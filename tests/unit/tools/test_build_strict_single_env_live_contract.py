# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano"
BUILDER_PATH = RECIPE_DIR / "build_strict_single_env_live_contract.py"
COLLECTOR_PATH = RECIPE_DIR / "collect_strict_single_env_wandb.py"
EVALUATOR_PATH = RECIPE_DIR / "evaluate_strict_single_env_live.py"
TERMINAL_COLLECTOR_PATH = RECIPE_DIR / "collect_strict_single_env_terminal_jobs.py"
LIVE_TEST_PATH = Path(__file__).with_name("test_evaluate_strict_single_env_live.py")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module("strict_live_contract_builder", BUILDER_PATH)
LIVE_TEST = _load_module("strict_live_contract_fixture", LIVE_TEST_PATH)
LIVE = LIVE_TEST.EVALUATOR


def _lineage_document(fixture: Any) -> Any:
    return LIVE.document_from_value(
        {
            "schema": BUILDER.BUILDER_INPUT_SCHEMA,
            "unverified_lineage_metadata": fixture.contract["provenance"]["trusted_oob_declarations"],
        },
        trailing_lf=False,
    )


def _build(
    fixture: Any,
    *,
    lineage_document: Any | None = None,
    terminal_collector_sha256: str | None = None,
    wandb_collector_sha256: str | None = None,
) -> dict[str, Any]:
    return BUILDER.build_contract(
        live=LIVE,
        pair_document=fixture.artifacts["pair_manifest"],
        submission_document=fixture.artifacts["submission_receipt"],
        holdout_document=fixture.artifacts["holdout"],
        execution_documents={
            "off": fixture.artifacts["off_execution"],
            "on": fixture.artifacts["on_execution"],
        },
        job_documents={
            "off": fixture.artifacts["off_job_exit"],
            "on": fixture.artifacts["on_job_exit"],
        },
        terminal_document=fixture.artifacts["terminal_scheduler"],
        unverified_lineage_document=lineage_document or _lineage_document(fixture),
        collector_path=COLLECTOR_PATH,
        expected_terminal_collector_sha256=(
            terminal_collector_sha256 or hashlib.sha256(TERMINAL_COLLECTOR_PATH.read_bytes()).hexdigest()
        ),
        expected_wandb_collector_sha256=(
            wandb_collector_sha256 or hashlib.sha256(COLLECTOR_PATH.read_bytes()).hexdigest()
        ),
    )


def _write_document(path: Path, document: Any) -> None:
    path.write_bytes(document.raw)


@pytest.mark.parametrize("environment", ["reasoning_gym", "citation", "freeform"])
def test_builder_derives_valid_current_live_contract_for_every_environment(
    environment: str,
) -> None:
    fixture = LIVE_TEST._fixture(environment)
    contract = _build(fixture)
    pair = fixture.artifacts["pair_manifest"]

    LIVE.validate_contract(contract)
    assert contract["pair"]["environment"] == environment
    assert contract["campaign"] == pair.value["campaign"]
    assert contract["provenance"]["common"]["pair_manifest_sha256"] == pair.sha256
    assert contract["provenance"]["common"]["wandb_exporter_sha256"] == (
        hashlib.sha256(COLLECTOR_PATH.read_bytes()).hexdigest()
    )
    assert contract["configs"]["off"]["shared_prefix_mode"] == "observe"
    assert contract["configs"]["on"]["shared_prefix_mode"] == "train"
    for arm in ("off", "on"):
        assert (
            contract["provenance"]["arms"][arm]["snapshot_manifest_sha256"]
            == pair.value["source"]["snapshots"][arm]["manifest_sha256"]
        )
    assert (
        contract["provenance"]["arms"]["off"]["snapshot_manifest_sha256"]
        == contract["provenance"]["arms"]["on"]["snapshot_manifest_sha256"]
    )


def test_builder_rejects_lineage_that_claims_an_observable_pair_pin() -> None:
    fixture = LIVE_TEST._fixture()
    value = copy.deepcopy(_lineage_document(fixture).value)
    value["unverified_lineage_metadata"]["common"]["pair_manifest_sha256"] = "0" * 64
    lineage = LIVE.document_from_value(value, trailing_lf=False)

    with pytest.raises(LIVE.EvidenceError, match="unverified lineage common metadata fields differ"):
        _build(fixture, lineage_document=lineage)


def test_builder_rejects_a_semantically_tampered_execution_receipt() -> None:
    fixture = LIVE_TEST._fixture()
    value = copy.deepcopy(fixture.artifacts["off_execution"].value)
    value["shared_prefix_mode"] = "train"
    fixture.artifacts["off_execution"] = LIVE.document_from_value(value)

    with pytest.raises(LIVE.EvidenceError, match="execution receipt.shared_prefix_mode differs"):
        _build(fixture)


def test_builder_authenticates_retained_evaluator_source() -> None:
    with pytest.raises(BUILDER.ContractBuildError, match="trusted OOB"):
        BUILDER._load_live_evaluator("0" * 64)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("terminal", "terminal scheduler collector differs"),
        ("wandb", "W&B collector differs"),
    ],
)
def test_builder_rejects_unreviewed_collector_bytes(field: str, message: str) -> None:
    fixture = LIVE_TEST._fixture()
    arguments = {f"{field}_collector_sha256": "0" * 64}

    with pytest.raises(BUILDER.ContractBuildError, match=message):
        _build(fixture, **arguments)


def test_exclusive_stager_publishes_canonical_read_only_bytes_once(
    tmp_path: Path,
) -> None:
    fixture = LIVE_TEST._fixture()
    contract = _build(fixture)
    raw = BUILDER._canonical_contract_bytes(LIVE, contract)
    output = tmp_path / "acceptance-contract.json"

    BUILDER._stage_exclusive(output, raw)

    assert output.read_bytes() == raw
    assert not raw.endswith(b"\n")
    assert json.loads(raw) == contract
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert output.stat().st_nlink == 1
    assert not list(tmp_path.glob(".*.candidate-*"))
    with pytest.raises(BUILDER.ContractBuildError, match="output already exists"):
        BUILDER._stage_exclusive(output, b"hostile replacement")
    assert output.read_bytes() == raw


def test_exclusive_stager_removes_publication_if_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "acceptance-contract.json"
    original_fsync = BUILDER.os.fsync
    calls = 0

    def fail_first_directory_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(BUILDER.os, "fsync", fail_first_directory_sync)

    with pytest.raises(BUILDER.ContractBuildError, match="cannot stage acceptance contract"):
        BUILDER._stage_exclusive(output, b"sealed acceptance contract")
    assert not output.exists()
    assert not list(tmp_path.glob(".*.candidate-*"))


def test_exclusive_stager_rejects_relative_and_symlinked_output_ancestry(tmp_path: Path) -> None:
    with pytest.raises(BUILDER.ContractBuildError, match="lexical-canonical absolute"):
        BUILDER._stage_exclusive(Path("acceptance-contract.json"), b"sealed acceptance contract")

    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    os.symlink(real_parent, linked_parent)
    with pytest.raises(BUILDER.ContractBuildError, match="cannot authenticate output ancestry"):
        BUILDER._stage_exclusive(linked_parent / "acceptance-contract.json", b"sealed acceptance contract")


def test_builder_cli_authenticates_inputs_and_stages_one_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = LIVE_TEST._fixture("freeform")
    documents = {
        "pair": fixture.artifacts["pair_manifest"],
        "submission": fixture.artifacts["submission_receipt"],
        "holdout": fixture.artifacts["holdout"],
        "off-execution": fixture.artifacts["off_execution"],
        "on-execution": fixture.artifacts["on_execution"],
        "off-exit": fixture.artifacts["off_job_exit"],
        "on-exit": fixture.artifacts["on_job_exit"],
        "terminal": fixture.artifacts["terminal_scheduler"],
        "lineage": _lineage_document(fixture),
    }
    paths = {name: tmp_path / f"{name}.json" for name in documents}
    for name, document in documents.items():
        _write_document(paths[name], document)
    output = tmp_path / "acceptance-contract.json"
    arguments = [
        "--expected-evaluator-sha256",
        hashlib.sha256(EVALUATOR_PATH.read_bytes()).hexdigest(),
        "--expected-terminal-collector-sha256",
        hashlib.sha256(TERMINAL_COLLECTOR_PATH.read_bytes()).hexdigest(),
        "--expected-wandb-collector-sha256",
        hashlib.sha256(COLLECTOR_PATH.read_bytes()).hexdigest(),
        "--pair-manifest",
        str(paths["pair"]),
        "--submission-receipt",
        str(paths["submission"]),
        "--holdout-receipt",
        str(paths["holdout"]),
        "--off-execution-receipt",
        str(paths["off-execution"]),
        "--on-execution-receipt",
        str(paths["on-execution"]),
        "--off-job-exit-receipt",
        str(paths["off-exit"]),
        "--on-job-exit-receipt",
        str(paths["on-exit"]),
        "--terminal-scheduler-receipt",
        str(paths["terminal"]),
        "--unverified-lineage-metadata",
        str(paths["lineage"]),
        "--collector",
        str(COLLECTOR_PATH),
        "--output",
        str(output),
    ]

    assert BUILDER.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["path"] == str(output.resolve())
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert BUILDER.main(arguments) == 2
    assert "output already exists" in capsys.readouterr().err


def test_builder_reports_evaluator_load_failure_without_unbound_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_load(expected_sha256: str) -> ModuleType:
        raise BUILDER.ContractBuildError("synthetic evaluator load failure")

    monkeypatch.setattr(BUILDER, "_load_live_evaluator", fail_load)
    arguments = [
        "--expected-evaluator-sha256",
        "0" * 64,
        "--expected-terminal-collector-sha256",
        "0" * 64,
        "--expected-wandb-collector-sha256",
        "0" * 64,
        "--pair-manifest",
        "unused",
        "--submission-receipt",
        "unused",
        "--holdout-receipt",
        "unused",
        "--off-execution-receipt",
        "unused",
        "--on-execution-receipt",
        "unused",
        "--off-job-exit-receipt",
        "unused",
        "--on-job-exit-receipt",
        "unused",
        "--terminal-scheduler-receipt",
        "unused",
        "--unverified-lineage-metadata",
        "unused",
        "--output",
        "unused",
    ]

    assert BUILDER.main(arguments) == 2
    assert "synthetic evaluator load failure" in capsys.readouterr().err
