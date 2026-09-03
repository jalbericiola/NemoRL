# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Synthetic contract tests for the production V2 replay coordinator.

These tests exercise local publication and public-API orchestration only.  They
are not an HSG scheduler, container, GPU, scorer, or lifecycle acceptance run.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shlex
import signal
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import nemo_rl.utils.strict_captured_replay_coordinator_v2 as coordinator
from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
    PAIR_MANIFEST_SCHEMA,
    PAIR_SLURM_EXPORT_SCHEMA,
    REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    REPLAY_SUBMISSION_CONTRACT_SCHEMA,
    SLURM_EXPORT_ALLOWED_NAMES,
    AuthenticatedOffSourceCapture,
    AuthenticatedReplayStaticInputs,
)
from nemo_rl.utils.strict_captured_replay_profiles import (
    get_strict_captured_replay_profile,
)


def _source(
    tmp_path: Path,
    *,
    environment: str = "citation",
) -> AuthenticatedOffSourceCapture:
    profile_id = {
        "citation": "citation-string-match-v1",
        "freeform": "freeform-regex-v1",
    }[environment]
    profile = get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    results_root = tmp_path / "results"
    results_root.mkdir(mode=0o700)
    results_root.chmod(0o700)
    source_root = tmp_path / "source"
    on_snapshot = tmp_path / "on-snapshot"
    pair = {
        "schema": PAIR_MANIFEST_SCHEMA,
        "pair_id": f"pair-{environment}",
        "paths": {
            "results_root": str(results_root),
        },
        "selection": {
            "environment": environment,
        },
        "container_entry_boundary": {
            "bash_args": ["-p"],
            "bash_path": "/bin/bash",
            "env_path": "/usr/bin/env",
            "sha256sum": {"path": "/usr/bin/sha256sum", "sha256": "9" * 64},
            "unset_environment": ["BASH_ENV", "ENV"],
        },
        "source": {
            "root": str(source_root),
            "gym": {
                "gitlink_commit": "1" * 40,
                "tree": "2" * 40,
            },
            "snapshots": {
                "on": {
                    "path": str(on_snapshot),
                },
            },
        },
        "artifacts": {
            "fixture": {
                "path": str(source_root / profile.fixture_path),
                "rows": profile.fixture_rows,
                "sha256": profile.fixture_sha256,
            },
        },
        "slurm_export_boundary": {
            "schema": PAIR_SLURM_EXPORT_SCHEMA,
            "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
        },
    }
    return AuthenticatedOffSourceCapture(
        source_capture={},
        pair_manifest=pair,
        pair_manifest_sha256="a" * 64,
        pair_submission_receipt={},
        pair_submission_receipt_sha256="b" * 64,
        trusted_off_exit_receipt_path=str(tmp_path / "EXIT.json"),
        trusted_off_exit_receipt_sha256="c" * 64,
        pre_receipt={},
        pre_receipt_sha256="d" * 64,
        exit_receipt={},
        exit_receipt_sha256="c" * 64,
        main_ledger={},
        transcript_bundle={},
        transport_bundle={},
        transport_manifest={},
        transport_records=(),
    )


def _profile_id(environment: str) -> str:
    return {
        "citation": "citation-string-match-v1",
        "freeform": "freeform-regex-v1",
    }[environment]


def _export(
    source: AuthenticatedOffSourceCapture,
    *,
    attempt_id: str = "replay-1",
) -> coordinator.Pair79ReplayExport:
    environment = source.pair_manifest["selection"]["environment"]
    return coordinator._build_pair79_replay_export(
        authenticated_source=source,
        attempt_id=attempt_id,
        expected_environment=environment,
        expected_profile_id=_profile_id(environment),
    )


def _prepared(
    source: AuthenticatedOffSourceCapture,
    *,
    launcher_path: Path,
    launcher_sha256: str,
    launcher_argv: tuple[str, ...] | None = None,
) -> coordinator.PreparedReplaySubmission:
    pair = source.pair_manifest
    pair_id = pair["pair_id"]
    results_root = pair["paths"]["results_root"]
    manifest_path = f"{results_root}/captured_replay/manifests/{pair_id}/replay-1.json"
    exact_argv = coordinator._build_replay_launcher_argv(
        authenticated_source=source,
        replay_manifest_path=manifest_path,
        replay_manifest_sha256="5" * 64,
        expected_environment=pair["selection"]["environment"],
        expected_profile_id=_profile_id(pair["selection"]["environment"]),
    )
    return coordinator.PreparedReplaySubmission(
        pair_id=pair_id,
        attempt_id="replay-1",
        environment=pair["selection"]["environment"],
        profile_id=_profile_id(pair["selection"]["environment"]),
        pair_manifest_path=f"{results_root}/PAIR_MANIFEST.json",
        pair_manifest_sha256=source.pair_manifest_sha256,
        pair_submission_receipt_path=(f"{results_root}/PAIR_SUBMISSION_RECEIPT.json"),
        pair_submission_receipt_sha256=source.pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=source.trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=(source.trusted_off_exit_receipt_sha256),
        slurm_export_path=(f"{results_root}/captured_replay/slurm_exports/{pair_id}/replay-1.env"),
        slurm_export_sha256="7" * 64,
        submission_contract_path=(
            f"{results_root}/captured_replay/replay_submission_state/{pair_id}/" "replay-1.submission-contract.json"
        ),
        submission_contract_sha256="8" * 64,
        replay_manifest_path=manifest_path,
        replay_manifest_sha256="5" * 64,
        launcher_path=str(launcher_path),
        launcher_sha256=launcher_sha256,
        launcher_cwd=pair["source"]["snapshots"]["on"]["path"],
        launcher_argv=exact_argv if launcher_argv is None else launcher_argv,
    )


def _static_inputs_for_prepared(
    prepared: coordinator.PreparedReplaySubmission,
) -> AuthenticatedReplayStaticInputs:
    return AuthenticatedReplayStaticInputs(
        attempt_id=prepared.attempt_id,
        container_asset={},
        source_snapshot={"path": prepared.launcher_cwd},
        gym_source_root={},
        replay_program={
            "submission_launcher": {
                "path": "strict_pair_replay_launch_v2.sh",
                "sha256": prepared.launcher_sha256,
            }
        },
        slurm_export_path=prepared.slurm_export_path,
        slurm_export_sha256=prepared.slurm_export_sha256,
        slurm_export_values=(),
        submission_contract_path=prepared.submission_contract_path,
        submission_contract_sha256=prepared.submission_contract_sha256,
        submission_contract={
            "submission_launcher": {
                "path": prepared.launcher_path,
                "sha256": prepared.launcher_sha256,
            }
        },
    )


def _manifest_for_prepared(
    prepared: coordinator.PreparedReplaySubmission,
) -> dict[str, Any]:
    return {
        "pair_id": prepared.pair_id,
        "attempt_id": prepared.attempt_id,
        "environment": prepared.environment,
        "scorer_profile": {"profile_id": prepared.profile_id},
    }


def _authenticated_result_snapshot(
    tmp_path: Path,
    *,
    environment: str = "citation",
) -> dict[str, Any]:
    pair_id = f"pair-{environment}"
    profile_id = _profile_id(environment)
    scorer_boot_id = "12345678-1234-1234-1234-123456789abc"
    result_root = tmp_path / "results/captured_replay/replay-1"
    manifest_path = tmp_path / f"results/captured_replay/manifests/{pair_id}/replay-1.json"
    submission_path = (
        tmp_path / f"results/captured_replay/replay_submission_state/{pair_id}/" "replay-1/submission-receipt.json"
    )
    receipt_root = tmp_path / f"results/captured_replay/replay_job_state/{pair_id}/replay-1/" "93001-0/receipts"

    def reference(path: Path, schema: str, digest: str) -> dict[str, str]:
        return {"path": str(path), "schema": schema, "sha256": digest}

    outputs = {
        "scorer_call_index": reference(
            result_root / "strict_gym_child_runtime/format-verification-call-index.json",
            "nemo-rl-strict-format-verification-call-index-v1",
            "1" * 64,
        ),
        "transport_consumption": reference(
            result_root / "model-transport-replay-consumption.json",
            "nemo-rl-strict-model-transport-replay-consumption-v3",
            "2" * 64,
        ),
        "transcript_bundle": reference(
            result_root / "transcript-bundle.json",
            "nemo-rl-strict-step1-transcript-bundle-v4",
            "3" * 64,
        ),
        "replay_ledger": reference(
            result_root / "replay-ledger.json",
            "nemo-rl-strict-captured-replay-step1-ledger-v5",
            "4" * 64,
        ),
    }
    return {
        "schema": coordinator.AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA,
        "pair_id": pair_id,
        "environment": environment,
        "profile_id": profile_id,
        "attempt_id": "replay-1",
        "candidate_job_id": "93001",
        "authenticated_job_id": "93001",
        "run_id": hashlib.sha256(
            (f"nemo-rl-strict-replay-v2:{environment}:{pair_id}:replay-1").encode("ascii")
        ).hexdigest(),
        "driver_process": {
            "boot_id_sha256": hashlib.sha256((scorer_boot_id + "\n").encode("ascii")).hexdigest(),
            "pid": 31001,
            "start_time_ticks": 71001,
        },
        "scorer_process_identity": {
            "boot_id": scorer_boot_id,
            "hostname": "strict-scorer-0",
            "pid": 31002,
            "start_ticks": 71002,
        },
        "manifest": reference(
            manifest_path,
            REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
            "5" * 64,
        ),
        "submission_receipt": reference(
            submission_path,
            "nemo-rl-strict-captured-replay-submission-receipt-v5",
            "6" * 64,
        ),
        "pre_receipt": reference(
            receipt_root / "PRE.json",
            "nemo-rl-strict-captured-replay-job-pre-receipt-v3",
            "7" * 64,
        ),
        "exit_receipt": reference(
            receipt_root / "EXIT.json",
            "nemo-rl-strict-captured-replay-job-exit-receipt-v6",
            "8" * 64,
        ),
        "result_final_receipt": reference(
            receipt_root / "FINAL.json",
            "nemo-rl-strict-captured-replay-result-final-receipt-v1",
            "9" * 64,
        ),
        "result_root": str(result_root),
        "result_inventory": reference(
            result_root / "result-inventory-v2.json",
            "nemo-rl-strict-captured-replay-result-inventory-v2",
            "a" * 64,
        ),
        "evidence_index": reference(
            result_root / "evidence-index.json",
            "nemo-rl-strict-captured-replay-evidence-index-v4",
            "b" * 64,
        ),
        "outputs": outputs,
        "samples": [
            {
                "sample_index": index,
                "fixture_row_index": 0,
                "rollout_index": index,
                "generation_seed": 41000 + index,
                "model_transport_entry_sha256": "c" * 64,
                "model_transport_request_body_sha256": "d" * 64,
                "model_transport_response_body_sha256": "e" * 64,
                "model_response_sha256": "f" * 64,
                "match_details": (
                    {
                        "expected": ["[1]"],
                        "missing": [],
                        "spurious": [],
                        "passed": True,
                    }
                    if environment == "citation"
                    else {
                        "matching_lines": 1,
                        "min_matches": 1,
                        "passed": True,
                    }
                ),
                "raw_environment_reward": 1.0,
            }
            for index in range(4)
        ],
    }


def _install_prepared_reauthentication_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: AuthenticatedOffSourceCapture,
    prepared: coordinator.PreparedReplaySubmission,
) -> None:
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_source",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_static_inputs",
        lambda **_kwargs: _static_inputs_for_prepared(prepared),
    )
    monkeypatch.setattr(
        coordinator,
        "load_replay_execution_manifest_v2",
        lambda **_kwargs: (_manifest_for_prepared(prepared), "5" * 64),
    )


@pytest.mark.parametrize("environment", ["citation", "freeform"])
def test_pair79_export_has_exact_order_framing_and_six_nonempty_values(
    tmp_path: Path,
    environment: str,
) -> None:
    source = _source(tmp_path, environment=environment)
    document = _export(source)

    assert len(document.records) == 79
    assert tuple(name for name, _ in document.records) == SLURM_EXPORT_ALLOWED_NAMES
    assert document.raw.endswith(b"\0")
    assert not document.raw.endswith(b"\0\0")
    assert b"\n" not in document.raw
    assert hashlib.sha256(document.raw).hexdigest() == document.sha256

    values = dict(document.records)
    assert {name for name, value in document.records if value} == {
        "EXPECTED_GYM_GITLINK_COMMIT",
        "EXPECTED_GYM_TREE",
        "PAIR_ID",
        "RESULTS_DIR",
        "STRICT_PAIR_ENVIRONMENT",
        "STRICT_PREBUILT_SNAPSHOT_DIR",
    }
    pair = source.pair_manifest
    assert values["EXPECTED_GYM_GITLINK_COMMIT"] == b"1" * 40
    assert values["EXPECTED_GYM_TREE"] == b"2" * 40
    assert values["PAIR_ID"] == pair["pair_id"].encode("ascii")
    assert values["STRICT_PAIR_ENVIRONMENT"] == environment.encode("ascii")
    assert values["RESULTS_DIR"] == (f"{pair['paths']['results_root']}/captured_replay/replay-1".encode("ascii"))
    assert values["STRICT_PREBUILT_SNAPSHOT_DIR"] == pair["source"]["snapshots"]["on"]["path"].encode("ascii")


def test_pair79_export_publication_is_atomic_exclusive_and_mode_0400(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    document = _export(source)
    path, digest = coordinator._publish_pair79_replay_export(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
        document=document,
    )

    metadata = path.lstat()
    assert path.read_bytes() == document.raw
    assert digest == document.sha256
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_uid == os.geteuid()
    assert metadata.st_nlink == 1
    with pytest.raises(FileExistsError, match="Pair79 export already exists"):
        coordinator._publish_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
            document=document,
        )
    assert path.read_bytes() == document.raw


def test_pair79_export_partial_write_never_publishes_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    document = _export(source)

    def partial_write(descriptor: int, payload: bytes) -> None:
        assert os.write(descriptor, payload[:7]) == 7
        raise OSError("synthetic partial write")

    monkeypatch.setattr(coordinator, "_write_all", partial_write)
    with pytest.raises(OSError, match="synthetic partial write"):
        coordinator._publish_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
            document=document,
        )
    target = Path(document.path)
    assert not target.exists()
    assert not (target.parent / ".replay-1.env.candidate").exists()


def test_pair79_export_rejects_poisoned_parent_mode_without_repairing_it(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    document = _export(source)
    captured_replay = Path(source.pair_manifest["paths"]["results_root"]) / "captured_replay"
    captured_replay.mkdir(mode=0o755)
    captured_replay.chmod(0o755)

    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="mode-0700",
    ):
        coordinator._publish_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
            document=document,
        )
    assert stat.S_IMODE(captured_replay.stat().st_mode) == 0o755
    assert not Path(document.path).exists()


def test_pair79_export_rejects_symlink_parent_and_preexisting_symlink_target(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    document = _export(source)
    results_root = Path(source.pair_manifest["paths"]["results_root"])
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (results_root / "captured_replay").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        coordinator._publish_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
            document=document,
        )
    assert list(outside.iterdir()) == []

    (results_root / "captured_replay").unlink()
    parent = results_root / "captured_replay/slurm_exports/pair-citation"
    parent.mkdir(mode=0o700, parents=True)
    for directory in (results_root / "captured_replay", parent.parent, parent):
        directory.chmod(0o700)
    poison = tmp_path / "poison"
    poison.write_bytes(b"unchanged")
    Path(document.path).symlink_to(poison)
    with pytest.raises(FileExistsError, match="Pair79 export already exists"):
        coordinator._publish_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
            document=document,
        )
    assert poison.read_bytes() == b"unchanged"


def test_pair79_export_rejects_exact_type_subclasses_and_profile_mismatch_before_writes(
    tmp_path: Path,
) -> None:
    class TextSubclass(str):
        pass

    class PairSubclass(dict[str, Any]):
        pass

    source = _source(tmp_path)
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="exact string",
    ):
        coordinator._build_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment=TextSubclass("citation"),
            expected_profile_id="citation-string-match-v1",
        )
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="environment differs",
    ):
        coordinator._build_pair79_replay_export(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )
    poisoned_source = replace(
        source,
        pair_manifest=PairSubclass(source.pair_manifest),
    )
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="exact dictionary",
    ):
        coordinator._build_pair79_replay_export(
            authenticated_source=poisoned_source,
            attempt_id="replay-1",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )
    results_root = Path(source.pair_manifest["paths"]["results_root"])
    assert not (results_root / "captured_replay").exists()


@pytest.mark.parametrize("environment", ["citation", "freeform"])
def test_prepare_uses_public_contract_and_v4_apis_and_selects_v2_exact20_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    profile_id = _profile_id(environment)
    source = _source(tmp_path, environment=environment)
    pair = source.pair_manifest
    pair_id = pair["pair_id"]
    results_root = Path(pair["paths"]["results_root"])
    export = _export(source)
    contract_path = (
        results_root / "captured_replay/replay_submission_state" / pair_id / "replay-1.submission-contract.json"
    )
    manifest_path = results_root / "captured_replay/manifests" / pair_id / "replay-1.json"
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    launcher.write_bytes(b"#!/bin/sh\nexit 99\n")
    launcher.chmod(0o555)
    launcher_sha = hashlib.sha256(launcher.read_bytes()).hexdigest()
    contract_sha = "4" * 64
    manifest_sha = "5" * 64
    contract = {
        "schema": REPLAY_SUBMISSION_CONTRACT_SCHEMA,
        "submission_launcher": {
            "path": str(launcher),
            "sha256": launcher_sha,
        },
    }
    static_inputs = AuthenticatedReplayStaticInputs(
        attempt_id="replay-1",
        container_asset={},
        source_snapshot={"path": str(launcher.parent)},
        gym_source_root={},
        replay_program={
            "submission_launcher": {
                "path": "strict_pair_replay_launch_v2.sh",
                "sha256": launcher_sha,
            }
        },
        slurm_export_path=export.path,
        slurm_export_sha256=export.sha256,
        slurm_export_values=export.records,
        submission_contract_path=str(contract_path),
        submission_contract_sha256=contract_sha,
        submission_contract=contract,
    )
    manifest = {
        "schema": REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
        "pair_id": pair_id,
        "attempt_id": "replay-1",
    }
    calls: list[str] = []

    def load_source(**kwargs: str) -> AuthenticatedOffSourceCapture:
        assert kwargs["expected_environment"] == environment
        assert kwargs["expected_profile_id"] == profile_id
        calls.append("load-source")
        return source

    def build_contract(**kwargs: Any) -> dict[str, Any]:
        assert Path(export.path).read_bytes() == export.raw
        assert kwargs["authenticated_source"] is source
        calls.append("build-contract")
        return contract

    def publish_contract(**kwargs: Any) -> tuple[Path, str]:
        assert kwargs["document"] is contract
        calls.append("publish-contract")
        return contract_path, contract_sha

    def load_static(**kwargs: Any) -> AuthenticatedReplayStaticInputs:
        calls.append("load-static")
        return static_inputs

    def build_manifest(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["expected_environment"] == environment
        assert kwargs["expected_profile_id"] == profile_id
        calls.append("build-manifest-v4")
        return manifest

    def publish_manifest(**kwargs: Any) -> tuple[Path, str]:
        assert kwargs["output"] == str(manifest_path)
        assert kwargs["document"] is manifest
        calls.append("publish-manifest-v4")
        return manifest_path, manifest_sha

    def load_manifest(**kwargs: Any) -> tuple[dict[str, Any], str]:
        calls.append("load-manifest-v4")
        return manifest, manifest_sha

    monkeypatch.setattr(coordinator, "load_authenticated_replay_source", load_source)
    monkeypatch.setattr(coordinator, "build_replay_submission_contract", build_contract)
    monkeypatch.setattr(coordinator, "publish_replay_submission_contract", publish_contract)
    monkeypatch.setattr(coordinator, "load_authenticated_replay_static_inputs", load_static)
    monkeypatch.setattr(coordinator, "build_replay_execution_manifest_v2", build_manifest)
    monkeypatch.setattr(coordinator, "publish_replay_execution_manifest_v2", publish_manifest)
    monkeypatch.setattr(coordinator, "load_replay_execution_manifest_v2", load_manifest)

    prepared = coordinator.prepare_replay_submission(
        pair_manifest_path=str(results_root / "PAIR_MANIFEST.json"),
        pair_manifest_sha256="a" * 64,
        pair_submission_receipt_path=str(results_root / "PAIR_SUBMISSION_RECEIPT.json"),
        pair_submission_receipt_sha256="b" * 64,
        trusted_off_exit_receipt_path=str(tmp_path / "EXIT.json"),
        trusted_off_exit_receipt_sha256="c" * 64,
        attempt_id="replay-1",
        submission_nonce="3" * 64,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )

    assert calls == [
        "load-source",
        "build-contract",
        "publish-contract",
        "load-static",
        "build-manifest-v4",
        "publish-manifest-v4",
        "load-manifest-v4",
    ]
    assert prepared.launcher_path == str(launcher)
    assert prepared.launcher_sha256 == launcher_sha
    assert prepared.launcher_cwd == str(launcher.parent)
    assert Path(prepared.launcher_path).name == "strict_pair_replay_launch_v2.sh"
    assert "strict_pair_replay_launch.sh" not in prepared.launcher_path
    assert len(prepared.launcher_argv) == 20
    assert prepared.launcher_argv[::2] == (
        "--pair-manifest",
        "--pair-manifest-sha256",
        "--pair-submission-receipt",
        "--pair-submission-receipt-sha256",
        "--off-exit-receipt",
        "--off-exit-receipt-sha256",
        "--replay-manifest",
        "--replay-manifest-sha256",
        "--environment",
        "--profile-id",
    )
    assert prepared.launcher_argv[-2:] == (
        "--profile-id",
        profile_id,
    )


def test_invoke_executes_exact20_v2_launcher_with_empty_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    argv = coordinator._build_replay_launcher_argv(
        authenticated_source=source,
        replay_manifest_path=str(tmp_path / "results/captured_replay/manifests/pair-citation/replay-1.json"),
        replay_manifest_sha256="5" * 64,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    checks = ['[ "$#" -eq 20 ] || exit 31']
    checks.extend(
        f'[ "${{{index}}}" = {shlex.quote(value)} ] || exit {31 + index}' for index, value in enumerate(argv, start=1)
    )
    checks.extend(
        [
            '[ -z "${PYTHONPATH+x}" ] || exit 70',
            '[ -z "${WANDB_API_KEY+x}" ] || exit 71',
        ]
    )
    receipt_sha = "6" * 64
    output = (
        '{"attempt_id":"replay-1","candidate_job_id":"93001",'
        f'"pair_id":"pair-citation","submission_receipt_sha256":"{receipt_sha}"}}'
    )
    launcher.write_text(
        "#!/bin/sh\n" + "\n".join(checks) + f"\nprintf '%s\\n' {shlex.quote(output)}\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
        launcher_argv=argv,
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )

    assert coordinator.invoke_replay_launcher(prepared) == {
        "attempt_id": "replay-1",
        "candidate_job_id": "93001",
        "pair_id": "pair-citation",
        "submission_receipt_sha256": receipt_sha,
    }


def test_invoke_executes_retained_launcher_inode_during_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    marker = tmp_path / "attacker-executed"
    receipt_sha = "6" * 64
    output = (
        '{"attempt_id":"replay-1","candidate_job_id":"93001",'
        f'"pair_id":"pair-citation","submission_receipt_sha256":"{receipt_sha}"}}'
    )
    launcher.write_text(
        f"#!/bin/bash -p\nprintf '%s\\n' {shlex.quote(output)}\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )
    real_run = coordinator._run_bounded_launcher
    saved = launcher.with_name("saved-launcher")

    def swap_run(argv: list[str], **kwargs: Any) -> Any:
        assert argv[:3] == [
            "/bin/bash",
            "-p",
            coordinator._submission_launcher_fd_path(kwargs["pass_fds"][0]),
        ]
        launcher.rename(saved)
        launcher.write_text(
            f"#!/bin/bash -p\ntouch {shlex.quote(str(marker))}\n",
            encoding="ascii",
        )
        launcher.chmod(0o555)
        try:
            return real_run(argv, **kwargs)
        finally:
            launcher.unlink()
            saved.rename(launcher)

    monkeypatch.setattr(coordinator, "_run_bounded_launcher", swap_run)
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="authenticated stable bytes",
    ):
        coordinator.invoke_replay_launcher(prepared)
    assert not marker.exists()


def test_invoke_executes_sealed_copy_during_in_place_launcher_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    marker = tmp_path / "attacker-executed"
    receipt_sha = "6" * 64
    output = (
        '{"attempt_id":"replay-1","candidate_job_id":"93001",'
        f'"pair_id":"pair-citation","submission_receipt_sha256":"{receipt_sha}"}}'
    )
    launcher.write_text(
        f"#!/bin/bash -p\nprintf '%s\\n' {shlex.quote(output)}\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    authenticated_raw = launcher.read_bytes()
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(authenticated_raw).hexdigest(),
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )
    real_run = coordinator._run_bounded_launcher

    def mutate_run(argv: list[str], **kwargs: Any) -> Any:
        launcher.chmod(0o700)
        launcher.write_text(
            f"#!/bin/bash -p\ntouch {shlex.quote(str(marker))}\n",
            encoding="ascii",
        )
        launcher.chmod(0o555)
        try:
            return real_run(argv, **kwargs)
        finally:
            launcher.chmod(0o700)
            launcher.write_bytes(authenticated_raw)
            launcher.chmod(0o555)

    monkeypatch.setattr(coordinator, "_run_bounded_launcher", mutate_run)
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="authenticated stable bytes",
    ):
        coordinator.invoke_replay_launcher(prepared)
    assert not marker.exists()


def test_invoke_timeout_reaps_entire_launcher_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    marker = tmp_path / "orphan-marker"
    launcher.write_text(
        "#!/bin/bash -p\n" f"(sleep 0.4; touch {shlex.quote(str(marker))}) &\n" "sleep 5\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )
    monkeypatch.setattr(coordinator, "_LAUNCH_TIMEOUT_SECONDS", 0.05)
    signals: list[int] = []
    real_killpg = coordinator.os.killpg

    def observed_killpg(process_group: int, selected_signal: int) -> None:
        signals.append(selected_signal)
        real_killpg(process_group, selected_signal)

    monkeypatch.setattr(coordinator.os, "killpg", observed_killpg)

    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="execution deadline",
    ):
        coordinator.invoke_replay_launcher(prepared)
    time.sleep(0.6)
    assert not marker.exists()
    assert signals == [signal.SIGTERM]


def test_invoke_rejects_launcher_output_before_unbounded_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    launcher.write_text(
        "#!/bin/bash -p\nfor ((i=0; i<70000; i++)); do printf x; done\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )

    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="strict bound",
    ):
        coordinator.invoke_replay_launcher(prepared)


@pytest.mark.parametrize("poison_mode", [0o500, 0o755])
def test_invoke_rejects_launcher_mode_mismatch_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison_mode: int,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    marker = tmp_path / "executed"
    launcher.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n", encoding="ascii")
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    _install_prepared_reauthentication_mocks(
        monkeypatch,
        source=source,
        prepared=prepared,
    )
    launcher.chmod(poison_mode)

    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="mode-0555",
    ):
        coordinator.invoke_replay_launcher(prepared)
    assert not marker.exists()


def test_invoke_rejects_forged_nonexact_argv_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    launcher.write_bytes(b"#!/bin/sh\nexit 90\n")
    launcher.chmod(0o555)
    prepared = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    forged = replace(
        prepared,
        launcher_argv=("--pair-manifest", prepared.pair_manifest_path),
    )
    executed = False

    def forbidden_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executed
        executed = True

    monkeypatch.setattr(coordinator, "_run_bounded_launcher", forbidden_run)
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="exact20 authority",
    ):
        coordinator.invoke_replay_launcher(forged)
    assert executed is False


def test_invoke_rejects_self_signed_forged_prepared_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    launcher = tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"
    launcher.parent.mkdir(mode=0o700)
    marker = tmp_path / "executed"
    launcher.write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\n",
        encoding="ascii",
    )
    launcher.chmod(0o555)
    forged = _prepared(
        source,
        launcher_path=launcher,
        launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
    )
    authenticated = False
    executed = False

    def reject_forged_source(**_kwargs: Any) -> None:
        nonlocal authenticated
        authenticated = True
        raise ValueError("synthetic unauthenticated source anchors")

    def forbidden_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executed
        executed = True

    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_source",
        reject_forged_source,
    )
    monkeypatch.setattr(coordinator, "_run_bounded_launcher", forbidden_run)
    with pytest.raises(ValueError, match="unauthenticated source anchors"):
        coordinator.invoke_replay_launcher(forged)
    assert authenticated is True
    assert executed is False
    assert not marker.exists()


def test_released_receipt_is_reloaded_through_public_profile_bound_v2_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    replay_manifest_path = tmp_path / "results/captured_replay/manifests/pair-citation/replay-1.json"
    submission_receipt_path = (
        tmp_path / "results/captured_replay/replay_submission_state/pair-citation/replay-1/submission-receipt.json"
    )
    prepared = _prepared(
        source,
        launcher_path=(tmp_path / "on-snapshot/strict_pair_replay_launch_v2.sh"),
        launcher_sha256="4" * 64,
    )
    receipt_sha = "6" * 64
    launcher_receipt = {
        "attempt_id": "replay-1",
        "candidate_job_id": "93001",
        "pair_id": "pair-citation",
        "submission_receipt_sha256": receipt_sha,
    }
    manifest = {
        **_manifest_for_prepared(prepared),
        "scheduler_submission": {
            "receipt": {
                "path": str(submission_receipt_path),
            }
        },
    }
    document = {
        "schema": "synthetic-public-v2-submission-receipt",
        "status": "released",
        "pair_id": "pair-citation",
        "attempt_id": "replay-1",
        "candidate_job_id": "93001",
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def load_source(**kwargs: Any) -> AuthenticatedOffSourceCapture:
        calls.append(("source", kwargs))
        return source

    def load_manifest(**kwargs: Any) -> tuple[dict[str, Any], str]:
        calls.append(("manifest", kwargs))
        return manifest, "5" * 64

    def load_static(**kwargs: Any) -> AuthenticatedReplayStaticInputs:
        calls.append(("static", kwargs))
        return _static_inputs_for_prepared(prepared)

    def load_submission_receipt(**kwargs: Any) -> tuple[dict[str, Any], str]:
        calls.append(("receipt-v2", kwargs))
        return document, receipt_sha

    monkeypatch.setattr(coordinator, "load_authenticated_replay_source", load_source)
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_static_inputs",
        load_static,
    )
    monkeypatch.setattr(coordinator, "load_replay_execution_manifest_v2", load_manifest)
    monkeypatch.setattr(
        coordinator,
        "load_captured_replay_submission_receipt_v2",
        load_submission_receipt,
    )

    loaded = coordinator.load_released_submission_receipt(
        prepared=prepared,
        launcher_receipt=launcher_receipt,
        pair_manifest_path=str(tmp_path / "results/PAIR_MANIFEST.json"),
        pair_manifest_sha256="a" * 64,
        pair_submission_receipt_path=str(tmp_path / "results/PAIR_SUBMISSION_RECEIPT.json"),
        pair_submission_receipt_sha256="b" * 64,
        trusted_off_exit_receipt_path=str(tmp_path / "EXIT.json"),
        trusted_off_exit_receipt_sha256="c" * 64,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )

    assert loaded is document
    assert [name for name, _ in calls] == [
        "source",
        "static",
        "manifest",
        "receipt-v2",
    ]
    receipt_kwargs = calls[-1][1]
    assert receipt_kwargs == {
        "path": str(submission_receipt_path),
        "expected_sha256": receipt_sha,
        "replay_execution_manifest": manifest,
        "authenticated_source": source,
        "expected_environment": "citation",
        "expected_profile_id": "citation-string-match-v1",
    }

    poisoned_document = dict(document)
    poisoned_document["candidate_job_id"] = "93002"
    monkeypatch.setattr(
        coordinator,
        "load_captured_replay_submission_receipt_v2",
        lambda **_kwargs: (poisoned_document, receipt_sha),
    )
    with pytest.raises(
        coordinator.StrictCapturedReplayCoordinatorError,
        match="receipt identity differs",
    ):
        coordinator.load_released_submission_receipt(
            prepared=prepared,
            launcher_receipt=launcher_receipt,
            pair_manifest_path=str(tmp_path / "results/PAIR_MANIFEST.json"),
            pair_manifest_sha256="a" * 64,
            pair_submission_receipt_path=str(tmp_path / "results/PAIR_SUBMISSION_RECEIPT.json"),
            pair_submission_receipt_sha256="b" * 64,
            trusted_off_exit_receipt_path=str(tmp_path / "EXIT.json"),
            trusted_off_exit_receipt_sha256="c" * 64,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def _consume_kwargs(
    tmp_path: Path,
    *,
    environment: str = "citation",
) -> dict[str, str]:
    pair_id = f"pair-{environment}"
    return {
        "pair_manifest_path": str(tmp_path / "results/PAIR_MANIFEST.json"),
        "pair_manifest_sha256": "a" * 64,
        "pair_submission_receipt_path": str(tmp_path / "results/PAIR_SUBMISSION_RECEIPT.json"),
        "pair_submission_receipt_sha256": "b" * 64,
        "trusted_off_exit_receipt_path": str(tmp_path / "EXIT.json"),
        "trusted_off_exit_receipt_sha256": "c" * 64,
        "replay_manifest_path": str(tmp_path / f"results/captured_replay/manifests/{pair_id}/replay-1.json"),
        "replay_manifest_sha256": "5" * 64,
        "submission_receipt_sha256": "6" * 64,
        "candidate_job_id": "93001",
        "result_final_receipt_path": str(
            tmp_path / f"results/captured_replay/replay_job_state/{pair_id}/" "replay-1/93001-0/receipts/FINAL.json"
        ),
        "result_final_receipt_sha256": "9" * 64,
        "expected_environment": environment,
        "expected_profile_id": _profile_id(environment),
    }


@pytest.mark.parametrize("environment", ["citation", "freeform"])
def test_consume_uses_external_final_loader_and_detached_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
) -> None:
    source = _source(tmp_path, environment=environment)
    snapshot = _authenticated_result_snapshot(tmp_path, environment=environment)
    capability = object()
    calls: list[tuple[str, Any]] = []

    def load_source(**kwargs: Any) -> AuthenticatedOffSourceCapture:
        calls.append(("source", kwargs))
        return source

    def load_result(**kwargs: Any) -> object:
        calls.append(("authenticated-result", kwargs))
        return capability

    def detach(value: object) -> dict[str, Any]:
        assert value is capability
        calls.append(("snapshot", value))
        return snapshot

    monkeypatch.setattr(coordinator, "load_authenticated_replay_source", load_source)
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_captured_replay_result_v2",
        load_result,
    )
    monkeypatch.setattr(
        coordinator,
        "snapshot_authenticated_captured_replay_result_v2",
        detach,
    )

    consumed = coordinator.consume_replay_result(**_consume_kwargs(tmp_path, environment=environment))

    assert [name for name, _ in calls] == [
        "source",
        "authenticated-result",
        "snapshot",
    ]
    assert consumed.snapshot == snapshot
    assert consumed.snapshot is not snapshot
    assert calls[1][1] == {
        "authenticated_source": source,
        "replay_execution_manifest_path": snapshot["manifest"]["path"],
        "replay_execution_manifest_sha256": snapshot["manifest"]["sha256"],
        "submission_receipt_sha256": snapshot["submission_receipt"]["sha256"],
        "candidate_job_id": snapshot["candidate_job_id"],
        "result_final_receipt_path": snapshot["result_final_receipt"]["path"],
        "result_final_receipt_sha256": snapshot["result_final_receipt"]["sha256"],
        "expected_environment": snapshot["environment"],
        "expected_profile_id": snapshot["profile_id"],
    }


@pytest.mark.parametrize(
    "poison",
    [
        "pair-relabel",
        "attempt-relabel",
        "run-relabel",
        "driver-missing",
        "driver-extra",
        "driver-type",
        "scorer-missing",
        "scorer-extra",
        "scorer-type",
        "boot-mismatch",
        "process-alias",
        "reference-schema",
        "result-root",
        "output-missing",
        "output-extra",
        "output-schema",
        "sample-missing",
        "sample-extra",
        "sample-type",
        "fixture-row-nonzero",
        "reward-int-alias",
        "reward-negative-zero",
        "citation-passed-inconsistent",
    ],
)
def test_consume_rejects_malformed_authenticated_snapshot_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    source = _source(tmp_path)
    snapshot = copy.deepcopy(_authenticated_result_snapshot(tmp_path))
    if poison == "pair-relabel":
        snapshot["pair_id"] = "pair-other"
    elif poison == "attempt-relabel":
        snapshot["attempt_id"] = "replay-2"
    elif poison == "run-relabel":
        snapshot["run_id"] = "f" * 64
    elif poison == "driver-missing":
        snapshot["driver_process"].pop("boot_id_sha256")
    elif poison == "driver-extra":
        snapshot["driver_process"]["extra"] = 1
    elif poison == "driver-type":
        snapshot["driver_process"]["pid"] = True
    elif poison == "scorer-missing":
        snapshot["scorer_process_identity"].pop("hostname")
    elif poison == "scorer-extra":
        snapshot["scorer_process_identity"]["extra"] = 1
    elif poison == "scorer-type":
        snapshot["scorer_process_identity"]["start_ticks"] = True
    elif poison == "boot-mismatch":
        snapshot["driver_process"]["boot_id_sha256"] = "0a" * 32
    elif poison == "process-alias":
        snapshot["scorer_process_identity"]["pid"] = snapshot["driver_process"]["pid"]
        snapshot["scorer_process_identity"]["start_ticks"] = snapshot["driver_process"]["start_time_ticks"]
    elif poison == "reference-schema":
        snapshot["exit_receipt"]["schema"] = "synthetic-exit"
    elif poison == "result-root":
        snapshot["result_root"] = str(tmp_path / "alternate-result")
    elif poison == "output-missing":
        snapshot["outputs"].pop("replay_ledger")
    elif poison == "output-extra":
        snapshot["outputs"]["extra"] = copy.deepcopy(snapshot["outputs"]["replay_ledger"])
    elif poison == "output-schema":
        snapshot["outputs"]["transcript_bundle"]["schema"] = "synthetic-transcript"
    elif poison == "sample-missing":
        snapshot["samples"][0].pop("model_response_sha256")
    elif poison == "sample-extra":
        snapshot["samples"][0]["extra"] = None
    elif poison == "sample-type":
        snapshot["samples"][0]["sample_index"] = True
    elif poison == "fixture-row-nonzero":
        snapshot["samples"][1]["fixture_row_index"] = 1
    elif poison == "reward-int-alias":
        snapshot["samples"][0]["raw_environment_reward"] = 1
    elif poison == "reward-negative-zero":
        snapshot["samples"][0]["match_details"]["passed"] = False
        snapshot["samples"][0]["raw_environment_reward"] = -0.0
    elif poison == "citation-passed-inconsistent":
        snapshot["samples"][0]["match_details"]["missing"] = ["[1]"]
    else:  # pragma: no cover - the parameter list closes this set.
        raise AssertionError("unreachable snapshot poison")

    capability = object()
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_source",
        lambda **_kwargs: source,
    )
    monkeypatch.setattr(
        coordinator,
        "load_authenticated_captured_replay_result_v2",
        lambda **_kwargs: capability,
    )
    monkeypatch.setattr(
        coordinator,
        "snapshot_authenticated_captured_replay_result_v2",
        lambda value: snapshot if value is capability else None,
    )

    with pytest.raises(coordinator.StrictCapturedReplayCoordinatorError):
        coordinator.consume_replay_result(**_consume_kwargs(tmp_path))


@pytest.mark.parametrize(
    ("field", "poison"),
    [
        ("submission_receipt_sha256", "0" * 64),
        ("candidate_job_id", "93002"),
        ("result_final_receipt_sha256", "e" * 64),
    ],
)
def test_consume_defers_oob_authority_rejection_to_authenticated_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    poison: str,
) -> None:
    source = _source(tmp_path)
    kwargs = _consume_kwargs(tmp_path)
    kwargs[field] = poison
    seen: dict[str, Any] = {}
    detached = False

    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_source",
        lambda **_kwargs: source,
    )

    def reject(**loader_kwargs: Any) -> None:
        seen.update(loader_kwargs)
        raise ValueError("synthetic authenticated FINAL rejection")

    def forbidden_snapshot(_value: object) -> None:
        nonlocal detached
        detached = True

    monkeypatch.setattr(
        coordinator,
        "load_authenticated_captured_replay_result_v2",
        reject,
    )
    monkeypatch.setattr(
        coordinator,
        "snapshot_authenticated_captured_replay_result_v2",
        forbidden_snapshot,
    )

    with pytest.raises(ValueError, match="authenticated FINAL rejection"):
        coordinator.consume_replay_result(**kwargs)
    assert seen[field] == poison
    assert detached is False


def test_consume_forwards_alternate_manifest_path_to_fail_closed_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    kwargs = _consume_kwargs(tmp_path)
    alternate = tmp_path / "alternate/replay-1.json"
    kwargs["replay_manifest_path"] = str(alternate)
    seen_path: str | None = None
    detached = False

    monkeypatch.setattr(
        coordinator,
        "load_authenticated_replay_source",
        lambda **_kwargs: source,
    )

    def reject(**loader_kwargs: Any) -> None:
        nonlocal seen_path
        seen_path = loader_kwargs["replay_execution_manifest_path"]
        raise ValueError("manifest is not at authenticated publication path")

    def forbidden_snapshot(_value: object) -> None:
        nonlocal detached
        detached = True

    monkeypatch.setattr(
        coordinator,
        "load_authenticated_captured_replay_result_v2",
        reject,
    )
    monkeypatch.setattr(
        coordinator,
        "snapshot_authenticated_captured_replay_result_v2",
        forbidden_snapshot,
    )

    with pytest.raises(ValueError, match="authenticated publication path"):
        coordinator.consume_replay_result(**kwargs)
    assert seen_path == str(alternate)
    assert detached is False


def test_consume_parser_requires_external_final_authorities_not_inventory() -> None:
    parser = coordinator._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, coordinator.argparse._SubParsersAction)
    )
    consume = subparsers.choices["consume"]
    options = {option for action in consume._actions for option in action.option_strings}
    assert {
        "--replay-manifest",
        "--replay-manifest-sha256",
        "--submission-receipt-sha256",
        "--candidate-job-id",
        "--result-final-receipt",
        "--result-final-receipt-sha256",
    } <= options
    assert "--result-inventory-sha256" not in options


def test_main_consume_forwards_exact_external_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _consume_kwargs(tmp_path)
    snapshot = _authenticated_result_snapshot(tmp_path)
    seen: dict[str, str] = {}

    def consume(**kwargs: str) -> coordinator.ConsumedReplayResult:
        seen.update(kwargs)
        return coordinator.ConsumedReplayResult(snapshot=snapshot)

    monkeypatch.setattr(coordinator, "consume_replay_result", consume)
    argv = [
        "consume",
        "--pair-manifest",
        expected["pair_manifest_path"],
        "--pair-manifest-sha256",
        expected["pair_manifest_sha256"],
        "--pair-submission-receipt",
        expected["pair_submission_receipt_path"],
        "--pair-submission-receipt-sha256",
        expected["pair_submission_receipt_sha256"],
        "--off-exit-receipt",
        expected["trusted_off_exit_receipt_path"],
        "--off-exit-receipt-sha256",
        expected["trusted_off_exit_receipt_sha256"],
        "--environment",
        expected["expected_environment"],
        "--profile-id",
        expected["expected_profile_id"],
        "--replay-manifest",
        expected["replay_manifest_path"],
        "--replay-manifest-sha256",
        expected["replay_manifest_sha256"],
        "--submission-receipt-sha256",
        expected["submission_receipt_sha256"],
        "--candidate-job-id",
        expected["candidate_job_id"],
        "--result-final-receipt",
        expected["result_final_receipt_path"],
        "--result-final-receipt-sha256",
        expected["result_final_receipt_sha256"],
    ]

    assert coordinator.main(argv) == 0
    assert seen == expected
    assert json.loads(capsys.readouterr().out) == coordinator._consume_json(
        coordinator.ConsumedReplayResult(snapshot=snapshot)
    )


def test_consume_json_projects_only_authenticated_snapshot_authority(
    tmp_path: Path,
) -> None:
    snapshot = _authenticated_result_snapshot(tmp_path)
    result = coordinator.ConsumedReplayResult(snapshot=snapshot)

    output = coordinator._consume_json(result)

    assert output["schema"].endswith("-v2")
    assert output["status"] == "authenticated-and-snapshotted"
    assert output["replay_manifest"] == snapshot["manifest"]
    assert output["submission_receipt"] == snapshot["submission_receipt"]
    assert output["pre_receipt"] == snapshot["pre_receipt"]
    assert output["exit_receipt"] == snapshot["exit_receipt"]
    assert output["result_final_receipt"] == snapshot["result_final_receipt"]
    assert output["result"]["inventory"] == snapshot["result_inventory"]
    assert output["result"]["evidence_index"] == snapshot["evidence_index"]
    assert output["result"]["outputs"] == snapshot["outputs"]
    assert output["result"]["samples"] == snapshot["samples"]
    assert "files" not in output["result"]
    assert "inventory_sha256" not in output["result"]
    output["result"]["inventory"]["sha256"] = "0" * 64
    assert snapshot["result_inventory"]["sha256"] == "a" * 64


def test_coordinator_imports_only_public_lifecycle_apis() -> None:
    module_path = Path(coordinator.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    lifecycle_modules = {
        "nemo_rl.utils.strict_captured_replay_manifest_v2",
        "nemo_rl.utils.strict_captured_replay_evidence_v2",
    }
    imported: dict[str, set[str]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module in lifecycle_modules:
            imported[statement.module] = {alias.name for alias in statement.names}
    assert set(imported) == lifecycle_modules
    assert all(not name.startswith("_") for names in imported.values() for name in names)
    assert {
        "load_authenticated_captured_replay_result_v2",
        "snapshot_authenticated_captured_replay_result_v2",
    } <= imported["nemo_rl.utils.strict_captured_replay_evidence_v2"]
    source = module_path.read_text(encoding="utf-8")
    assert "strict_captured_replay_seal_v2" not in source
    assert "verify_sealed_result_v2" not in source
    assert "consume_verified_sealed_result_v2" not in source
    assert "build_pair79_replay_export" not in coordinator.__all__
    assert "publish_pair79_replay_export" not in coordinator.__all__
    assert "build_replay_launcher_argv" not in coordinator.__all__
