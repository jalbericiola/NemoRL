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

import base64
import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_PATH = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/collect_strict_single_env_terminal_jobs.py"


def _load_collector() -> ModuleType:
    spec = importlib.util.spec_from_file_location("strict_terminal_collector", COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = _load_collector()


@pytest.fixture(autouse=True)
def _allow_pytest_private_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    original = COLLECTOR._trusted_ancestor_metadata

    def validate(path: Path, metadata: os.stat_result) -> bool:
        if str(path) == "/private/tmp":
            return (
                stat.S_ISDIR(metadata.st_mode)
                and metadata.st_uid == 0
                and metadata.st_gid == 0
                and stat.S_IMODE(metadata.st_mode) == 0o1777
            )
        return original(path, metadata)

    monkeypatch.setattr(COLLECTOR, "_trusted_ancestor_metadata", validate)


def _digest(raw: bytes | str) -> str:
    if isinstance(raw, str):
        raw = raw.encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _document(value: Any) -> Any:
    raw = COLLECTOR.canonical_json_bytes(value, "test document") + b"\n"
    return COLLECTOR.Document(value=value, raw=raw, sha256=_digest(raw))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    tmp_path.chmod(0o700)
    scontrol = tmp_path / "scontrol"
    scontrol.write_bytes(b"synthetic pinned scontrol\n")
    scontrol.chmod(0o755)
    slurm_conf = tmp_path / "slurm.conf"
    slurm_conf.write_bytes(b"synthetic pinned slurm configuration\n")
    slurm_conf.chmod(0o644)
    collector = tmp_path / "terminal-collector.py"
    collector.write_bytes(COLLECTOR_PATH.read_bytes())
    collector.chmod(0o500)
    monkeypatch.setattr(
        COLLECTOR,
        "HSG_SLURM_CONF",
        {"path": str(slurm_conf), "sha256": _digest(slurm_conf.read_bytes())},
    )
    pair_id = "strict-spfx-ab"
    environment = "reasoning_gym"
    nonce = "nonce-123"
    pair_value = {
        "schema": COLLECTOR.PAIR_SCHEMA,
        "pair_id": pair_id,
        "selection": {"environment": environment},
        "arms": {"off": "observe", "on": "train"},
        "wandb": {
            "arms": {
                "off": {"name": f"off-{environment}-{pair_id}"},
                "on": {"name": f"on-{environment}-{pair_id}"},
            }
        },
        "runtime_tools": {
            "manifest": {"path": "/results/runtime-tools.json", "sha256": _digest("runtime-tools")},
            "document": {"host": {"scontrol": {"path": str(scontrol), "sha256": _digest(scontrol.read_bytes())}}},
        },
        "source": {
            "snapshots": {
                "off": {"path": "/results/snapshots/off"},
                "on": {"path": "/results/snapshots/on"},
            }
        },
        "scheduler_submission": {
            "contract": {"path": "/results/submission-contract.json", "sha256": _digest("contract")},
            "receipt": {"path": "/results/submission-receipt.json"},
        },
        "execution_environment": {
            "arms": {
                "off": {"results_dir": "/results/off"},
                "on": {"results_dir": "/results/on"},
            }
        },
        "determinism_receipt_dir": "shared_prefix_determinism_receipts",
        "campaign": {"slurm": {"account": "nemotron_sw_post", "partition": "batch"}},
    }
    pair = _document(pair_value)
    job_ids = {"off": "41001", "on": "41002"}
    comments = {arm: f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair.sha256}" for arm in ("off", "on")}
    submission = _document(
        {
            "schema": COLLECTOR.SUBMISSION_SCHEMA,
            "outcome": "released",
            "stage": "complete",
            "pair": {"id": pair_id, "manifest": {"sha256": pair.sha256}},
            "wandb": copy.deepcopy(pair_value["wandb"]),
            "submission_nonce": nonce,
            "held_submissions": {arm: {"candidate_job_id": job_ids[arm]} for arm in ("off", "on")},
            "authenticated_jobs": {
                arm: [
                    {
                        "job_id": job_ids[arm],
                        "job_name": pair_value["wandb"]["arms"][arm]["name"],
                        "comment": comments[arm],
                        "user_id": str(os.geteuid()),
                    }
                ]
                for arm in ("off", "on")
            },
        }
    )
    exits = {}
    for arm in ("off", "on"):
        exits[arm] = _document(
            {
                "schema": COLLECTOR.JOB_RECEIPT_SCHEMA,
                "phase": "EXIT",
                "post_verified": True,
                "driver_exit_code": 0,
                "pair_id": pair_id,
                "environment": environment,
                "arm": arm,
                "job_id": job_ids[arm],
                "job_name": pair_value["wandb"]["arms"][arm]["name"],
                "restart_count": 0,
                "pair_manifest_sha256": pair.sha256,
                "submission_receipt_sha256": submission.sha256,
                "submission_receipt_path": pair_value["scheduler_submission"]["receipt"]["path"],
                "submission_contract_path": pair_value["scheduler_submission"]["contract"]["path"],
                "submission_contract_sha256": pair_value["scheduler_submission"]["contract"]["sha256"],
                "runtime_tool_manifest_sha256": pair_value["runtime_tools"]["manifest"]["sha256"],
                "selection": copy.deepcopy(pair_value["selection"]),
                "source": copy.deepcopy(pair_value["source"]),
                "execution_environment": copy.deepcopy(pair_value["execution_environment"]),
                "job_account": "nemotron_sw_post",
                "job_partition": "batch",
                "job_num_nodes": 1,
                "gpus_per_node": 4,
                "runtime_attestation_receipt_dir": (
                    f"/results/{arm}/shared_prefix_determinism_receipts/{job_ids[arm]}-0"
                ),
            }
        )
    return SimpleNamespace(
        pair=pair,
        submission=submission,
        exits=exits,
        pair_value=pair_value,
        job_ids=job_ids,
        comments=comments,
        scontrol=scontrol,
        slurm_conf=slurm_conf,
        collector=collector,
    )


def _scontrol_raw(fixture: SimpleNamespace, arm: str, state: str = "COMPLETED") -> bytes:
    job_id = fixture.job_ids[arm]
    value = {
        "errors": [],
        "jobs": [
            {
                "job_id": int(job_id),
                "job_state": [state],
                "name": fixture.pair_value["wandb"]["arms"][arm]["name"],
                "comment": fixture.comments[arm],
                "user_id": os.geteuid(),
                "restart_cnt": 0,
                "current_working_directory": fixture.pair_value["source"]["snapshots"][arm]["path"],
                "state_reason": "None",
                "hold": False,
                "start_time": {"set": True, "infinite": False, "number": 1_788_000_000},
                "end_time": {"set": True, "infinite": False, "number": 1_788_000_100},
                "exit_code": {
                    "status": ["SUCCESS"],
                    "return_code": {"set": True, "infinite": False, "number": 0},
                    "signal": {
                        "id": {"set": False, "infinite": False, "number": 0},
                        "name": "",
                    },
                },
                "derived_exit_code": {
                    "status": ["SUCCESS"],
                    "return_code": {"set": True, "infinite": False, "number": 0},
                    "signal": {
                        "id": {"set": False, "infinite": False, "number": 0},
                        "name": "",
                    },
                },
                "billable_tres": {"number": 0.0},
            }
        ],
        "last_backfill": {"set": True, "infinite": False, "number": 1_788_437_041},
        "last_update": {"set": True, "infinite": False, "number": 0},
        "meta": {
            "plugin": {
                "type": "",
                "name": "",
                "data_parser": "data_parser/v0.0.44",
                "accounting_storage": "accounting_storage/slurmdbd",
            },
            "client": {"source": "", "user": "jalbericiola", "group": "dip"},
            "command": ["show", "job"],
            "slurm": {
                "version": {"major": "25", "micro": "6", "minor": "11"},
                "release": "25.11.6",
                "cluster": "oci-hsg-cs-001",
            },
        },
        "warnings": [],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


def _capture(
    fixture: SimpleNamespace,
    arm: str,
    output: Path,
    responses: list[bytes],
    *,
    collector_path: Path | None = None,
    expected_collector_sha256: str | None = None,
) -> dict[str, Any]:
    if collector_path is None:
        collector_path = fixture.collector
    calls = []

    def runner(argv: Any) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout=responses.pop(0), stderr=b"")

    sleeps = []
    value = COLLECTOR.capture_arm(
        pair_document=fixture.pair,
        submission_document=fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        expected_collector_sha256=(
            _digest(collector_path.read_bytes()) if expected_collector_sha256 is None else expected_collector_sha256
        ),
        arm=arm,
        output=output,
        collector_path=collector_path,
        poll_interval_seconds=0.1,
        timeout_seconds=10.0,
        runner=runner,
        sleeper=sleeps.append,
        monotonic=iter((0.0, 0.1, 0.2, 0.3)).__next__,
        time_ns=iter(
            (
                1_788_000_000_000_000_000,
                1_788_000_000_000_000_001,
                1_788_000_000_000_000_002,
                1_788_000_000_000_000_003,
            )
        ).__next__,
        system_owner=os.geteuid(),
        system_group=fixture.scontrol.stat().st_gid,
    )
    value["_test_calls"] = calls
    value["_test_sleeps"] = sleeps
    return value


def _capture_documents(fixture: SimpleNamespace, tmp_path: Path) -> dict[str, Any]:
    documents = {}
    for arm in ("off", "on"):
        path = tmp_path / f"{arm}-terminal.json"
        _capture(fixture, arm, path, [_scontrol_raw(fixture, arm)])
        documents[arm] = COLLECTOR.load_document(path, f"{arm} capture")
    return documents


def _compose_valid(
    fixture: SimpleNamespace,
    tmp_path: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    capture_documents = _capture_documents(fixture, tmp_path)
    output = tmp_path / "terminal-pair.json"
    receipt = COLLECTOR.compose_pair_receipt(
        pair_document=fixture.pair,
        submission_document=fixture.submission,
        exit_documents=fixture.exits,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
        expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        capture_documents=capture_documents,
        output=output,
        collector_path=fixture.collector,
    )
    return receipt, COLLECTOR.load_document(output, "terminal Pair receipt"), capture_documents


def test_capture_starts_without_exit_and_seals_active_then_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "off-terminal.json"

    capture = _capture(
        fixture,
        "off",
        output,
        [_scontrol_raw(fixture, "off", "RUNNING"), _scontrol_raw(fixture, "off")],
    )

    assert len(capture.pop("_test_calls")) == 2
    assert capture.pop("_test_sleeps") == [0.1]
    assert capture["terminal_record"]["job_state"] == "COMPLETED"
    assert capture["query"]["started_at_unix_ns"] == "1788000000000000002"
    assert capture["query"]["finished_at_unix_ns"] == "1788000000000000003"
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert output.stat().st_nlink == 1
    assert not list(tmp_path.glob(".*.candidate-*"))


def test_capture_requires_the_oob_collector_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "off-terminal.json"

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="trusted OOB"):
        _capture(
            fixture,
            "off",
            output,
            [_scontrol_raw(fixture, "off")],
            expected_collector_sha256=_digest("different reviewed collector"),
        )
    assert not output.exists()


@pytest.mark.parametrize("mode", [0o400, 0o700, 0o755])
def test_capture_rejects_nonexecutable_or_mutable_collector_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate-terminal-collector.py"
    candidate.write_bytes(COLLECTOR_PATH.read_bytes())
    candidate.chmod(mode)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="metadata differs"):
        _capture(
            fixture,
            "off",
            tmp_path / "off-terminal.json",
            [_scontrol_raw(fixture, "off")],
            collector_path=candidate,
        )


@pytest.mark.parametrize(
    "hostile_time",
    [100, True, "0", "01788000000000000000", "100000000000000000000"],
)
def test_arm_replay_requires_canonical_decimal_string_nanoseconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_time: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "off-terminal.json"
    _capture(fixture, "off", output, [_scontrol_raw(fixture, "off")])
    capture = copy.deepcopy(COLLECTOR.load_document(output, "OFF capture").value)
    capture["query"]["started_at_unix_ns"] = hostile_time
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="decimal nanoseconds"):
        COLLECTOR.validate_arm_capture(
            _document(capture),
            context=context,
            pair_sha256=fixture.pair.sha256,
            submission_sha256=fixture.submission.sha256,
            collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )


def test_capture_compose_and_pure_replay_close_both_exit_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    capture_documents = {}
    for arm in ("off", "on"):
        path = tmp_path / f"{arm}-terminal.json"
        _capture(fixture, arm, path, [_scontrol_raw(fixture, arm)])
        capture_documents[arm] = COLLECTOR.load_document(path, f"{arm} capture")
    output = tmp_path / "terminal-pair.json"

    receipt = COLLECTOR.compose_pair_receipt(
        pair_document=fixture.pair,
        submission_document=fixture.submission,
        exit_documents=fixture.exits,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
        expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        capture_documents=capture_documents,
        output=output,
        collector_path=fixture.collector,
    )

    document = COLLECTOR.load_document(output, "terminal Pair receipt")
    assert receipt == document.value
    assert (
        COLLECTOR.validate_pair_receipt(
            document,
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )
        == receipt
    )
    assert receipt["job_exit_receipt_sha256s"] == {arm: fixture.exits[arm].sha256 for arm in ("off", "on")}
    with pytest.raises(COLLECTOR.TerminalCollectionError, match="output already exists"):
        COLLECTOR.compose_pair_receipt(
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
            capture_documents=capture_documents,
            output=output,
            collector_path=fixture.collector,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [("post_verified", False), ("driver_exit_code", 1), ("job_id", "99999")],
)
def test_compose_requires_semantically_valid_exit_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    captures = _capture_documents(fixture, tmp_path)
    exits = dict(fixture.exits)
    poisoned = copy.deepcopy(exits["off"].value)
    poisoned[key] = value
    exits["off"] = _document(poisoned)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="OFF receipt|off EXIT receipt"):
        COLLECTOR.compose_pair_receipt(
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
            capture_documents=captures,
            output=tmp_path / "terminal-pair.json",
            collector_path=fixture.collector,
        )


def test_compose_rejects_missing_exit_field_without_leaking_keyerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    captures = _capture_documents(fixture, tmp_path)
    exits = dict(fixture.exits)
    poisoned = copy.deepcopy(exits["off"].value)
    del poisoned["post_verified"]
    exits["off"] = _document(poisoned)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="structure is incomplete"):
        COLLECTOR.compose_pair_receipt(
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
            capture_documents=captures,
            output=tmp_path / "terminal-pair.json",
            collector_path=fixture.collector,
        )


def test_pure_validator_replays_poisoned_raw_scheduler_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt, _, _ = _compose_valid(fixture, tmp_path)
    poisoned = copy.deepcopy(receipt)
    capture = poisoned["captures"]["off"]
    raw = base64.b64decode(capture["query"]["raw_stdout_base64"])
    scheduler = json.loads(raw)
    scheduler["jobs"][0]["job_state"] = ["CANCELLED"]
    raw = json.dumps(scheduler, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    capture["query"]["raw_stdout_base64"] = base64.b64encode(raw).decode("ascii")
    capture["query"]["raw_stdout_sha256"] = _digest(raw)
    capture["query"]["raw_stdout_byte_count"] = len(raw)
    capture_raw = COLLECTOR.canonical_json_bytes(capture, "poisoned capture") + b"\n"
    poisoned["capture_sha256s"]["off"] = _digest(capture_raw)
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": fixture.pair.sha256,
        "submission_receipt_sha256": fixture.submission.sha256,
        "job_exit_receipt_sha256s": poisoned["job_exit_receipt_sha256s"],
        "capture_sha256s": poisoned["capture_sha256s"],
    }
    poisoned["composition_sha256"] = _digest(COLLECTOR.canonical_json_bytes(composition, "poisoned composition"))

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="non-success terminal state"):
        COLLECTOR.validate_pair_receipt(
            _document(poisoned),
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )


def test_pure_validator_replays_and_rejects_nonzero_derived_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt, _, _ = _compose_valid(fixture, tmp_path)
    poisoned = copy.deepcopy(receipt)
    capture = poisoned["captures"]["off"]
    raw = base64.b64decode(capture["query"]["raw_stdout_base64"])
    scheduler = json.loads(raw)
    scheduler["jobs"][0]["derived_exit_code"]["return_code"]["number"] = 49
    raw = json.dumps(scheduler, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    capture["query"]["raw_stdout_base64"] = base64.b64encode(raw).decode("ascii")
    capture["query"]["raw_stdout_sha256"] = _digest(raw)
    capture["query"]["raw_stdout_byte_count"] = len(raw)
    capture_raw = COLLECTOR.canonical_json_bytes(capture, "poisoned capture") + b"\n"
    poisoned["capture_sha256s"]["off"] = _digest(capture_raw)
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": fixture.pair.sha256,
        "submission_receipt_sha256": fixture.submission.sha256,
        "job_exit_receipt_sha256s": poisoned["job_exit_receipt_sha256s"],
        "capture_sha256s": poisoned["capture_sha256s"],
    }
    poisoned["composition_sha256"] = _digest(COLLECTOR.canonical_json_bytes(composition, "poisoned composition"))

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="derived_exit_code return_code"):
        COLLECTOR.validate_pair_receipt(
            _document(poisoned),
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )


def test_pure_validator_accepts_an_equivalent_foreign_frozen_document_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass(frozen=True)
    class ForeignDocument:
        value: Any
        sha256: str
        raw: bytes

    fixture = _fixture(tmp_path, monkeypatch)
    receipt, document, _ = _compose_valid(fixture, tmp_path)
    foreign = ForeignDocument(value=document.value, sha256=document.sha256, raw=document.raw)

    assert (
        COLLECTOR.validate_pair_receipt(
            foreign,
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )
        == receipt
    )


def test_document_protocol_recomputes_value_raw_and_digest_consistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    poisoned_value = copy.deepcopy(fixture.pair.value)
    poisoned_value["pair_id"] = "different-pair"
    poisoned = COLLECTOR.Document(
        value=poisoned_value,
        raw=fixture.pair.raw,
        sha256=fixture.pair.sha256,
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="framing or internal digest differs"):
        COLLECTOR._submission_context(
            poisoned,
            fixture.submission,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            arm="off",
        )


@pytest.mark.parametrize("field", ["capture_sha256s", "composition_sha256"])
def test_pure_validator_recomputes_capture_and_composition_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt, _, _ = _compose_valid(fixture, tmp_path)
    poisoned = copy.deepcopy(receipt)
    if field == "capture_sha256s":
        poisoned[field]["off"] = _digest("different capture")
    else:
        poisoned[field] = _digest("different composition")

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="capture digest|composition digest"):
        COLLECTOR.validate_pair_receipt(
            _document(poisoned),
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest(COLLECTOR_PATH.read_bytes()),
        )


def test_compose_requires_the_same_oob_collector_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    captures = _capture_documents(fixture, tmp_path)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="trusted OOB"):
        COLLECTOR.compose_pair_receipt(
            pair_document=fixture.pair,
            submission_document=fixture.submission,
            exit_documents=fixture.exits,
            expected_pair_sha256=fixture.pair.sha256,
            expected_submission_sha256=fixture.submission.sha256,
            expected_exit_sha256s={arm: fixture.exits[arm].sha256 for arm in ("off", "on")},
            expected_collector_sha256=_digest("different reviewed collector"),
            capture_documents=captures,
            output=tmp_path / "terminal-pair.json",
            collector_path=fixture.collector,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("jobs", 0, "job_id"), True),
        (("jobs", 0, "restart_cnt"), False),
        (("jobs", 0, "hold"), 0),
        (("jobs", 0, "exit_code", "return_code", "number"), False),
        (("jobs", 0, "exit_code", "signal", "id", "number"), False),
        (("jobs", 0, "derived_exit_code", "return_code", "number"), False),
        (("jobs", 0, "derived_exit_code", "signal", "id", "number"), False),
        (("last_update", "number"), False),
    ],
)
def test_terminal_parser_rejects_integer_boolean_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[Any, ...],
    value: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    raw_value = json.loads(_scontrol_raw(fixture, "off"))
    target = raw_value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    raw = json.dumps(raw_value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError):
        COLLECTOR.normalize_scontrol_terminal(raw, context)


def test_terminal_parser_requires_derived_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    value = json.loads(_scontrol_raw(fixture, "off"))
    del value["jobs"][0]["derived_exit_code"]
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="derived_exit_code"):
        COLLECTOR.normalize_scontrol_terminal(raw, context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", ["FAILED"]),
        (
            "signal",
            {
                "id": {"set": True, "infinite": False, "number": 9},
                "name": "SIGKILL",
            },
        ),
    ],
)
def test_terminal_parser_rejects_failed_or_signaled_derived_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    scheduler = json.loads(_scontrol_raw(fixture, "off"))
    scheduler["jobs"][0]["derived_exit_code"][field] = value
    raw = json.dumps(scheduler, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="derived_exit_code"):
        COLLECTOR.normalize_scontrol_terminal(raw, context)


@pytest.mark.parametrize("state", ["CANCELLED", "FAILED", "TIMEOUT"])
def test_success_zero_never_overrides_noncompleted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="non-success terminal state"):
        COLLECTOR.normalize_scontrol_terminal(_scontrol_raw(fixture, "off", state), context)


@pytest.mark.parametrize("job_count", [0, 2])
def test_terminal_parser_rejects_missing_or_duplicate_exact_id_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_count: int,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    value = json.loads(_scontrol_raw(fixture, "off"))
    value["jobs"] = value["jobs"] * job_count
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    context = COLLECTOR._submission_context(
        fixture.pair,
        fixture.submission,
        expected_pair_sha256=fixture.pair.sha256,
        expected_submission_sha256=fixture.submission.sha256,
        arm="off",
    )

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="disappeared or duplicated"):
        COLLECTOR.normalize_scontrol_terminal(raw, context)


@pytest.mark.parametrize("hostile", [b'{"errors":[],"errors":[]}\n', b'{"x":-0}\n', b'{"x":-0.0}\n', b'{"x":NaN}\n'])
def test_strict_json_rejects_ambiguous_numeric_or_duplicate_encodings(hostile: bytes) -> None:
    with pytest.raises(COLLECTOR.TerminalCollectionError):
        COLLECTOR._parse_json(hostile, "hostile scheduler JSON")


def test_regular_0644_config_is_accepted_but_executable_files_are_enforced(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "tool-or-config"
    path.write_bytes(b"stable bytes\n")
    path.chmod(0o644)

    assert COLLECTOR._file_sha256(
        path,
        "Slurm configuration",
        require_executable=False,
        expected_owner=os.geteuid(),
        expected_group=path.stat().st_gid,
        exact_mode=0o644,
    ) == _digest(path.read_bytes())
    for label in ("scontrol", "terminal collector"):
        with pytest.raises(COLLECTOR.TerminalCollectionError, match="metadata differs"):
            COLLECTOR._file_sha256(
                path,
                label,
                require_executable=True,
                expected_owner=os.geteuid(),
                expected_group=path.stat().st_gid,
                exact_mode=0o644,
            )


def test_stable_hash_rejects_mutation_during_fd_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path.chmod(0o700)
    path = tmp_path / "mutating-config"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    path.chmod(0o644)
    original_read = COLLECTOR.os.read
    mutated = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if not mutated:
            mutated = True
            with path.open("ab") as stream:
                stream.write(b"hostile mutation")
        return chunk

    monkeypatch.setattr(COLLECTOR.os, "read", racing_read)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="grew|changed"):
        COLLECTOR._file_sha256(
            path,
            "mutating config",
            require_executable=False,
            expected_owner=os.geteuid(),
            expected_group=path.stat().st_gid,
            exact_mode=0o644,
        )


def test_loaded_documents_require_mode_0400_and_one_link(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    raw = COLLECTOR.canonical_json_bytes({"schema": "test"}, "test document") + b"\n"
    wrong_mode = tmp_path / "wrong-mode.json"
    wrong_mode.write_bytes(raw)
    wrong_mode.chmod(0o600)
    linked = tmp_path / "linked.json"
    linked.write_bytes(raw)
    linked.chmod(0o400)
    alias = tmp_path / "linked-alias.json"
    os.link(linked, alias)

    for path in (wrong_mode, linked):
        with pytest.raises(COLLECTOR.TerminalCollectionError, match="metadata differs"):
            COLLECTOR.load_document(path, "hostile document")


def test_exclusive_stage_removes_publication_if_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "terminal.json"
    original_fsync = COLLECTOR.os.fsync
    calls = 0

    def fail_first_directory_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(COLLECTOR.os, "fsync", fail_first_directory_sync)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="cannot stage terminal evidence"):
        COLLECTOR._stage_exclusive(output, b"sealed evidence\n")
    assert not output.exists()
    assert not list(tmp_path.glob(".*.candidate-*"))


def test_pair_requires_the_final3a_runtime_attestation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    pair_value = copy.deepcopy(fixture.pair.value)
    pair_value["determinism_receipt_dir"] = "strict_pair_runtime_attestations"
    pair = _document(pair_value)
    submission_value = copy.deepcopy(fixture.submission.value)
    submission_value["pair"]["manifest"]["sha256"] = pair.sha256
    for arm in ("off", "on"):
        submission_value["authenticated_jobs"][arm][0][
            "comment"
        ] = f"nemo-rl-strict-pair-v1:{arm}:{submission_value['submission_nonce']}:{pair.sha256}"
    submission = _document(submission_value)

    with pytest.raises(COLLECTOR.TerminalCollectionError, match="attestation directory differs"):
        COLLECTOR._submission_context(
            pair,
            submission,
            expected_pair_sha256=pair.sha256,
            expected_submission_sha256=submission.sha256,
            arm="off",
        )


def test_hsg_shared_project_exception_is_exact() -> None:
    exact = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o775,
        st_uid=0,
        st_gid=20330,
    )
    path = Path(COLLECTOR.HSG_SHARED_PROJECT_ANCESTOR["path"])

    assert COLLECTOR._trusted_ancestor_metadata(path, exact)
    for mutation in (
        SimpleNamespace(st_mode=stat.S_IFDIR | 0o777, st_uid=0, st_gid=20330),
        SimpleNamespace(st_mode=stat.S_IFDIR | 0o775, st_uid=1, st_gid=20330),
        SimpleNamespace(st_mode=stat.S_IFDIR | 0o775, st_uid=0, st_gid=1),
    ):
        assert not COLLECTOR._trusted_ancestor_metadata(path, mutation)
    assert not COLLECTOR._trusted_ancestor_metadata(path / "hostile", exact)


def test_relative_and_dotdot_paths_fail_before_open(tmp_path: Path) -> None:
    for path in (Path("relative.json"), tmp_path / "child" / ".." / "receipt.json"):
        with pytest.raises(COLLECTOR.TerminalCollectionError, match="lexical-canonical absolute"):
            COLLECTOR._parent_and_leaf(path, "hostile path")
