# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemo_rl.algorithms import strict_captured_replay_runtime as replay

_RUNNER_PATH = Path(__file__).resolve().parents[3] / "examples/nemo_gym/run_strict_captured_replay.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("_strict_captured_replay_runner_test_module", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


def _resource_receipt(
    interpreter: str = "/strict/resource/.venv/bin/python",
) -> dict[str, Any]:
    return {
        "schema": "nemo-rl-strict-gym-child-receipt-v1",
        "hash_domain": "sha256-canonical-ascii-json-no-lf-v1",
        "environment": "reasoning_gym",
        "pair_id": "pair-1",
        "job_id": "12345",
        "stage": "isolated-runner-pre-entrypoint",
        "spec_sha256": "1" * 64,
        "target": {"role": "resource", "interpreter": interpreter},
        "server": {},
        "process": {"sys_executable": interpreter},
        "distribution_versions": {},
        "module_versions": {},
        "scorer": {},
    }


def _derived_request(
    text: str = "<answer>Zoey fool Zoey sage</answer>",
) -> dict[str, Any]:
    return {
        "question": "What are Zoey and Riley?",
        "answer": "Zoey fool Zoey sage; Riley sage",
        "metadata": {"source_dataset": "knights_knaves"},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        },
    }


def _verifier_response(*, extracted_answer: str = "Zoey fool Zoey sage", score: float = 0.5) -> dict[str, Any]:
    return {
        "task_name": "knights_knaves",
        "extracted_answer": extracted_answer,
        "score": score,
        "reward": score,
    }


def test_pair_id_grammar_rejects_unicode_alphanumeric() -> None:
    with pytest.raises(replay.StrictCapturedReplayError, match="pair_id"):
        replay._safe_pair_id("pair-é")


def test_independent_score_projection_rejects_forged_response_alias() -> None:
    with pytest.raises(replay.StrictCapturedReplayError, match="aliases differ"):
        replay.reasoning_gym_score_call_material(
            derived_verifier_request=_derived_request(),
            verifier_response=_verifier_response(extracted_answer="forged"),
        )


def test_independent_score_projection_rejects_reward_alias_mismatch() -> None:
    response = _verifier_response()
    response["reward"] = 0.25
    with pytest.raises(replay.StrictCapturedReplayError, match="outside the exact domain"):
        replay.reasoning_gym_score_call_material(
            derived_verifier_request=_derived_request(),
            verifier_response=response,
        )


def test_replay_uses_child_terminal_instead_of_secondary_python_scorer() -> None:
    runtime_source = Path(replay.__file__).read_text(encoding="utf-8")
    runner_source = _RUNNER_PATH.read_text(encoding="utf-8")
    assert "subprocess.run" not in runtime_source
    assert "verify_reasoning_gym_score_with_child" not in runtime_source
    assert "verify_reasoning_gym_score_with_child" not in runner_source
    assert "session.finalize_score_calls(" in runner_source
    assert "expected_calls, run_helper=run_helper" in runner_source


def test_runner_matches_authenticated_source_api_and_pair74_roster() -> None:
    from nemo_rl.utils.strict_captured_replay_evidence import (
        load_captured_replay_submission_receipt,
    )
    from nemo_rl.utils.strict_captured_replay_manifest import SLURM_EXPORT_ALLOWED_NAMES

    parameters = inspect.signature(load_captured_replay_submission_receipt).parameters
    assert parameters["authenticated_source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert runner._SLURM_EXPORT_ALLOWED_NAMES == SLURM_EXPORT_ALLOWED_NAMES
    assert dict(runner._RUNTIME_DEVICE_ENVIRONMENT_FIELDS) == {
        "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
        "GPU_DEVICE_ORDINAL": "gpu_device_ordinal",
        "NVIDIA_VISIBLE_DEVICES": "nvidia_visible_devices",
        "ROCR_VISIBLE_DEVICES": "rocr_visible_devices",
        "ZE_AFFINITY_MASK": "ze_affinity_mask",
    }
    assert set(dict(runner._RUNTIME_DEVICE_ENVIRONMENT_FIELDS)).isdisjoint(runner._SLURM_EXPORT_ALLOWED_NAMES)


class _FakeResponse:
    status = 200

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self, amount: int) -> bytes:
        del amount
        raw, self._raw = self._raw, b""
        return raw


class _FakeConnection:
    def __init__(self, raw: bytes) -> None:
        self.response = _FakeResponse(raw)

    def request(self, *args: Any, **kwargs: Any) -> None:
        pass

    def getresponse(self) -> _FakeResponse:
        return self.response

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "raw",
    [b'{"reward":0.0,"reward":1.0}', b'{"reward":NaN}', b"[]"],
)
def test_post_verify_rejects_malformed_http_json(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
    monkeypatch.setattr(
        replay.http.client,
        "HTTPConnection",
        lambda *args, **kwargs: _FakeConnection(raw),
    )
    with pytest.raises(replay.StrictCapturedReplayError):
        replay.post_resource_verify(
            host="127.0.0.1",
            port=5000,
            derived_verifier_request={"request": True},
            timeout_seconds=1.0,
        )


def test_replay_module_has_no_model_generation_imports() -> None:
    source = Path(replay.__file__).read_text(encoding="utf-8")
    for forbidden in ("vllm", "Policy", "generate(", '"/run"'):
        assert forbidden not in source


def test_score_call_terminal_is_required_before_transport_finalize() -> None:
    execute_parameters = inspect.signature(replay.execute_captured_replay_cohort).parameters
    finalize_parameters = inspect.signature(replay.StrictModelTransportReplaySource.finalize).parameters
    assert "finalize_score_call_evidence" in execute_parameters
    assert "reasoning_score_call_index_ref" in finalize_parameters
    assert "driver_scheduler_device_environment" in execute_parameters
    assert "scheduler_device_environment" in finalize_parameters
    source = Path(replay.__file__).read_text(encoding="utf-8")
    terminal_offset = source.index("reasoning_score_call_index_ref =")
    consumption_offset = source.index("transport_source.finalize(")
    assert terminal_offset < consumption_offset


def test_runtime_transcript_bindings_use_direct_on_snapshot_and_non_wandb_run_id() -> None:
    manifest = {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "attempt_id": "replay-1",
        "pair": {"manifest": {"sha256": "1" * 64}},
        "replay_contract": {
            "selected_config": {"sha256": "2" * 64},
            "source_snapshot": {
                "arm": "on",
                "ref": {
                    "config_sha256": "2" * 64,
                    "entrypoint_sha256": "3" * 64,
                    "manifest_sha256": "4" * 64,
                    "path": "/authenticated/on",
                },
            },
            "gym_scorer": {
                "resources": {"verifier_source": {"sha256": "5" * 64}},
            },
        },
        "artifacts": {"fixture": {"sha256": "6" * 64}},
        "wandb": {
            "enabled": False,
            "mode": "disabled",
            "reason": "scorer-only-replay-no-wandb-credentials-or-output",
        },
    }
    bindings = replay._transcript_bindings(
        manifest=manifest,
        submission_receipt_sha256="7" * 64,
        authenticated_job_id="12345",
    )
    assert bindings["snapshot_manifest_sha256"] == "4" * 64
    assert bindings["run_id"] == replay.replay_run_id(
        environment="reasoning_gym",
        pair_id="pair-1",
        attempt_id="replay-1",
    )
    assert "run_id" not in manifest["wandb"]


def test_runtime_normalizes_replay_verifier_material_to_owned_plain_dict() -> None:
    from nemo_rl.utils.strict_model_transport_replay import ReplayVerifierMaterial

    material = ReplayVerifierMaterial(
        rollout_index=0,
        generation_seed=42,
        model_response={"status": "completed"},
        agent_run_request={"metadata": {"source_dataset": "knights_knaves"}},
        derived_verifier_request={"question": "Q"},
        source_entry_sha256="1" * 64,
        request_body_sha256="2" * 64,
        response_body_sha256="3" * 64,
    )

    document = replay._material_document(material, rollout_index=0)

    assert type(document) is dict
    assert document == dict(material)
    document["model_response"]["status"] = "mutated"
    assert material.model_response == {"status": "completed"}


def _write_canonical(path: Path, document: dict[str, Any], *, trailing_lf: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode(
        "ascii"
    ) + (b"\n" if trailing_lf else b"")
    path.write_bytes(payload)
    path.chmod(0o400)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def immutable_tmp_path(tmp_path: Path):
    """Restore fixture permissions before pytest removes its temporary tree."""
    try:
        yield tmp_path
    finally:
        paths = sorted(tmp_path.rglob("*"), key=lambda item: len(item.parts), reverse=True)
        for path in paths:
            if path.is_symlink():
                continue
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except FileNotFoundError:
                pass
        tmp_path.chmod(0o700)


def _bootstrap_authority_files(tmp_path: Path, *, executable_cli: bool = False) -> SimpleNamespace:
    source_root = tmp_path / "snapshot"
    fake_program_payloads: dict[str, bytes] = {}
    gym_resource_payloads = {
        "config": b"reasoning-gym selected config\n",
        "requirements": b"reasoning-gym==test\n",
        "verifier_source": b"# authenticated reasoning-gym verifier\n",
    }
    resource_only_payload = (
        b"reasoning_gym:\n  resources_servers:\n    reasoning_gym:\n"
        b"      entrypoint: app.py\n      domain: knowledge\n      verified: false\n"
    )
    extra_snapshot_files: dict[str, bytes] = {
        f"{runner._GYM_SOURCE_RELATIVE}/nemo_gym/__init__.py": (
            b"from pathlib import Path\n"
            b"PARENT_DIR = Path(__file__).resolve().parents[1]\n"
            b"AUTHENTICATED_NEMO_GYM = True\n"
        ),
        **{
            f"{runner._GYM_SOURCE_RELATIVE}/{runner._GYM_RESOURCE_PATHS[name]}": payload
            for name, payload in gym_resource_payloads.items()
        },
        f"{runner._GYM_SOURCE_RELATIVE}/{runner._GYM_RESOURCE_ONLY_CONFIG['path']}": resource_only_payload,
    }
    if executable_cli:
        fake_program_payloads = {
            "entrypoint": _RUNNER_PATH.read_bytes(),
            "evidence_utility": b"""
import json
def load_captured_replay_submission_receipt(
    *, path, expected_sha256, replay_execution_manifest, authenticated_source
):
    if not hasattr(authenticated_source, 'pair_manifest'):
        raise RuntimeError('submission loader missing authenticated_source capability')
    with open(path, 'rb') as stream:
        return json.loads(stream.read().decode('ascii')), expected_sha256
def canonical_ascii_json(value): return b'{}'
def publish_evidence_document(**kwargs): raise AssertionError('not reached')
def validate_captured_replay_step1_ledger(*args, **kwargs): raise AssertionError('not reached')
def validate_transcript_bundle(*args, **kwargs): raise AssertionError('not reached')
""",
            "manifest_utility": b"""
import nemo_gym
from types import SimpleNamespace
if getattr(nemo_gym, 'AUTHENTICATED_NEMO_GYM', False) is not True:
    raise RuntimeError('AMBIENT_GYM_EXECUTED')
def load_authenticated_off_source_capture(**kwargs):
    return SimpleNamespace(pair_manifest=kwargs['pair_manifest'])
def load_replay_execution_manifest(*, path, expected_sha256, authenticated_source):
    if not hasattr(authenticated_source, 'pair_manifest'):
        raise RuntimeError('missing authenticated_source capability')
    raise RuntimeError('PRODUCTION_LOADER_AUTHENTICATED_SOURCE_REACHED')
""",
            "raw_transport_owner": b"""
def load_strict_model_transport_replay_source(**kwargs): raise AssertionError('not reached')
def publish_strict_model_transport_replay_consumption(**kwargs): raise AssertionError('not reached')
def validate_strict_model_transport_replay_consumption(*args, **kwargs): raise AssertionError('not reached')
""",
        }
        extra_snapshot_files.update(
            {
                "ray.py": b"# authenticated fake ray module\n",
                "nemo_rl/__init__.py": b"",
                "nemo_rl/utils/__init__.py": b"",
                "nemo_rl/algorithms/__init__.py": b"",
                "nemo_rl/environments/__init__.py": b"",
            }
        )
    program: dict[str, dict[str, str]] = {}
    for index, (name, relative) in enumerate(runner._PROGRAM_PATHS.items()):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = fake_program_payloads.get(name, f"# authenticated program member {index}: {name}\n".encode("ascii"))
        path.write_bytes(payload)
        path.chmod(0o400)
        program[name] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    for relative, payload in extra_snapshot_files.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o400)
    selected_config = source_root / "examples/selected.yaml"
    selected_config.write_bytes(b"selected: true\n")
    selected_config.chmod(0o400)
    selected_config_sha = hashlib.sha256(selected_config.read_bytes()).hexdigest()
    training_entrypoint = source_root / "examples/run_grpo_single_controller.py"
    training_entrypoint.write_bytes(b"# pinned training entrypoint\n")
    training_entrypoint.chmod(0o400)
    training_entrypoint_sha = hashlib.sha256(training_entrypoint.read_bytes()).hexdigest()
    symlink_manifest = source_root / runner._SNAPSHOT_SYMLINK_MANIFEST
    _write_canonical(
        symlink_manifest,
        {"schema": "nemo-rl-strict-snapshot-symlinks-v1", "symlinks": {}},
        trailing_lf=True,
    )
    regular_paths = [
        *[reference["path"] for reference in program.values()],
        "examples/selected.yaml",
        "examples/run_grpo_single_controller.py",
        *extra_snapshot_files,
        runner._SNAPSHOT_SYMLINK_MANIFEST,
        runner._SNAPSHOT_MODE_MANIFEST,
    ]
    mode_manifest = source_root / runner._SNAPSHOT_MODE_MANIFEST
    _write_canonical(
        mode_manifest,
        {
            "regular_file_executable": {relative: False for relative in regular_paths},
            "schema": "nemo-rl-strict-snapshot-modes-v1",
        },
        trailing_lf=True,
    )
    snapshot_manifest = source_root / runner._SNAPSHOT_SHA_MANIFEST
    snapshot_payload = b"".join(
        hashlib.sha256((source_root / relative).read_bytes()).hexdigest().encode("ascii")
        + b"  "
        + relative.encode("ascii")
        + b"\n"
        for relative in regular_paths
    )
    snapshot_manifest.write_bytes(snapshot_payload)
    snapshot_manifest.chmod(0o400)
    snapshot_manifest_sha = hashlib.sha256(snapshot_payload).hexdigest()
    snapshot_ref = {
        "config_sha256": selected_config_sha,
        "entrypoint_sha256": training_entrypoint_sha,
        "manifest_sha256": snapshot_manifest_sha,
        "path": str(source_root),
    }

    pair_path = tmp_path / "results/PAIR_MANIFEST.json"
    results_root = pair_path.parent
    host_tools = {
        name: {
            "path": f"/usr/bin/{name}",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name in ("env", "python", "sbatch", "scancel", "scontrol")
    }
    gym_resources = {
        name: {
            "path": runner._GYM_RESOURCE_PATHS[name],
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in gym_resource_payloads.items()
    }
    gym_source = {
        "gitlink_commit": "3" * 40,
        "path": str(tmp_path / "gym-source"),
        "tree": "4" * 40,
    }
    pair = {key: {} for key in runner._PAIR_ROOT_KEYS}
    pair.update(
        {
            "schema": runner._PAIR_SCHEMA,
            "pair_id": "pair-1",
            "selection": {
                "environment": "reasoning_gym",
                "config": {
                    "path": "examples/selected.yaml",
                    "sha256": selected_config_sha,
                },
                "gym_resources": gym_resources,
            },
            "campaign": {
                "nodes": 1,
                "slurm": {
                    "account": "nemotron_sw_post",
                    "partition": "batch",
                    "qos": "normal",
                },
            },
            "paths": {
                "results_root": str(results_root),
                "cache_root": str(tmp_path / "cache"),
                "hf_home": str(tmp_path / "hf"),
                "snapshot_parent": str(tmp_path / "snapshots"),
            },
            "source": {
                "snapshots": {
                    "off": {**snapshot_ref, "path": str(tmp_path / "unused-off")},
                    "on": snapshot_ref,
                },
                "gym": gym_source,
            },
            "runtime_tools": {"document": {"host": host_tools}},
            "slurm_export_boundary": {
                "allowed_names": list(runner._SLURM_EXPORT_ALLOWED_NAMES),
            },
        }
    )
    pair_sha = _write_canonical(pair_path, pair, trailing_lf=True)
    pair_submission_path = results_root / "PAIR_SUBMISSION_RECEIPT.json"
    pair_submission = {key: {} for key in runner._PAIR_SUBMISSION_ROOT_KEYS}
    pair_submission.update(
        {
            "schema": runner._PAIR_SUBMISSION_SCHEMA,
            "outcome": "released",
            "stage": "complete",
            "rollback_confirmed": None,
            "cancellations": [],
            "pre_cancel_queries": [],
            "post_cancel_queries": [],
            "recovery_query": None,
            "pair": {
                "id": "pair-1",
                "manifest": {"path": str(pair_path), "sha256": pair_sha},
            },
            "receipt": {
                "path": str(pair_submission_path),
                "schema": runner._PAIR_SUBMISSION_SCHEMA,
            },
        }
    )
    pair_submission_sha = _write_canonical(pair_submission_path, pair_submission, trailing_lf=True)
    output_root = tmp_path / "results/captured_replay/replay-1"
    export_values = {name: "" for name in runner._SLURM_EXPORT_ALLOWED_NAMES}
    export_values.update(
        {
            "EXPECTED_GYM_GITLINK_COMMIT": gym_source["gitlink_commit"],
            "EXPECTED_GYM_TREE": gym_source["tree"],
            "PAIR_ID": "pair-1",
            "RESULTS_DIR": str(output_root),
            "STRICT_PAIR_ENVIRONMENT": "reasoning_gym",
            "STRICT_PREBUILT_SNAPSHOT_DIR": str(source_root),
        }
    )
    export_path = results_root / "captured_replay/slurm_exports/pair-1/replay-1.env"
    export_path.parent.mkdir(parents=True)
    export_payload = b"".join(
        name.encode("ascii") + b"=" + export_values[name].encode("ascii") + b"\0"
        for name in runner._SLURM_EXPORT_ALLOWED_NAMES
    )
    export_path.write_bytes(export_payload)
    export_path.chmod(0o400)
    slurm_export_boundary = {
        "schema": runner._REPLAY_SLURM_EXPORT_SCHEMA,
        "allowed_names": list(runner._SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "attempt_id": "replay-1",
        "format": "nul-separated-name-value",
        "get_user_env": False,
        "job_argv_template": [
            "--pair-manifest",
            "{pair_manifest_path}",
            "--pair-manifest-sha256",
            "{pair_manifest_sha256}",
            "--pair-submission-receipt",
            "{pair_submission_receipt_path}",
            "--pair-submission-receipt-sha256",
            "{pair_submission_receipt_sha256}",
            "--off-exit-receipt",
            "{trusted_off_exit_receipt_path}",
            "--off-exit-receipt-sha256",
            "{trusted_off_exit_receipt_sha256}",
            "--replay-manifest",
            "{replay_manifest_path}",
            "--replay-manifest-sha256",
            "{replay_manifest_sha256}",
        ],
        "path": str(export_path),
        "sha256": hashlib.sha256(export_payload).hexdigest(),
    }
    replay_submission_root = results_root / "captured_replay/replay_submission_state/pair-1/replay-1"
    replay_submission_path = replay_submission_root / "submission-receipt.json"
    off_exit_ref = {
        "path": str(results_root / "reasoning_gym/off/receipts/EXIT.json"),
        "schema": "nemo-rl-strict-pair-job-receipt-v2",
        "sha256": "6" * 64,
    }
    submission_nonce = "9" * 64
    job_name = "strict-replay-replay-1-pair-1"
    manifest = {key: {} for key in runner._MANIFEST_ROOT_KEYS}
    manifest.update(
        {
            "schema": runner._MANIFEST_SCHEMA,
            "hash_domain": runner._MANIFEST_HASH_DOMAIN,
            "pair_id": "pair-1",
            "environment": "reasoning_gym",
            "arm": "on",
            "mode": "fresh_verifier_reward_replay",
            "attempt_id": "replay-1",
            "pair": {
                "id": "pair-1",
                "environment": "reasoning_gym",
                "manifest": {
                    "path": str(pair_path),
                    "schema": "nemo-rl-strict-single-env-pair-v2",
                    "sha256": pair_sha,
                },
                "submission_receipt": {
                    "path": str(pair_submission_path),
                    "schema": runner._PAIR_SUBMISSION_SCHEMA,
                    "sha256": pair_submission_sha,
                },
            },
            "replay_contract": {
                "execution_scope": "scorer-only",
                "policy_execution": {
                    "backward": False,
                    "forward": False,
                    "optimizer": False,
                    "violation": "fail-closed",
                },
                "program": program,
                "selected_config": {
                    "path": "examples/selected.yaml",
                    "sha256": selected_config_sha,
                },
                "source_snapshot": {"arm": "on", "ref": snapshot_ref},
                "gym_scorer": {
                    "launcher": {
                        "log_wrapper": "forbidden",
                        "resource_only_config": dict(runner._GYM_RESOURCE_ONLY_CONFIG),
                    },
                    "resources": gym_resources,
                    "source": gym_source,
                    "source_root": {
                        "snapshot_relative_path": runner._GYM_SOURCE_RELATIVE,
                        "host_path": str(source_root / runner._GYM_SOURCE_RELATIVE),
                        "container_path": runner._GYM_CONTAINER_ROOT,
                    },
                },
            },
            "artifacts": {"outputs": {"directory": {"path": str(output_root)}}},
            "scheduler_submission": {
                "nonce": submission_nonce,
                "identity": {"job_name": job_name, "submitter_euid": os.geteuid()},
                "contract": {
                    "path": str(tmp_path / "submission-contract.json"),
                    "sha256": "8" * 64,
                },
                "receipt": {
                    "path": str(replay_submission_path),
                    "schema": runner._REPLAY_SUBMISSION_SCHEMA,
                },
            },
            "runtime_attestation_requirements": {"schema": "test-runtime"},
            "wandb": {
                "enabled": False,
                "mode": "disabled",
                "reason": "scorer-only-replay-no-wandb-credentials-or-output",
            },
            "runtime_tools": pair["runtime_tools"],
            "slurm_export_boundary": slurm_export_boundary,
            "source_capture": {"job_receipts": {"exit": off_exit_ref}},
            "source": pair["source"],
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = _write_canonical(manifest_path, manifest, trailing_lf=False)
    candidate = "12345"
    comment = f"nemo-rl-strict-captured-replay-v1:replay-1:{submission_nonce}:{manifest_sha}"

    def scheduler_query(phase: str, state: str, held: bool, reason: str, base: Path):
        raw_path = Path(str(base) + ".raw")
        raw_document = {
            "errors": [],
            "jobs": [
                {
                    "comment": comment,
                    "current_working_directory": str(source_root),
                    "hold": held,
                    "job_id": int(candidate),
                    "job_state": [state],
                    "name": job_name,
                    "restart_cnt": 0,
                    "state_reason": reason,
                    "user_id": os.geteuid(),
                }
            ],
            "last_backfill": {},
            "last_update": {},
            "meta": {},
            "warnings": [],
        }
        raw = json.dumps(raw_document, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        raw_path.chmod(0o400)
        document = {
            "schema": runner._SCHEDULER_QUERY_SCHEMA,
            "phase": phase,
            "argv": [
                host_tools["scontrol"]["path"],
                "show",
                "job",
                "--json",
                candidate,
            ],
            "path": str(raw_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "line_count": raw.count(b"\n"),
            "status": 0,
            "normalization": {
                "algorithm": "scontrol-show-job-json-v1",
                "complete": True,
                "duplicate_keys_rejected": True,
                "negative_zero_rejected": True,
                "nonfinite_numbers_rejected": True,
            },
            "records": [
                {
                    "job_id": candidate,
                    "job_name": job_name,
                    "comment": comment,
                    "user_id": str(os.geteuid()),
                    "work_dir": str(source_root),
                    "job_state": state,
                    "reason": reason,
                    "held": held,
                    "restart_count": 0,
                }
            ],
            "match_count": 1,
        }
        document_path = Path(str(base) + "-query.json")
        digest = _write_canonical(document_path, document, trailing_lf=True)
        return {
            "path": str(document_path),
            "schema": runner._SCHEDULER_QUERY_SCHEMA,
            "sha256": digest,
        }

    submission_query_base = replay_submission_root / "PRE_RELEASE.scontrol"
    pre_release_query = scheduler_query("PRE_RELEASE", "PENDING", True, "JobHeldUser", submission_query_base)
    accepted_path = replay_submission_root / "accepted.job-id"
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_bytes = f"{candidate}\n".encode("ascii")
    accepted_path.write_bytes(accepted_bytes)
    accepted_path.chmod(0o400)
    accepted = {
        "path": str(accepted_path),
        "sha256": hashlib.sha256(accepted_bytes).hexdigest(),
        "parsed_candidate_job_id": candidate,
        "format": "ascii-positive-decimal-lf",
        "mode": "0400",
    }
    sbatch_argv = [
        "/usr/bin/sbatch",
        "--parsable",
        "strict_pair_replay_job_wrapper.sh",
        "--pair-manifest",
        str(pair_path),
        "--pair-manifest-sha256",
        pair_sha,
        "--pair-submission-receipt",
        str(pair_submission_path),
        "--pair-submission-receipt-sha256",
        pair_submission_sha,
        "--off-exit-receipt",
        off_exit_ref["path"],
        "--off-exit-receipt-sha256",
        off_exit_ref["sha256"],
        "--replay-manifest",
        str(manifest_path),
        "--replay-manifest-sha256",
        manifest_sha,
    ]
    replay_submission = {
        "schema": runner._REPLAY_SUBMISSION_SCHEMA,
        "phase": "SUBMISSION",
        "status": "held-candidate-not-in-job-authenticated",
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": "replay-1",
        "replay_execution_manifest": {
            "path": str(manifest_path),
            "schema": runner._MANIFEST_SCHEMA,
            "sha256": manifest_sha,
        },
        "replay_source_snapshot": {"arm": "on", "ref": snapshot_ref},
        "submission_contract": manifest["scheduler_submission"]["contract"],
        "slurm_export_boundary": manifest["slurm_export_boundary"],
        "submission_launcher": {
            "path": str(source_root / program["submission_launcher"]["path"]),
            "sha256": program["submission_launcher"]["sha256"],
        },
        "job_wrapper": {
            "path": str(source_root / program["job_wrapper"]["path"]),
            "sha256": program["job_wrapper"]["sha256"],
        },
        "scheduler_client_environment": {
            "ambient_merge": False,
            "env": host_tools["env"],
            "variables": {
                "LC_ALL": "C",
                "SLURM_CONF": {
                    "path": str(tmp_path / "slurm.conf"),
                    "sha256": "7" * 64,
                },
            },
        },
        "scheduler_tools": {name: host_tools[name] for name in ("sbatch", "scancel", "scontrol")},
        "submission_nonce": submission_nonce,
        "job_name": job_name,
        "comment": comment,
        "submitter_euid": os.geteuid(),
        "sbatch": {
            "path": host_tools["sbatch"]["path"],
            "sha256": host_tools["sbatch"]["sha256"],
            "argv": sbatch_argv,
            "argv_sha256": runner._domain_sha256("captured-replay-sbatch-argv", sbatch_argv),
            "parsable_stdout": f"{candidate}\n",
            "parsable_stdout_sha256": hashlib.sha256(accepted_bytes).hexdigest(),
        },
        "candidate_job_id": candidate,
        "accepted_id_record": accepted,
        "pre_release_scheduler_query": pre_release_query,
        "submitted_at_unix_ns": 1,
    }
    replay_submission_sha = _write_canonical(replay_submission_path, replay_submission, trailing_lf=True)
    job_root = results_root / "captured_replay/replay_job_state/pair-1/replay-1/12345-0"
    pre_query = scheduler_query("PRE", "RUNNING", False, "None", job_root / "queries/PRE.scontrol")
    pre_path = job_root / "receipts/PRE.json"
    static_boundary = {name: manifest[name] for name in runner._PRE_STATIC_BOUNDARY_KEYS}
    pre = {
        "schema": runner._PRE_SCHEMA,
        "phase": "PRE",
        "status": "authenticated-pre",
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": "replay-1",
        "replay_execution_manifest": {
            "path": str(manifest_path),
            "schema": runner._MANIFEST_SCHEMA,
            "sha256": manifest_sha,
        },
        "submission_receipt": {
            "path": str(replay_submission_path),
            "schema": runner._REPLAY_SUBMISSION_SCHEMA,
            "sha256": replay_submission_sha,
        },
        "candidate_job_id": candidate,
        "authenticated_job_id": candidate,
        "job": {
            "account": "nemotron_sw_post",
            "name": job_name,
            "num_nodes": 1,
            "partition": "batch",
            "qos": "normal",
            "gpus_per_node": 4,
            "restart_count": 0,
        },
        "static_boundary": static_boundary,
        "pre_scheduler_query": pre_query,
        "output_precondition": {
            "path": str(output_root),
            "mode": "0700",
            "status": "absent",
        },
        "runtime_attestation_contract": manifest["runtime_attestation_requirements"],
        "execution_source_root": str(source_root),
        "driver": {
            "entrypoint": runner._PROGRAM_PATHS["entrypoint"],
            "invocation": "python-isolated-no-bytecode",
            "pre_receipt_path": str(pre_path),
        },
        "post_verified": False,
    }
    pre_sha = _write_canonical(pre_path, pre, trailing_lf=True)
    for directory in sorted(
        [source_root, *[path for path in source_root.rglob("*") if path.is_dir()]],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    return SimpleNamespace(
        source_root=source_root,
        program=program,
        pair=pair,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        pre=pre,
        pre_path=pre_path,
        pre_sha=pre_sha,
        output_root=output_root,
        export_values=export_values,
    )


@contextlib.contextmanager
def _without_loaded_nemo_modules():
    saved = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "nemo_rl" or name.startswith("nemo_rl.") or name == "nemo_gym" or name.startswith("nemo_gym.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "nemo_rl" or name.startswith("nemo_rl.") or name == "nemo_gym" or name.startswith("nemo_gym."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _install_bootstrap_environment(monkeypatch: pytest.MonkeyPatch, fixture: SimpleNamespace) -> None:
    monkeypatch.setattr(
        runner,
        "__file__",
        str(fixture.source_root / runner._PROGRAM_PATHS["entrypoint"]),
    )
    expected = {
        **fixture.export_values,
        **_runtime_device_environment(),
        "STRICT_PAIR_BOUND_JOB_ID": "12345",
        "STRICT_REPLAY_EXECUTION_SOURCE_ROOT": str(fixture.source_root),
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)


def _runtime_device_environment() -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "GPU_DEVICE_ORDINAL": "0,1,2,3",
        "NVIDIA_VISIBLE_DEVICES": "0,1,2,3",
        "ROCR_VISIBLE_DEVICES": "0,1,2,3",
        "ZE_AFFINITY_MASK": "0.0,1.0,2.0,3.0",
    }


def test_runner_authenticates_pre_and_program_before_repo_import(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = immutable_tmp_path
    fixture = _bootstrap_authority_files(tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    with _without_loaded_nemo_modules():
        authority = runner._authenticate_before_repo_imports(
            manifest_path=str(fixture.manifest_path),
            manifest_sha256=fixture.manifest_sha,
            pre_receipt_path=str(fixture.pre_path),
            pre_receipt_sha256=fixture.pre_sha,
        )
    assert authority.authenticated_job_id == "12345"
    assert authority.execution_source_root == fixture.source_root
    assert authority.manifest == fixture.manifest
    assert authority.scheduler_device_environment == {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "0,1,2,3",
        "rocr_visible_devices": "0,1,2,3",
        "ze_affinity_mask": "0.0,1.0,2.0,3.0",
    }


def test_runner_main_exact_cli_authenticates_then_invokes_driver(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    observed: list[Any] = []
    monkeypatch.setattr(runner, "_require_isolated_driver_process", lambda: None)
    monkeypatch.setattr(runner, "_run_authenticated_driver", observed.append)
    with _without_loaded_nemo_modules():
        status = runner.main(
            [
                "--replay-driver-phase",
                "--replay-manifest",
                str(fixture.manifest_path),
                "--replay-manifest-sha256",
                fixture.manifest_sha,
                "--pre-receipt",
                str(fixture.pre_path),
                "--pre-receipt-sha256",
                fixture.pre_sha,
            ]
        )
    assert status == 0
    assert len(observed) == 1
    assert (
        observed[0].snapshot_authentication["source_snapshot_manifest_sha256"]
        == fixture.manifest["replay_contract"]["source_snapshot"]["ref"]["manifest_sha256"]
    )


def test_runner_production_cli_reaches_authenticated_source_manifest_loader(
    immutable_tmp_path: Path,
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path, executable_cli=True)
    environment = os.environ.copy()
    environment.update(
        {
            **fixture.export_values,
            **_runtime_device_environment(),
            "STRICT_PAIR_BOUND_JOB_ID": "12345",
        }
    )
    entrypoint = fixture.source_root / runner._PROGRAM_PATHS["entrypoint"]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(entrypoint),
            "--replay-driver-phase",
            "--replay-manifest",
            str(fixture.manifest_path),
            "--replay-manifest-sha256",
            fixture.manifest_sha,
            "--pre-receipt",
            str(fixture.pre_path),
            "--pre-receipt-sha256",
            fixture.pre_sha,
        ],
        cwd=fixture.source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert b"PRODUCTION_LOADER_AUTHENTICATED_SOURCE_REACHED" in completed.stderr
    assert b"missing authenticated_source capability" not in completed.stderr


def test_runner_production_cli_rejects_ambient_editable_gym_shadow(
    immutable_tmp_path: Path,
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path, executable_cli=True)
    ambient_root = immutable_tmp_path / "ambient-editable"
    ambient_package = ambient_root / "nemo_gym"
    ambient_package.mkdir(parents=True)
    (ambient_package / "__init__.py").write_text(
        "raise RuntimeError('AMBIENT_GYM_EXECUTED')\n",
        encoding="ascii",
    )
    environment = os.environ.copy()
    environment.update(
        {
            **fixture.export_values,
            **_runtime_device_environment(),
            "STRICT_PAIR_BOUND_JOB_ID": "12345",
        }
    )
    entrypoint = fixture.source_root / runner._PROGRAM_PATHS["entrypoint"]
    bootstrap = (
        "import runpy,sys;"
        "ambient=sys.argv.pop(1);target=sys.argv.pop(1);"
        "sys.path.insert(0,ambient);sys.argv[0]=target;"
        "runpy.run_path(target,run_name='__main__')"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            bootstrap,
            str(ambient_root),
            str(entrypoint),
            "--replay-driver-phase",
            "--replay-manifest",
            str(fixture.manifest_path),
            "--replay-manifest-sha256",
            fixture.manifest_sha,
            "--pre-receipt",
            str(fixture.pre_path),
            "--pre-receipt-sha256",
            fixture.pre_sha,
        ],
        cwd=fixture.source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert b"PRODUCTION_LOADER_AUTHENTICATED_SOURCE_REACHED" in completed.stderr
    assert b"AMBIENT_GYM_EXECUTED" not in completed.stderr


@pytest.mark.parametrize("python_flags", [["-B"], ["-I"]])
def test_runner_production_cli_rejects_missing_isolation_flag(
    immutable_tmp_path: Path, python_flags: list[str]
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path, executable_cli=True)
    environment = os.environ.copy()
    entrypoint = fixture.source_root / runner._PROGRAM_PATHS["entrypoint"]
    completed = subprocess.run(
        [
            sys.executable,
            *python_flags,
            str(entrypoint),
            "--replay-driver-phase",
            "--replay-manifest",
            str(fixture.manifest_path),
            "--replay-manifest-sha256",
            fixture.manifest_sha,
            "--pre-receipt",
            str(fixture.pre_path),
            "--pre-receipt-sha256",
            fixture.pre_sha,
        ],
        cwd=fixture.source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert b"requires live Python -I -B isolation" in completed.stderr


def test_runner_reauthenticates_snapshot_immediately_before_repo_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = SimpleNamespace(
        manifest={},
        manifest_sha256="1" * 64,
        pair_manifest={},
        execution_source_root=Path("/authenticated/on"),
        snapshot_authentication={"status": "authenticated"},
    )
    monkeypatch.setattr(runner, "_require_isolated_driver_process", lambda: None)
    monkeypatch.setattr(
        runner,
        "_authenticate_snapshot_program",
        lambda **kwargs: {"status": "changed"},
    )
    monkeypatch.setattr(
        runner,
        "_activate_authenticated_source_roots",
        lambda root: pytest.fail("changed snapshot must not reach sys.path"),
    )
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="changed before repository activation",
    ):
        runner._run_authenticated_driver(authority)


def test_runner_rejects_changed_program_member_before_repo_import(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = immutable_tmp_path
    fixture = _bootstrap_authority_files(tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    runtime_path = fixture.source_root / fixture.program["runtime"]["path"]
    runtime_path.chmod(0o600)
    runtime_path.write_text("changed\n", encoding="ascii")
    runtime_path.chmod(0o400)
    with (
        _without_loaded_nemo_modules(),
        pytest.raises(runner.StrictCapturedReplayEntrypointError, match="snapshot member differs"),
    ):
        runner._authenticate_before_repo_imports(
            manifest_path=str(fixture.manifest_path),
            manifest_sha256=fixture.manifest_sha,
            pre_receipt_path=str(fixture.pre_path),
            pre_receipt_sha256=fixture.pre_sha,
        )


def test_snapshot_symlink_resolution_admits_only_authenticated_in_root_target() -> None:
    runner._validate_snapshot_symlink_targets(
        symlinks={"dir/link": "../target"},
        regular_paths={"target"},
        directory_paths={"", "dir"},
    )
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="escapes authenticated root"):
        runner._validate_snapshot_symlink_targets(
            symlinks={"dir/link": "../../outside"},
            regular_paths={"target"},
            directory_paths={"", "dir"},
        )


def test_snapshot_symlink_resolution_rejects_cycles() -> None:
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="cycle detected"):
        runner._validate_snapshot_symlink_targets(
            symlinks={"one": "two", "two": "one"},
            regular_paths={"target"},
            directory_paths={""},
        )


def test_snapshot_symlink_resolution_does_not_collapse_before_nested_expansion() -> None:
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="escapes authenticated root"):
        runner._validate_snapshot_symlink_targets(
            symlinks={"a": "d/b/../..", "d/b": "../safe"},
            regular_paths={"safe"},
            directory_paths={"", "d"},
        )


def test_snapshot_inventory_rejects_valid_cached_bytecode(
    immutable_tmp_path: Path,
) -> None:
    snapshot = immutable_tmp_path / "snapshot-with-stale-bytecode"
    package = snapshot / "package"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("VALUE = 'authenticated source'\n", encoding="ascii")
    cached = cache / "module.cpython-313.pyc"
    py_compile.compile(str(source), cfile=str(cached), doraise=True)
    for path in (source, cached):
        path.chmod(0o400)
    for directory in (cache, package, snapshot):
        directory.chmod(0o500)
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="cached bytecode"):
        runner._snapshot_tree_inventory(snapshot)


def test_runner_rejects_wrapper_job_binding_mismatch(immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = immutable_tmp_path
    fixture = _bootstrap_authority_files(tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    monkeypatch.setenv("STRICT_PAIR_BOUND_JOB_ID", "12346")
    with (
        _without_loaded_nemo_modules(),
        pytest.raises(
            runner.StrictCapturedReplayEntrypointError,
            match="STRICT_PAIR_BOUND_JOB_ID",
        ),
    ):
        runner._authenticate_before_repo_imports(
            manifest_path=str(fixture.manifest_path),
            manifest_sha256=fixture.manifest_sha,
            pre_receipt_path=str(fixture.pre_path),
            pre_receipt_sha256=fixture.pre_sha,
        )


def test_runner_rejects_nonempty_unused_pair74_environment_value(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    assert fixture.export_values["COMMAND"] == ""
    monkeypatch.setenv("COMMAND", "forged-command")
    with (
        _without_loaded_nemo_modules(),
        pytest.raises(
            runner.StrictCapturedReplayEntrypointError,
            match="authenticated replay Slurm export environment differs: COMMAND",
        ),
    ):
        runner._authenticate_before_repo_imports(
            manifest_path=str(fixture.manifest_path),
            manifest_sha256=fixture.manifest_sha,
            pre_receipt_path=str(fixture.pre_path),
            pre_receipt_sha256=fixture.pre_sha,
        )


def test_runner_reconstructs_exact_scheduler_device_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_rl.utils import strict_captured_replay_evidence as evidence

    for name, value in _runtime_device_environment().items():
        monkeypatch.setenv(name, value)
    observed = runner._live_scheduler_device_environment()
    expected = {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "0,1,2,3",
        "rocr_visible_devices": "0,1,2,3",
        "ze_affinity_mask": "0.0,1.0,2.0,3.0",
    }
    assert observed == expected
    assert evidence._validate_scheduler_device_environment(observed) == expected


def test_runner_scheduler_device_environment_allows_absent_optional_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    for name in (
        "GPU_DEVICE_ORDINAL",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        monkeypatch.delenv(name, raising=False)
    assert runner._live_scheduler_device_environment() == {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": None,
        "nvidia_visible_devices": None,
        "rocr_visible_devices": None,
        "ze_affinity_mask": None,
    }


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("CUDA_VISIBLE_DEVICES", None, "cuda_visible_devices"),
        ("CUDA_VISIBLE_DEVICES", "00,1,2,3", "canonical numerically distinct"),
        ("CUDA_VISIBLE_DEVICES", "0,1,2,2", "canonical numerically distinct"),
        ("GPU_DEVICE_ORDINAL", "3,2,1,0", "must equal cuda_visible_devices"),
        ("NVIDIA_VISIBLE_DEVICES", "0,1,2", "invalid GPU identities"),
        (
            "NVIDIA_VISIBLE_DEVICES",
            "GPU-AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA,"
            "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb,"
            "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc,"
            "GPU-dddddddd-dddd-dddd-dddd-dddddddddddd",
            "invalid GPU identities",
        ),
        ("ROCR_VISIBLE_DEVICES", "00,1,2,3", "canonical numerically distinct"),
        ("ZE_AFFINITY_MASK", "00.0,1.0,2.0,3.0", "invalid device tokens"),
        ("ZE_AFFINITY_MASK", "0.0,1.0,1.0,3.0", "identities repeat"),
        ("ZE_AFFINITY_MASK", "0,0.0,1,2", "identities repeat"),
    ),
)
def test_runner_rejects_invalid_scheduler_device_environment(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
    message: str,
) -> None:
    for environment_name, environment_value in _runtime_device_environment().items():
        monkeypatch.setenv(environment_name, environment_value)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match=message):
        runner._live_scheduler_device_environment()


def test_runner_authentication_rejects_missing_scheduler_cuda_boundary(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path)
    _install_bootstrap_environment(monkeypatch, fixture)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with (
        _without_loaded_nemo_modules(),
        pytest.raises(
            runner.StrictCapturedReplayEntrypointError,
            match="cuda_visible_devices",
        ),
    ):
        runner._authenticate_before_repo_imports(
            manifest_path=str(fixture.manifest_path),
            manifest_sha256=fixture.manifest_sha,
            pre_receipt_path=str(fixture.pre_path),
            pre_receipt_sha256=fixture.pre_sha,
        )


def test_runner_scheduler_query_v2_recomputes_normalized_record_from_raw(
    immutable_tmp_path: Path,
) -> None:
    fixture = _bootstrap_authority_files(immutable_tmp_path)
    reference = dict(fixture.pre["pre_scheduler_query"])
    query_path = Path(reference["path"])
    query = json.loads(query_path.read_text(encoding="ascii"))
    raw_path = Path(query["path"])
    raw_document = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_document["jobs"][0]["job_id"] = 12346
    raw = json.dumps(raw_document, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    raw_path.chmod(0o600)
    raw_path.write_bytes(raw)
    raw_path.chmod(0o400)
    query["sha256"] = hashlib.sha256(raw).hexdigest()
    query["byte_count"] = len(raw)
    query["line_count"] = raw.count(b"\n")
    query_path.chmod(0o600)
    reference["sha256"] = _write_canonical(query_path, query, trailing_lf=True)
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="scheduler identity differs"):
        runner._authenticate_scheduler_query(
            reference=reference,
            phase="PRE",
            expected_document_path=query_path,
            expected_raw_path=raw_path,
            job_id="12345",
            job_name="strict-replay-replay-1-pair-1",
            comment=None,
            user_id=str(os.geteuid()),
            work_dir=str(fixture.source_root),
            job_state="RUNNING",
            held=False,
            reason=None,
            scontrol_path="/usr/bin/scontrol",
        )


def test_runner_scheduler_query_v2_rejects_numeric_negative_zero() -> None:
    raw = b'{"errors":[],"jobs":[{}],"last_backfill":-0,"last_update":{},' b'"meta":{},"warnings":[]}\n'
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="negative zero"):
        runner._strict_scheduler_source_job(raw, phase="PRE")


@pytest.mark.parametrize("reason", (123, "None\x01", "None\x7f"))
def test_runner_scheduler_query_v2_rejects_unclean_post_reason(reason: Any) -> None:
    source = {
        "comment": "strict-comment",
        "current_working_directory": "/snapshot",
        "hold": False,
        "job_id": 123,
        "job_state": ["RUNNING"],
        "name": "strict-job",
        "restart_cnt": 0,
        "state_reason": reason,
        "user_id": 42,
    }
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="state_reason",
    ):
        runner._normalize_scheduler_source_job(source, phase="POST")


@pytest.mark.parametrize("field", ("last_backfill", "last_update", "meta"))
@pytest.mark.parametrize("bad_value", (None, [], 0, False, "invalid"))
def test_runner_scheduler_query_v2_rejects_non_object_metadata_fields(field: str, bad_value: Any) -> None:
    document = {
        "errors": [],
        "jobs": [{}],
        "last_backfill": {},
        "last_update": {},
        "meta": {},
        "warnings": [],
    }
    document[field] = bad_value
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="scheduler raw JSON metadata fields are not exact objects",
    ):
        runner._strict_scheduler_source_job(raw, phase="PRE")


def test_runner_has_no_eager_repository_imports() -> None:
    tree = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))
    eager_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            eager_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            eager_imports.append(node.module)
    assert not any(
        name == "nemo_rl" or name.startswith("nemo_rl.") or name == "nemo_gym" or name.startswith("nemo_gym.")
        for name in eager_imports
    )


def test_resource_only_parser_disables_agent_model_and_log_wrapper(
    immutable_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = immutable_tmp_path
    execution_root = tmp_path / "source"
    gym_root = execution_root / "3rdparty/Gym-workspace/Gym"
    config_path = gym_root / "resources_servers/reasoning_gym/configs/resources_only.yaml"
    config_path.parent.mkdir(parents=True)
    config_payload = (
        b"reasoning_gym:\n  resources_servers:\n    reasoning_gym:\n"
        b"      entrypoint: app.py\n      domain: knowledge\n      verified: false\n"
    )
    config_path.write_bytes(config_payload)
    config_path.chmod(0o400)

    class ParserConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    resolved = {
        "config_paths": [str(config_path)],
        "reasoning_gym": {"resources_servers": {"reasoning_gym": {"entrypoint": "app.py"}}},
    }
    nemo_gym = ModuleType("nemo_gym")
    nemo_gym.__path__ = []  # type: ignore[attr-defined]
    nemo_gym.PARENT_DIR = gym_root  # type: ignore[attr-defined]
    cli = ModuleType("nemo_gym.cli")
    cli.GlobalConfigDictParserConfig = ParserConfig  # type: ignore[attr-defined]
    global_config = ModuleType("nemo_gym.global_config")
    global_config.NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = {"config_paths"}  # type: ignore[attr-defined]
    global_config.get_global_config_dict = lambda **kwargs: resolved  # type: ignore[attr-defined]
    omega = ModuleType("omegaconf")
    omega.DictConfig = dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_gym", nemo_gym)
    monkeypatch.setitem(sys.modules, "nemo_gym.cli", cli)
    monkeypatch.setitem(sys.modules, "nemo_gym.global_config", global_config)
    monkeypatch.setitem(sys.modules, "omegaconf", omega)
    manifest = {
        "replay_contract": {
            "gym_scorer": {
                "launcher": {
                    "log_wrapper": "forbidden",
                    "resource_only_config": {
                        "path": "resources_servers/reasoning_gym/configs/resources_only.yaml",
                        "sha256": hashlib.sha256(config_payload).hexdigest(),
                    },
                }
            }
        },
        "execution_environment": {
            "attempt": {
                "base_log_dir": str(tmp_path / "logs"),
                "persistent_cache": str(tmp_path / "cache"),
            }
        },
        "artifacts": {"outputs": {"directory": {"path": str(tmp_path / "out")}}},
    }
    fake_ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(gcs_address="127.0.0.1:10001"))
    parser = runner._build_reasoning_gym_resource_only_parser_config(
        manifest=manifest,
        execution_source_root=execution_root,
        ray_module=fake_ray,
    )
    initial = parser.kwargs["initial_global_config_dict"]
    assert "nemo_gym_log_dir" not in initial
    assert initial["config_paths"] == [str(config_path)]


def test_runner_finalizer_passes_run_helper_and_does_not_double_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _RUNNER_PATH.resolve().parents[2]
    runtime_path = source_root / runner._PROGRAM_PATHS["runtime"]
    child_path = source_root / runner._PROGRAM_PATHS["gym_child_runtime"]
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    child_sha = hashlib.sha256(child_path.read_bytes()).hexdigest()
    monkeypatch.setenv("STRICT_REPLAY_EXECUTION_SOURCE_ROOT", str(source_root))
    monkeypatch.setattr(
        runner,
        "_stable_regular_sha256",
        lambda path, *, name: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(runner, "_verify_imported_source_module_origins", lambda **kwargs: None)

    instances: list[Any] = []

    class ParserConfig:
        pass

    class RunHelper:
        def __init__(self) -> None:
            self.shutdown_count = 0
            self._processes = {"reasoning_gym": object()}
            self._head_server = object()
            instances.append(self)

        def start(self, *, global_config_dict_parser_config: Any) -> None:
            assert isinstance(global_config_dict_parser_config, ParserConfig)

        def shutdown(self) -> None:
            self.shutdown_count += 1
            self._processes = {}
            self._head_server = None

    cli = ModuleType("nemo_gym.cli")
    cli.GlobalConfigDictParserConfig = ParserConfig  # type: ignore[attr-defined]
    cli.RunHelper = RunHelper  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_gym.cli", cli)

    receipt = _resource_receipt(str(child_path))
    receipt_payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("ascii")

    class Session:
        receipt_root = Path("/strict/receipts")
        spec = {
            "gym": {
                "root": runner._GYM_CONTAINER_ROOT,
                "git_commit": "3" * 40,
                "tree": "4" * 40,
            },
            "targets": [
                {
                    "interpreter": str(child_path),
                    "scorer": {"sha256": "2" * 64},
                }
            ],
        }

        def launch_environment(self):
            return contextlib.nullcontext()

        def attest_started(self, run_helper: Any):
            return (
                {
                    "scope": "scorer-only",
                    "environment": "reasoning_gym",
                    "job_id": "12345",
                    "children": [
                        {
                            "role": "resource",
                            "observation": {"host": "127.0.0.1", "port": 5000},
                            "receipt": {
                                "path": "/strict/receipts/resource.json",
                                "schema": "nemo-rl-strict-gym-child-receipt-v1",
                                "sha256": hashlib.sha256(receipt_payload).hexdigest(),
                            },
                        }
                    ],
                },
                "3" * 64,
            )

        def finalize_score_calls(self, expected_calls: Any, *, run_helper: Any):
            assert len(expected_calls) == 4
            run_helper.shutdown()
            return ({"call_count": 4}, "4" * 64)

    child_runtime = ModuleType("nemo_rl.environments.strict_gym_child_runtime")
    child_runtime.__file__ = str(child_path)
    child_runtime.STRICT_GYM_SCORE_FINALIZER_SAFE = True  # type: ignore[attr-defined]
    child_runtime.prepare_strict_gym_child_runtime = lambda **kwargs: Session()  # type: ignore[attr-defined]
    child_runtime._load_canonical_document = lambda path: (receipt, receipt_payload)  # type: ignore[attr-defined]
    child_runtime.reasoning_score_call_expectation = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.environments.strict_gym_child_runtime",
        child_runtime,
    )
    import nemo_rl.environments

    monkeypatch.setattr(nemo_rl.environments, "strict_gym_child_runtime", child_runtime, raising=False)
    ray_module = ModuleType("ray")
    ray_module.is_initialized = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray_module)

    monkeypatch.setattr(
        replay,
        "reasoning_gym_score_call_material",
        lambda **kwargs: {
            "task_name": "knights_knaves",
            "answer": "A",
            "entry": {"question": "Q", "answer": "A", "metadata": {}},
            "float_result": 1.0,
        },
    )

    def execute(**kwargs: Any) -> str:
        assert kwargs["driver_scheduler_device_environment"] == {
            "schema": "nemo-rl-strict-scheduler-device-environment-v1",
            "cuda_visible_devices": "0,1,2,3",
            "gpu_device_ordinal": "0,1,2,3",
            "nvidia_visible_devices": "all",
            "rocr_visible_devices": None,
            "ze_affinity_mask": None,
        }
        for index in range(4):
            kwargs["independent_score_check"](index, {}, {})
        terminal = kwargs["finalize_score_call_evidence"]()
        assert terminal["sha256"] == "4" * 64
        return "complete"

    monkeypatch.setattr(replay, "execute_captured_replay_cohort", execute)
    manifest = {
        "environment": "reasoning_gym",
        "replay_contract": {
            "program": {
                "runtime": {
                    "path": runner._PROGRAM_PATHS["runtime"],
                    "sha256": runtime_sha,
                },
                "gym_child_runtime": {
                    "path": runner._PROGRAM_PATHS["gym_child_runtime"],
                    "sha256": child_sha,
                },
            },
            "source_snapshot": {
                "arm": "on",
                "ref": {
                    "config_sha256": "1" * 64,
                    "entrypoint_sha256": "2" * 64,
                    "manifest_sha256": "5" * 64,
                    "path": str(source_root),
                },
            },
            "gym_scorer": {
                "source": {
                    "gitlink_commit": "3" * 40,
                    "path": "/authenticated/gym-source",
                    "tree": "4" * 40,
                },
                "source_root": {
                    "snapshot_relative_path": runner._GYM_SOURCE_RELATIVE,
                    "host_path": str(source_root / runner._GYM_SOURCE_RELATIVE),
                    "container_path": runner._GYM_CONTAINER_ROOT,
                },
            },
        },
        "artifacts": {
            "outputs": {"reasoning_score_call_index": {"path": "/strict/receipts/reasoning-score-call-index.json"}}
        },
    }
    result = runner.run_reasoning_gym_replay_with_authenticated_resource_child(
        snapshot_authentication={
            "status": "authenticated",
            "replay_execution_manifest_sha256": "6" * 64,
            "source_snapshot_manifest_sha256": "5" * 64,
            "program_sha256": runner._domain_sha256(
                "captured-replay-program",
                {
                    "runtime": {
                        "path": runner._PROGRAM_PATHS["runtime"],
                        "sha256": runtime_sha,
                    },
                    "gym_child_runtime": {
                        "path": runner._PROGRAM_PATHS["gym_child_runtime"],
                        "sha256": child_sha,
                    },
                },
            ),
        },
        resource_parser_config=ParserConfig(),
        manifest=manifest,
        replay_execution_manifest_sha256="6" * 64,
        submission_receipt_sha256="7" * 64,
        authenticated_job_id="12345",
        driver_process={
            "boot_id_sha256": "8" * 64,
            "pid": 1,
            "start_time_ticks": 1,
        },
        driver_scheduler_device_environment={
            "schema": "nemo-rl-strict-scheduler-device-environment-v1",
            "cuda_visible_devices": "0,1,2,3",
            "gpu_device_ordinal": "0,1,2,3",
            "nvidia_visible_devices": "all",
            "rocr_visible_devices": None,
            "ze_affinity_mask": None,
        },
        source_transcript_document={},
        source_main_ledger_document={},
        transport_source=SimpleNamespace(),
    )
    assert result == "complete"
    assert instances[0].shutdown_count == 1


def test_runner_main_rejects_non_wrapper_invocation() -> None:
    with pytest.raises(runner.StrictCapturedReplayEntrypointError, match="wrapper driver phase"):
        runner.main(
            [
                "--replay-manifest",
                "/tmp/manifest.json",
                "--replay-manifest-sha256",
                "1" * 64,
                "--pre-receipt",
                "/tmp/pre.json",
                "--pre-receipt-sha256",
                "2" * 64,
            ]
        )
