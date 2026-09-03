# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import copy
import hashlib
import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import nemo_rl.utils.strict_captured_replay_evidence_v2 as evidence
from nemo_rl.environments.strict_gym_child_runtime_v2 import (
    format_verification_call_expectation,
)
from tests.unit.utils.test_strict_captured_replay_evidence import _format_entry

_PROFILE_PAIRS = (
    ("citation", "citation-string-match-v1"),
    ("freeform", "freeform-regex-v1"),
)


def _digest(value: str | bytes) -> str:
    payload = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _profile(environment: str, profile_id: str) -> dict[str, Any]:
    return {
        "environment": environment,
        "profile_id": profile_id,
        "verifier_type": "string_match" if environment == "citation" else "regex",
        "method": "_verify_string_match"
        if environment == "citation"
        else "_verify_regex",
        "resource_config_path_name": (
            "citation_format" if environment == "citation" else "freeform_formatting"
        ),
        "disabled_config_path_name": (
            "citation_format_simple_agent"
            if environment == "citation"
            else "freeform_formatting_simple_agent"
        ),
        "resource_app": {"path": "resource/app.py", "sha256": _digest("app")},
        "resource_config": {
            "path": "resource/config.yaml",
            "sha256": _digest("config"),
        },
        "requirements": {
            "path": "resource/requirements.txt",
            "sha256": _digest("requirements"),
        },
        "call_schema": "nemo-rl-strict-format-verification-call-v1",
        "closed_schema": "nemo-rl-strict-format-verification-closed-v1",
        "call_index_schema": evidence.FORMAT_VERIFICATION_CALL_INDEX_SCHEMA,
    }


def test_import_boundary_does_not_execute_profile_or_manifest_modules() -> None:
    source_root = Path(__file__).resolve().parents[3]
    code = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
import nemo_rl.utils.strict_captured_replay_evidence_v2
assert 'nemo_rl.utils.strict_captured_replay_profiles' not in sys.modules
assert 'nemo_rl.utils.strict_captured_replay_manifest_v2' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_program_closure_is_exact_14_member_stable_loaded_set(tmp_path: Path) -> None:
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        REPLAY_PROGRAM_V2_PATHS,
    )

    assert evidence.REPLAY_PROGRAM_V2_PATHS == REPLAY_PROGRAM_V2_PATHS
    assert len(evidence.REPLAY_PROGRAM_V2_PATHS) == 14
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(mode=0o700)
    program: dict[str, dict[str, str]] = {}
    for name, relative in evidence.REPLAY_PROGRAM_V2_PATHS.items():
        path = snapshot / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw = name.encode("ascii")
        path.write_bytes(raw)
        program[name] = {"path": relative, "sha256": _digest(raw)}
    manifest = {
        "replay_contract": {
            "program": program,
            "source_snapshot": {"ref": {"path": str(snapshot)}},
        }
    }
    evidence._authenticate_program_closure_v2(manifest)

    tampered = copy.deepcopy(manifest)
    tampered["replay_contract"]["program"]["runtime"]["sha256"] = _digest("wrong")
    with pytest.raises(ValueError, match="authenticated reference"):
        evidence._authenticate_program_closure_v2(tampered)

    extra = copy.deepcopy(manifest)
    extra["replay_contract"]["program"]["result-derived-plugin"] = {
        "path": "plugin.py",
        "sha256": _digest("plugin"),
    }
    with pytest.raises(ValueError, match="keyset mismatch"):
        evidence._authenticate_program_closure_v2(extra)


class _DictPoison(dict):
    pass


class _ListPoison(list):
    pass


@pytest.mark.parametrize(
    "value",
    [
        {"value": -0.0},
        {"value": math.nan},
        {"value": math.inf},
        _DictPoison(value=1),
        {"value": _ListPoison([1])},
    ],
)
def test_canonical_json_rejects_negative_zero_nonfinite_and_subclasses(
    value: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        evidence.canonical_ascii_json(value)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"x","value":-0.0}',
        b'{"schema":"x","value":NaN}',
        b'{"schema":"x","value":1,"value":2}',
    ],
)
def test_loader_rejects_negative_zero_nonfinite_and_duplicate_bytes(
    tmp_path: Path,
    raw: bytes,
) -> None:
    path = tmp_path / "poison.json"
    path.write_bytes(raw)
    path.chmod(0o400)
    with pytest.raises((TypeError, ValueError)):
        evidence.load_evidence_document(
            path=path,
            expected_sha256=_digest(raw),
            trailing_lf=False,
        )


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_format_terminal_join_recomputes_exact_profile_result(
    environment: str,
    profile_id: str,
) -> None:
    entries = [_format_entry(index, environment) for index in range(4)]
    calls: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        request = {
            name: copy.deepcopy(entry["derived_verifier_request"][name])
            for name in ("responses_create_params", "response", "verifier")
        }
        expected = format_verification_call_expectation(
            environment=environment,
            derived_verifier_request=request,
            verifier_response=entry["verifier_response"],
        )
        calls.append(
            {
                "sequence": index + 1,
                "method": expected["method"],
                "input": {
                    name: expected[name]
                    for name in (
                        "request_sha256",
                        "verifier_sha256",
                        "response_text_sha256",
                    )
                },
                "outcome": {
                    "kind": "returned",
                    "response_sha256": expected["response_sha256"],
                    "match_details_sha256": expected["match_details_sha256"],
                    "float_result": expected["float_result"],
                },
                "receipt": {
                    "sequence": index + 1,
                    "path": f"/receipts/call-{index + 1}.json",
                    "schema": "nemo-rl-strict-format-verification-call-v1",
                    "sha256": _digest(f"call-{index + 1}"),
                },
            }
        )
    terminal = {
        "schema": evidence.FORMAT_VERIFICATION_CALL_INDEX_SCHEMA,
        "environment": environment,
        "profile_id": profile_id,
        "quiescence": {"original_process_reaped": True},
        "calls": calls,
    }
    transcript = {"entries": entries}
    evidence._close_score_transcript_join(
        terminal,
        transcript,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )

    poisoned = copy.deepcopy(terminal)
    poisoned["calls"][0]["outcome"]["float_result"] = -0.0
    with pytest.raises(ValueError):
        evidence._close_score_transcript_join(
            poisoned,
            transcript,
            expected_environment=environment,
            expected_profile_id=profile_id,
        )

    unreaped = copy.deepcopy(terminal)
    unreaped["quiescence"]["original_process_reaped"] = 1
    with pytest.raises(ValueError, match="process reaping"):
        evidence._close_score_transcript_join(
            unreaped,
            transcript,
            expected_environment=environment,
            expected_profile_id=profile_id,
        )


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_evidence_index_v4_is_profile_bound_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    profile_id: str,
) -> None:
    scorer_profile = _profile(environment, profile_id)
    manifest = {
        "pair_id": "pair-v2",
        "environment": environment,
        "attempt_id": "replay-1",
        "scorer_profile": scorer_profile,
        "pair": {
            "submission_receipt": {
                "path": "/pair/submission.json",
                "schema": "nemo-rl-strict-pair-submission-receipt-v2",
                "sha256": _digest("pair-submission"),
            }
        },
        "source_capture": {"arm": "off"},
    }
    process = {
        "boot_id_sha256": _digest("boot"),
        "pid": 123,
        "start_time_ticks": 456,
    }
    outputs = {
        name: {
            "path": f"/results/{name}.json",
            "schema": f"schema-{name}",
            "sha256": _digest(name),
        }
        for name in evidence.REPLAY_OUTPUT_V2_KEYS
    }
    submission = {
        "replay_execution_manifest": {
            "path": "/manifest.json",
            "schema": evidence.REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
            "sha256": _digest("manifest"),
        }
    }
    pre = {
        "submission_receipt": {
            "path": "/submission.json",
            "schema": evidence.REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
            "sha256": _digest("submission"),
        }
    }
    exit_receipt = {
        "candidate_job_id": "82001",
        "authenticated_job_id": "82001",
        "driver_process": process,
        "driver_scheduler_device_environment": {},
        "outputs": outputs,
        "pre_receipt": {
            "path": "/pre.json",
            "schema": evidence.REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
            "sha256": _digest("pre"),
        },
    }
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        evidence,
        "_validated_lifecycle_manifest",
        lambda value, **kwargs: manifest,
    )
    monkeypatch.setattr(
        evidence,
        "validate_captured_replay_exit_receipt_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "_replay_job_receipt_path",
        lambda *args, **kwargs: Path("/exit.json"),
    )
    monkeypatch.setattr(
        evidence,
        "_require_loaded_document_matches",
        lambda *args, **kwargs: None,
    )

    def validate_outputs(value, **kwargs):
        calls.append(kwargs)
        return value

    monkeypatch.setattr(evidence, "_validate_captured_replay_outputs", validate_outputs)
    document = evidence.build_captured_replay_evidence_index_v2(
        replay_execution_manifest=manifest,
        authenticated_source=object(),
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert set(document) == set(evidence.REPLAY_POST_INDEX_V2_ROOT_KEYS)
    assert document["schema"] == evidence.REPLAY_POST_INDEX_V2_SCHEMA
    assert document["environment"] == environment
    assert document["profile_id"] == profile_id
    assert document["scorer_profile"] == scorer_profile
    assert type(document["original_process_reaped"]) is bool
    assert document["original_process_reaped"] is True
    assert set(document["outputs"]) == set(evidence.REPLAY_OUTPUT_V2_KEYS)
    assert calls[-1]["expected_environment"] == environment
    assert calls[-1]["expected_profile_id"] == profile_id

    for field, poison in (
        ("original_process_reaped", 1),
        ("profile_id", "wrong-profile"),
    ):
        altered = copy.deepcopy(document)
        altered[field] = poison
        with pytest.raises(ValueError):
            evidence.validate_captured_replay_evidence_index_v2(
                altered,
                replay_execution_manifest=manifest,
                submission_receipt=submission,
                pre_receipt=pre,
                exit_receipt=exit_receipt,
                authenticated_source=object(),
                expected_environment=environment,
                expected_profile_id=profile_id,
            )


def test_runtime_attestation_v2_binds_profile_reaping_and_generic_scorer_ref() -> None:
    environment, profile_id = _PROFILE_PAIRS[0]
    manifest = {
        "environment": environment,
        "scorer_profile": _profile(environment, profile_id),
        "runtime_attestation_requirements": {"schema": "requirements-v2"},
    }
    outputs = {
        name: {
            "path": f"/result/{name}.json",
            "schema": f"schema-{name}",
            "sha256": _digest(name),
        }
        for name in evidence.REPLAY_OUTPUT_V2_KEYS
    }
    result = evidence._replay_runtime_attestation(manifest, outputs)
    assert set(result) == set(evidence.REPLAY_RUNTIME_ATTESTATION_V2_KEYS)
    assert result["schema"] == evidence.REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA
    assert result["environment"] == environment
    assert result["profile_id"] == profile_id
    assert result["original_process_reaped"] is True
    assert "scorer_call_index" in result
    assert "reasoning_score_call_index" not in result


def test_public_lifecycle_surface_is_v2_only_and_profile_explicit() -> None:
    lifecycle_names = {
        name
        for name in evidence.__all__
        if name.startswith(
            ("build_captured", "validate_captured", "publish_captured", "load_captured")
        )
    }
    assert lifecycle_names
    assert all(name.endswith("_v2") for name in lifecycle_names)
    for name in lifecycle_names:
        parameters = inspect.signature(getattr(evidence, name)).parameters
        assert "expected_environment" in parameters
        assert "expected_profile_id" in parameters
