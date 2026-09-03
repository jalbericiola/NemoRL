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
from nemo_rl.environments import strict_gym_child_runtime_v2 as child_runtime
from nemo_rl.environments.strict_gym_child_runtime_v2 import (
    format_verification_call_expectation,
    reasoning_score_call_expectation,
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
        "method": "_verify_string_match" if environment == "citation" else "_verify_regex",
        "resource_config_path_name": ("citation_format" if environment == "citation" else "freeform_formatting"),
        "disabled_config_path_name": (
            "citation_format_simple_agent" if environment == "citation" else "freeform_formatting_simple_agent"
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


def _reasoning_terminal_and_transcript() -> tuple[dict[str, Any], dict[str, Any]]:
    rewards = (0.0, 0.25, 0.5, 1.0)
    entries: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for index, reward in enumerate(rewards):
        request = {
            "question": f"question-{index}",
            "answer": f"reference-answer-{index}",
            "metadata": {"source_dataset": "knights_knaves"},
        }
        extracted_answer = f"model-answer-{index}"
        response = {
            "task_name": "knights_knaves",
            "score": reward,
            "reward": reward,
            "extracted_answer": extracted_answer,
        }
        entry = {
            "sample_index": index,
            "fixture_row_index": 0,
            "rollout_index": index,
            "generation_seed": 100 + index,
            "model_transport_entry_sha256": _digest(f"entry-{index}"),
            "model_transport_request_body_sha256": _digest(f"request-{index}"),
            "model_transport_response_body_sha256": _digest(f"response-{index}"),
            "model_response_sha256": _digest(f"model-response-{index}"),
            "derived_verifier_request": request,
            "verifier_response": response,
            "raw_environment_reward": reward,
        }
        expectation = reasoning_score_call_expectation(
            task_name="knights_knaves",
            answer=extracted_answer,
            entry=request,
            float_result=reward,
        )
        entries.append(entry)
        calls.append(
            {
                "sequence": index + 1,
                "task_name": expectation["task_name"],
                "input": {
                    "answer_sha256": expectation["answer_sha256"],
                    "entry_sha256": expectation["entry_sha256"],
                },
                "float_result": expectation["float_result"],
                "receipt": {
                    "path": f"/result/reasoning-score-call-{index + 1:08d}.json",
                    "schema": "nemo-rl-strict-reasoning-score-call-v1",
                    "sha256": _digest(f"score-call-{index}"),
                },
            }
        )
    terminal = {
        "schema": evidence.REASONING_SCORE_CALL_INDEX_SCHEMA,
        "environment": "reasoning_gym",
        "quiescence": {"original_process_reaped": True},
        "calls": calls,
    }
    return terminal, {"entries": entries}


def test_reasoning_terminal_join_and_sample_projection_close_fractional_rewards() -> None:
    profile_id = "reasoning-gym-exact-match-v1"
    terminal, transcript = _reasoning_terminal_and_transcript()
    evidence._close_score_transcript_join(
        terminal,
        transcript,
        expected_environment="reasoning_gym",
        expected_profile_id=profile_id,
    )
    samples = evidence._project_authenticated_result_samples_v2(
        transcript,
        expected_environment="reasoning_gym",
        expected_profile_id=profile_id,
    )

    assert [sample["raw_environment_reward"] for sample in samples] == [
        0.0,
        0.25,
        0.5,
        1.0,
    ]
    assert all(set(sample) == set(evidence._AUTHENTICATED_RESULT_SAMPLE_V2_KEYS) for sample in samples)
    assert samples[1]["match_details"] == {
        "task_name": "knights_knaves",
        "score": 0.25,
        "extracted_answer": "model-answer-1",
    }
    assert type(samples[1]["match_details"]["score"]) is float


@pytest.mark.parametrize(
    ("target", "poison"),
    (
        ("inner_profile", "reasoning-gym-exact-match-v1"),
        ("schema", evidence.FORMAT_VERIFICATION_CALL_INDEX_SCHEMA),
        ("outer_profile", "citation-string-match-v1"),
        ("call_hash", "0" * 64),
        ("server_reward", 0.75),
        ("server_reward", True),
        ("score", -0.0),
        ("raw_reward", 0.75),
        ("response_task", "other_task"),
    ),
)
def test_reasoning_terminal_join_rejects_profile_and_reward_poison(
    target: str,
    poison: Any,
) -> None:
    terminal, transcript = _reasoning_terminal_and_transcript()
    profile_id = "reasoning-gym-exact-match-v1"
    if target == "inner_profile":
        terminal["profile_id"] = poison
    elif target == "schema":
        terminal["schema"] = poison
    elif target == "outer_profile":
        profile_id = poison
    elif target == "call_hash":
        terminal["calls"][0]["input"]["entry_sha256"] = poison
    elif target == "server_reward":
        transcript["entries"][0]["verifier_response"]["reward"] = poison
    elif target == "score":
        transcript["entries"][0]["verifier_response"]["score"] = poison
    elif target == "raw_reward":
        transcript["entries"][0]["raw_environment_reward"] = poison
    elif target == "response_task":
        transcript["entries"][0]["verifier_response"]["task_name"] = poison
    else:  # pragma: no cover - the closed parameter table owns this branch.
        raise AssertionError(target)

    with pytest.raises((TypeError, ValueError)):
        evidence._close_score_transcript_join(
            terminal,
            transcript,
            expected_environment="reasoning_gym",
            expected_profile_id=profile_id,
        )


@pytest.mark.parametrize(
    ("field", "poison"),
    (
        ("score", -0.0),
        ("reward", True),
        ("reward", 0.75),
        ("raw_environment_reward", True),
        ("raw_environment_reward", 0.75),
    ),
)
def test_reasoning_sample_projection_rejects_nonexact_or_divergent_reward(
    field: str,
    poison: Any,
) -> None:
    _, transcript = _reasoning_terminal_and_transcript()
    if field == "raw_environment_reward":
        transcript["entries"][0][field] = poison
    else:
        transcript["entries"][0]["verifier_response"][field] = poison
    with pytest.raises((TypeError, ValueError)):
        evidence._project_authenticated_result_samples_v2(
            transcript,
            expected_environment="reasoning_gym",
            expected_profile_id="reasoning-gym-exact-match-v1",
        )


def test_reasoning_verifier_response_binds_server_reward_to_score_and_raw() -> None:
    model_response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<answer>model-answer</answer>",
                    }
                ],
            }
        ]
    }
    verifier_response = {
        "responses_create_params": {},
        "response": model_response,
        "reward": 0.25,
        "task_name": "knights_knaves",
        "score": 0.25,
        "extracted_answer": "model-answer",
    }
    evidence._validate_environment_verifier_response(
        verifier_response,
        agent_run_request={"metadata": {"source_dataset": "knights_knaves"}},
        model_response=model_response,
        environment="reasoning_gym",
        reward=0.25,
        name="reasoning verifier response",
    )

    for poison in (True, -0.0, 0.5):
        altered = copy.deepcopy(verifier_response)
        altered["reward"] = poison
        with pytest.raises((TypeError, ValueError)):
            evidence._validate_environment_verifier_response(
                altered,
                agent_run_request={"metadata": {"source_dataset": "knights_knaves"}},
                model_response=model_response,
                environment="reasoning_gym",
                reward=0.25,
                name="reasoning verifier response",
            )


def test_retained_reasoning_scorer_dispatch_uses_only_payload_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, _ = _reasoning_terminal_and_transcript()
    roster = (("strict_gym_child_runtime/index.json", b"retained"),)
    expected_sha256 = _digest("terminal")
    observed: list[tuple[tuple[tuple[str, bytes], ...], dict[str, Any]]] = []

    def validate_reasoning(payload_roster, **kwargs):
        observed.append((payload_roster, kwargs))
        return copy.deepcopy(terminal), expected_sha256

    def forbidden_format(*args, **kwargs):
        del args, kwargs
        raise AssertionError("reasoning dispatch must not invoke format validation")

    def forbidden_path_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("retained-byte validation must not reopen named paths")

    monkeypatch.setattr(
        child_runtime,
        "validate_finalized_reasoning_score_call_index_payloads",
        validate_reasoning,
    )
    monkeypatch.setattr(
        child_runtime,
        "validate_finalized_format_verification_call_index_payloads",
        forbidden_format,
    )
    monkeypatch.setattr(Path, "read_bytes", forbidden_path_load)
    admitted, digest = evidence._validate_retained_scorer_call_index_v2(
        roster,
        expected_sha256=expected_sha256,
        expected_receipt_root=Path("/result/strict_gym_child_runtime"),
        expected_bootstrap_root=Path("/snapshot/bootstrap"),
        expected_bootstrap_sha256=_digest("bootstrap"),
        expected_pair_id="pair-v2",
        expected_job_id="12345",
        expected_environment="reasoning_gym",
        expected_profile_id="reasoning-gym-exact-match-v1",
    )

    assert admitted == terminal
    assert admitted is not terminal
    assert digest == expected_sha256
    assert observed == [
        (
            roster,
            {
                "expected_sha256": expected_sha256,
                "expected_receipt_root": Path("/result/strict_gym_child_runtime"),
                "expected_bootstrap_root": Path("/snapshot/bootstrap"),
                "expected_bootstrap_sha256": _digest("bootstrap"),
                "expected_pair_id": "pair-v2",
                "expected_job_id": "12345",
            },
        )
    ]

    poisoned_terminal = copy.deepcopy(terminal)
    poisoned_terminal["profile_id"] = "reasoning-gym-exact-match-v1"
    monkeypatch.setattr(
        child_runtime,
        "validate_finalized_reasoning_score_call_index_payloads",
        lambda *args, **kwargs: (poisoned_terminal, expected_sha256),
    )
    with pytest.raises(ValueError, match="outer profile authority"):
        evidence._validate_retained_scorer_call_index_v2(
            roster,
            expected_sha256=expected_sha256,
            expected_receipt_root=Path("/result/strict_gym_child_runtime"),
            expected_bootstrap_root=Path("/snapshot/bootstrap"),
            expected_bootstrap_sha256=_digest("bootstrap"),
            expected_pair_id="pair-v2",
            expected_job_id="12345",
            expected_environment="reasoning_gym",
            expected_profile_id="reasoning-gym-exact-match-v1",
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
        if name.startswith(("build_captured", "validate_captured", "publish_captured", "load_captured"))
    }
    assert lifecycle_names
    assert all(name.endswith("_v2") for name in lifecycle_names)
    for name in lifecycle_names:
        parameters = inspect.signature(getattr(evidence, name)).parameters
        assert "expected_environment" in parameters
        assert "expected_profile_id" in parameters
