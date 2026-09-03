# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.algorithms import strict_captured_replay_runtime_v2 as replay
from nemo_rl.utils import strict_captured_replay_profiles as replay_profiles
from nemo_rl.utils.strict_captured_replay_profiles import (
    get_strict_captured_replay_profile,
)


def _manifest_and_source(
    environment: str,
    profile_id: str,
    *,
    profile: replay_profiles.StrictCapturedReplayProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if profile is None:
        profile = get_strict_captured_replay_profile(
            expected_environment=environment,
            expected_profile_id=profile_id,
        )
    entries = []
    for index in range(4):
        entries.append(
            {
                "generation_seed": 100 + index,
                "generation_request": {"index": index},
                "model_response": {"response": index},
                "agent_run_request": {"request": index},
                "derived_verifier_request": {"derived": index},
                "model_transport_entry_sha256": f"{index + 1:x}" * 64,
                "model_transport_request_body_sha256": f"{index + 5:x}" * 64,
                "model_transport_response_body_sha256": f"{index + 9:x}" * 64,
            }
        )
    transcript = {
        "schema": replay.TRANSCRIPT_BUNDLE_SCHEMA,
        "pair_id": "pair-v2",
        "environment": environment,
        "arm": "off",
        "mode": "observe",
        "attempt_id": None,
        "generation": {"seed_base": 100},
        "fixture_row": {"value": {"fixture": environment}},
        "verifier_request_derivation": {"algorithm": "pinned"},
        "entries": entries,
    }
    ledger = {
        "schema": "nemo-rl-strict-main-step1-ledger-v5",
        "rows": [{"sample_id": str(index)} for index in range(4)],
    }
    transcript_sha = replay.document_sha256(transcript, trailing_lf=False)
    ledger_sha = replay.document_sha256(ledger, trailing_lf=False)
    terminal_name = (
        "reasoning-score-call-index.json"
        if environment == "reasoning_gym"
        else "format-verification-call-index.json"
    )
    scorer_path = f"/strict/{environment}/{terminal_name}"
    manifest = {
        "schema": "nemo-rl-strict-captured-replay-execution-manifest-v4",
        "pair_id": "pair-v2",
        "environment": environment,
        "scorer_profile": replay._scorer_profile(profile),
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": "replay-1",
        "wandb": {
            "enabled": False,
            "mode": "disabled",
            "reason": "scorer-only-replay-no-wandb-credentials-or-output",
        },
        "pair": {
            "manifest": {"sha256": "a" * 64},
            "pair_campaign_sha256": "b" * 64,
            "pair_campaign_reward_and_advantage_sha256": "c" * 64,
        },
        "source_capture": {
            "step1_evidence": {
                "transcript_bundle": {
                    "path": "/strict/source/transcript.json",
                    "schema": replay.TRANSCRIPT_BUNDLE_SCHEMA,
                    "sha256": transcript_sha,
                },
                "main_ledger": {
                    "path": "/strict/source/ledger.json",
                    "schema": "nemo-rl-strict-main-step1-ledger-v5",
                    "sha256": ledger_sha,
                },
                "model_transport": {
                    "bundle": {
                        "path": "/strict/source/transport.json",
                        "schema": "nemo-rl-strict-model-transport-bundle-v1",
                        "sha256": "d" * 64,
                    }
                },
            }
        },
        "replay_contract": {
            "execution_scope": "scorer-only",
            "policy_execution": {
                "backward": False,
                "forward": False,
                "optimizer": False,
                "violation": "fail-closed",
            },
            "selected_config": {"sha256": "e" * 64},
            "source_snapshot": {
                "arm": "on",
                "ref": {"manifest_sha256": "f" * 64},
            },
            "gym_scorer": {"resources": {"verifier_source": {"sha256": "1" * 64}}},
        },
        "artifacts": {
            "fixture": {"sha256": "2" * 64},
            "outputs": {
                "transcript_bundle": {
                    "path": f"/strict/{environment}/transcript.json",
                    "schema": replay.TRANSCRIPT_BUNDLE_SCHEMA,
                    "framing": "canonical-ascii-json-no-lf",
                    "mode": "0400",
                },
                "scorer_call_index": {
                    "path": scorer_path,
                    "schema": profile.call_index_schema,
                    "framing": "canonical-ascii-json-no-lf",
                    "mode": "0400",
                },
            },
        },
    }
    return manifest, transcript, ledger


class _Transport:
    def __init__(self, entries: list[dict[str, Any]], events: list[str]) -> None:
        self._entries = entries
        self.events = events

    def consume(self, *, rollout_index: int, generation_seed: int) -> dict[str, Any]:
        self.events.append(f"consume:{rollout_index}")
        entry = self._entries[rollout_index]
        assert generation_seed == entry["generation_seed"]
        return {
            "rollout_index": rollout_index,
            "generation_seed": generation_seed,
            "model_response": copy.deepcopy(entry["model_response"]),
            "agent_run_request": copy.deepcopy(entry["agent_run_request"]),
            "derived_verifier_request": copy.deepcopy(
                entry["derived_verifier_request"]
            ),
            "source_entry_sha256": entry["model_transport_entry_sha256"],
            "request_body_sha256": entry["model_transport_request_body_sha256"],
            "response_body_sha256": entry["model_transport_response_body_sha256"],
        }

    def record_fresh_verifier_result(
        self, *, rollout_index: int, verifier_response: Any
    ) -> None:
        assert verifier_response["reward"] == float(rollout_index % 2)
        self.events.append(f"record:{rollout_index}")

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        assert set(kwargs) == {
            "replay_execution_manifest_sha256",
            "authenticated_job_id",
            "process",
            "scheduler_device_environment",
            "scorer_call_index_ref",
        }
        assert kwargs["scorer_call_index_ref"]["sha256"] == "3" * 64
        self.events.append("transport-finalize")
        return {"schema": "nemo-rl-strict-model-transport-replay-consumption-v3"}


class _DictSubclass(dict[str, Any]):
    pass


def _reasoning_profile() -> replay_profiles.StrictCapturedReplayProfile:
    base = get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    return replace(
        base,
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
        verifier_type="score_answer",
        method="KnightsKnavesDataset.score_answer",
        resource_config_path_name="reasoning_gym",
        disabled_config_path_name="reasoning_gym_simple_agent",
        call_schema="nemo-rl-strict-reasoning-score-call-v1",
        closed_schema="nemo-rl-strict-reasoning-score-closed-v1",
        call_index_schema="nemo-rl-strict-reasoning-score-call-index-v1",
        scorer_terminal_index_path=(
            "strict_gym_child_runtime/reasoning-score-call-index.json"
        ),
    )


def _reasoning_request(text: str) -> dict[str, Any]:
    return {
        "responses_create_params": {"input": "question"},
        "question": "Who is truthful?",
        "answer": ["Alice"],
        "metadata": {"source_dataset": "knights_knaves", "difficulty": 2},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        },
    }


def _reasoning_response(
    request: dict[str, Any],
    *,
    answer: str,
    score: Any,
    reward: Any | None = None,
) -> dict[str, Any]:
    return {
        "responses_create_params": copy.deepcopy(request["responses_create_params"]),
        "response": copy.deepcopy(request["response"]),
        "task_name": "knights_knaves",
        "extracted_answer": answer,
        "score": score,
        "reward": score if reward is None else reward,
    }


def _install_reasoning_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> replay_profiles.StrictCapturedReplayProfile:
    profile = _reasoning_profile()
    original = replay_profiles.get_strict_captured_replay_profile

    def get_profile(
        *, expected_environment: str, expected_profile_id: str
    ) -> replay_profiles.StrictCapturedReplayProfile:
        if (expected_environment, expected_profile_id) == (
            profile.environment,
            profile.profile_id,
        ):
            return profile
        return original(
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )

    monkeypatch.setattr(
        replay_profiles,
        "get_strict_captured_replay_profile",
        get_profile,
    )
    return profile


def _patch_evidence_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "validate_transcript_bundle", lambda document: None)
    monkeypatch.setattr(replay, "validate_main_step1_ledger", lambda document: None)
    monkeypatch.setattr(
        replay, "validate_ledger_transcript_join", lambda **kwargs: None
    )
    monkeypatch.setattr(
        replay, "validate_captured_replay_source_join", lambda **kwargs: None
    )
    monkeypatch.setattr(
        replay, "validate_captured_replay_step1_ledger", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(replay, "_build_replay_ledger_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        replay,
        "build_transcript_bundle",
        lambda **kwargs: {
            "schema": replay.TRANSCRIPT_BUNDLE_SCHEMA,
            "entries": kwargs["entry_inputs"],
        },
    )
    monkeypatch.setattr(
        replay,
        "build_captured_replay_step1_ledger",
        lambda **kwargs: {"schema": "nemo-rl-strict-captured-replay-step1-ledger-v5"},
    )


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_profiled_runtime_executes_exact_k4_then_finalizes_child_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    profile_id: str,
) -> None:
    manifest, transcript, ledger = _manifest_and_source(environment, profile_id)
    _patch_evidence_builders(monkeypatch)
    events: list[str] = []
    transport = _Transport(transcript["entries"], events)

    def post(index: int, seed: int, request: Any) -> dict[str, Any]:
        assert seed == 100 + index and request == {"derived": index}
        events.append(f"post:{index}")
        return {"reward": float(index % 2)}

    def check(index: int, request: Any, response: Any) -> None:
        assert request == {"derived": index}
        assert response == {"reward": float(index % 2)}
        events.append(f"check:{index}")

    def finalize_child() -> dict[str, Any]:
        events.append("child-finalize")
        return {
            "path": manifest["artifacts"]["outputs"]["scorer_call_index"]["path"],
            "schema": "nemo-rl-strict-format-verification-call-index-v1",
            "sha256": "3" * 64,
        }

    documents = replay.execute_profiled_captured_replay_cohort(
        manifest=manifest,
        replay_execution_manifest_sha256="4" * 64,
        submission_receipt_sha256="5" * 64,
        authenticated_job_id="12345",
        driver_process={"pid": 1},
        driver_scheduler_device_environment={"cuda_visible_devices": "0,1,2,3"},
        source_transcript_document=transcript,
        source_main_ledger_document=ledger,
        transport_source=transport,
        post_verifier=post,
        independent_scorer_check=check,
        finalize_scorer_call_evidence=finalize_child,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )

    assert isinstance(documents, replay.ReplayDocumentsV2)
    assert events == [
        *[
            event
            for index in range(4)
            for event in (
                f"consume:{index}",
                f"post:{index}",
                f"check:{index}",
                f"record:{index}",
            )
        ],
        "child-finalize",
        "transport-finalize",
    ]


def test_profile_mismatch_fails_before_transport_consumption() -> None:
    manifest, transcript, ledger = _manifest_and_source(
        "citation", "citation-string-match-v1"
    )
    events: list[str] = []
    with pytest.raises(ValueError, match="unsupported"):
        replay.execute_profiled_captured_replay_cohort(
            manifest=manifest,
            replay_execution_manifest_sha256="4" * 64,
            submission_receipt_sha256="5" * 64,
            authenticated_job_id="12345",
            driver_process={},
            driver_scheduler_device_environment={},
            source_transcript_document=transcript,
            source_main_ledger_document=ledger,
            transport_source=_Transport(transcript["entries"], events),
            post_verifier=lambda *args: {"reward": 1.0},
            independent_scorer_check=lambda *args: None,
            finalize_scorer_call_evidence=lambda: {},
            expected_environment="citation",
            expected_profile_id="freeform-regex-v1",
        )
    assert events == []


@pytest.mark.parametrize("field", ["environment", "profile_id"])
def test_profile_string_subclasses_fail_before_transport(field: str) -> None:
    manifest, transcript, ledger = _manifest_and_source(
        "citation", "citation-string-match-v1"
    )
    events: list[str] = []
    kwargs = {
        "expected_environment": "citation",
        "expected_profile_id": "citation-string-match-v1",
    }
    kwargs[f"expected_{field}"] = type("StringSubclass", (str,), {})(
        kwargs[f"expected_{field}"]
    )
    with pytest.raises(ValueError, match="exact strings"):
        replay.execute_profiled_captured_replay_cohort(
            manifest=manifest,
            replay_execution_manifest_sha256="4" * 64,
            submission_receipt_sha256="5" * 64,
            authenticated_job_id="12345",
            driver_process={},
            driver_scheduler_device_environment={},
            source_transcript_document=transcript,
            source_main_ledger_document=ledger,
            transport_source=_Transport(transcript["entries"], events),
            post_verifier=lambda *args: {"reward": 1.0},
            independent_scorer_check=lambda *args: None,
            finalize_scorer_call_evidence=lambda: {},
            **kwargs,
        )
    assert events == []


def test_transport_material_mapping_subclass_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, transcript, ledger = _manifest_and_source(
        "citation", "citation-string-match-v1"
    )
    monkeypatch.setattr(replay, "validate_transcript_bundle", lambda document: None)
    monkeypatch.setattr(replay, "validate_main_step1_ledger", lambda document: None)
    monkeypatch.setattr(
        replay, "validate_ledger_transcript_join", lambda **kwargs: None
    )
    transport = _Transport(transcript["entries"], [])
    original_consume = transport.consume
    transport.consume = lambda **kwargs: _DictSubclass(original_consume(**kwargs))  # type: ignore[method-assign]
    with pytest.raises(replay.StrictCapturedReplayError, match="exact JSON object"):
        replay.execute_profiled_captured_replay_cohort(
            manifest=manifest,
            replay_execution_manifest_sha256="4" * 64,
            submission_receipt_sha256="5" * 64,
            authenticated_job_id="12345",
            driver_process={},
            driver_scheduler_device_environment={},
            source_transcript_document=transcript,
            source_main_ledger_document=ledger,
            transport_source=transport,
            post_verifier=lambda *args: {"reward": 1.0},
            independent_scorer_check=lambda *args: None,
            finalize_scorer_call_evidence=lambda: {},
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_scorer_finalizer_mapping_subclass_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, transcript, ledger = _manifest_and_source(
        "citation", "citation-string-match-v1"
    )
    _patch_evidence_builders(monkeypatch)
    terminal = _DictSubclass(
        {
            "path": manifest["artifacts"]["outputs"]["scorer_call_index"]["path"],
            "schema": "nemo-rl-strict-format-verification-call-index-v1",
            "sha256": "3" * 64,
        }
    )
    with pytest.raises(replay.StrictCapturedReplayError, match="exact JSON object"):
        replay.execute_profiled_captured_replay_cohort(
            manifest=manifest,
            replay_execution_manifest_sha256="4" * 64,
            submission_receipt_sha256="5" * 64,
            authenticated_job_id="12345",
            driver_process={},
            driver_scheduler_device_environment={},
            source_transcript_document=transcript,
            source_main_ledger_document=ledger,
            transport_source=_Transport(transcript["entries"], []),
            post_verifier=lambda index, *_: {"reward": float(index % 2)},
            independent_scorer_check=lambda *args: None,
            finalize_scorer_call_evidence=lambda: terminal,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


@pytest.mark.parametrize(
    "raw",
    [b'{"reward":-0}', b'{"reward":-0.0}', b'{"reward":NaN}', b'{"x":1,"x":2}', b"[]"],
)
def test_http_response_parser_rejects_non_strict_json(raw: bytes) -> None:
    with pytest.raises(replay.StrictCapturedReplayError):
        replay._strict_json_object(raw, name="response")


def test_runtime_surface_is_profiled_and_generic() -> None:
    parameters = inspect.signature(
        replay.execute_profiled_captured_replay_cohort
    ).parameters
    assert "expected_environment" in parameters
    assert "expected_profile_id" in parameters
    assert "independent_scorer_check" in parameters
    assert "finalize_scorer_call_evidence" in parameters
    assert "independent_format_check" not in parameters
    assert "finalize_format_call_evidence" not in parameters
    assert (
        "scorer_call_index_ref"
        in inspect.signature(
            replay.StrictModelTransportReplaySourceV3.finalize
        ).parameters
    )
    assert (
        "format_verification_call_index_ref"
        not in inspect.signature(
            replay.StrictModelTransportReplaySourceV3.finalize
        ).parameters
    )
    source = Path(replay.__file__).read_text(encoding="utf-8")
    assert "reasoning_gym_score_call_material" in replay.__all__
    assert "IndependentScorerCheck" in replay.__all__
    assert "FinalizeScorerCallEvidence" in replay.__all__
    assert "IndependentFormatCheck" not in replay.__all__
    assert "FinalizeFormatCallEvidence" not in replay.__all__
    assert "reasoning_score_call_index" not in source
    assert source.index("finalize_scorer_call_evidence()") < source.index(
        "transport_source.finalize("
    )


@pytest.mark.parametrize(
    ("text", "expected_answer"),
    [
        (
            "<answer>discarded</answer>\nwork\n<answer> Alice </answer>",
            "Alice",
        ),
        (r"work \boxed{Alice}", "Alice"),
        ("  Alice  ", "Alice"),
    ],
)
def test_reasoning_gym_score_call_material_projects_independent_exact_input(
    text: str,
    expected_answer: str,
) -> None:
    request = _reasoning_request(text)
    response = _reasoning_response(
        request,
        answer=expected_answer,
        score=0.375,
    )
    assert replay.reasoning_gym_score_call_material(
        derived_verifier_request=request,
        verifier_response=response,
    ) == {
        "answer": expected_answer,
        "entry": {
            "question": request["question"],
            "answer": request["answer"],
            "metadata": request["metadata"],
        },
        "task_name": "knights_knaves",
        "float_result": 0.375,
    }


@pytest.mark.parametrize(
    "response_update",
    [
        {"task_name": "other"},
        {"task_name": True},
        {"extracted_answer": "forged"},
        {"extracted_answer": True},
        {"score": 1, "reward": 1.0},
        {"score": True, "reward": 1.0},
        {"score": 1.0, "reward": True},
        {"score": math.inf, "reward": math.inf},
        {"score": math.nan, "reward": math.nan},
        {"score": -0.0, "reward": -0.0},
        {"score": -0.001, "reward": -0.001},
        {"score": 1.001, "reward": 1.001},
        {"reward": 0.5},
    ],
)
def test_reasoning_gym_score_call_material_rejects_forged_alias_or_score(
    response_update: dict[str, Any],
) -> None:
    request = _reasoning_request("<answer>Alice</answer>")
    response = {
        **_reasoning_response(request, answer="Alice", score=0.375),
        **response_update,
    }
    with pytest.raises((replay.StrictCapturedReplayError, ValueError)):
        replay.reasoning_gym_score_call_material(
            derived_verifier_request=request,
            verifier_response=response,
        )


def test_reasoning_gym_score_call_material_rejects_wrong_dataset() -> None:
    request = _reasoning_request("<answer>Alice</answer>")
    request["metadata"]["source_dataset"] = "other"
    with pytest.raises(replay.StrictCapturedReplayError, match="task_name"):
        replay.reasoning_gym_score_call_material(
            derived_verifier_request=request,
            verifier_response={
                **_reasoning_response(request, answer="Alice", score=1.0),
                "task_name": "other",
            },
        )


def test_reasoning_gym_score_call_material_rejects_response_keyset_drift() -> None:
    request = _reasoning_request("<answer>Alice</answer>")
    response = _reasoning_response(request, answer="Alice", score=1.0)
    response.pop("response")
    with pytest.raises(replay.StrictCapturedReplayError, match="keyset"):
        replay.reasoning_gym_score_call_material(
            derived_verifier_request=request,
            verifier_response=response,
        )


def test_profiled_runtime_accepts_reasoning_fraction_and_finalizes_generic_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _install_reasoning_profile(monkeypatch)
    manifest, transcript, ledger = _manifest_and_source(
        profile.environment,
        profile.profile_id,
        profile=profile,
    )
    scores = (0.125, 0.375, 0.625, 0.875)
    for index, entry in enumerate(transcript["entries"]):
        request = _reasoning_request(f"<answer>answer-{index}</answer>")
        entry["model_response"] = copy.deepcopy(request["response"])
        entry["derived_verifier_request"] = request
    manifest["source_capture"]["step1_evidence"]["transcript_bundle"]["sha256"] = (
        replay.document_sha256(transcript, trailing_lf=False)
    )
    _patch_evidence_builders(monkeypatch)
    events: list[str] = []
    transport = _Transport(transcript["entries"], events)

    def record(*, rollout_index: int, verifier_response: Any) -> None:
        assert verifier_response["reward"] == scores[rollout_index]
        events.append(f"record:{rollout_index}")

    transport.record_fresh_verifier_result = record  # type: ignore[method-assign]
    projected: list[dict[str, Any]] = []

    def post(index: int, seed: int, request: Any) -> dict[str, Any]:
        assert seed == 100 + index
        events.append(f"post:{index}")
        return _reasoning_response(
            request,
            answer=f"answer-{index}",
            score=scores[index],
        )

    def check(index: int, request: Any, response: Any) -> None:
        projected.append(
            replay.reasoning_gym_score_call_material(
                derived_verifier_request=request,
                verifier_response=response,
            )
        )
        events.append(f"check:{index}")

    def finalize_child() -> dict[str, Any]:
        events.append("child-finalize")
        return {
            "path": manifest["artifacts"]["outputs"]["scorer_call_index"]["path"],
            "schema": profile.call_index_schema,
            "sha256": "3" * 64,
        }

    documents = replay.execute_profiled_captured_replay_cohort(
        manifest=manifest,
        replay_execution_manifest_sha256="4" * 64,
        submission_receipt_sha256="5" * 64,
        authenticated_job_id="12345",
        driver_process={"pid": 1},
        driver_scheduler_device_environment={"cuda_visible_devices": "0,1,2,3"},
        source_transcript_document=transcript,
        source_main_ledger_document=ledger,
        transport_source=transport,
        post_verifier=post,
        independent_scorer_check=check,
        finalize_scorer_call_evidence=finalize_child,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )

    assert [item["float_result"] for item in projected] == list(scores)
    assert documents.scorer_call_index["schema"] == profile.call_index_schema
    assert events[-2:] == ["child-finalize", "transport-finalize"]


def test_profiled_runtime_rejects_reasoning_negative_zero_before_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _install_reasoning_profile(monkeypatch)
    manifest, transcript, ledger = _manifest_and_source(
        profile.environment,
        profile.profile_id,
        profile=profile,
    )
    request = _reasoning_request("<answer>Alice</answer>")
    for entry in transcript["entries"]:
        entry["model_response"] = copy.deepcopy(request["response"])
        entry["derived_verifier_request"] = copy.deepcopy(request)
    manifest["source_capture"]["step1_evidence"]["transcript_bundle"]["sha256"] = (
        replay.document_sha256(transcript, trailing_lf=False)
    )
    _patch_evidence_builders(monkeypatch)
    terminal_called = False
    transport = _Transport(transcript["entries"], [])
    transport.record_fresh_verifier_result = lambda **kwargs: None  # type: ignore[method-assign]
    response = _reasoning_response(request, answer="Alice", score=-0.0)

    def finalize_child() -> dict[str, Any]:
        nonlocal terminal_called
        terminal_called = True
        return {}

    with pytest.raises(
        (replay.StrictCapturedReplayError, ValueError), match="negative zero"
    ):
        replay.execute_profiled_captured_replay_cohort(
            manifest=manifest,
            replay_execution_manifest_sha256="4" * 64,
            submission_receipt_sha256="5" * 64,
            authenticated_job_id="12345",
            driver_process={},
            driver_scheduler_device_environment={},
            source_transcript_document=transcript,
            source_main_ledger_document=ledger,
            transport_source=transport,
            post_verifier=lambda *args: response,
            independent_scorer_check=lambda *args: None,
            finalize_scorer_call_evidence=finalize_child,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
    assert terminal_called is False
