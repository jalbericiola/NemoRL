# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import copy
import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import nemo_rl.environments.strict_gym_child_runtime_v2 as child_runtime_v2
from nemo_rl.utils import strict_captured_replay_profiles as replay_profiles
from nemo_rl.utils.strict_captured_replay_evidence import (
    TRANSCRIPT_BUNDLE_SCHEMA,
    canonical_ascii_json,
)
from nemo_rl.utils.strict_main_step_ledger import MAIN_STEP1_LEDGER_SCHEMA
from nemo_rl.utils.strict_model_transport import (
    MODEL_TRANSPORT_BUNDLE_SCHEMA,
    MODEL_TRANSPORT_CALL_SCHEMA,
    MODEL_TRANSPORT_MANIFEST_SCHEMA,
)
from nemo_rl.utils.strict_model_transport_replay_v3 import (
    MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA,
    ReplayVerifierMaterial,
    StrictModelTransportReplayError,
    StrictModelTransportReplaySourceV3,
    publish_strict_model_transport_replay_consumption_v3,
    validate_strict_model_transport_replay_consumption_v3,
)
from tests.unit.utils.test_strict_captured_replay_evidence import (
    _entry,
    _format_entry,
)

_PROFILE_PAIRS = (
    ("citation", "citation-string-match-v1"),
    ("freeform", "freeform-regex-v1"),
)
_REASONING_SCORES = (0.125, 0.375, 0.625, 0.875)


def _reasoning_profile() -> replay_profiles.StrictCapturedReplayProfile:
    base = replay_profiles.get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    result_files = tuple(
        name.replace("format-verification", "reasoning-score")
        for name in base.result_files
    )
    result_file_schemas = tuple(
        schema.replace("format-verification", "reasoning-score")
        for schema in base.result_file_schemas
    )
    terminal_path = "strict_gym_child_runtime/reasoning-score-call-index.json"
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
        result_files=result_files,
        result_file_schemas=result_file_schemas,
        result_anchor_paths=frozenset(
            terminal_path if name == base.scorer_terminal_index_path else name
            for name in base.result_anchor_paths
        ),
        scorer_terminal_index_path=terminal_path,
    )


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


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(path: Path, schema: str, label: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema": schema,
        "sha256": _digest(label),
    }


def _device_environment() -> dict[str, Any]:
    return {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "all",
        "rocr_visible_devices": None,
        "ze_affinity_mask": None,
    }


def _source(
    tmp_path: Path,
    *,
    environment: str,
    profile_id: str,
) -> tuple[
    StrictModelTransportReplaySourceV3,
    list[dict[str, Any]],
    Path,
]:
    attempt_root = tmp_path / "captured_replay" / "replay-1"
    attempt_root.mkdir(mode=0o700, parents=True)
    if environment == "reasoning_gym":
        raw_entries = [
            _entry(index, reward=_REASONING_SCORES[index]) for index in range(4)
        ]
    else:
        raw_entries = [_format_entry(index, environment) for index in range(4)]
    materials = tuple(
        ReplayVerifierMaterial(
            rollout_index=index,
            generation_seed=entry["generation_seed"],
            model_response=copy.deepcopy(entry["model_response"]),
            agent_run_request=copy.deepcopy(entry["agent_run_request"]),
            derived_verifier_request=copy.deepcopy(entry["derived_verifier_request"]),
            source_entry_sha256=entry["model_transport_entry_sha256"],
            request_body_sha256=entry["model_transport_request_body_sha256"],
            response_body_sha256=entry["model_transport_response_body_sha256"],
        )
        for index, entry in enumerate(raw_entries)
    )
    transcript = {
        "entries": [
            {
                "generation_request_sha256": _digest(f"generation-request-{index}"),
                "model_response_sha256": _digest(f"model-response-{index}"),
                "agent_run_request_sha256": _digest(f"agent-run-request-{index}"),
                "derived_verifier_request_sha256": _digest(
                    f"derived-verifier-request-{index}"
                ),
            }
            for index in range(4)
        ]
    }
    refs = {
        "main_ledger": _artifact(
            tmp_path / "off" / "main-ledger.json",
            MAIN_STEP1_LEDGER_SCHEMA,
            "main-ledger",
        ),
        "transcript_bundle": _artifact(
            tmp_path / "off" / "transcript-bundle.json",
            TRANSCRIPT_BUNDLE_SCHEMA,
            "transcript",
        ),
        "transport_bundle": _artifact(
            tmp_path / "off" / "model-transport-bundle.json",
            MODEL_TRANSPORT_BUNDLE_SCHEMA,
            "transport-bundle",
        ),
        "transport_manifest": _artifact(
            tmp_path / "off" / "model-transport-manifest.json",
            MODEL_TRANSPORT_MANIFEST_SCHEMA,
            "transport-manifest",
        ),
        "raw_log": {
            "path": str(tmp_path / "off" / "model-transport.jsonl"),
            "record_schema": MODEL_TRANSPORT_CALL_SCHEMA,
            "record_count": 4,
            "sha256": _digest("raw-log"),
        },
        "ordered_entries_sha256": _digest("source-ordered-entries"),
    }
    source = StrictModelTransportReplaySourceV3(
        expected_environment=environment,
        expected_profile_id=profile_id,
        pair_id="pair-001",
        attempt_id="replay-1",
        replay_attempt_root=str(attempt_root),
        source_job_id="12345",
        pair_manifest_sha256=_digest("pair-manifest"),
        source_submission_receipt_sha256=_digest("pair-receipt"),
        source_refs=refs,
        materials=materials,
        transcript_document=transcript,
        main_ledger_document={},
    )
    responses = [copy.deepcopy(entry["verifier_response"]) for entry in raw_entries]
    return source, responses, attempt_root


def _terminal(
    *,
    environment: str,
    profile_id: str,
    source: StrictModelTransportReplaySourceV3,
    responses: list[dict[str, Any]],
    attempt_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = replay_profiles.get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    root = attempt_root / "strict_gym_child_runtime"
    calls: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        material = source._materials[index]
        sequence = index + 1
        if environment == "reasoning_gym":
            source._fresh_results[index] = {
                "fresh_native_reward": response["reward"],
                "verifier_response": copy.deepcopy(response),
            }
            expected = source._expected_reasoning_score_call(index)
            del source._fresh_results[index]
            calls.append(
                {
                    "sequence": sequence,
                    "task_name": expected["task_name"],
                    "input": expected["input"],
                    "float_result": expected["float_result"],
                    "receipt": _artifact(
                        root / f"reasoning-score-call-{sequence:08d}.json",
                        profile.call_schema,
                        f"call-{sequence}",
                    ),
                }
            )
            continue
        request = {
            name: copy.deepcopy(material.derived_verifier_request[name])
            for name in (
                "responses_create_params",
                "response",
                "verifier",
            )
        }
        expected = child_runtime_v2.format_verification_call_expectation(
            environment=environment,
            derived_verifier_request=request,
            verifier_response=response,
        )
        calls.append(
            {
                "sequence": sequence,
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
                "receipt": _artifact(
                    root / f"format-verification-call-{sequence:08d}.json",
                    profile.call_schema,
                    f"call-{sequence}",
                ),
            }
        )
    terminal = {
        "schema": profile.call_index_schema,
        "environment": environment,
        "scope": "scorer-only",
        "pair_id": "pair-001",
        "job_id": "67890",
        "quiescence": {"original_process_reaped": True},
        "call_count": 4,
        "calls": calls,
    }
    if environment != "reasoning_gym":
        terminal["profile_id"] = profile_id
    terminal_ref = _artifact(
        attempt_root / profile.scorer_terminal_index_path,
        profile.call_index_schema,
        "terminal-index",
    )
    return terminal, terminal_ref


def _finalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    environment: str,
    profile_id: str,
) -> tuple[dict[str, Any], Path]:
    source, responses, attempt_root = _source(
        tmp_path,
        environment=environment,
        profile_id=profile_id,
    )
    terminal, terminal_ref = _terminal(
        environment=environment,
        profile_id=profile_id,
        source=source,
        responses=responses,
        attempt_root=attempt_root,
    )

    def load_terminal(path: Path, **kwargs):
        assert path == Path(terminal_ref["path"])
        expected_kwargs = {
            "expected_sha256": terminal_ref["sha256"],
            "expected_receipt_root": (attempt_root / "strict_gym_child_runtime"),
            "expected_pair_id": "pair-001",
            "expected_job_id": "67890",
        }
        if environment != "reasoning_gym":
            expected_kwargs.update(
                {
                    "expected_environment": environment,
                    "expected_profile_id": profile_id,
                }
            )
        assert kwargs == expected_kwargs
        return copy.deepcopy(terminal), terminal_ref["sha256"]

    loader_name = (
        "load_finalized_reasoning_score_call_index"
        if environment == "reasoning_gym"
        else "load_finalized_format_verification_call_index"
    )
    monkeypatch.setattr(child_runtime_v2, loader_name, load_terminal)
    for index, response in enumerate(responses):
        material = source.consume(
            rollout_index=index,
            generation_seed=source._materials[index].generation_seed,
        )
        assert material.rollout_index == index
        source.record_fresh_verifier_result(
            rollout_index=index,
            verifier_response=response,
        )
    document = source.finalize(
        replay_execution_manifest_sha256=_digest("replay-manifest"),
        authenticated_job_id="67890",
        process={
            "boot_id_sha256": _digest("boot-id"),
            "pid": 4321,
            "start_time_ticks": 987654,
        },
        scheduler_device_environment=_device_environment(),
        scorer_call_index_ref=terminal_ref,
    )
    assert source.phase == "complete-terminal"
    return document, attempt_root


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_profiled_transport_v3_binds_terminal_graph_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: str,
    profile_id: str,
) -> None:
    document, attempt_root = _finalize(
        monkeypatch,
        tmp_path,
        environment=environment,
        profile_id=profile_id,
    )
    profile = replay_profiles.get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert document["schema"] == MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA
    assert document["environment"] == environment
    assert document["scorer_profile"] == {
        "environment": environment,
        "profile_id": profile_id,
        "verifier_type": profile.verifier_type,
        "method": profile.method,
        "resource_config_path_name": profile.resource_config_path_name,
        "disabled_config_path_name": profile.disabled_config_path_name,
        "resource_app": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
        "resource_config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "fixture": {
            "path": profile.fixture_path,
            "sha256": profile.fixture_sha256,
            "rows": profile.fixture_rows,
        },
        "call_schema": profile.call_schema,
        "closed_schema": profile.closed_schema,
        "call_index_schema": profile.call_index_schema,
    }
    scorer = document["replay"]["scorer_evidence"]
    assert scorer["original_process_reaped"] is True
    assert scorer["terminal_index"]["path"].endswith(
        f"/{profile.scorer_terminal_index_path}"
    )
    assert len(profile.result_files) == 13
    assert [entry["rollout_index"] for entry in document["entries"]] == [
        0,
        1,
        2,
        3,
    ]
    validate_strict_model_transport_replay_consumption_v3(
        document,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    output = attempt_root / "model-transport-replay-consumption.json"
    published, digest = publish_strict_model_transport_replay_consumption_v3(
        output=output,
        document=document,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert published == output
    assert published.read_bytes() == canonical_ascii_json(document)
    assert hashlib.sha256(published.read_bytes()).hexdigest() == digest
    assert (published.stat().st_mode & 0o777) == 0o400


def test_profiled_transport_v3_rejects_cross_profile_and_unreaped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document, _ = _finalize(
        monkeypatch,
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    with pytest.raises(ValueError, match="profile|environment"):
        validate_strict_model_transport_replay_consumption_v3(
            document,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )
    poisoned = copy.deepcopy(document)
    poisoned["replay"]["scorer_evidence"]["original_process_reaped"] = False
    with pytest.raises(ValueError, match="reaped"):
        validate_strict_model_transport_replay_consumption_v3(
            poisoned,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_profiled_transport_v3_finalizer_poison_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, responses, attempt_root = _source(
        tmp_path,
        environment="freeform",
        profile_id="freeform-regex-v1",
    )
    terminal, terminal_ref = _terminal(
        environment="freeform",
        profile_id="freeform-regex-v1",
        source=source,
        responses=responses,
        attempt_root=attempt_root,
    )
    terminal["quiescence"]["original_process_reaped"] = False
    monkeypatch.setattr(
        child_runtime_v2,
        "load_finalized_format_verification_call_index",
        lambda *args, **kwargs: (terminal, terminal_ref["sha256"]),
    )
    for index, response in enumerate(responses):
        source.consume(
            rollout_index=index,
            generation_seed=source._materials[index].generation_seed,
        )
        source.record_fresh_verifier_result(
            rollout_index=index,
            verifier_response=response,
        )
    with pytest.raises(StrictModelTransportReplayError, match="not reaped"):
        source.finalize(
            replay_execution_manifest_sha256=_digest("replay-manifest"),
            authenticated_job_id="67890",
            process={
                "boot_id_sha256": _digest("boot-id"),
                "pid": 4321,
                "start_time_ticks": 987654,
            },
            scheduler_device_environment=_device_environment(),
            scorer_call_index_ref=terminal_ref,
        )
    assert source.phase == "poisoned"


def test_reasoning_transport_v3_dispatches_terminal_and_accepts_fractional_rewards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _install_reasoning_profile(monkeypatch)
    document, _ = _finalize(
        monkeypatch,
        tmp_path,
        environment=profile.environment,
        profile_id=profile.profile_id,
    )

    assert document["environment"] == "reasoning_gym"
    assert document["scorer_profile"]["verifier_type"] == "score_answer"
    assert document["scorer_profile"]["method"] == ("KnightsKnavesDataset.score_answer")
    assert [entry["fresh_native_reward"] for entry in document["entries"]] == list(
        _REASONING_SCORES
    )
    assert set(document["replay"]["scorer_evidence"]) == {
        "status",
        "terminal_index",
        "original_process_reaped",
    }
    validate_strict_model_transport_replay_consumption_v3(
        document,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )


def _reasoning_terminal_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    replay_profiles.StrictCapturedReplayProfile,
    StrictModelTransportReplaySourceV3,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    profile = _install_reasoning_profile(monkeypatch)
    source, responses, attempt_root = _source(
        tmp_path,
        environment=profile.environment,
        profile_id=profile.profile_id,
    )
    terminal, terminal_ref = _terminal(
        environment=profile.environment,
        profile_id=profile.profile_id,
        source=source,
        responses=responses,
        attempt_root=attempt_root,
    )
    for index, response in enumerate(responses):
        source.consume(
            rollout_index=index,
            generation_seed=source._materials[index].generation_seed,
        )
        source.record_fresh_verifier_result(
            rollout_index=index,
            verifier_response=response,
        )
    return profile, source, responses, terminal, terminal_ref


@pytest.mark.parametrize(
    "poison",
    [
        lambda terminal: terminal.__setitem__("environment", "freeform"),
        lambda terminal: terminal.__setitem__("scope", "main"),
        lambda terminal: terminal.__setitem__("pair_id", "pair-other"),
        lambda terminal: terminal.__setitem__("job_id", "67891"),
        lambda terminal: terminal.__setitem__("call_count", 3),
        lambda terminal: terminal["quiescence"].__setitem__(
            "original_process_reaped", False
        ),
        lambda terminal: terminal["calls"].pop(),
        lambda terminal: terminal["calls"][0].__setitem__("task_name", "other"),
        lambda terminal: terminal["calls"][0]["input"].__setitem__(
            "answer_sha256", _digest("forged-answer")
        ),
        lambda terminal: terminal["calls"][0]["input"].__setitem__(
            "entry_sha256", _digest("forged-entry")
        ),
        lambda terminal: terminal["calls"][0].__setitem__("float_result", 0.5),
    ],
)
def test_reasoning_transport_v3_rejects_terminal_identity_or_k4_join_poison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    poison: Any,
) -> None:
    _, source, _, terminal, terminal_ref = _reasoning_terminal_fixture(
        monkeypatch,
        tmp_path,
    )
    poison(terminal)
    monkeypatch.setattr(
        child_runtime_v2,
        "load_finalized_reasoning_score_call_index",
        lambda *args, **kwargs: (copy.deepcopy(terminal), terminal_ref["sha256"]),
    )

    with pytest.raises(StrictModelTransportReplayError):
        source.finalize(
            replay_execution_manifest_sha256=_digest("replay-manifest"),
            authenticated_job_id="67890",
            process={
                "boot_id_sha256": _digest("boot-id"),
                "pid": 4321,
                "start_time_ticks": 987654,
            },
            scheduler_device_environment=_device_environment(),
            scorer_call_index_ref=terminal_ref,
        )
    assert source.phase == "poisoned"


def test_reasoning_transport_v3_rejects_terminal_path_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, source, _, _, terminal_ref = _reasoning_terminal_fixture(
        monkeypatch,
        tmp_path,
    )
    terminal_ref["path"] = str(tmp_path / "attacker" / "reasoning-index.json")
    loader_called = False

    def load_terminal(*args: Any, **kwargs: Any) -> Any:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("reasoning terminal loader must not run")

    monkeypatch.setattr(
        child_runtime_v2,
        "load_finalized_reasoning_score_call_index",
        load_terminal,
    )
    with pytest.raises(StrictModelTransportReplayError, match="path differs"):
        source.finalize(
            replay_execution_manifest_sha256=_digest("replay-manifest"),
            authenticated_job_id="67890",
            process={
                "boot_id_sha256": _digest("boot-id"),
                "pid": 4321,
                "start_time_ticks": 987654,
            },
            scheduler_device_environment=_device_environment(),
            scorer_call_index_ref=terminal_ref,
        )
    assert loader_called is False


@pytest.mark.parametrize(
    "reward",
    [-0.0, True, 1, math.inf, math.nan, -0.001, 1.001],
)
def test_reasoning_transport_v3_rejects_invalid_fractional_reward_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reward: Any,
) -> None:
    profile = _install_reasoning_profile(monkeypatch)
    document, _ = _finalize(
        monkeypatch,
        tmp_path,
        environment=profile.environment,
        profile_id=profile.profile_id,
    )
    document["entries"][0]["fresh_native_reward"] = reward
    with pytest.raises((TypeError, ValueError)):
        validate_strict_model_transport_replay_consumption_v3(
            document,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )


def test_format_transport_v3_still_rejects_fractional_reward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document, _ = _finalize(
        monkeypatch,
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    document["entries"][0]["fresh_native_reward"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        validate_strict_model_transport_replay_consumption_v3(
            document,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_legacy_transport_import_does_not_execute_profile_registry() -> None:
    repository = Path(__file__).parents[3]
    script = r"""
import builtins
import sys

repository = sys.argv[1]
sys.path.insert(0, repository)
profile_module = "nemo_rl.utils.strict_captured_replay_profiles"
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == profile_module:
        raise RuntimeError("legacy transport imported profile registry")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import nemo_rl.utils.strict_model_transport_replay
assert profile_module not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(repository)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert completed.returncode == 0, completed.stderr
