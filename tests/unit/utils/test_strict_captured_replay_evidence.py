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

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import struct
from pathlib import Path

import pytest

from nemo_rl.utils.strict_captured_replay_evidence import (
    CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA,
    CROSS_ARM_PARITY_FIELDS,
    HASH_DOMAIN,
    MODEL_TRANSPORT_BUNDLE_SCHEMA,
    REPLAY_LEDGER_ROOT_KEYS,
    TRANSCRIPT_BUNDLE_ROOT_KEYS,
    TRANSCRIPT_BUNDLE_SCHEMA,
    TRANSCRIPT_ENTRY_KEYS,
    build_captured_replay_step1_ledger,
    build_transcript_bundle as _build_transcript_bundle,
    build_verifier_request_derivation,
    canonical_ascii_json,
    derive_nemo_gym_request_seed,
    document_sha256,
    domain_sha256,
    load_evidence_document,
    load_strict_fixture_row0,
    main_transcript_bundle_path,
    publish_evidence_document,
    publish_main_transcript_bundle,
    replay_job_name,
    replay_run_id,
    validate_captured_replay_step1_ledger,
    validate_captured_replay_source_join,
    validate_fresh_verifier_response,
    validate_ledger_transcript_join,
    validate_transcript_bundle,
    validate_verifier_request_derivation,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_load_strict_fixture_row0_stable_reads_exact_five_lf_rows(
    tmp_path: Path,
) -> None:
    rows = [
        {"agent_ref": {"name": "reasoning_gym_simple_agent"}, "text": "café"},
        *({"row": index} for index in range(1, 5)),
    ]
    raw = b"".join(
        ("  " + json.dumps(row, ensure_ascii=False) + "  \n").encode("utf-8")
        for row in rows
    )
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    assert (
        load_strict_fixture_row0(path=fixture.resolve(), expected_sha256=digest)
        == rows[0]
    )

    with pytest.raises(ValueError, match="Pair SHA-256"):
        load_strict_fixture_row0(
            path=fixture.resolve(), expected_sha256=_digest("different")
        )

    link = tmp_path / "fixture-link.jsonl"
    link.symlink_to(fixture)
    with pytest.raises(OSError):
        load_strict_fixture_row0(path=link, expected_sha256=digest)


@pytest.mark.parametrize(
    "bad_row",
    [
        b'{"duplicate":1,"duplicate":2}',
        b'{"negative_zero":-0}',
        b'{"negative_float_zero":-0.0}',
        b'{"nonfinite":NaN}',
        b"[]",
    ],
)
def test_load_strict_fixture_row0_rejects_non_strict_rows(
    tmp_path: Path, bad_row: bytes
) -> None:
    raw = bad_row + b"\n" + b'{"ok":true}\n' * 4
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_bytes(raw)
    with pytest.raises((TypeError, ValueError)):
        load_strict_fixture_row0(
            path=fixture.resolve(), expected_sha256=hashlib.sha256(raw).hexdigest()
        )


def _generation() -> dict[str, object]:
    return {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }


def _params(seed: int) -> dict[str, object]:
    return {
        "input": [{"role": "user", "content": "Zoey and Riley?"}],
        "max_output_tokens": 768,
        "metadata": {
            "extra_body": json.dumps(
                {"seed": seed}, sort_keys=True, separators=(",", ":")
            )
        },
        "temperature": 1.0,
        "top_p": 1.0,
    }


def _fixture_row() -> dict[str, object]:
    return {
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "reasoning_gym_simple_agent",
        },
        "answer": "Zoey",
        "metadata": {"source_dataset": "knights_knaves"},
        "question": "Zoey and Riley?",
        "responses_create_params": {
            "input": [{"role": "user", "content": "Zoey and Riley?"}]
        },
    }


def _expanded_params(params: dict[str, object]) -> dict[str, object]:
    message = copy.deepcopy(params["input"][0])
    message["type"] = "message"
    return {
        "background": None,
        "include": None,
        "input": [message],
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": copy.deepcopy(params["metadata"]),
        "model": None,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": None,
        "service_tier": None,
        "store": None,
        "stream": None,
        "temperature": 1.0,
        "text": None,
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": None,
        "top_p": 1.0,
        "truncation": None,
        "user": None,
    }


def _model_response(index: int, params: dict[str, object]) -> dict[str, object]:
    return {
        "background": None,
        "conversation": None,
        "created_at": 1.0,
        "error": None,
        "id": f"resp_{index:012x}40008000{index:012x}",
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": copy.deepcopy(params["metadata"]),
        "model": "/immutable/model",
        "object": "response",
        "output": [
            {
                "content": None,
                "encrypted_content": None,
                "generation_log_probs": [-0.1, -0.2],
                "generation_token_ids": [200 + index, 300 + index],
                "id": f"rs_{index:012x}40008000{index:012x}",
                "prompt_token_ids": [101, 102],
                "routed_experts": None,
                "summary": [{"text": f"answer {index}", "type": "summary_text"}],
                "type": "reasoning",
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "prompt_cache_key": None,
        "reasoning": None,
        "safety_identifier": None,
        "service_tier": None,
        "status": "completed",
        "temperature": 1.0,
        "text": None,
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": None,
        "top_p": 1.0,
        "truncation": None,
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 4,
        },
        "user": None,
    }


def _transport_ref() -> dict[str, str]:
    return {
        "path": (
            "/campaign/results/off/strict_model_transport/model-transport-bundle.json"
        ),
        "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
        "sha256": _digest("off-model-transport-bundle"),
    }


def _derivation() -> dict[str, object]:
    return build_verifier_request_derivation(
        gym_gitlink_commit="354babf7e3554fcd006807c86e80ef476aec9408",
        gym_tree="1234567890abcdef1234567890abcdef12345678",
        openai_version="2.6.1",
        pydantic_version="2.13.4",
    )


def build_transcript_bundle(**kwargs):
    """Keep fixtures concise while every bundle carries the pinned derivation."""
    kwargs.setdefault("verifier_request_derivation", _derivation())
    return _build_transcript_bundle(**kwargs)


def _entry(index: int, *, reward: float | None = None) -> dict[str, object]:
    seed = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=index
    )
    params = _params(seed)
    model_response = _model_response(index, params)
    raw_reward = float(index % 2) if reward is None else reward
    agent_run_request = {
        "_ng_rollout_index": index,
        "_ng_task_index": 3381613064844692360,
        "_rowidx": index,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "reasoning_gym_simple_agent",
        },
        "responses_create_params": copy.deepcopy(params),
        "answer": "Zoey",
        "metadata": {"source_dataset": "knights_knaves"},
        "question": "Zoey and Riley?",
    }
    verifier_response = {
        "responses_create_params": _expanded_params(params),
        "response": copy.deepcopy(model_response),
        "reward": raw_reward,
        "task_name": "knights_knaves",
        "score": raw_reward,
        "extracted_answer": "",
    }
    derived_verifier_request = copy.deepcopy(agent_run_request)
    derived_verifier_request["responses_create_params"] = _expanded_params(params)
    derived_verifier_request["response"] = copy.deepcopy(model_response)
    return {
        "sample_index": index,
        "fixture_row_index": 0,
        "rollout_index": index,
        "generation_seed": seed,
        "generation_request": params,
        "model_response": model_response,
        "agent_run_request": agent_run_request,
        "derived_verifier_request": derived_verifier_request,
        "verifier_response": verifier_response,
        "raw_environment_reward": raw_reward,
        "model_transport_entry_sha256": _digest(f"transport-entry-{index}"),
        "model_transport_request_body_sha256": _digest(f"transport-request-{index}"),
        "model_transport_response_body_sha256": _digest(f"transport-response-{index}"),
    }


def _format_fixture_row(environment: str) -> dict[str, object]:
    if environment == "citation":
        verifier = {
            "type": "string_match",
            "patterns": [r"\[web:\d+\]"],
            "expected_markers": ["[web:1]"],
        }
        agent = "citation_format_simple_agent"
    elif environment == "freeform":
        verifier = {
            "type": "regex",
            "pattern_id": "bullet_double_dash",
            "verify_regex": [r"^-- .+"],
            "verify_min_matches": 1,
        }
        agent = "freeform_formatting_simple_agent"
    else:
        raise AssertionError(f"unexpected format environment {environment!r}")
    return {
        "agent_ref": {"type": "responses_api_agents", "name": agent},
        "responses_create_params": {
            "input": [{"role": "user", "content": "format this answer"}]
        },
        "verifier": verifier,
    }


def _format_entry(index: int, environment: str) -> dict[str, object]:
    seed = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=index
    )
    params = {
        "input": [{"role": "user", "content": "format this answer"}],
        "max_output_tokens": 768,
        "metadata": {
            "extra_body": json.dumps(
                {"seed": seed}, sort_keys=True, separators=(",", ":")
            )
        },
        "temperature": 1.0,
        "top_p": 1.0,
    }
    fixture = _format_fixture_row(environment)
    agent_run_request = copy.deepcopy(fixture)
    agent_run_request.update(
        {
            "_ng_task_index": 3381613064844692360,
            "_rowidx": index,
            "_ng_rollout_index": index,
            "responses_create_params": copy.deepcopy(params),
        }
    )
    model_response = _model_response(index, params)
    derived_verifier_request = copy.deepcopy(agent_run_request)
    derived_verifier_request["responses_create_params"] = _expanded_params(params)
    derived_verifier_request["response"] = copy.deepcopy(model_response)
    if environment == "citation":
        match_details = {
            "expected": ["[web:1]"],
            "missing": ["[web:1]"],
            "spurious": [],
            "passed": False,
        }
    else:
        match_details = {"matching_lines": 0, "min_matches": 1, "passed": False}
    return {
        "sample_index": index,
        "fixture_row_index": 0,
        "rollout_index": index,
        "generation_seed": seed,
        "generation_request": params,
        "model_response": model_response,
        "agent_run_request": agent_run_request,
        "derived_verifier_request": derived_verifier_request,
        "verifier_response": {
            "responses_create_params": _expanded_params(params),
            "response": copy.deepcopy(model_response),
            "reward": 0.0,
            "verifier": copy.deepcopy(fixture["verifier"]),
            "match_details": match_details,
        },
        "raw_environment_reward": 0.0,
        "model_transport_entry_sha256": _digest(f"format-entry-{index}"),
        "model_transport_request_body_sha256": _digest(f"format-request-{index}"),
        "model_transport_response_body_sha256": _digest(f"format-response-{index}"),
    }


def _bindings(*, job_id: str = "81001", run_id: str | None = None) -> dict[str, str]:
    names = (
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "fixture_sha256",
        "verifier_source_sha256",
        "config_sha256",
        "snapshot_manifest_sha256",
    )
    result = {name: _digest(name) for name in names}
    result["job_id"] = job_id
    result["run_id"] = run_id or _digest("main-run")
    return result


def _main_bundle() -> dict[str, object]:
    return build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        verifier_request_derivation=_derivation(),
        entry_inputs=[_entry(index) for index in range(4)],
    )


def _replay_bundle(
    attempt_id: str = "replay-1", job_id: str = "82001"
) -> dict[str, object]:
    run_id = replay_run_id(
        environment="reasoning_gym",
        pair_id="rg-calibration",
        attempt_id=attempt_id,
    )
    return build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="on",
        mode="captured_replay",
        attempt_id=attempt_id,
        generation=_generation(),
        bindings=_bindings(job_id=job_id, run_id=run_id),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        verifier_request_derivation=_derivation(),
        entry_inputs=[_entry(index) for index in range(4)],
    )


def _ledger_rows(bundle: dict[str, object]) -> list[dict[str, object]]:
    group_id = "305fe72a-79e3-4b88-9eb2-000000000000"
    rloo_advantages = (
        -1.1546986103057861,
        1.1546984910964966,
        -1.1546986103057861,
        1.1546984910964966,
    )
    rows = []
    for entry in bundle["entries"]:
        prompt = entry["model_response"]["output"][0]["prompt_token_ids"]
        completion = entry["model_response"]["output"][0]["generation_token_ids"]
        token_ids = [*prompt, *completion]
        reward = entry["raw_environment_reward"]
        rows.append(
            {
                "sample_index": entry["sample_index"],
                "sample_id": f"{group_id}_g{entry['sample_index']}",
                "shared_prefix_group_id": group_id,
                "fixture_row_index": 0,
                "rollout_index": entry["rollout_index"],
                "generation_seed": entry["generation_seed"],
                "request_sha256": entry["generation_request_sha256"],
                "response_sha256": entry["model_response_sha256"],
                "agent_run_request_sha256": entry["agent_run_request_sha256"],
                "derived_verifier_request_sha256": entry[
                    "derived_verifier_request_sha256"
                ],
                "verifier_response_sha256": entry["verifier_response_sha256"],
                "token_ids": token_ids,
                "input_length": len(token_ids),
                "prompt_token_ids": prompt,
                "completion_token_ids": completion,
                "token_loss_mask": [0.0] * len(prompt) + [1.0] * len(completion),
                "raw_environment_reward": reward,
                "pre_penalty_environment_reward": reward,
                "penalty_flags": {
                    "reasoning_equal_to_final_answer": False,
                    "empty_final_answer": False,
                    "unwanted_token": False,
                    "malformed_think_tag": False,
                    "invalid_tool_call": False,
                    "malformed_thinking": False,
                    "raw_invalid_tool_call": False,
                    "raw_malformed_thinking": False,
                    "invalid_and_malformed": False,
                },
                "verifier_reward": reward,
                "processed_reward": reward,
                "sample_mask": 1.0,
                "advantages": [rloo_advantages[entry["sample_index"]]] * len(token_ids),
                "valid_loss_tokens": len(completion),
                "total_tokens": len(token_ids),
            }
        )
    return rows


def _replay_ledger(
    attempt_id: str = "replay-1", job_id: str = "82001"
) -> tuple[dict[str, object], dict[str, object]]:
    source = _main_bundle()
    replay = _replay_bundle(attempt_id, job_id)
    source_ref = {
        "path": "/campaign/results/off/strict_pair_step1_evidence/transcript-bundle.json",
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(source, trailing_lf=False),
    }
    replay_ref = {
        "path": f"/campaign/results/captured_replay/{attempt_id}/transcript-bundle.json",
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(replay, trailing_lf=False),
    }
    bindings = {
        **_bindings(
            job_id=job_id,
            run_id=replay_run_id(
                environment="reasoning_gym",
                pair_id="rg-calibration",
                attempt_id=attempt_id,
            ),
        ),
        "restart_count": 0,
        "pair_campaign_sha256": _digest("campaign"),
        "pair_campaign_reward_and_advantage_sha256": _digest("reward-policy"),
        "process": {
            "boot_id_sha256": _digest(f"{attempt_id}-boot"),
            "pid": 41001 if attempt_id == "replay-1" else 41002,
            "start_time_ticks": 51001 if attempt_id == "replay-1" else 51002,
        },
    }
    ledger = build_captured_replay_step1_ledger(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        attempt_id=attempt_id,
        source_main_ledger_sha256=_digest("off-main-ledger"),
        source_transcript_bundle=source_ref,
        source_transcript_document=source,
        generation=_generation(),
        bindings=bindings,
        transcript_bundle=replay_ref,
        transcript_document=replay,
        row_inputs=_ledger_rows(replay),
    )
    return ledger, replay


def _source_ref() -> dict[str, str]:
    bundle = _main_bundle()
    return {
        "path": "/campaign/results/off/strict_pair_step1_evidence/transcript-bundle.json",
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(bundle, trailing_lf=False),
    }


def test_transcript_bundle_has_exact_frozen_schema_and_separate_domains() -> None:
    bundle = _main_bundle()
    assert set(bundle) == set(TRANSCRIPT_BUNDLE_ROOT_KEYS)
    assert bundle["schema"] == TRANSCRIPT_BUNDLE_SCHEMA
    assert bundle["hash_domain"] == HASH_DOMAIN
    assert bundle["sample_count"] == 4
    assert bundle["attempt_id"] is None
    assert [entry["fixture_row_index"] for entry in bundle["entries"]] == [0] * 4

    entry = bundle["entries"][0]
    assert set(entry) == set(TRANSCRIPT_ENTRY_KEYS)
    assert entry["generation_request_sha256"] == domain_sha256(
        "step1-generation-request", entry["generation_request"]
    )
    assert entry["model_response_sha256"] == domain_sha256(
        "step1-model-response", entry["model_response"]
    )
    assert entry["agent_run_request_sha256"] == domain_sha256(
        "step1-agent-run-request", entry["agent_run_request"]
    )
    assert entry["derived_verifier_request_sha256"] == domain_sha256(
        "step1-derived-verifier-request", entry["derived_verifier_request"]
    )
    assert entry["verifier_response_sha256"] == domain_sha256(
        "step1-verifier-response", entry["verifier_response"]
    )
    assert bundle["entries_sha256"] == domain_sha256(
        "step1-transcript-entries", bundle["entries"]
    )
    raw = canonical_ascii_json(bundle)
    assert not raw.endswith(b"\n")
    assert raw == json.dumps(
        bundle,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    validate_transcript_bundle(bundle)

    generation = _generation()
    generation["seed_base"] = 43
    with pytest.raises(ValueError, match="generation policy is not frozen"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=generation,
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=[_entry(index) for index in range(4)],
        )


def test_r334_derivation_contract_and_old_evidence_versions_are_rejected() -> None:
    derivation = _derivation()
    assert derivation == {
        "schema": "nemo-rl-strict-derived-verifier-request-v1",
        "assurance": "deterministic-reconstruction-not-wire-capture",
        "algorithm": "pinned-simple-agent-model-dump-v1",
        "gym_gitlink_commit": "354babf7e3554fcd006807c86e80ef476aec9408",
        "gym_tree": "1234567890abcdef1234567890abcdef12345678",
        "runtime": {"openai_version": "2.6.1", "pydantic_version": "2.13.4"},
        "sources": {
            "simple_agent": {
                "path": "responses_api_agents/simple_agent/app.py",
                "sha256": "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a",
            },
            "base_resources": {
                "path": "nemo_gym/base_resources_server.py",
                "sha256": "b106a97397cdce8da2c1dbacd0b0b4b862ec03e664704e38044025fc9046693d",
            },
            "openai_utils": {
                "path": "nemo_gym/openai_utils.py",
                "sha256": "2e612f284de3cd290f76ccea8eccf577805127cfe3ea92d24f95b4ca4a068dce",
            },
        },
    }
    validate_verifier_request_derivation(derivation)

    changed = copy.deepcopy(derivation)
    changed["runtime"]["openai_version"] = "2.7.2"
    with pytest.raises(ValueError, match="pinned contract"):
        validate_verifier_request_derivation(changed)

    missing = copy.deepcopy(derivation)
    del missing["runtime"]["openai_version"]
    with pytest.raises(ValueError, match="runtime keyset mismatch"):
        validate_verifier_request_derivation(missing)

    with pytest.raises(TypeError, match="openai_version"):
        build_verifier_request_derivation(
            gym_gitlink_commit="354babf7e3554fcd006807c86e80ef476aec9408",
            gym_tree="1234567890abcdef1234567890abcdef12345678",
            pydantic_version="2.13.4",
        )

    old_transcript = _main_bundle()
    old_transcript["schema"] = "nemo-rl-strict-step1-transcript-bundle-v3"
    with pytest.raises(ValueError, match="unexpected transcript bundle schema"):
        validate_transcript_bundle(old_transcript)

    old_ledger, transcript = _replay_ledger()
    old_ledger["schema"] = "nemo-rl-strict-captured-replay-step1-ledger-v4"
    with pytest.raises(ValueError, match="unexpected captured replay ledger schema"):
        validate_captured_replay_step1_ledger(
            old_ledger,
            source_transcript_document=_main_bundle(),
            transcript_document=transcript,
        )


def test_model_response_digest_excludes_reward_and_verifier_transcript() -> None:
    first = _main_bundle()
    entries = [_entry(index) for index in range(4)]
    entries[0] = _entry(0, reward=1.0)
    second = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=entries,
    )
    assert (
        first["entries"][0]["model_response_sha256"]
        == second["entries"][0]["model_response_sha256"]
    )
    assert (
        first["entries"][0]["verifier_response_sha256"]
        != second["entries"][0]["verifier_response_sha256"]
    )
    assert first["entries"][0]["entry_sha256"] != second["entries"][0]["entry_sha256"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda entries: entries[0].update(fixture_row_index=1),
            "fixture_row_index=0",
        ),
        (
            lambda entries: entries[0].update(generation_seed=7),
            "generation seed does not close",
        ),
        (
            lambda entries: entries[0]["generation_request"].update(top_p=0.5),
            "fixture-to-generation transform differs",
        ),
        (
            lambda entries: entries[0]["model_response"]["output"][0][
                "generation_token_ids"
            ].__setitem__(0, 2_147_483_648)
            or entries[0]["verifier_response"].update(
                response=copy.deepcopy(entries[0]["model_response"])
            )
            or entries[0]["derived_verifier_request"].update(
                response=copy.deepcopy(entries[0]["model_response"])
            ),
            "2147483647",
        ),
        (
            lambda entries: (
                entries[0].update(raw_environment_reward=1.25),
                entries[0]["verifier_response"].update(reward=1.25, score=1.25),
            ),
            "must be in \\[0, 1\\]",
        ),
        (
            lambda entries: entries[0]["verifier_response"].update(extra=True),
            "verifier_response keyset mismatch",
        ),
    ],
)
def test_transcript_builder_rejects_identity_transport_and_reward_drift(
    mutation, match: str
) -> None:
    entries = [_entry(index) for index in range(4)]
    mutation(entries)
    with pytest.raises((TypeError, ValueError), match=match):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )


def test_validator_recomputes_every_transcript_digest() -> None:
    bundle = copy.deepcopy(_main_bundle())
    bundle["entries"][0]["model_response_sha256"] = _digest("forged")
    with pytest.raises(ValueError, match="changed or non-derived"):
        validate_transcript_bundle(bundle)

    bundle = copy.deepcopy(_main_bundle())
    bundle["bindings"]["fixture_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nonzero lowercase SHA-256"):
        validate_transcript_bundle(bundle)


def test_transcript_rejects_nested_numeric_alias_wrong_agent_and_response() -> None:
    entries = [_entry(index) for index in range(4)]
    entries[0]["agent_run_request"]["responses_create_params"]["max_output_tokens"] = (
        768.0
    )
    with pytest.raises(ValueError, match="fixture-to-agent-run transform differs"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )

    entries = [_entry(index) for index in range(4)]
    entries[0]["agent_run_request"]["agent_ref"]["name"] = "wrong_agent"
    with pytest.raises(ValueError, match="verifier agent"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )

    entries = [_entry(index) for index in range(4)]
    entries[0]["model_response"]["max_output_tokens"] = 768.0
    entries[0]["verifier_response"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    with pytest.raises(ValueError, match="exact integer 768"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )

    entries = [_entry(index) for index in range(4)]
    del entries[0]["generation_request"]["input"]
    del entries[0]["agent_run_request"]["responses_create_params"]["input"]
    del entries[0]["verifier_response"]["responses_create_params"]["input"]
    entries[0]["verifier_response"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    with pytest.raises(ValueError, match="fixture-to-agent-run transform differs"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )

    entries = [_entry(index) for index in range(4)]
    entries[0]["model_response"]["id"] = "resp_not-a-uuid"
    entries[0]["verifier_response"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    with pytest.raises(ValueError, match="canonical UUID4 resp_ identifier"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )

    entries = [_entry(index) for index in range(4)]
    entries[0]["model_response"]["reward"] = 1.0
    entries[0]["verifier_response"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        entries[0]["model_response"]
    )
    with pytest.raises(ValueError, match="must not contain a root reward"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda entries: entries[0]["verifier_response"].update(
                task_name="wrong_dataset"
            ),
            "task_name differs",
        ),
        (
            lambda entries: entries[0]["verifier_response"].update(score=1.0),
            "score differs",
        ),
        (
            lambda entries: entries[0]["verifier_response"].update(
                extracted_answer="invented"
            ),
            "extracted_answer differs",
        ),
        (
            lambda entries: entries[0]["verifier_response"]["responses_create_params"][
                "input"
            ][0].pop("type"),
            "pinned Pydantic expansion",
        ),
    ],
)
def test_reasoning_gym_r331_verifier_response_is_exact(mutation, match: str) -> None:
    entries = [_entry(index) for index in range(4)]
    mutation(entries)
    with pytest.raises((TypeError, ValueError), match=match):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )


def test_reasoning_gym_r332_accepts_partial_reward() -> None:
    entries = [_entry(index) for index in range(4)]
    partial_reward = 0.3 + (0.7 * 1 / 2)
    entries[0]["raw_environment_reward"] = partial_reward
    entries[0]["verifier_response"].update(reward=partial_reward, score=partial_reward)
    bundle = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=entries,
    )
    assert bundle["entries"][0]["raw_environment_reward"] == partial_reward


@pytest.mark.parametrize(("field", "value"), [("question", 123), ("answer", 123)])
def test_reasoning_gym_fixture_rejects_non_pinned_scalar_types(
    field: str, value: object
) -> None:
    fixture = _fixture_row()
    fixture[field] = value
    with pytest.raises(TypeError, match=f"fixture_row.{field}"):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=fixture,
            model_transport_bundle=_transport_ref(),
            entry_inputs=[_entry(index) for index in range(4)],
        )


def test_reasoning_gym_fixture_accepts_nullable_answer() -> None:
    fixture = _fixture_row()
    fixture["answer"] = None
    entries = [_entry(index) for index in range(4)]
    for entry in entries:
        entry["agent_run_request"]["answer"] = None
        entry["derived_verifier_request"]["answer"] = None
    build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=fixture,
        model_transport_bundle=_transport_ref(),
        entry_inputs=entries,
    )


@pytest.mark.parametrize(
    ("reward", "match"),
    [
        (-0.0, "negative zero"),
        (float("nan"), "finite float"),
        (float("inf"), "finite float"),
        (-0.01, "must be in \\[0, 1\\]"),
        (1.01, "must be in \\[0, 1\\]"),
        (1, "exact finite float"),
    ],
)
def test_reasoning_gym_r332_rejects_invalid_partial_reward(
    reward: object, match: str
) -> None:
    entries = [_entry(index) for index in range(4)]
    entries[0]["raw_environment_reward"] = reward
    entries[0]["verifier_response"].update(reward=reward, score=reward)
    with pytest.raises((TypeError, ValueError), match=match):
        build_transcript_bundle(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_fixture_row(),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )


@pytest.mark.parametrize(
    ("text", "extracted_answer"),
    [
        (
            "<answer> first </answer> ignored <answer>\n final \n</answer>",
            "final",
        ),
        (r"work \boxed{ 42 }", "42"),
        ("  plain answer  ", "plain answer"),
    ],
)
def test_reasoning_gym_r331_extracted_answer_matches_pinned_precedence(
    text: str, extracted_answer: str
) -> None:
    entries = [_entry(index) for index in range(4)]
    model_response = entries[0]["model_response"]
    model_response["output"] = [
        {
            "content": [
                {
                    "annotations": [],
                    "logprobs": None,
                    "text": text,
                    "type": "output_text",
                }
            ],
            "generation_log_probs": [-0.1, -0.2],
            "generation_token_ids": [200, 300],
            "id": "msg_00000000000040008000000000000000",
            "prompt_token_ids": [101, 102],
            "role": "assistant",
            "routed_experts": None,
            "status": "completed",
            "type": "message",
        }
    ]
    entries[0]["verifier_response"]["response"] = copy.deepcopy(model_response)
    entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(model_response)
    entries[0]["verifier_response"]["extracted_answer"] = extracted_answer

    bundle = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=entries,
    )
    validate_transcript_bundle(bundle)


def test_fresh_verifier_response_closes_to_derived_request_and_captured_output() -> (
    None
):
    entry = _entry(0, reward=0.6499999999999999)
    reward = validate_fresh_verifier_response(
        environment="reasoning_gym",
        agent_run_request=entry["agent_run_request"],
        derived_verifier_request=entry["derived_verifier_request"],
        model_response=entry["model_response"],
        verifier_response=entry["verifier_response"],
    )
    assert reward == 0.6499999999999999

    forged = copy.deepcopy(entry["derived_verifier_request"])
    forged["response"]["id"] = "resp_00000000000040008000000000000001"
    with pytest.raises(ValueError, match="pinned reconstruction"):
        validate_fresh_verifier_response(
            environment="reasoning_gym",
            agent_run_request=entry["agent_run_request"],
            derived_verifier_request=forged,
            model_response=entry["model_response"],
            verifier_response=entry["verifier_response"],
        )

    forged_result = copy.deepcopy(entry["verifier_response"])
    forged_result["score"] = 1.0
    with pytest.raises(ValueError, match="score differs"):
        validate_fresh_verifier_response(
            environment="reasoning_gym",
            agent_run_request=entry["agent_run_request"],
            derived_verifier_request=entry["derived_verifier_request"],
            model_response=entry["model_response"],
            verifier_response=forged_result,
        )


@pytest.mark.parametrize("environment", ["citation", "freeform"])
def test_format_r331_verifier_response_closes_exact_semantics(environment: str) -> None:
    entries = [_format_entry(index, environment) for index in range(4)]
    bundle = build_transcript_bundle(
        pair_id=f"{environment}-calibration",
        environment=environment,
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_format_fixture_row(environment),
        model_transport_bundle=_transport_ref(),
        entry_inputs=entries,
    )
    validate_transcript_bundle(bundle)

    entries[0]["verifier_response"]["match_details"]["passed"] = True
    with pytest.raises(ValueError, match="match_details differs"):
        build_transcript_bundle(
            pair_id=f"{environment}-calibration",
            environment=environment,
            arm="off",
            mode="observe",
            attempt_id=None,
            generation=_generation(),
            bindings=_bindings(),
            fixture_row=_format_fixture_row(environment),
            model_transport_bundle=_transport_ref(),
            entry_inputs=entries,
        )


def test_captured_replay_source_join_binds_inputs_but_not_fresh_reward() -> None:
    source = _main_bundle()
    replay_bindings = _bindings(
        job_id="82001",
        run_id=replay_run_id(
            environment="reasoning_gym",
            pair_id="rg-calibration",
            attempt_id="replay-1",
        ),
    )
    replay_bindings["snapshot_manifest_sha256"] = _digest("on-snapshot")
    replay = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="on",
        mode="captured_replay",
        attempt_id="replay-1",
        generation=_generation(),
        bindings=replay_bindings,
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=[_entry(index) for index in range(4)],
    )
    validate_captured_replay_source_join(
        source_transcript_bundle=source, replay_transcript_bundle=replay
    )

    fresh_uuid_inputs = [_entry(index) for index in range(4)]
    fresh_uuid_inputs[0]["model_response"]["id"] = (
        "resp_00000000009940008000000000000099"
    )
    fresh_uuid_inputs[0]["model_response"]["output"][0]["id"] = (
        "rs_00000000009940008000000000000099"
    )
    fresh_uuid_inputs[0]["verifier_response"]["response"] = copy.deepcopy(
        fresh_uuid_inputs[0]["model_response"]
    )
    fresh_uuid_inputs[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        fresh_uuid_inputs[0]["model_response"]
    )
    fresh_uuid_replay = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="on",
        mode="captured_replay",
        attempt_id="replay-1",
        generation=_generation(),
        bindings=replay_bindings,
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=fresh_uuid_inputs,
    )
    validate_captured_replay_source_join(
        source_transcript_bundle=source,
        replay_transcript_bundle=fresh_uuid_replay,
    )

    changed_reward = _replay_bundle()
    changed_reward["entries"][0]["verifier_response"]["reward"] = 1.0
    changed_reward["entries"][0]["verifier_response"]["score"] = 1.0
    changed_reward["entries"][0]["raw_environment_reward"] = 1.0
    changed_reward["entries"][0]["verifier_response_sha256"] = domain_sha256(
        "step1-verifier-response", changed_reward["entries"][0]["verifier_response"]
    )
    changed_reward["entries"][0]["entry_sha256"] = domain_sha256(
        "step1-transcript-entry",
        {
            key: value
            for key, value in changed_reward["entries"][0].items()
            if key != "entry_sha256"
        },
    )
    changed_reward["entries_sha256"] = domain_sha256(
        "step1-transcript-entries", changed_reward["entries"]
    )
    validate_captured_replay_source_join(
        source_transcript_bundle=source, replay_transcript_bundle=changed_reward
    )

    changed_request = copy.deepcopy(replay)
    changed_request["entries"][0]["agent_run_request"]["question"] = "other"
    changed_request["entries"][0]["agent_run_request_sha256"] = domain_sha256(
        "step1-agent-run-request",
        changed_request["entries"][0]["agent_run_request"],
    )
    changed_request["entries"][0]["entry_sha256"] = domain_sha256(
        "step1-transcript-entry",
        {
            key: value
            for key, value in changed_request["entries"][0].items()
            if key != "entry_sha256"
        },
    )
    changed_request["entries_sha256"] = domain_sha256(
        "step1-transcript-entries", changed_request["entries"]
    )
    with pytest.raises(ValueError, match="fixture-to-agent-run transform differs"):
        validate_captured_replay_source_join(
            source_transcript_bundle=source,
            replay_transcript_bundle=changed_request,
        )


def test_replay_ledger_v5_builds_and_closes_to_transcript_preimages() -> None:
    ledger, transcript = _replay_ledger()
    assert set(ledger) == set(REPLAY_LEDGER_ROOT_KEYS)
    assert ledger["schema"] == CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA
    assert ledger["mode"] == "captured_replay"
    assert ledger["attempt_id"] == "replay-1"
    assert ledger["source_main_ledger_sha256"] == _digest("off-main-ledger")
    assert ledger["transcript_bundle"]["sha256"] == document_sha256(
        transcript, trailing_lf=False
    )
    assert ledger["compared_fields"] == list(CROSS_ARM_PARITY_FIELDS)
    assert len(ledger["rows"]) == 4
    assert all(row["fixture_row_index"] == 0 for row in ledger["rows"])
    validate_captured_replay_step1_ledger(
        ledger,
        source_transcript_document=_main_bundle(),
        transcript_document=transcript,
    )
    validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)


def test_replay_ledger_r333_joins_native_reward_through_exact_float32() -> None:
    native_reward = 0.3 + (0.7 * 1 / 2)
    trainer_reward = struct.unpack(">f", struct.pack(">f", native_reward))[0]
    entry_inputs = [_entry(index, reward=native_reward) for index in range(4)]
    source = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=_bindings(),
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=copy.deepcopy(entry_inputs),
    )
    replay_bindings = _bindings(
        job_id="82001",
        run_id=replay_run_id(
            environment="reasoning_gym",
            pair_id="rg-calibration",
            attempt_id="replay-1",
        ),
    )
    replay_bindings["snapshot_manifest_sha256"] = _digest("on-snapshot")
    replay = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="on",
        mode="captured_replay",
        attempt_id="replay-1",
        generation=_generation(),
        bindings=replay_bindings,
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=copy.deepcopy(entry_inputs),
    )
    rows = _ledger_rows(replay)
    for row in rows:
        for field in (
            "raw_environment_reward",
            "pre_penalty_environment_reward",
            "verifier_reward",
            "processed_reward",
        ):
            row[field] = trainer_reward
        row["advantages"] = [0.0] * row["input_length"]
    source_ref = {
        "path": "/campaign/results/off/strict_pair_step1_evidence/transcript-bundle.json",
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(source, trailing_lf=False),
    }
    replay_ref = {
        "path": "/campaign/results/captured_replay/replay-1/transcript-bundle.json",
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(replay, trailing_lf=False),
    }
    ledger_bindings = {
        **replay_bindings,
        "restart_count": 0,
        "pair_campaign_sha256": _digest("campaign"),
        "pair_campaign_reward_and_advantage_sha256": _digest("reward-policy"),
        "process": {
            "boot_id_sha256": _digest("partial-boot"),
            "pid": 41001,
            "start_time_ticks": 51001,
        },
    }
    ledger = build_captured_replay_step1_ledger(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        attempt_id="replay-1",
        source_main_ledger_sha256=_digest("off-main-ledger"),
        source_transcript_bundle=source_ref,
        source_transcript_document=source,
        generation=_generation(),
        bindings=ledger_bindings,
        transcript_bundle=replay_ref,
        transcript_document=replay,
        row_inputs=rows,
    )
    assert ledger["rows"][0]["raw_environment_reward"] == trainer_reward
    assert trainer_reward != native_reward
    validate_ledger_transcript_join(ledger=ledger, transcript_bundle=replay)

    bad = copy.deepcopy(ledger)
    bad["rows"][0]["raw_environment_reward"] = native_reward
    with pytest.raises(ValueError, match="float32 transcript raw_environment_reward"):
        validate_ledger_transcript_join(ledger=bad, transcript_bundle=replay)


def test_replay_ledger_builder_requires_joined_source_documents() -> None:
    ledger, replay = _replay_ledger()
    changed_bindings = _bindings()
    changed_bindings["fixture_sha256"] = _digest("other-fixture")
    changed_source = build_transcript_bundle(
        pair_id="rg-calibration",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings=changed_bindings,
        fixture_row=_fixture_row(),
        model_transport_bundle=_transport_ref(),
        entry_inputs=[_entry(index) for index in range(4)],
    )
    changed_source_ref = copy.deepcopy(ledger["source_transcript_bundle"])
    changed_source_ref["sha256"] = document_sha256(changed_source, trailing_lf=False)
    with pytest.raises(ValueError, match="binding fixture_sha256 differs"):
        build_captured_replay_step1_ledger(
            pair_id="rg-calibration",
            environment="reasoning_gym",
            attempt_id="replay-1",
            source_main_ledger_sha256=_digest("off-main-ledger"),
            source_transcript_bundle=changed_source_ref,
            source_transcript_document=changed_source,
            generation=_generation(),
            bindings=ledger["bindings"],
            transcript_bundle=ledger["transcript_bundle"],
            transcript_document=replay,
            row_inputs=_ledger_rows(replay),
        )


def test_ledger_transcript_join_binds_prompt_uuid_to_task_index() -> None:
    ledger, transcript = _replay_ledger()
    different_group_id = "305fe72a-79e3-4b88-9eb2-000000000001"
    for index, row in enumerate(ledger["rows"]):
        row["shared_prefix_group_id"] = different_group_id
        row["sample_id"] = f"{different_group_id}_g{index}"
    with pytest.raises(ValueError, match="does not close to transcript task index"):
        validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("response_sha256", _digest("wrong-response"), "response_sha256"),
        (
            "agent_run_request_sha256",
            _digest("wrong-verifier-request"),
            "agent_run_request_sha256",
        ),
        (
            "derived_verifier_request_sha256",
            _digest("wrong-derived-verifier-request"),
            "derived_verifier_request_sha256",
        ),
        (
            "verifier_response_sha256",
            _digest("wrong-verifier-response"),
            "verifier_response_sha256",
        ),
    ],
)
def test_ledger_transcript_join_rejects_each_transport_digest_drift(
    field: str, replacement: str, match: str
) -> None:
    ledger, transcript = _replay_ledger()
    ledger["rows"][0][field] = replacement
    with pytest.raises(ValueError, match=match):
        validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)


def test_replay_ledger_validator_rejects_bool_int_alias_and_forged_process() -> None:
    ledger, transcript = _replay_ledger()
    ledger["step"] = True
    with pytest.raises(ValueError, match="step 1 K=4"):
        validate_captured_replay_step1_ledger(
            ledger,
            source_transcript_document=_main_bundle(),
            transcript_document=transcript,
        )

    ledger, transcript = _replay_ledger()
    ledger["bindings"]["process"]["pid"] = 1 << 31
    with pytest.raises(ValueError, match="exceeds"):
        validate_captured_replay_step1_ledger(
            ledger,
            source_transcript_document=_main_bundle(),
            transcript_document=transcript,
        )


def test_pair_id_exactly_accepts_64_and_rejects_65_ascii_characters() -> None:
    assert replay_job_name(pair_id="p" * 64, attempt_id="replay-1").endswith("p" * 64)
    with pytest.raises(ValueError, match="exceeds 64 ASCII bytes"):
        replay_job_name(pair_id="p" * 65, attempt_id="replay-1")


def test_frozen_paths_and_deterministic_identities() -> None:
    assert main_transcript_bundle_path("/campaign/results/off") == Path(
        "/campaign/results/off/strict_pair_step1_evidence/transcript-bundle.json"
    )
    assert (
        replay_run_id(
            environment="reasoning_gym",
            pair_id="rg-calibration",
            attempt_id="replay-2",
        )
        == hashlib.sha256(
            b"nemo-rl-strict-replay-v2:reasoning_gym:rg-calibration:replay-2"
        ).hexdigest()
    )


def test_publish_and_load_preserve_framing_mode_and_no_clobber(
    tmp_path: Path,
) -> None:
    os.chmod(tmp_path, 0o700)
    bundle = _main_bundle()
    bundle_path, bundle_sha = publish_main_transcript_bundle(
        results_dir=str(tmp_path), document=bundle
    )
    raw = bundle_path.read_bytes()
    assert not raw.endswith(b"\n")
    assert stat.S_IMODE(os.lstat(bundle_path).st_mode) == 0o400
    assert os.lstat(bundle_path).st_nlink == 1
    loaded, actual_sha = load_evidence_document(
        path=bundle_path, expected_sha256=bundle_sha, trailing_lf=False
    )
    assert loaded == bundle
    assert actual_sha == bundle_sha
    with pytest.raises(FileExistsError, match="already exists"):
        publish_main_transcript_bundle(results_dir=str(tmp_path), document=bundle)

    receipt_directory = tmp_path / "replay"
    receipt_directory.mkdir(mode=0o700)
    submission = {"schema": "test-receipt", "status": "complete"}
    receipt_path, receipt_sha = publish_evidence_document(
        output=receipt_directory / "submission-receipt.json",
        document=submission,
        trailing_lf=True,
    )
    assert receipt_path.read_bytes().endswith(b"\n")
    loaded_receipt, loaded_sha = load_evidence_document(
        path=receipt_path,
        expected_sha256=receipt_sha,
        trailing_lf=True,
    )
    assert loaded_receipt == submission
    assert loaded_sha == receipt_sha


def test_load_rejects_hard_link_and_framing_confusion(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    directory = tmp_path / "evidence"
    directory.mkdir(mode=0o700)
    bundle = _main_bundle()
    path, digest = publish_evidence_document(
        output=directory / "bundle.json", document=bundle, trailing_lf=False
    )
    os.link(path, directory / "alias.json")
    with pytest.raises(RuntimeError, match="single-link"):
        load_evidence_document(path=path, expected_sha256=digest, trailing_lf=False)

    submission = {"schema": "test-receipt", "status": "complete"}
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    receipt_path, receipt_digest = publish_evidence_document(
        output=other / "submission.json", document=submission, trailing_lf=True
    )
    with pytest.raises(ValueError, match="must not end in LF"):
        load_evidence_document(
            path=receipt_path,
            expected_sha256=receipt_digest,
            trailing_lf=False,
        )
