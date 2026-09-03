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

import asyncio
import json
import uuid
from copy import deepcopy
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import (
    _stamp_strict_main_step1_tags,
)
from nemo_rl.experience.rollout_manager import (
    AsyncNemoGymRolloutImpl,
    _publish_strict_main_step1_transcript,
)
from nemo_rl.experience.interfaces import (
    NEMO_GYM_REQUEST_SEEDS_METADATA_KEY,
    NEMO_GYM_REWARD_PENALTY_FLAG_KEYS,
    NEMO_GYM_REWARD_PENALTY_FLAGS_KEY,
    NEMO_GYM_TRANSCRIPT_BUNDLE_SHA256_METADATA_KEY,
    Completion,
    PromptGroupRecord,
)
from nemo_rl.experience.rollouts import apply_reward_penalties
from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_REWARD_PENALTY_TAGS,
    MAIN_STEP1_TAG_FIXTURE_ROW_INDEX,
    MAIN_STEP1_TAG_GENERATION_SEED,
    MAIN_STEP1_TAG_ROLLOUT_INDEX,
    MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256,
    derive_nemo_gym_request_seed,
)


_STRICT_GROUP_ID = "12345678-1234-4678-9234-567812345678"
_STRICT_TASK_INDEX = (
    uuid.UUID(_STRICT_GROUP_ID).int ^ (uuid.UUID(_STRICT_GROUP_ID).int >> 64)
) & ((1 << 63) - 1)
_FIXTURE_ROW = {
    "agent_ref": {
        "type": "responses_api_agents",
        "name": "reasoning_gym_simple_agent",
    },
    "answer": "2",
    "metadata": {"source_dataset": "basic_arithmetic"},
    "question": "1+1?",
    "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
}


def _flags(**updates: bool) -> dict[str, bool]:
    values = {key: False for key in NEMO_GYM_REWARD_PENALTY_FLAG_KEYS}
    values.update(updates)
    return values


def test_reward_penalties_preserve_exact_per_completion_decisions() -> None:
    result = {
        "message_log": [
            {"role": "user", "token_ids": [10]},
            {"role": "assistant", "token_ids": [2, 11]},
        ],
        "full_result": {
            "reward": 1.0,
            "response": {
                "output": [
                    {"type": "reasoning", "summary": [{"text": "same"}]},
                    {"type": "message", "content": [{"text": "same"}]},
                ]
            },
        },
    }

    apply_reward_penalties(
        [result],
        {
            "penalize_duplicated_reasoning": True,
            "penalize_unwanted_tokens": True,
            "token_ids": {"unwanted": [2]},
        },
    )

    assert result[NEMO_GYM_REWARD_PENALTY_FLAGS_KEY] == _flags(
        reasoning_equal_to_final_answer=True,
        unwanted_token=True,
    )


def test_tq_tags_bind_each_consumed_row_to_request_and_penalty_decisions() -> None:
    group_id = "12345678-1234-4678-9234-567812345678"
    seeds = [101, 102, 103, 104]
    completions = [
        Completion(
            message_log=[],
            env_extras=None,
            truncated=False,
            reward=float(index % 2),
            reward_penalty_flags=_flags(unwanted_token=index == 3),
        )
        for index in range(4)
    ]
    record = PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info=None,
        metadata={
            NEMO_GYM_REQUEST_SEEDS_METADATA_KEY: seeds,
            NEMO_GYM_TRANSCRIPT_BUNDLE_SHA256_METADATA_KEY: "a" * 64,
        },
        completions=completions,
        rollout_metrics={},
    )
    sample_ids = [f"{group_id}_g{index}" for index in range(4)]
    tags = [{"weight_version": 0} for _ in range(4)]

    _stamp_strict_main_step1_tags(
        record=record,
        sample_ids=sample_ids,
        tags=tags,
    )

    for index, tag in enumerate(tags):
        assert tag[MAIN_STEP1_TAG_FIXTURE_ROW_INDEX] == 0
        assert tag[MAIN_STEP1_TAG_ROLLOUT_INDEX] == index
        assert tag[MAIN_STEP1_TAG_GENERATION_SEED] == seeds[index]
        assert tag[MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256] == "a" * 64
        for flag_name, tag_name in MAIN_STEP1_REWARD_PENALTY_TAGS.items():
            assert tag[tag_name] is completions[index].reward_penalty_flags[flag_name]


def _strict_transcript_completion(index: int) -> tuple[Completion, int]:
    seed = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=index
    )
    request = {
        "input": [{"role": "user", "content": "question"}],
        "max_output_tokens": 768,
        "temperature": 1.0,
        "top_p": 1.0,
        "metadata": {"extra_body": json.dumps({"seed": seed}, separators=(",", ":"))},
    }
    response = {
        "background": None,
        "conversation": None,
        "created_at": 1.0,
        "error": None,
        "id": f"resp_{index:012x}40008000{index:012x}",
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": deepcopy(request["metadata"]),
        "model": "/immutable/model",
        "object": "response",
        "output": [
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": None,
                        "text": str(index),
                        "type": "output_text",
                    }
                ],
                "id": f"msg_{index:012x}40008000{index:012x}",
                "prompt_token_ids": [10, 11],
                "generation_token_ids": [100 + index, 200 + index],
                "generation_log_probs": [-0.25, -0.5],
                "role": "assistant",
                "routed_experts": None,
                "status": "completed",
                "type": "message",
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
    agent_run_request = {
        **_FIXTURE_ROW,
        "_ng_task_index": _STRICT_TASK_INDEX,
        "_rowidx": index,
        "_ng_rollout_index": index,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "reasoning_gym_simple_agent",
        },
        "responses_create_params": request,
    }
    expanded_responses_create_params = {
        "background": None,
        "include": None,
        "input": [
            {
                **deepcopy(request["input"][0]),
                "type": "message",
            }
        ],
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": deepcopy(request["metadata"]),
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
    derived_verifier_request = {
        **deepcopy(agent_run_request),
        "responses_create_params": deepcopy(expanded_responses_create_params),
        "response": deepcopy(response),
    }
    reward = float(index % 2)
    return (
        Completion(
            message_log=[],
            env_extras=None,
            truncated=False,
            reward=reward,
            raw_environment_reward=reward,
            pre_penalty_reward=reward,
            reward_penalty_flags=_flags(),
            strict_transcript={
                "generation_request": request,
                "model_response": response,
                "agent_run_request": agent_run_request,
                "derived_verifier_request": derived_verifier_request,
                "verifier_response": {
                    "responses_create_params": expanded_responses_create_params,
                    "response": response,
                    "reward": reward,
                    "task_name": "basic_arithmetic",
                    "score": reward,
                    "extracted_answer": str(index),
                },
                "verifier_request_derivation_runtime": {
                    "openai_version": "2.6.1",
                    "pydantic_version": "2.13.4",
                },
            },
        ),
        seed,
    )


def _strict_runtime_contract() -> dict:
    return {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "off",
        "mode": "observe",
        "results_dir": "/results/off",
        "fixture_path": "/fixtures/reasoning.jsonl",
        "model_transport_policy_sha256": "a" * 64,
        "gym_gitlink_commit": "1" * 40,
        "gym_tree": "2" * 40,
        "bindings": {
            "pair_manifest_sha256": "1" * 64,
            "submission_receipt_sha256": "2" * 64,
            "job_id": "123",
            "run_id": "3" * 64,
            "restart_count": 0,
            "fixture_sha256": "4" * 64,
            "verifier_source_sha256": "5" * 64,
            "snapshot_manifest_sha256": "6" * 64,
            "config_sha256": "7" * 64,
            "pair_campaign_sha256": "8" * 64,
            "pair_campaign_reward_and_advantage_sha256": "9" * 64,
        },
    }


def _transport_evidence() -> tuple[dict, dict, dict, dict]:
    entries = [
        {
            "entry_sha256": f"{index + 1:x}" * 64,
            "request_body_sha256": "a" * 64,
            "response_body_sha256": "b" * 64,
        }
        for index in range(4)
    ]
    capture = {"record_count": 4}
    reference = {
        "path": "/results/off/strict_model_transport/model-transport-bundle.json",
        "schema": "nemo-rl-strict-model-transport-bundle-v1",
        "sha256": "c" * 64,
    }
    return capture, reference, {"entries": entries}, {"policy_sha256": "d" * 64}


@pytest.mark.asyncio
async def test_rollout_finalizer_calls_the_single_model_owner_with_exact_transcripts(
    monkeypatch,
) -> None:
    completions = [_strict_transcript_completion(index)[0] for index in range(4)]
    capture, reference, bundle, policy = _transport_evidence()
    future = asyncio.get_running_loop().create_future()
    future.set_result((capture, reference, bundle))

    class WorkerGroup:
        dp_size = 1

        def get_dp_leader_worker_idx(self, dp_index):
            assert dp_index == 0
            return 7

        def run_single_worker_single_data(self, **kwargs):
            self.call = kwargs
            return future

    worker_group = WorkerGroup()
    rollout = object.__new__(AsyncNemoGymRolloutImpl)
    rollout._policy_generation = SimpleNamespace(worker_group=worker_group)
    rollout._generation_config = {"model_name": "/immutable/model"}
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.main_step1_runtime_contract",
        _strict_runtime_contract,
    )
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.load_runtime_model_transport_policy",
        lambda **kwargs: policy,
    )

    result = await rollout._attest_strict_model_transport_step1(completions)

    assert result == (capture, reference, bundle, policy, "/immutable/model")
    assert worker_group.call["method_name"] == "attest_strict_model_transport_step1"
    assert worker_group.call["worker_idx"] == 7
    assert (
        worker_group.call["expected_generation_requests"][2]
        == (completions[2].strict_transcript["generation_request"])
    )
    assert (
        worker_group.call["expected_model_responses"][1]
        == (completions[1].strict_transcript["model_response"])
    )


def test_rollout_publishes_exact_main_transcript_and_returns_digest(
    monkeypatch,
) -> None:
    contract = _strict_runtime_contract()
    completions_and_seeds = [_strict_transcript_completion(i) for i in range(4)]
    completions = [item[0] for item in completions_and_seeds]
    seeds = [item[1] for item in completions_and_seeds]
    published = {}
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.main_step1_runtime_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.load_strict_fixture_row0",
        lambda **kwargs: _FIXTURE_ROW,
    )

    def publish(*, results_dir, document):
        published["results_dir"] = results_dir
        published["document"] = document
        return (
            "/results/off/strict_pair_step1_evidence/transcript-bundle.json",
            "a" * 64,
        )

    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.publish_main_transcript_bundle",
        publish,
    )
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.validate_transcript_model_transport_join",
        lambda **kwargs: None,
    )

    def build(**kwargs):
        published["build_kwargs"] = kwargs
        return {
            "attempt_id": kwargs["attempt_id"],
            "entries": kwargs["entry_inputs"],
        }

    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.build_transcript_bundle", build
    )
    capture, transport_ref, transport_bundle, transport_policy = _transport_evidence()

    digest = _publish_strict_main_step1_transcript(
        input_sample={"idx": 0, "extra_env_info": _FIXTURE_ROW},
        completions=completions,
        request_seeds=seeds,
        generation_config={
            "nemo_gym_per_rollout_seed_base": 42,
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
        },
        rollout_group_id=_STRICT_GROUP_ID,
        model_transport_capture=capture,
        model_transport_bundle_ref=transport_ref,
        model_transport_bundle=transport_bundle,
        model_transport_policy=transport_policy,
        model_path="/immutable/model",
    )

    assert digest == "a" * 64
    assert published["results_dir"] == "/results/off"
    assert published["document"]["attempt_id"] is None
    assert published["document"]["entries"][3]["generation_seed"] == seeds[3]
    assert published["build_kwargs"]["fixture_row"] == _FIXTURE_ROW
    assert published["build_kwargs"]["model_transport_bundle"] == transport_ref
    assert published["build_kwargs"]["verifier_request_derivation"] == {
        "schema": "nemo-rl-strict-derived-verifier-request-v1",
        "assurance": "deterministic-reconstruction-not-wire-capture",
        "algorithm": "pinned-simple-agent-model-dump-v1",
        "gym_gitlink_commit": "1" * 40,
        "gym_tree": "2" * 40,
        "runtime": {
            "openai_version": "2.6.1",
            "pydantic_version": "2.13.4",
        },
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
    assert (
        published["document"]["entries"][2]["model_transport_entry_sha256"]
        == transport_bundle["entries"][2]["entry_sha256"]
    )


def test_rollout_rejects_reward_injected_into_raw_model_response(monkeypatch) -> None:
    completions_and_seeds = [_strict_transcript_completion(i) for i in range(4)]
    completions = [item[0] for item in completions_and_seeds]
    seeds = [item[1] for item in completions_and_seeds]
    assert completions[0].strict_transcript is not None
    completions[0].strict_transcript["model_response"]["reward"] = 1.0
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.main_step1_runtime_contract",
        _strict_runtime_contract,
    )
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.load_strict_fixture_row0",
        lambda **kwargs: _FIXTURE_ROW,
    )
    capture, transport_ref, transport_bundle, transport_policy = _transport_evidence()

    with pytest.raises(RuntimeError, match="only raw model output"):
        _publish_strict_main_step1_transcript(
            input_sample={"idx": 0, "extra_env_info": _FIXTURE_ROW},
            completions=completions,
            request_seeds=seeds,
            generation_config={
                "nemo_gym_per_rollout_seed_base": 42,
                "max_new_tokens": 768,
                "temperature": 1.0,
                "top_k": None,
                "top_p": 1.0,
            },
            rollout_group_id=_STRICT_GROUP_ID,
            model_transport_capture=capture,
            model_transport_bundle_ref=transport_ref,
            model_transport_bundle=transport_bundle,
            model_transport_policy=transport_policy,
            model_path="/immutable/model",
        )
