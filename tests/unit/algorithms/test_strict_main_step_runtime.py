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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.strict_main_step_runtime import StrictMainStep1Recorder
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    MALFORMED_THINKING_MESSAGE_COUNT,
    PRE_PENALTY_REWARD,
    RAW_ENVIRONMENT_REWARD,
)
from nemo_rl.utils.strict_captured_replay_evidence import (
    build_verifier_request_derivation,
)
from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_REWARD_PENALTY_TAGS,
    MAIN_STEP1_TAG_FIXTURE_ROW_INDEX,
    MAIN_STEP1_TAG_GENERATION_SEED,
    MAIN_STEP1_TAG_ROLLOUT_INDEX,
    MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256,
    derive_nemo_gym_request_seed,
)
from tests.unit.experience.test_strict_step1_runtime_plumbing import (
    _strict_transcript_completion,
)


def test_recorder_captures_exact_final_unpadded_k4_trainer_rows(monkeypatch) -> None:
    recorder = object.__new__(StrictMainStep1Recorder)
    recorder.enabled = True
    recorder._row_inputs = []
    recorder._published = False
    recorder._contract = {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "train",
        "results_dir": "/results",
        "bindings": {},
    }
    recorder._generation = {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }
    recorder._transcript_bundle_ref = None
    recorder._transcript_bundle_document = {}
    transcript_ref = {
        "path": "/results/strict_pair_step1_evidence/transcript-bundle.json",
        "schema": "nemo-rl-strict-step1-transcript-bundle-v4",
        "sha256": "a" * 64,
    }

    def bind_rows(*, rows, expected_sha256):
        assert expected_sha256 == "a" * 64
        return (
            [
                {
                    **row,
                    "request_sha256": "1" * 64,
                    "response_sha256": "2" * 64,
                    "agent_run_request_sha256": "3" * 64,
                    "derived_verifier_request_sha256": "4" * 64,
                    "verifier_response_sha256": "5" * 64,
                }
                for row in rows
            ],
            transcript_ref,
        )

    recorder._bind_transcript_rows = bind_rows
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.build_main_step1_ledger",
        lambda **kwargs: pytest.fail(
            "ledger must not be built before optimizer success"
        ),
    )

    group_id = "12345678-1234-4678-9234-567812345678"
    input_ids = torch.tensor(
        [[10, 11, 100 + row, 200 + row, 0] for row in range(4)],
        dtype=torch.long,
    )
    input_lengths = torch.tensor([4, 4, 4, 4], dtype=torch.long)
    prompt_ids = torch.tensor([[10, 11] for _ in range(4)], dtype=torch.long)
    token_mask = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0, 0.0] for _ in range(4)], dtype=torch.float32
    )
    advantages = torch.tensor(
        [[0.0, 0.0, float(row), float(row), 0.0] for row in range(4)],
        dtype=torch.float32,
    )
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "prompt_ids_for_adv": prompt_ids,
            "token_mask": token_mask,
            "sample_mask": torch.ones(4, dtype=torch.float32),
            RAW_ENVIRONMENT_REWARD: rewards,
            PRE_PENALTY_REWARD: rewards,
            SHARED_PREFIX_GROUP_ID: np.asarray([group_id] * 4, dtype=object),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([2, 2, 2, 2], dtype=torch.long),
            GENERATED_ASSISTANT_MESSAGE_COUNT: torch.ones(4, dtype=torch.long),
            INVALID_TOOL_CALL_MESSAGE_COUNT: torch.zeros(4, dtype=torch.long),
            MALFORMED_THINKING_MESSAGE_COUNT: torch.zeros(4, dtype=torch.long),
            INVALID_AND_MALFORMED_MESSAGE_COUNT: torch.zeros(4, dtype=torch.long),
        },
        batch_size=[4],
    )
    tags = []
    for row in range(4):
        tags.append(
            {
                MAIN_STEP1_TAG_FIXTURE_ROW_INDEX: 0,
                MAIN_STEP1_TAG_ROLLOUT_INDEX: row,
                MAIN_STEP1_TAG_GENERATION_SEED: derive_nemo_gym_request_seed(
                    seed_base=42, fixture_row_index=0, rollout_index=row
                ),
                MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256: "a" * 64,
                **{
                    tag_name: False
                    for tag_name in MAIN_STEP1_REWARD_PENALTY_TAGS.values()
                },
            }
        )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=[f"{group_id}_g{row}" for row in range(4)],
        tags=tags,
    )

    recorder.capture_consumed_rows(
        step_index=0,
        meta=meta,
        data=data,
        prompt_ids=prompt_ids,
        verifier_rewards=rewards,
        processed_rewards=rewards,
        token_loss_mask=token_mask,
        sample_mask=torch.ones(4, dtype=torch.float32),
        advantages=advantages,
        invalid_advantage_enabled=True,
        malformed_advantage_enabled=False,
    )

    assert len(recorder._row_inputs) == 4
    assert recorder._row_inputs[3]["token_ids"] == [10, 11, 103, 203]
    assert recorder._row_inputs[3]["prompt_token_ids"] == [10, 11]
    assert recorder._row_inputs[3]["completion_token_ids"] == [103, 203]
    assert recorder._row_inputs[3]["advantages"] == [0.0, 0.0, 3.0, 3.0]
    assert recorder._row_inputs[3]["valid_loss_tokens"] == 2
    assert not any(recorder._row_inputs[3]["penalty_flags"].values())
    assert recorder._row_inputs[3]["response_sha256"] == "2" * 64
    assert recorder._transcript_bundle_ref == transcript_ref


def test_recorder_stable_loads_and_joins_raw_transcript_before_ledger(
    monkeypatch, tmp_path
) -> None:
    generation = {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }
    transcript_bindings: dict[str, object] = {
        "pair_manifest_sha256": "1" * 64,
        "submission_receipt_sha256": "2" * 64,
        "job_id": "123",
        "run_id": "3" * 64,
        "fixture_sha256": "4" * 64,
        "verifier_source_sha256": "5" * 64,
        "config_sha256": "6" * 64,
        "snapshot_manifest_sha256": "7" * 64,
    }
    main_bindings = {
        **transcript_bindings,
        "restart_count": 0,
        "pair_campaign_sha256": "8" * 64,
        "pair_campaign_reward_and_advantage_sha256": "9" * 64,
    }
    entries = []
    rows = []
    for index in range(4):
        completion, seed = _strict_transcript_completion(index)
        assert completion.strict_transcript is not None
        request = completion.strict_transcript["generation_request"]
        response = completion.strict_transcript["model_response"]
        agent_run_request = completion.strict_transcript["agent_run_request"]
        derived_verifier_request = completion.strict_transcript[
            "derived_verifier_request"
        ]
        verifier_response = completion.strict_transcript["verifier_response"]
        reward = completion.raw_environment_reward
        entries.append(
            {
                "sample_index": index,
                "fixture_row_index": 0,
                "rollout_index": index,
                "generation_seed": seed,
                "generation_request": request,
                "model_response": response,
                "agent_run_request": agent_run_request,
                "derived_verifier_request": derived_verifier_request,
                "verifier_response": verifier_response,
                "raw_environment_reward": reward,
                "generation_request_sha256": "1" * 64,
                "model_response_sha256": "2" * 64,
                "agent_run_request_sha256": "3" * 64,
                "derived_verifier_request_sha256": "4" * 64,
                "verifier_response_sha256": "5" * 64,
            }
        )
        group_id = "12345678-1234-4678-9234-567812345678"
        rows.append(
            {
                "sample_index": index,
                "sample_id": f"{group_id}_g{index}",
                "shared_prefix_group_id": group_id,
                "fixture_row_index": 0,
                "rollout_index": index,
                "generation_seed": seed,
                "token_ids": [10, 11, 100 + index, 200 + index],
                "input_length": 4,
                "prompt_token_ids": [10, 11],
                "completion_token_ids": [100 + index, 200 + index],
                "token_loss_mask": [0.0, 0.0, 1.0, 1.0],
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
                "advantages": [0.0, 0.0, 1.0, 1.0],
                "valid_loss_tokens": 2,
                "total_tokens": 4,
            }
        )

    transport_ref = {
        "path": str(
            tmp_path / "strict_model_transport" / "model-transport-bundle.json"
        ),
        "schema": "nemo-rl-strict-model-transport-bundle-v1",
        "sha256": "a" * 64,
    }
    bundle = {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "train",
        "attempt_id": None,
        "step": 1,
        "sample_count": 4,
        "generation": generation,
        "bindings": transcript_bindings,
        "verifier_request_derivation": build_verifier_request_derivation(
            gym_gitlink_commit="1" * 40,
            gym_tree="2" * 40,
            openai_version="2.6.1",
            pydantic_version="2.13.4",
        ),
        "model_transport_bundle": transport_ref,
        "entries": entries,
    }
    digest = "b" * 64
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.load_evidence_document",
        lambda **kwargs: (bundle, digest),
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.validate_transcript_bundle",
        lambda document: None,
    )
    raw_capture_ref = {
        "path": str(tmp_path / "strict_model_transport" / "model-transport.jsonl"),
        "record_count": 4,
        "record_schema": "nemo-rl-strict-model-transport-call-v1",
        "sha256": "c" * 64,
    }
    transport_document = {
        "capture_server": {"server_instance_id": "d" * 64},
        "entry_count": 4,
        "ordered_entries_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.load_finalized_model_transport_capture",
        lambda **kwargs: (
            {"policy_sha256": "f" * 64},
            transport_document,
            raw_capture_ref,
            transport_ref,
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.validate_transcript_model_transport_join",
        lambda **kwargs: None,
    )
    recorder = object.__new__(StrictMainStep1Recorder)
    recorder._contract = {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "train",
        "results_dir": str(tmp_path),
        "model_transport_policy_sha256": "f" * 64,
        "gym_gitlink_commit": "1" * 40,
        "gym_tree": "2" * 40,
        "bindings": main_bindings,
    }
    recorder._generation = generation
    recorder._model_path = "/immutable/model"

    native_partial_reward = 0.6499999999999999
    trainer_float32_reward = 0.6499999761581421
    entries[0]["raw_environment_reward"] = native_partial_reward
    rows[0]["raw_environment_reward"] = trainer_float32_reward

    bound_rows, reference = recorder._bind_transcript_rows(
        rows=rows, expected_sha256=digest
    )

    assert reference["sha256"] == digest
    assert (
        bound_rows[2]["response_sha256"]
        == bundle["entries"][2]["model_response_sha256"]
    )
    assert (
        bound_rows[2]["verifier_response_sha256"]
        == bundle["entries"][2]["verifier_response_sha256"]
    )
    assert recorder._model_transport_bundle_ref == transport_ref
    assert recorder._model_transport_capture_ref == raw_capture_ref

    wrong_precision_rows = [dict(row) for row in rows]
    wrong_precision_rows[0]["raw_environment_reward"] = native_partial_reward
    with pytest.raises(RuntimeError, match="round to the consumed float32"):
        recorder._bind_transcript_rows(
            rows=wrong_precision_rows, expected_sha256=digest
        )


def test_publish_requires_applied_update_then_writes_terminal_manifest(
    monkeypatch, tmp_path
) -> None:
    recorder = object.__new__(StrictMainStep1Recorder)
    recorder.enabled = True
    recorder._published = False
    recorder._row_inputs = [{}, {}, {}, {}]
    recorder._contract = {
        "pair_id": "pair-1",
        "environment": "reasoning_gym",
        "arm": "on",
        "mode": "train",
        "results_dir": str(tmp_path),
        "model_transport_policy_sha256": "a" * 64,
        "bindings": {
            "pair_manifest_sha256": "1" * 64,
            "submission_receipt_sha256": "2" * 64,
            "job_id": "123",
        },
    }
    recorder._generation = {"seed_base": 42}
    recorder._transcript_bundle_ref = {
        "path": str(tmp_path / "strict_pair_step1_evidence/transcript-bundle.json"),
        "schema": "nemo-rl-strict-step1-transcript-bundle-v4",
        "sha256": "3" * 64,
    }
    recorder._transcript_bundle_document = {"entries": [{}, {}, {}, {}]}
    recorder._model_transport_policy = {"policy_sha256": "a" * 64}
    recorder._model_transport_bundle_ref = {
        "path": str(tmp_path / "strict_model_transport/model-transport-bundle.json"),
        "schema": "nemo-rl-strict-model-transport-bundle-v1",
        "sha256": "4" * 64,
    }
    recorder._model_transport_bundle_document = {
        "capture_server": {"server_instance_id": "5" * 64},
        "entry_count": 4,
        "ordered_entries_sha256": "6" * 64,
    }
    recorder._model_transport_capture_ref = {
        "path": str(tmp_path / "strict_model_transport/model-transport.jsonl"),
        "record_count": 4,
        "record_schema": "nemo-rl-strict-model-transport-call-v1",
        "sha256": "7" * 64,
    }
    recorder._model_path = "/immutable/model"
    calls: dict[str, object] = {}
    ledger_path = tmp_path / "strict_pair_step1_evidence/main-ledger.json"

    def build_ledger(**kwargs):
        calls["ledger_inputs"] = kwargs
        return {
            "schema": "nemo-rl-strict-main-step1-ledger-v5",
            "update_successful": kwargs["update_successful"],
        }

    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.build_main_step1_ledger",
        build_ledger,
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.validate_ledger_transcript_join",
        lambda **kwargs: calls.setdefault("ledger_join", kwargs),
    )

    def publish_ledger(**kwargs):
        calls["published_ledger"] = kwargs
        return ledger_path, "8" * 64

    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.publish_main_step1_ledger",
        publish_ledger,
    )

    def build_manifest(**kwargs):
        calls["manifest_inputs"] = kwargs
        return {"schema": "nemo-rl-strict-model-transport-manifest-v1"}

    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.build_model_transport_manifest",
        build_manifest,
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.strict_main_step_runtime.publish_model_transport_manifest",
        lambda **kwargs: calls.setdefault("published_manifest", kwargs),
    )

    with pytest.raises(RuntimeError, match="successful optimizer update"):
        recorder.publish_after_successful_step(step_index=0, update_successful=False)
    assert calls == {}

    result = recorder.publish_after_successful_step(
        step_index=0, update_successful=True
    )
    assert result == (ledger_path, "8" * 64)
    assert calls["ledger_inputs"]["update_successful"] is True
    assert calls["published_ledger"]["document"]["update_successful"] is True
    assert calls["manifest_inputs"]["authenticated_job_id"] == "123"
    assert calls["manifest_inputs"]["main_ledger"]["sha256"] == "8" * 64
    assert calls["published_manifest"]["transport_directory"] == (
        tmp_path / "strict_model_transport"
    )
