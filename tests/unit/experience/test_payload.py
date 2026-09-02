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

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane.codec import materialize
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    INVALID_TOOL_CALL_TOKEN_MASK,
    MALFORMED_THINKING_MESSAGE_COUNT,
    MALFORMED_THINKING_TOKEN_MASK,
    RESPONSE_TOKEN_LENGTHS,
    ROLLOUT_TRUNCATED,
)
from nemo_rl.experience.payload import pack_payload, record_to_train_batch


def _routes(start: int, count: int) -> torch.Tensor:
    token_routes = torch.arange(start, start + count, dtype=torch.int16).view(
        count, 1, 1
    )
    topk_offsets = torch.arange(2, dtype=torch.int16).view(1, 1, 2)
    return (token_routes + topk_offsets).expand(count, 2, 2).contiguous()


def _fallback_routes(count: int) -> torch.Tensor:
    return torch.arange(2, dtype=torch.int16).view(1, 1, 2).expand(count, 2, 2)


def _completion(
    route_start: int,
    reward: float,
    *,
    env_token_ids: tuple[int, ...] = (30,),
    with_routes: bool = True,
) -> Completion:
    message_log = [
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
            "routed_experts": _routes(route_start, 2),
        },
        {
            "role": "assistant",
            "content": "answer",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
            "routed_experts": _routes(route_start + 2, 2),
        },
        {
            "role": "user",
            "content": "environment",
            "token_ids": torch.tensor(env_token_ids),
            "routed_experts": _fallback_routes(len(env_token_ids)),
        },
    ]
    if not with_routes:
        for message in message_log:
            message.pop("routed_experts")
    return Completion(
        message_log=message_log,
        env_extras=None,
        truncated=False,
        reward=reward,
    )


def _record(completions: list[Completion]) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        extra_env_info=None,
        metadata={"task_name": "test", "loss_multiplier": 1.0},
        completions=completions,
        rollout_metrics={},
    )


def _assistant_history_record(
    *, history_has_logprobs: bool = False
) -> PromptGroupRecord:
    prompt_messages = [
        {
            "role": "user",
            "content": "first question",
            "token_ids": torch.tensor([10]),
            "routed_experts": _routes(10, 1),
        },
        {
            "role": "assistant",
            "content": "answer in prompt history",
            "token_ids": torch.tensor([11, 12, 13]),
            "routed_experts": _routes(11, 3),
        },
        {
            "role": "user",
            "content": "follow-up question",
            "token_ids": torch.tensor([14, 15]),
            "routed_experts": _routes(14, 2),
        },
    ]
    if history_has_logprobs:
        # Reconstructed message logs may carry synthetic zero logprobs on prompt
        # assistants; the prompt boundary must still keep this turn loss-masked.
        prompt_messages[1]["generation_logprobs"] = torch.zeros(3)
    generated = {
        "role": "assistant",
        "content": "new response",
        "token_ids": torch.tensor([20, 21]),
        "generation_logprobs": torch.tensor([-0.1, -0.2]),
        "routed_experts": _routes(20, 2),
        "is_invalid_tool_call": True,
    }
    completion = Completion(
        message_log=[*prompt_messages, generated],
        env_extras=None,
        truncated=False,
        reward=1.0,
    )
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[dict(message) for message in prompt_messages],
        extra_env_info=None,
        metadata={"task_name": "test", "loss_multiplier": 1.0},
        completions=[completion],
        rollout_metrics={},
    )


def test_record_to_train_batch_preserves_routed_experts_in_tq_payload() -> None:
    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(
                route_start=30,
                reward=2.0,
                env_token_ids=(30, 31),
            ),
        ]
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    expected_routes = [
        torch.cat((_routes(10, 4), _fallback_routes(1))),
        torch.cat((_routes(30, 4), _fallback_routes(2))),
    ]
    assert train_batch["input_lengths"].tolist() == [5, 6]
    assert train_batch["routed_experts"].shape == (2, 6, 2, 2)
    assert torch.equal(
        train_batch["routed_experts"][0, :5],
        expected_routes[0],
    )
    assert torch.equal(
        train_batch["routed_experts"][1],
        expected_routes[1],
    )

    sample_ids, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert sample_ids == ["group_g0", "group_g1"]
    assert "routed_experts" in fields
    packed_routes = fields["routed_experts"]
    assert packed_routes.is_nested
    packed_rows = list(packed_routes.unbind())
    assert torch.equal(packed_rows[0], expected_routes[0])
    assert torch.equal(packed_rows[1], expected_routes[1])
    assert tags == [{"weight_version": 3}, {"weight_version": 3}]


def test_record_to_train_batch_omits_routed_experts_when_absent() -> None:
    record = _record([_completion(route_start=10, reward=1.0, with_routes=False)])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )
    assert "routed_experts" not in train_batch

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert "routed_experts" not in fields


def test_reward_processing_payload_preserves_masks_counts_and_loss_gate() -> None:
    completion = _completion(route_start=10, reward=1.0)
    assistant = completion.message_log[1]
    assistant["is_invalid_tool_call"] = True
    assistant["has_malformed_thinking"] = True
    completion.env_extras = {"instance_config": {"mask_sample": True}}
    completion.env_masked = True
    completion.truncated = True
    record = _record([completion])
    record.metadata["loss_multiplier"] = 0.5

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_reward_processing_metadata=True,
        include_message_advantage_metadata=True,
    )

    assert train_batch["sample_mask"].tolist() == [0.0]
    assert train_batch[ROLLOUT_TRUNCATED].tolist() == [True]
    assert train_batch[RESPONSE_TOKEN_LENGTHS].tolist() == [2]
    assert train_batch[GENERATED_ASSISTANT_MESSAGE_COUNT].tolist() == [1]
    assert train_batch[INVALID_TOOL_CALL_MESSAGE_COUNT].tolist() == [1]
    assert train_batch[MALFORMED_THINKING_MESSAGE_COUNT].tolist() == [1]
    assert train_batch[INVALID_AND_MALFORMED_MESSAGE_COUNT].tolist() == [1]
    assert train_batch[INVALID_TOOL_CALL_TOKEN_MASK][0, :5].tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert train_batch[MALFORMED_THINKING_TOKEN_MASK][0, :5].tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert fields[INVALID_TOOL_CALL_TOKEN_MASK].is_nested
    assert fields[MALFORMED_THINKING_TOKEN_MASK].is_nested


def test_response_length_and_masks_skip_assistant_prompt_history() -> None:
    record = _assistant_history_record(history_has_logprobs=True)

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_reward_processing_metadata=True,
        include_message_advantage_metadata=True,
    )

    assert train_batch[RESPONSE_TOKEN_LENGTHS].tolist() == [2]
    assert train_batch[GENERATED_ASSISTANT_MESSAGE_COUNT].tolist() == [1]
    assert train_batch[INVALID_TOOL_CALL_MESSAGE_COUNT].tolist() == [1]
    assert train_batch["token_mask"][0, :8].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    assert train_batch[INVALID_TOOL_CALL_TOKEN_MASK][0, :8].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_record_to_train_batch_is_idempotent_and_does_not_mutate_record() -> None:
    record = _assistant_history_record()

    kwargs = {
        "pad_value_dict": {"token_ids": 0, "input_ids": 0},
        "include_reward_processing_metadata": True,
        "include_message_advantage_metadata": True,
    }
    first = record_to_train_batch(record, **kwargs)
    second = record_to_train_batch(record, **kwargs)

    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert all(
        "token_loss_mask" not in message
        for message in record.completions[0].message_log
    )
    assert "generation_logprobs" not in record.completions[0].message_log[1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_invalid_tool_call", 1),
        ("has_malformed_thinking", "false"),
        ("is_invalid_tool_call", torch.tensor(True)),
    ],
)
def test_message_detector_flags_require_bool(field: str, value: object) -> None:
    completion = _completion(route_start=10, reward=1.0)
    completion.message_log[1][field] = value

    with pytest.raises(TypeError, match=rf"message\.{field} must be a bool"):
        record_to_train_batch(
            _record([completion]),
            pad_value_dict={"token_ids": 0, "input_ids": 0},
            include_message_advantage_metadata=True,
        )


@pytest.mark.parametrize("value", [1, "false", torch.tensor(False)])
def test_completion_env_masked_requires_bool(value: object) -> None:
    completion = _completion(route_start=10, reward=1.0)
    completion.env_masked = value  # type: ignore[assignment]

    with pytest.raises(TypeError, match=r"Completion\.env_masked must be a bool"):
        record_to_train_batch(
            _record([completion]),
            pad_value_dict={"token_ids": 0, "input_ids": 0},
        )


def test_native_env_extras_cannot_implicitly_mask_sample() -> None:
    completion = _completion(route_start=10, reward=1.0)
    completion.env_extras = {"instance_config": {"mask_sample": True}}

    train_batch = record_to_train_batch(
        _record([completion]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["sample_mask"].tolist() == [1.0]


def test_record_to_train_batch_requires_loss_multiplier() -> None:
    record = _record([_completion(route_start=10, reward=1.0)])
    record.metadata.pop("loss_multiplier")

    with pytest.raises(ValueError, match="requires loss_multiplier"):
        record_to_train_batch(
            record,
            pad_value_dict={"token_ids": 0, "input_ids": 0},
        )


def test_record_to_train_batch_preserves_base_loss_multiplier() -> None:
    record = _record([_completion(route_start=10, reward=1.0)])
    record.metadata["loss_multiplier"] = 0.25

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["sample_mask"].tolist() == [0.25]


def test_shared_prefix_payload_metadata_is_explicitly_opt_in() -> None:
    # Keep the optional TransferQueue runtime out of payload-test collection.
    from nemo_rl.data_plane.adapters.transfer_queue import (
        _from_wire,
        _promote_1d_leaves,
    )

    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(route_start=30, reward=2.0),
        ]
    )

    disabled_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )
    assert set(disabled_batch) == {
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
        "prompt_ids_for_adv",
        "total_reward",
        "routed_experts",
    }
    _, disabled_fields, _ = pack_payload(
        disabled_batch,
        weight_version=3,
        group_id="group",
    )
    assert set(disabled_fields.keys()) == set(disabled_batch)

    observed_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_shared_prefix_metadata=True,
    )
    assert observed_batch[SHARED_PREFIX_PROMPT_LENGTHS].tolist() == [2, 2]
    _, observed_fields, _ = pack_payload(
        observed_batch,
        weight_version=3,
        group_id="group",
        include_shared_prefix_metadata=True,
    )
    wire_fields = _promote_1d_leaves(observed_fields)
    assert wire_fields[SHARED_PREFIX_PROMPT_LENGTHS].shape == (2, 1)
    restored_fields = _from_wire(wire_fields)
    assert restored_fields[SHARED_PREFIX_PROMPT_LENGTHS].shape == (2,)

    materialized = materialize(restored_fields)
    group_ids = materialized[SHARED_PREFIX_GROUP_ID]
    assert isinstance(group_ids, np.ndarray)
    assert group_ids.dtype == object
    assert group_ids.tolist() == ["group", "group"]
    assert materialized[SHARED_PREFIX_PROMPT_LENGTHS].shape == (2,)
    assert materialized[SHARED_PREFIX_PROMPT_LENGTHS].tolist() == [2, 2]


def test_shared_prefix_payload_rejects_missing_prompt_lengths() -> None:
    train_batch = record_to_train_batch(
        _record([_completion(route_start=10, reward=1.0)]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    with pytest.raises(ValueError, match="requires prompt lengths"):
        pack_payload(
            train_batch,
            weight_version=3,
            group_id="group",
            include_shared_prefix_metadata=True,
        )


@pytest.mark.parametrize(
    ("prompt_lengths", "match"),
    [
        (torch.tensor([[2], [2]]), r"shape \(2,\)"),
        (torch.tensor([2, 6]), "no larger than input_lengths"),
    ],
)
def test_shared_prefix_payload_rejects_invalid_prompt_lengths(
    prompt_lengths: torch.Tensor,
    match: str,
) -> None:
    train_batch = record_to_train_batch(
        _record(
            [
                _completion(route_start=10, reward=1.0),
                _completion(route_start=30, reward=2.0),
            ]
        ),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
        include_shared_prefix_metadata=True,
    )
    train_batch[SHARED_PREFIX_PROMPT_LENGTHS] = prompt_lengths

    with pytest.raises(ValueError, match=match):
        pack_payload(
            train_batch,
            weight_version=3,
            group_id="group",
            include_shared_prefix_metadata=True,
        )


def _failed_completion() -> Completion:
    """A trajectory whose first generation raised: prompt only, no routes."""
    return Completion(
        message_log=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        env_extras=None,
        truncated=False,
        reward=0.0,
    )


def test_record_to_train_batch_backfills_routes_for_failed_completion() -> None:
    """A group is packable when only some completions generated (and so have routes)."""
    record = _record([_completion(route_start=10, reward=1.0), _failed_completion()])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["input_lengths"].tolist() == [5, 2]
    routes = train_batch["routed_experts"]
    assert routes.shape == (2, 5, 2, 2)
    assert torch.equal(routes[0, :5], torch.cat((_routes(10, 4), _fallback_routes(1))))
    # The completion that never generated gets the all--1 missing-route sentinel,
    # so Megatron routes those tokens with its own router.
    assert torch.equal(routes[1, :2], torch.full((2, 2, 2), -1, dtype=routes.dtype))
    # It is fully loss-masked either way.
    assert train_batch["token_mask"][1, :2].tolist() == [0, 0]

    _, fields, _ = pack_payload(train_batch, weight_version=3, group_id="group")
    assert "routed_experts" in fields
    assert list(fields["routed_experts"].unbind())[1].shape == (2, 2, 2)
