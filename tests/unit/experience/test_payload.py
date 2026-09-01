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
        metadata={"task_name": "test"},
        completions=completions,
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
        "truncated",
        "response_token_lengths",
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
    assert wire_fields["response_token_lengths"].shape == (2, 1)
    assert wire_fields["truncated"].shape == (2, 1)
    restored_fields = _from_wire(wire_fields)
    assert restored_fields[SHARED_PREFIX_PROMPT_LENGTHS].shape == (2,)
    assert restored_fields["response_token_lengths"].shape == (2,)
    assert restored_fields["response_token_lengths"].tolist() == [2, 2]
    assert restored_fields["truncated"].shape == (2,)
    assert restored_fields["truncated"].tolist() == [False, False]

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


def test_record_to_train_batch_preserves_reward_shaping_metadata() -> None:
    first = _completion(route_start=10, reward=1.0)
    first.truncated = True
    first.message_log.append(
        {
            "role": "assistant",
            "content": "later assistant turn",
            "token_ids": torch.tensor([40, 41, 42]),
            "generation_logprobs": torch.tensor([-0.3, -0.4, -0.5]),
            "routed_experts": _fallback_routes(3),
        }
    )
    record = _record([first, _failed_completion()])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["truncated"].dtype is torch.bool
    assert train_batch["truncated"].tolist() == [True, False]
    # Match decompose_message_log: use the first assistant response, and zero when
    # generation failed before producing any assistant turn.
    assert train_batch["response_token_lengths"].tolist() == [2, 0]

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert "truncated" in fields
    assert "response_token_lengths" in fields


def test_response_length_keeps_zero_length_first_assistant() -> None:
    completion = _completion(route_start=10, reward=1.0)
    first_assistant = completion.message_log[1]
    first_assistant["token_ids"] = torch.empty(0, dtype=torch.long)
    first_assistant["generation_logprobs"] = torch.empty(0)
    first_assistant["routed_experts"] = _routes(12, 0)
    completion.message_log.append(
        {
            "role": "assistant",
            "content": "later nonempty assistant",
            "token_ids": torch.tensor([40, 41, 42]),
            "generation_logprobs": torch.tensor([-0.3, -0.4, -0.5]),
            "routed_experts": _fallback_routes(3),
        }
    )

    train_batch = record_to_train_batch(
        _record([completion]),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["response_token_lengths"].tolist() == [0]


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
