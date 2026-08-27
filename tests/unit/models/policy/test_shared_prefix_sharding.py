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

from types import SimpleNamespace

import torch

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_GROUP_ID,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import SharedPrefixTrainingConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import TQPolicy


def _policy_with_dp_size(dp_size: int) -> Policy:
    policy = Policy.__new__(Policy)
    policy.shared_prefix_training_config = SharedPrefixTrainingConfig(mode="train")
    policy.sharding_annotations = SimpleNamespace(
        get_axis_size=lambda axis: dp_size if axis == "data_parallel" else 1
    )
    policy.cfg = {
        "make_sequence_length_divisible_by": 1,
        "sequence_packing": {
            "logprob_mb_tokens": 10,
            "train_mb_tokens": 10,
        },
    }
    return policy


def test_shared_prefix_policy_sharding_keeps_groups_complete_and_reorders_back():
    policy = _policy_with_dp_size(2)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(24).reshape(8, 3),
            "input_lengths": torch.tensor([9, 9, 8, 8, 2, 2, 1, 1]),
            SHARED_PREFIX_GROUP_ID: ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )

    shards, rank_order = policy._shard_for_logprob(data)

    assert [shard[SHARED_PREFIX_GROUP_ID] for shard in shards] == [
        ["a", "a", "d", "d"],
        ["b", "b", "c", "c"],
    ]
    for shard in shards:
        group_slots: dict[str, set[int]] = {}
        for group_id, slot_id in zip(
            shard[SHARED_PREFIX_GROUP_ID],
            shard[SHARED_PREFIX_EXECUTION_SLOT].tolist(),
            strict=True,
        ):
            group_slots.setdefault(group_id, set()).add(slot_id)
        assert all(slot_ids == {0, 1} for slot_ids in group_slots.values())
    assert rank_order == [0, 1, 6, 7, 2, 3, 4, 5]
    aggregated = BatchedDataDict.from_batches(shards)
    aggregated.reorder_data(rank_order)
    torch.testing.assert_close(aggregated["input_ids"], data["input_ids"])


def test_shared_prefix_train_and_logprob_use_the_same_dp_assignment():
    policy = _policy_with_dp_size(2)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(24).reshape(8, 3),
            "input_lengths": torch.tensor([9, 9, 8, 8, 2, 2, 1, 1]),
            SHARED_PREFIX_GROUP_ID: ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )

    logprob_shards, _ = policy._shard_for_logprob(data)
    train_shards = policy._shard_for_train(data, batch_size=8)

    assert [shard[SHARED_PREFIX_GROUP_ID] for shard in train_shards] == [
        shard[SHARED_PREFIX_GROUP_ID] for shard in logprob_shards
    ]


def test_shared_prefix_train_sharding_preserves_global_batch_boundaries():
    policy = _policy_with_dp_size(2)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(24).reshape(8, 3),
            "input_lengths": torch.tensor([9, 9, 8, 8, 2, 2, 1, 1]),
            SHARED_PREFIX_GROUP_ID: ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )

    train_shards = policy._shard_for_train(data, batch_size=4)

    assert [shard[SHARED_PREFIX_GROUP_ID] for shard in train_shards] == [
        ["a", "a", "c", "c"],
        ["b", "b", "d", "d"],
    ]


def test_tq_policy_only_fetches_shared_prefix_fields_in_train_mode():
    policy = TQPolicy.__new__(TQPolicy)
    fields = ["input_ids"]

    for mode in ("disabled", "observe"):
        policy.shared_prefix_training_config = SharedPrefixTrainingConfig(mode=mode)
        assert policy._with_shared_prefix_fields(fields) == fields

    policy.shared_prefix_training_config = SharedPrefixTrainingConfig(mode="train")
    assert policy._with_shared_prefix_fields(fields) == [
        "input_ids",
        "shared_prefix_group_id",
        "shared_prefix_prompt_lengths",
    ]
