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

import pytest
import torch

from nemo_rl.data.packing import (
    SharedPrefixFallbackReason,
    build_shared_prefix_tensor_plan,
)
from nemo_rl.data.packing.shared_prefix import (
    SharedPrefixLayout,
    SharedPrefixRow,
    build_shared_prefix_layout,
)
from nemo_rl.data.packing.shared_prefix_tensors import (
    build_star_attention_allow_mask,
    get_shared_prefix_context_parallel_indices,
    materialize_shared_prefix_layout,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    return (
        torch.tensor(
            [
                [10, 11, 12, 20, 21, -1],
                [10, 11, 12, 30, 31, 32],
                [40, 41, 50, 51, -1, -1],
            ],
            dtype=torch.long,
        ),
        torch.tensor([5, 6, 4]),
        torch.tensor([3, 3, 2]),
        ["shared", "shared", "singleton"],
    )


def test_tensor_plan_materializes_prompt_once_and_preserves_fallbacks() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()

    plan = build_shared_prefix_tensor_plan(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
    )

    assert len(plan.shared_bins) == 1
    shared_bin = plan.shared_bins[0]
    assert shared_bin.layout.row_indices == (1, 0)
    torch.testing.assert_close(
        shared_bin.packed_input_ids,
        torch.tensor([10, 11, 12, 30, 31, 32, 20, 21]),
    )
    torch.testing.assert_close(
        shared_bin.position_ids,
        torch.tensor([0, 1, 2, 3, 4, 5, 3, 4]),
    )
    assert plan.fallback_row_indices == (2,)
    assert plan.fallbacks[0].reason is SharedPrefixFallbackReason.NO_EXACT_PROMPT_PEER
    assert plan.fallbacks[0].fits_block_diagonal_bin


def test_tensor_indices_encode_gather_fanout_and_scatter() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()
    shared_bin = build_shared_prefix_tensor_plan(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
    ).shared_bins[0]
    indices = shared_bin.indices

    torch.testing.assert_close(
        indices.token_gather_rows,
        torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]),
    )
    torch.testing.assert_close(
        indices.token_gather_columns,
        torch.tensor([0, 1, 2, 3, 4, 5, 3, 4]),
    )
    torch.testing.assert_close(
        indices.completion_positions,
        torch.tensor([3, 4, 5, 6, 7]),
    )
    torch.testing.assert_close(
        indices.predecessor_positions,
        torch.tensor([2, 3, 4, 2, 6]),
    )
    torch.testing.assert_close(
        indices.completion_scatter_rows,
        torch.tensor([1, 1, 1, 0, 0]),
    )
    torch.testing.assert_close(
        indices.completion_scatter_columns,
        torch.tensor([2, 3, 4, 2, 3]),
    )


def test_context_parallel_indices_match_standard_two_chunk_zigzag() -> None:
    rank0 = get_shared_prefix_context_parallel_indices(
        16,
        cp_rank=0,
        cp_size=2,
    )
    rank1 = get_shared_prefix_context_parallel_indices(
        16,
        cp_rank=1,
        cp_size=2,
    )

    torch.testing.assert_close(rank0, torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]))
    torch.testing.assert_close(rank1, torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    torch.testing.assert_close(
        torch.cat((rank0, rank1)).sort().values,
        torch.arange(16),
    )


def test_context_parallel_shard_pads_only_after_real_star_tokens() -> None:
    input_ids = torch.tensor(
        [
            [10, 11, 12, 20, 0],
            [10, 11, 12, 30, 31],
        ]
    )
    tensor_bin = build_shared_prefix_tensor_plan(
        input_ids=input_ids,
        input_lengths=torch.tensor([4, 5]),
        prompt_lengths=torch.tensor([3, 3]),
        group_ids=["g", "g"],
        bin_capacity=8,
        materialize_attention_mask=False,
    ).shared_bins[0]

    rank0 = shard_shared_prefix_tensor_bin_for_context_parallel(
        tensor_bin,
        cp_rank=0,
        cp_size=2,
    )
    rank1 = shard_shared_prefix_tensor_bin_for_context_parallel(
        tensor_bin,
        cp_rank=1,
        cp_size=2,
    )

    assert tensor_bin.layout.total_length == 6
    assert rank0.padded_total_length == rank1.padded_total_length == 8
    torch.testing.assert_close(rank0.global_token_indices, torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(rank1.global_token_indices, torch.tensor([2, 3, 4, 5]))
    torch.testing.assert_close(rank0.packed_input_ids, torch.tensor([10, 11, 0, 0]))
    torch.testing.assert_close(rank1.packed_input_ids, torch.tensor([12, 30, 31, 20]))


@pytest.mark.parametrize(
    "padded_length,cp_rank,cp_size,message",
    [
        (8, 0, 0, "cp_size must be positive"),
        (8, 2, 2, "cp_rank must be in"),
        (6, 0, 2, r"divisible by 2 \* cp_size"),
    ],
)
def test_context_parallel_indices_reject_invalid_topology(
    padded_length: int,
    cp_rank: int,
    cp_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        get_shared_prefix_context_parallel_indices(
            padded_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )


def test_cp1_star_attention_mask_has_exact_branch_isolation() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()
    mask = (
        build_shared_prefix_tensor_plan(
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
            group_ids=group_ids,
            bin_capacity=16,
        )
        .shared_bins[0]
        .attention_allow_mask
    )

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)


def test_fused_consumer_can_skip_dense_attention_oracle() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()

    shared_bin = build_shared_prefix_tensor_plan(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
        materialize_attention_mask=False,
    ).shared_bins[0]

    assert shared_bin.attention_allow_mask is None
    torch.testing.assert_close(
        shared_bin.packed_input_ids,
        torch.tensor([10, 11, 12, 30, 31, 32, 20, 21]),
    )


def test_group_id_and_exact_prompt_are_both_required() -> None:
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, 4],
            [1, 9, 5],
            [1, 2, 6],
        ]
    )
    plan = build_shared_prefix_tensor_plan(
        input_ids=input_ids,
        input_lengths=torch.tensor([3, 3, 3, 3]),
        prompt_lengths=torch.tensor([2, 2, 2, 2]),
        group_ids=["a", "b", "a", "a"],
        bin_capacity=8,
    )

    assert [shared.layout.row_indices for shared in plan.shared_bins] == [(0, 3)]
    assert plan.fallback_row_indices == (1, 2)
    assert [fallback.reason for fallback in plan.fallbacks] == [
        SharedPrefixFallbackReason.NO_EXACT_PROMPT_PEER,
        SharedPrefixFallbackReason.PROMPT_MISMATCH,
    ]


def test_materializer_rejects_stale_prompt_tokens() -> None:
    layout = build_shared_prefix_layout(
        [
            SharedPrefixRow(0, "g", (1, 2), 1),
            SharedPrefixRow(1, "g", (1, 2), 1),
        ]
    )
    changed_input_ids = torch.tensor([[1, 2, 3], [1, 9, 4]])

    with pytest.raises(ValueError, match="differs from the planned exact prompt"):
        materialize_shared_prefix_layout(
            changed_input_ids,
            input_lengths=torch.tensor([3, 3]),
            layout=layout,
        )


def test_materializer_rejects_a_layout_past_the_unpadded_row() -> None:
    layout = build_shared_prefix_layout(
        [
            SharedPrefixRow(0, "g", (1, 2), 2),
            SharedPrefixRow(1, "g", (1, 2), 1),
        ]
    )
    input_ids = torch.tensor([[1, 2, 3, -1], [1, 2, 4, -1]])

    with pytest.raises(ValueError, match="layout requires 4 tokens"):
        materialize_shared_prefix_layout(
            input_ids,
            input_lengths=torch.tensor([3, 3]),
            layout=layout,
        )


def test_mask_builder_rejects_noncontiguous_branch_spans() -> None:
    invalid = SharedPrefixLayout(
        group_id="g",
        prompt_token_ids=(1, 2),
        row_indices=(0, 1),
        completion_lengths=(1, 1),
        total_length=4,
        branch_starts=(2, 2),
        position_ids=(0, 1, 2, 2),
        token_gather_rows=(0, 0, 0, 1),
        token_gather_columns=(0, 1, 2, 2),
        completion_positions=(2, 3),
        predecessor_positions=(1, 1),
        completion_scatter_rows=(0, 1),
        completion_scatter_columns=(1, 1),
    )

    with pytest.raises(ValueError, match="positive and contiguous"):
        build_star_attention_allow_mask(invalid)


@pytest.mark.parametrize(
    "input_ids, input_lengths, prompt_lengths, group_ids, message",
    [
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([3]),
            torch.tensor([2]),
            ["g"],
            "input_ids must have shape",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor([3, 3]),
            torch.tensor([2, 2]),
            ["g", "g"],
            "integer dtype",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 4]),
            torch.tensor([2, 2]),
            ["g", "g"],
            "input_lengths must be within",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 2]),
            torch.tensor([2, 3]),
            ["g", "g"],
            "prompt_lengths must satisfy",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 3]),
            torch.tensor([2, 2]),
            ["g"],
            "group_ids must have 2 entries",
        ),
    ],
)
def test_tensor_plan_rejects_invalid_batch_metadata(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_shared_prefix_tensor_plan(
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
            group_ids=group_ids,
            bin_capacity=8,
        )
