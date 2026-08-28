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

"""Two-rank CUDA coverage for shared-prefix CP scalar reconstruction."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytestmark = pytest.mark.mcore


def _dense_conventional_logprob_oracle(
    global_padded_logits: torch.Tensor,
    tensor_bin,
    source_input_ids: torch.Tensor,
) -> torch.Tensor:
    """Expand the tiny test star to conventional rows as an independent oracle."""
    layout = tensor_bin.layout
    row_count, sequence_length = source_input_ids.shape
    vocab_size = global_padded_logits.shape[-1]
    prompt_prediction_count = layout.prompt_length - 1

    dense_logits = global_padded_logits.new_zeros(
        (row_count, sequence_length, vocab_size)
    )
    dense_logits[:, :prompt_prediction_count] = global_padded_logits[
        :, :prompt_prediction_count
    ].expand(row_count, -1, -1)

    source_to_local = {
        source_row: local_row
        for local_row, source_row in enumerate(layout.row_indices)
    }
    scatter_rows = torch.tensor(
        [source_to_local[row] for row in layout.completion_scatter_rows],
        dtype=torch.long,
        device=global_padded_logits.device,
    )
    scatter_columns = tensor_bin.indices.completion_scatter_columns.to(
        device=global_padded_logits.device
    )
    dense_logits[scatter_rows, scatter_columns] = global_padded_logits[
        0,
        tensor_bin.indices.predecessor_positions.to(
            device=global_padded_logits.device
        ),
    ]

    local_input_ids = source_input_ids[list(layout.row_indices)].to(
        device=global_padded_logits.device
    )
    dense_logprobs = F.log_softmax(dense_logits[:, :-1].float(), dim=-1)
    dense_logprobs = dense_logprobs.gather(
        dim=-1,
        index=local_input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    valid_predictions = torch.zeros_like(dense_logprobs, dtype=torch.bool)
    valid_predictions[:, :prompt_prediction_count] = True
    valid_predictions[scatter_rows, scatter_columns] = True
    return torch.where(valid_predictions, dense_logprobs, 0.0)


def _gradient_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, float]:
    """Return relative L2 error, cosine similarity, and maximum absolute error."""
    actual_fp64 = actual.detach().to(torch.float64).flatten()
    expected_fp64 = expected.detach().to(torch.float64).flatten()
    difference = actual_fp64 - expected_fp64
    expected_norm = torch.linalg.vector_norm(expected_fp64)
    actual_norm = torch.linalg.vector_norm(actual_fp64)
    relative_l2 = torch.linalg.vector_norm(difference) / expected_norm
    cosine = torch.dot(actual_fp64, expected_fp64) / (
        actual_norm * expected_norm
    )
    return (
        relative_l2.item(),
        cosine.item(),
        difference.abs().max().item(),
    )


def _assert_gradient_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    """Apply dtype-derived tolerances and return diagnostics for logging."""
    relative_l2, cosine, max_absolute = _gradient_metrics(actual, expected)
    epsilon = torch.finfo(dtype).eps
    if dtype == torch.float32:
        # Different autograd graphs accumulate repeated prompt-row gradients in
        # a different order. Permit a small multiple of FP32 machine epsilon,
        # while requiring the aggregate error to remain near machine precision.
        torch.testing.assert_close(
            actual,
            expected,
            rtol=32 * epsilon,
            atol=32 * epsilon,
        )
        assert relative_l2 <= 32 * epsilon
        assert cosine >= 1.0 - 64 * epsilon
    else:
        assert dtype == torch.bfloat16
        # Both tensors are already rounded BF16 gradients. One BF16 epsilon is
        # the natural elementwise/relative-L2 bound for an accumulation-order
        # difference; the cosine bound is the corresponding second-order error.
        torch.testing.assert_close(
            actual,
            expected,
            rtol=epsilon,
            atol=epsilon,
        )
        assert relative_l2 <= epsilon
        assert cosine >= 1.0 - 4 * epsilon**2
    return relative_l2, cosine, max_absolute


def _run_shared_prefix_cp2_scalar_reduce(rank: int, world_size: int) -> None:
    from nemo_rl.data.packing import (
        build_shared_prefix_tensor_plan,
        get_shared_prefix_context_parallel_indices,
    )
    from nemo_rl.models.megatron.data import SharedPrefixForwardMetadata
    from nemo_rl.models.megatron.train import shared_prefix_next_token_logprobs

    assert world_size == 2
    assert dist.is_initialized()
    assert dist.get_world_size() == world_size
    assert dist.get_rank() == rank
    device = torch.device("cuda", rank)

    # Logical star length is seven, while CP2 requires a minimally padded
    # physical length divisible by 2 * CP, namely eight.
    source_input_ids = torch.tensor(
        [
            [10, 11, 12, 20, 21, 0],
            [10, 11, 12, 30, 31, 0],
        ],
        dtype=torch.long,
    )
    tensor_bin = build_shared_prefix_tensor_plan(
        input_ids=source_input_ids,
        input_lengths=torch.tensor([5, 5]),
        prompt_lengths=torch.tensor([3, 3]),
        group_ids=["g", "g"],
        bin_capacity=16,
        materialize_attention_mask=False,
    ).shared_bins[0]
    logical_length = tensor_bin.layout.total_length
    padded_length = 8
    vocab_size = 43
    assert logical_length == 7 < padded_length

    # Generate one deterministic global leaf on every rank. The test oracle
    # consumes the global tensor; production consumes only a two-chunk zigzag
    # selection. Both FP32 and BF16 run through the real collective below.
    generator = torch.Generator(device="cpu").manual_seed(20260827)
    base_global_logits = torch.randn(
        1,
        padded_length,
        vocab_size,
        generator=generator,
        dtype=torch.float32,
    )
    global_indices = get_shared_prefix_context_parallel_indices(
        padded_length,
        cp_rank=rank,
        cp_size=world_size,
        device=device,
    )
    predictor_count = (
        tensor_bin.layout.prompt_length
        - 1
        + tensor_bin.indices.predecessor_positions.numel()
    )
    reduce_payloads: list[tuple[torch.Size, torch.dtype, torch.device]] = []
    real_differentiable_all_reduce = torch.distributed.nn.functional.all_reduce
    case_outputs: dict[tuple[torch.dtype, int | None], torch.Tensor] = {}
    case_gradients: dict[tuple[torch.dtype, int | None], torch.Tensor] = {}

    def tracked_real_all_reduce(value, *args, **kwargs):
        reduce_payloads.append((value.shape, value.dtype, value.device))
        return real_differentiable_all_reduce(value, *args, **kwargs)

    def reject_vocab_gather(*_args, **_kwargs):
        raise AssertionError("shared-prefix CP logprobs must not all-gather logits")

    for dtype in (torch.float32, torch.bfloat16):
        for chunk_size in (None, 2):
            global_padded_logits = (
                base_global_logits.to(device=device, dtype=dtype)
                .clone()
                .requires_grad_(True)
            )
            local_logits = (
                global_padded_logits.detach()
                .index_select(1, global_indices)
                .clone()
                .requires_grad_(True)
            )
            oracle = _dense_conventional_logprob_oracle(
                global_padded_logits,
                tensor_bin,
                source_input_ids,
            )

            with (
                patch(
                    "nemo_rl.models.megatron.train.get_context_parallel_group",
                    return_value=dist.group.WORLD,
                ),
                patch(
                    "torch.distributed.nn.functional.all_reduce",
                    side_effect=tracked_real_all_reduce,
                ),
                patch(
                    "torch.distributed.nn.functional.all_gather",
                    side_effect=reject_vocab_gather,
                ),
                patch(
                    "torch.distributed.all_gather",
                    side_effect=reject_vocab_gather,
                ),
                patch(
                    "torch.distributed.all_gather_into_tensor",
                    side_effect=reject_vocab_gather,
                ),
            ):
                restored = shared_prefix_next_token_logprobs(
                    local_logits,
                    SharedPrefixForwardMetadata(
                        tensor_bin=tensor_bin,
                        source_sequence_length=source_input_ids.shape[1],
                        cp_rank=rank,
                        cp_size=world_size,
                        padded_total_length=padded_length,
                    ),
                    chunk_size=chunk_size,
                )

                # LossPostProcessor divides the loss replicated on every CP rank
                # by CP. The differentiable SUM then sums those rank-local
                # cotangents in backward, recovering the dense/global scale.
                weights = torch.arange(
                    1,
                    restored.numel() + 1,
                    dtype=restored.dtype,
                    device=device,
                ).view_as(restored)
                (restored.mul(weights).sum() / world_size).backward()

            torch.testing.assert_close(restored, oracle, rtol=0.0, atol=0.0)
            (oracle.mul(weights).sum()).backward()
            expected_local_grad = global_padded_logits.grad.index_select(
                1, global_indices
            )
            relative_l2, cosine, max_absolute = _gradient_metrics(
                local_logits.grad,
                expected_local_grad,
            )
            print(
                "shared_prefix_cp2_gradient "
                f"rank={rank} dtype={dtype} chunk_size={chunk_size} "
                f"relative_l2={relative_l2:.9e} cosine={cosine:.12f} "
                f"max_absolute={max_absolute:.9e}",
                flush=True,
            )
            _assert_gradient_parity(
                local_logits.grad,
                expected_local_grad,
                dtype=dtype,
            )
            case_outputs[(dtype, chunk_size)] = restored.detach().clone()
            case_gradients[(dtype, chunk_size)] = local_logits.grad.detach().clone()

            # Explicitly check that the physical padding row is loss-dead on
            # its owner for every dtype and chunking mode.
            padding_local_offset = torch.nonzero(
                global_indices == logical_length,
                as_tuple=False,
            ).flatten()
            if padding_local_offset.numel():
                assert (
                    torch.count_nonzero(
                        local_logits.grad[0, padding_local_offset]
                    ).item()
                    == 0
                )

    assert reduce_payloads == [
        (torch.Size([predictor_count]), torch.float32, device)
    ] * 4
    assert predictor_count < (padded_length // world_size) * vocab_size
    assert predictor_count != vocab_size

    # Chunking changes only the maximum temporary vocabulary rows. Report and
    # bound its numerical effect separately from distributed-vs-dense parity.
    for dtype in (torch.float32, torch.bfloat16):
        torch.testing.assert_close(
            case_outputs[(dtype, None)],
            case_outputs[(dtype, 2)],
            rtol=0.0,
            atol=0.0,
        )
        relative_l2, cosine, max_absolute = _gradient_metrics(
            case_gradients[(dtype, None)],
            case_gradients[(dtype, 2)],
        )
        print(
            "shared_prefix_cp2_chunk_gradient "
            f"rank={rank} dtype={dtype} none_vs_2_relative_l2={relative_l2:.9e} "
            f"cosine={cosine:.12f} max_absolute={max_absolute:.9e}",
            flush=True,
        )
        _assert_gradient_parity(
            case_gradients[(dtype, None)],
            case_gradients[(dtype, 2)],
            dtype=dtype,
        )


def test_shared_prefix_cp2_uses_real_scalar_all_reduce_with_dense_gradient_parity(
    distributed_test_runner,
) -> None:
    """Run the CP2 scalar forward/backward contract on two real CUDA ranks."""
    distributed_test_runner(
        _run_shared_prefix_cp2_scalar_reduce,
        world_size=2,
        backend="nccl",
    )
