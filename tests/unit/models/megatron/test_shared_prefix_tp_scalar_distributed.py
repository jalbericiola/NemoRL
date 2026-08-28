"""CUDA coverage for TP-sharded shared-prefix scalar log-probabilities."""

from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytestmark = pytest.mark.mcore


def _dense_logprob_oracle(
    global_logits: torch.Tensor,
    tensor_bin,
    source_input_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Expand the tiny star to conventional rows with a full-vocab oracle."""
    layout = tensor_bin.layout
    row_count, sequence_length = source_input_ids.shape
    prompt_prediction_count = layout.prompt_length - 1
    dense_logits = global_logits.new_zeros(
        row_count,
        sequence_length,
        global_logits.shape[-1],
    )
    dense_logits[:, :prompt_prediction_count] = global_logits[
        :, :prompt_prediction_count
    ].expand(row_count, -1, -1)

    source_to_local = {
        source_row: local_row for local_row, source_row in enumerate(layout.row_indices)
    }
    scatter_rows = torch.tensor(
        [source_to_local[row] for row in layout.completion_scatter_rows],
        dtype=torch.long,
        device=global_logits.device,
    )
    scatter_columns = tensor_bin.indices.completion_scatter_columns.to(
        device=global_logits.device
    )
    dense_logits[scatter_rows, scatter_columns] = global_logits[
        0,
        tensor_bin.indices.predecessor_positions.to(device=global_logits.device),
    ]

    local_input_ids = source_input_ids[list(layout.row_indices)].to(
        device=global_logits.device
    )
    logprobs = F.log_softmax((dense_logits[:, :-1] / temperature).float(), dim=-1)
    logprobs = logprobs.gather(
        dim=-1,
        index=local_input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    valid = torch.zeros_like(logprobs, dtype=torch.bool)
    valid[:, :prompt_prediction_count] = True
    valid[scatter_rows, scatter_columns] = True
    return torch.where(valid, logprobs, 0.0)


def _gradient_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_fp64 = actual.detach().double().flatten()
    expected_fp64 = expected.detach().double().flatten()
    difference = actual_fp64 - expected_fp64
    expected_norm = expected_fp64.norm().clamp_min(torch.finfo(torch.float64).tiny)
    actual_norm = actual_fp64.norm().clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "relative_l2": (difference.norm() / expected_norm).item(),
        "cosine": (
            torch.dot(actual_fp64, expected_fp64) / (actual_norm * expected_norm)
        ).item(),
        "absolute_max": difference.abs().max().item(),
    }


def _run_tp_shared_prefix_scalar_parity(
    rank: int,
    world_size: int,
    *,
    tp_size: int,
    cp_size: int,
) -> None:
    from megatron.core import parallel_state

    from nemo_rl.data.packing import (
        build_shared_prefix_tensor_plan,
        get_shared_prefix_context_parallel_indices,
        get_shared_prefix_physical_alignment,
    )
    from nemo_rl.models.megatron.data import SharedPrefixForwardMetadata
    from nemo_rl.models.megatron.train import shared_prefix_next_token_logprobs

    assert world_size == tp_size * cp_size
    device = torch.device("cuda", rank)
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        source_input_ids = torch.tensor(
            [
                [10, 11, 12, 20, 21, 0],
                [10, 11, 12, 30, 31, 0],
            ],
            dtype=torch.long,
        )
        alignment = get_shared_prefix_physical_alignment(
            tp_size=tp_size,
            cp_size=cp_size,
        )
        tensor_bin = build_shared_prefix_tensor_plan(
            input_ids=source_input_ids,
            input_lengths=torch.tensor([5, 5]),
            prompt_lengths=torch.tensor([3, 3]),
            group_ids=["g", "g"],
            bin_capacity=16,
            materialize_attention_mask=False,
            sequence_length_pad_multiple=alignment,
        ).shared_bins[0]
        logical_length = tensor_bin.layout.total_length
        physical_length = tensor_bin.layout.physical_total_length
        padded_length = ((physical_length + alignment - 1) // alignment) * alignment
        assert logical_length == 7
        assert physical_length == 13
        assert padded_length == 16

        vocab_size = 48
        local_vocab_size = vocab_size // tp_size
        vocab_start = tp_rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size
        cp_indices = get_shared_prefix_context_parallel_indices(
            padded_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            device=device,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260828)
        base_logits = torch.randn(
            1,
            padded_length,
            vocab_size,
            generator=generator,
            dtype=torch.float32,
        )
        predictor_count = (
            tensor_bin.layout.prompt_length
            - 1
            + tensor_bin.indices.predecessor_positions.numel()
        )
        temperature = 1.7

        for dtype in (torch.float32, torch.bfloat16):
            for chunk_size in (None, 2):
                global_logits = (
                    base_logits.to(device=device, dtype=dtype)
                    .clone()
                    .requires_grad_(True)
                )
                local_logits = (
                    global_logits.detach()
                    .index_select(1, cp_indices)[..., vocab_start:vocab_end]
                    .clone()
                    .requires_grad_(True)
                )
                oracle = _dense_logprob_oracle(
                    global_logits,
                    tensor_bin,
                    source_input_ids,
                    temperature=temperature,
                )
                reduction_payloads: list[torch.Size] = []
                real_all_reduce = dist.all_reduce
                real_functional_all_reduce = torch.distributed.nn.functional.all_reduce

                def tracked_all_reduce(value, *args, **kwargs):
                    reduction_payloads.append(value.shape)
                    return real_all_reduce(value, *args, **kwargs)

                def tracked_functional_all_reduce(value, *args, **kwargs):
                    reduction_payloads.append(value.shape)
                    return real_functional_all_reduce(value, *args, **kwargs)

                def reject_gather(*_args, **_kwargs):
                    raise AssertionError(
                        "shared-prefix TP scalar routing must not gather vocabulary logits"
                    )

                with (
                    patch(
                        "torch.distributed.all_reduce", side_effect=tracked_all_reduce
                    ),
                    patch(
                        "torch.distributed.nn.functional.all_reduce",
                        side_effect=tracked_functional_all_reduce,
                    ),
                    patch("torch.distributed.all_gather", side_effect=reject_gather),
                    patch(
                        "torch.distributed.all_gather_into_tensor",
                        side_effect=reject_gather,
                    ),
                    patch(
                        "torch.distributed.nn.functional.all_gather",
                        side_effect=reject_gather,
                    ),
                ):
                    restored = shared_prefix_next_token_logprobs(
                        local_logits,
                        SharedPrefixForwardMetadata(
                            tensor_bin=tensor_bin,
                            source_sequence_length=source_input_ids.shape[1],
                            cp_rank=cp_rank,
                            cp_size=cp_size,
                            padded_total_length=padded_length,
                            padding_multiple=alignment,
                        ),
                        chunk_size=chunk_size,
                        temperature=temperature,
                    )
                    weights = torch.arange(
                        1,
                        restored.numel() + 1,
                        dtype=restored.dtype,
                        device=device,
                    ).view_as(restored)
                    # LossPostProcessor divides each CP-replicated loss by CP.
                    (restored.mul(weights).sum() / cp_size).backward()

                torch.testing.assert_close(restored, oracle, rtol=2e-6, atol=2e-6)
                oracle.mul(weights).sum().backward()
                expected_local_grad = global_logits.grad.index_select(1, cp_indices)[
                    ..., vocab_start:vocab_end
                ]
                stats = _gradient_stats(local_logits.grad, expected_local_grad)
                print(
                    "shared_prefix_tp_scalar "
                    f"rank={rank} TP={tp_size} CP={cp_size} dtype={dtype} "
                    f"chunk_size={chunk_size} stats={stats} "
                    f"payloads={reduction_payloads}",
                    flush=True,
                )
                if dtype == torch.float32:
                    assert stats["relative_l2"] < 1e-5, stats
                    assert stats["cosine"] > 0.999999, stats
                    assert stats["absolute_max"] < 2e-6, stats
                else:
                    epsilon = torch.finfo(torch.bfloat16).eps
                    assert stats["relative_l2"] < epsilon, stats
                    assert stats["cosine"] > 1.0 - 4 * epsilon**2, stats
                    assert stats["absolute_max"] <= epsilon, stats

                assert reduction_payloads
                assert all(
                    payload.numel() <= predictor_count for payload in reduction_payloads
                )
                global_padding = torch.cat(
                    (
                        tensor_bin.indices.physical_padding_positions.to(device),
                        torch.arange(
                            physical_length,
                            padded_length,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                )
                local_padding = torch.nonzero(
                    torch.isin(cp_indices, global_padding),
                    as_tuple=False,
                ).flatten()
                if local_padding.numel():
                    assert (
                        torch.count_nonzero(local_logits.grad[:, local_padding]).item()
                        == 0
                    )
    finally:
        torch.cuda.synchronize(device)
        parallel_state.destroy_model_parallel()


@pytest.mark.timeout(240)
@pytest.mark.parametrize(("cp_size", "world_size"), [(1, 2), (2, 4)])
def test_tp2_shared_prefix_scalar_forward_backward_parity(
    distributed_test_runner,
    cp_size: int,
    world_size: int,
) -> None:
    """Validate TP2/CP{1,2} scalar routing without any vocab/logit gather."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix TP scalar parity requires CUDA BF16 support")
    distributed_test_runner(
        partial(
            _run_tp_shared_prefix_scalar_parity,
            tp_size=2,
            cp_size=cp_size,
        ),
        world_size=world_size,
        backend="nccl",
    )
