"""TP/SP HybridModel-to-NeMo shared-prefix loss integration coverage."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore


class _CompletionLogprobLoss:
    """Deterministic completion-only objective with distinct branch weights."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __call__(
        self,
        *,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del global_valid_seqs, global_valid_toks
        loss = -(next_token_logprobs * data["loss_weights"]).sum()
        return loss, {"completion_logprob_loss": loss.detach()}


def _scheduled_loss(
    restored_logprobs: torch.Tensor,
    *,
    data: BatchedDataDict[Any],
    cp_size: int,
) -> torch.Tensor:
    """Use LossPostProcessor and algebraically emulate schedule scaling."""
    from nemo_rl.models.megatron.train import LossPostProcessor

    num_microbatches = 3
    processor = LossPostProcessor(
        loss_fn=_CompletionLogprobLoss(),
        cfg={"sequence_packing": {"enabled": True, "fuse_loss": True}},
        num_microbatches=num_microbatches,
        cp_normalize=True,
    )
    loss_fn = processor(
        data_dict=data,
        packed_seq_params=None,
        global_valid_seqs=torch.tensor(data.size, device=restored_logprobs.device),
        global_valid_toks=torch.count_nonzero(data["loss_weights"]),
        input_is_next_token_logprobs=True,
    )
    prescaled_loss, _ = loss_fn(restored_logprobs)
    return prescaled_loss * cp_size / num_microbatches


def _gradient_stats(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_fp64 = reference.detach().double().flatten()
    actual_fp64 = actual.detach().double().flatten()
    difference = actual_fp64 - reference_fp64
    reference_norm = reference_fp64.norm()
    actual_norm = actual_fp64.norm()
    denominator = (reference_norm * actual_norm).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    return {
        "relative_l2": (
            difference.norm()
            / reference_norm.clamp_min(torch.finfo(torch.float64).tiny)
        ).item(),
        "cosine": (torch.dot(reference_fp64, actual_fp64) / denominator).item(),
        "reference_norm": reference_norm.item(),
        "actual_norm": actual_norm.item(),
        "absolute_l2": difference.norm().item(),
        "absolute_max": difference.abs().max().item(),
    }


def _assert_gradient_parity(stats: dict[str, float], *, label: str) -> None:
    if stats["reference_norm"] <= 1e-10:
        assert stats["absolute_max"] <= 1e-6, f"{label}: {stats}"
        return
    assert stats["relative_l2"] < 0.03, f"{label}: {stats}"
    assert stats["cosine"] > 0.995, f"{label}: {stats}"


def _max_conditioned_error(
    reference: torch.Tensor,
    actual: torch.Tensor,
    absolute_sum: torch.Tensor,
) -> float:
    return (
        (
            (reference.detach().double() - actual.detach().double()).abs()
            / absolute_sum.detach().double().clamp_min(1e-300)
        )
        .max()
        .item()
    )


def _all_reduced_parameter_grads(
    model: torch.nn.Module,
    tp_group: dist.ProcessGroup,
    cp_group: dist.ProcessGroup,
) -> dict[str, torch.Tensor]:
    model_config = model.config
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().clone()
        needs_tp_sum = (
            bool(model_config.sequence_parallel)
            and bool(getattr(parameter, "sequence_parallel", False))
        ) or (
            bool(model_config.qk_layernorm)
            and ("q_layernorm" in name or "k_layernorm" in name)
        )
        if tp_group.size() > 1 and needs_tp_sum:
            # Match MCore's production gradient finalization. Dense branches
            # and the star can place selected tokens on different SP shards,
            # so replicated SP parameters are comparable only after this SUM.
            dist.all_reduce(gradient, group=tp_group)
        if cp_group.size() > 1:
            dist.all_reduce(gradient, group=cp_group)
        gradients[name] = gradient
    return gradients


def _run_tp_sp_hybrid_to_nemo_loss_parity(
    rank: int,
    world_size: int,
    *,
    cp_size: int,
) -> None:
    from megatron.core import parallel_state
    from megatron.core.models.hybrid import shared_prefix as mcore_shared_prefix
    from megatron.core.models.hybrid.hybrid_layer_allocation import (
        validate_segment_layers,
    )
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.hybrid.shared_prefix import SharedPrefixLayout
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm import mamba_mixer as mamba_mixer_module
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.data.packing import (
        build_shared_prefix_tensor_plan,
        get_shared_prefix_context_parallel_indices,
        get_shared_prefix_physical_alignment,
        shard_shared_prefix_tensor_bin_for_context_parallel,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.model_utils import from_parallel_logits_to_logprobs
    from nemo_rl.models.megatron.data import (
        ProcessedMicrobatch,
        SharedPrefixForwardMetadata,
    )
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        forward_with_post_processing_fn,
        shared_prefix_next_token_logprobs,
    )

    tp_size = 2
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
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        tp_rank = tp_group.rank()
        cp_rank = cp_group.rank()

        # The two first-completion targets deliberately live on different TP
        # vocabulary shards. The planner orders the longer completion first.
        source_input_ids = torch.tensor(
            [
                [10, 11, 12, 13, 30, 31, 32, 0],
                [10, 11, 12, 13, 80, 81, 82, 83],
            ],
            dtype=torch.long,
        )
        source_lengths = torch.tensor((7, 8), dtype=torch.long)
        physical_alignment = get_shared_prefix_physical_alignment(
            tp_size=tp_size,
            cp_size=cp_size,
        )
        padding_multiple = 16
        assert padding_multiple > physical_alignment
        assert padding_multiple % physical_alignment == 0
        tensor_bin = build_shared_prefix_tensor_plan(
            input_ids=source_input_ids,
            input_lengths=source_lengths,
            prompt_lengths=torch.tensor((4, 4), dtype=torch.long),
            group_ids=("g", "g"),
            bin_capacity=32,
            materialize_attention_mask=False,
            sequence_length_pad_multiple=padding_multiple,
        ).shared_bins[0]
        layout = tensor_bin.layout
        mcore_layout = SharedPrefixLayout(
            prefix_len=layout.prompt_length,
            completion_lens=layout.physical_completion_lengths,
            logical_completion_lens=layout.completion_lengths,
            padding_multiple=padding_multiple,
        )
        assert layout.total_length == 11
        assert tuple(layout.completion_lengths) == (4, 3)
        assert tuple(layout.physical_completion_lengths) == (12, 12)
        assert layout.physical_total_length == 28
        shared_physical_length = (
            (layout.physical_total_length + padding_multiple - 1) // padding_multiple
        ) * padding_multiple
        assert shared_physical_length == 32
        dense_physical_length = padding_multiple

        hidden_size = 256
        vocab_size = 128
        local_vocab_size = vocab_size // tp_size
        pattern = "M*"
        torch.manual_seed(20260828)
        model_parallel_cuda_manual_seed(20260828)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(validate_segment_layers(pattern)),
            num_attention_heads=32,
            num_query_groups=2,
            kv_channels=128,
            mamba_num_heads=64,
            mamba_num_groups=8,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=True,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=True,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd", "dp_cp"]
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=shared_physical_length,
            hybrid_layer_pattern=pattern,
            position_embedding_type="rope",
            pre_process=True,
            post_process=True,
            parallel_output=True,
            pg_collection=process_groups,
        ).to(device=device)
        model.train()
        cp_source_rank = dist.get_global_rank(cp_group, 0)
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=cp_source_rank, group=cp_group)

        mamba_mixer = model.decoder.layers[0].mixer
        scan_impl = mamba_mixer_module.mamba_chunk_scan_combined
        assert mcore_shared_prefix.mamba_chunk_scan_combined is scan_impl

        @contextmanager
        def capture_mamba_scans(
            records: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> Any:
            def record_scan(*args: Any, **kwargs: Any) -> Any:
                result = scan_impl(*args, **kwargs)
                scan_output = result[0] if isinstance(result, tuple) else result
                scan_output.retain_grad()
                records.append((args[0], scan_output))
                return result

            with (
                patch.object(
                    mamba_mixer_module,
                    "mamba_chunk_scan_combined",
                    record_scan,
                ),
                patch.object(
                    mcore_shared_prefix,
                    "mamba_chunk_scan_combined",
                    record_scan,
                ),
            ):
                yield

        def reconstruct_mamba_d_vjp(
            records: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            local_vjp = None
            local_absolute_sum = None
            for scan_input, scan_output in records:
                if scan_output.grad is None:
                    raise RuntimeError(
                        "captured Mamba scan output did not receive a gradient"
                    )
                product = (
                    scan_output.grad.detach().double() * scan_input.detach().double()
                )
                contribution = product.sum(dim=(0, 1))
                absolute_sum = product.abs().sum(dim=(0, 1))
                if not mamba_mixer.D_has_hdim:
                    contribution = contribution.sum(dim=-1)
                    absolute_sum = absolute_sum.sum(dim=-1)
                contribution = contribution.reshape(-1)
                absolute_sum = absolute_sum.reshape(-1)
                local_vjp = (
                    contribution if local_vjp is None else local_vjp + contribution
                )
                local_absolute_sum = (
                    absolute_sum
                    if local_absolute_sum is None
                    else local_absolute_sum + absolute_sum
                )
            if local_vjp is None or local_absolute_sum is None:
                raise RuntimeError("Mamba D diagnostics captured no scan calls")

            full_vjp = torch.zeros_like(mamba_mixer.D, dtype=torch.float64)
            full_absolute_sum = torch.zeros_like(
                mamba_mixer.D,
                dtype=torch.float64,
            )
            if cp_size == 1:
                full_vjp.reshape(-1).copy_(local_vjp)
                full_absolute_sum.reshape(-1).copy_(local_absolute_sum)
            else:
                if local_vjp.numel() * cp_size != mamba_mixer.D.numel():
                    raise RuntimeError(
                        "CP-local Mamba D formula does not cover the TP-local D shard"
                    )
                start = cp_rank * local_vjp.numel()
                full_vjp.reshape(-1)[start : start + local_vjp.numel()] = local_vjp
                full_absolute_sum.reshape(-1)[
                    start : start + local_absolute_sum.numel()
                ] = local_absolute_sum
                dist.all_reduce(full_vjp, group=cp_group)
                dist.all_reduce(full_absolute_sum, group=cp_group)
            return full_vjp, full_absolute_sum

        row_indices = list(layout.row_indices)
        local_source_ids = source_input_ids[row_indices].to(device=device)
        local_source_lengths = source_lengths[row_indices].tolist()
        first_completion_targets = local_source_ids[:, layout.prompt_length]
        assert sorted((first_completion_targets // local_vocab_size).tolist()) == [0, 1]

        loss_weights = torch.zeros(
            len(row_indices),
            source_input_ids.shape[1] - 1,
            dtype=torch.float32,
            device=device,
        )
        for row, source_length in enumerate(local_source_lengths):
            completion_count = source_length - layout.prompt_length
            loss_weights[row, layout.prompt_length - 1 : source_length - 1] = (
                torch.arange(
                    1,
                    completion_count + 1,
                    dtype=torch.float32,
                    device=device,
                )
                * (1.0 + 0.25 * row)
            )
        loss_data = BatchedDataDict(
            {"input_ids": local_source_ids, "loss_weights": loss_weights}
        )

        def cp_localize(global_tensor: torch.Tensor) -> torch.Tensor:
            indices = get_shared_prefix_context_parallel_indices(
                global_tensor.shape[1],
                cp_rank=cp_rank,
                cp_size=cp_size,
                device=device,
            )
            return global_tensor.to(device=device).index_select(1, indices)

        # Dense conventional oracle: each branch runs independently through the
        # same TP/SP HybridModel, then uses NeMo's existing TP log-prob primitive.
        dense_logits = []
        dense_rows = []
        dense_scan_records: list[tuple[torch.Tensor, torch.Tensor]] = []
        with capture_mamba_scans(dense_scan_records):
            for row, source_length in enumerate(local_source_lengths):
                global_tokens = F.pad(
                    local_source_ids[row, :source_length],
                    (0, dense_physical_length - source_length),
                ).unsqueeze(0)
                global_positions = torch.arange(
                    dense_physical_length,
                    device=device,
                ).unsqueeze(0)
                branch_logits = model(
                    cp_localize(global_tokens),
                    cp_localize(global_positions),
                    None,
                    runtime_gather_output=False,
                )
                assert branch_logits.shape == (
                    1,
                    dense_physical_length // cp_size,
                    local_vocab_size,
                )
                branch_logits.retain_grad()
                branch_logprobs = from_parallel_logits_to_logprobs(
                    branch_logits,
                    global_tokens,
                    tp_rank * local_vocab_size,
                    (tp_rank + 1) * local_vocab_size,
                    tp_group,
                    inference_only=False,
                    cp_group=cp_group,
                    chunk_size=2,
                )[0]
                dense_rows.append(
                    F.pad(
                        branch_logprobs[: source_length - 1],
                        (0, source_input_ids.shape[1] - source_length),
                    )
                )
                dense_logits.append(branch_logits)
        dense_restored = torch.stack(dense_rows)
        dense_loss = _scheduled_loss(
            dense_restored,
            data=loss_data,
            cp_size=cp_size,
        )
        dense_loss.backward()
        dense_parameter_grads = _all_reduced_parameter_grads(
            model,
            tp_group,
            cp_group,
        )
        dense_d_formula, dense_d_absolute_sum = reconstruct_mamba_d_vjp(
            dense_scan_records
        )

        last_prompt_position = layout.prompt_length - 1
        dense_last_prompt_vjp = torch.zeros(
            local_vocab_size,
            dtype=dense_logits[0].grad.dtype,
            device=device,
        )
        dense_cp_indices = get_shared_prefix_context_parallel_indices(
            dense_physical_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            device=device,
        )
        dense_local_predictor = torch.nonzero(
            dense_cp_indices == last_prompt_position,
            as_tuple=False,
        ).flatten()
        if dense_local_predictor.numel():
            local_offset = int(dense_local_predictor.item())
            dense_last_prompt_vjp = sum(
                logits.grad[0, local_offset] for logits in dense_logits
            )
        if cp_size > 1:
            dist.all_reduce(dense_last_prompt_vjp, group=cp_group)

        model.zero_grad(set_to_none=True)
        shared_shard = shard_shared_prefix_tensor_bin_for_context_parallel(
            tensor_bin,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_size=tp_size,
            padding_multiple=padding_multiple,
        )
        shared_input_ids = shared_shard.packed_input_ids.to(device).unsqueeze(0)
        shared_positions = (
            mcore_layout.padded_position_ids(
                shared_physical_length,
                device,
            )
            .index_select(
                0,
                shared_shard.global_token_indices.to(device),
            )
            .unsqueeze(0)
        )
        shared_scan_records: list[tuple[torch.Tensor, torch.Tensor]] = []
        shared_metadata = SharedPrefixForwardMetadata(
            tensor_bin=tensor_bin,
            source_sequence_length=source_input_ids.shape[1],
            cp_rank=cp_rank,
            cp_size=cp_size,
            padded_total_length=shared_physical_length,
            padding_multiple=padding_multiple,
        )

        def reject_cp_logit_gather(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "shared-prefix NeMo loss routing must not gather vocabulary logits"
            )

        with (
            capture_mamba_scans(shared_scan_records),
            patch(
                "nemo_rl.models.megatron.train.allgather_cp_sharded_tensor",
                side_effect=reject_cp_logit_gather,
            ),
        ):
            captured_logits: list[torch.Tensor] = []
            real_scalar_router = shared_prefix_next_token_logprobs

            def capture_scalar_router(
                logits: torch.Tensor,
                metadata: SharedPrefixForwardMetadata,
                **kwargs: Any,
            ) -> torch.Tensor:
                logits.retain_grad()
                captured_logits.append(logits)
                return real_scalar_router(logits, metadata, **kwargs)

            processor = LossPostProcessor(
                loss_fn=_CompletionLogprobLoss(),
                cfg={
                    "logprob_chunk_size": 2,
                    "sequence_packing": {"enabled": True, "fuse_loss": True},
                },
                num_microbatches=3,
                cp_normalize=True,
            )
            processed = ProcessedMicrobatch(
                data_dict=loss_data,
                input_ids=local_source_ids,
                input_ids_cp_sharded=shared_input_ids,
                attention_mask=None,
                position_ids=shared_positions,
                packed_seq_params=None,
                cu_seqlens_padded=None,
                shared_prefix=shared_metadata,
                shared_prefix_train_mode=True,
            )
            with patch(
                "nemo_rl.models.megatron.train.shared_prefix_next_token_logprobs",
                side_effect=capture_scalar_router,
            ):
                shared_restored, shared_loss_fn = forward_with_post_processing_fn(
                    iter((processed,)),
                    model,
                    processor,
                    defer_fp32_logits=True,
                    global_valid_seqs=torch.tensor(loss_data.size, device=device),
                    global_valid_toks=torch.count_nonzero(loss_weights),
                )
            assert len(captured_logits) == 1
            shared_logits = captured_logits[0]
            assert shared_logits.shape == (
                1,
                shared_physical_length // cp_size,
                local_vocab_size,
            )
            assert shared_logits.dtype == torch.bfloat16
            shared_prescaled_loss, _ = shared_loss_fn(shared_restored)
            shared_loss = shared_prescaled_loss * cp_size / 3
            shared_loss.backward()

        shared_parameter_grads = _all_reduced_parameter_grads(
            model,
            tp_group,
            cp_group,
        )
        shared_d_formula, shared_d_absolute_sum = reconstruct_mamba_d_vjp(
            shared_scan_records
        )
        assert dense_parameter_grads.keys() == shared_parameter_grads.keys()

        shared_cp_indices = get_shared_prefix_context_parallel_indices(
            shared_physical_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
            device=device,
        )
        shared_local_predictor = torch.nonzero(
            shared_cp_indices == last_prompt_position,
            as_tuple=False,
        ).flatten()
        shared_last_prompt_vjp = torch.zeros_like(dense_last_prompt_vjp)
        if shared_local_predictor.numel():
            shared_last_prompt_vjp = shared_logits.grad[
                0, int(shared_local_predictor.item())
            ]
        if cp_size > 1:
            dist.all_reduce(shared_last_prompt_vjp, group=cp_group)

        loss_stats = {
            "dense": dense_loss.detach().item(),
            "shared": shared_loss.detach().item(),
            "absolute": abs((shared_loss.detach() - dense_loss.detach()).item()),
        }
        loss_stats["relative"] = loss_stats["absolute"] / max(
            abs(loss_stats["dense"]),
            torch.finfo(torch.float64).tiny,
        )
        restored_stats = _gradient_stats(dense_restored, shared_restored)
        predictor_stats = _gradient_stats(
            dense_last_prompt_vjp,
            shared_last_prompt_vjp,
        )
        parameter_stats = {
            name: _gradient_stats(
                dense_parameter_grads[name],
                shared_parameter_grads[name],
            )
            for name in dense_parameter_grads
        }
        d_parameter_name = next(
            name for name in parameter_stats if name.endswith(".mixer.D")
        )
        d_diagnostics = {
            "dense_formula_vs_autograd": _max_conditioned_error(
                dense_d_formula,
                dense_parameter_grads[d_parameter_name],
                dense_d_absolute_sum,
            ),
            "shared_formula_vs_autograd": _max_conditioned_error(
                shared_d_formula,
                shared_parameter_grads[d_parameter_name],
                shared_d_absolute_sum,
            ),
            "dense_vs_shared_formula": _max_conditioned_error(
                dense_d_formula,
                shared_d_formula,
                dense_d_absolute_sum,
            ),
        }
        if tp_rank == 0 and cp_rank == 0:
            print(
                "SHARED_PREFIX_TP_SP_HYBRID_NEMO_DIAGNOSTICS "
                f"TP={tp_size} CP={cp_size} loss={loss_stats} "
                f"restored={restored_stats} predictor={predictor_stats} "
                f"d={d_diagnostics} parameters={parameter_stats}",
                flush=True,
            )

        assert loss_stats["relative"] < 0.03, loss_stats
        _assert_gradient_parity(restored_stats, label="restored selected logprobs")
        _assert_gradient_parity(
            predictor_stats,
            label="summed last-prompt predictor VJP",
        )
        for name, stats in parameter_stats.items():
            if name == d_parameter_name:
                assert (
                    d_diagnostics["dense_formula_vs_autograd"]
                    < torch.finfo(torch.float32).eps
                ), d_diagnostics
                assert (
                    d_diagnostics["shared_formula_vs_autograd"]
                    < torch.finfo(torch.float32).eps
                ), d_diagnostics
                assert (
                    d_diagnostics["dense_vs_shared_formula"]
                    < torch.finfo(torch.bfloat16).eps
                ), d_diagnostics
                assert stats["cosine"] > 0.995, stats
                continue
            _assert_gradient_parity(stats, label=f"parameter gradient {name}")
    finally:
        torch.cuda.synchronize(device)
        parallel_state.destroy_model_parallel()


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("cp_size", "world_size"), [(1, 2), (2, 4)])
def test_tp2_sp_real_hybrid_model_to_nemo_scalar_loss_gradient_parity(
    distributed_test_runner,
    cp_size: int,
    world_size: int,
) -> None:
    """Exercise candidate TP/SP Hybrid-output-to-NeMo-loss contracts."""
    from megatron.core.models.hybrid.shared_prefix import (
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
    )

    from nemo_rl.models.megatron.setup import (
        SUPPORTED_SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SUPPORTED_SHARED_PREFIX_TP_CP_SP_TRAINING_CAPABILITY,
        SUPPORTED_SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
    )

    assert (
        SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY
    )
    assert (
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_TP_CP_SP_TRAINING_CAPABILITY
    )
    assert (
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix TP/SP integration requires CUDA BF16 support")
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    distributed_test_runner(
        partial(_run_tp_sp_hybrid_to_nemo_loss_parity, cp_size=cp_size),
        world_size=world_size,
        backend="nccl",
    )
