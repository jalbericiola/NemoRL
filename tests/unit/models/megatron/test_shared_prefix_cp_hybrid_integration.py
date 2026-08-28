"""CP2 integration parity from a real HybridModel through NeMo's scalar loss path."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
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


class _WeightedLogprobLoss:
    """Small deterministic LOGPROB loss that exposes every selected-token VJP."""

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
        return loss, {"weighted_logprob_loss": loss.detach()}


def _pad_sequence_for_cp(tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    multiple = 2 * cp_size
    padding = (-tensor.shape[0]) % multiple
    if not padding:
        return tensor
    return torch.cat(
        (tensor, tensor.new_zeros(padding, *tensor.shape[1:])),
        dim=0,
    )


def _gather_canonical(
    tensor: torch.Tensor, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather detached standard-zigzag shards for post-backward assertions."""
    from megatron.core.models.hybrid.shared_prefix_fused import _undo_cp_zigzag

    gathered = [torch.empty_like(tensor) for _ in range(cp_group.size())]
    dist.all_gather(gathered, tensor, group=cp_group)
    return _undo_cp_zigzag(torch.cat(gathered, dim=0), cp_group.size())


def _all_reduced_parameter_grads(
    model: torch.nn.Module, cp_group: dist.ProcessGroup
) -> dict[str, torch.Tensor]:
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().clone()
        dist.all_reduce(gradient, group=cp_group)
        gradients[name] = gradient
    return gradients


def _scheduled_loss(
    restored_logprobs: torch.Tensor,
    *,
    data: BatchedDataDict[Any],
    cp_size: int,
    num_microbatches: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run LossPostProcessor plus an algebraic emulation of MCore schedule scaling."""
    from nemo_rl.models.megatron.train import LossPostProcessor

    processor = LossPostProcessor(
        loss_fn=_WeightedLogprobLoss(),
        cfg={"sequence_packing": {"enabled": True}},
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
    prescaled_loss, metrics = loss_fn(restored_logprobs)
    # Algebraically emulate the factor that
    # megatron.core.pipeline_parallel.schedules applies before backward.
    scheduled_loss = prescaled_loss * cp_size / num_microbatches
    return scheduled_loss, metrics


def _gradient_stats(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, float]:
    reference_fp64 = reference.detach().double().flatten()
    actual_fp64 = actual.detach().double().flatten()
    difference = reference_fp64 - actual_fp64
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


def _max_elementwise_conditioned_error(
    reference: torch.Tensor,
    actual: torch.Tensor,
    absolute_sum: torch.Tensor,
) -> float:
    """Scale each error by the uncancelled magnitude of its own reduction."""
    return (
        (
            (reference.detach().double() - actual.detach().double()).abs()
            / absolute_sum.detach().double().clamp_min(1e-300)
        )
        .max()
        .item()
    )


def _assert_gradient_stats(
    stats: dict[str, float],
    *,
    label: str,
    relative_tolerance: float = 0.03,
    cosine_tolerance: float = 0.995,
    near_zero_reference_norm: float = 1e-10,
    near_zero_absolute_tolerance: float = 1e-6,
) -> None:
    if stats["reference_norm"] <= near_zero_reference_norm:
        assert stats["absolute_max"] <= near_zero_absolute_tolerance, (
            f"{label}: near-zero reference exceeded absolute tolerance: {stats}"
        )
        return
    assert stats["relative_l2"] < relative_tolerance, f"{label}: {stats}"
    assert stats["cosine"] > cosine_tolerance, f"{label}: {stats}"


def _run_cp2_hybrid_to_nemo_loss_parity(rank: int, world_size: int) -> None:
    # Defer the optional MCore/CUDA model stack so ordinary test collection
    # remains available in CPU-only NeMo development environments.
    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_allocation import (
        validate_segment_layers,
    )
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.hybrid import shared_prefix as mcore_shared_prefix
    from megatron.core.models.hybrid.shared_prefix import (
        SharedPrefixLayout,
        _forward_mamba_layer_shared_prefix_cp_replay,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm import mamba_mixer as mamba_mixer_module
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from nemo_rl.data.packing import (
        build_shared_prefix_tensor_plan,
        get_shared_prefix_context_parallel_indices,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor
    from nemo_rl.models.megatron.data import SharedPrefixForwardMetadata
    from nemo_rl.models.megatron.train import shared_prefix_next_token_logprobs

    assert world_size == 2
    device = torch.device("cuda", rank)
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = cp_group.rank()
        assert cp_rank == rank

        # Match the MCore promotion geometry: two branches share twelve prompt
        # tokens, have distinct first completion targets, and form a 31-token
        # logical star with exactly one trailing token of CP2 physical padding.
        prompt_tokens = torch.arange(10, 22, dtype=torch.long)
        first_completion = torch.arange(30, 38, dtype=torch.long)
        second_completion = torch.arange(50, 61, dtype=torch.long)
        source_input_ids = torch.stack(
            (
                F.pad(
                    torch.cat((prompt_tokens, first_completion)),
                    (0, second_completion.numel() - first_completion.numel()),
                ),
                torch.cat((prompt_tokens, second_completion)),
            )
        )
        source_lengths = torch.tensor((20, 23), dtype=torch.long)
        tensor_bin = build_shared_prefix_tensor_plan(
            input_ids=source_input_ids,
            input_lengths=source_lengths,
            prompt_lengths=torch.tensor((12, 12), dtype=torch.long),
            group_ids=("g", "g"),
            bin_capacity=64,
            materialize_attention_mask=False,
        ).shared_bins[0]
        layout = tensor_bin.layout
        mcore_layout = SharedPrefixLayout(
            prefix_len=layout.prompt_length,
            completion_lens=layout.completion_lengths,
        )
        assert mcore_layout.total_len == 31
        assert source_input_ids[0, 12] != source_input_ids[1, 12]

        hidden_size = 256
        vocab_size = 128
        pattern = "M*"
        model_parallel_cuda_manual_seed(20260827)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=len(validate_segment_layers(pattern)),
            num_attention_heads=4,
            num_query_groups=1,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_mamba_mem_eff_path=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            context_parallel_size=world_size,
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "pp", "cp", "embd", "dp_cp"]
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=32,
            hybrid_layer_pattern=pattern,
            position_embedding_type="rope",
            pre_process=True,
            post_process=True,
            parallel_output=False,
            pg_collection=process_groups,
        ).to(device=device)
        model.train()
        source_rank = dist.get_global_rank(cp_group, 0)
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=source_rank, group=cp_group)

        mamba_mixer = model.decoder.layers[0].mixer
        scan_impl = mamba_mixer_module.mamba_chunk_scan_combined
        assert mcore_shared_prefix.mamba_chunk_scan_combined is scan_impl

        @contextmanager
        def capture_mamba_scans(
            records: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> Any:
            """Capture scan inputs and output VJPs for an independent FP64 D formula."""

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
            """Return FP64 analytic D VJP and its elementwise cancellation scale."""
            local_vjp = None
            local_absolute_sum = None
            for scan_input, scan_output in records:
                if scan_output.grad is None:
                    raise RuntimeError(
                        "captured Mamba scan output did not receive a gradient"
                    )
                # D is the scan's direct skip: dD = sum(dy * x). This formula
                # deliberately does not read D.grad, so it independently checks
                # production autograd while retaining FP64 reduction precision.
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
            if local_vjp.numel() * world_size != mamba_mixer.D.numel():
                raise RuntimeError(
                    "Mamba CP scan shard does not cover the full D parameter: "
                    f"local={local_vjp.numel()}, CP={world_size}, "
                    f"D={mamba_mixer.D.numel()}"
                )

            full_vjp = torch.zeros_like(mamba_mixer.D, dtype=torch.float64)
            full_absolute_sum = torch.zeros_like(
                mamba_mixer.D,
                dtype=torch.float64,
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
        loss_weights = torch.zeros(
            len(row_indices),
            source_input_ids.shape[1] - 1,
            dtype=torch.float32,
            device=device,
        )
        for row, source_length in enumerate(local_source_lengths):
            first_completion_predictor = layout.prompt_length - 1
            completion_target_count = source_length - layout.prompt_length
            # Match RL training: prompt targets are masked, while the final
            # prompt position predicts the first completion token and every
            # remaining completion target contributes to the policy loss.
            loss_weights[row, first_completion_predictor : source_length - 1] = (
                torch.arange(
                    1,
                    completion_target_count + 1,
                    device=device,
                    dtype=torch.float32,
                )
            ) * (1.0 + 0.25 * row)
        loss_data = BatchedDataDict(
            {"input_ids": local_source_ids, "loss_weights": loss_weights}
        )

        generator = torch.Generator(device=device).manual_seed(17)
        global_prefix = torch.randn(
            layout.prompt_length,
            1,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        global_completions = [
            torch.randn(
                completion_length,
                1,
                hidden_size,
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            for completion_length in layout.completion_lengths
        ]

        def localize(global_tensor: torch.Tensor) -> torch.Tensor:
            indices = get_shared_prefix_context_parallel_indices(
                global_tensor.shape[0],
                cp_rank=cp_rank,
                cp_size=world_size,
                device=device,
            )
            return (
                global_tensor.index_select(0, indices)
                .detach()
                .clone()
                .requires_grad_(True)
            )

        # Conventional dense oracle: one CP-sharded full sequence per branch,
        # followed by NeMo's existing differentiable CP logit gather.
        dense_inputs = []
        dense_local_logits = []
        dense_rows = []
        dense_scan_records: list[tuple[torch.Tensor, torch.Tensor]] = []
        with capture_mamba_scans(dense_scan_records):
            for row, (completion, source_length) in enumerate(
                zip(global_completions, local_source_lengths, strict=True)
            ):
                global_dense_input = _pad_sequence_for_cp(
                    torch.cat((global_prefix, completion), dim=0), world_size
                )
                local_dense_input = localize(global_dense_input)
                local_logits = model(
                    input_ids=None,
                    position_ids=None,
                    attention_mask=None,
                    decoder_input=local_dense_input,
                )
                local_logits.retain_grad()
                global_logits = allgather_cp_sharded_tensor(
                    local_logits, cp_group, seq_dim=1
                )
                real_logits = global_logits[:, :source_length]
                target_tokens = local_source_ids[row, 1:source_length]
                row_logprobs = (
                    F.log_softmax(real_logits[:, :-1].float(), dim=-1)
                    .gather(
                        dim=-1,
                        index=target_tokens.view(1, -1, 1),
                    )
                    .squeeze(0)
                    .squeeze(-1)
                )
                dense_rows.append(
                    F.pad(
                        row_logprobs,
                        (0, source_input_ids.shape[1] - source_length),
                    )
                )
                dense_inputs.append(local_dense_input)
                dense_local_logits.append(local_logits)
        dense_restored = torch.stack(dense_rows, dim=0)
        dense_loss, _ = _scheduled_loss(
            dense_restored,
            data=loss_data,
            cp_size=world_size,
            num_microbatches=3,
        )
        dense_unscaled_loss = -(dense_restored * loss_weights).sum()
        torch.testing.assert_close(
            dense_loss.detach() * world_size,
            dense_unscaled_loss.detach(),
            rtol=1e-6,
            atol=1e-6,
        )
        dense_loss.backward()
        dense_input_grads = [
            _gather_canonical(value.grad, cp_group) for value in dense_inputs
        ]
        dense_logit_grads = [
            _gather_canonical(value.grad[0], cp_group) for value in dense_local_logits
        ]
        dense_parameter_grads = _all_reduced_parameter_grads(model, cp_group)
        dense_d_formula, dense_d_absolute_sum = reconstruct_mamba_d_vjp(
            dense_scan_records
        )
        d_parameter_name = next(
            name for name in dense_parameter_grads if name.endswith(".mixer.D")
        )
        assert dense_parameter_grads[d_parameter_name].dtype == torch.float32

        global_shared_input = _pad_sequence_for_cp(
            torch.cat((global_prefix, *global_completions), dim=0), world_size
        )
        assert global_shared_input.shape[0] == 32

        def reject_logit_gather(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "shared-prefix CP must exchange selected scalars, not vocabulary logits"
            )

        dense_prefix_grad = sum(
            gradient[: layout.prompt_length] for gradient in dense_input_grads
        )
        last_prompt_predictor = layout.prompt_length - 1
        dense_last_prompt_vjp = sum(
            gradient[last_prompt_predictor] for gradient in dense_logit_grads
        )
        local_shared_indices = get_shared_prefix_context_parallel_indices(
            global_shared_input.shape[0],
            cp_rank=cp_rank,
            cp_size=world_size,
            device=device,
        )
        local_predictor = torch.nonzero(
            local_shared_indices == last_prompt_predictor, as_tuple=False
        ).flatten()

        def run_shared(
            mamba_impl: Any = None,
        ) -> dict[str, Any]:
            model.zero_grad(set_to_none=True)
            shared_input = localize(global_shared_input)
            scan_records: list[tuple[torch.Tensor, torch.Tensor]] = []
            mamba_context = (
                nullcontext()
                if mamba_impl is None
                else patch.object(
                    mcore_shared_prefix,
                    "_forward_mamba_layer_shared_prefix_cp",
                    mamba_impl,
                )
            )
            # The shared model uses only sequence/head all-to-alls, and the NeMo
            # adapter uses one differentiable scalar all-reduce. Any all-gather
            # in this scope would regress the vocabulary-memory contract.
            with (
                mamba_context,
                capture_mamba_scans(scan_records),
                patch("torch.distributed.all_gather", side_effect=reject_logit_gather),
                patch(
                    "torch.distributed.all_gather_into_tensor",
                    side_effect=reject_logit_gather,
                ),
                patch(
                    "torch.distributed.nn.functional.all_gather",
                    side_effect=reject_logit_gather,
                ),
                patch(
                    "nemo_rl.models.megatron.train.allgather_cp_sharded_tensor",
                    side_effect=reject_logit_gather,
                ),
            ):
                shared_local_logits = model(
                    input_ids=None,
                    position_ids=None,
                    attention_mask=None,
                    decoder_input=shared_input,
                    shared_prefix_layout=mcore_layout,
                )
                shared_local_logits.retain_grad()
                shared_restored = shared_prefix_next_token_logprobs(
                    shared_local_logits,
                    SharedPrefixForwardMetadata(
                        tensor_bin=tensor_bin,
                        source_sequence_length=source_input_ids.shape[1],
                        cp_rank=cp_rank,
                        cp_size=world_size,
                        padded_total_length=global_shared_input.shape[0],
                    ),
                    chunk_size=2,
                )
                shared_loss, _ = _scheduled_loss(
                    shared_restored,
                    data=loss_data,
                    cp_size=world_size,
                    num_microbatches=3,
                )
                shared_unscaled_loss = -(shared_restored * loss_weights).sum()
                shared_loss.backward()

            d_formula, d_absolute_sum = reconstruct_mamba_d_vjp(scan_records)
            parameter_grads = _all_reduced_parameter_grads(model, cp_group)
            assert parameter_grads[d_parameter_name].dtype == torch.float32

            return {
                "loss": shared_loss.detach(),
                "unscaled_loss": shared_unscaled_loss.detach(),
                "input_grad": _gather_canonical(shared_input.grad, cp_group),
                "logit_grad": _gather_canonical(shared_local_logits.grad[0], cp_group),
                "local_logits": shared_local_logits.detach(),
                "local_logit_grad": shared_local_logits.grad.detach(),
                "parameter_grads": parameter_grads,
                "d_formula": d_formula,
                "d_absolute_sum": d_absolute_sum,
            }

        # The named replay implementation is an uninterrupted correctness
        # baseline. The advertised production default uses the optimized
        # state-fork and must independently pass the same contract.
        shared_cases = {
            "replay": run_shared(_forward_mamba_layer_shared_prefix_cp_replay),
            "production-default": run_shared(),
        }
        case_diagnostics: dict[str, dict[str, Any]] = {}
        for case_name, case in shared_cases.items():
            shared_input_grad = case["input_grad"]
            shared_logit_grad = case["logit_grad"]
            shared_parameter_grads = case["parameter_grads"]
            assert dense_parameter_grads.keys() == shared_parameter_grads.keys()

            completion_stats = {}
            for branch, (dense_gradient, completion_slice) in enumerate(
                zip(
                    dense_input_grads,
                    mcore_layout.completion_slices(),
                    strict=True,
                )
            ):
                completion_length = completion_slice.stop - completion_slice.start
                completion_stats[f"branch_{branch}"] = _gradient_stats(
                    dense_gradient[
                        layout.prompt_length : layout.prompt_length + completion_length
                    ],
                    shared_input_grad[completion_slice],
                )

            parameter_stats = {
                name: _gradient_stats(
                    dense_parameter_grads[name], shared_parameter_grads[name]
                )
                for name in dense_parameter_grads
            }
            shared_d_formula = case["d_formula"]
            shared_d_absolute_sum = case["d_absolute_sum"]
            d_formula_diagnostics = {
                "all_finite": all(
                    torch.isfinite(value).all().item()
                    for value in (
                        dense_parameter_grads[d_parameter_name],
                        shared_parameter_grads[d_parameter_name],
                        dense_d_formula,
                        shared_d_formula,
                        dense_d_absolute_sum,
                        shared_d_absolute_sum,
                    )
                ),
                "dense_formula_vs_autograd": _gradient_stats(
                    dense_d_formula,
                    dense_parameter_grads[d_parameter_name],
                ),
                "shared_formula_vs_autograd": _gradient_stats(
                    shared_d_formula,
                    shared_parameter_grads[d_parameter_name],
                ),
                "dense_vs_shared_formula": _gradient_stats(
                    dense_d_formula,
                    shared_d_formula,
                ),
                "dense_formula_vs_autograd_max_elementwise_conditioned": (
                    _max_elementwise_conditioned_error(
                        dense_d_formula,
                        dense_parameter_grads[d_parameter_name],
                        dense_d_absolute_sum,
                    )
                ),
                "shared_formula_vs_autograd_max_elementwise_conditioned": (
                    _max_elementwise_conditioned_error(
                        shared_d_formula,
                        shared_parameter_grads[d_parameter_name],
                        shared_d_absolute_sum,
                    )
                ),
                "dense_vs_shared_formula_max_elementwise_conditioned": (
                    _max_elementwise_conditioned_error(
                        dense_d_formula,
                        shared_d_formula,
                        dense_d_absolute_sum,
                    )
                ),
                "dense_sum_abs_dy_x": dense_d_absolute_sum.sum().item(),
                "shared_sum_abs_dy_x": shared_d_absolute_sum.sum().item(),
            }
            loss_absolute = abs((case["loss"] - dense_loss.detach()).item())
            loss_relative = loss_absolute / max(
                abs(dense_loss.detach().item()), torch.finfo(torch.float64).tiny
            )
            schedule_absolute = abs(
                (case["loss"] * world_size - case["unscaled_loss"]).item()
            )
            schedule_relative = schedule_absolute / max(
                abs(case["unscaled_loss"].item()),
                torch.finfo(torch.float64).tiny,
            )
            analytic_vjp_stats = None
            if local_predictor.numel():
                local_offset = int(local_predictor.item())
                logits = case["local_logits"][0, local_offset].float()
                expected_vjp = (
                    logits.softmax(dim=-1)
                    * loss_weights[:, last_prompt_predictor].sum()
                )
                for row in range(len(row_indices)):
                    target = int(local_source_ids[row, layout.prompt_length].item())
                    expected_vjp[target] -= loss_weights[row, last_prompt_predictor]
                # Autograd casts the FP32 log-softmax VJP back to the BF16 output.
                expected_vjp = expected_vjp.to(case["local_logits"].dtype).float()
                analytic_vjp_stats = _gradient_stats(
                    expected_vjp,
                    case["local_logit_grad"][0, local_offset].float(),
                )

            case_diagnostics[case_name] = {
                "loss": {
                    "dense": dense_loss.detach().item(),
                    "shared": case["loss"].item(),
                    "absolute": loss_absolute,
                    "relative": loss_relative,
                },
                "schedule_scaling": {
                    "absolute": schedule_absolute,
                    "relative": schedule_relative,
                },
                "prompt_input": _gradient_stats(
                    dense_prefix_grad,
                    shared_input_grad[: layout.prompt_length],
                ),
                "completion_inputs": completion_stats,
                "padding_input_absolute_max": shared_input_grad[
                    mcore_layout.total_len :
                ]
                .abs()
                .max()
                .item(),
                "last_prompt_predictor": _gradient_stats(
                    dense_last_prompt_vjp,
                    shared_logit_grad[last_prompt_predictor],
                ),
                "analytic_last_prompt_predictor": analytic_vjp_stats,
                "parameters": parameter_stats,
                "mamba_d_formula": d_formula_diagnostics,
            }

        for case_name, diagnostics in case_diagnostics.items():
            parameter_outliers = {
                name: stats
                for name, stats in diagnostics["parameters"].items()
                if (stats["reference_norm"] <= 1e-10 and stats["absolute_max"] > 1e-6)
                or (
                    stats["reference_norm"] > 1e-10
                    and (stats["relative_l2"] >= 0.03 or stats["cosine"] <= 0.995)
                )
            }
            if rank == 0:
                print(
                    "SHARED_PREFIX_CP2_HYBRID_NEMO_DIAGNOSTICS "
                    f"case={case_name} summary="
                    f"{ {key: value for key, value in diagnostics.items() if key not in ('parameters', 'mamba_d_formula')} }",
                    flush=True,
                )
                print(
                    "SHARED_PREFIX_CP2_HYBRID_NEMO_PARAMETER_STATS "
                    f"case={case_name} parameters={diagnostics['parameters']}",
                    flush=True,
                )
                print(
                    "SHARED_PREFIX_CP2_HYBRID_NEMO_PARAMETER_OUTLIERS "
                    f"case={case_name} outliers={parameter_outliers}",
                    flush=True,
                )
                print(
                    "SHARED_PREFIX_CP2_HYBRID_NEMO_MAMBA_D_FORMULA "
                    f"case={case_name} diagnostics="
                    f"{diagnostics['mamba_d_formula']}",
                    flush=True,
                )
            if diagnostics["analytic_last_prompt_predictor"] is not None:
                print(
                    "SHARED_PREFIX_CP2_HYBRID_NEMO_ANALYTIC_VJP "
                    f"rank={rank} case={case_name} "
                    f"stats={diagnostics['analytic_last_prompt_predictor']}",
                    flush=True,
                )

        # Assert only after both implementations have emitted complete diagnostics.
        for case_name in ("replay", "production-default"):
            diagnostics = case_diagnostics[case_name]
            assert diagnostics["loss"]["relative"] < 0.03, (
                f"{case_name} forward loss: {diagnostics['loss']}"
            )
            assert diagnostics["schedule_scaling"]["relative"] < 1e-6, (
                f"{case_name} schedule scaling: {diagnostics['schedule_scaling']}"
            )
            _assert_gradient_stats(
                diagnostics["prompt_input"],
                label=f"{case_name} shared prompt input gradient",
            )
            for branch, stats in diagnostics["completion_inputs"].items():
                _assert_gradient_stats(
                    stats,
                    label=f"{case_name} completion input gradient {branch}",
                )
            assert diagnostics["padding_input_absolute_max"] <= 1e-6, (
                f"{case_name} padding input gradient: "
                f"{diagnostics['padding_input_absolute_max']}"
            )
            _assert_gradient_stats(
                diagnostics["last_prompt_predictor"],
                label=f"{case_name} last-prompt shared predictor VJP",
            )
            if diagnostics["analytic_last_prompt_predictor"] is not None:
                _assert_gradient_stats(
                    diagnostics["analytic_last_prompt_predictor"],
                    label=f"{case_name} analytic last-prompt predictor VJP",
                )
            for name, stats in diagnostics["parameters"].items():
                if name == d_parameter_name:
                    d_diagnostics = diagnostics["mamba_d_formula"]
                    assert d_diagnostics["all_finite"], (
                        f"{case_name} Mamba D formula contains non-finite values: "
                        f"{d_diagnostics}"
                    )
                    # D is a cancellation-heavy direct-skip reduction. Its
                    # near-zero final norm can make an ordinary relative L2
                    # ratio large even when each accumulation is accurate.
                    # Independently validate the real FP32 autograd result
                    # against the analytic FP64 sum(dy*x). The dense and
                    # shared paths each contain an independent BF16 rounding,
                    # so their pairwise condition-aware bound is 2u=epsilon.
                    assert (
                        d_diagnostics[
                            "dense_formula_vs_autograd_max_elementwise_conditioned"
                        ]
                        < torch.finfo(torch.float32).eps
                    ), f"{case_name} dense Mamba D autograd: {d_diagnostics}"
                    assert (
                        d_diagnostics[
                            "shared_formula_vs_autograd_max_elementwise_conditioned"
                        ]
                        < torch.finfo(torch.float32).eps
                    ), f"{case_name} shared Mamba D autograd: {d_diagnostics}"
                    assert (
                        d_diagnostics[
                            "dense_vs_shared_formula_max_elementwise_conditioned"
                        ]
                        < torch.finfo(torch.bfloat16).eps
                    ), f"{case_name} dense/shared Mamba D formula: {d_diagnostics}"
                    assert stats["cosine"] > 0.995, (
                        f"{case_name} parameter gradient {name}: {stats}"
                    )
                    continue
                _assert_gradient_stats(
                    stats,
                    label=f"{case_name} parameter gradient {name}",
                )
    finally:
        torch.cuda.synchronize(device)
        parallel_state.destroy_model_parallel()


@pytest.mark.timeout(420)
def test_cp2_real_hybrid_model_to_nemo_scalar_loss_gradient_parity(
    distributed_test_runner,
) -> None:
    """Exercise the advertised CP Hybrid-output-to-NeMo-loss contract."""
    from megatron.core.models.hybrid.shared_prefix import (
        SHARED_PREFIX_CP_TRAINING_CAPABILITY,
        SHARED_PREFIX_TRAINING_CAPABILITIES,
    )

    assert SHARED_PREFIX_CP_TRAINING_CAPABILITY in SHARED_PREFIX_TRAINING_CAPABILITIES
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("shared-prefix CP2 integration requires CUDA BF16 support")
    pytest.importorskip("causal_conv1d")
    pytest.importorskip("flash_attn")
    pytest.importorskip("mamba_ssm")
    distributed_test_runner(
        _run_cp2_hybrid_to_nemo_loss_parity,
        world_size=2,
        backend="nccl",
    )
