"""WORLD16 RLVR41 shared-prefix Hybrid-to-NeMo integration gate."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType, ModuleType
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType

pytestmark = pytest.mark.mcore

_WORLD_SIZE = 16
_TP_SIZE = 4
_CP_SIZE = 4
_EP_SIZE = 16
_ETP_SIZE = 1
_PADDING_MULTIPLE = 32
_PREFIX = (10, 11, 12, 13, 14)
_LOGICAL_COMPLETION_LENGTHS = tuple(range(1, 17))
_STAR_PHYSICAL_LENGTH = 448
_GATE_ENV = "NEMORL_SHARED_PREFIX_RLVR41_GATE"


class _CompletionLogprobLoss:
    """Completion-only objective with distinct branch/token weights."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __call__(
        self,
        *,
        next_token_logprobs: torch.Tensor,
        data: Any,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del global_valid_seqs, global_valid_toks
        loss = -(next_token_logprobs * data["loss_weights"]).sum()
        return loss, {"completion_logprob_loss": loss.detach()}


def _load_mcore_test_module(relative_path: str, module_name: str) -> ModuleType:
    """Load an MCore test helper without colliding with NeMo's ``tests`` package."""
    import megatron.core

    if megatron.core.__file__ is None:
        raise RuntimeError("cannot resolve the installed megatron.core package")
    mcore_root = Path(megatron.core.__file__).resolve().parents[2]
    module_path = mcore_root / relative_path
    if not module_path.is_file():
        raise RuntimeError(f"required MCore test helper is missing: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MCore test helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_gate_condition(condition: bool, reason: str, *, gate_mode: bool) -> None:
    """Skip exploratory collection but fail closed in the authoritative gate."""
    if condition:
        return
    if gate_mode:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _require_gate_dependency(module_name: str, *, gate_mode: bool) -> None:
    """Require an optional CUDA dependency without allowing gate-mode skips."""
    try:
        __import__(module_name)
    except ImportError as error:
        reason = f"RLVR41 NeMo integration requires {module_name}: {error}"
        if gate_mode:
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)


def _install_finalize_probe(model: torch.nn.Module) -> None:
    """Give the raw PP1 fixture the DDP surface used by the real finalizer."""
    from megatron.core.distributed import DistributedDataParallelConfig

    model.ddp_config = DistributedDataParallelConfig()
    model._shared_prefix_finish_grad_sync_calls = 0

    def finish_grad_sync(this: torch.nn.Module, force_all_reduce: bool = False) -> None:
        del force_all_reduce
        this._shared_prefix_finish_grad_sync_calls += 1

    model.finish_grad_sync = MethodType(finish_grad_sync, model)


def _build_source_tensor_bin() -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Build the exact P5/K16/M32 NeMo layout used by the MCore fixture."""
    from nemo_rl.data.packing import (
        SharedPrefixRow,
        build_shared_prefix_layout,
        materialize_shared_prefix_layout,
    )

    source_width = len(_PREFIX) + max(_LOGICAL_COMPLETION_LENGTHS)
    input_ids = torch.zeros(
        len(_LOGICAL_COMPLETION_LENGTHS),
        source_width,
        dtype=torch.long,
    )
    input_lengths = []
    rows = []
    for row_index, completion_length in enumerate(_LOGICAL_COMPLETION_LENGTHS):
        input_ids[row_index, : len(_PREFIX)] = torch.tensor(_PREFIX)
        # The first completion targets cover all four 128-token TP vocabulary
        # shards; all ids remain inside the fixture's padded vocabulary of 512.
        completion_start = 16 + 30 * row_index
        input_ids[
            row_index,
            len(_PREFIX) : len(_PREFIX) + completion_length,
        ] = torch.arange(completion_start, completion_start + completion_length)
        input_lengths.append(len(_PREFIX) + completion_length)
        rows.append(
            SharedPrefixRow(
                row_index=row_index,
                group_id="rlvr41",
                prompt_token_ids=_PREFIX,
                completion_length=completion_length,
            )
        )

    layout = build_shared_prefix_layout(
        rows,
        sequence_length_pad_multiple=_PADDING_MULTIPLE,
    )
    tensor_bin = materialize_shared_prefix_layout(
        input_ids,
        input_lengths=torch.tensor(input_lengths, dtype=torch.long),
        layout=layout,
        materialize_attention_mask=False,
    )
    return input_ids, torch.tensor(input_lengths, dtype=torch.long), tensor_bin


def _completion_loss_weights(
    input_lengths: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Weight only logical completion targets, including every first token."""
    weights = torch.zeros(
        input_lengths.numel(),
        int(input_lengths.max().item()) - 1,
        dtype=torch.float32,
        device=device,
    )
    first_predictor = len(_PREFIX) - 1
    for row_index, completion_length in enumerate(_LOGICAL_COMPLETION_LENGTHS):
        weights[
            row_index,
            first_predictor : first_predictor + completion_length,
        ] = torch.arange(
            1,
            completion_length + 1,
            dtype=torch.float32,
            device=device,
        ) * (1.0 + row_index / len(_LOGICAL_COMPLETION_LENGTHS))
    return weights


def _expected_last_prompt_vjp(
    prompt_logits: torch.Tensor,
    *,
    targets: torch.Tensor,
    weights: torch.Tensor,
    tp_group: dist.ProcessGroup,
    tp_rank: int,
) -> torch.Tensor:
    """Compute the summed first-completion VJP with scalar TP reductions."""
    local_logits = prompt_logits.detach().float()
    global_max = local_logits.max()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    local_exp = torch.exp(local_logits - global_max)
    global_denominator = local_exp.sum()
    dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM, group=tp_group)
    expected = weights.sum() * (local_exp / global_denominator)

    local_vocab_size = prompt_logits.numel()
    vocab_start = tp_rank * local_vocab_size
    vocab_end = vocab_start + local_vocab_size
    for target, weight in zip(targets.tolist(), weights.tolist(), strict=True):
        if vocab_start <= target < vocab_end:
            expected[target - vocab_start] -= weight
    return expected


@pytest.mark.timeout(1800)
def test_rlvr41_world16_shared_prefix_reaches_nemo_loss_and_finalizes_once() -> None:
    """Conjoin TP4/CP4/EP16/SP/M32/recompute with NeMo scalar loss routing."""
    gate_mode = os.environ.get(_GATE_ENV) == "1"
    _require_gate_condition(
        int(os.environ.get("WORLD_SIZE", "1")) == _WORLD_SIZE,
        "RLVR41 NeMo integration requires torchrun WORLD_SIZE=16",
        gate_mode=gate_mode,
    )
    _require_gate_condition(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "RLVR41 NeMo integration requires CUDA BF16 support",
        gate_mode=gate_mode,
    )
    for dependency in ("causal_conv1d", "flash_attn", "mamba_ssm"):
        _require_gate_dependency(dependency, gate_mode=gate_mode)

    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.hybrid.shared_prefix import (
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
        SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias

    from nemo_rl.data.packing import shard_shared_prefix_tensor_bin_for_context_parallel
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import train as train_module
    from nemo_rl.models.megatron.data import (
        ProcessedMicrobatch,
        SharedPrefixForwardMetadata,
    )
    from nemo_rl.models.megatron.setup import (
        SUPPORTED_SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SUPPORTED_SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
        SUPPORTED_SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
        SUPPORTED_SHARED_PREFIX_TP_CP_SP_TRAINING_CAPABILITY,
    )
    from nemo_rl.models.megatron.train import (
        LossPostProcessor,
        forward_with_post_processing_fn,
        shared_prefix_next_token_logprobs,
    )

    assert (
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_TP_CP_SP_TRAINING_CAPABILITY
    )
    assert (
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY
    )
    assert (
        SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY
    )
    assert (
        SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY
    )

    fixture = _load_mcore_test_module(
        "tests/unit_tests/ssm/test_hybrid_shared_prefix_rlvr41_distributed.py",
        "_mcore_rlvr41_shared_prefix_fixture_for_nemo",
    )
    test_utilities = _load_mcore_test_module(
        "tests/unit_tests/test_utilities.py",
        "_mcore_test_utilities_for_nemo_rlvr41",
    )
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert global_rank % torch.cuda.device_count() == local_rank
    test_utilities.Utils.world_size = _WORLD_SIZE
    test_utilities.Utils.rank = global_rank

    previous_qk_layer_scaling = os.environ.get("NVTE_APPLY_QK_LAYER_SCALING")
    os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
    test_utilities.Utils.initialize_model_parallel(
        tensor_model_parallel_size=_TP_SIZE,
        pipeline_model_parallel_size=1,
        context_parallel_size=_CP_SIZE,
        expert_model_parallel_size=_EP_SIZE,
        expert_tensor_parallel_size=_ETP_SIZE,
    )
    try:
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = process_groups.tp
        cp_group = process_groups.cp
        assert process_groups.tp_dp_cp.size() == _WORLD_SIZE
        assert parallel_state.get_expert_model_parallel_world_size() == _EP_SIZE
        assert parallel_state.get_expert_tensor_parallel_world_size() == _ETP_SIZE
        tp_rank = tp_group.rank()
        cp_rank = cp_group.rank()
        device = torch.device("cuda", local_rank)

        torch.manual_seed(20260828)
        model_parallel_cuda_manual_seed(20260828)
        model = fixture.build_rlvr41_shared_prefix_model(process_groups)
        model.train()
        assert model.config.recompute_granularity == "full"
        assert model.config.recompute_method == "uniform"
        assert model.config.recompute_num_layers == 1
        router = fixture.freeze_rlvr41_router(model)
        _install_finalize_probe(model)
        initial_expert_bias = router.expert_bias.detach().clone()

        input_ids, input_lengths, tensor_bin = _build_source_tensor_bin()
        layout = tensor_bin.layout
        mcore_layout = fixture.rlvr41_shared_prefix_layout()
        assert layout.prompt_length == mcore_layout.prefix_len == len(_PREFIX)
        assert tuple(layout.completion_lengths) == tuple(
            mcore_layout.logical_completion_lens
        )
        assert tuple(layout.physical_completion_lengths) == tuple(
            mcore_layout.completion_lens
        )
        assert layout.physical_total_length == mcore_layout.total_len == 437

        shared_shard = shard_shared_prefix_tensor_bin_for_context_parallel(
            tensor_bin,
            cp_rank=cp_rank,
            cp_size=_CP_SIZE,
            tp_size=_TP_SIZE,
            padding_multiple=_PADDING_MULTIPLE,
        )
        assert shared_shard.padded_total_length == _STAR_PHYSICAL_LENGTH
        shared_input_ids = shared_shard.packed_input_ids.to(device).unsqueeze(0)
        mcore_positions = mcore_layout.padded_position_ids(
            _STAR_PHYSICAL_LENGTH,
            device,
        ).index_select(0, shared_shard.global_token_indices.to(device))
        torch.testing.assert_close(
            shared_shard.position_ids.to(device),
            mcore_positions,
        )

        source_input_ids = input_ids.to(device)
        loss_weights = _completion_loss_weights(input_lengths, device=device)
        loss_data = BatchedDataDict(
            {"input_ids": source_input_ids, "loss_weights": loss_weights}
        )
        metadata = SharedPrefixForwardMetadata(
            tensor_bin=tensor_bin,
            source_sequence_length=source_input_ids.shape[1],
            padding_multiple=_PADDING_MULTIPLE,
            cp_rank=cp_rank,
            cp_size=_CP_SIZE,
            padded_total_length=_STAR_PHYSICAL_LENGTH,
        )
        processed = ProcessedMicrobatch(
            data_dict=loss_data,
            input_ids=source_input_ids,
            input_ids_cp_sharded=shared_input_ids,
            attention_mask=None,
            position_ids=mcore_positions.unsqueeze(0),
            packed_seq_params=None,
            cu_seqlens_padded=None,
            shared_prefix=metadata,
            shared_prefix_train_mode=True,
        )
        num_microbatches = 3
        processor = LossPostProcessor(
            loss_fn=_CompletionLogprobLoss(),
            cfg={
                "logprob_chunk_size": 2,
                "sequence_packing": {"enabled": True, "fuse_loss": True},
            },
            num_microbatches=num_microbatches,
            cp_normalize=True,
        )

        captured_logits: list[torch.Tensor] = []
        distributed_logprob_calls: list[tuple[torch.Size, torch.Size]] = []
        real_scalar_router = shared_prefix_next_token_logprobs
        real_distributed_logprob = train_module.DistributedLogprob.apply

        def capture_scalar_router(
            logits: torch.Tensor,
            shared_metadata: SharedPrefixForwardMetadata,
            **kwargs: Any,
        ) -> torch.Tensor:
            logits.retain_grad()
            captured_logits.append(logits)
            return real_scalar_router(logits, shared_metadata, **kwargs)

        def capture_distributed_logprob(
            logits: torch.Tensor,
            targets: torch.Tensor,
            *args: Any,
        ) -> torch.Tensor:
            distributed_logprob_calls.append((logits.shape, targets.shape))
            return real_distributed_logprob(logits, targets, *args)

        def reject_vocab_logit_gather(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "shared-prefix NeMo loss must reduce selected scalars, not gather logits"
            )

        router.local_tokens_per_expert.zero_()
        with (
            patch(
                "nemo_rl.models.megatron.train.shared_prefix_next_token_logprobs",
                side_effect=capture_scalar_router,
            ),
            patch.object(
                train_module.DistributedLogprob,
                "apply",
                side_effect=capture_distributed_logprob,
            ),
            patch(
                "nemo_rl.models.megatron.train.allgather_cp_sharded_tensor",
                side_effect=reject_vocab_logit_gather,
            ),
            patch(
                "nemo_rl.models.megatron.train.from_parallel_logits_to_logprobs",
                side_effect=reject_vocab_logit_gather,
            ),
        ):
            restored_logprobs, loss_fn = forward_with_post_processing_fn(
                iter((processed,)),
                model,
                processor,
                defer_fp32_logits=True,
                global_valid_seqs=torch.tensor(loss_data.size, device=device),
                global_valid_toks=torch.count_nonzero(loss_weights),
            )
            prescaled_loss, _ = loss_fn(restored_logprobs)
            # Algebraically apply the multiplier used by the production
            # Megatron schedule; LossPostProcessor already owns CP normalization.
            loss = prescaled_loss * _CP_SIZE / num_microbatches
            loss.backward()

        assert router.weight.grad is None

        assert len(captured_logits) == 1
        logits = captured_logits[0]
        assert logits.shape == (1, _STAR_PHYSICAL_LENGTH // _CP_SIZE, 512 // _TP_SIZE)
        assert logits.dtype == torch.bfloat16
        assert restored_logprobs.shape == (
            len(_LOGICAL_COMPLETION_LENGTHS),
            source_input_ids.shape[1] - 1,
        )
        assert restored_logprobs.dtype == torch.float32
        assert torch.isfinite(restored_logprobs).all()
        assert torch.isfinite(loss)
        assert distributed_logprob_calls
        assert all(
            rows <= 2 and vocab == 512 // _TP_SIZE and targets == rows
            for (rows, vocab), (targets,) in distributed_logprob_calls
        )

        local_global_indices = shared_shard.global_token_indices.to(device)
        physical_padding = tensor_bin.indices.physical_padding_positions.to(device)
        topology_padding = torch.arange(
            layout.physical_total_length,
            _STAR_PHYSICAL_LENGTH,
            device=device,
        )
        invalid_positions = torch.cat((physical_padding, topology_padding))
        invalid_local_rows = torch.isin(local_global_indices, invalid_positions)
        assert bool(invalid_local_rows.any().item())
        assert torch.count_nonzero(logits.grad[0, invalid_local_rows]) == 0

        last_prompt_position = len(_PREFIX) - 1
        local_prompt_row = torch.nonzero(
            local_global_indices == last_prompt_position,
            as_tuple=False,
        ).flatten()
        prompt_logits = torch.zeros(
            logits.shape[-1],
            dtype=logits.dtype,
            device=device,
        )
        prompt_vjp = torch.zeros_like(prompt_logits)
        if local_prompt_row.numel():
            local_row = int(local_prompt_row.item())
            prompt_logits.copy_(logits.detach()[0, local_row])
            prompt_vjp.copy_(logits.grad[0, local_row])
        dist.all_reduce(prompt_logits, group=cp_group)
        dist.all_reduce(prompt_vjp, group=cp_group)
        first_completion_targets = source_input_ids[:, len(_PREFIX)]
        first_completion_weights = loss_weights[:, last_prompt_position]
        expected_prompt_vjp = _expected_last_prompt_vjp(
            prompt_logits,
            targets=first_completion_targets,
            weights=first_completion_weights,
            tp_group=tp_group,
            tp_rank=tp_rank,
        )
        torch.testing.assert_close(
            prompt_vjp.float(),
            expected_prompt_vjp,
            rtol=0.03,
            atol=0.03,
        )

        counts_local = router.local_tokens_per_expert.detach().clone()
        global_counts = counts_local.clone()
        dist.all_reduce(global_counts, group=process_groups.tp_dp_cp)
        expected_counts = torch.zeros_like(global_counts)
        selected_experts = torch.topk(
            initial_expert_bias,
            model.config.moe_router_topk,
        ).indices
        expected_counts[selected_experts] = _PADDING_MULTIPLE * len(
            _LOGICAL_COMPLETION_LENGTHS
        )
        torch.testing.assert_close(global_counts, expected_counts, rtol=0.0, atol=0.0)
        expected_updated_bias = get_updated_expert_bias(
            counts_local,
            initial_expert_bias.clone(),
            model.config.moe_router_bias_update_rate,
            tp_dp_cp_group=process_groups.tp_dp_cp,
        )
        finalize_model_grads([model], pg_collection=process_groups)
        assert model._shared_prefix_finish_grad_sync_calls == 1
        torch.testing.assert_close(
            router.expert_bias,
            expected_updated_bias,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.count_nonzero(router.local_tokens_per_expert) == 0
        assert model.output_layer.weight.grad is not None
        assert torch.isfinite(model.output_layer.weight.grad).all()
        assert torch.count_nonzero(model.output_layer.weight.grad) > 0

        if global_rank == 0:
            print(
                "RLVR41_WORLD16_NEMO_SHARED_PREFIX_GREEN "
                f"loss={loss.detach().item()} "
                f"selected_logprobs={restored_logprobs.numel()} "
                f"expert_counts={global_counts.tolist()}",
                flush=True,
            )
    finally:
        test_utilities.Utils.destroy_model_parallel()
        if previous_qk_layer_scaling is None:
            os.environ.pop("NVTE_APPLY_QK_LAYER_SCALING", None)
        else:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = previous_qk_layer_scaling
