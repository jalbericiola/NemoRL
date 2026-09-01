"""Production shared-prefix + repeated-MTP NeMo policy-worker gate.

This is intentionally an opt-in HSG integration test.  It verifies that MCore
advertises the validated MTP capability through both the defining module and
HybridModel's bound runtime view, then exercises NeMo's production capability
negotiation, real policy-worker constructor, and optimizer path without widening
the capability set in the test process.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ray
import torch

pytestmark = pytest.mark.mcore

_GATE_ENV = "NEMORL_SHARED_PREFIX_MTP_WORKER_GATE"
_WORLD_SIZE = 16
_TP_SIZE = 4
_CP_SIZE = 4
_EP_SIZE = 16
_ETP_SIZE = 1
_PADDING_MULTIPLE = 32
_BATCH_SIZE = 4
_BIN_CAPACITY = 256
_PROMPT_LENGTH = 8
_COMPLETION_LENGTHS = (8, 12, 16, 20)


def _require_gate_condition(condition: bool, reason: str, *, gate_mode: bool) -> None:
    """Skip ordinary collection, but turn every missing prerequisite into a failure."""
    if condition:
        return
    if gate_mode:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def _build_policy_config(model_path: str) -> dict[str, Any]:
    """Resolve the real Nano recipe with only workload-scale gate overrides."""
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    repo_root = Path(__file__).resolve().parents[4]
    recipe = repo_root / "examples/nemo_gym/nemotron-3.5-nano/rlvr.yaml"
    if not recipe.is_file():
        raise RuntimeError(f"Nano RLVR recipe is missing: {recipe}")

    register_omegaconf_resolvers()
    config = load_config(recipe)
    overrides: dict[str, Any] = {
        "policy.model_name": model_path,
        "policy.tokenizer.name": model_path,
        "policy.train_global_batch_size": _BATCH_SIZE,
        "policy.train_micro_batch_size": 1,
        "policy.max_total_sequence_length": _BIN_CAPACITY,
        "policy.logprob_chunk_size": 64,
        "policy.shared_prefix_training.mode": "train",
        "policy.sequence_packing.train_mb_tokens": _BIN_CAPACITY,
        "policy.sequence_packing.logprob_mb_tokens": _BIN_CAPACITY,
        "policy.make_sequence_length_divisible_by": _PADDING_MULTIPLE,
        "policy.megatron_cfg.tensor_model_parallel_size": _TP_SIZE,
        "policy.megatron_cfg.context_parallel_size": _CP_SIZE,
        "policy.megatron_cfg.expert_model_parallel_size": _EP_SIZE,
        "policy.megatron_cfg.expert_tensor_parallel_size": _ETP_SIZE,
        "policy.megatron_cfg.pipeline_model_parallel_size": 1,
        "policy.megatron_cfg.sequence_parallel": True,
        # Algorithm setup normally derives this before worker construction.
        # This direct-worker gate performs exactly one optimizer step.
        "policy.megatron_cfg.train_iters": 1,
        "policy.megatron_cfg.activation_checkpointing": True,
        "policy.megatron_cfg.recompute_granularity": "full",
        "policy.megatron_cfg.recompute_method": "uniform",
        "policy.megatron_cfg.recompute_num_layers": 1,
        "policy.megatron_cfg.moe_aux_loss_coeff": 0.0,
        "policy.megatron_cfg.moe_z_loss_coeff": 0.0,
        "policy.megatron_cfg.moe_input_jitter_eps": None,
        "policy.megatron_cfg.moe_router_load_balancing_type": "none",
        "policy.megatron_cfg.moe_shared_expert_overlap": False,
        "policy.megatron_cfg.mtp_num_layers": 5,
        "policy.megatron_cfg.mtp_use_repeated_layer": True,
        "policy.megatron_cfg.mtp_detach_heads": True,
        "policy.megatron_cfg.cuda_graph_impl": "none",
        "policy.megatron_cfg.fp8_cfg.enabled": False,
    }
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)

    policy = OmegaConf.to_container(config.policy, resolve=True)
    if not isinstance(policy, dict):
        raise TypeError(f"resolved policy must be a dict, got {type(policy).__name__}")
    return policy


def _build_worker_sharding():
    """Match Policy's PP, DP, CP, TP rank layout for a single DP replica."""
    from nemo_rl.distributed.named_sharding import NamedSharding

    return NamedSharding(
        layout=np.arange(_WORLD_SIZE).reshape(1, 1, _CP_SIZE, _TP_SIZE),
        names=[
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ],
    )


def _build_shared_prefix_batch():
    """Create one exact-prefix K=4 star with completion-only loss masks."""
    from nemo_rl.data.packing.shared_prefix_metadata import (
        SHARED_PREFIX_EXECUTION_SLOT,
        SHARED_PREFIX_GROUP_ID,
        SHARED_PREFIX_PROMPT_LENGTHS,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    width = _PROMPT_LENGTH + max(_COMPLETION_LENGTHS)
    input_ids = torch.zeros((_BATCH_SIZE, width), dtype=torch.long)
    token_mask = torch.zeros((_BATCH_SIZE, width), dtype=torch.float32)
    input_lengths = torch.empty(_BATCH_SIZE, dtype=torch.int32)
    prompt = torch.arange(101, 101 + _PROMPT_LENGTH, dtype=torch.long)
    for row, completion_length in enumerate(_COMPLETION_LENGTHS):
        input_length = _PROMPT_LENGTH + completion_length
        input_ids[row, :_PROMPT_LENGTH] = prompt
        input_ids[row, _PROMPT_LENGTH:input_length] = torch.arange(
            1001 + 64 * row,
            1001 + 64 * row + completion_length,
            dtype=torch.long,
        )
        # Column j says whether token j is a valid target; the first completion
        # token is therefore predicted from the final shared prompt state.
        token_mask[row, _PROMPT_LENGTH:input_length] = 1.0
        input_lengths[row] = input_length

    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": torch.arange(width).unsqueeze(0)
            < input_lengths.unsqueeze(1),
            "token_mask": token_mask,
            "sample_mask": torch.ones(_BATCH_SIZE, dtype=torch.float32),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.full(
                (_BATCH_SIZE,), _PROMPT_LENGTH, dtype=torch.int32
            ),
            SHARED_PREFIX_GROUP_ID: ["mtp-worker-gate"] * _BATCH_SIZE,
            SHARED_PREFIX_EXECUTION_SLOT: torch.zeros(_BATCH_SIZE, dtype=torch.int32),
        }
    )


@pytest.mark.timeout(3600)
def test_nano_policy_worker_builds_and_optimizes_shared_prefix_mtp5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the real worker and execute one TP4/CP4 MTP5 optimizer step."""
    gate_mode = os.environ.get(_GATE_ENV) == "1"
    _require_gate_condition(
        int(os.environ.get("WORLD_SIZE", "1")) == _WORLD_SIZE,
        "Nano MTP worker gate requires torchrun WORLD_SIZE=16",
        gate_mode=gate_mode,
    )
    _require_gate_condition(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "Nano MTP worker gate requires CUDA BF16 support",
        gate_mode=gate_mode,
    )
    model_path = os.environ.get("NEMORL_NANO_MODEL_PATH", "")
    _require_gate_condition(
        bool(model_path) and (Path(model_path) / "config.json").is_file(),
        "NEMORL_NANO_MODEL_PATH must name the mounted Nano HF checkpoint",
        gate_mode=gate_mode,
    )
    for module_name in ("causal_conv1d", "flash_attn", "mamba_ssm"):
        try:
            __import__(module_name)
        except ImportError as error:
            _require_gate_condition(
                False,
                f"Nano MTP worker gate requires {module_name}: {error}",
                gate_mode=gate_mode,
            )

    from megatron.core.models.hybrid import hybrid_model as mcore_hybrid_model
    from megatron.core.models.hybrid import shared_prefix as mcore_shared_prefix
    from megatron.core.utils import unwrap_model

    from nemo_rl.algorithms.loss import NLLLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.models.megatron.setup import (
        SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
        _get_mcore_shared_prefix_training_capability,
        destroy_parallel_state,
    )
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    advertised = frozenset(mcore_shared_prefix.SHARED_PREFIX_TRAINING_CAPABILITIES)
    hybrid_model_advertised = frozenset(
        mcore_hybrid_model.SHARED_PREFIX_TRAINING_CAPABILITIES
    )
    negotiated = _get_mcore_shared_prefix_training_capability()
    assert SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY in advertised, (
        "MCore must advertise the production MTP capability"
    )
    assert (
        SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY in hybrid_model_advertised
    ), "HybridModel's bound production capability must include MTP"
    assert SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY in negotiated, (
        "NeMo must negotiate MCore's production MTP capability"
    )
    assert negotiated == advertised
    assert (
        mcore_shared_prefix.SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY
        == SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [local_rank])
    config = _build_policy_config(model_path)
    tokenizer = get_tokenizer(config["tokenizer"])
    worker = None
    try:
        worker = MegatronPolicyWorkerImpl(
            config=config,
            tokenizer=tokenizer,
            init_optimizer=True,
            init_reference_model=False,
            worker_sharding_annotations=_build_worker_sharding(),
        )
        model_config = worker._get_model_config()
        assert model_config is not None
        assert model_config.tensor_model_parallel_size == _TP_SIZE
        assert model_config.context_parallel_size == _CP_SIZE
        assert model_config.expert_model_parallel_size == _EP_SIZE
        assert model_config.expert_tensor_parallel_size == _ETP_SIZE
        assert model_config.sequence_parallel is True
        assert model_config.num_moe_experts == 128
        assert model_config.recompute_granularity == "full"
        assert model_config.recompute_method == "uniform"
        assert model_config.recompute_num_layers == 1
        assert model_config.mtp_num_layers == 5
        assert model_config.mtp_use_repeated_layer is True
        assert model_config.mtp_hybrid_override_pattern == "*E"
        main_pattern, *predictor_patterns = model_config.hybrid_layer_pattern.split("/")
        assert "M" in main_pattern and "E" in main_pattern
        assert predictor_patterns == ["*E"] * model_config.mtp_num_layers

        unwrapped = unwrap_model(worker.model)
        chunks = unwrapped if isinstance(unwrapped, (list, tuple)) else [unwrapped]
        mtp_parameter_names = [
            name
            for chunk in chunks
            for name, _parameter in chunk.named_parameters()
            if "mtp" in name.lower()
        ]
        assert mtp_parameter_names, "constructed worker has no physical MTP parameters"

        scheduler_steps_before = worker.scheduler.num_steps
        metrics = worker.train(
            _build_shared_prefix_batch(),
            NLLLossFn(),
            gbs=_BATCH_SIZE,
            mbs=1,
        )
        assert worker.scheduler.num_steps == scheduler_steps_before + _BATCH_SIZE
        assert torch.isfinite(metrics["global_loss"]).all()
        assert torch.isfinite(metrics["grad_norm"]).all()
        assert float(metrics["grad_norm"].max().item()) > 0.0
        assert "mtp_metrics" in metrics and metrics["mtp_metrics"]
        mtp_metrics = metrics["mtp_metrics"]
        expected_mtp_metrics = {
            *(f"mtp_{depth}_loss" for depth in range(1, 6)),
            *(f"mtp_{depth}_acceptance_rate" for depth in range(1, 6)),
            "grad_norm",
        }
        assert set(mtp_metrics) == expected_mtp_metrics, sorted(mtp_metrics)
        mtp_grad_norm = mtp_metrics.get("grad_norm")
        assert mtp_grad_norm is not None
        mtp_grad_norm_value = float(mtp_grad_norm)
        assert math.isfinite(mtp_grad_norm_value) and mtp_grad_norm_value > 0.0
        for metric_name, metric_value in mtp_metrics.items():
            if isinstance(metric_value, (int, float)):
                assert math.isfinite(float(metric_value)), (
                    metric_name,
                    metric_value,
                )

        if torch.distributed.get_rank() == 0:
            print(
                "NEMORL_SHARED_PREFIX_MTP_WORKER_GREEN "
                f"loss={float(metrics['global_loss'].item())} "
                f"grad_norm={float(metrics['grad_norm'].max().item())} "
                f"mtp_grad_norm={mtp_grad_norm_value} "
                f"physical_mtp_params={len(mtp_parameter_names)}",
                flush=True,
            )
    finally:
        if worker is not None:
            worker.shutdown()
        destroy_parallel_state()
