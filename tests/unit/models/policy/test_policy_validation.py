# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Unit tests for Policy class validation logic.

This module tests the early validation checks in the Policy class, particularly
the world_size compatibility validation that prevents confusing reshape errors
when the cluster size is insufficient for the specified parallelism configuration.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.policy import (
    PolicyConfig,
    SharedPrefixTrainingConfig,
    get_shared_prefix_training_config,
    shared_prefix_deterministic_execution_required,
    validate_shared_prefix_training_config,
)
from nemo_rl.models.policy.lm_policy import Policy


def test_shutdown_succeeds_before_worker_group_is_initialized(capsys) -> None:
    policy = Policy.__new__(Policy)

    assert policy.shutdown()
    assert capsys.readouterr().out == ""


def create_mock_cluster(world_size: int):
    """Create a mock cluster with the specified world size."""
    cluster = MagicMock()
    cluster.world_size.return_value = world_size

    # Mock get_master_address_and_port method to return valid address and port
    cluster.get_master_address_and_port.return_value = ("127.0.0.1", 29500)

    # Mock get_placement_groups method to return a list of mock placement groups
    mock_pg = MagicMock()
    mock_pg.bundle_count = world_size  # Each placement group has world_size bundles
    cluster.get_placement_groups.return_value = [mock_pg]

    # Mock get_available_address_and_port method
    cluster.get_available_address_and_port.return_value = ("127.0.0.1", 29501)

    return cluster


def create_mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


def create_dtensor_config(
    model_name: str, tp: int, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a DTensor configuration for testing."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_new_tokens": 16,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            "enabled": True,
            "cpu_offload": False,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
    }


def create_megatron_config(
    model_name: str, tp: int, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a Megatron configuration for testing."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_new_tokens": 16,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "context_parallel_size": cp,
            "sequence_parallel": False,
            "activation_checkpointing": False,
        },
        "dynamic_batching": {
            "enabled": pp == 1,  # Only enable for single pipeline parallel stage
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
    }


def _add_shared_prefix_determinism_contract(config: PolicyConfig) -> PolicyConfig:
    """Install the exact deterministic actor/provider test contract."""
    megatron_config = cast(dict[str, Any], config["megatron_cfg"])
    megatron_config["env_vars"] = {
        "MAMBA_DETERMINISTIC": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
    }
    megatron_config["model_overrides"] = {
        "deterministic_mode": True,
        "cross_entropy_loss_fusion": False,
        "tp_comm_overlap": False,
    }
    return config


def create_shared_prefix_train_config(
    *, tp: int = 1, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a topology-valid shared-prefix policy slice."""
    config = _add_shared_prefix_determinism_contract(
        create_megatron_config("test-model", tp=tp, pp=pp, cp=cp)
    )
    cast(dict[str, Any], config["megatron_cfg"])["sequence_parallel"] = tp > 1
    config["precision"] = "bfloat16"
    config["sequence_packing"] = {
        "enabled": True,
        "train_mb_tokens": 128,
        "logprob_mb_tokens": 128,
        "algorithm": "modified_first_fit_decreasing",
    }
    config["shared_prefix_training"] = {"mode": "train"}
    return config


def create_shared_prefix_mtp5_train_config() -> PolicyConfig:
    """Create the one validated shared-prefix MTP provider/topology slice."""
    config = create_shared_prefix_train_config(tp=2, cp=2)
    config["make_sequence_length_divisible_by"] = 128
    cast(dict[str, Any], config["megatron_cfg"]).update(
        {
            "mtp_num_layers": 5,
            "mtp_loss_scaling_factor": 0.3,
            "mtp_use_repeated_layer": True,
            "mtp_detach_heads": True,
            "tensor_model_parallel_size": 2,
            "context_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
        }
    )
    return config


def test_shared_prefix_training_defaults_to_disabled_for_legacy_config() -> None:
    config = create_dtensor_config("test-model", tp=1)

    resolved_config = get_shared_prefix_training_config(config)

    assert resolved_config == SharedPrefixTrainingConfig(mode="disabled")


def test_shared_prefix_observe_mode_is_backend_neutral() -> None:
    config = create_dtensor_config("test-model", tp=1, cp=2)
    config["shared_prefix_training"] = {"mode": "observe"}

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "observe"


def test_shared_prefix_config_rejects_misspelled_determinism_flag() -> None:
    config = create_megatron_config("test-model", tp=1)
    config["shared_prefix_training"] = {
        "mode": "observe",
        "require_determinstic_execution": True,
    }

    with pytest.raises(ValueError, match="require_determinstic_execution"):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize("value", [0, 1, "false", "true"])
def test_shared_prefix_config_rejects_coerced_determinism_flag(
    value: object,
) -> None:
    config = create_megatron_config("test-model", tp=1)
    config["shared_prefix_training"] = {
        "mode": "observe",
        "require_deterministic_execution": value,
    }

    with pytest.raises(ValueError, match="require_deterministic_execution"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_disabled_mode_rejects_deterministic_execution_opt_in() -> None:
    config = create_dtensor_config("test-model", tp=1)
    config["shared_prefix_training"] = {
        "mode": "disabled",
        "require_deterministic_execution": True,
    }

    with pytest.raises(ValueError, match="mode=disabled must preserve"):
        shared_prefix_deterministic_execution_required(config)
    with pytest.raises(ValueError, match="mode=disabled must preserve"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_strict_observe_requires_megatron() -> None:
    config = create_dtensor_config("test-model", tp=1)
    config["shared_prefix_training"] = {
        "mode": "observe",
        "require_deterministic_execution": True,
    }

    with pytest.raises(ValueError, match="policy.megatron_cfg.enabled=true"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_strict_observe_accepts_exact_contract_without_train_gates() -> (
    None
):
    config = _add_shared_prefix_determinism_contract(
        create_megatron_config("test-model", tp=1, pp=2, cp=2)
    )
    config["shared_prefix_training"] = {
        "mode": "observe",
        "require_deterministic_execution": True,
    }

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "observe"
    assert shared_prefix_deterministic_execution_required(config)


def test_shared_prefix_plain_observe_does_not_require_deterministic_contract() -> None:
    config = create_megatron_config("test-model", tp=1)
    config["shared_prefix_training"] = {"mode": "observe"}

    assert not shared_prefix_deterministic_execution_required(config)
    assert validate_shared_prefix_training_config(config).mode == "observe"


@pytest.mark.parametrize(
    ("block", "name", "value"),
    [
        ("env_vars", "MAMBA_DETERMINISTIC", "0"),
        ("model_overrides", "deterministic_mode", False),
    ],
)
def test_shared_prefix_strict_observe_rejects_determinism_contract_drift(
    block: str,
    name: str,
    value: object,
) -> None:
    config = _add_shared_prefix_determinism_contract(
        create_megatron_config("test-model", tp=1)
    )
    config["shared_prefix_training"] = {
        "mode": "observe",
        "require_deterministic_execution": True,
    }
    cast(dict[str, Any], config["megatron_cfg"])[block][name] = value

    with pytest.raises(ValueError, match=rf"{block}\.{name}="):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAMBA_DETERMINISTIC", None),
        ("MAMBA_DETERMINISTIC", "0"),
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
        ("NCCL_ALGO", "Tree"),
    ],
)
def test_shared_prefix_train_requires_exact_determinism_environment(
    name: str,
    value: object,
) -> None:
    config = create_shared_prefix_train_config()
    env_vars = cast(dict[str, Any], config["megatron_cfg"])["env_vars"]
    if value is None:
        env_vars.pop(name)
    else:
        env_vars[name] = value

    with pytest.raises(ValueError, match=rf"env_vars\.{name}="):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("deterministic_mode", False),
        ("cross_entropy_loss_fusion", True),
        ("tp_comm_overlap", True),
    ],
)
def test_shared_prefix_train_requires_exact_determinism_model_overrides(
    name: str,
    value: object,
) -> None:
    config = create_shared_prefix_train_config()
    overrides = cast(dict[str, Any], config["megatron_cfg"])["model_overrides"]
    overrides[name] = value

    with pytest.raises(ValueError, match=rf"model_overrides\.{name}="):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    "name",
    [
        "TRITON_CACHE_AUTOTUNING",
        "TRITON_AUTOTUNE_BLOCK_SIZE_M",
        "TRITON_AUTOTUNE_BLOCK_T",
    ],
)
def test_shared_prefix_determinism_rejects_triton_autotuning(name: str) -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"])["env_vars"][name] = "1"

    with pytest.raises(ValueError, match=rf"unset.*{name}"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_train_mode_accepts_first_slice_topology() -> None:
    config = create_shared_prefix_train_config()

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


def test_shared_prefix_train_mode_accepts_cp_config_for_late_capability_gate() -> None:
    config = create_shared_prefix_train_config(cp=2)

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


@pytest.mark.parametrize(("tp", "cp"), [(2, 1), (2, 2), (4, 4)])
def test_shared_prefix_train_mode_accepts_tp_sp_for_late_capability_gate(
    tp: int,
    cp: int,
) -> None:
    config = create_shared_prefix_train_config(tp=tp, cp=cp)

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


def test_shared_prefix_train_mode_accepts_exact_mtp5_target_early() -> None:
    config = create_shared_prefix_mtp5_train_config()

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"
    assert config["make_sequence_length_divisible_by"] == 128


@pytest.mark.parametrize(
    "hybrid_layer_pattern",
    [
        None,
        "",
        "M",
        "M/*E",
        "M/*E/*E/*E/*E/*E/*E",
        "M/*A/*A/*A/*A/*A",
        "M/*E/*E/*A/*E/*E",
    ],
)
def test_shared_prefix_mtp5_resolved_pattern_requires_five_exact_dense_heads(
    hybrid_layer_pattern: object,
) -> None:
    from nemo_rl.models.policy import get_shared_prefix_mtp_target_mismatch

    config = create_shared_prefix_mtp5_train_config()
    values = dict(cast(dict[str, Any], config["megatron_cfg"]))
    values.update(
        {
            "mtp_hybrid_override_pattern": "*E",
            "position_embedding_type": "none",
            "bf16": True,
            "fp16": False,
            "hybrid_layer_pattern": hybrid_layer_pattern,
        }
    )

    mismatch = get_shared_prefix_mtp_target_mismatch(
        values,
        require_provider_pattern=True,
        resolved_world_size=4,
    )

    assert mismatch is not None
    assert mismatch[0] == "hybrid_layer_pattern"


@pytest.mark.parametrize("mtp_num_layers", [None, 0])
def test_shared_prefix_train_mode_preserves_disabled_mtp_legacy_path(
    mtp_num_layers: int | None,
) -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"])["mtp_num_layers"] = mtp_num_layers

    assert validate_shared_prefix_training_config(config).mode == "train"


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("mtp_num_layers", 5.0, "mtp_num_layers=5"),
        ("mtp_loss_scaling_factor", None, "mtp_loss_scaling_factor=0.3"),
        ("mtp_use_repeated_layer", 1, "mtp_use_repeated_layer=True"),
        ("mtp_detach_heads", False, "mtp_detach_heads=True"),
        ("tensor_model_parallel_size", 4, "tensor_model_parallel_size=2"),
        ("context_parallel_size", 4, "context_parallel_size=2"),
        (
            "sequence_parallel",
            False,
            "sequence_parallel=true exactly when TP>1",
        ),
        ("pipeline_model_parallel_size", 2, "pipeline_model_parallel_size=1"),
        ("expert_model_parallel_size", 2, "expert_model_parallel_size=4"),
        ("expert_tensor_parallel_size", 2, "expert_tensor_parallel_size=1"),
    ],
)
def test_shared_prefix_train_mode_rejects_exact_mtp5_target_drift_early(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    config = create_shared_prefix_mtp5_train_config()
    cast(dict[str, Any], config["megatron_cfg"])[field] = value

    with pytest.raises(ValueError, match=expected_error):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_model_parallel_size", True),
        ("tensor_model_parallel_size", 2.0),
        ("tensor_model_parallel_size", "2"),
        ("context_parallel_size", False),
        ("context_parallel_size", 2.0),
        ("context_parallel_size", "2"),
        ("sequence_parallel", 1),
        ("sequence_parallel", "false"),
    ],
)
def test_shared_prefix_train_mode_rejects_coercible_topology_values(
    field: str,
    value: object,
) -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"])[field] = value

    with pytest.raises(ValueError, match="positive integer|boolean"):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    "config,expected_config_path",
    [
        (
            create_dtensor_config("test-model", tp=1),
            "policy.megatron_cfg.enabled=true",
        ),
        (
            _add_shared_prefix_determinism_contract(
                create_megatron_config("test-model", tp=1)
            ),
            "policy.sequence_packing.enabled=true",
        ),
        (
            _add_shared_prefix_determinism_contract(
                create_megatron_config("test-model", tp=1, pp=2)
            ),
            "policy.megatron_cfg.pipeline_model_parallel_size=1",
        ),
        (
            _add_shared_prefix_determinism_contract(
                create_megatron_config("test-model", tp=2)
            ),
            "sequence_parallel=true exactly when TP>1",
        ),
    ],
)
def test_shared_prefix_train_mode_rejects_unsupported_first_slice_topology(
    config: PolicyConfig,
    expected_config_path: str,
) -> None:
    config["shared_prefix_training"] = {"mode": "train"}
    if "megatron_cfg" in config:
        config["sequence_packing"] = {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "algorithm": "modified_first_fit_decreasing",
        }
    if expected_config_path == "policy.sequence_packing.enabled=true":
        config["sequence_packing"] = {"enabled": False}

    with pytest.raises(ValueError, match=expected_config_path):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    "megatron_overrides,policy_overrides,expected_config_path",
    [
        (
            {"sequence_parallel": True},
            {},
            "sequence_parallel=true exactly when TP>1",
        ),
        (
            {"mtp_num_layers": 1},
            {},
            "policy.megatron_cfg.mtp_num_layers=5",
        ),
        (
            {"cuda_graph_impl": "local"},
            {},
            "policy.megatron_cfg.cuda_graph_impl='none'",
        ),
        (
            {"fp8_cfg": {"enabled": True}},
            {},
            "policy.megatron_cfg.fp8_cfg.enabled=false",
        ),
        (
            {},
            {"quant_cfg": "nvfp4"},
            "policy.quant_cfg=null",
        ),
    ],
)
def test_shared_prefix_train_mode_rejects_unsupported_runtime_features_early(
    megatron_overrides: dict[str, object],
    policy_overrides: dict[str, object],
    expected_config_path: str,
) -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"]).update(megatron_overrides)
    cast(dict[str, Any], config).update(policy_overrides)

    with pytest.raises(ValueError, match=expected_config_path):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    "config_path,value",
    [
        ("train_mb_tokens", 126),
        ("logprob_mb_tokens", 130),
    ],
)
def test_shared_prefix_cp_requires_aligned_microbatch_capacities(
    config_path: str,
    value: int,
) -> None:
    config = create_shared_prefix_train_config(cp=2)
    cast(dict[str, Any], config["sequence_packing"])[config_path] = value

    with pytest.raises(ValueError, match=f"sequence_packing.{config_path}"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_cp_requires_aligned_sequence_divisibility() -> None:
    config = create_shared_prefix_train_config(cp=2)
    config["make_sequence_length_divisible_by"] = 2

    with pytest.raises(ValueError, match="make_sequence_length_divisible_by"):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize("raw", [True, False, 0, -8, 8.0, "8"])
def test_shared_prefix_rejects_invalid_explicit_physical_multiple(raw: object) -> None:
    config = create_shared_prefix_train_config(tp=2, cp=2)
    cast(dict[str, Any], config)["make_sequence_length_divisible_by"] = raw

    with pytest.raises(ValueError, match="make_sequence_length_divisible_by"):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize("raw", [None, 16])
def test_shared_prefix_tp_cp_resolves_legacy_or_explicit_m(raw: int | None) -> None:
    config = create_shared_prefix_train_config(tp=2, cp=2)
    if raw is not None:
        config["make_sequence_length_divisible_by"] = raw
    else:
        cast(dict[str, Any], config)["make_sequence_length_divisible_by"] = None

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


def test_shared_prefix_capacities_must_follow_explicit_m_not_only_q() -> None:
    config = create_shared_prefix_train_config(tp=2, cp=2)
    config["make_sequence_length_divisible_by"] = 32
    cast(dict[str, Any], config["sequence_packing"])["train_mb_tokens"] = 128
    cast(dict[str, Any], config["sequence_packing"])["logprob_mb_tokens"] = 96

    assert validate_shared_prefix_training_config(config).mode == "train"

    cast(dict[str, Any], config["sequence_packing"])["logprob_mb_tokens"] = 104
    with pytest.raises(ValueError, match="resolved padding M=32"):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_train_mode_allows_selective_activation_recompute() -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"]).update(
        {
            "activation_checkpointing": True,
            "recompute_granularity": "selective",
            "recompute_modules": ["core_attn"],
        }
    )

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


def test_shared_prefix_train_mode_allows_pipeclean_default_full_recompute() -> None:
    config = create_shared_prefix_train_config()
    megatron_config = cast(dict[str, Any], config["megatron_cfg"])
    megatron_config["activation_checkpointing"] = True
    assert "recompute_granularity" not in megatron_config
    assert "recompute_method" not in megatron_config
    assert "recompute_num_layers" not in megatron_config

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "train"


@pytest.mark.parametrize(
    ("recompute_overrides", "expected_config_path"),
    [
        (
            {
                "activation_checkpointing": True,
                "recompute_granularity": "full",
                "recompute_method": "block",
            },
            "policy.megatron_cfg.recompute_method='uniform'",
        ),
        (
            {
                "activation_checkpointing": True,
                "recompute_granularity": "full",
                "recompute_num_layers": 2,
            },
            "policy.megatron_cfg.recompute_num_layers=1",
        ),
    ],
)
def test_shared_prefix_train_mode_rejects_incompatible_raw_full_recompute(
    recompute_overrides: dict[str, object],
    expected_config_path: str,
) -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"]).update(recompute_overrides)

    with pytest.raises(ValueError, match=expected_config_path):
        validate_shared_prefix_training_config(config)


@pytest.mark.parametrize(
    "world_size,tp,cp,should_pass,expected_error_type,description",
    [
        # Valid cases - DTensor backend (PP is always 1 for DTensor)
        (8, 8, 1, True, None, "Valid: DP=1, TP=8, PP=1, CP=1"),
        (16, 8, 1, True, None, "Valid: DP=2, TP=8, PP=1, CP=1"),
        (8, 4, 2, True, None, "Valid: DP=1, TP=4, PP=1, CP=2"),
        (16, 4, 2, True, None, "Valid: DP=2, TP=4, PP=1, CP=2"),
        (1, 1, 1, True, None, "Valid: Minimal config DP=1, TP=1, PP=1, CP=1"),
        # Invalid cases - insufficient world_size (DP < 1)
        (4, 8, 1, False, "insufficient", "Invalid: DP=0.5, TP=8, PP=1, CP=1"),
        (2, 8, 1, False, "insufficient", "Invalid: DP=0.25, TP=8, PP=1, CP=1"),
        (4, 4, 2, False, "insufficient", "Invalid: DP=0.5, TP=4, PP=1, CP=2"),
        # Invalid cases - not divisible (DP not integer)
        (10, 4, 2, False, "divisible", "Invalid: DP=1.25, TP=4, PP=1, CP=2"),
        (9, 8, 1, False, "divisible", "Invalid: DP=1.125, TP=8, PP=1, CP=1"),
        (6, 4, 1, False, "divisible", "Invalid: DP=1.5, TP=4, PP=1, CP=1"),
    ],
)
@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_world_size_validation_dtensor(
    mock_ray_worker_group,
    tiny_llama_model_path,
    world_size,
    tp,
    cp,
    should_pass,
    expected_error_type,
    description,
):
    """Test world_size validation with DTensor backend.

    Note: DTensor backend always uses PP=1 (no pipeline parallelism support).
    Tests the constraint: world_size = DP * PP * CP * TP where DP >= 1 and DP must be integer.
    """
    cluster = create_mock_cluster(world_size)
    tokenizer = create_mock_tokenizer()
    config = create_dtensor_config(
        tiny_llama_model_path, tp, pp=1, cp=cp
    )  # DTensor always has PP=1

    # Mock RayWorkerGroup to prevent actual worker creation
    mock_worker_group_instance = MagicMock()
    mock_ray_worker_group.return_value = mock_worker_group_instance

    if should_pass:
        # Should succeed without raising an exception
        try:
            policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
            # Verify the calculated DP makes sense
            expected_dp = world_size // (1 * cp * tp)  # PP=1 for DTensor
            assert expected_dp >= 1, f"Expected DP should be >= 1, got {expected_dp}"
            # Verify that worker group was created (validation passed)
            mock_ray_worker_group.assert_called_once()
        except Exception as e:
            pytest.fail(f"Expected success for {description}, but got error: {e}")
    else:
        # Should raise ValueError with specific error type
        with pytest.raises(ValueError) as exc_info:
            Policy(cluster=cluster, config=config, tokenizer=tokenizer)

        error_msg = str(exc_info.value)
        if expected_error_type == "insufficient":
            assert "insufficient" in error_msg, (
                f"Expected 'insufficient' error for {description}"
            )
            assert "DP must be ≥ 1" in error_msg, (
                f"Expected DP constraint message for {description}"
            )
        elif expected_error_type == "divisible":
            assert "must be divisible" in error_msg, (
                f"Expected 'divisible' error for {description}"
            )
            assert "not an integer" in error_msg, (
                f"Expected integer constraint message for {description}"
            )
        # For failing cases, worker group should not be created
        mock_ray_worker_group.assert_not_called()


@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_dtensor_dp_replicate_size_sets_batching_dp(
    mock_ray_worker_group,
    tiny_llama_model_path,
):
    """Test that dp_replicate_size is separated from the batching DP axis."""
    cluster = create_mock_cluster(world_size=8)
    tokenizer = create_mock_tokenizer()
    config = create_dtensor_config(tiny_llama_model_path, tp=1)
    config["dtensor_cfg"]["_v2"] = True
    config["dtensor_cfg"]["dp_replicate_size"] = 2

    policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)

    assert policy.sharding_annotations.shape["data_parallel"] == 8
    assert policy.sharding_annotations.get_axis_size("data_parallel") == 8
    mock_ray_worker_group.assert_called_once()


@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_dtensor_hsdp_dispatches_distinct_batches(
    mock_ray_worker_group,
    tiny_llama_model_path,
):
    """Test that HSDP (dp_replicate_size > 1) dispatches distinct batches to all replicas.

    The bug was that dp_replicate workers received identical batches.
    By unifying dp_shard and dp_replicate into a single data_parallel axis,
    we ensure data.shard_by_batch_size is called with the FULL DP product,
    and run_all_workers_sharded_data shards across all of them.
    """
    cluster = create_mock_cluster(world_size=8)
    tokenizer = create_mock_tokenizer()
    config = create_dtensor_config(tiny_llama_model_path, tp=1)
    config["dtensor_cfg"]["_v2"] = True
    config["dtensor_cfg"]["dp_replicate_size"] = 2  # HSDP enabled

    policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)

    # Mock data
    mock_data = MagicMock()
    # Create 8 distinct shards to prove each of the 8 DP workers gets unique data
    mock_lengths = MagicMock()
    mock_lengths.tolist.return_value = [10]
    mock_shards = [
        {"input_lengths": mock_lengths, "id": f"shard_{i}"} for i in range(8)
    ]
    mock_data.shard_by_batch_size.return_value = (mock_shards, None)

    mock_loss_fn = MagicMock()

    # Call train to trigger data dispatch
    policy.train(
        data=mock_data,
        loss_fn=mock_loss_fn,
        gbs=32,
        mbs=4,
    )

    # 1. Assert data was sharded into 8 distinct pieces (the full DP product)
    mock_data.shard_by_batch_size.assert_called_once()
    called_dp_size = mock_data.shard_by_batch_size.call_args[0][0]
    assert called_dp_size == 8, (
        f"Data should be sharded into 8 pieces, got {called_dp_size}"
    )

    # 2. Assert the 8 distinct pieces were sent to the workers sharded across data_parallel
    mock_worker_group = mock_ray_worker_group.return_value
    mock_worker_group.run_all_workers_sharded_data.assert_any_call(
        "train",
        data=mock_shards,  # The 8 distinct shards are passed directly
        in_sharded_axes=["data_parallel"],  # They are sharded across the unified axis
        replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
        output_is_replicated=[
            "context_parallel",
            "tensor_parallel",
            "pipeline_parallel",
        ],
        common_kwargs={
            "loss_fn": mock_loss_fn,
            "eval_mode": False,
            "gbs": 32,
            "mbs": 4,
            "check_dim_skip_keys": None,
        },
    )


@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_dtensor_dp_replicate_size_requires_v2(
    mock_ray_worker_group,
    tiny_llama_model_path,
):
    """Test that HSDP requires the Automodel DTensor v2 worker."""
    cluster = create_mock_cluster(world_size=8)
    tokenizer = create_mock_tokenizer()
    config = create_dtensor_config(tiny_llama_model_path, tp=1)
    config["dtensor_cfg"]["dp_replicate_size"] = 2

    with pytest.raises(ValueError, match="_v2: true"):
        Policy(cluster=cluster, config=config, tokenizer=tokenizer)

    mock_ray_worker_group.assert_not_called()


@pytest.mark.parametrize(
    "world_size,tp,pp,cp,should_pass,expected_error_type,description",
    [
        # Valid cases - Megatron backend (supports PP > 1)
        (
            32,
            8,
            4,
            1,
            True,
            None,
            "Valid: DP=1, TP=8, PP=4, CP=1 (original error case fixed)",
        ),
        (64, 8, 4, 1, True, None, "Valid: DP=2, TP=8, PP=4, CP=1"),
        (16, 4, 2, 2, True, None, "Valid: DP=1, TP=4, PP=2, CP=2"),
        # Invalid cases - insufficient world_size (DP < 1)
        (
            8,
            8,
            4,
            1,
            False,
            "insufficient",
            "Invalid: DP=0.25, TP=8, PP=4, CP=1 (original error)",
        ),
        (16, 8, 4, 1, False, "insufficient", "Invalid: DP=0.5, TP=8, PP=4, CP=1"),
        # Invalid cases - not divisible (DP not integer)
        (33, 8, 4, 1, False, "divisible", "Invalid: DP=1.03, TP=8, PP=4, CP=1"),
        (18, 4, 2, 2, False, "divisible", "Invalid: DP=1.125, TP=4, PP=2, CP=2"),
    ],
)
@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_world_size_validation_megatron(
    mock_ray_worker_group,
    tiny_llama_model_path,
    world_size,
    tp,
    pp,
    cp,
    should_pass,
    expected_error_type,
    description,
):
    """Test world_size validation with Megatron backend.

    Megatron backend supports pipeline parallelism (PP > 1) unlike DTensor.
    Tests the constraint: world_size = DP * PP * CP * TP where DP >= 1 and DP must be integer.
    Note: Expert Parallelism (EP) is handled internally by Megatron-Core, not at the worker level.
    """
    cluster = create_mock_cluster(world_size)
    tokenizer = create_mock_tokenizer()
    config = create_megatron_config(tiny_llama_model_path, tp, pp, cp)

    # Mock RayWorkerGroup to prevent actual worker creation
    mock_worker_group_instance = MagicMock()
    mock_ray_worker_group.return_value = mock_worker_group_instance

    if should_pass:
        # Should succeed without raising an exception
        try:
            policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
            # Verify the calculated DP makes sense
            expected_dp = world_size // (pp * cp * tp)
            assert expected_dp >= 1, f"Expected DP should be >= 1, got {expected_dp}"
            # Verify that worker group was created (validation passed)
            mock_ray_worker_group.assert_called_once()
        except Exception as e:
            pytest.fail(f"Expected success for {description}, but got error: {e}")
    else:
        # Should raise ValueError with specific error type
        with pytest.raises(ValueError) as exc_info:
            Policy(cluster=cluster, config=config, tokenizer=tokenizer)

        error_msg = str(exc_info.value)
        if expected_error_type == "insufficient":
            assert "insufficient" in error_msg, (
                f"Expected 'insufficient' error for {description}"
            )
            assert "DP must be ≥ 1" in error_msg, (
                f"Expected DP constraint message for {description}"
            )
        elif expected_error_type == "divisible":
            assert "must be divisible" in error_msg, (
                f"Expected 'divisible' error for {description}"
            )
            assert "not an integer" in error_msg, (
                f"Expected integer constraint message for {description}"
            )
        # For failing cases, worker group should not be created
        mock_ray_worker_group.assert_not_called()
