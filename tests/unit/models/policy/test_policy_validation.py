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

from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.policy import (
    PolicyConfig,
    SharedPrefixTrainingConfig,
    get_shared_prefix_training_config,
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


def create_shared_prefix_train_config(
    *, tp: int = 1, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a topology-valid shared-prefix policy slice."""
    config = create_megatron_config("test-model", tp=tp, pp=pp, cp=cp)
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


def test_shared_prefix_training_defaults_to_disabled_for_legacy_config() -> None:
    config = create_dtensor_config("test-model", tp=1)

    resolved_config = get_shared_prefix_training_config(config)

    assert resolved_config == SharedPrefixTrainingConfig(mode="disabled")


def test_shared_prefix_observe_mode_is_backend_neutral() -> None:
    config = create_dtensor_config("test-model", tp=1, cp=2)
    config["shared_prefix_training"] = {"mode": "observe"}

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == "observe"


@pytest.mark.parametrize("mode", ["disabled", "observe"])
def test_shared_prefix_non_train_modes_allow_megatron_peft(
    mode: Literal["disabled", "observe"],
) -> None:
    config = create_shared_prefix_train_config()
    config["shared_prefix_training"] = {"mode": mode}
    cast(dict[str, Any], config["megatron_cfg"])["peft"] = {"enabled": True}

    resolved_config = validate_shared_prefix_training_config(config)

    assert resolved_config.mode == mode


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
            create_megatron_config("test-model", tp=1),
            "policy.sequence_packing.enabled=true",
        ),
        (
            create_megatron_config("test-model", tp=1, pp=2),
            "policy.megatron_cfg.pipeline_model_parallel_size=1",
        ),
        (
            create_megatron_config("test-model", tp=2),
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


def test_shared_prefix_train_mode_rejects_megatron_peft_early() -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"])["peft"] = {"enabled": True}

    with pytest.raises(
        ValueError,
        match=r"policy\.megatron_cfg\.peft\.enabled=false.*PEFT/LoRA",
    ):
        validate_shared_prefix_training_config(config)


def test_shared_prefix_train_mode_defers_mtp_to_mcore_capability_validation() -> None:
    config = create_shared_prefix_train_config()
    cast(dict[str, Any], config["megatron_cfg"]).update(
        {
            "mtp_num_layers": 5,
            "mtp_use_repeated_layer": True,
        }
    )

    assert validate_shared_prefix_training_config(config).mode == "train"


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
