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

"""Tests for SingleController initialization and pump lifecycle."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    RewardScalingConfig,
    _initial_grpo_save_state,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.algorithms.shared_prefix_metrics import SharedPrefixOpportunity
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    _reduce_rollout_step_metrics,
    _reduce_shared_prefix_step_metrics,
    _resolve_train_dispatch_group_multiple,
    _train_selection_group_bounds,
    _unpack_sampler_selection,
    _validate_train_dispatch_buffer_capacity,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    INVALID_TOOL_CALL_TOKEN_MASK,
    MALFORMED_THINKING_MESSAGE_COUNT,
    MALFORMED_THINKING_TOKEN_MASK,
    RESPONSE_TOKEN_LENGTHS,
    ROLLOUT_TRUNCATED,
)
from nemo_rl.utils.timer import TimeoutChecker, Timer


class FakeWeightSynchronizer:
    pass


def _checkpointing_config(tmp_path) -> dict:
    """Minimal checkpointing block for actors built through __init__."""
    return {
        "enabled": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "metric_name": None,
        "higher_is_better": True,
        "keep_top_k": None,
        "save_period": 10,
        "save_optimizer": True,
        "checkpoint_must_save_by": None,
    }


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
        env={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=None,
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"num_prompts_per_step \* num_generations_per_prompt \(8\) "
            r"must equal policy.train_global_batch_size \(4\)"
        ),
    ):
        controller_cls(
            master_config=master_config,
            actor_args=actor_args,
            setup_timing_metrics=SetupTimingMetrics(),
        )


def test_logs_hyperparameters_and_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


def test_logs_setup_timing_metrics(monkeypatch, tmp_path) -> None:
    """setup_timing_metrics is forwarded to Logger.log_metrics under timing/setup."""
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    setup_metrics = SetupTimingMetrics(
        generation_init_time_s=1.5, policy_init_time_s=2.5
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
        # A real field of SingleControllerActorArgs. Read directly rather than via a
        # getattr default, so omitting it breaks here instead of silently degrading
        # watchdog.gym_subprocess_check into a no-op at runtime.
        env_handles={},
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=setup_metrics,
    )

    logger.log_metrics.assert_called_once_with(
        setup_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
    )


@pytest.mark.parametrize(
    ("recompute_kv_cache", "expected_invalidation_calls"),
    [(False, 0), (True, 1)],
)
def test_sync_weights_honors_recompute_kv_cache_config(
    recompute_kv_cache: bool,
    expected_invalidation_calls: int,
) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(
        recompute_kv_cache_after_weight_updates=recompute_kv_cache
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=False,
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    ctrl._inflight_by_group_id = {}
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})

    asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    assert ctrl._gen.invalidate_kv_cache.call_count == expected_invalidation_calls
    ctrl._rollout_manager.set_weight_version.assert_called_once_with(3)
    assert ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(
        calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}})
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    ctrl._inflight_by_group_id = {}
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})
    calibration_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    asyncio.run(ctrl._sync_weights(calibration_data=calibration_data))

    ctrl._trainer.calibrate_qkv_fp8_scales.assert_called_once_with(
        calibration_data,
        include_q=True,
    )
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(
        kv_scales={"layer.0": 0.5}
    )


class _AdvantageDataPlane:
    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selected_fields: list[str] | None = None
        self.written_fields: TensorDict | None = None

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selected_fields = list(select_fields)
        return self._data

    def put_samples(self, *, fields, **kwargs) -> None:
        del kwargs
        self.written_fields = fields


class _MaskRecordingAdvantageEstimator:
    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None

    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del kwargs
        self.mask = mask.clone()
        return rewards.unsqueeze(-1).expand_as(mask).clone()


def _valid_message_penalty_data() -> TensorDict:
    batch_size, sequence_length = 2, 4
    return TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.25, 0.75]),
            "token_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "sample_mask": torch.ones(batch_size),
            INVALID_TOOL_CALL_TOKEN_MASK: torch.tensor(
                [
                    [False, False, True, True],
                    [False, False, False, False],
                ]
            ),
            MALFORMED_THINKING_TOKEN_MASK: torch.tensor(
                [
                    [False, False, True, True],
                    [False, False, False, False],
                ]
            ),
            GENERATED_ASSISTANT_MESSAGE_COUNT: torch.tensor([1, 1]),
            INVALID_TOOL_CALL_MESSAGE_COUNT: torch.tensor([1, 0]),
            MALFORMED_THINKING_MESSAGE_COUNT: torch.tensor([1, 0]),
            INVALID_AND_MALFORMED_MESSAGE_COUNT: torch.tensor([1, 0]),
        },
        batch_size=[batch_size],
    )


def _run_message_penalty_stage(
    data: TensorDict,
    *,
    invalid_advantage: float | None = -5.0,
    malformed_advantage: float | None = -3.0,
) -> tuple[SingleControllerActor, _AdvantageDataPlane]:
    data_plane = _AdvantageDataPlane(data)
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = _MaskRecordingAdvantageEstimator()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(
            seq_logprob_error_threshold=None,
            reward_scaling=None,
            reward_shaping=None,
            overlong_filtering=False,
            invalid_tool_call_advantage=invalid_advantage,
            malformed_thinking_advantage=malformed_advantage,
            advantage_clip_low=None,
            advantage_clip_high=None,
        )
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
        "message_level_penalty_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(data.batch_size[0])],
        fields=list(data.keys()),
    )

    asyncio.run(ctrl._advantage_stage(meta))
    return ctrl, data_plane


def test_advantage_stage_matches_scale_shape_message_penalty_and_clip_order() -> None:
    batch_size, sequence_length = 3, 4
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 0.5, 1.0]),
            "token_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]] * batch_size),
            "sample_mask": torch.ones(batch_size),
            ROLLOUT_TRUNCATED: torch.zeros(batch_size, dtype=torch.bool),
            RESPONSE_TOKEN_LENGTHS: torch.tensor([2, 3, 4]),
            INVALID_TOOL_CALL_TOKEN_MASK: torch.tensor(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, False, False],
                ]
            ),
            MALFORMED_THINKING_TOKEN_MASK: torch.tensor(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, True, False],
                ]
            ),
            GENERATED_ASSISTANT_MESSAGE_COUNT: torch.tensor([1, 1, 1]),
            INVALID_TOOL_CALL_MESSAGE_COUNT: torch.tensor([0, 1, 0]),
            MALFORMED_THINKING_MESSAGE_COUNT: torch.tensor([0, 1, 1]),
            INVALID_AND_MALFORMED_MESSAGE_COUNT: torch.tensor([0, 1, 0]),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(
            seq_logprob_error_threshold=None,
            reward_scaling=RewardScalingConfig(
                enabled=True,
                source_min=0.0,
                source_max=1.0,
                target_min=0.0,
                target_max=2.0,
            ),
            reward_shaping=RewardShapingConfig(
                enabled=True,
                overlong_buffer_length=2,
                overlong_buffer_penalty=1.0,
                max_response_length=4,
            ),
            overlong_filtering=False,
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=-3.0,
            advantage_clip_low=-2.0,
            advantage_clip_high=2.0,
        )
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
        "message_level_penalty_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    asyncio.run(ctrl._advantage_stage(meta))

    assert estimator.mask is not None
    assert torch.equal(ctrl._step_log_dict["rewards"][0], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        ctrl._step_log_dict["masked_advantages"][0],
        torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]),
    )
    assert data_plane.written_fields is not None
    advantages = data_plane.written_fields["advantages"]
    assert advantages[1, 2:].tolist() == [-2.0, -2.0]
    assert advantages[2].tolist() == [1.0, 1.0, -2.0, 1.0]
    assert ctrl._step_log_dict["message_level_penalty_metrics"] == [
        {
            "num_invalid_tool_calls": 1,
            "num_malformed_thinking": 1,
            "num_assistant_messages": 3,
            "num_raw_invalid_tool_calls": 1,
            "num_raw_malformed_thinking": 2,
            "num_invalid_and_malformed_messages": 1,
        }
    ]


@pytest.mark.parametrize(
    ("field_name", "replacement", "error_match"),
    [
        (
            INVALID_TOOL_CALL_TOKEN_MASK,
            torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.int64),
            "must have dtype torch.bool",
        ),
        (
            MALFORMED_THINKING_TOKEN_MASK,
            torch.zeros(2, 5, dtype=torch.bool),
            "shape does not match advantages",
        ),
        (
            GENERATED_ASSISTANT_MESSAGE_COUNT,
            torch.tensor([1.0, 1.0]),
            "must have an integral, non-bool dtype",
        ),
        (
            GENERATED_ASSISTANT_MESSAGE_COUNT,
            torch.ones(2, 2, dtype=torch.long),
            "must have shape",
        ),
        (
            GENERATED_ASSISTANT_MESSAGE_COUNT,
            torch.tensor([True, True]),
            "must have an integral, non-bool dtype",
        ),
        (
            INVALID_TOOL_CALL_MESSAGE_COUNT,
            torch.tensor([-1, 0]),
            "must be nonnegative",
        ),
    ],
)
def test_advantage_stage_rejects_invalid_message_penalty_field_contract(
    field_name: str,
    replacement: torch.Tensor,
    error_match: str,
) -> None:
    data = _valid_message_penalty_data()
    data[field_name] = replacement

    with pytest.raises(RuntimeError, match=error_match):
        _run_message_penalty_stage(data)


def test_advantage_stage_rejects_detector_mask_outside_generated_response() -> None:
    data = _valid_message_penalty_data()
    data[INVALID_TOOL_CALL_TOKEN_MASK][0, 0] = True

    with pytest.raises(RuntimeError, match="outside the generated response token_mask"):
        _run_message_penalty_stage(data)


def test_advantage_stage_validates_message_counts_per_row_not_only_in_total() -> None:
    data = _valid_message_penalty_data()
    # The aggregate assistant count remains two, but row 0 cannot contain its
    # invalid/malformed message while declaring zero generated assistants.
    data[GENERATED_ASSISTANT_MESSAGE_COUNT] = torch.tensor([0, 2])

    with pytest.raises(RuntimeError, match="per-row message-level"):
        _run_message_penalty_stage(data)


def test_advantage_stage_requires_zero_assistant_count_to_have_empty_mask() -> None:
    data = _valid_message_penalty_data()
    data[GENERATED_ASSISTANT_MESSAGE_COUNT] = torch.tensor([0, 1])
    data[INVALID_TOOL_CALL_MESSAGE_COUNT] = torch.tensor([0, 0])
    data[MALFORMED_THINKING_MESSAGE_COUNT] = torch.tensor([0, 0])
    data[INVALID_AND_MALFORMED_MESSAGE_COUNT] = torch.tensor([0, 0])
    data[INVALID_TOOL_CALL_TOKEN_MASK][0] = False
    data[MALFORMED_THINKING_TOKEN_MASK][0] = False

    with pytest.raises(RuntimeError, match="generated assistant count/mask presence"):
        _run_message_penalty_stage(data)


@pytest.mark.parametrize(
    ("count_field", "mask_field", "receipt_name"),
    [
        (
            INVALID_TOOL_CALL_MESSAGE_COUNT,
            INVALID_TOOL_CALL_TOKEN_MASK,
            "raw invalid-tool-call",
        ),
        (
            MALFORMED_THINKING_MESSAGE_COUNT,
            MALFORMED_THINKING_TOKEN_MASK,
            "raw malformed-thinking",
        ),
    ],
)
def test_advantage_stage_requires_detector_count_and_mask_presence_to_match(
    count_field: str,
    mask_field: str,
    receipt_name: str,
) -> None:
    data = _valid_message_penalty_data()
    data[count_field] = torch.tensor([0, 0])
    data[INVALID_AND_MALFORMED_MESSAGE_COUNT] = torch.tensor([0, 0])
    if mask_field == INVALID_TOOL_CALL_TOKEN_MASK:
        data[MALFORMED_THINKING_MESSAGE_COUNT] = torch.tensor([0, 0])
        data[MALFORMED_THINKING_TOKEN_MASK][0] = False
    else:
        data[INVALID_TOOL_CALL_MESSAGE_COUNT] = torch.tensor([0, 0])
        data[INVALID_TOOL_CALL_TOKEN_MASK][0] = False

    with pytest.raises(RuntimeError, match=receipt_name):
        _run_message_penalty_stage(data)


def test_advantage_stage_requires_overlap_count_and_mask_presence_to_match() -> None:
    data = _valid_message_penalty_data()
    data[GENERATED_ASSISTANT_MESSAGE_COUNT] = torch.tensor([2, 1])
    data[INVALID_AND_MALFORMED_MESSAGE_COUNT] = torch.tensor([0, 0])

    with pytest.raises(RuntimeError, match="invalid/malformed overlap"):
        _run_message_penalty_stage(data)


def test_advantage_stage_validates_invalid_precedence_against_message_counts() -> None:
    data = _valid_message_penalty_data()
    # The receipt says the sole malformed message is the overlapping invalid
    # message. A malformed-only token after removing invalid tokens therefore
    # proves that the producer's count and masks disagree.
    data[INVALID_TOOL_CALL_TOKEN_MASK][0] = torch.tensor([False, False, True, False])

    with pytest.raises(RuntimeError, match="post-precedence malformed-thinking"):
        _run_message_penalty_stage(data)


def test_advantage_stage_applies_seq_logprob_error_mask_before_streaming_train(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 4, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    # exp(abs(1 - 0)) > the configured threshold of 2, so only row 2
    # should be removed from the loss while the other rows remain trainable.
    generation_logprobs[2, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=2.0)
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "input_lengths" not in data_plane.selected_fields
    assert SHARED_PREFIX_GROUP_ID not in data_plane.selected_fields
    assert SHARED_PREFIX_PROMPT_LENGTHS not in data_plane.selected_fields
    assert "prev_logprobs" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert torch.equal(
        data_plane.written_fields["sample_mask"],
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )
    assert estimator.mask is not None
    assert estimator.mask[2].count_nonzero() == 0
    assert estimator.mask[[0, 1, 3]].all()
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 1
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.0)
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_reports_seq_logprob_metrics_without_masking() -> None:
    batch_size, sequence_length = 2, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    generation_logprobs[1, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 1.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=None)
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    _, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "prev_logprobs" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert "sample_mask" not in data_plane.written_fields
    assert estimator.mask is not None
    assert estimator.mask.all()
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 0
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(math.e)


def test_advantage_stage_skips_estimator_when_seq_mask_removes_whole_chunk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": torch.ones(batch_size, sequence_length),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = MagicMock()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=2.0)
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert not has_valid_training_tokens
    estimator.compute_advantage.assert_not_called()
    assert data_plane.written_fields is not None
    assert not data_plane.written_fields["sample_mask"].bool().any()
    assert torch.equal(
        data_plane.written_fields["advantages"],
        torch.zeros(batch_size, sequence_length),
    )
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_skips_preexisting_empty_mask_without_seq_threshold() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.zeros(batch_size),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = MagicMock()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=None)
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert not has_valid_training_tokens
    estimator.compute_advantage.assert_not_called()
    assert data_plane.selected_fields is not None
    assert "prev_logprobs" not in data_plane.selected_fields
    assert "generation_logprobs" not in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert "sample_mask" not in data_plane.written_fields
    assert torch.equal(
        data_plane.written_fields["advantages"],
        torch.zeros(batch_size, sequence_length),
    )
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_observes_exact_prompt_opportunity() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.tensor([[10, 11], [10, 11]]),
            SHARED_PREFIX_GROUP_ID: np.asarray(
                ["stable-group", "stable-group"], dtype=object
            ),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([2, 2]),
            "input_lengths": torch.tensor([5, 4]),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ),
            "sample_mask": torch.ones(batch_size),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = _MaskRecordingAdvantageEstimator()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(
            seq_logprob_error_threshold=None,
            num_generations_per_prompt=2,
        )
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="observe")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["group_g0", "group_g1"],
        fields=list(data.keys()),
    )

    _, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "input_lengths" in data_plane.selected_fields
    assert SHARED_PREFIX_GROUP_ID in data_plane.selected_fields
    assert SHARED_PREFIX_PROMPT_LENGTHS in data_plane.selected_fields
    assert len(ctrl._step_shared_prefix_opportunities) == 1
    opportunity = ctrl._step_shared_prefix_opportunities[0]
    assert opportunity.total_sequences == 2
    assert opportunity.complete_groups == 1
    assert opportunity.total_tokens == 9
    assert opportunity.prompt_tokens == 4
    assert opportunity.valid_loss_tokens == 5.0
    assert opportunity.shareable_prompt_tokens == 2


def test_reduce_shared_prefix_step_metrics_weights_streaming_chunks_exactly() -> None:
    first = SharedPrefixOpportunity(
        total_sequences=2,
        eligible_sequences=2,
        complete_groups=1,
        fallback_sequences=0,
        total_tokens=10,
        prompt_tokens=4,
        valid_loss_tokens=6.0,
        non_loss_suffix_tokens=0.0,
        shareable_prompt_tokens=2,
        ideal_shared_token_work=8,
        loss_ratio_upper_bound_saved_tokens=2.0,
    )
    second = SharedPrefixOpportunity(
        total_sequences=2,
        eligible_sequences=2,
        complete_groups=1,
        fallback_sequences=0,
        total_tokens=14,
        prompt_tokens=6,
        valid_loss_tokens=8.0,
        non_loss_suffix_tokens=0.0,
        shareable_prompt_tokens=3,
        ideal_shared_token_work=11,
        loss_ratio_upper_bound_saved_tokens=3.0,
    )

    metrics = _reduce_shared_prefix_step_metrics(
        [first, second],
        expected_total_tokens=24.0,
    )

    assert metrics["total_num_tokens"] == 24
    assert metrics["mean_prompt_length"] == pytest.approx(2.5)
    assert metrics["shared_prefix/total_sequences"] == 4
    assert metrics["shared_prefix/complete_groups"] == 2
    assert metrics["shared_prefix/prompt_tokens"] == 10
    assert metrics["shared_prefix/valid_loss_tokens"] == 14.0
    assert metrics["shared_prefix/shareable_prompt_tokens"] == 5
    assert metrics["shared_prefix/ideal_shared_token_work"] == 19
    assert metrics["shared_prefix/ideal_token_work_speedup"] == pytest.approx(24 / 19)

    with pytest.raises(RuntimeError, match="exact optimizer step"):
        _reduce_shared_prefix_step_metrics(
            [first, second],
            expected_total_tokens=23,
        )


def test_reduce_rollout_step_metrics_uses_exact_selected_cohort() -> None:
    common = {
        "cohort/samples": 2,
        "cohort/generated_tokens": 7,
        "cohort/total_tokens": 13,
        "cohort/duplicated_reasoning_count": 0,
        "cohort/empty_final_answer_count": 0,
        "cohort/unwanted_token_count": 0,
        "cohort/malformed_think_tag_count": 0,
    }
    train_metrics, timing_metrics = _reduce_rollout_step_metrics(
        [
            {
                **common,
                "cohort/pre_penalty_reward_sum": 2.0,
                "cohort/raw_environment_reward_sum": 2.0,
                "cohort/effort_low_sample_count": 0,
                "cohort/effort_reward_delta_sum": 0.0,
                "cohort/env_masked_sample_count": 0,
                "cohort/post_penalty_reward_sum": 1.0,
                "cohort/duplicated_reasoning_count": 1,
                "cohort/rollout_started_at_s": 10.0,
                "cohort/rollout_finished_at_s": 13.0,
                "timing/rollout/total": 3.0,
                "total_reward/mean": 0.5,
            },
            {
                **common,
                "cohort/pre_penalty_reward_sum": 1.0,
                "cohort/raw_environment_reward_sum": 1.0,
                "cohort/effort_low_sample_count": 0,
                "cohort/effort_reward_delta_sum": 0.0,
                "cohort/env_masked_sample_count": 0,
                "cohort/post_penalty_reward_sum": 1.0,
                "cohort/empty_final_answer_count": 1,
                "cohort/rollout_started_at_s": 11.0,
                "cohort/rollout_finished_at_s": 16.0,
                "timing/rollout/total": 5.0,
                "total_reward/mean": 0.5,
            },
        ]
    )

    assert train_metrics["rollout/samples"] == 4
    assert train_metrics["rollout/generated_tokens"] == 14
    assert train_metrics["rollout/total_tokens"] == 26
    assert train_metrics["raw_environment_reward"] == pytest.approx(0.75)
    assert train_metrics["pre_penalty_environment_reward"] == pytest.approx(0.75)
    assert train_metrics["effort_low_sample_count"] == 0
    assert train_metrics["effort_low_sample_rate"] == 0.0
    assert train_metrics["effort_reward_delta"] == 0.0
    assert train_metrics["num_mask_sample_filtered"] == 0
    assert train_metrics["verifier_reward"] == pytest.approx(0.5)
    assert train_metrics["reasoning_equal_to_final_answer_rate"] == pytest.approx(0.25)
    assert train_metrics["empty_final_answer_rate"] == pytest.approx(0.25)
    assert train_metrics["unwanted_token_rate"] == 0.0
    assert train_metrics["malformed_think_tag_rate"] == 0.0
    assert train_metrics["total_reward/mean"] == pytest.approx(0.5)
    assert timing_metrics["rollout_generation_cohort"] == 6.0
    assert timing_metrics["rollout_generation_active"] == 6.0
    assert timing_metrics["rollout_generation_work"] == 8.0


def test_reduce_rollout_step_metrics_does_not_average_noncomposable_summaries() -> None:
    def _receipt(*, samples: int, reward_sum: float, mean_turns: float) -> dict:
        return {
            "cohort/samples": samples,
            "cohort/generated_tokens": samples,
            "cohort/total_tokens": samples,
            "cohort/raw_environment_reward_sum": reward_sum,
            "cohort/pre_penalty_reward_sum": reward_sum,
            "cohort/effort_low_sample_count": 0,
            "cohort/effort_reward_delta_sum": 0.0,
            "cohort/env_masked_sample_count": 0,
            "cohort/post_penalty_reward_sum": reward_sum,
            "cohort/duplicated_reasoning_count": 0,
            "cohort/empty_final_answer_count": 0,
            "cohort/unwanted_token_count": 0,
            "cohort/malformed_think_tag_count": 0,
            "cohort/rollout_started_at_s": 10.0,
            "cohort/rollout_finished_at_s": 11.0,
            "timing/rollout/total": 1.0,
            "total_reward/mean": reward_sum / samples,
            "total_reward/min": reward_sum / samples,
            "total_reward/max": reward_sum / samples,
            # These are valid only inside this prompt group. Their average is
            # neither the selected cohort's statistic nor a useful bound.
            "total_reward/median": reward_sum / samples,
            "total_reward/stddev": 0.0,
            "turns_per_sample/mean": mean_turns,
            "turns_per_sample/p95": mean_turns,
            "truncation_rate": mean_turns / 2.0,
        }

    train_metrics, _ = _reduce_rollout_step_metrics(
        [
            _receipt(samples=1, reward_sum=0.0, mean_turns=0.0),
            _receipt(samples=3, reward_sum=3.0, mean_turns=2.0),
        ]
    )

    assert train_metrics["total_reward/mean"] == pytest.approx(0.75)
    assert train_metrics["total_reward/min"] == 0.0
    assert train_metrics["total_reward/max"] == 1.0
    assert train_metrics["turns_per_sample/mean"] == pytest.approx(1.5)
    assert train_metrics["truncation_rate"] == pytest.approx(0.75)
    assert "total_reward/median" not in train_metrics
    assert "total_reward/stddev" not in train_metrics
    assert "turns_per_sample/p95" not in train_metrics


def test_reduce_rollout_step_metrics_fails_closed_on_incomplete_receipt() -> None:
    with pytest.raises(RuntimeError, match="missing exact cohort metrics"):
        _reduce_rollout_step_metrics(
            [{"cohort/samples": 2, "timing/rollout/total": 1.0}]
        )


def _test_rollout_receipt(groups: int = 1) -> list[dict[str, int | float]]:
    return [
        {
            "cohort/samples": 1,
            "cohort/generated_tokens": 1,
            "cohort/total_tokens": 1,
            "cohort/pre_penalty_reward_sum": 0.0,
            "cohort/raw_environment_reward_sum": 0.0,
            "cohort/effort_low_sample_count": 0,
            "cohort/effort_reward_delta_sum": 0.0,
            "cohort/env_masked_sample_count": 0,
            "cohort/post_penalty_reward_sum": 0.0,
            "cohort/duplicated_reasoning_count": 0,
            "cohort/empty_final_answer_count": 0,
            "cohort/unwanted_token_count": 0,
            "cohort/malformed_think_tag_count": 0,
            "cohort/rollout_started_at_s": 10.0,
            "cohort/rollout_finished_at_s": 10.0,
            "timing/rollout/total": 0.0,
        }
        for _ in range(groups)
    ]


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    async def select(self, **kwargs):
        del kwargs
        return None, 0, []


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0, []
        meta = self._meta
        self._meta = None
        return meta, 1, _test_rollout_receipt()


class _EvictingSampler(_OneThenEmptySampler):
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 2

    async def select(self, **kwargs):
        meta, num_groups, rollout_metrics = await super().select(**kwargs)
        if num_groups:
            rollout_metrics.extend(_test_rollout_receipt())
        return meta, 2 if num_groups else 0, rollout_metrics


class _ChunkedSampler(_EmptySampler):
    """Assembles one step out of several single-group chunks, then goes empty.

    This is the shape the streaming path actually produces and the reason
    ``keep_train_buffers`` exists: every chunk after the first runs against an
    already-open train step.
    """

    def __init__(self, meta: KVBatchMeta, chunks: int) -> None:
        self._meta = meta
        self._remaining = chunks

    async def select(self, **kwargs):
        del kwargs
        if self._remaining == 0:
            return None, 0, []
        self._remaining -= 1
        return self._meta, 1, _test_rollout_receipt()


class _SequenceSampler(_EmptySampler):
    def __init__(self, metas: list[KVBatchMeta]) -> None:
        self._metas = list(metas)

    async def select(self, **kwargs):
        del kwargs
        if not self._metas:
            return None, 0, []
        return self._metas.pop(0), 1, _test_rollout_receipt()


class _MetricSequenceSampler(_EmptySampler):
    def __init__(
        self,
        selections: list[tuple[KVBatchMeta, dict[str, int | float]]],
    ) -> None:
        self._selections = list(selections)

    async def select(self, **kwargs):
        del kwargs
        if not self._selections:
            return None, 0, []
        meta, rollout_metrics = self._selections.pop(0)
        return meta, 1, [dict(rollout_metrics)]


class _ExactBoundsSampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta, chunks: int) -> None:
        self._meta = meta
        self._remaining = chunks
        self.bounds: list[tuple[int, int]] = []

    async def select(self, **kwargs):
        minimum = kwargs["min_prompt_groups"]
        maximum = kwargs["max_prompt_groups"]
        self.bounds.append((minimum, maximum))
        if self._remaining == 0:
            return None, 0, []
        self._remaining -= 1
        assert minimum == maximum
        return self._meta, maximum, _test_rollout_receipt(maximum)


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _NoOpTrainer:
    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        del meta

    def finish_train_step(self) -> dict:
        return {}


class _LpRecordingTrainer(_NoOpTrainer):
    """Records the ``keep_train_buffers`` flag the pump passes on each chunk."""

    def __init__(self) -> None:
        self.keep_train_buffers_calls: list[bool] = []

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        self.keep_train_buffers_calls.append(keep_train_buffers)

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        ),
        # The pump's step epilogue reads the save triggers even when saving
        # is disabled.
        checkpointing={"enabled": False, "save_period": 10},
    )
    ctrl._async_cfg = SimpleNamespace(
        min_groups_for_streaming_train=1,
        rollout_failure=SimpleNamespace(min_step_batch_fraction=0.9),
    )
    ctrl._consumed_samples = 0
    ctrl._total_valid_tokens = 0
    ctrl._timeout = TimeoutChecker(timeout=None, fit_last_save_time=True)
    ctrl._timeout.start_iterations()
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._advantage_estimator = None
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._train_dispatch_group_multiple = 1
    ctrl._partition_id = "rollout_data"
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
    ctrl._gen = SimpleNamespace(requires_kv_scale_sync=False)
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._batch_shortfall = {}
    ctrl._batch_replacements = {}
    ctrl._batch_promotions = {}
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
    }
    ctrl._step_shared_prefix_opportunities = []
    return ctrl


def test_train_pump_stops_after_rollout_exhaustion_and_buffer_drain() -> None:
    ctrl = _train_pump_controller(sampler=_EmptySampler())

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 0


def test_train_pump_logs_only_selected_rollout_cohort(monkeypatch) -> None:
    def _meta(group: str) -> KVBatchMeta:
        return KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=[f"{group}_g0", f"{group}_g1"],
            fields=[],
            sequence_lengths=[1, 1],
            tags=[{"weight_version": 0}, {"weight_version": 0}],
        )

    common = {
        "cohort/samples": 2,
        "cohort/generated_tokens": 7,
        "cohort/total_tokens": 13,
        "cohort/duplicated_reasoning_count": 0,
        "cohort/empty_final_answer_count": 0,
        "cohort/unwanted_token_count": 0,
        "cohort/malformed_think_tag_count": 0,
    }
    first = {
        **common,
        "cohort/pre_penalty_reward_sum": 2.0,
        "cohort/raw_environment_reward_sum": 2.0,
        "cohort/effort_low_sample_count": 0,
        "cohort/effort_reward_delta_sum": 0.0,
        "cohort/env_masked_sample_count": 0,
        "cohort/post_penalty_reward_sum": 1.0,
        "cohort/duplicated_reasoning_count": 1,
        "cohort/rollout_started_at_s": 10.0,
        "cohort/rollout_finished_at_s": 13.0,
        "timing/rollout/total": 3.0,
    }
    second = {
        **common,
        "cohort/pre_penalty_reward_sum": 1.0,
        "cohort/raw_environment_reward_sum": 1.0,
        "cohort/effort_low_sample_count": 0,
        "cohort/effort_reward_delta_sum": 0.0,
        "cohort/env_masked_sample_count": 0,
        "cohort/post_penalty_reward_sum": 1.0,
        "cohort/empty_final_answer_count": 1,
        "cohort/rollout_started_at_s": 11.0,
        "cohort/rollout_finished_at_s": 16.0,
        "timing/rollout/total": 5.0,
    }
    # This third receipt is deliberately never selected into the two-group step.
    sentinel = {
        **common,
        "cohort/samples": 1000,
        "cohort/generated_tokens": 999999,
        "cohort/pre_penalty_reward_sum": 999999.0,
        "cohort/raw_environment_reward_sum": 999999.0,
        "cohort/effort_low_sample_count": 0,
        "cohort/effort_reward_delta_sum": 0.0,
        "cohort/env_masked_sample_count": 0,
        "cohort/post_penalty_reward_sum": 999999.0,
        "cohort/rollout_started_at_s": 0.0,
        "cohort/rollout_finished_at_s": 999999.0,
        "timing/rollout/total": 999999.0,
    }
    sampler = _MetricSequenceSampler(
        [
            (_meta("first"), first),
            (_meta("second"), second),
            (_meta("sentinel"), sentinel),
        ]
    )
    ctrl = _train_pump_controller(sampler=sampler)

    async def _advantage_stage(meta):
        ctrl._step_log_dict["rewards"].append(torch.tensor([0.5, 0.5]))
        ctrl._step_log_dict["masked_advantages"].append(torch.tensor([0.0]))
        return meta, True

    ctrl._advantage_stage = _advantage_stage
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_call, timing_call = ctrl._logger.log_metrics.call_args_list
    assert train_call.kwargs["prefix"] == "train"
    train_metrics = train_call.args[0]
    assert train_metrics["rollout/samples"] == 4
    assert train_metrics["rollout/generated_tokens"] == 14
    assert train_metrics["rollout/total_tokens"] == 26
    assert train_metrics["raw_environment_reward"] == pytest.approx(0.75)
    assert train_metrics["pre_penalty_environment_reward"] == pytest.approx(0.75)
    assert train_metrics["verifier_reward"] == pytest.approx(0.5)
    assert train_metrics["reward"] == pytest.approx(0.5)
    assert train_metrics["reward_processing_delta"] == pytest.approx(0.0)
    assert train_metrics["reasoning_equal_to_final_answer_rate"] == pytest.approx(0.25)
    assert train_metrics["empty_final_answer_rate"] == pytest.approx(0.25)
    assert train_metrics["unwanted_token_rate"] == 0.0
    assert train_metrics["malformed_think_tag_rate"] == 0.0

    assert timing_call.kwargs["prefix"] == "timing/train"
    assert timing_call.args[0]["rollout_generation_cohort"] == 6.0
    assert timing_call.args[0]["rollout_generation_active"] == 6.0
    assert timing_call.args[0]["rollout_generation_work"] == 8.0
    assert len(sampler._selections) == 1


@pytest.mark.parametrize("mode", ["disabled", "observe"])
def test_non_train_mode_preserves_greedy_streaming_bounds(mode: str) -> None:
    assert (
        _resolve_train_dispatch_group_multiple(
            shared_prefix_mode=mode,
            num_prompts_per_step=5,
            data_parallel_size=4,
        )
        == 1
    )
    assert _train_selection_group_bounds(
        remaining_groups=8,
        configured_min_groups=3,
        group_multiple=1,
    ) == (3, 8)


def test_shared_prefix_train_rounds_streaming_minimum_to_exact_dp_width() -> None:
    assert (
        _resolve_train_dispatch_group_multiple(
            shared_prefix_mode="train",
            num_prompts_per_step=8,
            data_parallel_size=2,
        )
        == 2
    )
    assert _train_selection_group_bounds(
        remaining_groups=8,
        configured_min_groups=3,
        group_multiple=2,
    ) == (4, 4)
    assert _train_selection_group_bounds(
        remaining_groups=2,
        configured_min_groups=3,
        group_multiple=2,
    ) == (2, 2)


def test_shared_prefix_train_rejects_non_dp_divisible_step_and_shortfall() -> None:
    with pytest.raises(ValueError, match="num_prompts_per_step.*divisible"):
        _resolve_train_dispatch_group_multiple(
            shared_prefix_mode="train",
            num_prompts_per_step=7,
            data_parallel_size=2,
        )
    with pytest.raises(RuntimeError, match="remaining=3, DP=2"):
        _train_selection_group_bounds(
            remaining_groups=3,
            configured_min_groups=1,
            group_multiple=2,
        )


def test_shared_prefix_train_rejects_buffer_smaller_than_rounded_claim() -> None:
    with pytest.raises(ValueError, match="rounds the streaming claim to 4"):
        _validate_train_dispatch_buffer_capacity(
            num_prompts_per_step=8,
            configured_min_groups=3,
            group_multiple=2,
            max_buffered_rollouts=3,
        )

    _validate_train_dispatch_buffer_capacity(
        num_prompts_per_step=8,
        configured_min_groups=3,
        group_multiple=2,
        max_buffered_rollouts=4,
    )


def test_sampler_selection_rejects_legacy_two_value_contract() -> None:
    with pytest.raises(RuntimeError, match="legacy two-value sampler contract"):
        _unpack_sampler_selection((None, 0))


def test_sampler_selection_requires_receipt_list() -> None:
    with pytest.raises(RuntimeError, match="rollout_metric_receipts as a list"):
        _unpack_sampler_selection((None, 0, ()))


def _sampler_selection_meta(samples: int = 1) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{idx}" for idx in range(samples)],
    )


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        ((None, 1, [{"cohort/samples": 1}]), "returned no metadata"),
        ((None, 0, [{"cohort/samples": 1}]), "returned no metadata"),
        ((_sampler_selection_meta(), 0, []), "non-positive num_groups"),
        ((_sampler_selection_meta(), -1, []), "non-positive num_groups"),
        (
            (_sampler_selection_meta(), 1, []),
            "one rollout metric receipt per selected prompt group",
        ),
        (
            (_sampler_selection_meta(), 2, [{}, {}]),
            "fewer samples than selected prompt groups",
        ),
        (({"sample_ids": ["sample-0"]}, 1, [{}]), "non-KVBatchMeta"),
    ],
)
def test_sampler_selection_rejects_inconsistent_empty_count_and_meta(
    selection: object,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        _unpack_sampler_selection(selection)


def test_sampler_selection_accepts_exact_receipt_contract() -> None:
    receipts = [{"cohort/samples": 1}]
    meta = _sampler_selection_meta()
    assert _unpack_sampler_selection((meta, 1, receipts)) == (meta, 1, receipts)
    assert _unpack_sampler_selection((None, 0, [])) == (None, 0, [])


def test_shared_prefix_train_pump_claims_only_exact_dp_divisible_chunks(
    monkeypatch,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a_g0", "a_g1", "b_g0", "b_g1"],
        fields=[],
        sequence_lengths=[1, 1, 1, 1],
        tags=[{"weight_version": 0} for _ in range(4)],
    )
    sampler = _ExactBoundsSampler(meta, chunks=2)
    ctrl = _train_pump_controller(sampler=sampler)
    ctrl._master_config.grpo.num_prompts_per_step = 4
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="train")
    ctrl._train_dispatch_group_multiple = 2
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = {}
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert sampler.bounds == [(2, 2), (2, 2)]
    assert trainer.train_microbatches_from_meta.call_count == 2
    assert ctrl._train_steps == 1


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))

    with pytest.raises(
        RuntimeError,
        match=(
            r"rollout exhausted before a complete training step was assembled: "
            r"dispatched 1/2 prompt groups"
        ),
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))


class _DroppingSampler(_OneThenEmptySampler):
    """Yields one group, then reports the second as never coming.

    Stands in for a prompt that was stamped for this step and then given up on: the
    credit lands while the pump is already waiting for the group, which is the only
    way it happens in a real run and the case that waits forever without the fix.

    ``select`` validates its bounds through the real sampler's own check, so a pump
    that asks for a non-positive batch fails here exactly as it would in production
    rather than being quietly tolerated by a permissive fake.
    """

    def __init__(self, meta: KVBatchMeta, *, credit_in_evict: bool) -> None:
        super().__init__(meta)
        self._credit_in_evict = credit_in_evict
        self.ctrl = None

    def _credit(self) -> None:
        self.ctrl._batch_shortfall[self.ctrl._trainer_version] = 1

    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        # The window between the pump's two reads of the step target. A credit landing
        # here shrinks the target after the loop condition has already passed.
        if self._credit_in_evict and self._meta is None:
            self._credit()
        return 0

    async def select(self, **kwargs):
        BaseSampler._validate_group_bounds(
            kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
        )
        meta, num_groups, rollout_metrics = await super().select(**kwargs)
        if meta is None and not self._credit_in_evict:
            self._credit()
        return meta, num_groups, rollout_metrics


def _dropping_controller(*, credit_in_evict: bool):
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    sampler = _DroppingSampler(meta, credit_in_evict=credit_in_evict)
    ctrl = _train_pump_controller(sampler=sampler)
    sampler.ctrl = ctrl
    # ceil(0.9 * 2) is 2, so the harness's default floor forbids any short step at
    # all; at 0.5 the floor is 1 and closing on the one group it got is legal.
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    # Rollouts are still running, so the "rollout exhausted" escape is unavailable:
    # a pump that does not act on the credit waits on a group nobody is generating.
    ctrl._rollout_exhausted.clear()
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    return ctrl


@pytest.mark.parametrize("credit_in_evict", [False, True])
def test_train_pump_closes_a_step_short_when_a_stamped_prompt_is_dropped(
    monkeypatch, credit_in_evict
) -> None:
    """Both windows a shortfall can be credited in have to close the step.

    Credited before the loop condition is re-read, the target simply falls to what is
    already dispatched. Credited after it, between the two reads, the batch the pump
    would ask for is empty -- which the sampler rejects, so the pump has to notice
    instead of asking. Either way the step trains on what it got.
    """
    ctrl = _dropping_controller(credit_in_evict=credit_in_evict)
    ctrl._batch_replacements = {0: 1}
    ctrl._batch_promotions = {0: 2, 3: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    # One group, not the configured two: the step is what shrank, not the run.
    assert ctrl._consumed_samples == 1
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["dropped_prompt_groups"] == 1
    assert train_metrics["replaced_prompt_groups"] == 1
    assert train_metrics["promoted_prompt_groups"] == 2
    # Read against version_during_step, not the already-incremented _trainer_version,
    # which would report 0 for every step forever.
    assert ctrl._trainer_version == 1
    # This step's counts are pruned; a later step's survive to be reported by it.
    assert ctrl._batch_shortfall == {}
    assert ctrl._batch_replacements == {}
    assert ctrl._batch_promotions == {3: 1}


def test_train_pump_prunes_stamps_older_than_the_step_that_just_closed(
    monkeypatch,
) -> None:
    """A straggler credited for a step that already closed must not outlive it.

    Popping only the current step's entry would leak those, and they are unreachable
    afterwards: the target is only ever read for the step being assembled.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 3}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    ctrl._trainer_version = 3
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    ctrl._batch_shortfall = {2: 1, 3: 1, 5: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._batch_shortfall == {5: 1}


def test_train_pump_rejects_step_with_no_valid_training_chunks() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._master_config.grpo.num_prompts_per_step = 1
    ctrl._advantage_stage = AsyncMock(return_value=(meta, False))
    trainer = MagicMock(spec=_NoOpTrainer)
    ctrl._trainer = trainer

    with pytest.raises(
        RuntimeError,
        match="no valid response tokens after filtering",
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    trainer.prepare_for_training.assert_called_once_with()
    trainer.begin_train_step.assert_not_called()
    trainer.train_microbatches_from_meta.assert_not_called()
    trainer.finish_train_step.assert_not_called()


def test_train_pump_skips_empty_chunk_and_trains_later_valid_chunk(
    monkeypatch,
) -> None:
    empty_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["empty-sample"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    valid_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["valid-sample"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_SequenceSampler([empty_meta, valid_meta]))
    ctrl._advantage_stage = AsyncMock(
        side_effect=[
            (empty_meta, False),
            (valid_meta, True),
        ]
    )
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = {}
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert trainer.prepare_for_training.call_count == 2
    trainer.begin_train_step.assert_called_once_with(None)
    trainer.train_microbatches_from_meta.assert_called_once_with(valid_meta)
    trainer.finish_train_step.assert_called_once_with()
    assert ctrl._train_steps == 1


def test_train_pump_observe_metrics_exclude_fully_filtered_chunk(
    monkeypatch,
) -> None:
    """Observed token work must cover only chunks that entered the optimizer step."""
    empty_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["empty-sample"],
        fields=[],
        sequence_lengths=[5],
        tags=[{"weight_version": 0}],
    )
    valid_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["valid-sample"],
        fields=[],
        sequence_lengths=[3],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_SequenceSampler([empty_meta, valid_meta]))
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="observe")
    ctrl._advantage_stage = AsyncMock(
        side_effect=[
            (empty_meta, False),
            (valid_meta, True),
        ]
    )
    ctrl._step_shared_prefix_opportunities = [
        SharedPrefixOpportunity(
            total_sequences=1,
            eligible_sequences=0,
            complete_groups=0,
            fallback_sequences=1,
            total_tokens=3,
            prompt_tokens=1,
            valid_loss_tokens=2.0,
            non_loss_suffix_tokens=0.0,
            shareable_prompt_tokens=0,
            ideal_shared_token_work=3,
            loss_ratio_upper_bound_saved_tokens=0.0,
        )
    ]
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = {}
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["total_num_tokens"] == 3
    assert train_metrics["shared_prefix/total_tokens"] == 3


def test_train_pump_logs_nonzero_stale_group_metrics(monkeypatch) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0", "sample-1"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_EvictingSampler(meta))
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["evicted_stale_prompt_groups"] == 2
    assert train_metrics["aborted_stale_inflight_groups"] == 1


def test_train_pump_keeps_train_buffers_once_the_step_is_open(monkeypatch) -> None:
    """The logprob detour between chunks must not offload the trainer's grad
    buffers, because mcore's offload frees the gradients the earlier chunks of
    this step accumulated rather than copying them out.

    First chunk: no step open yet, nothing to preserve, so the offload is still
    worth taking. Every later chunk: step open, buffers must stay resident.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    # num_prompts_per_step is 2 in the harness, so two single-group chunks close
    # the step.
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = True
    trainer = _LpRecordingTrainer()
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert trainer.keep_train_buffers_calls == [False, True]
