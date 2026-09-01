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
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
from nemo_rl.algorithms.grpo import (GRPOConfig, RewardScalingConfig,
                                     _initial_grpo_save_state)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.reward_functions import RewardShapingConfig
from nemo_rl.algorithms.shared_prefix_metrics import SharedPrefixOpportunity
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor, _reduce_shared_prefix_step_metrics,
    _resolve_train_dispatch_group_multiple, _string_object_field,
    _train_selection_group_bounds, _validate_train_dispatch_buffer_capacity)
from nemo_rl.algorithms.single_controller_utils.config import (AdvantageConfig,
                                                               AsyncRLConfig,
                                                               MasterConfig)
from nemo_rl.algorithms.single_controller_utils.utils import (
    reduce_advantage_pump_metrics,
)
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID, SHARED_PREFIX_PROMPT_LENGTHS)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import RolloutStats
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


def _run_lifecycle_controller(
    *,
    train_error: Exception | None = None,
    weight_shutdown_error: Exception | None = None,
    logger_finish_error: Exception | None = None,
    checkpointer_shutdown_error: Exception | None = None,
):
    """Build a minimal in-process actor that records run() teardown ordering."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    events: list[str] = []

    async def sync_weights() -> None:
        events.append("initial_sync")

    async def restore_step() -> None:
        return None

    async def idle_pump(name: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            events.append(f"{name}_joined")

    async def train_pump() -> None:
        try:
            await asyncio.sleep(0)
            ctrl._train_steps = 7
            ctrl._trainer_version = 7
            if train_error is not None:
                raise train_error
        finally:
            events.append("train_joined")

    def weight_sync_shutdown() -> None:
        events.append("weight_sync_shutdown")
        if weight_shutdown_error is not None:
            raise weight_shutdown_error

    def logger_finish() -> None:
        events.append("logger_finish")
        if logger_finish_error is not None:
            raise logger_finish_error

    def checkpointer_shutdown() -> None:
        events.append("checkpointer_shutdown")
        if checkpointer_shutdown_error is not None:
            raise checkpointer_shutdown_error

    logger = MagicMock()
    logger.log_metrics.side_effect = lambda *args, **kwargs: events.append("terminal_log")
    logger.finish.side_effect = logger_finish

    ctrl._sync_weights = sync_weights
    ctrl._maybe_restore_rollout_admission_state = restore_step
    ctrl._maybe_restore_replay_buffer = restore_step
    ctrl._finalize_rollout_admission_restore = lambda: None
    ctrl._discard_legacy_unstamped_replay = restore_step
    ctrl._maybe_restore_replacement_reserve = restore_step
    ctrl._rollout_pump = lambda: idle_pump("rollout")
    ctrl._train_pump = train_pump
    ctrl._stall_watchdog_pump = lambda: idle_pump("watchdog")
    ctrl._gen_fleet = None
    ctrl._rollout_manager = SimpleNamespace(stats=RolloutStats(committed=3, skipped=1))
    ctrl._train_steps = 0
    ctrl._trainer_version = 0
    ctrl._weight_synchronizer = SimpleNamespace(shutdown=weight_sync_shutdown)
    ctrl._logger = logger
    ctrl._checkpointer = SimpleNamespace(shutdown=checkpointer_shutdown)

    async def run_controller():
        return await asyncio.wait_for(ctrl.run(), timeout=1.0)

    return ctrl, logger, events, run_controller


def test_run_logs_final_rollout_snapshot_after_pumps_join_and_before_finish() -> None:
    ctrl, logger, events, run_controller = _run_lifecycle_controller()

    result = asyncio.run(run_controller())

    assert result == {"train_steps": 7, "trainer_version": 7}
    logger.log_metrics.assert_called_once()
    metrics = logger.log_metrics.call_args.args[0]
    assert metrics == {
        **ctrl._rollout_manager.stats.as_metrics(),
        "rollout/counters_finalized": 1.0,
    }
    assert list(metrics)[-1] == "rollout/counters_finalized"
    assert logger.log_metrics.call_args.kwargs == {
        "step": 7,
        "step_finished": True,
    }
    terminal_index = events.index("terminal_log")
    for pump in ("rollout_joined", "train_joined", "watchdog_joined"):
        assert events.index(pump) < terminal_index
    assert terminal_index < events.index("logger_finish")
    logger.finish.assert_called_once_with()


def test_run_failure_skips_finalized_marker_but_still_finishes_logger() -> None:
    error = RuntimeError("injected train failure")
    _, logger, events, run_controller = _run_lifecycle_controller(train_error=error)

    with pytest.raises(RuntimeError, match="injected train failure"):
        asyncio.run(run_controller())

    logger.log_metrics.assert_not_called()
    finish_index = events.index("logger_finish")
    for pump in ("rollout_joined", "train_joined", "watchdog_joined"):
        assert events.index(pump) < finish_index
    logger.finish.assert_called_once_with()


def test_run_primary_failure_survives_all_teardown_failures() -> None:
    """Cleanup remains best-effort without replacing the pump traceback."""
    primary_error = RuntimeError("injected train failure")
    _, logger, events, run_controller = _run_lifecycle_controller(
        train_error=primary_error,
        weight_shutdown_error=ValueError("injected weight shutdown failure"),
        logger_finish_error=OSError("injected logger shutdown failure"),
        checkpointer_shutdown_error=LookupError("injected checkpointer shutdown failure"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_controller())

    assert exc_info.value is primary_error
    logger.log_metrics.assert_not_called()
    assert events[-3:] == [
        "weight_sync_shutdown",
        "logger_finish",
        "checkpointer_shutdown",
    ]
    logger.finish.assert_called_once_with()


def test_run_success_retains_first_teardown_failure_and_attempts_later_cleanup() -> None:
    """A successful pump run surfaces teardown failure after all attempts."""
    logger_error = OSError("injected logger shutdown failure")
    checkpointer_error = LookupError("injected checkpointer shutdown failure")
    _, logger, events, run_controller = _run_lifecycle_controller(
        logger_finish_error=logger_error,
        checkpointer_shutdown_error=checkpointer_error,
    )

    with pytest.raises(OSError) as exc_info:
        asyncio.run(run_controller())

    assert exc_info.value is logger_error
    logger.log_metrics.assert_called_once()
    assert events[-3:] == [
        "terminal_log",
        "logger_finish",
        "checkpointer_shutdown",
    ]


def test_run_success_surfaces_checkpointer_shutdown_failure() -> None:
    """The last teardown stage is not silently discarded on success."""
    checkpointer_error = LookupError("injected checkpointer shutdown failure")
    _, logger, events, run_controller = _run_lifecycle_controller(checkpointer_shutdown_error=checkpointer_error)

    with pytest.raises(LookupError) as exc_info:
        asyncio.run(run_controller())

    assert exc_info.value is checkpointer_error
    logger.log_metrics.assert_called_once()
    assert events[-2:] == ["logger_finish", "checkpointer_shutdown"]


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
    setup_metrics = SetupTimingMetrics(generation_init_time_s=1.5, policy_init_time_s=2.5)
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

    logger.log_metrics.assert_called_once_with(setup_metrics.to_metrics_dict(), step=0, prefix="timing/setup")


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
    ctrl._async_cfg = AsyncRLConfig(recompute_kv_cache_after_weight_updates=recompute_kv_cache)
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(return_value=True),
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


def test_sync_weights_drains_gym_before_refit_and_invalidates_cache() -> None:
    async def _main() -> None:
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._async_cfg = AsyncRLConfig(recompute_kv_cache_after_weight_updates=False)
        ctrl._rollout_permitted = asyncio.Event()
        ctrl._rollout_permitted.set()
        ctrl._inflight_registry_changed = asyncio.Event()
        ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
        ctrl._gen = SimpleNamespace(
            invalidate_kv_cache=MagicMock(return_value=True),
            requires_kv_scale_sync=False,
        )
        ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
        ctrl._trainer_version = 3
        ctrl._master_config = SimpleNamespace(
            env={"should_use_nemo_gym": True},
            policy={
                "generation": {
                    "backend": "vllm",
                    "vllm_cfg": {
                        "async_engine": True,
                        "expose_http_server": True,
                    },
                }
            },
        )

        release_generation = asyncio.Event()

        async def _inflight_generation() -> None:
            await release_generation.wait()
            ctrl._inflight_by_group_id.pop("gym-group")
            ctrl._inflight_registry_changed.set()

        inflight_task = asyncio.create_task(_inflight_generation())
        ctrl._inflight_by_group_id = {"gym-group": (inflight_task, 2)}

        sync_task = asyncio.create_task(ctrl._sync_weights())
        await asyncio.sleep(0)

        assert not ctrl._rollout_permitted.is_set()
        ctrl._weight_synchronizer.sync_weights.assert_not_called()

        release_generation.set()
        assert await sync_task == 0
        await inflight_task

        ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
        ctrl._gen.invalidate_kv_cache.assert_called_once_with()
        ctrl._rollout_manager.set_weight_version.assert_called_once_with(3)
        assert ctrl._rollout_permitted.is_set()

    asyncio.run(_main())


def test_sync_weights_fails_closed_when_cache_invalidation_fails() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(recompute_kv_cache_after_weight_updates=True)
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(return_value=False),
        requires_kv_scale_sync=False,
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    ctrl._inflight_by_group_id = {}
    ctrl._master_config = SimpleNamespace(env={})

    with pytest.raises(RuntimeError, match="cache invalidation failed"):
        asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    ctrl._rollout_manager.set_weight_version.assert_not_called()
    assert not ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(return_value=True),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}}))
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
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales={"layer.0": 0.5})


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
        self.rewards: torch.Tensor | None = None

    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del kwargs
        self.mask = mask.clone()
        self.rewards = rewards.clone()
        return rewards.unsqueeze(-1).expand_as(mask).clone()


def _reward_stage_controller(
    data: TensorDict,
    *,
    reward_scaling: RewardScalingConfig,
    reward_shaping: RewardShapingConfig,
    log_train_data: bool = False,
):
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
            reward_scaling=reward_scaling,
            reward_shaping=reward_shaping,
        )
    )
    ctrl._shared_prefix_training_config = SimpleNamespace(mode="disabled")
    ctrl._step_shared_prefix_opportunities = []
    ctrl._step_log_dict = {
        "rewards": [],
        "verifier_rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    ctrl._log_train_data = log_train_data
    ctrl._step_train_data_records = []
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(data.batch_size[0])],
        fields=list(data.keys()),
    )
    return ctrl, data_plane, estimator, meta


def test_advantage_stage_disabled_reward_processing_preserves_fields() -> None:
    batch_size, sequence_length = 2, 4
    raw_rewards = torch.tensor([0.25, 0.75])
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
            "total_reward": raw_rewards,
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
        },
        batch_size=[batch_size],
    )
    ctrl, data_plane, estimator, meta = _reward_stage_controller(
        data,
        reward_scaling=RewardScalingConfig(enabled=False),
        reward_shaping=RewardShapingConfig(enabled=False),
    )

    result_meta, _ = asyncio.run(ctrl._advantage_stage(meta))

    assert data_plane.selected_fields is not None
    assert "truncated" not in data_plane.selected_fields
    assert "response_token_lengths" not in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert "total_reward" not in data_plane.written_fields
    assert set(result_meta.fields or []) == set(meta.fields or []) | {"advantages"}
    assert estimator.rewards is not None
    assert torch.equal(estimator.rewards, raw_rewards)
    assert torch.equal(ctrl._step_log_dict["rewards"][0], raw_rewards)
    assert torch.equal(ctrl._step_log_dict["verifier_rewards"][0], raw_rewards)


def test_advantage_stage_applies_dapo_shaping_and_persists_shaped_rewards() -> None:
    batch_size, sequence_length = 4, 4
    data = TensorDict(
        {
            "input_ids": torch.zeros(batch_size, sequence_length, dtype=torch.long),
            "input_lengths": torch.full((batch_size,), sequence_length),
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
            "total_reward": torch.zeros(batch_size),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 1.0]] * batch_size),
            "sample_mask": torch.ones(batch_size),
            "generation_logprobs": torch.zeros(batch_size, sequence_length),
            "truncated": torch.tensor([True, True, True, False]),
            "response_token_lengths": torch.tensor([256, 256, 256, 149]),
        },
        batch_size=[batch_size],
    )
    ctrl, data_plane, estimator, meta = _reward_stage_controller(
        data,
        reward_scaling=RewardScalingConfig(enabled=False),
        reward_shaping=RewardShapingConfig(
            enabled=True,
            max_response_length=256,
            overlong_buffer_length=128,
            overlong_buffer_penalty=1.0,
        ),
        log_train_data=True,
    )

    result_meta, _ = asyncio.run(ctrl._advantage_stage(meta))

    expected = torch.tensor([-1.0, -1.0, -1.0, -0.1640625])
    assert data_plane.selected_fields is not None
    assert "truncated" in data_plane.selected_fields
    assert "response_token_lengths" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert "prev_logprobs" not in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert torch.equal(data_plane.written_fields["total_reward"], expected)
    assert "total_reward" in (result_meta.fields or [])
    assert estimator.rewards is not None
    assert torch.equal(estimator.rewards, expected)
    assert torch.equal(ctrl._step_log_dict["rewards"][0], expected)
    assert torch.equal(ctrl._step_log_dict["verifier_rewards"][0], torch.zeros(4))
    assert [row["rewards"] for row in ctrl._step_train_data_records] == pytest.approx(expected.tolist())
    assert [row["verifier_rewards"] for row in ctrl._step_train_data_records] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert all(row["generation_logprobs"] == [0.0] * sequence_length for row in ctrl._step_train_data_records)
    assert all("prev_logprobs" not in row for row in ctrl._step_train_data_records)


def test_train_data_capture_preserves_zero_completion_row() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 0, 0], [20, 21, 101, 102, 0]]),
            "input_lengths": torch.tensor([3, 4]),
            "prompt_ids_for_adv": torch.tensor([[10, 11, 12, 0, 0], [20, 21, 0, 0, 0]]),
            "generation_logprobs": torch.zeros(batch_size, sequence_length),
        },
        batch_size=[batch_size],
    )
    ctrl, _data_plane, _estimator, meta = _reward_stage_controller(
        data,
        reward_scaling=RewardScalingConfig(enabled=False),
        reward_shaping=RewardShapingConfig(enabled=False),
        log_train_data=True,
    )
    token_mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]])

    ctrl._capture_train_data_chunk(
        meta=meta,
        data=data,
        rewards=torch.tensor([0.0, 1.0]),
        verifier_rewards=torch.tensor([0.0, 1.0]),
        token_mask=token_mask,
        sample_mask=torch.ones(batch_size),
        advantages=torch.zeros(batch_size, sequence_length),
    )

    assert len(ctrl._step_train_data_records) == 2
    assert ctrl._step_train_data_records[0]["prompt_ids"] == [10, 11, 12]
    assert ctrl._step_train_data_records[0]["token_loss_mask"] == [0.0] * sequence_length
    assert ctrl._step_train_data_records[1]["prompt_ids"] == [20, 21]


def test_advantage_stage_scales_then_applies_stop_properly_penalty() -> None:
    batch_size, sequence_length = 2, 4
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
            "total_reward": torch.tensor([0.25, 0.5]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "truncated": torch.tensor([False, True]),
            "response_token_lengths": torch.tensor([12, 12]),
        },
        batch_size=[batch_size],
    )
    ctrl, data_plane, estimator, meta = _reward_stage_controller(
        data,
        reward_scaling=RewardScalingConfig(
            enabled=True,
            source_min=0.0,
            source_max=1.0,
            target_min=0.0,
            target_max=2.0,
        ),
        reward_shaping=RewardShapingConfig(
            enabled=True,
            stop_properly_penalty_coef=0.5,
        ),
    )

    asyncio.run(ctrl._advantage_stage(meta))

    # Scaling yields [0.5, 1.0], then only the truncated row is halved.
    expected = torch.tensor([0.5, 0.5])
    assert data_plane.written_fields is not None
    assert torch.equal(data_plane.written_fields["total_reward"], expected)
    assert estimator.rewards is not None
    assert torch.equal(estimator.rewards, expected)
    assert torch.equal(
        ctrl._step_log_dict["verifier_rewards"][0], torch.tensor([0.25, 0.5])
    )


def test_advantage_stage_reward_metrics_match_trained_streaming_cohort() -> None:
    sequence_length = 4

    def make_data(rewards: torch.Tensor, sample_mask: torch.Tensor) -> TensorDict:
        batch_size = int(rewards.shape[0])
        return TensorDict(
            {
                "input_ids": torch.arange(
                    batch_size * sequence_length, dtype=torch.long
                ).reshape(batch_size, sequence_length),
                "input_lengths": torch.full((batch_size,), sequence_length),
                "prompt_ids_for_adv": torch.zeros(
                    batch_size, sequence_length, dtype=torch.long
                ),
                "total_reward": rewards,
                "token_mask": torch.tensor(
                    [[0.0, 1.0, 1.0, 1.0]] * batch_size
                ),
                "sample_mask": sample_mask,
                "generation_logprobs": torch.zeros(batch_size, sequence_length),
            },
            batch_size=[batch_size],
        )

    filtered_data = make_data(
        torch.tensor([100.0, 200.0]),
        torch.zeros(2),
    )
    ctrl, _filtered_plane, _estimator, filtered_meta = _reward_stage_controller(
        filtered_data,
        reward_scaling=RewardScalingConfig(
            enabled=True,
            source_min=0.0,
            source_max=4.0,
            target_min=0.0,
            target_max=8.0,
        ),
        reward_shaping=RewardShapingConfig(enabled=False),
        log_train_data=True,
    )

    _, filtered_has_valid_tokens = asyncio.run(
        ctrl._advantage_stage(filtered_meta)
    )

    assert not filtered_has_valid_tokens
    assert ctrl._step_log_dict["rewards"] == []
    assert ctrl._step_log_dict["verifier_rewards"] == []
    assert ctrl._step_train_data_records == []

    partial_data = make_data(
        torch.tensor([1.0, 3.0, 4.0]),
        torch.tensor([1.0, 0.0, 1.0]),
    )
    partial_plane = _AdvantageDataPlane(partial_data)
    ctrl._dp_client = partial_plane
    partial_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["partial-0", "partial-1", "partial-2"],
        fields=list(partial_data.keys()),
    )

    _, partial_has_valid_tokens = asyncio.run(ctrl._advantage_stage(partial_meta))

    assert partial_has_valid_tokens
    assert len(ctrl._step_train_data_records) == 3
    assert ctrl._step_train_data_records[1]["sample_loss_mask"] == 0.0
    assert ctrl._step_train_data_records[1]["rewards"] == pytest.approx(6.0)
    assert ctrl._step_train_data_records[1]["verifier_rewards"] == pytest.approx(
        3.0
    )
    metrics = reduce_advantage_pump_metrics(**ctrl._step_log_dict)
    assert metrics["reward"] == pytest.approx(16.0 / 3.0)
    assert metrics["verifier_reward"] == pytest.approx(8.0 / 3.0)
    assert metrics["reward_processing_delta"] == pytest.approx(8.0 / 3.0)


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
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
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
    ctrl._master_config = SimpleNamespace(grpo=SimpleNamespace(seq_logprob_error_threshold=2.0))
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
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
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
    ctrl._master_config = SimpleNamespace(grpo=SimpleNamespace(seq_logprob_error_threshold=None))
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
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
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
    ctrl._master_config = SimpleNamespace(grpo=SimpleNamespace(seq_logprob_error_threshold=2.0))
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
    assert ctrl._step_log_dict["rewards"] == []
    assert ctrl._step_log_dict.get("verifier_rewards", []) == []
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_skips_preexisting_empty_mask_without_seq_threshold() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(batch_size, sequence_length, dtype=torch.long),
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
    ctrl._master_config = SimpleNamespace(grpo=SimpleNamespace(seq_logprob_error_threshold=None))
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
    assert ctrl._step_log_dict["rewards"] == []
    assert ctrl._step_log_dict.get("verifier_rewards", []) == []
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_observes_exact_prompt_opportunity() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.tensor([[10, 11], [10, 11]]),
            SHARED_PREFIX_GROUP_ID: np.asarray(["stable-group", "stable-group"], dtype=object),
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


def test_advantage_stage_captures_and_flushes_exact_train_payload() -> None:
    batch_size, sequence_length = 2, 5
    input_ids = torch.tensor([[10, 11, 101, 102, 0], [10, 11, 201, 202, 203]], dtype=torch.long)
    token_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
    data = TensorDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([4, 5]),
            "prompt_ids_for_adv": torch.tensor([[10, 11, 0, 0, 0], [10, 11, 0, 0, 0]]),
            SHARED_PREFIX_GROUP_ID: np.asarray(["stable-group", "stable-group"], dtype=object),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([2, 2]),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.tensor([[0.0, 0.0, -0.1, -0.2, 0.0], [0.0, 0.0, -0.3, -0.4, -0.5]]),
            "generation_logprobs": torch.tensor([[0.0, 0.0, -0.11, -0.21, 0.0], [0.0, 0.0, -0.31, -0.41, -0.51]]),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = _MaskRecordingAdvantageEstimator()
    ctrl._policy_logprobs_required = True
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
    ctrl._log_train_data = True
    ctrl._step_train_data_records = []
    ctrl._logger = SimpleNamespace(log_batched_dict_as_jsonl=MagicMock())
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["group_g0", "group_g1"],
        fields=list(data.keys()),
    )

    _, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "input_ids" in data_plane.selected_fields
    assert "input_lengths" in data_plane.selected_fields
    assert len(ctrl._step_train_data_records) == 2
    first, second = ctrl._step_train_data_records
    assert first["sample_id"] == "group_g0"
    assert first["token_ids"] == [10, 11, 101, 102, 0]
    assert first["input_lengths"] == 4
    assert first["token_loss_mask"] == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert first["rewards"] == 1.0
    assert first[SHARED_PREFIX_GROUP_ID] == "stable-group"
    assert first[SHARED_PREFIX_PROMPT_LENGTHS] == 2
    assert first["prompt_ids"] == [10, 11]
    assert first["generation_logprobs"][2] == pytest.approx(-0.11)
    assert first["prev_logprobs"][2] == pytest.approx(-0.1)
    assert second["advantages"] == [0.0] * sequence_length

    # A real SingleController optimizer step may stream more than one replay
    # chunk.  The second chunk must append in dispatch order and the logger must
    # publish one complete step file, never one file per chunk.
    second_data = TensorDict(
        {
            **{key: value.clone() for key, value in data.items() if isinstance(value, torch.Tensor)},
            SHARED_PREFIX_GROUP_ID: np.asarray(["second-group", "second-group"], dtype=object),
        },
        batch_size=[batch_size],
    )
    second_plane = _AdvantageDataPlane(second_data)
    ctrl._dp_client = second_plane
    second_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["second-group_g0", "second-group_g1"],
        fields=list(second_data.keys()),
    )
    _, second_has_valid_tokens = asyncio.run(ctrl._advantage_stage(second_meta))
    assert second_has_valid_tokens
    assert len(ctrl._step_train_data_records) == 4

    # Fail closed without clearing buffered evidence if an upstream streaming
    # bug supplies a duplicate sample ID.
    ctrl._step_train_data_records.append(dict(ctrl._step_train_data_records[0]))
    with pytest.raises(RuntimeError, match="missing or duplicate sample IDs"):
        ctrl._flush_train_data_records(step=1)
    assert len(ctrl._step_train_data_records) == 5
    ctrl._step_train_data_records.pop()

    ctrl._flush_train_data_records(step=1)

    ctrl._logger.log_batched_dict_as_jsonl.assert_called_once()
    columns, filename = ctrl._logger.log_batched_dict_as_jsonl.call_args.args
    assert filename == "train_data_step1.jsonl"
    assert columns["sample_id"] == [
        "group_g0",
        "group_g1",
        "second-group_g0",
        "second-group_g1",
    ]
    assert columns["token_ids"] == input_ids.tolist() * 2
    assert columns["generation_logprobs"][3][4] == pytest.approx(-0.51)
    assert ctrl._step_train_data_records == []

    with pytest.raises(RuntimeError, match="produced no rows"):
        ctrl._flush_train_data_records(step=1)


def test_string_object_field_decodes_numpy_object_column_wrapper() -> None:
    data = TensorDict(
        {SHARED_PREFIX_GROUP_ID: np.asarray(["stable-group-0", "stable-group-1"], dtype=object)},
        batch_size=[2],
    )

    assert _string_object_field(data, SHARED_PREFIX_GROUP_ID) == [
        "stable-group-0",
        "stable-group-1",
    ]


def test_string_object_field_decodes_real_nontensor_stack_wrappers() -> None:
    data = TensorDict(
        {
            SHARED_PREFIX_GROUP_ID: NonTensorStack(
                NonTensorData(data=np.asarray("stable-group-0", dtype=object)),
                NonTensorData(data=np.asarray("stable-group-1", dtype=object)),
            )
        },
        batch_size=[2],
    )

    assert _string_object_field(data, SHARED_PREFIX_GROUP_ID) == [
        "stable-group-0",
        "stable-group-1",
    ]


def test_string_object_field_rejects_non_scalar_wrapped_rows() -> None:
    data = TensorDict(
        {
            SHARED_PREFIX_GROUP_ID: NonTensorStack(
                NonTensorData(data=np.asarray(["stable-group-0"], dtype=object)),
                NonTensorData(data="stable-group-1"),
            )
        },
        batch_size=[2],
    )

    with pytest.raises(TypeError, match="must contain scalar strings"):
        _string_object_field(data, SHARED_PREFIX_GROUP_ID)


def test_string_object_field_rejects_row_count_mismatch() -> None:
    data = TensorDict(
        {SHARED_PREFIX_GROUP_ID: NonTensorData(data=["stable-group-0"], batch_size=[2])},
        batch_size=[2],
    )

    with pytest.raises(TypeError, match="exactly one value per row"):
        _string_object_field(data, SHARED_PREFIX_GROUP_ID)


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


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    async def select(self, **kwargs):
        del kwargs
        return None, 0


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0
        meta = self._meta
        self._meta = None
        return meta, 1


class _EvictingSampler(_OneThenEmptySampler):
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 2

    async def select(self, **kwargs):
        meta, num_groups = await super().select(**kwargs)
        return meta, 2 if num_groups else 0


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
            return None, 0
        self._remaining -= 1
        return self._meta, 1


class _SequenceSampler(_EmptySampler):
    def __init__(self, metas: list[KVBatchMeta]) -> None:
        self._metas = list(metas)

    async def select(self, **kwargs):
        del kwargs
        if not self._metas:
            return None, 0
        return self._metas.pop(0), 1


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
            return None, 0
        self._remaining -= 1
        assert minimum == maximum
        return self._meta, maximum


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
    ctrl._next_rollout_admission = 0
    ctrl._pending_rollout_admissions = deque()
    ctrl._legacy_untracked_replay_group_ids = set()
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
    metric_calls = ctrl._logger.log_metrics.call_args_list
    assert metric_calls[-2].kwargs == {
        "step": 1,
        "prefix": "train",
    }
    assert metric_calls[-1].kwargs == {
        "step": 1,
        "prefix": "timing/train",
        "step_finished": True,
    }


def test_train_data_flush_precedes_version_publish_and_weight_sync(monkeypatch) -> None:
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
    events: list[str] = []

    def flush(*, step: int) -> None:
        assert step == 1
        assert (ctrl._trainer_version, ctrl._train_steps) == (0, 0)
        events.append("flush")

    async def sync_weights(*, calibration_data) -> int:
        assert calibration_data is None
        assert (ctrl._trainer_version, ctrl._train_steps) == (1, 1)
        events.append("sync")
        return 0

    ctrl._flush_train_data_records = flush
    ctrl._sync_weights = sync_weights
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert events == ["flush", "sync"]


def test_train_data_flush_failure_does_not_publish_version_or_sync(monkeypatch) -> None:
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

    def fail_flush(*, step: int) -> None:
        assert step == 1
        assert (ctrl._trainer_version, ctrl._train_steps) == (0, 0)
        raise RuntimeError("injected train-data logging failure")

    ctrl._flush_train_data_records = fail_flush
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    with pytest.raises(RuntimeError, match="injected train-data logging failure"):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert (ctrl._trainer_version, ctrl._train_steps) == (0, 0)
    ctrl._sync_weights.assert_not_awaited()


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
        match=(r"rollout exhausted before a complete training step was assembled: " r"dispatched 1/2 prompt groups"),
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
        BaseSampler._validate_group_bounds(kwargs["min_prompt_groups"], kwargs["max_prompt_groups"])
        meta, num_groups = await super().select(**kwargs)
        if meta is None and not self._credit_in_evict:
            self._credit()
        return meta, num_groups


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
def test_train_pump_closes_a_step_short_when_a_stamped_prompt_is_dropped(monkeypatch, credit_in_evict) -> None:
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


def test_train_pump_logs_first_step_scalar_moe_and_mtp5_metrics(monkeypatch) -> None:
    """TQ worker scalars reach the first train log under their exact tags."""
    from nemo_rl.models.policy.tq_policy import _aggregate_train_results

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
    worker_result = {
        "global_loss": torch.tensor(0.5),
        "grad_norm": torch.tensor([1.5]),
        "all_mb_metrics": {},
        "moe_metrics": {"load_balancing_loss": 0.125},
        "mtp_metrics": {
            "mtp_1_loss": 1.0,
            "mtp_1_acceptance_rate": 50.0,
            "mtp_2_loss": 2.0,
            "mtp_2_acceptance_rate": 40.0,
            "mtp_3_loss": 3.0,
            "mtp_3_acceptance_rate": 30.0,
            "mtp_4_loss": 4.0,
            "mtp_4_acceptance_rate": 40.0,
            "mtp_5_loss": 5.0,
            "mtp_5_acceptance_rate": 25.0,
            "grad_norm": 2.5,
        },
    }
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = _aggregate_train_results([worker_result])
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_call = next(call for call in ctrl._logger.log_metrics.call_args_list if call.kwargs["prefix"] == "train")
    assert train_call.kwargs["step"] == 1
    train_metrics = train_call.args[0]
    assert train_metrics["moe/load_balancing_loss"] == pytest.approx(0.125)
    logged_mtp = {
        f"{train_call.kwargs['prefix']}/{key}": value for key, value in train_metrics.items() if key.startswith("mtp/")
    }
    assert logged_mtp == pytest.approx(
        {
            "train/mtp/mtp_1_loss": 1.0,
            "train/mtp/mtp_1_acceptance_rate": 50.0,
            "train/mtp/mtp_2_loss": 2.0,
            "train/mtp/mtp_2_acceptance_rate": 40.0,
            "train/mtp/mtp_3_loss": 3.0,
            "train/mtp/mtp_3_acceptance_rate": 30.0,
            "train/mtp/mtp_4_loss": 4.0,
            "train/mtp/mtp_4_acceptance_rate": 40.0,
            "train/mtp/mtp_5_loss": 5.0,
            "train/mtp/mtp_5_acceptance_rate": 25.0,
            "train/mtp/grad_norm": 2.5,
        }
    )


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
