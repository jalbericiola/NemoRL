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

"""SingleController: asyncio orchestrator for the RL training loop.

CPU-only Ray actor that runs two concurrent pumps plus a watchdog, and
coordinates the other actors via lightweight RPCs. SC sends control signals
and reads metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → gen.generate_and_push(prompt, dp_client) ← RPC to GenWorker
                     GenWorker → dp_client.put_samples(...)
  _train_pump    → sampler.evict/select against TQReplayBuffer
                 → _advantage_stage(meta) → dp_client.get_samples(...)
                                        → adv_estimator.compute_advantage(...)
                                        → dp_client.put_samples(...)
                 → trainer.begin/train_microbatches/finish_train_step (split API,
                     driver-side TQPolicy via asyncio.to_thread)
                     Trainer → dp_client.get_samples(...)   (via its own client)
                 → dp_client.clear_samples(...)             ← SC clears after train
  _sync_weights  → WeightSynchronizer.sync_weights()
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from functools import partial
from typing import Any, Awaitable, Callable, Optional, Union

import numpy as np
import ray
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from nemo_rl.algorithms.grpo import (
    GRPOSaveState, _should_log_nemo_gym_responses,
    _write_latest_checkpoint_status,
    compute_and_apply_seq_logprob_error_masking, scale_rewards)
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.reward_functions import apply_reward_shaping
from nemo_rl.algorithms.shared_prefix_metrics import (
    SharedPrefixOpportunity, combine_shared_prefix_opportunities,
    observe_shared_prefix_opportunity)
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig, MasterConfig, validate_sampler_buffer_capacity,
    validate_single_controller_config)
from nemo_rl.algorithms.single_controller_utils.setup import \
    SingleControllerActorArgs
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics, fields_for_put, reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim, tensor_field)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID, SHARED_PREFIX_PROMPT_LENGTHS)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.codec import unwrap_wire_stripped_payload
from nemo_rl.data_plane.schema import DP_CALIB_INPUT_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import should_use_nemo_gym
from nemo_rl.experience.failures import RolloutStall
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy import get_shared_prefix_training_config
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager, PathLike
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration]

# Named `log` rather than `logger` to keep it distinct from the experiment
# Logger this module also uses as `self._logger`.
log = logging.getLogger(__name__)

_ROLLOUT_ADMISSION_STATE_FILE = "rollout_admission.pt"
_ROLLOUT_ADMISSION_STATE_VERSION = 2


class _AdmissionInflightRegistry:
    """Bind RolloutManager's attempt group ID to one durable prompt record.

    ``RolloutManager.generate_and_push`` creates the group ID internally. Its
    registry hook is therefore the only controller-side point at which the durable
    admission ledger can learn the exact ID later serialized by TQReplayBuffer.
    """

    def __init__(
        self,
        backing: dict[str, tuple[asyncio.Task[None], int]],
        group_record: dict[str, Any],
        rollout_permitted: asyncio.Event,
        registry_changed: asyncio.Event,
    ) -> None:
        self._backing = backing
        self._group_record = group_record
        self._rollout_permitted = rollout_permitted
        self._registry_changed = registry_changed

    async def wait_until_permitted(self) -> None:
        """Block every rollout attempt, including retries, during refit."""
        await self._rollout_permitted.wait()

    def is_permitted(self) -> bool:
        """Recheck admission synchronously immediately before registration."""
        return self._rollout_permitted.is_set()

    def __setitem__(self, group_id: str, value: tuple[asyncio.Task[None], int]) -> None:
        self._group_record["group_id"] = group_id
        self._backing[group_id] = value
        self._registry_changed.set()

    def pop(self, group_id: str, default: Any = None) -> Any:
        value = self._backing.pop(group_id, default)
        self._registry_changed.set()
        return value


def _resolve_train_dispatch_group_multiple(
    *,
    shared_prefix_mode: str,
    num_prompts_per_step: int,
    data_parallel_size: int,
) -> int:
    """Return the prompt-group quantum required by the active train path."""
    if shared_prefix_mode != "train":
        return 1
    if data_parallel_size < 1:
        raise ValueError(f"shared-prefix train mode requires positive DP size, got {data_parallel_size}")
    if num_prompts_per_step % data_parallel_size != 0:
        raise ValueError(
            "policy.shared_prefix_training.mode=train requires "
            "grpo.num_prompts_per_step to be divisible by the trainer data-parallel "
            f"size; got {num_prompts_per_step} prompt groups and DP="
            f"{data_parallel_size}."
        )
    return data_parallel_size


def _train_selection_group_bounds(
    *,
    remaining_groups: int,
    configured_min_groups: int,
    group_multiple: int,
) -> tuple[int, int]:
    """Choose sampler bounds, forcing exact DP-width chunks when required.

    Ordinary execution retains the historical greedy ``[min, remaining]``
    bounds. Shared-prefix train mode passes its DP size as ``group_multiple``;
    there, exact ``min == max`` bounds prevent a greedy sampler from claiming a
    currently available but non-divisible group count.
    """
    if remaining_groups < 1:
        raise ValueError(f"remaining_groups must be positive, got {remaining_groups}")
    if configured_min_groups < 1:
        raise ValueError(f"configured_min_groups must be positive, got {configured_min_groups}")
    if group_multiple < 1:
        raise ValueError(f"group_multiple must be positive, got {group_multiple}")

    historical_minimum = min(configured_min_groups, remaining_groups)
    if group_multiple == 1:
        return historical_minimum, remaining_groups
    if remaining_groups % group_multiple != 0:
        raise RuntimeError(
            "shared-prefix streaming train cannot dispatch the remaining prompt "
            f"groups evenly across DP ranks: remaining={remaining_groups}, "
            f"DP={group_multiple}. A dropped-prompt shortfall must preserve a "
            "DP-divisible step size; use replacement handling or reduce the step "
            "by a whole DP-width of prompt groups."
        )

    rounded_minimum = ((historical_minimum + group_multiple - 1) // group_multiple) * group_multiple
    exact_groups = min(rounded_minimum, remaining_groups)
    return exact_groups, exact_groups


def _validate_train_dispatch_buffer_capacity(
    *,
    num_prompts_per_step: int,
    configured_min_groups: int,
    group_multiple: int,
    max_buffered_rollouts: int,
) -> None:
    """Reject backpressure caps that cannot fill one required train claim."""
    required_groups, _ = _train_selection_group_bounds(
        remaining_groups=num_prompts_per_step,
        configured_min_groups=configured_min_groups,
        group_multiple=group_multiple,
    )
    if max_buffered_rollouts < required_groups:
        raise ValueError(
            "policy.shared_prefix_training.mode=train rounds the streaming "
            f"claim to {required_groups} prompt groups for DP={group_multiple}, "
            "but async_rl.max_buffered_rollouts="
            f"{max_buffered_rollouts}. Increase the buffer cap to at least "
            f"{required_groups}; otherwise backpressure prevents one claim from "
            "ever becoming ready."
        )


def _reduce_shared_prefix_step_metrics(
    opportunities: list[SharedPrefixOpportunity],
    *,
    expected_total_tokens: float | int | None = None,
) -> dict[str, float | int]:
    """Reduce disjoint streaming chunks into exact step-level metrics."""
    if not opportunities:
        return {}
    combined = combine_shared_prefix_opportunities(opportunities)
    if combined.total_sequences <= 0:
        raise RuntimeError("shared-prefix observation produced a step with no sequences")
    if expected_total_tokens is not None and float(expected_total_tokens) != float(combined.total_tokens):
        raise RuntimeError(
            "shared-prefix observation did not cover the exact optimizer step: "
            f"observed total_num_tokens={combined.total_tokens}, "
            f"step total_num_tokens={expected_total_tokens}"
        )
    metrics = combined.as_metrics()
    # SingleController historically omitted these two standard GRPO metrics.
    # Emit exact sample-weighted values so W&B can compare the original V/T
    # ceiling with exact prompt accounting on the same optimizer step.
    metrics["total_num_tokens"] = combined.total_tokens
    metrics["mean_prompt_length"] = combined.prompt_tokens / combined.total_sequences
    return metrics


def _string_object_field(data: TensorDict, field_name: str) -> list[str]:
    """Decode a TQ object column and require one non-empty string per row."""
    value: Any = None
    # pyrefly: inference cycle on tensordict.items() loop vars.
    for key, item in data.items(include_nested=False):  # type: ignore[bad-assignment]
        if str(key) == field_name:
            value = item
            break
    if isinstance(value, NonTensorStack):
        # TensorDict may preserve NumPy object columns as a stack of
        # NonTensorData wrappers.  Walk the stack structurally: in some
        # versions ``tolist()`` leaves those per-row payloads wrapped.
        items = list(value.unbind(0))
    elif isinstance(value, NonTensorData):
        payload = value.data
        if isinstance(payload, np.ndarray) and payload.dtype == object:
            if payload.ndim != 1:
                raise TypeError(f"object field {field_name!r} must be a one-dimensional column")
            items = payload.tolist()
        elif isinstance(payload, (list, tuple)):
            items = list(payload)
        elif value.batch_dims:
            items = list(value.unbind(0))
        else:
            items = [payload]
    elif isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim != 1:
            raise TypeError(f"object field {field_name!r} must be a one-dimensional column")
        items = value.tolist()
    else:
        raise TypeError(f"expected object field {field_name!r}; got {type(value).__name__}")
    if len(data.batch_size) != 1 or len(items) != data.batch_size[0]:
        raise TypeError(f"object field {field_name!r} must contain exactly one value per row")
    decoded: list[str] = []
    for raw_item in items:
        decoded_item = unwrap_wire_stripped_payload(raw_item)
        while isinstance(decoded_item, NonTensorData):
            decoded_item = unwrap_wire_stripped_payload(decoded_item.data)
        if isinstance(decoded_item, np.ndarray) and decoded_item.dtype == object:
            if decoded_item.ndim != 0:
                raise TypeError(f"object field {field_name!r} must contain scalar strings")
            decoded_item = decoded_item.item()
        if not isinstance(decoded_item, str) or not decoded_item:
            raise TypeError(f"object field {field_name!r} must contain non-empty strings")
        decoded.append(decoded_item)
    return decoded


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns three concurrent asyncio tasks:
      - _rollout_pump:  dispatches prompts to GenerationWorkerActor
      - _train_pump:    claims DataPlane meta, trains, clears consumed rows,
                        then runs _sync_weights (drain gate + weight
                        synchronization) inline after each optimizer step
      - _stall_watchdog_pump: publishes rollout counters and reports stalls or
                        unhealthy environments, which are the failures that
                        otherwise produce no signal at all

    Plus _gen_fleet_probe_pump when fleet health is enabled, which probes generation
    shard liveness on its own, much shorter clock.

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        actor_args: SingleControllerActorArgs,
        setup_timing_metrics: SetupTimingMetrics,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            actor_args: Pre-built actor args from setup_single_controller.
            setup_timing_metrics: Driver-side setup timings; logged here (Logger isn't cloudpickleable).
        """
        validate_single_controller_config(master_config)

        self._advantage_cfg = AdvantageConfig()
        self._partition_id: str = actor_args.partition_id

        self._master_config = master_config
        self._shared_prefix_training_config = get_shared_prefix_training_config(master_config.policy)
        self._async_cfg = master_config.async_rl
        self._policy_logprobs_required = not (
            master_config.loss_fn.force_on_policy_ratio and master_config.grpo.seq_logprob_error_threshold is None
        )
        self._reference_logprobs_required = not bool(master_config.grpo.skip_reference_policy_logprobs_calculation)
        self._dp_client = actor_args.dp_client
        self._gen: Generation = actor_args.gen_handle
        self._trainer: TQPolicy = actor_args.trainer_handle
        self._train_dispatch_group_multiple = _resolve_train_dispatch_group_multiple(
            shared_prefix_mode=self._shared_prefix_training_config.mode,
            num_prompts_per_step=master_config.grpo.num_prompts_per_step,
            data_parallel_size=(
                int(self._trainer.data_parallel_size) if self._shared_prefix_training_config.mode == "train" else 1
            ),
        )
        if self._shared_prefix_training_config.mode == "train":
            _validate_train_dispatch_buffer_capacity(
                num_prompts_per_step=master_config.grpo.num_prompts_per_step,
                configured_min_groups=(master_config.async_rl.min_groups_for_streaming_train),
                group_multiple=self._train_dispatch_group_multiple,
                max_buffered_rollouts=master_config.async_rl.max_buffered_rollouts,
            )
        self._dataloader = actor_args.dataloader
        self._weight_synchronizer = actor_args.weight_synchronizer
        self._advantage_estimator = actor_args.advantage_estimator
        self._loss_fn = actor_args.loss_fn
        self._buffer = actor_args.tq_buffer
        self._rollout_manager = actor_args.rollout_manager
        # Direct access, deliberately. A getattr default here reads as defensive but
        # buys a silent failure mode: rename or drop the field and
        # watchdog.gym_subprocess_check: true degrades to a health check that iterates
        # nothing and reports nothing -- the exact class of silent failure this work
        # exists to remove. A missing field should break loudly at construction, where
        # it costs five minutes, not quietly at hour three of a run.
        self._env_handles = actor_args.env_handles
        # These two keep the getattr for a genuinely different reason: None is a
        # meaningful value meaning "feature off", and it is also their default. Absence
        # therefore degrades to the documented off state rather than to a broken one.
        self._gen_fleet = getattr(actor_args, "fleet_monitor", None)
        self._generation_router = getattr(actor_args, "generation_router", None)
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._logger.log_hyperparams(master_config.model_dump())
        self._logger.log_metrics(setup_timing_metrics.to_metrics_dict(), step=0, prefix="timing/setup")
        self._timer = Timer()

        # Also built here, not on the driver: TimeoutChecker must capture
        # wall-clock start times inside the actor, not at driver setup time.
        # actor_args only carries the driver-side restore products
        # (save_state, last_checkpoint_path).
        self._checkpointer = CheckpointManager(master_config.checkpointing)
        self._timeout = TimeoutChecker(
            timeout=master_config.checkpointing["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        self._timeout.start_iterations()

        # Loaded (or initial) GRPOSaveState from setup; _get_grpo_save_state
        # already defaulted any fields missing from older checkpoints.
        self._save_state: GRPOSaveState = actor_args.save_state
        self._last_checkpoint_path: Optional[str] = actor_args.last_checkpoint_path
        self._consumed_samples: int = actor_args.save_state.consumed_samples
        self._total_valid_tokens: int = actor_args.save_state.total_valid_tokens

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = actor_args.train_cluster
        self._inference_cluster = actor_args.inference_cluster

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        self._sampler = create_sampler(self._buffer, self._async_cfg.sampler)
        # The trainer step and rollout admission cursors are equal only on a fresh
        # start. Lookahead makes admission run ahead, and the dataloader checkpoint is
        # already positioned after those admitted batches. Persist the independent
        # cursor plus the not-yet-trained batches so a restart can requeue only the
        # in-flight groups omitted by TQReplayBuffer.state_dict().
        self._next_rollout_admission: int = actor_args.save_state.current_step
        self._pending_rollout_admissions: deque[dict[str, Any]] = deque()
        self._rollout_admission_state_restored = False
        # Ready replay rows inherited from a checkpoint that predates the identity
        # ledger. Carry their exact IDs through v2 checkpoints until selected/evicted,
        # so a second resume does not mistake them for a corrupt snapshot pair.
        self._legacy_untracked_replay_group_ids: set[str] = set()
        self._sampler.set_dispatch_index(self._next_rollout_admission)
        required_capacity = self._sampler.required_buffer_capacity(num_prompts_per_step)
        validate_sampler_buffer_capacity(
            self._async_cfg,
            required_capacity=required_capacity,
            sampler_name=type(self._sampler).__name__,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # One atomic cut for dataloader advancement, reserve transfers, admission
        # bookkeeping, and the replay snapshot paired with their checkpoint. It is
        # deliberately separate from _rollout_permitted: already-admitted generation
        # may finish while a checkpoint is being written.
        self._rollout_admission_lock = asyncio.Lock()

        # Set only after _rollout_pump exhausts its configured epochs and all
        # dispatched tasks finish successfully. Rollout failures propagate
        # through run() instead of being reported as normal exhaustion.
        self._rollout_exhausted: asyncio.Event = asyncio.Event()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        self._inflight_by_group_id: dict[str, tuple[asyncio.Task[None], int]] = {}
        # Wakes the refit drain when an attempt enters or leaves the generation
        # section. The rollout gate alone cannot stop an already-admitted retry.
        self._inflight_registry_changed = asyncio.Event()

        # Groups that will never arrive, keyed by the training step they were stamped
        # for. A sampler that matches batches to steps exactly (InOrderSampler) can only
        # ever select num_prompts_per_step groups carrying that stamp, so a dropped
        # prompt leaves that step permanently one short and the train pump waits on a
        # group no one is generating. The pump subtracts these to close the step short.
        # Only stamped prompts appear here: with an unstamped sampler the batch fills
        # from whatever is ready, so a drop costs throughput but strands nothing.
        self._batch_shortfall: dict[int, int] = {}

        # Spare prompts that on_dropped_prompt="replace" substitutes for dropped ones,
        # and the per-step counts of how each step got made whole (logged so a step's
        # batch size stays explainable after the fact). The reserve is filled only by the
        # rollout pump, which owns the dataloader iterator; see the config docstring for
        # why a dispatch task cannot pull from it directly.
        #
        # The two counters read from opposite ends of a borrow: a step that filled a
        # hole with a later step's finished group counts a promotion, and the step it
        # borrowed from counts the replacement that repaid it.
        self._replacement_reserve: deque[DatumSpec] = deque()
        self._batch_replacements: dict[int, int] = {}
        self._batch_promotions: dict[int, int] = {}
        # Whether the sampler has ever handed back a target step. Only stamped prompts
        # can strand a step, so this gates the pool: filling it for a sampler that never
        # stamps would divert a batch of prompts that nothing is ever able to draw on.
        # Learned from admit rather than the sampler's type, because a custom sampler's
        # stamping is not knowable until it answers.
        self._sampler_stamps_target_steps: bool = False

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(self._async_cfg.max_buffered_rollouts)
        self._restored_replay_group_count = 0

        self._trainer_version: int = actor_args.save_state.current_step
        self._train_steps: int = actor_args.save_state.current_step
        self._current_epoch: int = actor_args.save_state.current_epoch
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "verifier_rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
            "seq_logprob_error_metrics": [],
        }
        self._step_shared_prefix_opportunities: list[SharedPrefixOpportunity] = []
        # Match the legacy GRPO meaning of env.should_log_nemo_gym_responses:
        # when Gym is not responsible for full response logging, preserve the
        # exact tensors that reached SingleController training in a local JSONL.
        # DataPlane samples are cleared chunk-by-chunk, so capture CPU records in
        # _advantage_stage and flush the whole ordered optimizer step only after
        # finish_train_step succeeds.
        self._log_train_data = not _should_log_nemo_gym_responses(master_config)
        self._step_train_data_records: list[dict[str, Any]] = []

        print(
            f"SingleControllerActor: "
            f"sampler={self._async_cfg.sampler.name} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"weight_sync={type(self._weight_synchronizer).__name__}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        # Synchronize weights before starting the pumps
        await self._sync_weights()

        await self._maybe_restore_rollout_admission_state()
        await self._maybe_restore_replay_buffer()
        self._finalize_rollout_admission_restore()
        await self._discard_legacy_unstamped_replay()
        await self._maybe_restore_replacement_reserve()

        # Start the rollout and train pumps, plus the watchdog
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        watchdog_task = asyncio.create_task(self._stall_watchdog_pump())
        tasks = [rollout_task, train_task, watchdog_task]
        # Only with fleet health on. Created unconditionally it would be a timer firing
        # every probe_interval_s for every run that does not use the feature, which is
        # the default.
        probe_task = asyncio.create_task(self._gen_fleet_probe_pump()) if self._gen_fleet is not None else None
        if probe_task is not None:
            tasks.append(probe_task)
        run_succeeded = False
        primary_error_active = False
        first_teardown_error: BaseException | None = None

        def record_teardown_error(stage: str, error: BaseException) -> None:
            """Retain the first cleanup failure while allowing cleanup to continue."""
            nonlocal first_teardown_error
            if first_teardown_error is None:
                first_teardown_error = error
            try:
                print(
                    f"Error during {stage}: {type(error).__name__}: {error}",
                    flush=True,
                )
            except BaseException:
                # A closed or broken output stream is another cleanup failure;
                # it cannot be allowed to replace the active pump exception.
                pass

        try:
            done, _ = await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
            if probe_task is not None and probe_task in done:
                # Loops forever like the watchdog, so finishing at all means it raised.
                await probe_task
            if watchdog_task in done:
                # The watchdog loops forever, so finishing at all means it raised --
                # a stall or an unhealthy environment. Surface that ahead of the
                # pumps, whose own symptom would just be "waiting".
                await watchdog_task
            if rollout_task in done:
                # Propagate rollout failures immediately. A normally exhausted
                # rollout pump leaves the train pump to drain committed groups.
                await rollout_task
            await train_task
            run_succeeded = True
        except BaseException:
            # Keep the active pump/train exception alive across every teardown
            # attempt. The bare raise after finally preserves its traceback.
            primary_error_active = True
            raise
        finally:
            for task in tasks:
                try:
                    task.cancel()
                except BaseException as error:
                    record_teardown_error("pump cancellation", error)

            pumps_joined = False
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                pumps_joined = True
            except BaseException as error:
                record_teardown_error("pump join", error)

            try:
                self._weight_synchronizer.shutdown()
            except BaseException as error:
                record_teardown_error("weight-synchronizer shutdown", error)

            # A terminal receipt is valid only after the main run succeeded and
            # every pump is joined, so its counter snapshot is immutable.
            if run_succeeded and pumps_joined:
                try:
                    terminal_metrics = dict(self._rollout_manager.stats.as_metrics())
                    # Dict insertion order is the backend emission order. Append
                    # the receipt last so it cannot precede its counter snapshot.
                    terminal_metrics["rollout/counters_finalized"] = 1.0
                    self._logger.log_metrics(
                        terminal_metrics,
                        step=self._train_steps,
                        step_finished=True,
                    )
                except BaseException as error:
                    record_teardown_error("terminal counter logging", error)

            try:
                self._logger.finish()
            except BaseException as error:
                record_teardown_error("logger shutdown", error)

            try:
                await asyncio.to_thread(self._checkpointer.shutdown)
            except BaseException as error:
                record_teardown_error("checkpointer shutdown", error)

            # Cleanup failures are actionable only when there is no active
            # pump/train failure for them to obscure.
            if first_teardown_error is not None and not primary_error_active:
                raise first_teardown_error

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
        }

    async def ping(self) -> dict[str, Any]:
        """Liveness check — returns immediately if event loop is running."""
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
            "train_steps": self._train_steps,
            "inflight_rollouts": self._inflight_rollouts,
            "rollout_permitted": self._rollout_permitted.is_set(),
            "epoch": self._current_epoch,
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _maybe_restore_rollout_admission_state(self) -> None:
        """Restore the rollout cursor and admitted-but-untrained prompt batches.

        The dataloader state is captured after every batch yielded to the rollout
        pump, including lookahead batches. The replay-buffer checkpoint, however,
        intentionally omits unready slots. This small companion checkpoint closes
        that gap: it owns the original prompt batches until their optimizer step
        finishes, so missing groups can be regenerated without advancing the restored
        dataloader a second time.
        """
        if self._last_checkpoint_path is None:
            return
        state_path = os.path.join(self._last_checkpoint_path, _ROLLOUT_ADMISSION_STATE_FILE)
        if not os.path.exists(state_path):
            print(
                f"⚠️ No rollout admission checkpoint found at {state_path}. "
                "Reconstructing the cursor from restored replay metadata; in-flight "
                "rollouts from this legacy checkpoint cannot be recovered.",
                flush=True,
            )
            return

        state = await asyncio.to_thread(torch.load, state_path, weights_only=False)
        if not isinstance(state, dict):
            raise ValueError("rollout admission checkpoint must be a mapping, got " f"{type(state).__name__}")
        required = {
            "version",
            "sampler_name",
            "next_rollout_admission",
            "pending_admissions",
            "legacy_untracked_group_ids",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"rollout admission checkpoint missing required keys: {missing}")
        if state["version"] != _ROLLOUT_ADMISSION_STATE_VERSION:
            raise ValueError("unsupported rollout admission checkpoint version: " f"{state['version']!r}")
        saved_sampler_name = state["sampler_name"]
        current_sampler_name = self._async_cfg.sampler.name
        if saved_sampler_name != current_sampler_name:
            raise ValueError(
                "rollout admission checkpoint was saved with sampler "
                f"{saved_sampler_name!r}, but this run uses "
                f"{current_sampler_name!r}. Resume with the original sampler so "
                "already-consumed dataloader batches retain their admission policy."
            )

        next_admission = state["next_rollout_admission"]
        if (
            not isinstance(next_admission, int)
            or isinstance(next_admission, bool)
            or next_admission < self._train_steps
        ):
            raise ValueError(
                "rollout admission cursor must be an integer no smaller than the "
                f"restored train step {self._train_steps}, got {next_admission!r}"
            )

        pending_state = state["pending_admissions"]
        if not isinstance(pending_state, list):
            raise ValueError("pending_admissions must be a list")
        raw_legacy_group_ids = state["legacy_untracked_group_ids"]
        if (
            not isinstance(raw_legacy_group_ids, list)
            or any(not isinstance(group_id, str) or not group_id for group_id in raw_legacy_group_ids)
            or len(set(raw_legacy_group_ids)) != len(raw_legacy_group_ids)
        ):
            raise ValueError("legacy_untracked_group_ids must be a list of unique non-empty " "strings")
        pending: deque[dict[str, Any]] = deque()
        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        seen_dispatch_indices: set[int] = set()
        seen_group_ids: set[str] = set()
        for index, raw in enumerate(pending_state):
            if not isinstance(raw, dict):
                raise ValueError(f"pending admission {index} must be a mapping, got " f"{type(raw).__name__}")
            if set(raw) != {"dispatch_index", "admitted", "source", "groups"}:
                raise ValueError(f"pending admission {index} has invalid keys: {set(raw)}")
            admitted = raw["admitted"]
            if not isinstance(admitted, bool):
                raise ValueError(f"pending admission {index} admitted must be bool, got " f"{admitted!r}")
            source = raw["source"]
            if source not in {"dataloader", "reserve"}:
                raise ValueError(f"pending admission {index} has invalid source={source!r}")
            dispatch_index = raw["dispatch_index"]
            dispatch_index_invalid = admitted and (
                not isinstance(dispatch_index, int)
                or isinstance(dispatch_index, bool)
                or dispatch_index < 0
                or dispatch_index >= next_admission
                or dispatch_index in seen_dispatch_indices
            )
            if dispatch_index_invalid or (not admitted and dispatch_index is not None):
                raise ValueError(
                    f"pending admission {index} has invalid or duplicate "
                    f"dispatch_index={dispatch_index!r} for next admission "
                    f"{next_admission}"
                )
            if admitted:
                seen_dispatch_indices.add(dispatch_index)
            raw_groups = raw["groups"]
            if not isinstance(raw_groups, list) or not (1 <= len(raw_groups) <= num_prompts_per_step):
                raise ValueError(
                    f"pending admission {index} must hold between 1 and " f"{num_prompts_per_step} group record(s)"
                )
            groups: list[dict[str, Any]] = []
            for group_index, raw_group in enumerate(raw_groups):
                if not isinstance(raw_group, dict) or set(raw_group) != {
                    "target_step",
                    "group_id",
                    "prompt",
                }:
                    raise ValueError(
                        f"pending admission {index} group {group_index} must have "
                        "exactly target_step, group_id, and prompt"
                    )
                target_step = raw_group["target_step"]
                if target_step is not None and (
                    not isinstance(target_step, int) or isinstance(target_step, bool) or target_step < self._train_steps
                ):
                    raise ValueError(
                        f"pending admission {index} group {group_index} has "
                        f"invalid target_step={target_step!r} for restored train "
                        f"step {self._train_steps}"
                    )
                if not admitted and target_step is not None:
                    raise ValueError(
                        f"unadmitted pending admission {index} group {group_index} " "cannot have a target_step"
                    )
                group_id = raw_group["group_id"]
                if group_id is not None and (
                    not isinstance(group_id, str) or not group_id or group_id in seen_group_ids
                ):
                    raise ValueError(
                        f"pending admission {index} group {group_index} has "
                        f"invalid or duplicate group_id={group_id!r}"
                    )
                if group_id is not None:
                    seen_group_ids.add(group_id)
                if not admitted and group_id is not None:
                    raise ValueError(
                        f"unadmitted pending admission {index} group {group_index} " "cannot have a group_id"
                    )
                groups.append(
                    {
                        "target_step": target_step,
                        "group_id": group_id,
                        "prompt": raw_group["prompt"],
                    }
                )
            pending.append(
                {
                    "dispatch_index": dispatch_index,
                    "admitted": admitted,
                    "source": source,
                    "groups": groups,
                }
            )

        unadmitted_indices = [index for index, entry in enumerate(pending) if not entry["admitted"]]
        if unadmitted_indices and unadmitted_indices != [len(pending) - 1]:
            raise ValueError(
                "rollout admission checkpoint may contain only one unadmitted "
                "dataloader batch, at the tail of pending_admissions"
            )

        stamped = [
            group["target_step"] is not None for entry in pending if entry["admitted"] for group in entry["groups"]
        ]
        if stamped and any(stamped) and not all(stamped):
            raise ValueError("rollout admission checkpoint mixes stamped and unstamped batches")

        self._next_rollout_admission = next_admission
        self._pending_rollout_admissions = pending
        self._legacy_untracked_replay_group_ids = set(raw_legacy_group_ids)
        self._rollout_admission_state_restored = True
        self._sampler.set_dispatch_index(self._next_rollout_admission)
        if stamped and stamped[0]:
            self._sampler_stamps_target_steps = True
        print(
            f"📦 Restored {len(pending)} pending rollout admission(s); next "
            f"admission={self._next_rollout_admission}: {state_path}",
            flush=True,
        )

    async def _discard_legacy_unstamped_replay(self) -> None:
        """Drop restored unstamped rows that cannot resume at the current weight.

        A checkpoint predating ``rollout_admission.pt`` has no prompt identity
        ledger, so every unstamped row is discarded and the complete remaining
        admission budget is regenerated. Retaining an incomplete older WeightFifo
        bucket would make it the oldest selectable weight forever, while every fresh
        replacement uses the current weight.

        V2 normally reconciles by exact group ID. It still has to discard an older
        WeightFifo row (or a row the sampler explicitly says is now stale): if its
        sibling was in flight and omitted from replay, requeueing only that sibling at
        the current weight splits one prompt cohort across two strict FIFO buckets.
        Removing the ready row makes the exact ledger requeue the whole affected
        cohort at one weight. Legacy carry IDs inside a transitional v2 checkpoint
        are also discarded because they have no prompt record to requeue.
        """
        targets = list(getattr(self._buffer, "target_step_list", ()))
        if not targets or any(target is not None for target in targets):
            return

        group_ids = list(getattr(self._buffer, "_group_ids", ()))
        start_weights = list(
            getattr(
                self._buffer,
                "start_weight_list",
                [self._train_steps] * len(targets),
            )
        )
        if not len(group_ids) == len(start_weights) == len(targets):
            raise RuntimeError("restored replay metadata is misaligned before admission cleanup")
        if self._rollout_admission_state_restored:
            exact_ledger_group_ids = {
                group["group_id"]
                for entry in self._pending_rollout_admissions
                for group in entry["groups"]
                if group["group_id"] is not None
            }
            unexpected_group_ids = set(group_ids) - (exact_ledger_group_ids | self._legacy_untracked_replay_group_ids)
            if unexpected_group_ids:
                raise RuntimeError(
                    "restored replay buffer contains group ID(s) absent from the "
                    "paired admission ledger or its legacy carry set: "
                    f"{sorted(unexpected_group_ids)}"
                )

        drop_indices: list[int] = []
        dropped_legacy_group_ids: set[str] = set()
        sampler_name = self._async_cfg.sampler.name
        for index, (group_id, start_weight) in enumerate(zip(group_ids, start_weights, strict=True)):
            is_legacy = (
                not self._rollout_admission_state_restored or group_id in self._legacy_untracked_replay_group_ids
            )
            drop_for_weight_fifo = sampler_name == "weight_fifo" and start_weight < self._train_steps
            drop_as_stale = self._sampler.should_abort_inflight(
                start_weight_version=start_weight,
                current_train_weight=self._train_steps,
            )
            if is_legacy or drop_for_weight_fifo or drop_as_stale:
                drop_indices.append(index)
                if is_legacy:
                    dropped_legacy_group_ids.add(group_id)

        if not drop_indices:
            return
        dropped_group_ids = {group_ids[index] for index in drop_indices}
        removed = await self._buffer.remove(drop_indices, remove_in_dp=True)
        if removed != len(drop_indices):
            raise RuntimeError("unstamped replay cleanup removed " f"{removed}/{len(drop_indices)} group(s)")
        self._legacy_untracked_replay_group_ids.difference_update(dropped_group_ids)
        if dropped_legacy_group_ids and self._rollout_admission_state_restored:
            admitted_dispatch_indices = [
                entry["dispatch_index"] for entry in self._pending_rollout_admissions if entry["admitted"]
            ]
            exact_frontier = max([self._train_steps] + [index + 1 for index in admitted_dispatch_indices])
            if self._next_rollout_admission > exact_frontier:
                self._next_rollout_admission = exact_frontier
                self._sampler.set_dispatch_index(exact_frontier)
        restored_permits = min(removed, getattr(self, "_restored_replay_group_count", 0))
        for _ in range(restored_permits):
            self._buffer_capacity.release()
        self._restored_replay_group_count = max(
            0,
            getattr(self, "_restored_replay_group_count", 0) - removed,
        )
        print(
            f"🧹 Discarded {removed} unstamped replay group(s) that could not "
            "resume safely at the current trainer weight.",
            flush=True,
        )

    def _finalize_rollout_admission_restore(self) -> None:
        """Seed a legacy checkpoint's cursor from the replay groups it restored."""
        targets = list(getattr(self._buffer, "target_step_list", ()))
        has_stamped_targets = any(target is not None for target in targets)
        stamped_targets = [target for target in targets if target is not None and target >= self._train_steps]
        if has_stamped_targets:
            self._sampler_stamps_target_steps = True

        if self._rollout_admission_state_restored:
            return

        self._legacy_untracked_replay_group_ids = set(getattr(self._buffer, "_group_ids", ()))

        # Checkpoints predating rollout_admission.pt cannot recover unready prompt
        # payloads, but the ready replay groups still prove how far admission got.
        # A partial stamped cohort is different: seed at its target so a dataloader or
        # reserve batch can fill its missing groups without spending another absolute
        # step. Full cohorts are skipped by advancing beyond their largest target.
        if has_stamped_targets:
            groups_per_step = self._master_config.grpo.num_prompts_per_step
            if stamped_targets:
                target_counts = {target: stamped_targets.count(target) for target in set(stamped_targets)}
                # A completely absent cohort can also be an admitted batch whose every
                # rollout was in flight when the legacy replay snapshot was taken. Start
                # at the first non-full target in the contiguous lookahead range, then let
                # the pump advance across later full targets without consuming data.
                incomplete_targets = [
                    target
                    for target in range(self._train_steps, max(stamped_targets) + 1)
                    if target_counts.get(target, 0) < groups_per_step
                ]
                if incomplete_targets:
                    self._next_rollout_admission = max(self._train_steps, min(incomplete_targets))
                else:
                    self._next_rollout_admission = max(self._train_steps, max(stamped_targets) + 1)
            else:
                # A stale stamped straggler still proves the sampler policy. It says
                # nothing about the next untrained cohort, so do not spend that cohort's
                # absolute admission budget before the stale row is evicted.
                self._next_rollout_admission = self._train_steps
        elif targets:
            # An old unstamped replay snapshot cannot prove how many *selectable*
            # cohorts it owns. Rows may have become stale immediately after the saved
            # train-step increment, and custom samplers need not expose their
            # selectability rule. Counting those rows can therefore spend admission
            # budget on work the train pump will evict and end the run short. Restart
            # from the absolute train frontier and tolerate any restored surplus. New
            # v2 checkpoints use exact group IDs and never take this fallback.
            self._next_rollout_admission = self._train_steps
            print(
                "⚠️ Legacy unstamped replay has no admission identity ledger; "
                "regenerating the complete remaining admission budget so stale "
                "restored groups cannot make the run finish short.",
                flush=True,
            )
        self._sampler.set_dispatch_index(self._next_rollout_admission)

    def _rollout_admission_state_dict(self) -> dict[str, Any]:
        """Return the admission state paired with the dataloader checkpoint."""
        return {
            "version": _ROLLOUT_ADMISSION_STATE_VERSION,
            "sampler_name": self._async_cfg.sampler.name,
            "next_rollout_admission": self._next_rollout_admission,
            "legacy_untracked_group_ids": sorted(self._legacy_untracked_replay_group_ids),
            "pending_admissions": [
                {
                    "dispatch_index": entry["dispatch_index"],
                    "admitted": entry["admitted"],
                    "source": entry["source"],
                    "groups": [
                        {
                            "target_step": group["target_step"],
                            "group_id": group["group_id"],
                            "prompt": group["prompt"],
                        }
                        for group in entry["groups"]
                    ],
                }
                for entry in self._pending_rollout_admissions
            ],
        }

    def _retire_rollout_group_ids(self, group_ids: set[str]) -> int:
        """Remove exact committed groups from the durable admission ledger."""
        if not group_ids:
            return 0
        self._legacy_untracked_replay_group_ids.difference_update(group_ids)
        removed = 0
        retained: deque[dict[str, Any]] = deque()
        for entry in self._pending_rollout_admissions:
            groups = []
            for group in entry["groups"]:
                if group["group_id"] in group_ids:
                    removed += 1
                else:
                    groups.append(group)
            if groups:
                entry["groups"] = groups
                retained.append(entry)
        self._pending_rollout_admissions = retained
        return removed

    def _drop_pending_group(self, group_record: dict[str, Any]) -> None:
        """Forget one prompt that definitively produced no replay group."""
        retained: deque[dict[str, Any]] = deque()
        found = False
        for entry in self._pending_rollout_admissions:
            groups = []
            for candidate in entry["groups"]:
                if candidate is group_record:
                    found = True
                else:
                    groups.append(candidate)
            if groups:
                entry["groups"] = groups
                retained.append(entry)
        self._pending_rollout_admissions = retained
        if not found:
            raise RuntimeError("rollout prompt record disappeared before completion")

    def _retarget_pending_group(self, group_id: str, target_step: int) -> None:
        """Keep the ledger aligned when a ready group is promoted to another step."""
        for entry in self._pending_rollout_admissions:
            for group in entry["groups"]:
                if group["group_id"] == group_id:
                    group["target_step"] = target_step
                    return

    def _retire_rollout_admission(
        self,
        trained_step: int,
        *,
        trained_group_ids: set[str] | None = None,
    ) -> None:
        """Forget the exact groups durably consumed by one optimizer step."""
        if trained_group_ids:
            self._retire_rollout_group_ids(trained_group_ids)
        if not self._pending_rollout_admissions:
            return
        admitted_groups = [
            group for entry in self._pending_rollout_admissions if entry["admitted"] for group in entry["groups"]
        ]
        if not admitted_groups:
            return
        stamped = admitted_groups[0]["target_step"] is not None
        if any(
            (group["target_step"] is not None) != stamped
            for entry in self._pending_rollout_admissions
            if entry["admitted"]
            for group in entry["groups"]
        ):
            raise RuntimeError("pending rollout admissions mix stamp policies")

        if stamped:
            retained: deque[dict[str, Any]] = deque()
            for entry in self._pending_rollout_admissions:
                if not entry["admitted"]:
                    retained.append(entry)
                    continue
                groups = [group for group in entry["groups"] if group["target_step"] > trained_step]
                if groups:
                    entry["groups"] = groups
                    retained.append(entry)
            self._pending_rollout_admissions = retained
            return

        # Real replay groups have exact IDs. This count fallback is only for legacy
        # checkpoints/test doubles whose restored rows predate the identity ledger.
        if trained_group_ids:
            return
        groups_to_retire = self._master_config.grpo.num_prompts_per_step
        retained = deque()
        for entry in self._pending_rollout_admissions:
            if not entry["admitted"] or not groups_to_retire:
                retained.append(entry)
                continue
            entry_groups = len(entry["groups"])
            if entry_groups <= groups_to_retire:
                groups_to_retire -= entry_groups
                continue
            entry["groups"] = entry["groups"][groups_to_retire:]
            groups_to_retire = 0
            retained.append(entry)
        self._pending_rollout_admissions = retained

    async def _maybe_restore_replay_buffer(self) -> None:
        """Restore replay-buffer groups from the previous run's checkpoint.

        Skipped with a warning when the checkpoint was written under a
        different sampler: restored groups carry the saving sampler's
        weight/target-step stamps, which another policy may never select.
        """
        if self._last_checkpoint_path is None:
            return
        buffer_path = os.path.join(self._last_checkpoint_path, "replay_buffer.pt")
        if not os.path.exists(buffer_path):
            print(
                f"⚠️ No replay buffer checkpoint found at {buffer_path}. " "Starting with an empty replay buffer.",
                flush=True,
            )
            return
        saved_sampler_name = self._save_state.sampler_name
        current_sampler_name = self._async_cfg.sampler.name
        if saved_sampler_name != current_sampler_name:
            print(
                f"⚠️ Replay buffer checkpoint was saved with sampler "
                f"{saved_sampler_name!r} but this run uses "
                f"{current_sampler_name!r}; skipping the buffer restore.",
                flush=True,
            )
            return
        print(f"📦 Restoring replay buffer from checkpoint: {buffer_path}")
        # weights_only=False: groups hold pickled KVBatchMeta/TensorDicts,
        # not plain tensors. The checkpoint is a trusted same-job artifact.
        buffer_state = await asyncio.to_thread(torch.load, buffer_path, weights_only=False)
        restored = await self._buffer.load_state_dict(
            buffer_state,
            max_groups=self._async_cfg.max_buffered_rollouts,
            expected_partition_id=self._partition_id,
            expected_group_size=self._master_config.grpo.num_generations_per_prompt,
        )
        # Each buffered group holds one _buffer_capacity permit; the load
        # truncation guarantees restored <= capacity, so this never blocks.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()
        self._restored_replay_group_count = restored

    async def _maybe_restore_replacement_reserve(self) -> None:
        """Restore spare prompts diverted before the previous run's checkpoint.

        These were pulled from the dataloader and never dispatched, so the restored
        dataloader resumes past them. Nothing else in the checkpoint holds them, and
        without this they are simply gone: one batch of the dataset per divert, plus
        the training step ``_clamp_max_num_steps`` had budgeted for it.

        No sampler-name guard, unlike the buffer restore. Spares carry no stamp -- they
        are prompts that never reached ``admit`` -- so nothing about them depends on
        which sampler wrote the checkpoint. They are restored even into a run that has
        since switched to ``on_dropped_prompt="shrink"``, where the pool is never drawn
        on but is still drained back into training at the end of the dataloader.
        """
        if self._last_checkpoint_path is None:
            return
        reserve_path = os.path.join(self._last_checkpoint_path, "replacement_reserve.pt")
        # Absent for every run that never diverted a batch, which is every run that
        # does not use "replace" -- so silence here rather than the buffer restore's
        # warning, since this is the ordinary case rather than a lost artifact.
        if not os.path.exists(reserve_path):
            return
        # weights_only=False: spares are pickled DatumSpecs, and the checkpoint is a
        # trusted same-job artifact (the replay buffer restore loads on the same terms).
        reserve_state = await asyncio.to_thread(torch.load, reserve_path, weights_only=False)
        self._replacement_reserve.extend(reserve_state)
        print(
            f"📦 Restored {len(reserve_state)} pooled spare prompt(s) from checkpoint: " f"{reserve_path}",
            flush=True,
        )

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(self, method_name: str, **kwargs) -> Any:
        """Call a DataPlaneClient method or a Ray actor exposing that method."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch:
          0. Under on_dropped_prompt="replace", divert the batch into the spare pool
             if the pool is below its low-water mark, and skip admission entirely.
             Otherwise await sampler.admit(...) to wait until the batch may dispatch
             and obtain its target_step stamp.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. If the prompt was dropped, substitute a spare and repeat step 4 -- for this
             step, or for whichever step lends this one a finished group in its place
             (see _take_replacement, _promote_into_step) -- or credit the step short so
             the train pump can close it
          6. Decrement _inflight_rollouts

        Once every epoch is done, whatever is left in the spare pool is dispatched as
        ordinary steps, up to the remaining ``grpo.max_num_steps`` budget, rather than
        discarded (see _drain_reserve_into_steps).
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        self._rollout_exhausted.clear()
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            group_record: dict[str, Any],
            task_started_event: asyncio.Event,
        ) -> None:
            task_started_event.set()
            self._inflight_rollouts += 1
            prompt = group_record["prompt"]
            target_step = group_record["target_step"]
            # This task owns one slot of a step, which can outlive both the prompt it
            # started with and the step it started on: a dropped prompt is substituted in
            # place and the loop runs again, and the slot is re-aimed at whichever step
            # lends this one a finished group. Both permits are held across
            # substitutions because the slot stays occupied either way -- they are
            # released once something commits, or once a step is credited short.
            replacements = 0
            try:
                while True:
                    try:
                        outcome = await self._rollout_manager.generate_and_push(
                            prompt,
                            target_step=target_step,
                            inflight_registry=_AdmissionInflightRegistry(
                                self._inflight_by_group_id,
                                group_record,
                                self._rollout_permitted,
                                self._inflight_registry_changed,
                            ),  # type: ignore[arg-type]
                        )
                    except BaseException:
                        # On success ownership transfers to the train pump, which
                        # releases this permit after consuming the committed group.
                        self._buffer_capacity.release()
                        raise

                    if outcome is not RolloutOutcome.SKIPPED:
                        break

                    async with self._rollout_admission_lock:
                        replacement = self._take_replacement(target_step, replacements)
                        if replacement is None:
                            # Nothing was committed, so the train pump will never see
                            # this group and never release its permit on our behalf.
                            self._buffer_capacity.release()
                            self._credit_shortfall(target_step)
                            self._drop_pending_group(group_record)
                            return

                        replacements += 1
                        prompt = replacement
                        group_record["prompt"] = replacement
                        group_record["group_id"] = None
                        # Attempted only now that a spare is in hand, because the
                        # borrow is a debt and the spare is what repays it.
                        lender_step = self._promote_into_step(target_step)
                        if lender_step is not None:
                            target_step = lender_step
                            group_record["target_step"] = lender_step
                    print(
                        f"  target_step={target_step}: substituting a spare prompt for "
                        f"the dropped group (replacement {replacements}/"
                        f"{self._async_cfg.rollout_failure.max_replacement_attempts}, "
                        f"{len(self._replacement_reserve)} spare(s) left)",
                        flush=True,
                    )
                    # A substitution is a fresh rollout, not a continuation of the one
                    # that failed, so it observes the same pause a first dispatch does
                    # instead of pushing new generation into a weight-sync window.
                    await self._rollout_permitted.wait()
            finally:
                self._inflight_rollouts -= 1
                sem.release()

            if replacements and target_step is not None:
                # Counted per slot, not per attempt: a step got its group back, which is
                # the fact that explains why its batch is full despite a drop. Recorded
                # against the step the spare actually committed to, which after a borrow
                # is the lender rather than the step that was dropped from.
                self._batch_replacements[target_step] = self._batch_replacements.get(target_step, 0) + 1

            if self._async_cfg.diagnostics:
                content = ""
                for i in range(len(prompt["message_log"])):
                    if prompt["message_log"][i]["role"] == "user":
                        content = prompt["message_log"][i]["content"]
                        break
                print(f"  rollout done for prompt='{content[:20]}...'", flush=True)

        def _release_permits_if_task_not_started(
            _: asyncio.Task[Any],
            *,
            task_started_event: asyncio.Event,
        ) -> None:
            if not task_started_event.is_set():
                self._buffer_capacity.release()
                sem.release()

        async def _launch(group_record: dict[str, Any]) -> None:
            # check if buffer is full
            await self._buffer_capacity.acquire()
            # check if inflight rollouts is full
            await sem.acquire()
            # wait for rollout to be permitted
            await self._rollout_permitted.wait()

            task_started_event = asyncio.Event()
            # dispatch rollout
            task = rollout_tasks.create_task(_dispatch_one_prompt(group_record, task_started_event))
            self._dispatched_rollouts.add(task)
            task.add_done_callback(self._dispatched_rollouts.discard)
            task.add_done_callback(
                partial(
                    _release_permits_if_task_not_started,
                    task_started_event=task_started_event,
                )
            )

        def _materialize_batch(
            prompt_batch: BatchedDataDict[DatumSpec],
        ) -> list[DatumSpec]:
            return [
                {k: v[prompt_idx] for k, v in prompt_batch.items()}  # type: ignore
                for prompt_idx in range(prompt_batch.size)
            ]

        def _record_yielded_batch(prompts: list[DatumSpec], *, source: str = "dataloader") -> dict[str, Any]:
            if not prompts:
                raise RuntimeError("rollout admission cannot hold an empty batch")
            entry = {
                "dispatch_index": None,
                "admitted": False,
                "source": source,
                "groups": [
                    {
                        "target_step": None,
                        "group_id": None,
                        "prompt": prompt,
                    }
                    for prompt in prompts
                ],
            }
            self._pending_rollout_admissions.append(entry)
            return entry

        def _mark_batch_admitted(entry: dict[str, Any], target_step: Optional[int]) -> None:
            if entry["admitted"] or entry["dispatch_index"] is not None:
                raise RuntimeError("rollout batch was admitted more than once")
            entry["dispatch_index"] = self._next_rollout_admission
            entry["admitted"] = True
            for group_record in entry["groups"]:
                group_record["target_step"] = target_step
            self._next_rollout_admission += 1

        def _drop_pending_entry(entry: dict[str, Any]) -> None:
            self._pending_rollout_admissions = deque(
                candidate for candidate in self._pending_rollout_admissions if candidate is not entry
            )

        async def _admit_yielded_batch(entry: dict[str, Any], *, max_num_steps: int) -> bool:
            """Admit a durably owned dataloader/reserve batch outside the cut lock."""
            target_step = await self._sampler.admit(trainer_version_fn=lambda: self._trainer_version)
            if target_step is not None and target_step >= max_num_steps:
                raise RuntimeError(
                    "rollout sampler admitted target_step=" f"{target_step} beyond grpo.max_num_steps={max_num_steps}"
                )
            async with self._rollout_admission_lock:
                _mark_batch_admitted(entry, target_step)
                if target_step is not None:
                    self._sampler_stamps_target_steps = True

                if entry["source"] == "reserve" and target_step is not None:
                    buffered = min(
                        self._buffer.count_for_target_step(target_step),
                        len(entry["groups"]),
                    )
                    if buffered:
                        missing = len(entry["groups"]) - buffered
                        returned_groups = entry["groups"][missing:]
                        entry["groups"] = entry["groups"][:missing]
                        self._replacement_reserve.extendleft(reversed([group["prompt"] for group in returned_groups]))
                        if not entry["groups"]:
                            _drop_pending_entry(entry)
                        print(
                            f"  pooled target_step={target_step}: {buffered} group(s) "
                            "already buffered; preserved the same number of spares",
                            flush=True,
                        )
            return bool(entry["groups"])

        async def _dispatch_admission(entry: dict[str, Any], restored_group_ids: set[str]) -> None:
            """Dispatch exactly the prompt groups absent from restored replay."""
            if not entry["admitted"]:
                raise RuntimeError("cannot dispatch a yielded batch before admission")
            # Reserve reconciliation can return every prompt to the pool when the
            # restored target is already full, removing this ledger entry entirely.
            if not entry["groups"]:
                return
            buffered = 0
            legacy_count_credit = 0
            target_step = entry["groups"][0]["target_step"]
            if (
                not self._rollout_admission_state_restored
                and entry["source"] == "dataloader"
                and target_step is not None
                and all(group["group_id"] is None for group in entry["groups"])
            ):
                legacy_count_credit = min(
                    self._buffer.count_for_target_step(target_step),
                    len(entry["groups"]),
                )
                if legacy_count_credit:
                    missing = len(entry["groups"]) - legacy_count_credit
                    returned_groups = entry["groups"][missing:]
                    entry["groups"] = entry["groups"][:missing]
                    self._replacement_reserve.extendleft(reversed([group["prompt"] for group in returned_groups]))
                    if not entry["groups"]:
                        _drop_pending_entry(entry)
            for group_record in entry["groups"]:
                group_id = group_record["group_id"]
                if group_id is not None and group_id in restored_group_ids:
                    restored_group_ids.remove(group_id)
                    buffered += 1
                    continue
                await _launch(group_record)
            buffered += legacy_count_credit
            if buffered:
                reconciliation = "legacy target-count credit" if legacy_count_credit else "exact group ID(s)"
                print(
                    f"  dispatch_index={entry['dispatch_index']}: {buffered} "
                    f"{reconciliation} already buffered; requeued only missing "
                    "prompts",
                    flush=True,
                )

        async def _advance_full_legacy_target(max_num_steps: int) -> bool:
            """Consume sampler admission, but no prompt, for a full legacy target."""
            if (
                not getattr(self, "_sampler_stamps_target_steps", False)
                or not self._legacy_untracked_replay_group_ids
                or self._next_rollout_admission >= max_num_steps
            ):
                return False
            target_step = self._next_rollout_admission
            groups_per_step = self._master_config.grpo.num_prompts_per_step
            if self._buffer.count_for_target_step(target_step) < groups_per_step:
                return False
            if self._rollout_admission_state_restored:
                target_group_ids = [
                    group_id
                    for buffered_target, group_id in zip(
                        self._buffer.target_step_list,
                        self._buffer._group_ids,
                        strict=True,
                    )
                    if buffered_target == target_step
                ]
                # A v2 checkpoint written while a legacy resume was still in
                # progress carries the old replay IDs forward. Only those exact IDs
                # authorize prompt-free admission; a full cohort produced under v2
                # has its own ledger entries and must reconcile there instead.
                if not target_group_ids or any(
                    group_id not in self._legacy_untracked_replay_group_ids for group_id in target_group_ids
                ):
                    return False

            # Do not hold the checkpoint/admission lock across a sampler gate: the
            # trainer may need the same lock to retire the cohort that opens it.
            admitted_target = await self._sampler.admit(trainer_version_fn=lambda: self._trainer_version)
            if admitted_target != target_step:
                raise RuntimeError(
                    "legacy replay target reconciliation expected sampler target_step="
                    f"{target_step}, got {admitted_target}"
                )
            async with self._rollout_admission_lock:
                self._next_rollout_admission += 1
            print(
                f"  legacy target_step={target_step} is already complete in the "
                "restored buffer; advanced admission without consuming prompts",
                flush=True,
            )
            return True

        # A checkpoint can contain ready groups and omit sibling slots whose rollout
        # was still in flight. Reconcile by exact group ID before touching the restored
        # dataloader; counts cannot distinguish "A ready, B in flight" from the
        # opposite completion order and would regenerate the wrong prompt.
        restored_group_ids = set(getattr(self._buffer, "_group_ids", ()))
        self._legacy_untracked_replay_group_ids.intersection_update(restored_group_ids)
        async with asyncio.TaskGroup() as rollout_tasks:
            # Training may retire the left edge while one of these launches awaits
            # capacity, so iterate a stable snapshot rather than the live deque.
            for entry in list(self._pending_rollout_admissions):
                if not entry["admitted"]:
                    await _admit_yielded_batch(
                        entry,
                        max_num_steps=self._master_config.grpo.max_num_steps,
                    )
                await _dispatch_admission(entry, restored_group_ids)
        unexpected_restored_group_ids = restored_group_ids - self._legacy_untracked_replay_group_ids
        if unexpected_restored_group_ids and self._rollout_admission_state_restored:
            raise RuntimeError(
                "restored replay buffer contains group ID(s) absent from the paired "
                "admission ledger or its legacy carry set: "
                f"{sorted(unexpected_restored_group_ids)}"
            )

        max_epochs = self._master_config.grpo.max_num_epochs
        max_num_steps = self._master_config.grpo.max_num_steps
        admission_budget_exhausted = self._next_rollout_admission >= max_num_steps
        async with asyncio.TaskGroup() as rollout_tasks:
            while not admission_budget_exhausted and (max_epochs is None or self._current_epoch < max_epochs):
                dataloader_iter = iter(self._dataloader)
                while True:
                    entry: dict[str, Any] | None = None
                    epoch_exhausted = False
                    if await _advance_full_legacy_target(max_num_steps):
                        if self._next_rollout_admission >= max_num_steps:
                            admission_budget_exhausted = True
                            break
                        continue
                    async with self._rollout_admission_lock:
                        if (
                            self._next_rollout_admission >= max_num_steps
                            and not self._should_divert_next_batch_to_reserve()
                        ):
                            admission_budget_exhausted = True
                        else:
                            try:
                                prompt_batch = next(dataloader_iter)
                            except StopIteration:
                                epoch_exhausted = True

                        if epoch_exhausted:
                            self._current_epoch += 1
                        elif admission_budget_exhausted:
                            pass
                        elif self._divert_batch_to_reserve(prompt_batch):
                            pass
                        else:
                            # Own the yielded batch before sampler.admit can block.
                            # A checkpoint now has its prompts even though neither the
                            # sampler cursor nor absolute admission cursor advanced.
                            entry = _record_yielded_batch(_materialize_batch(prompt_batch))

                    if epoch_exhausted or admission_budget_exhausted:
                        break
                    if entry is not None:
                        await _admit_yielded_batch(entry, max_num_steps=max_num_steps)
                        await _dispatch_admission(entry, restored_group_ids)

        # Only now that every dispatched rollout has settled is the pool genuinely
        # spare. Draining it inside the group above would race them for it, and a
        # rollout that was about to be dropped has the better claim: it needs a spare to
        # keep its step whole, whereas an extra step is only worth having if one is
        # left over. A second group because the first is closed to new tasks.
        while await _advance_full_legacy_target(max_num_steps):
            pass
        async with asyncio.TaskGroup() as rollout_tasks:
            await self._drain_reserve_into_steps(
                _launch,
                max_steps=max(0, max_num_steps - self._next_rollout_admission),
            )
        # A reserve top-up may have exposed a later full lookahead cohort. Advance it
        # even when max_epochs already put the dataloader loop at EOF; otherwise a
        # final checkpoint can persist next_admission < train_steps after training
        # consumes that cohort and then reject its own state on the next resume.
        while await _advance_full_legacy_target(max_num_steps):
            pass

        # Drain in-flight so return implies "all rollouts in TQ".
        inflight = list(self._dispatched_rollouts)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

        self._rollout_exhausted.set()
        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    def _should_divert_next_batch_to_reserve(self) -> bool:
        """Whether the next dataloader batch is owed to the replacement pool."""
        failure_cfg = self._async_cfg.rollout_failure
        return (
            failure_cfg.on_dropped_prompt == "replace"
            and self._sampler_stamps_target_steps
            and len(self._replacement_reserve) < failure_cfg.replacement_reserve_prompts
        )

    def _divert_batch_to_reserve(self, prompt_batch: BatchedDataDict[DatumSpec]) -> bool:
        """Consume a whole batch as spare prompts instead of admitting it as a step.

        Returns whether the batch was taken, in which case the caller must not admit it.
        Diverting before ``admit`` is what keeps the stamp sequence honest: admitting a
        batch and then dispatching nothing for it would leave a target step that no
        group is ever generated for, which is exactly the hang the shortfall accounting
        exists to prevent.

        A whole batch at a time because the dataloader only yields batches. The spares
        that go unused are not wasted work -- nothing has been generated for them -- and
        they stay in the pool for later steps.

        Nothing is diverted until the sampler has actually stamped a batch, so a run
        whose sampler never stamps does not lose a batch of prompts to a pool it can
        never draw on. The cost is that the first batch is always admitted rather than
        diverted; in practice the pool is filled while that first batch's rollouts are
        still running, so it is available by the time any of them can be given up on.
        """
        if not self._should_divert_next_batch_to_reserve():
            return False
        failure_cfg = self._async_cfg.rollout_failure

        for prompt_idx in range(prompt_batch.size):
            spare: DatumSpec = {k: v[prompt_idx] for k, v in prompt_batch.items()}  # type: ignore
            self._replacement_reserve.append(spare)
        print(
            f"  spare pool refilled with {prompt_batch.size} prompt(s) "
            f"(low-water mark {failure_cfg.replacement_reserve_prompts}); this batch is "
            "not admitted as a training step",
            flush=True,
        )
        return True

    async def _drain_reserve_into_steps(
        self,
        launch: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        max_steps: int,
    ) -> int:
        """Train on the leftover spares once the dataloader has nothing more to give.

        Spares were consumed from the dataset like any other prompt, so leaving them in
        the pool at the end of the last epoch throws away data the run already paid for.

        It also restores the step count. ``_clamp_max_num_steps`` derives
        ``max_num_steps`` from ``len(dataloader)``, and every diverted batch is one
        fewer batch the loop can admit -- so without this a replace-mode run quietly
        finishes one step short of the budget it was configured with, per divert.

        Whole steps only. A partial pool dispatched as a step is short by construction,
        and ``min_step_batch_fraction`` would then reject it and fail a run that had
        otherwise completed cleanly. In the ordinary case the pool holds exactly one
        batch (the dataloader uses ``batch_size=num_prompts_per_step``), so the common
        outcome is that the whole thing is recovered.

        ``max_steps`` is the unused global training-step budget. It distinguishes
        dataloader exhaustion, where a diverted batch should restore a missing step,
        from an explicit ``grpo.max_num_steps`` stop, where draining the same batch
        would generate work the trainer can never consume.

        Not gated on ``on_dropped_prompt``: an empty pool makes this a no-op anyway, and
        only "replace" ever fills one, so the gate would buy nothing while stranding a
        pool restored from a checkpoint into a run that has since switched to "shrink".

        Returns:
            The number of additional training steps dispatched from the pool.
        """
        if max_steps < 0:
            raise ValueError(f"max_steps must be non-negative, got {max_steps}")

        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        max_num_steps = self._master_config.grpo.max_num_steps
        dispatched_steps = 0
        while dispatched_steps < max_steps:
            async with self._rollout_admission_lock:
                if len(self._replacement_reserve) < num_prompts_per_step:
                    break
                # Own K prompts before sampler.admit can block. A checkpoint sees
                # them either in the reserve or in this unadmitted entry.
                step_prompts = [self._replacement_reserve.popleft() for _ in range(num_prompts_per_step)]
                entry = {
                    "dispatch_index": None,
                    "admitted": False,
                    "source": "reserve",
                    "groups": [
                        {
                            "target_step": None,
                            "group_id": None,
                            "prompt": prompt,
                        }
                        for prompt in step_prompts
                    ],
                }
                self._pending_rollout_admissions.append(entry)

            target_step = await self._sampler.admit(trainer_version_fn=lambda: self._trainer_version)
            if target_step is not None and target_step >= max_num_steps:
                raise RuntimeError(
                    "rollout sampler admitted pooled target_step="
                    f"{target_step} beyond grpo.max_num_steps={max_num_steps}"
                )

            async with self._rollout_admission_lock:
                entry["dispatch_index"] = self._next_rollout_admission
                entry["admitted"] = True
                for group_record in entry["groups"]:
                    group_record["target_step"] = target_step
                self._next_rollout_admission += 1
                if target_step is not None:
                    self._sampler_stamps_target_steps = True
                    buffered = self._buffer.count_for_target_step(target_step)
                else:
                    buffered = 0
                buffered = min(buffered, num_prompts_per_step)
                missing = num_prompts_per_step - buffered
                returned_groups = entry["groups"][missing:]
                entry["groups"] = entry["groups"][:missing]
                self._replacement_reserve.extendleft(reversed([group["prompt"] for group in returned_groups]))
                if not entry["groups"]:
                    self._pending_rollout_admissions = deque(
                        candidate for candidate in self._pending_rollout_admissions if candidate is not entry
                    )

            if not missing:
                dispatched_steps += 1
                print(
                    f"  pooled target_step={target_step} is already complete in the "
                    "restored buffer; preserving the reserve and dispatching 0 "
                    "duplicate prompts",
                    flush=True,
                )
                continue
            print(
                f"  dataloader exhausted; training on {missing} pooled "
                f"spare(s) as target_step={target_step}; {buffered} group(s) "
                "already buffered",
                flush=True,
            )
            for group_record in entry["groups"]:
                await launch(group_record)
            dispatched_steps += 1

        if self._replacement_reserve:
            if len(self._replacement_reserve) < num_prompts_per_step:
                detail = f"fewer than the {num_prompts_per_step} a step needs"
            else:
                detail = "the training-step budget is exhausted"
            print(
                f"  {len(self._replacement_reserve)} pooled spare(s) left over; "
                f"{detail}, so they are not trained on",
                flush=True,
            )
        return dispatched_steps

    def _take_replacement(self, target_step: Optional[int], replacements_used: int) -> Optional[DatumSpec]:
        """A spare prompt to stand in for a dropped group, or None to shrink instead.

        None covers the four ways a replacement can be unavailable: it was not asked
        for, the sampler did not stamp this prompt so no step is waiting on it, the
        per-slot budget is spent, or the pool is empty because the dataloader is
        exhausted. Every one of them falls back to ``on_dropped_prompt="shrink"`` rather
        than waiting, because a step whose replacements keep failing still has to close.
        """
        failure_cfg = self._async_cfg.rollout_failure
        if failure_cfg.on_dropped_prompt != "replace":
            return None
        if target_step is None:
            return None
        if replacements_used >= failure_cfg.max_replacement_attempts:
            return None
        if not self._replacement_reserve:
            return None
        return self._replacement_reserve.popleft()

    def _promote_into_step(self, target_step: Optional[int]) -> Optional[int]:
        """Fill a dropped step from a later step's finished work, and name the lender.

        Where a replacement goes, rather than whether one happens. The lost step closes
        on generation that already exists instead of waiting out a rollout with the
        trainer idle, and the caller redirects its spare prompt to the lender, which is
        due a training step later and has the slack to absorb the wait. The same prompt
        is generated either way.

        Only ever reached with a spare already in hand, which is what makes the borrow
        safe to take: an unrepaid loan is the same hole one step later.

        Returns None -- leaving the caller filling the dropped step directly -- when
        nothing is stamped so no step is stranded, when the trainer has already moved
        past this step (a second drop can land after the first one closed it short, and
        a group stamped for a finished step would only be evicted), or when no later
        step has a finished group to lend. The last is always the case at
        ``in_order.max_lookahead_versions=0``, where the next batch is not dispatched
        until this step trains.

        Returns:
            The step that lent the group, which the caller now owes a rollout, or None.
        """
        if target_step is None:
            return None
        if target_step < self._trainer_version:
            return None
        targets_before = list(getattr(self._buffer, "target_step_list", ()))
        lender_step = self._buffer.promote_ready_group(to_target_step=target_step)
        if lender_step is None:
            return None
        targets_after = list(getattr(self._buffer, "target_step_list", ()))
        group_ids = list(getattr(self._buffer, "_group_ids", ()))
        if len(group_ids) == len(targets_before) == len(targets_after):
            promoted_indices = [
                index
                for index, (before, after) in enumerate(zip(targets_before, targets_after))
                if before == lender_step and after == target_step
            ]
            if len(promoted_indices) == 1:
                self._retarget_pending_group(group_ids[promoted_indices[0]], target_step)
        self._batch_promotions[target_step] = self._batch_promotions.get(target_step, 0) + 1
        print(
            f"  target_step={target_step}: filled the dropped group by promoting a "
            f"finished group from target_step={lender_step}; the spare prompt is "
            "dispatched to repay that step instead",
            flush=True,
        )
        return lender_step

    def _credit_shortfall(self, target_step: Optional[int]) -> None:
        """Record that a stamped step will never receive a group it is waiting for."""
        if target_step is None:
            return
        self._batch_shortfall[target_step] = self._batch_shortfall.get(target_step, 0) + 1
        print(
            f"  target_step={target_step} is one group short "
            f"({self._batch_shortfall[target_step]} total); the train pump "
            "will close that step early",
            flush=True,
        )

    def _target_groups_for_step(self, step: int) -> int:
        """How many prompt groups this step should train on, after dropped prompts.

        ``num_prompts_per_step`` is the target; groups stamped for this step that were
        given up on are subtracted, because they are never arriving and a sampler that
        matches batches to steps exactly cannot substitute another step's groups for
        them. Without this the pump waits on a group no one is generating.

        The step trains on fewer samples than configured, which is the point: a smaller
        step beats a stalled run. The count is logged as ``dropped_prompt_groups`` so
        the batch size a step actually used is recoverable afterwards.

        How much smaller is bounded by ``min_step_batch_fraction``, and that bound has
        to live here because neither drop budget provides it. Both budgets are
        run-scoped -- the consecutive counter is cleared by any commit, including
        commits for other steps -- so drops landing on one step while other steps
        succeed can shrink it without ever tripping them.

        Raises:
            RuntimeError: The step fell below ``min_step_batch_fraction`` of
                ``num_prompts_per_step``. Training a fraction of a batch is a silent
                change to the gradient estimate, so it is refused rather than absorbed.
        """
        num_prompts_per_step = self._master_config.grpo.num_prompts_per_step
        dropped = self._batch_shortfall.get(step, 0)
        target = num_prompts_per_step - dropped
        fraction = self._async_cfg.rollout_failure.min_step_batch_fraction
        # ceil, so the floor is never rounded down into allowing one more drop than the
        # fraction states. With fraction > 0 this is always >= 1, which also rules out
        # the empty step.
        floor = math.ceil(num_prompts_per_step * fraction)
        if target < floor:
            raise RuntimeError(
                f"training step {step} lost {dropped} of {num_prompts_per_step} prompt "
                f"group(s), leaving {target}, below the floor of {floor} set by "
                f"async_rl.rollout_failure.min_step_batch_fraction={fraction}. "
                "Either the generation fleet is failing a whole step's worth of "
                "prompts, or the drop budgets are set too high to catch it: they are "
                "run-scoped and cannot bound how short a single step gets."
            )
        return target

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_stage(train_meta).
          4. trainer.train_microbatches_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        grpo_cfg = self._master_config.grpo

        while self._train_steps < grpo_cfg.max_num_steps:
            version_during_step = self._trainer_version
            groups_dispatched = 0
            evicted_stale_prompt_groups = 0
            min_sample_version = None
            step_open = False
            chunks_dispatched = 0
            calibration_batches: list[BatchedDataDict[Any]] = []
            trained_group_ids: set[str] = set()

            with self._timer.time("total_step_time"):
                # Re-read on every iteration rather than once: a prompt stamped for this
                # step can be dropped while the pump is already waiting for it, which is
                # precisely the case that would otherwise wait forever.
                while groups_dispatched < self._target_groups_for_step(version_during_step):
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        group_ids_before_evict = set(getattr(self._buffer, "_group_ids", ()))
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        if evicted:
                            evicted_group_ids = group_ids_before_evict - set(getattr(self._buffer, "_group_ids", ()))
                            self._retire_rollout_group_ids(evicted_group_ids)
                        evicted_stale_prompt_groups += evicted
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch. Read the target again rather than reusing
                        # the loop condition's value: the awaits above are a window in
                        # which a prompt stamped for this step can be dropped, and the
                        # target would then be stale by the time it is subtracted. It
                        # can also have fallen to what is already dispatched, which is
                        # not a batch the sampler can be asked for -- select() rejects
                        # a min below 1 -- so close the step instead.
                        target_groups = self._target_groups_for_step(version_during_step)
                        max_prompt_groups = target_groups - groups_dispatched
                        if max_prompt_groups <= 0:
                            break
                        min_prompt_groups, max_prompt_groups = _train_selection_group_bounds(
                            remaining_groups=max_prompt_groups,
                            configured_min_groups=(self._async_cfg.min_groups_for_streaming_train),
                            group_multiple=self._train_dispatch_group_multiple,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        if (
                            train_meta is not None
                            and self._train_dispatch_group_multiple > 1
                            and num_groups != min_prompt_groups
                        ):
                            raise RuntimeError(
                                "shared-prefix train sampler violated its exact "
                                "DP-divisible claim: requested "
                                f"{min_prompt_groups} prompt groups, got {num_groups}."
                            )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            if self._rollout_exhausted.is_set():
                                buffered_groups = len(self._buffer)
                                if groups_dispatched == 0 and buffered_groups == 0:
                                    print(
                                        "train_pump: rollout exhausted and " "buffer drained",
                                        flush=True,
                                    )
                                    return
                                # Against the step's own target, not the configured
                                # batch size: a step that legitimately shrank would
                                # otherwise be reported as missing groups it was
                                # already excused from.
                                raise RuntimeError(
                                    "rollout exhausted before a complete training "
                                    f"step was assembled: dispatched "
                                    f"{groups_dispatched}/{target_groups} prompt "
                                    f"groups with {buffered_groups} group(s) "
                                    f"remaining in the buffer"
                                )
                            await asyncio.sleep(0.005)
                            continue

                        # Sample IDs are ``{group_id}_g{i}``. Track the exact groups
                        # selected because ready-first/freshest policies may consume a
                        # different admission order than FIFO completion.
                        selected_group_ids = {
                            head
                            for sample_id in train_meta.sample_ids
                            for head, separator, row in [sample_id.rpartition("_g")]
                            if head and separator and row.isdigit()
                        }
                        if len(selected_group_ids) == num_groups:
                            trained_group_ids.update(selected_group_ids)

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    if self._policy_logprobs_required or self._reference_logprobs_required:
                        with self._timer.time("logprob_inference_prep"):
                            # Once the step is open, gradients are accumulating
                            # in the trainer's grad buffers across chunks. The
                            # Megatron buffer offload frees that storage outright
                            # and its reload zeroes it, so offloading here would
                            # discard every chunk but the last while the 1/N
                            # normalizer still counts all of them.
                            await asyncio.to_thread(
                                self._trainer.prepare_for_lp_inference,
                                keep_train_buffers=step_open,
                            )
                        with self._timer.time("policy_and_reference_logprobs"):
                            if self._policy_logprobs_required:
                                await asyncio.to_thread(self._trainer.get_logprobs_from_meta, train_meta)
                            if self._reference_logprobs_required:
                                await asyncio.to_thread(
                                    self._trainer.get_reference_policy_logprobs_from_meta,
                                    train_meta,
                                )

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        (
                            train_meta,
                            has_valid_training_tokens,
                        ) = await self._advantage_stage(train_meta)

                    # Filtering can leave a streaming chunk with no training tokens.
                    # Consume that chunk without F/B, then continue the same optimizer
                    # step with the next chunk. Always restore training mode because
                    # log-prob inference may have switched the model to inference mode.
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    if has_valid_training_tokens:
                        with self._timer.time("policy_training"):
                            if not step_open:
                                await asyncio.to_thread(
                                    self._trainer.begin_train_step,
                                    self._loss_fn,
                                )
                                step_open = True
                            await asyncio.to_thread(
                                self._trainer.train_microbatches_from_meta,
                                train_meta,
                            )

                    # Keep token accounting scoped to the exact optimizer step.
                    # A fully filtered streaming chunk is consumed but never enters
                    # F/B, and _advantage_stage intentionally records no shared-prefix
                    # opportunity for it either.
                    if has_valid_training_tokens and train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(int(s) for s in train_meta.sequence_lengths)

                    if getattr(self._gen, "requires_kv_scale_sync", False):
                        calibration_fields = [
                            field for field in (train_meta.fields or []) if field in DP_CALIB_INPUT_FIELDS
                        ]
                        calibration_batches.append(
                            await asyncio.to_thread(
                                self._trainer.read_from_dataplane,
                                train_meta,
                                select_fields=calibration_fields,
                            )
                        )

                    # Refresh min_sample_version
                    curr_min_sample_version = min(t["weight_version"] for t in train_meta.tags)  # type: ignore
                    if min_sample_version is not None:
                        min_sample_version = min(min_sample_version, curr_min_sample_version)
                    else:
                        min_sample_version = curr_min_sample_version

                    # Remove consumed sample_ids from the buffer
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=list(train_meta.sample_ids),
                        partition_id=self._partition_id,
                    )

                    groups_dispatched += num_groups
                    chunks_dispatched += 1
                    # How many chunks a step is split into decides how many times
                    # gradients accumulate before the single reduce, so record it
                    # rather than leaving it to be inferred from phase timings.
                    #
                    # These reach a run's output only because nemo_rl/__init__.py
                    # sets the `nemo_rl` logger to NRL_LOG_LEVEL (INFO by
                    # default); the bare basicConfig() there pins the root logger
                    # at WARNING and no later basicConfig can raise it. Note that
                    # the handler writes to stderr while the progress prints
                    # around this write to stdout, so the two are not guaranteed
                    # to interleave in order in a Ray driver log.
                    log.info(
                        "train_pump: step %d chunk %d: %d group(s), %d/%d dispatched",
                        version_during_step,
                        chunks_dispatched,
                        num_groups,
                        groups_dispatched,
                        grpo_cfg.num_prompts_per_step,
                    )

                log.info(
                    "train_pump: step %d closing on %d chunk(s), %d group(s)",
                    version_during_step,
                    chunks_dispatched,
                    groups_dispatched,
                )
                if not step_open:
                    raise RuntimeError(
                        "SingleController has no valid response tokens after "
                        "filtering. Check grpo.seq_logprob_error_threshold to "
                        "avoid an optimizer step with an empty batch."
                    )

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(reduce_advantage_pump_metrics(**self._step_log_dict))
                self._step_log_dict = {k: [] for k in self._step_log_dict}
                if self._step_shared_prefix_opportunities:
                    step_metrics.update(
                        _reduce_shared_prefix_step_metrics(
                            self._step_shared_prefix_opportunities,
                            expected_total_tokens=step_metrics.get("total_num_tokens"),
                        )
                    )
                    self._step_shared_prefix_opportunities = []

                # The JSONL can be large. Keep the published trainer version at
                # the generation-visible version while it is written; otherwise
                # this await lets the rollout pump admit version N+1 work before
                # _sync_weights has moved generation off version N.
                await asyncio.to_thread(
                    self._flush_train_data_records,
                    step=self._train_steps + 1,
                )
                # Retire before publishing the incremented trainer step. Checkpointing
                # happens later in this same iteration, so its cursor, dataloader, and
                # pending-admission ledger all describe one coherent boundary.
                self._retire_rollout_admission(
                    version_during_step,
                    trained_group_ids=trained_group_ids,
                )
                self._trainer_version += 1
                self._train_steps += 1
                dropped_prompt_groups = self._batch_shortfall.get(version_during_step, 0)
                replaced_prompt_groups = self._batch_replacements.get(version_during_step, 0)
                promoted_prompt_groups = self._batch_promotions.get(version_during_step, 0)
                # Prune every stamp this step or older. Popping only this step's entry
                # would leak the ones belonging to a step that was already closed when
                # a straggler stamped for it was finally given up on.
                self._batch_shortfall = {
                    step: dropped for step, dropped in self._batch_shortfall.items() if step > version_during_step
                }
                self._batch_replacements = {
                    step: replaced for step, replaced in self._batch_replacements.items() if step > version_during_step
                }
                self._batch_promotions = {
                    step: promoted for step, promoted in self._batch_promotions.items() if step > version_during_step
                }
                with self._timer.time("weight_sync"):
                    calibration_data = (
                        BatchedDataDict.from_batches(calibration_batches) if calibration_batches else None
                    )
                    aborted_stale_inflight_groups = await self._sync_weights(calibration_data=calibration_data)
                    step_metrics.update(
                        {
                            "evicted_stale_prompt_groups": evicted_stale_prompt_groups,
                            "aborted_stale_inflight_groups": aborted_stale_inflight_groups,
                            # Non-zero means this step trained on a smaller batch than
                            # num_prompts_per_step, which any comparison of step metrics
                            # across steps has to account for.
                            "dropped_prompt_groups": dropped_prompt_groups,
                            # Groups filled by a spare prompt this step waited on --
                            # either one it lost itself, or one it lent to an earlier
                            # step and was repaid for. Non-zero here with zero above is
                            # the healthy shape of on_dropped_prompt="replace": the
                            # batch stayed whole, and the cost was the wall-clock spent
                            # waiting on the spare.
                            "replaced_prompt_groups": replaced_prompt_groups,
                            # Groups this step lost and filled by borrowing finished work
                            # from a later step. The better shape of the same thing: the
                            # batch stayed whole and nothing waited for it, with the
                            # repayment showing up as a replacement on the lender.
                            "promoted_prompt_groups": promoted_prompt_groups,
                        }
                    )

                # Checkpointing (mirrors async_grpo_train's save block).
                # What the step actually trained on, which is num_prompts_per_step only
                # when nothing was dropped. Counted from the dispatch tally rather than
                # derived from the shortfall so the figure does not depend on the
                # bookkeeping staying exact; this lands in the checkpoint.
                self._consumed_samples += groups_dispatched
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                is_last_step = self._train_steps >= grpo_cfg.max_num_steps or (
                    self._rollout_exhausted.is_set() and len(self._buffer) == 0
                )
                ft_save_period = self._master_config.checkpointing.get("ft_save_period")
                # _train_steps was already incremented above, so it equals
                # the legacy loop's 1-indexed `step + 1`.
                should_save_by_step = (
                    is_last_step
                    or self._train_steps % self._master_config.checkpointing["save_period"] == 0
                    or (ft_save_period is not None and self._train_steps % ft_save_period == 0)
                )
                should_save_by_timeout = self._timeout.check_save()

                if self._master_config.checkpointing["enabled"] and (should_save_by_step or should_save_by_timeout):
                    with self._timer.time("checkpointing"):
                        await self._save_checkpoint(step_metrics)

            timing_metrics: dict[str, float] = self._timer.get_timing_metrics(reduction_op="sum")  # type: ignore

            total_time = timing_metrics.get("total_step_time", 0.0)
            total_num_gpus = int(ray.cluster_resources().get("GPU", 0))
            if total_time > 0 and total_num_gpus > 0 and "global_valid_toks" in step_metrics:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    step_metrics["global_valid_toks"] / total_time / total_num_gpus
                )

            print("\n⏱️  Timing:")
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
                if k == "total_step_time":
                    continue
                percent = (v / total_time * 100) if total_time > 0 else 0.0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            # TODO: vllm metrics logger, histogram log, rollout_metrics,
            #   pretty-print "Training Results" block, print_performance_metrics.
            print(f"step_metrics={step_metrics}", flush=True)
            self._logger.log_metrics(step_metrics, step=self._train_steps, prefix="train")
            self._logger.log_metrics(
                timing_metrics,
                step=self._train_steps,
                prefix="timing/train",
                step_finished=True,
            )
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = training version - oldest sample version.
            lag = version_during_step - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg.max_num_steps}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                break

    async def _stall_watchdog_pump(self) -> None:
        """Report rollout health, and detect stalls nothing else catches.

        Progress is the pair (committed groups, completed train steps) rather than a
        timestamp: both counters already exist, and "neither has moved" is the property
        that actually matters.

        Deliberately *not* conditioned on rollouts being in flight. An earlier version
        required that, on the reasoning that an idle controller has legitimately no
        work -- and a fault-injection run walked straight through the gap. Killing a
        generation worker wedged the loop with zero rollouts in flight and zero
        failures recorded: the rollout pump was blocked on backpressure behind a train
        pump that could no longer finish a step, so nothing was in flight to count.
        The watchdog observed six minutes of idleness and said nothing.

        What separates a real stall from an idle gap is whether work remains, so that
        is what is checked instead.
        """
        watchdog_cfg = self._async_cfg.stall_watchdog
        max_num_steps = self._master_config.grpo.max_num_steps
        last_progress = (-1, -1)
        last_progress_at = time.monotonic()

        while True:
            await asyncio.sleep(watchdog_cfg.interval_s)
            now = time.monotonic()

            stats = self._rollout_manager.stats
            progress = (stats.committed, self._train_steps)
            if progress != last_progress:
                last_progress = progress
                last_progress_at = now
            idle_s = now - last_progress_at

            metrics = dict(stats.as_metrics())
            metrics["rollout/inflight"] = float(self._inflight_rollouts)
            metrics["rollout/idle_s"] = idle_s
            metrics["rollout/train_steps"] = float(self._train_steps)
            if self._gen_fleet is not None:
                metrics.update(self._gen_fleet.as_metrics())
            if self._generation_router is not None:
                # router/* counters are exactly what you want when a backend starts
                # failing; computed since P2 landed but never published until now.
                # Best-effort like the membership push: a router being recreated must
                # not cost a metrics tick.
                try:
                    metrics.update(await self._ray_get(self._generation_router.metrics.remote()))
                except Exception as error:  # noqa: BLE001 - metrics are advisory
                    print(
                        f"watchdog: router metrics unavailable this tick: " f"{type(error).__name__}: {error}",
                        flush=True,
                    )
            self._logger.log_metrics(metrics, step=self._train_steps)

            if watchdog_cfg.gym_subprocess_check:
                # Bounded by one tick so a wedged environment cannot stop the pump, and
                # routed through stall_action so "warn" means warn -- see
                # _check_env_health.
                problems = await self._check_env_health(watchdog_cfg.interval_s)
                if problems:
                    detail = "; ".join(problems)
                    if watchdog_cfg.stall_action == "abort":
                        raise RuntimeError(f"environment health check failed -- {detail}")
                    print(f"WARNING: environment health -- {detail}", flush=True)

            if self._gen_fleet is not None:
                # Raises once too few shards remain for the run to be worth continuing.
                # Checked after publishing so the final state is on record.
                self._gen_fleet.raise_if_exhausted()

            work_remains = self._train_steps < max_num_steps
            if work_remains and idle_s > watchdog_cfg.stall_timeout_s:
                message = (
                    f"no rollout committed and no train step completed in "
                    f"{idle_s:.0f}s ({self._inflight_rollouts} rollouts in flight, "
                    f"{stats.committed} groups committed, step "
                    f"{self._train_steps}/{max_num_steps}, "
                    f"stall_timeout_s={watchdog_cfg.stall_timeout_s})"
                )
                if watchdog_cfg.stall_action == "abort":
                    raise RolloutStall(message)
                print(f"WARNING: rollout stall -- {message}", flush=True)

    async def _gen_fleet_probe_pump(self) -> None:
        """Probe the generation fleet on its own clock.

        Separate from the watchdog because the two cadences answer different questions.
        The watchdog publishes counters and notices a stalled run, which is a
        minutes-scale concern; liveness detection is the input to every recovery
        decision and has to be seconds-scale.

        Sharing the watchdog's loop made ``probe_interval_s`` decorative -- probes ran at
        ``watchdog.interval_s`` and nothing read the configured value. With the shipped
        defaults that put detection at ``30s * unhealthy_threshold``, i.e. 60-90s, which
        is *longer* than the refit deadline: by the time a hung refit aborted, the monitor
        still had the dead shard as SUSPECT, so the rebuild that abort exists to trigger
        saw an empty absent set and did nothing. Arithmetic, not a race -- it could never
        have worked. Job 5925668.
        """
        interval_s = self._async_cfg.generation_fleet_health.probe_interval_s
        while True:
            await asyncio.sleep(interval_s)
            await self._probe_generation_fleet()
            # Both of these are best-effort: they talk to a max_restarts=-1 actor that
            # may be mid-recreation, and run() awaits this task and re-raises, so an
            # unguarded RayActorError here would end the training job over a push that
            # the next tick would have retried anyway. GenerationFleetExhausted from the
            # watchdog stays the only fatal path -- the same bounded-failure contract
            # _check_env_health follows.
            try:
                await self._drain_router_failures()
                # Pushed here rather than on the watchdog's clock so a membership change
                # reaches the router at detection speed.
                await self._push_router_membership()
            except Exception as error:  # noqa: BLE001 - best-effort, retried next tick
                print(
                    f"fleet probe: router update failed, retrying next tick: " f"{type(error).__name__}: {error}",
                    flush=True,
                )

    async def _probe_generation_fleet(self) -> None:
        """Ask every serving generation shard whether it is still alive.

        Ray actor liveness is the cheap authoritative signal for "the process is gone",
        and it is what the probe uses. It does not catch every failure -- a vLLM engine
        core can die while the worker process and its HTTP thread survive -- which is
        why the routing adapters also report the failures they observe. The two signals
        feed the same counters.

        Only serving shards are probed: a quarantined shard answering again says nothing
        about whether its weights are current, and the monitor ignores such probes
        anyway.

        Shards are probed concurrently. Sequentially, a tick costs up to
        ``probe_timeout_s`` per shard, so a fleet of four would take 8s to complete a
        round the config promises every 5s -- and config validation only checks
        ``probe_timeout_s < probe_interval_s``, which silently assumes one probe per
        tick. Concurrent, a round is bounded by ``probe_timeout_s`` at any fleet size.
        """
        if self._gen_fleet is None:
            return

        fleet_cfg = self._async_cfg.generation_fleet_health
        worker_group = self._gen.worker_group

        async def probe(shard_idx: int) -> None:
            worker_idx = worker_group.get_dp_leader_worker_idx(shard_idx)
            try:
                await asyncio.wait_for(
                    self._ray_get(worker_group.workers[worker_idx].is_alive.remote()),
                    timeout=fleet_cfg.probe_timeout_s,
                )
            except (Exception, asyncio.TimeoutError) as error:
                self._gen_fleet.record_probe(shard_idx, ok=False, error=f"{type(error).__name__}: {error}")
            else:
                self._gen_fleet.record_probe(shard_idx, ok=True)

        await asyncio.gather(*(probe(idx) for idx in self._gen_fleet.serving_shards()))

    async def _push_router_membership(self) -> None:
        """Tell the NeMo-Gym router which backends are currently serving.

        Pushed as the full set rather than a delta, so a dropped or reordered update --
        or a restarted router, which comes up believing every backend serves -- converges
        on the next tick without sequence numbers or replay.

        Pushed unconditionally, not gated on the membership epoch moving. The gate looked
        free -- an unchanged serving set costs nothing to skip -- but it made the router's
        own restart unrecoverable: a recreated actor rebuilds ``_serving`` as *every*
        backend, while the epoch it was last pushed at has not moved, so the gate blocked
        every corrective push and Gym routed to a quarantined shard for the rest of the
        run. The payload is a short list of strings on a probe-interval timer; the gate
        bought nothing and cost the guarantee both docstrings advertised.

        It is also what makes the router's reflex drop safe: dropping a failing backend
        locally is only correct because a later push puts it back.
        """
        if self._generation_router is None or self._gen_fleet is None:
            return
        await self._ray_get(self._generation_router.set_serving_backends.remote(self._gen_fleet.serving_base_urls()))

    async def _drain_router_failures(self) -> None:
        """Fold the router's observed backend failures into the fleet ledger.

        The router is the only component that sees a *wedged* engine: it answers
        ``is_alive`` from a healthy worker process, so no probe can condemn it. The
        router holds no monitor reference by design -- membership flows one way -- so it
        counts failures per backend URL and this drains them here, on the tick that
        already talks to it.
        """
        if self._generation_router is None or self._gen_fleet is None:
            return
        counts: dict[str, int] = await self._ray_get(self._generation_router.drain_backend_failures.remote())
        for url, count in counts.items():
            shard_idx = self._gen_fleet.shard_for_base_url(url)
            if shard_idx is None:
                continue
            for _ in range(count):
                self._gen_fleet.report_failure(
                    shard_idx,
                    RuntimeError(f"router: {count} failed request(s) to {url}"),
                )

    async def _check_env_health(self, timeout_s: float) -> list[str]:
        """Ask each environment actor that exposes a health check whether it is whole.

        Returns the problems found, empty when everything is well. It *reports* rather
        than raises so the caller can route the verdict through ``stall_action``, the
        same way the stall path does. Raising here bypassed ``stall_action`` entirely:
        under the documented default (``"warn"``, which promises to "only report"), and
        with ``gym_subprocess_check`` defaulting to true, an unhealthy environment killed
        the run -- a run-ending path switched on by default, in a feature whose whole
        posture is inert-by-default.

        Each probe is bounded. ``NemoGym`` is an asyncio actor, so a *wedged* environment
        -- precisely the case this check exists to catch -- left the await hanging
        forever, the pump never ticked again, and stall detection was dead exactly when
        it was needed. A probe that does not answer within one tick IS the unhealthy
        signal; it is not a reason to stop watching.

        Environments without the method are skipped rather than treated as unhealthy;
        only NeMo-Gym has subprocess servers to lose.
        """
        problems: list[str] = []
        for env_name, handle in self._env_handles.items():
            health_check = getattr(handle, "health_check", None)
            if health_check is None:
                continue
            try:
                await asyncio.wait_for(self._ray_get(health_check.remote()), timeout=timeout_s)
            except asyncio.TimeoutError:
                problems.append(f"environment {env_name!r} did not answer its health check within " f"{timeout_s}s")
            except Exception as error:
                problems.append(f"environment {env_name!r} reported unhealthy: {error}")
        return problems

    async def _abort_stale_inflight(self) -> int:
        """Abort in-flight rollouts that the sampler can no longer select."""
        stale_groups = [
            (group_id, task)
            for group_id, (task, start_version) in self._inflight_by_group_id.items()
            if self._sampler.should_abort_inflight(
                start_weight_version=start_version,
                current_train_weight=self._trainer_version,
            )
        ]
        if not stale_groups:
            return 0

        stale_tasks = [task for _, task in stale_groups]
        self._retire_rollout_group_ids({group_id for group_id, _ in stale_groups})
        for task in stale_tasks:
            task.cancel()

        results = await asyncio.gather(*stale_tasks, return_exceptions=True)
        failures = [
            result
            for result in results
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
        ]
        if failures:
            raise BaseExceptionGroup(
                "stale in-flight rollout cleanup failed",
                failures,
            )

        print(
            f"  aborted {len(stale_tasks)} stale in-flight rollout(s)",
            flush=True,
        )
        return len(stale_tasks)

    async def _drain_gym_inflight_generations(self) -> int:
        """Wait until every Gym request admitted before the refit gate has left generation.

        Cancelling the local Gym task is insufficient: the HTTP request may keep
        running in Gym/vLLM after its caller is cancelled. RolloutManager consults
        the same gate before every attempt, so once the gate is cleared the registry
        can only shrink. Waiting for the registry to become empty therefore proves
        that no remote generation can overlap the weight update.
        """
        drained_group_ids: set[str] = set()
        while self._inflight_by_group_id:
            drained_group_ids.update(self._inflight_by_group_id)
            self._inflight_registry_changed.clear()
            # Avoid losing a removal that happened between the loop condition and
            # clear(); asyncio tasks can only mutate this mapping at await points.
            if not self._inflight_by_group_id:
                continue
            await self._inflight_registry_changed.wait()

        if drained_group_ids:
            print(
                f"  drained {len(drained_group_ids)} Gym in-flight generation(s)",
                flush=True,
            )
        return len(drained_group_ids)

    async def _save_checkpoint(self, step_metrics: dict[str, Any]) -> None:
        """Write a full checkpoint for the just-finished train step.

        Everything except the (possibly async) policy weight write must be
        on disk before begin_finalization; rollouts keep running throughout.
        """
        save_state = self._save_state
        # Hold the admission cut until replay_buffer.state_dict has captured its ready
        # slots. Otherwise a new unstamped batch can enter replay during the trainer
        # save awaits without appearing in this checkpoint's cursor or ledger.
        async with self._rollout_admission_lock:
            save_state.current_step = self._train_steps
            save_state.total_steps = self._train_steps
            save_state.current_epoch = self._current_epoch
            save_state.consumed_samples = self._consumed_samples
            save_state.total_valid_tokens = self._total_valid_tokens
            # The restore skips replay when the sampler changes because its stamps
            # might never be selectable under the new policy.
            save_state.sampler_name = self._async_cfg.sampler.name
            dataloader_state = self._dataloader.state_dict()
            reserve_state = list(self._replacement_reserve)
            rollout_admission_state = self._rollout_admission_state_dict()
            buffer_state = await self._buffer.state_dict(saved_capacity=self._async_cfg.max_buffered_rollouts)
        # SC has no validation loop yet; drop the default sentinel instead of
        # persisting a bogus val_reward.
        if hasattr(save_state, "val_reward"):
            delattr(save_state, "val_reward")

        # validate_single_controller_config already rejected anything but a
        # "train:" prefix, so step_metrics is the only source to consult.
        full_metric_name = self._master_config.checkpointing["metric_name"]
        if full_metric_name is not None:
            metric_name = full_metric_name.split(":", 1)[1]
            if metric_name not in step_metrics:
                raise ValueError(f"Metric {metric_name} not found in train metrics")
            setattr(save_state, full_metric_name, step_metrics[metric_name])

        # Flush the previous checkpoint's background finalization first;
        # re-raises a failure from it.
        await asyncio.to_thread(self._checkpointer.finalize_pending)

        print(f"Saving checkpoint for step {self._train_steps}...")
        checkpoint_path: PathLike = (
            await asyncio.to_thread(  # pyrefly: ignore[bad-assignment]  the PathLike alias resolves inconsistently under pyrefly's import-cycle breaking
                self._checkpointer.init_tmp_checkpoint,
                self._train_steps,
                vars(save_state),
                self._master_config,
            )
        )
        # With async_save this returns after D2H staging; disk writes finish
        # in the background.
        await asyncio.to_thread(
            self._trainer.save_checkpoint,
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=(
                os.path.join(checkpoint_path, "policy", "optimizer") if self._checkpointer.save_optimizer else None
            ),
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._master_config.checkpointing,
        )
        await asyncio.to_thread(
            torch.save,
            dataloader_state,
            os.path.join(checkpoint_path, "train_dataloader.pt"),
        )
        await asyncio.to_thread(
            torch.save,
            rollout_admission_state,
            os.path.join(checkpoint_path, _ROLLOUT_ADMISSION_STATE_FILE),
        )
        if reserve_state:
            await asyncio.to_thread(
                torch.save,
                reserve_state,
                os.path.join(checkpoint_path, "replacement_reserve.pt"),
            )
        await asyncio.to_thread(
            torch.save,
            buffer_state,
            os.path.join(checkpoint_path, "replay_buffer.pt"),
        )
        # Rename happens in the background once the async weight writes
        # finish; flushed at the next save or on exit.
        self._checkpointer.begin_finalization(
            checkpoint_path,
            wait_fn=self._trainer.finalize_async_save,
        )
        await asyncio.to_thread(
            _write_latest_checkpoint_status,
            self._checkpointer,
            last_checkpoint_step=self._train_steps,
        )

    async def _sync_weights(
        self,
        *,
        calibration_data: Optional[BatchedDataDict[Any]] = None,
    ) -> int:
        """Pause rollout generation, synchronize weights, invalidate caches, resume.

        Native rollouts retain the sampler's stale-abort semantics. Gym rollouts
        are drained because cancelling a local Gym request does not prove the
        corresponding remote vLLM request stopped.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. Abort stale native rollouts or drain all Gym generations.
          3. Optionally calibrate FP8 KV-cache scales.
          4. weight_synchronizer.sync_weights(kv_scales=...)
          5. Invalidate reusable generation caches when configured, and always
             for Gym where repeated prompts otherwise reuse pre-refit KV state.
          6. _rollout_permitted.set()   — resume

        Args:
            calibration_data: Optional data used to calibrate FP8 KV-cache
                scales before synchronizing weights.

        Returns:
            The number of stale in-flight rollout groups aborted before the
            weight synchronization.
        """
        self._rollout_permitted.clear()

        use_nemo_gym = should_use_nemo_gym(self._master_config)
        if use_nemo_gym:
            await self._drain_gym_inflight_generations()
            aborted_stale_inflight_groups = 0
        else:
            aborted_stale_inflight_groups = await self._abort_stale_inflight()

        t0 = time.monotonic()
        kv_scales = None
        if getattr(self._gen, "requires_kv_scale_sync", False) and calibration_data is not None:
            print("▶ Computing KV cache scales...", flush=True)
            calibration_result = await asyncio.to_thread(
                self._trainer.calibrate_qkv_fp8_scales,
                calibration_data,
                include_q=True,
            )
            kv_scales = calibration_result["layers"]

        await asyncio.to_thread(
            self._weight_synchronizer.sync_weights,
            kv_scales=kv_scales,
        )
        if use_nemo_gym or self._async_cfg.recompute_kv_cache_after_weight_updates:
            # to_thread, like every other call into the workers here. Run directly on
            # the loop this is a blocking Ray call, and a wedged generation worker would
            # freeze the event loop itself -- taking the watchdog, which is an asyncio
            # task on that same loop, down with it.
            invalidated = await asyncio.to_thread(self._gen.invalidate_kv_cache)
            if not invalidated:
                raise RuntimeError("Generation prefix/KV cache invalidation failed after weight update")
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()
        return aborted_stale_inflight_groups

    async def _advantage_stage(self, meta: KVBatchMeta) -> tuple[KVBatchMeta, bool]:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.

        Returns:
            The updated batch metadata and whether the batch contains at least
            one valid training token.
        """
        if self._advantage_estimator is None:
            return meta, True
        adv_cfg = self._advantage_cfg

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = tensor_field(data, adv_cfg.prompt_ids_field)
        verifier_rewards = (
            squeeze_trailing_unit_dim(tensor_field(data, adv_cfg.reward_field))
            .float()
            .detach()
            .clone()
        )
        rewards = verifier_rewards
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(tensor_field(data, adv_cfg.sample_mask_field)).float()

        repeated_batch = BatchedDataDict(
            {
                "total_reward": rewards,
                **{
                    field_name: squeeze_trailing_unit_dim(tensor_field(data, field_name))
                    for field_name in adv_cfg.repeated_batch_fields
                },
            }
        )
        reward_scaling_cfg = getattr(self._master_config.grpo, "reward_scaling", None)
        reward_shaping_cfg = getattr(self._master_config.grpo, "reward_shaping", None)
        reward_scaling_enabled = bool(getattr(reward_scaling_cfg, "enabled", False))
        reward_shaping_enabled = bool(getattr(reward_shaping_cfg, "enabled", False))
        if reward_scaling_cfg is not None:
            repeated_batch = scale_rewards(repeated_batch, reward_scaling_cfg)
        if reward_shaping_enabled:
            repeated_batch["truncated"] = squeeze_trailing_unit_dim(tensor_field(data, "truncated")).bool()
            repeated_batch["response_token_lengths"] = squeeze_trailing_unit_dim(
                tensor_field(data, "response_token_lengths")
            ).long()
            repeated_batch = apply_reward_shaping(repeated_batch, reward_shaping_cfg)
        rewards = repeated_batch["total_reward"]

        seq_logprob_error_threshold = self._master_config.grpo.seq_logprob_error_threshold
        # Match the legacy path: whenever real policy logprobs are available,
        # report sequence-level generation/training mismatch. A threshold adds
        # masking; leaving it unset keeps this metrics-only.
        if self._policy_logprobs_required:
            masking_data = BatchedDataDict(
                {
                    "token_mask": token_mask,
                    "sample_mask": sample_mask,
                    "prev_logprobs": tensor_field(
                        data,
                        adv_cfg.policy_logprobs_field,
                    ),
                    "generation_logprobs": tensor_field(
                        data,
                        adv_cfg.generation_logprobs_field,
                    ),
                }
            )
            num_valid_seqs_before = float(
                ((token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(dim=-1) > 0).sum().item()
            )
            seq_error_metrics = compute_and_apply_seq_logprob_error_masking(
                train_data=masking_data,
                rewards=rewards,
                seq_logprob_error_threshold=seq_logprob_error_threshold,
            )
            sample_mask = masking_data["sample_mask"]
            num_valid_seqs_after = float(((token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum(dim=-1) > 0).sum().item())
            seq_error_metrics["num_masked_seqs_by_logprob_error"] = seq_error_metrics.pop("num_masked_seqs")
            seq_error_metrics["_num_valid_seqs_before"] = num_valid_seqs_before
            seq_error_metrics["_num_valid_seqs_after"] = num_valid_seqs_after
            self._step_log_dict["seq_logprob_error_metrics"].append(seq_error_metrics)

        mask = token_mask * sample_mask.unsqueeze(-1)

        kwargs: dict[str, torch.Tensor] = {}
        if self._policy_logprobs_required:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if self._reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        # Training predicts token t from position t - 1, so token_mask[:, 1:]
        # is the exact mask used when global_valid_toks and the loss are built.
        has_valid_training_tokens = bool(mask[:, 1:].bool().any().item())
        if has_valid_training_tokens:
            if self._shared_prefix_training_config.mode != "disabled":
                self._step_shared_prefix_opportunities.append(
                    observe_shared_prefix_opportunity(
                        group_ids=_string_object_field(data, SHARED_PREFIX_GROUP_ID),
                        prompt_token_ids=prompt_ids,
                        prompt_lengths=squeeze_trailing_unit_dim(tensor_field(data, SHARED_PREFIX_PROMPT_LENGTHS)),
                        input_lengths=squeeze_trailing_unit_dim(tensor_field(data, "input_lengths")),
                        token_mask=token_mask,
                        sample_mask=sample_mask,
                        expected_group_size=(self._master_config.grpo.num_generations_per_prompt),
                    )
                )
            advantages = self._advantage_estimator.compute_advantage(
                prompt_ids=prompt_ids,
                rewards=rewards,
                mask=mask,
                repeated_batch=repeated_batch,
                **kwargs,
            )
        else:
            advantages = torch.zeros_like(mask)
        response_advantages = torch.masked_select(advantages, mask.bool())
        # Keep reward metrics on the same streaming cohort that reaches F/B and
        # train-data JSONL capture.  A partially masked chunk still reaches the
        # trainer as one payload (with sample_loss_mask identifying inactive
        # rows), so retain all of its rows.  A fully filtered chunk is consumed
        # without F/B and is not captured; including it here would make
        # train/reward and train/verifier_reward describe a different cohort
        # from the optimizer step.
        if has_valid_training_tokens:
            self._step_log_dict["rewards"].append(rewards.detach().cpu())
            self._step_log_dict.setdefault("verifier_rewards", []).append(
                verifier_rewards.detach().cpu()
            )
        self._step_log_dict["masked_advantages"].append(response_advantages.detach().cpu())

        if getattr(self, "_log_train_data", False) and has_valid_training_tokens:
            self._capture_train_data_chunk(
                meta=meta,
                data=data,
                rewards=rewards,
                verifier_rewards=verifier_rewards,
                token_mask=token_mask,
                sample_mask=sample_mask,
                advantages=advantages,
            )

        fields_to_put = {adv_cfg.output_field: advantages}
        if reward_scaling_enabled or reward_shaping_enabled:
            fields_to_put[adv_cfg.reward_field] = rewards
        if seq_logprob_error_threshold is not None:
            fields_to_put[adv_cfg.sample_mask_field] = sample_mask

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(meta, fields_to_put),
        )
        return (
            meta.with_fields(list(fields_to_put)),
            has_valid_training_tokens,
        )

    def _capture_train_data_chunk(
        self,
        *,
        meta: KVBatchMeta,
        data: TensorDict,
        rewards: torch.Tensor,
        verifier_rewards: torch.Tensor,
        token_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        """Preserve the exact ordered trainer payload before DataPlane clears it."""
        input_ids = tensor_field(data, "input_ids")
        input_lengths = squeeze_trailing_unit_dim(tensor_field(data, "input_lengths"))
        prompt_ids = tensor_field(data, self._advantage_cfg.prompt_ids_field)
        batch_size = int(input_ids.shape[0])
        if len(meta.sample_ids) != batch_size:
            raise RuntimeError(
                "SingleController train-data logging sample-id mismatch: "
                f"{len(meta.sample_ids)} ids for batch size {batch_size}"
            )

        group_ids: list[str] | None = None
        prompt_lengths: torch.Tensor | None = None
        if self._shared_prefix_training_config.mode != "disabled":
            group_ids = _string_object_field(data, SHARED_PREFIX_GROUP_ID)
            prompt_lengths = squeeze_trailing_unit_dim(tensor_field(data, SHARED_PREFIX_PROMPT_LENGTHS))

        policy_logprobs = (
            tensor_field(data, self._advantage_cfg.policy_logprobs_field) if self._policy_logprobs_required else None
        )
        generation_logprobs = (
            tensor_field(data, self._advantage_cfg.generation_logprobs_field)
            if getattr(self, "_log_train_data", False) or self._policy_logprobs_required
            else None
        )
        reference_logprobs = (
            tensor_field(data, self._advantage_cfg.reference_logprobs_field)
            if self._reference_logprobs_required
            else None
        )

        def row_list(tensor: torch.Tensor, row: int) -> Any:
            return tensor[row].detach().cpu().tolist()

        for row in range(batch_size):
            if prompt_lengths is not None:
                prompt_length = int(prompt_lengths[row].detach().cpu().item())
            else:
                active_loss_indices = torch.nonzero(token_mask[row].bool(), as_tuple=False).flatten()
                if not len(active_loss_indices):
                    # A valid zero-completion row has no active loss token. In
                    # that case the complete input is the prompt, so retain the
                    # diagnostic row instead of allowing default-on logging to
                    # abort an otherwise valid optimizer step.
                    prompt_length = int(input_lengths[row].detach().cpu().item())
                else:
                    prompt_length = int(active_loss_indices[0].detach().cpu().item())
            if not 0 < prompt_length <= int(prompt_ids.shape[1]):
                raise RuntimeError(
                    "SingleController train-data logging prompt length is invalid "
                    f"for sample {meta.sample_ids[row]}: {prompt_length}"
                )
            record: dict[str, Any] = {
                "sample_id": str(meta.sample_ids[row]),
                "rewards": float(rewards[row].detach().cpu().item()),
                "verifier_rewards": float(
                    verifier_rewards[row].detach().cpu().item()
                ),
                "input_lengths": int(input_lengths[row].detach().cpu().item()),
                "token_ids": row_list(input_ids, row),
                "prompt_ids": prompt_ids[row, :prompt_length].detach().cpu().tolist(),
                "token_loss_mask": row_list(token_mask, row),
                "sample_loss_mask": float(sample_mask[row].detach().cpu().item()),
                "advantages": row_list(advantages, row),
            }
            if group_ids is not None and prompt_lengths is not None:
                record[SHARED_PREFIX_GROUP_ID] = group_ids[row]
                record[SHARED_PREFIX_PROMPT_LENGTHS] = prompt_length
            if policy_logprobs is not None:
                record["prev_logprobs"] = row_list(policy_logprobs, row)
            if generation_logprobs is not None:
                record["generation_logprobs"] = row_list(generation_logprobs, row)
            if reference_logprobs is not None:
                record["reference_policy_logprobs"] = row_list(reference_logprobs, row)
            self._step_train_data_records.append(record)

    def _flush_train_data_records(self, *, step: int) -> None:
        """Write one complete optimizer-step JSONL using the legacy filename."""
        if not getattr(self, "_log_train_data", False):
            return
        if not self._step_train_data_records:
            raise RuntimeError(
                "SingleController train-data logging produced no rows for a " f"successful optimizer step {step}"
            )

        keys = tuple(self._step_train_data_records[0])
        expected_keys = set(keys)
        for index, record in enumerate(self._step_train_data_records[1:], start=1):
            if set(record) != expected_keys:
                raise RuntimeError(
                    "SingleController train-data logging field mismatch at row "
                    f"{index}: expected {sorted(expected_keys)}, got "
                    f"{sorted(record)}"
                )
        sample_ids = [record.get("sample_id") for record in self._step_train_data_records]
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids) or len(
            set(sample_ids)
        ) != len(sample_ids):
            raise RuntimeError("SingleController train-data logging contains missing or duplicate " "sample IDs")
        columns = {key: [record[key] for record in self._step_train_data_records] for key in keys}
        self._logger.log_batched_dict_as_jsonl(columns, f"train_data_step{step}.jsonl")
        self._step_train_data_records.clear()

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._advantage_cfg
        fields = [
            adv_cfg.prompt_ids_field,
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
        ]
        reward_shaping_cfg = getattr(self._master_config.grpo, "reward_shaping", None)
        if bool(getattr(reward_shaping_cfg, "enabled", False)):
            fields.extend(["truncated", "response_token_lengths"])
        if self._shared_prefix_training_config.mode != "disabled":
            fields.extend(
                [
                    "input_lengths",
                    SHARED_PREFIX_GROUP_ID,
                    SHARED_PREFIX_PROMPT_LENGTHS,
                ]
            )
        if getattr(self, "_log_train_data", False):
            fields.extend(["input_ids", "input_lengths"])
        if self._policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
        if getattr(self, "_log_train_data", False) or self._policy_logprobs_required:
            fields.append(adv_cfg.generation_logprobs_field)
        if self._reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
