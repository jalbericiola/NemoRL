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

"""End-to-end test: SC._rollout_pump writes the expected rows to TQ."""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import ray
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSampler,
    WeightFifoSampler,
    WindowedSampler,
    WindowedSamplerConfig,
)
from nemo_rl.algorithms.grpo import GRPOConfig, _initial_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import RolloutManager, RolloutOutcome

# Reuse fixtures from the experience tests; same shape as test_async_rollout_manager.
from tests.unit.experience.test_rollout_manager import (
    single_multi_step_calculator_input_sample,  # noqa: F401
)
from tests.unit.experience.test_rollouts import (
    initial_multi_step_calculator_batch,  # noqa: F401
    multi_step_calculator_environment,  # noqa: F401
    multi_step_setup_vllm_async,  # noqa: F401
    rollout_cluster,  # noqa: F401
    rollout_tokenizer,  # noqa: F401
)
from tests.unit.single_controller._dp_fakes import (
    _BULK_FIELDS,
    _PARTITION_ID,
    _SyncDPAdapter,
    _TQActor,
)


def _failure_cfg(
    *,
    on_dropped_prompt: str = "shrink",
    max_replacement_attempts: int = 1,
    replacement_reserve_prompts: int = 1,
    min_step_batch_fraction: float = 0.9,
) -> SimpleNamespace:
    """The rollout_failure block the pump reads, defaulted to today's behaviour."""
    return SimpleNamespace(
        on_dropped_prompt=on_dropped_prompt,
        max_replacement_attempts=max_replacement_attempts,
        replacement_reserve_prompts=replacement_reserve_prompts,
        min_step_batch_fraction=min_step_batch_fraction,
    )


def _init_pump_ledgers(ctrl: Any, *, current_step: int = 0) -> None:
    """Set the per-step bookkeeping the pump reads unconditionally.

    Empty is the neutral state, so tests that do not care about these can call this
    and forget them. They are collected here because every hand-built controller in
    this file needs all of them, and a field added to only some of the builders is a
    missing attribute rather than a wrong answer.
    """
    ctrl._batch_shortfall = {}
    ctrl._batch_replacements = {}
    ctrl._batch_promotions = {}
    ctrl._replacement_reserve = deque()
    ctrl._train_steps = current_step
    ctrl._next_rollout_admission = current_step
    ctrl._pending_rollout_admissions = deque()
    ctrl._rollout_admission_state_restored = False
    ctrl._legacy_untracked_replay_group_ids = set()
    ctrl._rollout_admission_lock = asyncio.Lock()
    if not hasattr(ctrl, "_buffer"):
        ctrl._buffer = _RecordingBuffer()


class _RecordingBuffer:
    """TQReplayBuffer stand-in recording the target_step of each reserve.

    Seeded entries stand for groups whose rollout has already finished; anything the
    pump reserves during the test is in flight and so unready, as in the real buffer.
    """

    def __init__(
        self,
        target_step_list: list[int | None] | None = None,
        ready_list: list[bool] | None = None,
        group_ids: list[str] | None = None,
        start_weight_list: list[int] | None = None,
    ) -> None:
        self.target_step_list: list[int | None] = list(target_step_list or [])
        self.ready_list: list[bool] = (
            list(ready_list)
            if ready_list is not None
            else [True] * len(self.target_step_list)
        )
        self._group_ids = (
            list(group_ids)
            if group_ids is not None
            else [f"restored-{index}" for index in range(len(self.target_step_list))]
        )
        self.start_weight_list = (
            list(start_weight_list)
            if start_weight_list is not None
            else [0] * len(self.target_step_list)
        )

    def reserve(
        self, *, target_step: int | None, start_weight: int = 0
    ) -> None:
        self.target_step_list.append(target_step)
        self.ready_list.append(False)
        self._group_ids.append(f"new-{len(self._group_ids)}")
        self.start_weight_list.append(start_weight)

    async def remove(self, idxs, *, remove_in_dp: bool) -> int:
        del remove_in_dp
        for index in sorted(idxs, reverse=True):
            del self.target_step_list[index]
            del self.ready_list[index]
            del self._group_ids[index]
            del self.start_weight_list[index]
        return len(idxs)

    def count_for_target_step(self, target_step: int) -> int:
        return sum(1 for target in self.target_step_list if target == target_step)

    # The real thing, so the pump tests exercise the promotion the controller ships
    # rather than a second implementation of it that could drift.
    promote_ready_group = TQReplayBuffer.promote_ready_group


class _RecordingRolloutManager:
    def __init__(
        self, buffer: _RecordingBuffer, *, start_weight: int = 0
    ) -> None:
        self._buffer = buffer
        self._start_weight = start_weight
        self.prompts_seen: list[str] = []

    async def generate_and_push(
        self,
        prompt: Any,
        *,
        target_step: int | None = None,
        inflight_registry: dict[str, tuple[asyncio.Task[None], int]] | None = None,
    ) -> None:
        del inflight_registry
        self.prompts_seen.append(prompt["message_log"][0]["content"])
        self._buffer.reserve(
            target_step=target_step, start_weight=self._start_weight
        )


@pytest.mark.parametrize(
    ("make_sampler", "expected_target_steps"),
    [
        # weight_fifo gates but does not stamp target_step.
        (lambda buf: WeightFifoSampler(buf, max_staleness_versions=1), [None, None]),
        # in_order stamps the dispatch index as target_step.
        (lambda buf: InOrderSampler(buf, max_lookahead_versions=1), [0, 1]),
    ],
)
def test_rollout_pump_stamps_target_steps(
    make_sampler,
    expected_target_steps: list[int | None],
) -> None:
    buffer = _RecordingBuffer()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._buffer = buffer
    ctrl._async_cfg = SimpleNamespace(
        max_inflight_prompts=2,
        diagnostics=False,
        rollout_failure=_failure_cfg(),
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_epochs=1, max_num_steps=2)
    )
    ctrl._rollout_manager = _RecordingRolloutManager(buffer)
    # The sampler owns admission + target_step stamping (the dispatch counter
    # lives on the sampler, not the actor).
    ctrl._sampler = make_sampler(buffer)
    prompt_batch = BatchedDataDict(
        {"message_log": [[{"role": "user", "content": "prompt"}]]}
    )
    ctrl._dataloader = [prompt_batch, prompt_batch]
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._inflight_rollouts = 0
    ctrl._inflight_by_group_id = {}
    ctrl._dispatched_rollouts = set()
    ctrl._trainer_version = 0
    ctrl._current_epoch = 0
    _init_pump_ledgers(ctrl)

    asyncio.run(ctrl._rollout_pump())

    assert buffer.target_step_list == expected_target_steps
    assert ctrl._rollout_exhausted.is_set()


@pytest.mark.parametrize(
    ("outcome", "expect_permit_released"),
    [
        # A committed group transfers permit ownership to the train pump, which
        # releases it after consuming the group.
        (RolloutOutcome.COMMITTED, False),
        # A skipped prompt never reaches the buffer, so the train pump will never
        # see it and the dispatcher must release the permit itself. Getting this
        # wrong leaks one backpressure slot per skipped prompt until the pump wedges.
        (RolloutOutcome.SKIPPED, True),
    ],
)
def test_rollout_pump_releases_capacity_only_for_uncommitted_prompts(
    outcome: RolloutOutcome, expect_permit_released: bool
) -> None:
    class _OutcomeRolloutManager:
        async def generate_and_push(
            self,
            prompt: Any,
            *,
            target_step: int | None = None,
            inflight_registry: dict[str, Any] | None = None,
        ) -> RolloutOutcome:
            del prompt, target_step, inflight_registry
            return outcome

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = SimpleNamespace(
        max_inflight_prompts=2, diagnostics=False, rollout_failure=_failure_cfg()
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_epochs=1)
    )
    ctrl._rollout_manager = _OutcomeRolloutManager()
    ctrl._sampler = WindowedSampler(None, max_staleness_versions=1)
    ctrl._dataloader = [
        BatchedDataDict({"message_log": [[{"role": "user", "content": "prompt"}]]})
    ]
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._inflight_rollouts = 0
    ctrl._dispatched_rollouts = set()
    ctrl._trainer_version = 0
    ctrl._current_epoch = 0
    ctrl._inflight_by_group_id = {}
    _init_pump_ledgers(ctrl)

    asyncio.run(ctrl._rollout_pump())

    # One prompt was dispatched out of a semaphore sized 2.
    expected = 2 if expect_permit_released else 1
    assert ctrl._buffer_capacity._value == expected
    assert ctrl._inflight_rollouts == 0


@pytest.mark.parametrize(
    ("restored", "expected_new_dispatches"),
    [
        # Room left for a partial top-up.
        (1, 1),
        # Target step already full: the whole batch is dropped.
        (2, 0),
        # More restored than a batch: still zero, never negative.
        (3, 0),
    ],
)
def test_rollout_pump_tops_up_restored_target_step(
    restored: int,
    expected_new_dispatches: int,
) -> None:
    # On resume the buffer holds groups still stamped for the next target
    # step. In-order selection consumes a target step as one fixed-size batch,
    # so the pump must dispatch only the shortfall — a full batch on top would
    # leave surplus groups that are never selected and whose capacity permits
    # are held until evict.
    buffer = _RecordingBuffer([0] * restored)
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._buffer = buffer
    ctrl._async_cfg = SimpleNamespace(
        max_inflight_prompts=2, diagnostics=False, rollout_failure=_failure_cfg()
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_epochs=1)
    )
    ctrl._rollout_manager = _RecordingRolloutManager(buffer)
    # lookahead=0 keeps the single batch on target_step 0.
    ctrl._sampler = InOrderSampler(buffer, max_lookahead_versions=0)
    ctrl._dataloader = [
        BatchedDataDict(
            {
                "message_log": [
                    [{"role": "user", "content": "p0"}],
                    [{"role": "user", "content": "p1"}],
                ]
            }
        )
    ]
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_capacity = asyncio.Semaphore(4)
    ctrl._inflight_rollouts = 0
    ctrl._inflight_by_group_id = {}
    ctrl._dispatched_rollouts = set()
    ctrl._trainer_version = 0
    ctrl._current_epoch = 0
    _init_pump_ledgers(ctrl)

    asyncio.run(ctrl._rollout_pump())

    # Only the shortfall was dispatched on top of the restored groups.
    assert buffer.target_step_list == [0] * (restored + expected_new_dispatches)
    # A dispatched prompt keeps its permit (the train pump releases it after
    # consuming the group), so exactly one permit per dispatch is held and the
    # dropped prompts consume none.
    assert ctrl._buffer_capacity._value == 4 - expected_new_dispatches
    assert ctrl._inflight_rollouts == 0
    assert ctrl._rollout_exhausted.is_set()


class _SkippingRolloutManager:
    """Every prompt is given up on within budget, so nothing is ever committed."""

    async def generate_and_push(
        self,
        prompt: Any,
        *,
        target_step: int | None = None,
        inflight_registry: dict[str, Any] | None = None,
    ) -> RolloutOutcome:
        del prompt, target_step, inflight_registry
        return RolloutOutcome.SKIPPED


@pytest.mark.parametrize(
    ("make_sampler", "expected_shortfall"),
    [
        # in_order matches a batch to one step exactly, so a group that is never
        # generated leaves that step permanently short. Without the credit the train
        # pump waits on it forever, turning a tolerated drop into a hung run.
        (lambda buf: InOrderSampler(buf, max_lookahead_versions=1), {0: 1, 1: 1}),
        # weight_fifo does not stamp a step: the batch fills from whatever is ready, so
        # a drop costs throughput and strands nothing. Crediting here would shrink an
        # unrelated step.
        (lambda buf: WeightFifoSampler(buf, max_staleness_versions=1), {}),
    ],
)
def test_rollout_pump_credits_shortfall_only_for_stamped_prompts(
    make_sampler,
    expected_shortfall: dict[int, int],
) -> None:
    buffer = _RecordingBuffer()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._buffer = buffer
    ctrl._async_cfg = SimpleNamespace(
        max_inflight_prompts=2, diagnostics=False, rollout_failure=_failure_cfg()
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(max_num_epochs=1)
    )
    ctrl._rollout_manager = _SkippingRolloutManager()
    ctrl._sampler = make_sampler(buffer)
    prompt_batch = BatchedDataDict(
        {"message_log": [[{"role": "user", "content": "prompt"}]]}
    )
    ctrl._dataloader = [prompt_batch, prompt_batch]
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._inflight_rollouts = 0
    ctrl._inflight_by_group_id = {}
    ctrl._dispatched_rollouts = set()
    ctrl._trainer_version = 0
    ctrl._current_epoch = 0
    _init_pump_ledgers(ctrl)

    asyncio.run(ctrl._rollout_pump())

    assert ctrl._batch_shortfall == expected_shortfall
    # A dropped prompt commits nothing, so every permit comes back either way.
    assert ctrl._buffer_capacity._value == 2


class _ScriptedRolloutManager:
    """Returns a scripted outcome per call, recording which prompt each one saw."""

    def __init__(self, outcomes: list[RolloutOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.prompts_seen: list[str] = []
        self.target_steps_seen: list[int | None] = []

    async def generate_and_push(
        self,
        prompt: Any,
        *,
        target_step: int | None = None,
        inflight_registry: dict[str, Any] | None = None,
    ) -> RolloutOutcome:
        del inflight_registry
        self.prompts_seen.append(prompt["message_log"][0]["content"])
        self.target_steps_seen.append(target_step)
        if self._outcomes:
            return self._outcomes.pop(0)
        return RolloutOutcome.COMMITTED


def _batch(*contents: str) -> BatchedDataDict:
    """A dataloader batch holding one prompt per content string."""
    return BatchedDataDict(
        {"message_log": [[{"role": "user", "content": c}] for c in contents]}
    )


class _CursorDataloader:
    """Minimal stateful cursor matching the restart behavior under test."""

    def __init__(self, batches: list[BatchedDataDict], *, cursor: int = 0) -> None:
        self._batches = batches
        self.cursor = cursor

    def __iter__(self):
        while self.cursor < len(self._batches):
            batch = self._batches[self.cursor]
            self.cursor += 1
            yield batch

    def __len__(self) -> int:
        return len(self._batches)

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}


def _pending_admission(
    dispatch_index: int,
    target_step: int | None,
    *contents: str,
    group_ids: list[str | None] | None = None,
) -> dict[str, Any]:
    if group_ids is None:
        group_ids = [None] * len(contents)
    return {
        "dispatch_index": dispatch_index,
        "admitted": True,
        "source": "dataloader",
        "groups": [
            {
                "target_step": target_step,
                "group_id": group_id,
                "prompt": {
                    "message_log": [{"role": "user", "content": content}]
                },
            }
            for content, group_id in zip(contents, group_ids, strict=True)
        ],
    }


def _pending_prompts(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [group["prompt"] for group in entry["groups"]]


def _pump_controller(
    manager: Any,
    dataloader: list[BatchedDataDict],
    *,
    on_dropped_prompt: str = "replace",
    max_replacement_attempts: int = 1,
    num_prompts_per_step: int = 1,
    buffer: _RecordingBuffer | None = None,
    trainer_version: int = 0,
    current_step: int = 0,
    max_num_steps: int | None = None,
):
    buffer = _RecordingBuffer() if buffer is None else buffer
    if max_num_steps is None:
        max_num_steps = len(dataloader)
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._buffer = buffer
    ctrl._async_cfg = SimpleNamespace(
        max_inflight_prompts=2,
        diagnostics=False,
        sampler=SimpleNamespace(name="in_order"),
        rollout_failure=_failure_cfg(
            on_dropped_prompt=on_dropped_prompt,
            max_replacement_attempts=max_replacement_attempts,
        ),
    )
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            max_num_epochs=1,
            max_num_steps=max_num_steps,
            num_prompts_per_step=num_prompts_per_step,
        )
    )
    ctrl._rollout_manager = manager
    ctrl._sampler = InOrderSampler(buffer, max_lookahead_versions=1)
    ctrl._sampler.set_dispatch_index(current_step)
    ctrl._dataloader = dataloader
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._buffer_capacity = asyncio.Semaphore(4)
    ctrl._inflight_rollouts = 0
    ctrl._inflight_by_group_id = {}
    ctrl._dispatched_rollouts = set()
    ctrl._trainer_version = trainer_version
    ctrl._current_epoch = 0
    _init_pump_ledgers(ctrl, current_step=current_step)
    ctrl._sampler_stamps_target_steps = False
    return ctrl


class TestRolloutAdmissionBudget:
    """Rollout lookahead must stop where the optimizer-step budget stops."""

    def test_one_step_run_keeps_spare_but_never_admits_lookahead_row(self):
        """Regression for job 6695984's surplus target_step=1 K=4 cohort."""
        manager = _ScriptedRolloutManager([])
        ctrl = _pump_controller(
            manager,
            [_batch("target0"), _batch("would-be-spare"), _batch("target1")],
            max_num_steps=1,
        )

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["target0"]
        assert manager.target_steps_seen == [0]
        assert len(ctrl._replacement_reserve) == 1
        assert ctrl._replacement_reserve[0]["message_log"][0]["content"] == (
            "would-be-spare"
        )

    def test_two_step_cap_keeps_unused_reserve_but_does_not_drain_it(self):
        manager = _ScriptedRolloutManager([])
        ctrl = _pump_controller(
            manager,
            [
                _batch("target0"),
                _batch("spare"),
                _batch("target1"),
                _batch("forbidden-target2"),
            ],
            max_num_steps=2,
        )

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["target0", "target1"]
        assert manager.target_steps_seen == [0, 1]
        assert len(ctrl._replacement_reserve) == 1

    def test_unstamped_sampler_obeys_the_same_admission_cap(self):
        buffer = _RecordingBuffer()
        manager = _ScriptedRolloutManager([])
        ctrl = _pump_controller(
            manager,
            [_batch("target0"), _batch("forbidden-target1")],
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["target0"]
        assert manager.target_steps_seen == [None]
        assert len(ctrl._replacement_reserve) == 0

    def test_resume_uses_the_absolute_remaining_step_budget(self):
        manager = _ScriptedRolloutManager([])
        ctrl = _pump_controller(
            manager,
            [_batch("target5"), _batch("forbidden-target6")],
            trainer_version=5,
            current_step=5,
            max_num_steps=6,
        )

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["target5"]
        assert manager.target_steps_seen == [5]
        assert len(ctrl._replacement_reserve) == 1

    def test_final_epoch_resume_requeues_only_the_missing_stamped_groups(self):
        """A final-epoch checkpoint omits unready slots but loses no prompts.

        This is the restart shape the old ``current_epoch`` counter could not
        represent: the dataloader cursor is already at EOF, epoch equals its finite
        maximum, target 5 is fully durable, and only one of target 6's two admitted
        prompts reached replay before the checkpoint.
        """
        buffer = _RecordingBuffer(
            [5, 5, 6], group_ids=["p5a-id", "p5b-id", "p6a-id"]
        )
        dataloader = _CursorDataloader(
            [_batch("p5a", "p5b"), _batch("p6a", "p6b")], cursor=2
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=5,
            current_step=5,
            max_num_steps=7,
        )
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 7
        ctrl._pending_rollout_admissions = deque(
            [
                _pending_admission(
                    5,
                    5,
                    "p5a",
                    "p5b",
                    group_ids=["p5a-id", "p5b-id"],
                ),
                _pending_admission(
                    6,
                    6,
                    "p6a",
                    "p6b",
                    group_ids=["p6a-id", None],
                ),
            ]
        )
        ctrl._rollout_admission_state_restored = True
        ctrl._sampler.set_dispatch_index(7)
        ctrl._sampler_stamps_target_steps = True

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [5, 5, 6, 6]
        assert manager.prompts_seen == ["p6b"]
        assert ctrl._next_rollout_admission == 7
        assert ctrl._current_epoch == 1
        assert dataloader.state_dict() == {"cursor": 2}

    def test_full_restored_next_step_never_duplicates_from_the_reserve(self):
        buffer = _RecordingBuffer([5, 5], group_ids=["p5a-id", "p5b-id"])
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=5,
            current_step=5,
            max_num_steps=6,
        )
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 6
        ctrl._pending_rollout_admissions = deque(
            [
                _pending_admission(
                    5,
                    5,
                    "p5a",
                    "p5b",
                    group_ids=["p5a-id", "p5b-id"],
                )
            ]
        )
        ctrl._replacement_reserve.extend(
            _pending_prompts(_pending_admission(6, None, "spare-a", "spare-b"))
        )
        ctrl._sampler.set_dispatch_index(6)
        ctrl._sampler_stamps_target_steps = True
        ctrl._rollout_admission_state_restored = True

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [5, 5]
        assert len(ctrl._replacement_reserve) == 2

    def test_legacy_reserve_drain_skips_a_full_restored_cohort(self):
        """Defense in depth when an old checkpoint has no admission ledger."""
        buffer = _RecordingBuffer([0, 0])
        ctrl = _pump_controller(
            _RecordingRolloutManager(buffer),
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._current_epoch = 1
        ctrl._replacement_reserve.extend(
            _pending_prompts(_pending_admission(0, None, "spare-a", "spare-b"))
        )
        ctrl._sampler_stamps_target_steps = True
        ctrl._finalize_rollout_admission_restore()

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [0, 0]
        assert len(ctrl._replacement_reserve) == 2
        assert ctrl._next_rollout_admission == 1

    def test_legacy_reserve_drain_tops_up_partial_cohort_without_losing_spare(self):
        """Exercise the real legacy finalizer before filling the final target."""
        buffer = _RecordingBuffer([0])
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._current_epoch = 1
        ctrl._replacement_reserve.extend(
            _pending_prompts(_pending_admission(0, None, "spare-a", "spare-b"))
        )
        ctrl._sampler_stamps_target_steps = True
        ctrl._finalize_rollout_admission_restore()

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["spare-a"]
        assert buffer.target_step_list == [0, 0]
        assert [
            prompt["message_log"][0]["content"]
            for prompt in ctrl._replacement_reserve
        ] == ["spare-b"]

    def test_legacy_partial_then_full_lookahead_does_not_consume_fresh_batch(self):
        """A full later cohort advances admission without eating dataset prompts."""
        buffer = _RecordingBuffer([5, 6, 6])
        dataloader = _CursorDataloader(
            [_batch("fresh7a", "fresh7b"), _batch("fresh8a", "fresh8b")]
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=5,
            current_step=5,
            max_num_steps=7,
        )
        ctrl._finalize_rollout_admission_restore()

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["fresh7a"]
        assert buffer.count_for_target_step(5) == 2
        assert buffer.count_for_target_step(6) == 2
        assert dataloader.state_dict() == {"cursor": 1}
        assert ctrl._next_rollout_admission == 7
        assert [
            prompt["message_log"][0]["content"]
            for prompt in ctrl._replacement_reserve
        ] == ["fresh7b"]

    def test_final_epoch_legacy_reserve_topup_advances_later_full_cohort(self):
        """The final saved cursor covers a full lookahead exposed by a pool top-up."""
        buffer = _RecordingBuffer([5, 6, 6])
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=5,
            current_step=5,
            max_num_steps=7,
        )
        ctrl._current_epoch = 1
        ctrl._replacement_reserve.extend(
            _pending_prompts(_pending_admission(5, None, "spare-a", "spare-b"))
        )
        ctrl._finalize_rollout_admission_restore()

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["spare-a"]
        assert buffer.count_for_target_step(5) == 2
        assert buffer.count_for_target_step(6) == 2
        assert ctrl._next_rollout_admission == 7
        assert [
            prompt["message_log"][0]["content"]
            for prompt in ctrl._replacement_reserve
        ] == ["spare-b"]

    def test_unstamped_resume_credits_restored_groups_before_requeue(self):
        """Unstamped replay gets one global credit instead of K duplicates/batch."""
        buffer = _RecordingBuffer(
            [None, None, None], group_ids=["u5a-id", "u5b-id", "u6b-id"]
        )
        dataloader = _CursorDataloader(
            [_batch("u5a", "u5b"), _batch("u6a", "u6b")], cursor=2
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=5,
            current_step=5,
            max_num_steps=7,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._sampler.set_dispatch_index(7)
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 7
        ctrl._pending_rollout_admissions = deque(
            [
                _pending_admission(
                    5,
                    None,
                    "u5a",
                    "u5b",
                    group_ids=["u5a-id", "u5b-id"],
                ),
                _pending_admission(
                    6,
                    None,
                    "u6a",
                    "u6b",
                    group_ids=[None, "u6b-id"],
                ),
            ]
        )
        ctrl._rollout_admission_state_restored = True

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [None, None, None, None]
        assert manager.prompts_seen == ["u6a"]
        assert ctrl._next_rollout_admission == 7
        assert ctrl._current_epoch == 1
        assert dataloader.state_dict() == {"cursor": 2}

    def test_true_legacy_unstamped_partial_cohort_regenerates_full_frontier(self):
        """Unknown legacy selectability cannot spend an absolute admission."""
        buffer = _RecordingBuffer([None], group_ids=["legacy-ready"])
        dataloader = _CursorDataloader(
            [
                _batch("already-consumed-a", "already-consumed-b"),
                _batch("fresh-a", "fresh-b"),
            ],
            cursor=1,
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._finalize_rollout_admission_restore()

        assert ctrl._next_rollout_admission == 0

        async def resume():
            await ctrl._discard_legacy_unstamped_replay()
            await ctrl._rollout_pump()

        asyncio.run(resume())

        assert manager.prompts_seen == ["fresh-a", "fresh-b"]
        assert buffer.target_step_list == [None, None]
        assert ctrl._next_rollout_admission == 1
        assert dataloader.state_dict() == {"cursor": 2}
        assert not ctrl._replacement_reserve

    def test_second_resume_discards_unstamped_legacy_carry_owned_before_admit(self):
        """A v2 checkpoint at the gate rewinds carry but retains owned prompts."""
        buffer = _RecordingBuffer([None], group_ids=["legacy-ready"])
        dataloader = _CursorDataloader(
            [
                _batch("already-consumed-a", "already-consumed-b"),
                _batch("fresh-a", "fresh-b"),
            ],
            cursor=2,
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._sampler.set_dispatch_index(0)
        ctrl._current_epoch = 1
        ctrl._rollout_admission_state_restored = True
        ctrl._legacy_untracked_replay_group_ids = {"legacy-ready"}
        unadmitted = _pending_admission(0, None, "fresh-a", "fresh-b")
        unadmitted["dispatch_index"] = None
        unadmitted["admitted"] = False
        ctrl._pending_rollout_admissions.append(unadmitted)

        async def resume():
            await ctrl._discard_legacy_unstamped_replay()
            await ctrl._rollout_pump()

        asyncio.run(resume())

        assert manager.prompts_seen == ["fresh-a", "fresh-b"]
        assert buffer.target_step_list == [None, None]
        assert ctrl._next_rollout_admission == 1
        assert dataloader.state_dict() == {"cursor": 2}
        assert not ctrl._replacement_reserve

    def test_v2_weight_fifo_requeues_ready_older_sibling_at_current_weight(self):
        """Exact identity cannot leave one cohort split across strict FIFO buckets."""
        buffer = _RecordingBuffer(
            [None],
            group_ids=["ready-old"],
            start_weight_list=[0],
        )
        manager = _RecordingRolloutManager(buffer, start_weight=1)
        ctrl = _pump_controller(
            manager,
            [],
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=1,
            current_step=1,
            max_num_steps=2,
        )
        ctrl._async_cfg.sampler = SimpleNamespace(name="weight_fifo")
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._sampler.set_dispatch_index(2)
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 2
        ctrl._pending_rollout_admissions.append(
            _pending_admission(
                1,
                None,
                "ready-prompt",
                "missing-sibling",
                group_ids=["ready-old", None],
            )
        )
        ctrl._rollout_admission_state_restored = True
        ctrl._restored_replay_group_count = 1
        # One restored row already owns one of the four capacity permits.
        ctrl._buffer_capacity = asyncio.Semaphore(3)

        async def resume():
            await ctrl._discard_legacy_unstamped_replay()
            assert buffer._group_ids == []
            assert ctrl._buffer_capacity._value == 4
            await ctrl._rollout_pump()

        asyncio.run(resume())

        assert manager.prompts_seen == ["ready-prompt", "missing-sibling"]
        assert buffer.start_weight_list == [1, 1]
        assert buffer.target_step_list == [None, None]
        assert ctrl._next_rollout_admission == 2

    def test_v2_sanitizer_rejects_old_group_absent_from_paired_ledger(self):
        """Cleanup must not hide a corrupt replay/admission snapshot pair."""
        buffer = _RecordingBuffer(
            [None],
            group_ids=["orphan-old"],
            start_weight_list=[0],
        )
        ctrl = _pump_controller(
            _RecordingRolloutManager(buffer, start_weight=1),
            [],
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=1,
            current_step=1,
            max_num_steps=2,
        )
        ctrl._async_cfg.sampler = SimpleNamespace(name="weight_fifo")
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._rollout_admission_state_restored = True
        ctrl._pending_rollout_admissions.append(
            _pending_admission(
                1,
                None,
                "different-prompt",
                "missing-sibling",
                group_ids=["different-id", None],
            )
        )

        with pytest.raises(
            RuntimeError,
            match="absent from the paired admission ledger",
        ):
            asyncio.run(ctrl._discard_legacy_unstamped_replay())

        assert buffer._group_ids == ["orphan-old"]

    def test_true_legacy_unstamped_reserve_regenerates_full_frontier(self):
        """EOF reserve drain also refuses unsafe pre-ledger count credit."""
        buffer = _RecordingBuffer([None], group_ids=["legacy-ready"])
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            max_num_steps=1,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)
        ctrl._current_epoch = 1
        ctrl._replacement_reserve.extend(
            _pending_prompts(_pending_admission(0, None, "spare-a", "spare-b"))
        )
        ctrl._finalize_rollout_admission_restore()

        async def resume():
            await ctrl._discard_legacy_unstamped_replay()
            await ctrl._rollout_pump()

        asyncio.run(resume())

        assert manager.prompts_seen == ["spare-a", "spare-b"]
        assert buffer.target_step_list == [None, None]
        assert ctrl._next_rollout_admission == 1
        assert not ctrl._replacement_reserve

    def test_true_legacy_stale_unstamped_full_cohort_spends_no_budget(self):
        """A full but stale Windowed cohort cannot make two future steps short."""
        buffer = _RecordingBuffer(
            [None, None], group_ids=["stale-a", "stale-b"]
        )
        dataloader = _CursorDataloader(
            [
                _batch("already-consumed-a", "already-consumed-b"),
                _batch("step-1-a", "step-1-b"),
                _batch("step-2-a", "step-2-b"),
            ],
            cursor=1,
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=1,
            current_step=1,
            max_num_steps=3,
        )
        ctrl._sampler = WindowedSampler(
            buffer, max_staleness_versions=0
        )
        ctrl._finalize_rollout_admission_restore()

        assert ctrl._next_rollout_admission == 1

        async def resume():
            await ctrl._discard_legacy_unstamped_replay()
            await ctrl._rollout_pump()

        asyncio.run(resume())

        assert manager.prompts_seen == [
            "step-1-a",
            "step-1-b",
            "step-2-a",
            "step-2-b",
        ]
        assert ctrl._next_rollout_admission == 3
        assert dataloader.state_dict() == {"cursor": 3}

    def test_true_legacy_stale_stamped_row_does_not_spend_next_step_budget(self):
        """A stale target still identifies InOrder policy, but not an admission."""
        buffer = _RecordingBuffer([0], group_ids=["late-step-0"])
        dataloader = _CursorDataloader(
            [
                _batch("already-consumed-a", "already-consumed-b"),
                _batch("step-1-a", "step-1-b"),
            ],
            cursor=1,
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            dataloader,  # type: ignore[arg-type]
            on_dropped_prompt="shrink",
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=1,
            current_step=1,
            max_num_steps=2,
        )
        ctrl._finalize_rollout_admission_restore()

        assert ctrl._sampler_stamps_target_steps is True
        assert ctrl._next_rollout_admission == 1
        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["step-1-a", "step-1-b"]
        assert buffer.target_step_list == [0, 1, 1]
        assert ctrl._next_rollout_admission == 2
        assert dataloader.state_dict() == {"cursor": 2}

    def test_second_resume_accepts_legacy_groups_carried_through_v2_checkpoint(self):
        """Untrained lookahead from a legacy checkpoint survives a roll-forward."""
        legacy_ids = ["legacy-6a", "legacy-6b"]
        buffer = _RecordingBuffer([6, 6], group_ids=legacy_ids)
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=6,
            current_step=6,
            max_num_steps=7,
        )
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 7
        ctrl._sampler.set_dispatch_index(7)
        ctrl._sampler_stamps_target_steps = True
        ctrl._rollout_admission_state_restored = True
        ctrl._legacy_untracked_replay_group_ids = set(legacy_ids)

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == []
        assert buffer.target_step_list == [6, 6]
        assert sorted(ctrl._legacy_untracked_replay_group_ids) == legacy_ids

    def test_second_resume_advances_full_legacy_carry_after_exact_partial(self):
        """A v2 checkpoint mid-legacy-resume still finishes the old lookahead."""
        legacy_ids = ["legacy-7a", "legacy-7b"]
        buffer = _RecordingBuffer(
            [6, 7, 7],
            group_ids=["exact-6a", *legacy_ids],
        )
        manager = _RecordingRolloutManager(buffer)
        ctrl = _pump_controller(
            manager,
            [],
            num_prompts_per_step=2,
            buffer=buffer,
            trainer_version=6,
            current_step=6,
            max_num_steps=8,
        )
        ctrl._current_epoch = 1
        ctrl._next_rollout_admission = 7
        ctrl._pending_rollout_admissions = deque(
            [
                _pending_admission(
                    6,
                    6,
                    "exact-6a",
                    "missing-6b",
                    group_ids=["exact-6a", None],
                )
            ]
        )
        ctrl._sampler.set_dispatch_index(7)
        ctrl._sampler_stamps_target_steps = True
        ctrl._rollout_admission_state_restored = True
        ctrl._legacy_untracked_replay_group_ids = set(legacy_ids)

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["missing-6b"]
        assert buffer.count_for_target_step(6) == 2
        assert buffer.count_for_target_step(7) == 2
        assert ctrl._next_rollout_admission == 8
        assert sorted(ctrl._legacy_untracked_replay_group_ids) == legacy_ids

    def test_checkpoint_cut_persists_batch_blocked_in_sampler_admit(self):
        """A cursor cannot checkpoint past an unowned batch at a full gate."""

        class _BlockingSampler(InOrderSampler):
            def __init__(self, buffer):
                super().__init__(buffer, max_lookahead_versions=1)
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def admit(self, *, trainer_version_fn):
                self.entered.set()
                await self.release.wait()
                return await super().admit(trainer_version_fn=trainer_version_fn)

        async def exercise():
            buffer = _RecordingBuffer()
            dataloader = _CursorDataloader([_batch("blocked")])
            ctrl = _pump_controller(
                _RecordingRolloutManager(buffer),
                dataloader,  # type: ignore[arg-type]
                buffer=buffer,
                max_num_steps=1,
            )
            sampler = _BlockingSampler(buffer)
            sampler.set_dispatch_index(0)
            ctrl._sampler = sampler

            pump_task = asyncio.create_task(ctrl._rollout_pump())
            await sampler.entered.wait()

            async def checkpoint_cut():
                async with ctrl._rollout_admission_lock:
                    return (
                        dataloader.state_dict(),
                        ctrl._rollout_admission_state_dict(),
                    )

            checkpoint_task = asyncio.create_task(checkpoint_cut())
            dataloader_state, admission_state = await asyncio.wait_for(
                checkpoint_task, timeout=1.0
            )
            sampler.release.set()
            await pump_task
            return dataloader_state, admission_state

        dataloader_state, admission_state = asyncio.run(exercise())

        assert dataloader_state == {"cursor": 1}
        assert admission_state["next_rollout_admission"] == 0
        assert admission_state["pending_admissions"][0]["admitted"] is False
        assert admission_state["pending_admissions"][0]["groups"][0]["prompt"] == {
            "message_log": [{"role": "user", "content": "blocked"}]
        }

    def test_blocked_gate_does_not_deadlock_skipped_prompt_bookkeeping(self):
        """A full lookahead gate must not own the admission checkpoint lock."""

        class _GateObservingSampler(InOrderSampler):
            def __init__(self, buffer):
                super().__init__(buffer, max_lookahead_versions=0)
                self.calls = 0
                self.second_entered = asyncio.Event()

            async def admit(self, *, trainer_version_fn):
                self.calls += 1
                if self.calls == 2:
                    self.second_entered.set()
                return await super().admit(trainer_version_fn=trainer_version_fn)

        class _DelayedSkipManager:
            def __init__(self):
                self.calls = 0
                self.allow_skip = asyncio.Event()

            async def generate_and_push(self, prompt, **kwargs):
                del prompt, kwargs
                self.calls += 1
                if self.calls == 1:
                    await self.allow_skip.wait()
                    return RolloutOutcome.SKIPPED
                return RolloutOutcome.COMMITTED

        async def exercise():
            buffer = _RecordingBuffer()
            manager = _DelayedSkipManager()
            ctrl = _pump_controller(
                manager,
                [_batch("step0"), _batch("step1")],
                on_dropped_prompt="shrink",
                buffer=buffer,
                max_num_steps=2,
            )
            sampler = _GateObservingSampler(buffer)
            sampler.set_dispatch_index(0)
            ctrl._sampler = sampler

            pump_task = asyncio.create_task(ctrl._rollout_pump())
            await sampler.second_entered.wait()
            manager.allow_skip.set()

            async def wait_for_shortfall():
                while ctrl._batch_shortfall != {0: 1}:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_shortfall(), timeout=1.0)
            ctrl._trainer_version = 1
            await asyncio.wait_for(pump_task, timeout=1.0)
            return ctrl

        ctrl = asyncio.run(exercise())

        assert ctrl._batch_shortfall == {0: 1}
        assert len(ctrl._pending_rollout_admissions) == 1
        assert ctrl._pending_rollout_admissions[0]["dispatch_index"] == 1

    def test_reserve_drain_gate_does_not_deadlock_skipped_prompt(self):
        """Reserve ownership is durable without holding the lock across admit."""

        class _GateObservingSampler(InOrderSampler):
            def __init__(self, buffer):
                super().__init__(buffer, max_lookahead_versions=0)
                self.calls = 0
                self.second_entered = asyncio.Event()

            async def admit(self, *, trainer_version_fn):
                self.calls += 1
                if self.calls == 2:
                    self.second_entered.set()
                return await super().admit(trainer_version_fn=trainer_version_fn)

        class _DelayedSkipManager:
            def __init__(self):
                self.calls = 0
                self.allow_skip = asyncio.Event()

            async def generate_and_push(self, prompt, **kwargs):
                del prompt, kwargs
                self.calls += 1
                if self.calls == 1:
                    await self.allow_skip.wait()
                    return RolloutOutcome.SKIPPED
                return RolloutOutcome.COMMITTED

        async def exercise():
            buffer = _RecordingBuffer()
            manager = _DelayedSkipManager()
            ctrl = _pump_controller(
                manager,
                [],
                on_dropped_prompt="shrink",
                buffer=buffer,
                max_num_steps=2,
            )
            ctrl._current_epoch = 1
            ctrl._replacement_reserve.extend(
                _pending_prompts(
                    _pending_admission(0, None, "reserve0", "reserve1")
                )
            )
            sampler = _GateObservingSampler(buffer)
            sampler.set_dispatch_index(0)
            ctrl._sampler = sampler

            pump_task = asyncio.create_task(ctrl._rollout_pump())
            await sampler.second_entered.wait()
            manager.allow_skip.set()

            async def wait_for_shortfall():
                while ctrl._batch_shortfall != {0: 1}:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_shortfall(), timeout=1.0)
            ctrl._trainer_version = 1
            await asyncio.wait_for(pump_task, timeout=1.0)
            return ctrl

        ctrl = asyncio.run(exercise())

        assert ctrl._batch_shortfall == {0: 1}
        assert len(ctrl._pending_rollout_admissions) == 1
        assert ctrl._pending_rollout_admissions[0]["dispatch_index"] == 1


class TestReplaceDroppedPrompt:
    """on_dropped_prompt="replace": hold the batch size by substituting a spare."""

    def test_a_spare_stands_in_so_the_step_keeps_its_batch_size(self):
        # The first batch is admitted as step 0; the second is diverted into the pool
        # while step 0's rollout is still in flight, which is what makes it available.
        manager = _ScriptedRolloutManager([RolloutOutcome.SKIPPED])
        ctrl = _pump_controller(manager, [_batch("step0"), _batch("spare")])

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["step0", "spare"]
        assert ctrl._batch_shortfall == {}, (
            "a group that was replaced must not also shrink its step -- crediting both "
            "would drop a group the step is actually going to receive"
        )
        assert ctrl._batch_replacements == {0: 1}

    def test_a_replacement_that_also_fails_falls_back_to_shrinking(self):
        """Bounded rounds are what guarantee the step still closes."""
        manager = _ScriptedRolloutManager(
            [RolloutOutcome.SKIPPED, RolloutOutcome.SKIPPED]
        )
        ctrl = _pump_controller(
            manager,
            [_batch("step0"), _batch("spare")],
            max_replacement_attempts=1,
        )

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["step0", "spare"], (
            "budget of 1 was not exceeded"
        )
        assert ctrl._batch_shortfall == {0: 1}
        assert ctrl._batch_replacements == {}
        # Nothing committed, so the slot's permit must come back exactly once.
        assert ctrl._buffer_capacity._value == 4

    def test_an_unstamped_sampler_never_diverts_a_batch_into_the_pool(self):
        """No step can be stranded, so a pool would only cost prompts nothing draws on."""
        buffer = _RecordingBuffer()
        manager = _ScriptedRolloutManager([RolloutOutcome.SKIPPED])
        ctrl = _pump_controller(
            manager,
            [_batch("p0"), _batch("p1")],
            buffer=buffer,
        )
        ctrl._sampler = WeightFifoSampler(buffer, max_staleness_versions=1)

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["p0", "p1"], "no substitution attempted"
        assert len(ctrl._replacement_reserve) == 0, "no batch was spent on the pool"
        assert ctrl._batch_shortfall == {}

    @pytest.mark.parametrize("on_dropped_prompt", ["shrink", "replace"])
    def test_a_clean_run_loses_neither_a_prompt_nor_a_step_to_the_pool(
        self, on_dropped_prompt: str
    ):
        """Enabling replace must be free when nothing fails.

        The pool is filled by diverting a batch, so without draining it at the end that
        batch is never trained on -- and since max_num_steps is derived from
        len(dataloader), the run would also finish a step short of its budget.

        Stamps contiguous from 0 are what pin the diverting order. Diverting *after*
        admit would leave step 0 with no group at all, and these stamps would start at
        1: a step the train pump waits on forever.
        """
        buffer = _RecordingBuffer()
        ctrl = _pump_controller(
            _RecordingRolloutManager(buffer),
            [_batch("a"), _batch("b")],
            on_dropped_prompt=on_dropped_prompt,
            buffer=buffer,
        )

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [0, 1], "both batches became training steps"
        assert len(ctrl._replacement_reserve) == 0, "the pool was handed back"

    def test_a_partial_pool_is_not_dispatched_as_a_short_step(self):
        """A step short by construction would trip the floor and fail a finished run."""
        buffer = _RecordingBuffer()
        manager = _ScriptedRolloutManager([RolloutOutcome.SKIPPED])
        ctrl = _pump_controller(
            manager,
            [_batch("p0", "p1"), _batch("s0", "s1")],
            num_prompts_per_step=2,
            buffer=buffer,
        )

        asyncio.run(ctrl._rollout_pump())

        # s0 stood in for the dropped p0, which leaves the pool one short of a step.
        assert manager.prompts_seen == ["p0", "s0", "p1"]
        assert len(ctrl._replacement_reserve) == 1, "s1 is left over, not a short step"


class TestBorrowFromALaterStep:
    """Where a replacement is sent when a later step has finished work to lend.

    Same guarantee as any replacement -- the batch size holds -- but the dropped step
    closes on a group that already exists instead of one the trainer has to wait for.
    Each case seeds a finished group stamped for a later step, which is what a run with
    lookahead has in the buffer by the time a drop is declared.
    """

    @staticmethod
    def _with_lender(
        *,
        lender_step: int | None = 1,
        lender_ready: bool = True,
        trainer_version: int = 0,
        outcomes: list[RolloutOutcome] | None = None,
    ):
        buffer = _RecordingBuffer(
            target_step_list=[] if lender_step is None else [lender_step],
            ready_list=[] if lender_step is None else [lender_ready],
        )
        manager = _ScriptedRolloutManager(
            [RolloutOutcome.SKIPPED] if outcomes is None else outcomes
        )
        ctrl = _pump_controller(
            manager,
            [_batch("step0"), _batch("spare")],
            on_dropped_prompt="replace",
            buffer=buffer,
            trainer_version=trainer_version,
        )
        return ctrl, manager, buffer

    def test_the_lost_step_is_filled_now_and_the_lender_is_repaid_later(self):
        ctrl, manager, buffer = self._with_lender()

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [0], (
            "the finished group was re-stamped, so step 0 is whole immediately"
        )
        assert manager.prompts_seen == ["step0", "spare"]
        assert ctrl._batch_promotions == {0: 1}, "step 0 borrowed one group"
        assert ctrl._batch_replacements == {1: 1}, (
            "the spare repays the lender, not the step that was dropped from"
        )
        assert ctrl._batch_shortfall == {}, "neither step ends up short"

    def test_nothing_to_borrow_degrades_to_replacing_in_place(self):
        # The shape at max_lookahead_versions=0, where the next batch is not dispatched
        # until this step trains, so no later step can ever have finished work.
        ctrl, manager, _ = self._with_lender(lender_step=None)

        asyncio.run(ctrl._rollout_pump())

        assert manager.prompts_seen == ["step0", "spare"]
        assert ctrl._batch_promotions == {}
        assert ctrl._batch_replacements == {0: 1}, "the spare filled step 0 itself"
        assert ctrl._batch_shortfall == {}

    def test_an_in_flight_group_is_not_borrowed(self):
        """Borrowing one would inherit its wait, which is the thing being avoided."""
        ctrl, manager, buffer = self._with_lender(lender_ready=False)

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [1], "the reservation kept its own step"
        assert ctrl._batch_promotions == {}
        assert ctrl._batch_replacements == {0: 1}

    def test_a_step_the_trainer_has_passed_is_not_filled(self):
        """A second drop can land after the first already closed the step short.

        A group re-stamped for a finished step is selectable by nobody and evicted as
        stale, so the borrow would destroy the lender's work for nothing.
        """
        ctrl, manager, buffer = self._with_lender(lender_step=9, trainer_version=5)

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [9], "the lender kept its group"
        assert ctrl._batch_promotions == {}
        assert ctrl._batch_replacements == {0: 1}

    def test_a_borrow_is_never_taken_without_a_spare_to_repay_it(self):
        """An unrepaid loan is the same hole one step later, carried to the last step."""
        buffer = _RecordingBuffer(target_step_list=[1], ready_list=[True])
        manager = _ScriptedRolloutManager([RolloutOutcome.SKIPPED])
        # One batch, so nothing is ever diverted and the pool stays empty.
        ctrl = _pump_controller(
            manager,
            [_batch("step0")],
            on_dropped_prompt="replace",
            buffer=buffer,
        )

        asyncio.run(ctrl._rollout_pump())

        assert buffer.target_step_list == [1], "no group was moved off the lender"
        assert ctrl._batch_promotions == {}
        assert ctrl._batch_shortfall == {0: 1}, "step 0 shrinks instead"


class TestTakeReplacement:
    """Every way a substitution can be unavailable falls back to shrinking."""

    @staticmethod
    def _controller(
        *,
        on_dropped_prompt: str = "replace",
        max_replacement_attempts: int = 1,
        spares: int = 0,
    ):
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._async_cfg = SimpleNamespace(
            rollout_failure=_failure_cfg(
                on_dropped_prompt=on_dropped_prompt,
                max_replacement_attempts=max_replacement_attempts,
            )
        )
        ctrl._replacement_reserve = deque(_batch(f"spare{i}") for i in range(spares))
        return ctrl

    def test_a_spare_is_handed_out_and_leaves_the_pool(self):
        ctrl = self._controller(spares=2)
        assert ctrl._take_replacement(0, 0) is not None
        assert len(ctrl._replacement_reserve) == 1

    def test_shrink_mode_never_substitutes(self):
        ctrl = self._controller(on_dropped_prompt="shrink", spares=2)
        assert ctrl._take_replacement(0, 0) is None
        assert len(ctrl._replacement_reserve) == 2, "pool left untouched"

    def test_an_unstamped_prompt_has_no_short_step_to_repair(self):
        ctrl = self._controller(spares=2)
        assert ctrl._take_replacement(None, 0) is None

    def test_the_per_slot_budget_stops_substituting(self):
        ctrl = self._controller(max_replacement_attempts=2, spares=5)
        assert ctrl._take_replacement(0, 1) is not None
        assert ctrl._take_replacement(0, 2) is None

    def test_an_empty_pool_shrinks_instead_of_waiting(self):
        """At epoch end there is no more data; the step closes short, not never."""
        ctrl = self._controller(spares=0)
        assert ctrl._take_replacement(0, 0) is None


class TestTargetGroupsForStep:
    """How many groups a step waits for, once some are known not to be coming."""

    @staticmethod
    def _controller(
        num_prompts_per_step: int,
        shortfall: dict[int, int],
        fraction: float = 0.9,
    ):
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._master_config = SimpleNamespace(
            grpo=GRPOConfig.model_construct(
                num_prompts_per_step=num_prompts_per_step,
            )
        )
        ctrl._async_cfg = SimpleNamespace(
            rollout_failure=SimpleNamespace(min_step_batch_fraction=fraction)
        )
        ctrl._batch_shortfall = dict(shortfall)
        return ctrl

    def test_an_untouched_step_waits_for_the_configured_batch(self):
        ctrl = self._controller(128, {})
        assert ctrl._target_groups_for_step(3) == 128

    def test_a_dropped_group_shrinks_only_the_step_it_was_stamped_for(self):
        ctrl = self._controller(128, {3: 2})
        assert ctrl._target_groups_for_step(3) == 126
        assert ctrl._target_groups_for_step(4) == 128, "neighbouring steps unaffected"

    def test_a_step_below_the_floor_fails_rather_than_training_a_part_batch(self):
        """The drop budgets are run-scoped and cannot bound one step's shrinkage."""
        # floor = ceil(0.9 * 128) = 116, so 12 drops are allowed and 13 are not.
        ctrl = self._controller(128, {0: 12})
        assert ctrl._target_groups_for_step(0) == 116

        ctrl = self._controller(128, {0: 13})
        with pytest.raises(RuntimeError, match="below the floor of 116"):
            ctrl._target_groups_for_step(0)

    def test_a_small_batch_is_protected_more_strictly(self):
        """Losing 1 of 4 is a 25% batch reduction; the fraction says no."""
        ctrl = self._controller(4, {0: 1})
        with pytest.raises(RuntimeError, match="below the floor of 4"):
            ctrl._target_groups_for_step(0)

    def test_a_step_stripped_of_every_group_never_reaches_an_empty_batch(self):
        """ceil() with a positive fraction keeps the floor >= 1, so this is caught."""
        ctrl = self._controller(4, {0: 4}, fraction=0.01)
        with pytest.raises(RuntimeError, match="leaving 0"):
            ctrl._target_groups_for_step(0)

    def test_a_fraction_of_one_forbids_shrinking_at_all(self):
        ctrl = self._controller(128, {0: 1}, fraction=1.0)
        with pytest.raises(RuntimeError, match="below the floor of 128"):
            ctrl._target_groups_for_step(0)


def test_abort_stale_inflight_cancels_only_out_of_window_rollouts() -> None:
    async def _main() -> None:
        fresh = asyncio.create_task(asyncio.Event().wait())
        stale = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._sampler = WindowedSampler(None, max_staleness_versions=2)
        ctrl._trainer_version = 5
        ctrl._inflight_by_group_id = {"fresh": (fresh, 5), "stale": (stale, 1)}
        ctrl._pending_rollout_admissions = deque()
        ctrl._legacy_untracked_replay_group_ids = set()

        aborted = await ctrl._abort_stale_inflight()

        assert aborted == 1
        assert stale.cancelled()
        assert not fresh.cancelled()

        fresh.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fresh

    asyncio.run(_main())


def test_abort_stale_inflight_aggregates_cleanup_failures() -> None:
    async def _main() -> None:
        async def _boom() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("cleanup boom")

        task = asyncio.create_task(_boom())
        await asyncio.sleep(0)

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._sampler = WindowedSampler(None, max_staleness_versions=0)
        ctrl._trainer_version = 5
        ctrl._inflight_by_group_id = {"g": (task, 0)}
        ctrl._pending_rollout_admissions = deque()
        ctrl._legacy_untracked_replay_group_ids = set()

        with pytest.raises(BaseExceptionGroup) as exc_info:
            await ctrl._abort_stale_inflight()
        assert exc_info.value.subgroup(RuntimeError) is not None

    asyncio.run(_main())


def test_rollout_pump_failure_cancels_sibling_and_releases_capacity() -> None:
    class _FailingRolloutManager:
        def __init__(self) -> None:
            self._started = 0
            self._both_started = asyncio.Event()
            self.sibling_cancelled = False

        async def generate_and_push(
            self,
            prompt: Any,
            *,
            target_step: int | None = None,
            inflight_registry: dict[str, tuple[asyncio.Task[None], int]] | None = None,
        ) -> None:
            del target_step, inflight_registry
            self._started += 1
            if self._started == 2:
                self._both_started.set()
            await self._both_started.wait()

            content = prompt["message_log"][0]["content"]
            if content == "fail":
                raise RuntimeError("injected rollout failure")

            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.sibling_cancelled = True
                raise

    async def _main() -> None:
        manager = _FailingRolloutManager()
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._async_cfg = SimpleNamespace(
            max_inflight_prompts=2,
            diagnostics=False,
            rollout_failure=_failure_cfg(),
        )
        ctrl._master_config = SimpleNamespace(
            grpo=GRPOConfig.model_construct(max_num_epochs=1)
        )
        ctrl._rollout_manager = manager
        # Over-sampled windowed policy: admit never gates (buffer unused here).
        ctrl._sampler = WindowedSampler(None, max_staleness_versions=1)
        ctrl._dataloader = [
            BatchedDataDict(
                {
                    "message_log": [
                        [{"role": "user", "content": "fail"}],
                        [{"role": "user", "content": "sibling"}],
                    ]
                }
            )
        ]
        ctrl._rollout_permitted = asyncio.Event()
        ctrl._rollout_permitted.set()
        ctrl._rollout_exhausted = asyncio.Event()
        ctrl._buffer_capacity = asyncio.Semaphore(2)
        ctrl._inflight_rollouts = 0
        ctrl._inflight_by_group_id = {}
        ctrl._dispatched_rollouts = set()
        ctrl._trainer_version = 0
        ctrl._current_epoch = 0
        _init_pump_ledgers(ctrl)

        with pytest.raises(ExceptionGroup) as exc_info:
            await asyncio.wait_for(ctrl._rollout_pump(), timeout=1.0)

        assert exc_info.value.subgroup(RuntimeError) is not None
        assert manager.sibling_cancelled
        assert ctrl._inflight_rollouts == 0
        assert ctrl._buffer_capacity._value == 2
        assert ctrl._dispatched_rollouts == set()
        assert not ctrl._rollout_exhausted.is_set()

    asyncio.run(_main())


def test_rollout_pump_releases_permits_when_child_never_starts(monkeypatch) -> None:
    class _NeverCalledRolloutManager:
        async def generate_and_push(
            self,
            prompt: Any,
            *,
            target_step: int | None = None,
            inflight_registry: dict[str, tuple[asyncio.Task[None], int]] | None = None,
        ) -> None:
            del prompt, target_step, inflight_registry
            raise AssertionError("cancelled child unexpectedly started")

    class _CancelBeforeStartTaskGroup:
        def __init__(self) -> None:
            self._tasks: list[asyncio.Task[None]] = []

        async def __aenter__(self) -> _CancelBeforeStartTaskGroup:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            return False

        def create_task(self, coro: Any) -> asyncio.Task[None]:
            task = asyncio.create_task(coro)
            task.cancel()
            self._tasks.append(task)
            return task

    real_semaphore = asyncio.Semaphore
    created_semaphores: list[asyncio.Semaphore] = []

    def _recording_semaphore(value: int) -> asyncio.Semaphore:
        semaphore = real_semaphore(value)
        created_semaphores.append(semaphore)
        return semaphore

    monkeypatch.setattr(asyncio, "Semaphore", _recording_semaphore)
    monkeypatch.setattr(asyncio, "TaskGroup", _CancelBeforeStartTaskGroup)

    async def _main() -> None:
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        ctrl = object.__new__(controller_cls)
        ctrl._async_cfg = SimpleNamespace(
            max_inflight_prompts=1,
            diagnostics=False,
            rollout_failure=_failure_cfg(),
        )
        ctrl._master_config = SimpleNamespace(
            grpo=GRPOConfig.model_construct(max_num_epochs=1)
        )
        ctrl._rollout_manager = _NeverCalledRolloutManager()
        # Over-sampled windowed policy: admit never gates (buffer unused here).
        ctrl._sampler = WindowedSampler(None, max_staleness_versions=1)
        ctrl._dataloader = [
            BatchedDataDict({"message_log": [[{"role": "user", "content": "prompt"}]]})
        ]
        ctrl._rollout_permitted = asyncio.Event()
        ctrl._rollout_permitted.set()
        ctrl._rollout_exhausted = asyncio.Event()
        ctrl._buffer_capacity = real_semaphore(1)
        ctrl._inflight_rollouts = 0
        ctrl._inflight_by_group_id = {}
        ctrl._dispatched_rollouts = set()
        ctrl._trainer_version = 0
        ctrl._current_epoch = 0
        _init_pump_ledgers(ctrl)

        await ctrl._rollout_pump()
        await asyncio.sleep(0)

        assert ctrl._buffer_capacity._value == 1
        assert created_semaphores[0]._value == 1
        assert ctrl._inflight_rollouts == 0
        assert ctrl._dispatched_rollouts == set()
        assert ctrl._rollout_exhausted.is_set()

    asyncio.run(_main())


@pytest.mark.vllm
def test_rollout_pump_writes_expected_tq_data(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,  # noqa: F811
    tmp_path,
):
    """SC._rollout_pump writes num_prompts * num_generations rows to TQ with the expected fields and tags."""
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample

    num_generations = 2
    num_prompts = 2
    # TQReplayBuffer.commit writes ``num_generations`` training rows per prompt.
    expected_samples = num_prompts * num_generations
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    tq_actor = _TQActor.remote(
        partition_id=_PARTITION_ID,
        fields=_BULK_FIELDS,
        num_samples=expected_samples * 4,
        consumer_tasks=["train"],
    )
    dp_adapter = _SyncDPAdapter(tq_actor)

    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": expected_samples},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=num_prompts,
            num_generations_per_prompt=num_generations,
            max_num_steps=1,
            max_num_epochs=1,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            sampler=WindowedSamplerConfig(max_staleness_versions=1),
            min_groups_for_streaming_train=1,
            max_inflight_prompts=num_prompts,
            max_buffered_rollouts=num_prompts,
        ),
        logger={
            "log_dir": str(tmp_path / "logs"),
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "monitor_gpus": False,
        },
        # Actor __init__ builds a CheckpointManager + TimeoutChecker from
        # this block; enabled=False keeps the run write-free.
        checkpointing={
            "enabled": False,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "metric_name": None,
            "higher_is_better": False,
            "keep_top_k": None,
            "save_period": 10_000,
            "save_optimizer": False,
            "checkpoint_must_save_by": None,
        },
    )
    # Wrap each value in a single-element list so size==1 and v[0] returns the original field.
    batched_sample = BatchedDataDict({k: [v] for k, v in input_sample.items()})
    dataloader = [batched_sample] * num_prompts

    tq_buffer = TQReplayBuffer(
        dp_adapter,
        partition_id=_PARTITION_ID,
        pad_value_dict={"token_ids": int(tokenizer.pad_token_id or 0)},
    )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
        use_nemo_gym=False,
        tq_buffer=tq_buffer,
    )
    actor_args = SingleControllerActorArgs(
        gen_handle=vllm_generation,
        trainer_handle=object(),
        env_handles={},
        train_cluster=None,  # type: ignore[arg-type]  # unused by _rollout_pump
        inference_cluster=None,  # type: ignore[arg-type]
        dp_client=dp_adapter,
        dataloader=dataloader,
        weight_synchronizer=object(),  # type: ignore[arg-type]
        advantage_estimator=None,
        loss_fn=None,  # type: ignore[arg-type]
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=_PARTITION_ID,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    ctrl = SingleControllerActor.remote(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )

    vllm_generation.prepare_for_generation()
    ray.get(ctrl._rollout_pump.remote())
    vllm_generation.finish_generation()

    sample_ids = ray.get(tq_actor.list_sample_ids.remote(_PARTITION_ID))
    assert len(sample_ids) == expected_samples

    # pack_payload stamps sample_ids as ``{group_uuid}_g{i}``.
    group_ids: set[str] = set()
    for sid in sample_ids:
        head, _, tail = sid.rpartition("_g")
        assert head and tail.isdigit(), f"unexpected sample_id: {sid}"
        group_ids.add(head)
    assert len(group_ids) == num_prompts

    bulk = ray.get(
        tq_actor.get_samples.remote(
            sample_ids=sample_ids,
            partition_id=_PARTITION_ID,
            select_fields=_BULK_FIELDS,
        )
    )
    assert set(bulk.keys()) >= set(_BULK_FIELDS), (
        f"missing bulk fields: {set(_BULK_FIELDS) - set(bulk.keys())}"
    )

    input_lengths = bulk["input_lengths"].long()
    assert input_lengths.shape[0] == expected_samples
    assert torch.all(input_lengths > 0)
    assert torch.allclose(
        bulk["sample_mask"].float(),
        torch.ones(expected_samples, dtype=torch.float32),
    )

    # Same deterministic prompt as test_async_rollout_manager: the model
    # solves the calculator task every time -> reward == 1.0 and decoded
    # tail contains " 16".
    rewards = bulk["total_reward"].float().flatten()
    assert rewards.shape == (expected_samples,)
    assert torch.allclose(rewards, torch.ones(expected_samples)), (
        f"expected all rewards == 1.0, got {rewards.tolist()}"
    )

    input_ids = bulk["input_ids"]
    token_mask = bulk["token_mask"]
    for i in range(expected_samples):
        length = int(input_lengths[i])
        decoded = tokenizer.decode(
            input_ids[i, :length].tolist(), skip_special_tokens=False
        )
        assert " 16" in decoded[-64:], (
            f"sample {i}: decoded tail {decoded[-64:]!r} missing ' 16'"
        )
        assert int(token_mask[i, :length].sum().item()) > 0, (
            f"sample {i}: token_mask has no assistant tokens"
        )

    tags = ray.get(
        tq_actor.get_tags.remote(partition_id=_PARTITION_ID, sample_ids=sample_ids)
    )
    for tag in tags:
        assert tag["weight_version"] == 0
        # Slim tag schema: weight_version is the only field producers stamp.
        assert set(tag) == {"weight_version"}
