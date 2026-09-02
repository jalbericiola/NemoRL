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

"""Tests for RolloutManager.

Two groups:

* TestGenerateAndPushFlow — lightweight unit tests for the reserve→run→commit
  flow in generate_and_push (no Ray/vLLM; fakes for impl + tq_buffer).
* AsyncRollout / AsyncNemoGymRollout tests — vLLM/Ray-backed end-to-end checks
  for the underlying run_rollout paths (AsyncRolloutImpl / AsyncNemoGymRolloutImpl).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections import UserDict
from copy import deepcopy
from typing import cast

import pytest
import torch
from pydantic import BaseModel

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets.response_datasets import NemoGymDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.processors import nemo_gym_data_processor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.failures import GenerationUnavailable
from nemo_rl.experience.interfaces import (
    NEMO_GYM_ROLLOUT_INDEX_KEY,
    NEMO_GYM_TASK_INDEX_KEY,
    RAW_ENVIRONMENT_REWARD,
    Completion,
    PromptGroupRecord,
)
from nemo_rl.experience.rollout_manager import (
    AsyncNemoGymRolloutImpl,
    AsyncRolloutImpl,
    RolloutManager,
    RolloutRetryPolicy,
    RolloutStats,
    _add_cohort_rollout_metrics,
)
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
)
from nemo_rl.utils.timer import Timer

# Fixtures shared with the heavyweight rollout tests.
from tests.unit.environments.test_nemo_gym import (
    cluster,  # noqa: F401
    nemo_gym,  # noqa: F401
    nemo_gym_sanity_test_data,  # noqa: F401
    nemo_gym_tokenizer,  # noqa: F401
    nemo_gym_vllm_generation,  # noqa: F401
)
from tests.unit.experience.test_rollouts import (
    initial_multi_step_calculator_batch,  # noqa: F401
    multi_step_calculator_environment,  # noqa: F401
    multi_step_setup_vllm_async,  # noqa: F401
    rollout_cluster,  # noqa: F401
    rollout_tokenizer,  # noqa: F401
)
from tests.unit.test_envs import MultiStepCalcMetadata

_UNSET = object()


def _run(coro):
    return asyncio.run(coro)


def test_cohort_generated_tokens_exclude_prompt_history_assistants() -> None:
    metrics: dict = {}
    completion = Completion(
        message_log=[
            {"role": "user", "token_ids": torch.tensor([1])},
            {"role": "assistant", "token_ids": torch.tensor([2, 3])},
            {
                "role": "assistant",
                "token_ids": torch.tensor([4, 5, 6]),
                "generation_logprobs": torch.zeros(3),
            },
        ],
        env_extras=None,
        truncated=False,
        reward=1.0,
    )

    _add_cohort_rollout_metrics(metrics, [completion])

    assert metrics["cohort/generated_tokens"] == 3
    assert metrics["cohort/total_tokens"] == 6


def test_cohort_env_mask_count_uses_resolved_provenance_only() -> None:
    metrics: dict = {}
    raw_only = Completion(
        message_log=[{"role": "user", "token_ids": torch.tensor([1])}],
        env_extras={"instance_config": {"mask_sample": True}},
        truncated=False,
        reward=1.0,
        env_masked=False,
    )
    resolved = Completion(
        message_log=[{"role": "user", "token_ids": torch.tensor([2])}],
        env_extras={"instance_config": {"mask_sample": False}},
        truncated=False,
        reward=1.0,
        env_masked=True,
    )

    _add_cohort_rollout_metrics(metrics, [raw_only, resolved])

    assert metrics["cohort/env_masked_sample_count"] == 1


class _FakeBuffer:
    """Minimal TQReplayBuffer stand-in that records reserve/commit calls."""

    def __init__(self) -> None:
        self.reserve_calls: list[int] = []  # weight_versions passed to reserve
        self.commit_calls: list[tuple[str, object, int, int]] = []
        self.remove_calls: list[str] = []
        self.remove_in_dp_calls: list[bool] = []
        # reserve(weight_version=X) -> group_id; commit fills the slot.
        self._slots: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: int | None = None,
        group_id: str | None = None,
    ) -> str:
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.reserve_calls.append(weight_version)
        self._slots.append(group_id)
        return group_id

    async def commit(
        self,
        group_id: str,
        record,
        start_weight_version: int,
        end_weight_version: int,
    ):
        self.commit_calls.append(
            (group_id, record, start_weight_version, end_weight_version)
        )
        return record

    def replace_committed_rollout_metrics(self, group_id: str, rollout_metrics):
        assert self.commit_calls[-1][0] == group_id
        self.commit_calls[-1][1].rollout_metrics = dict(rollout_metrics)

    async def remove_group(self, group_id: str, *, remove_in_dp: bool = False) -> int:
        self.remove_calls.append(group_id)
        self.remove_in_dp_calls.append(remove_in_dp)
        self._slots.remove(group_id)
        return 1


class _FakeImpl:
    """Stand-in for AsyncRolloutImpl that returns a sentinel record."""

    def __init__(self, record="sentinel-record", on_run=None) -> None:
        self._record = record
        self._on_run = on_run
        self.rollout_group_ids: list[str | None] = []

    async def run_rollout(self, input_sample, *, rollout_group_id: str | None = None):
        self.rollout_group_ids.append(rollout_group_id)
        if self._on_run is not None:
            await self._on_run(input_sample)
        return self._record


def _make_manager(
    buffer: _FakeBuffer, impl: _FakeImpl, retry_policy: RolloutRetryPolicy | None = None
) -> RolloutManager:
    """Build a RolloutManager without firing the real __init__.

    The default policy is single-attempt, matching RolloutRetryPolicy's own default, so
    these tests keep exercising the no-retry path unless they ask for otherwise.
    """
    mgr = object.__new__(RolloutManager)
    mgr._impl = impl
    mgr._tokenizer = None
    mgr._num_generations_per_prompt = 1
    mgr._tq_buffer = buffer
    mgr._weight_version = 0
    mgr._retry_policy = (
        retry_policy
        if retry_policy is not None
        else RolloutRetryPolicy.single_attempt()
    )
    mgr._stats = RolloutStats()
    mgr._skipped_prompts = 0
    mgr._consecutive_infra_drops = 0
    return mgr


class TestGenerateAndPushFlow:
    def test_explicit_registry_tracks_only_inflight_generation(self):
        registry: dict[str, tuple[asyncio.Task[None], int]] = {}
        buf = _FakeBuffer()

        async def _assert_registered(_sample):
            assert len(registry) == 1
            task, start_version = next(iter(registry.values()))
            assert task is asyncio.current_task()
            assert start_version == 3

        mgr = _make_manager(buf, _FakeImpl(on_run=_assert_registered))
        mgr.set_weight_version(3)

        _run(
            mgr.generate_and_push(
                {"prompt": "p"},
                inflight_registry=registry,
            )
        )

        assert registry == {}

    def test_rollout_failure_removes_reserved_group(self):
        async def _fail_rollout(_sample):
            raise RuntimeError("injected rollout failure")

        registry: dict[str, tuple[asyncio.Task[None], int]] = {}
        buf = _FakeBuffer()
        mgr = _make_manager(buf, _FakeImpl(on_run=_fail_rollout))

        with pytest.raises(RuntimeError, match="injected rollout failure"):
            _run(mgr.generate_and_push({"prompt": "p"}, inflight_registry=registry))

        assert len(buf.reserve_calls) == 1
        assert len(buf.remove_calls) == 1
        assert mgr._impl.rollout_group_ids == buf.remove_calls
        assert buf._slots == []
        assert buf.commit_calls == []
        assert registry == {}

    def test_cleanup_failure_does_not_mask_original_exception(self):
        class _RaisingBuffer(_FakeBuffer):
            async def remove_group(self, group_id, *, remove_in_dp=False):
                raise RuntimeError("remove_group cleanup boom")

        class _OriginalError(Exception):
            pass

        async def _raise_original(_sample):
            raise _OriginalError("original rollout failure")

        buf = _RaisingBuffer()
        mgr = _make_manager(buf, _FakeImpl(on_run=_raise_original))

        with pytest.raises(_OriginalError):
            _run(mgr.generate_and_push({"prompt": "p"}))

    def test_reserves_then_runs_then_commits(self):
        events: list[str] = []
        buf = _FakeBuffer()

        async def _track_run(_sample):
            events.append("run")

        impl = _FakeImpl(record="r0", on_run=_track_run)
        mgr = _make_manager(buf, impl)

        # Wrap reserve/commit to log ordering.
        original_reserve = buf.reserve
        original_commit = buf.commit

        def _logged_reserve(**kwargs):
            events.append("reserve")
            return original_reserve(**kwargs)

        async def _logged_commit(*args, **kwargs):
            events.append("commit")
            return await original_commit(*args, **kwargs)

        buf.reserve = _logged_reserve  # type: ignore[method-assign]
        buf.commit = _logged_commit  # type: ignore[method-assign]

        _run(mgr.generate_and_push({"prompt": "p"}))

        assert events == ["reserve", "run", "commit"]
        assert buf.reserve_calls == [0]
        assert len(buf.commit_calls) == 1
        gid, record, start_v, end_v = buf.commit_calls[0]
        assert gid in buf._slots
        assert impl.rollout_group_ids == [gid]
        assert record == "r0"
        assert start_v == 0
        assert end_v == 0

    def test_redispatch_gets_fresh_group_id(self):
        attempts = 0

        async def _fail_first_attempt(_sample):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise GenerationUnavailable("injected first-attempt failure")

        buf = _FakeBuffer()
        impl = _FakeImpl(record="recovered", on_run=_fail_first_attempt)
        policy = RolloutRetryPolicy.single_attempt(
            max_infra_attempts=2,
            backoff_base_s=0.0,
        )
        mgr = _make_manager(buf, impl, retry_policy=policy)

        _run(mgr.generate_and_push({"prompt": "p"}))

        assert len(impl.rollout_group_ids) == 2
        assert impl.rollout_group_ids[0] != impl.rollout_group_ids[1]
        assert buf.remove_calls == [impl.rollout_group_ids[0]]
        assert [call[0] for call in buf.commit_calls] == [impl.rollout_group_ids[1]]

    def test_success_receipt_covers_failed_attempt_work_and_backoff(self, monkeypatch):
        attempts = 0

        async def _fail_first_attempt(_sample):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise GenerationUnavailable("injected first-attempt failure")

        record = PromptGroupRecord(
            prompt_idx=0,
            prompt=[],
            extra_env_info=None,
            metadata={},
            completions=[],
            # These attempt-local values must be replaced by the prompt lifecycle.
            rollout_metrics={
                "cohort/rollout_started_at_s": 8.0,
                "cohort/rollout_finished_at_s": 9.0,
                "timing/rollout/total": 1.0,
            },
        )
        buf = _FakeBuffer()
        impl = _FakeImpl(record=record, on_run=_fail_first_attempt)
        policy = RolloutRetryPolicy.single_attempt(
            max_infra_attempts=2,
            backoff_base_s=0.0,
        )
        mgr = _make_manager(buf, impl, retry_policy=policy)
        wall_timestamps = iter((100.0, 115.0, 120.0))
        work_timestamps = iter((1.0, 4.0, 5.0, 10.0))
        monkeypatch.setattr(
            "nemo_rl.experience.rollout_manager.time.time",
            lambda: next(wall_timestamps),
        )
        monkeypatch.setattr(
            "nemo_rl.experience.rollout_manager.time.perf_counter",
            lambda: next(work_timestamps),
        )

        _run(mgr.generate_and_push({"prompt": "p"}))

        _, committed, _, _ = buf.commit_calls[0]
        assert committed.rollout_metrics["cohort/rollout_started_at_s"] == 100.0
        assert committed.rollout_metrics["cohort/rollout_finished_at_s"] == 120.0
        # Two attempts took 3s and 5s.  The 7s between them is lifecycle/backoff,
        # represented by the wall interval but excluded from attempt work.
        assert committed.rollout_metrics["timing/rollout/total"] == 8.0

    def test_commit_latency_uses_same_lifecycle_boundary_on_retry(self, monkeypatch):
        class _Clock:
            def __init__(self) -> None:
                self.wall_s = 100.0
                self.work_s = 0.0

            def time(self) -> float:
                return self.wall_s

            def perf_counter(self) -> float:
                return self.work_s

            def advance_rollout(self, seconds: float) -> None:
                self.wall_s += seconds
                self.work_s += seconds

            def advance_wall(self, seconds: float) -> None:
                self.wall_s += seconds

        class _SlowCommitBuffer(_FakeBuffer):
            def __init__(self, clock: _Clock) -> None:
                super().__init__()
                self._clock = clock
                self._commit_durations = iter((7.0, 11.0))
                self.commit_attempts: list[str] = []
                self.receipts: dict[str, dict] = {}
                self.receipt_history: list[tuple[str, str, dict]] = []

            async def commit(
                self,
                group_id: str,
                record,
                start_weight_version: int,
                end_weight_version: int,
            ):
                self.commit_attempts.append(group_id)
                self._clock.advance_wall(next(self._commit_durations))
                if len(self.commit_attempts) == 1:
                    raise ConnectionError("injected slow failed commit")
                self.commit_calls.append(
                    (group_id, record, start_weight_version, end_weight_version)
                )
                receipt = deepcopy(record.rollout_metrics)
                self.receipts[group_id] = receipt
                self.receipt_history.append(("commit", group_id, receipt))
                return record

            def replace_committed_rollout_metrics(
                self, group_id: str, rollout_metrics
            ) -> None:
                receipt = deepcopy(rollout_metrics)
                self.receipts[group_id] = receipt
                self.receipt_history.append(("replace", group_id, receipt))

        clock = _Clock()
        rollout_durations = iter((2.0, 3.0))

        async def _run_for_attempt_duration(_sample) -> None:
            clock.advance_rollout(next(rollout_durations))

        async def _advance_backoff(seconds: float) -> None:
            clock.advance_wall(seconds)

        record = PromptGroupRecord(
            prompt_idx=0,
            prompt=[],
            extra_env_info=None,
            metadata={},
            completions=[],
            rollout_metrics={
                "cohort/rollout_started_at_s": 8.0,
                "cohort/rollout_finished_at_s": 9.0,
                "timing/rollout/total": 1.0,
            },
        )
        buf = _SlowCommitBuffer(clock)
        impl = _FakeImpl(record=record, on_run=_run_for_attempt_duration)
        policy = RolloutRetryPolicy.single_attempt(
            max_infra_attempts=2,
            backoff_base_s=5.0,
        )
        mgr = _make_manager(buf, impl, retry_policy=policy)
        monkeypatch.setattr("nemo_rl.experience.rollout_manager.time.time", clock.time)
        monkeypatch.setattr(
            "nemo_rl.experience.rollout_manager.time.perf_counter",
            clock.perf_counter,
        )
        monkeypatch.setattr(
            "nemo_rl.experience.rollout_manager.asyncio.sleep", _advance_backoff
        )

        _run(mgr.generate_and_push({"prompt": "p"}))

        failed_gid, successful_gid = buf.commit_attempts
        assert failed_gid != successful_gid
        assert buf.remove_calls == [failed_gid]
        assert buf.remove_in_dp_calls == [False]
        assert mgr._stats.redispatches_by_reason == {"ConnectionError": 1}
        assert [event for event, _, _ in buf.receipt_history] == [
            "commit",
            "replace",
        ]
        provisional_receipt = buf.receipt_history[0][2]
        assert provisional_receipt["cohort/rollout_finished_at_s"] == 117.0
        assert buf.receipts[successful_gid] == {
            **provisional_receipt,
            "cohort/rollout_started_at_s": 100.0,
            "cohort/rollout_finished_at_s": 128.0,
            "timing/rollout/total": 5.0,
        }

    def test_post_commit_receipt_failure_clears_materialized_rows(self):
        class _RejectingReceiptBuffer(_FakeBuffer):
            def replace_committed_rollout_metrics(
                self, group_id: str, rollout_metrics
            ) -> None:
                del group_id, rollout_metrics
                raise RuntimeError("injected receipt replacement failure")

        record = PromptGroupRecord(
            prompt_idx=0,
            prompt=[],
            extra_env_info=None,
            metadata={},
            completions=[],
            rollout_metrics={
                "cohort/rollout_started_at_s": 8.0,
                "cohort/rollout_finished_at_s": 9.0,
                "timing/rollout/total": 1.0,
            },
        )
        buf = _RejectingReceiptBuffer()
        mgr = _make_manager(buf, _FakeImpl(record=record))

        with pytest.raises(RuntimeError, match="receipt replacement failure"):
            _run(mgr.generate_and_push({"prompt": "p"}))

        assert buf.remove_calls == [buf.commit_calls[0][0]]
        assert buf.remove_in_dp_calls == [True]

    def test_start_weight_version_pinned_at_reserve_time(self):
        """If set_weight_version is called mid-rollout, start != end."""
        buf = _FakeBuffer()

        async def _bump_weight_mid_rollout(_sample):
            # Simulate a sync_weights bump during the rollout.
            mgr.set_weight_version(5)

        impl = _FakeImpl(record="r0", on_run=_bump_weight_mid_rollout)
        mgr = _make_manager(buf, impl)
        mgr.set_weight_version(3)

        _run(mgr.generate_and_push({"prompt": "p"}))

        # reserve happened before run_rollout → captured weight 3.
        assert buf.reserve_calls == [3]
        # commit's start is the same dispatch-time value; end reflects the post-rollout weight.
        _, _, start_v, end_v = buf.commit_calls[0]
        assert start_v == 3
        assert end_v == 5

    def test_no_weight_change_means_start_equals_end(self):
        buf = _FakeBuffer()
        impl = _FakeImpl(record="r0")
        mgr = _make_manager(buf, impl)
        mgr.set_weight_version(7)

        _run(mgr.generate_and_push({"prompt": "p"}))

        _, _, start_v, end_v = buf.commit_calls[0]
        assert start_v == 7
        assert end_v == 7

    def test_concurrent_dispatch_preserves_reserve_order(self):
        """Two concurrent generate_and_push calls must reserve before either commits.

        The contract: reserve order == dispatch order, even if rollouts finish
        out of order. Slot order in the buffer reflects the order reserve was
        called (not the order run_rollout completed).
        """
        buf = _FakeBuffer()

        # First call's rollout blocks until second call has reserved.
        first_reserved = asyncio.Event()
        second_reserved = asyncio.Event()

        async def _first_run(_sample):
            first_reserved.set()
            await second_reserved.wait()

        async def _second_run(_sample):
            # Second is dispatched only after first reserves, so by the time
            # second's reserve fires, slots[0] == first's gid.
            second_reserved.set()

        first_impl = _FakeImpl(record="r0", on_run=_first_run)
        second_impl = _FakeImpl(record="r1", on_run=_second_run)

        first_mgr = _make_manager(buf, first_impl)
        # Share buffer across two managers (mimics two dispatches from one pump).
        # Built through the shared helper so new RolloutManager attributes only have to
        # be added in one place.
        second_mgr = _make_manager(buf, second_impl)

        async def _drive():
            t1 = asyncio.create_task(first_mgr.generate_and_push({"prompt": "p1"}))
            # Wait until first has reserved before kicking off second so the
            # reserve ordering is deterministic.
            await first_reserved.wait()
            t2 = asyncio.create_task(second_mgr.generate_and_push({"prompt": "p2"}))
            await asyncio.gather(t1, t2)

        _run(_drive())

        # Slots in buffer == reserve order.
        first_gid, second_gid = buf._slots
        # Commit recorded both, in either order, but each maps to its own gid.
        commit_gids = [c[0] for c in buf.commit_calls]
        assert set(commit_gids) == {first_gid, second_gid}
        assert buf.reserve_calls == [0, 0]

    def test_requires_tq_buffer(self):
        mgr = _make_manager(_FakeBuffer(), _FakeImpl())
        mgr._tq_buffer = None
        with pytest.raises(AssertionError, match="tq_buffer"):
            _run(mgr.generate_and_push({"prompt": "p"}))


# ---------------------------------------------------------------------------
# Tests for RolloutManager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("impl_type", [AsyncRolloutImpl, AsyncNemoGymRolloutImpl])
def test_rollout_impl_requires_loss_multiplier_before_work(impl_type) -> None:
    impl = object.__new__(impl_type)

    with pytest.raises(ValueError, match="DatumSpec requires loss_multiplier"):
        _run(impl.run_rollout(cast(DatumSpec, {})))


def test_rollout_manager_raises_without_impl_params():
    """RolloutManager raises AssertionError when required params are missing."""
    common = {
        "tokenizer": None,
        "task_to_env": {},
        "num_generations_per_prompt": 1,
        "max_seq_len": 1,
    }

    with pytest.raises(AssertionError, match="num_generations_per_prompt must be >= 1"):
        updated_common = common.copy()
        updated_common["num_generations_per_prompt"] = 0
        RolloutManager(**updated_common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="policy_generation is required"):
        RolloutManager(**common, use_nemo_gym=False)

    with pytest.raises(AssertionError, match="generation_config is required"):
        RolloutManager(**common, use_nemo_gym=True)


def test_rollout_manager_forwards_mask_env_flagged_samples():
    """env.should_mask_flagged_samples reaches the NeMo-Gym impl through RolloutManager."""
    common = {
        "tokenizer": None,
        "task_to_env": {},
        "num_generations_per_prompt": 1,
        "max_seq_len": 1,
        "generation_config": {
            "stop_strings": None,
            "stop_token_ids": None,
            "top_k": None,
        },
        "use_nemo_gym": True,
    }

    assert RolloutManager(**common)._impl._mask_env_flagged_samples is True
    manager = RolloutManager(**common, mask_env_flagged_samples=False)
    assert manager._impl._mask_env_flagged_samples is False


def test_rollout_manager_rejects_enabled_penalties_on_native_path():
    with pytest.raises(ValueError, match="reward_penalties require use_nemo_gym=true"):
        RolloutManager(
            tokenizer=None,
            task_to_env={},
            num_generations_per_prompt=1,
            max_seq_len=1,
            policy_generation=object(),
            use_nemo_gym=False,
            reward_penalty_config={"penalize_empty_final_answer": True},
        )


def test_rollout_manager_rejects_unknown_enabled_penalty_flag():
    with pytest.raises(ValueError, match="unsupported enabled reward penalty"):
        RolloutManager(
            tokenizer=None,
            task_to_env={},
            num_generations_per_prompt=1,
            max_seq_len=1,
            policy_generation=object(),
            use_nemo_gym=False,
            reward_penalty_config={"penalize_empty_final_answers": True},
        )


def test_rollout_manager_rejects_unknown_enabled_penalty_on_base_model():
    class LoosePenaltyConfig(BaseModel, extra="allow"):
        pass

    with pytest.raises(ValueError, match="unsupported enabled reward penalty"):
        RolloutManager(
            tokenizer=None,
            task_to_env={},
            num_generations_per_prompt=1,
            max_seq_len=1,
            policy_generation=object(),
            use_nemo_gym=False,
            reward_penalty_config=LoosePenaltyConfig(penalize_empty_final_answers=True),
        )


def test_rollout_manager_forwards_resolved_penalties_to_nemo_gym_impl():
    manager = RolloutManager(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=1,
        max_seq_len=1,
        generation_config={
            "stop_strings": None,
            "stop_token_ids": None,
            "top_k": None,
        },
        use_nemo_gym=True,
        reward_penalty_config=UserDict({"penalize_empty_final_answer": True}),
    )

    assert manager._impl._reward_penalty_config == {"penalize_empty_final_answer": True}


def _nemo_gym_impl(
    mask_env_flagged_samples: bool,
    num_generations_per_prompt: int = 1,
    *,
    add_seed_per_rollout: object = _UNSET,
    seed_base: object = _UNSET,
) -> AsyncNemoGymRolloutImpl:
    generation_config = {
        "stop_strings": None,
        "stop_token_ids": None,
        "top_k": None,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 64,
    }
    if add_seed_per_rollout is not _UNSET:
        generation_config["nemo_gym_add_seed_per_rollout"] = add_seed_per_rollout
    if seed_base is not _UNSET:
        generation_config["nemo_gym_per_rollout_seed_base"] = seed_base
    return AsyncNemoGymRolloutImpl(
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=num_generations_per_prompt,
        max_seq_len=100,
        max_rollout_turns=1,
        generation_config=generation_config,
        mask_env_flagged_samples=mask_env_flagged_samples,
    )


def test_nemo_gym_inputs_use_attempt_identity_without_mutating_source():
    rollout_group_id = "12345678-1234-5678-1234-567812345678"
    source_task_index = 17
    input_sample = cast(
        DatumSpec,
        {
            "message_log": [],
            "extra_env_info": {
                "agent_ref": {"name": "test_agent"},
                "responses_create_params": {"max_output_tokens": 100},
                NEMO_GYM_TASK_INDEX_KEY: source_task_index,
                NEMO_GYM_ROLLOUT_INDEX_KEY: 99,
            },
            "task_name": "nemo_gym",
            "idx": 0,
        },
    )
    original = deepcopy(input_sample)
    impl = _nemo_gym_impl(mask_env_flagged_samples=True, num_generations_per_prompt=3)

    rows = impl._build_inputs(input_sample, rollout_group_id=rollout_group_id)

    task_indices = [row[NEMO_GYM_TASK_INDEX_KEY] for row in rows]
    assert len(set(task_indices)) == 1
    assert isinstance(task_indices[0], int)
    assert task_indices[0] != source_task_index
    assert 0 <= task_indices[0] < 1 << 63
    assert [row[NEMO_GYM_ROLLOUT_INDEX_KEY] for row in rows] == [0, 1, 2]
    assert [row["_rowidx"] for row in rows] == [0, 1, 2]
    assert input_sample == original


def test_nemo_gym_run_rollout_records_exact_wall_interval(monkeypatch):
    impl = _nemo_gym_impl(mask_env_flagged_samples=True)
    input_sample = cast(
        DatumSpec,
        {
            "message_log": [],
            "extra_env_info": {
                "agent_ref": {"name": "test_agent"},
                "responses_create_params": {"max_output_tokens": 100},
            },
            "task_name": "nemo_gym",
            "idx": 0,
            "loss_multiplier": 1.0,
        },
    )

    async def _run_rollouts(_inputs, _timer, _timer_prefix):
        return [], [], {}

    impl._run_rollouts = _run_rollouts
    timestamps = iter((10.0, 13.0))
    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.time.time", lambda: next(timestamps)
    )

    record = _run(impl.run_rollout(input_sample))

    assert record.rollout_metrics["cohort/rollout_started_at_s"] == 10.0
    assert record.rollout_metrics["cohort/rollout_finished_at_s"] == 13.0


def test_nemo_gym_run_rollouts_applies_penalty_and_preserves_reward_boundary():
    result = {
        "input_message_log": [{"role": "user", "token_ids": [1]}],
        "message_log": [
            {"role": "user", "token_ids": [1]},
            {
                "role": "assistant",
                "token_ids": [2, 3],
                "generation_logprobs": [0.0, 0.0],
            },
        ],
        "full_result": {
            "reward": 1.0,
            "response": {
                "output": [
                    {"type": "reasoning", "summary": [{"text": "same"}]},
                    {"type": "message", "content": [{"text": "same"}]},
                ]
            },
        },
    }
    impl = _nemo_gym_impl(mask_env_flagged_samples=True)
    impl._effort_config = EffortLevelsConfig(
        low_string="efficient",
        low_weight=1.0,
        low_penalty=1.0,
        low_ub=20,
    )
    impl._reward_penalty_config = {"penalize_duplicated_reasoning": True}

    async def _stream_rows(_env, _pending, results, _total_rows, _timer_prefix):
        results[0] = deepcopy(result)
        return {}

    impl._stream_rows = _stream_rows
    impl._task_to_env = {"nemo_gym": object()}
    completions, _, metrics = _run(
        impl._run_rollouts(
            [
                {
                    "_rowidx": 0,
                    "agent_ref": {"name": "agent"},
                    "responses_create_params": {
                        "input": [{"role": "user", "content": "be efficient"}]
                    },
                }
            ],
            timer=Timer(),
            timer_prefix="timing/test",
        )
    )

    completion = completions[0]
    assert completion.raw_environment_reward == 1.0
    assert completion.pre_penalty_reward == pytest.approx(1.9)
    assert completion.reward == 0.0
    assert metrics[f"{RAW_ENVIRONMENT_REWARD}/mean"] == 1.0
    assert metrics["pre_penalty_environment_reward/mean"] == pytest.approx(1.9)
    assert metrics["total_reward/mean"] == 0.0
    assert metrics["cohort/raw_environment_reward_sum"] == 1.0
    assert metrics["cohort/pre_penalty_reward_sum"] == pytest.approx(1.9)
    assert metrics["cohort/post_penalty_reward_sum"] == 0.0
    assert metrics["cohort/effort_low_sample_count"] == 1
    assert metrics["cohort/effort_reward_delta_sum"] == pytest.approx(0.9)
    assert metrics["mean_length_reward_low"] == pytest.approx(0.9)
    assert metrics["mean_reward_low"] == pytest.approx(1.9)
    assert metrics["mean_length_low"] == 2
    assert metrics["median_length_low"] == 2
    assert metrics["cohort/duplicated_reasoning_count"] == 1
    assert metrics["reasoning_equal_to_final_answer_rate"] == 1.0
    assert metrics["cohort/samples"] == 1
    assert metrics["cohort/generated_tokens"] == 2
    assert metrics["cohort/total_tokens"] == 3
    for count_key, rate_key in (
        ("empty_final_answer", "empty_final_answer_rate"),
        ("unwanted_token", "unwanted_token_rate"),
        ("malformed_think_tag", "malformed_think_tag_rate"),
    ):
        assert metrics[f"cohort/{count_key}_count"] == 0
        assert metrics[rate_key] == 0.0

    # Environment result remains mutable provenance, but reward boundaries are
    # immutable scalar snapshots and cannot be rewritten through that alias.
    completion.env_extras["reward"] = 99.0
    assert completion.raw_environment_reward == 1.0
    assert completion.pre_penalty_reward == pytest.approx(1.9)
    assert completion.reward == 0.0
    assert result["full_result"]["reward"] == 1.0


def test_nemo_gym_penalties_tensorize_token_ids_before_malformed_check():
    result = {
        "input_message_log": [{"role": "user", "token_ids": [12, 7]}],
        "message_log": [
            {"role": "user", "token_ids": [12, 7]},
            {
                "role": "assistant",
                "token_ids": [8],
                "generation_logprobs": [0.0],
            },
        ],
        "full_result": {"reward": 1.0},
    }
    impl = _nemo_gym_impl(mask_env_flagged_samples=True)
    impl._reward_penalty_config = {
        "penalize_malformed_think_tag": True,
        "token_ids": {"think_open": 12, "think_close": 13},
    }

    async def _stream_rows(_env, _pending, results, _total_rows, _timer_prefix):
        results[0] = deepcopy(result)
        return {}

    impl._stream_rows = _stream_rows
    impl._task_to_env = {"nemo_gym": object()}
    completions, _, metrics = _run(
        impl._run_rollouts(
            [{"_rowidx": 0, "agent_ref": {"name": "agent"}}],
            timer=Timer(),
            timer_prefix="timing/test",
        )
    )

    assert isinstance(completions[0].message_log[0]["token_ids"], torch.Tensor)
    assert completions[0].reward == 0.0
    assert metrics["cohort/malformed_think_tag_count"] == 1
    assert metrics["malformed_think_tag_rate"] == 1.0


def test_nemo_gym_inputs_reject_invalid_attempt_identity():
    input_sample = cast(
        DatumSpec,
        {"extra_env_info": {"responses_create_params": {}}},
    )
    impl = _nemo_gym_impl(mask_env_flagged_samples=True)

    with pytest.raises(ValueError, match="rollout_group_id must be a valid UUID"):
        impl._build_inputs(input_sample, rollout_group_id="not-a-uuid")


def test_nemo_gym_inputs_isolate_repeated_attempts():
    input_sample = cast(
        DatumSpec,
        {
            "extra_env_info": {
                "responses_create_params": {},
                NEMO_GYM_TASK_INDEX_KEY: None,
            },
        },
    )
    impl = _nemo_gym_impl(mask_env_flagged_samples=True)

    first = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )
    second = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000002",
    )

    assert first[0][NEMO_GYM_TASK_INDEX_KEY] != second[0][NEMO_GYM_TASK_INDEX_KEY]


def _nemo_gym_request_seeds(rows: list[dict]) -> list[int]:
    return [
        json.loads(row["responses_create_params"]["metadata"]["extra_body"])["seed"]
        for row in rows
    ]


def test_nemo_gym_inputs_add_stable_per_rollout_seeds_and_preserve_extra_body():
    input_sample = cast(
        DatumSpec,
        {
            "idx": 7,
            "extra_env_info": {
                "responses_create_params": {
                    "metadata": {
                        "request_label": "kept",
                        "extra_body": json.dumps(
                            {"guided_decoding": {"choice": ["A", "B"]}}
                        ),
                    }
                }
            },
        },
    )
    original = deepcopy(input_sample)
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        num_generations_per_prompt=4,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    rows = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )

    assert input_sample == original
    assert [row[NEMO_GYM_ROLLOUT_INDEX_KEY] for row in rows] == [0, 1, 2, 3]
    assert _nemo_gym_request_seeds(rows) == [
        8268443954876357215,
        178608576629793298,
        1550254623586035897,
        127339396338969742,
    ]
    assert len(set(_nemo_gym_request_seeds(rows))) == 4
    for row in rows:
        metadata = row["responses_create_params"]["metadata"]
        assert metadata["request_label"] == "kept"
        assert json.loads(metadata["extra_body"])["guided_decoding"] == {
            "choice": ["A", "B"]
        }


def test_nemo_gym_inputs_keep_seeds_stable_across_attempt_retries():
    input_sample = cast(
        DatumSpec,
        {"idx": 7, "extra_env_info": {"responses_create_params": {}}},
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        num_generations_per_prompt=3,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    first = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )
    retry = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000002",
    )

    assert first[0][NEMO_GYM_TASK_INDEX_KEY] != retry[0][NEMO_GYM_TASK_INDEX_KEY]
    assert _nemo_gym_request_seeds(first) == _nemo_gym_request_seeds(retry)


def test_nemo_gym_inputs_use_same_seed_identity_for_independent_ab_arms():
    input_sample = cast(
        DatumSpec,
        {"idx": 7, "extra_env_info": {"responses_create_params": {}}},
    )
    off_arm = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        num_generations_per_prompt=3,
        add_seed_per_rollout=True,
        seed_base=42,
    )
    on_arm = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        num_generations_per_prompt=3,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    off_rows = off_arm._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )
    on_rows = on_arm._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )

    assert _nemo_gym_request_seeds(off_rows) == _nemo_gym_request_seeds(on_rows)


def test_nemo_gym_inputs_use_sample_identity_in_request_seed():
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=42,
    )
    group_id = "00000000-0000-4000-8000-000000000001"

    first = impl._build_inputs(
        cast(
            DatumSpec,
            {"idx": 7, "extra_env_info": {"responses_create_params": {}}},
        ),
        rollout_group_id=group_id,
    )
    second = impl._build_inputs(
        cast(
            DatumSpec,
            {"idx": 8, "extra_env_info": {"responses_create_params": {}}},
        ),
        rollout_group_id=group_id,
    )

    assert _nemo_gym_request_seeds(first) != _nemo_gym_request_seeds(second)


@pytest.mark.parametrize("add_seed_per_rollout", [_UNSET, False])
def test_nemo_gym_inputs_leave_caller_seed_unchanged_when_disabled(
    add_seed_per_rollout: object,
) -> None:
    input_sample = cast(
        DatumSpec,
        {
            "extra_env_info": {
                "responses_create_params": {
                    "metadata": {"extra_body": json.dumps({"seed": 91})}
                }
            }
        },
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        num_generations_per_prompt=2,
        add_seed_per_rollout=add_seed_per_rollout,
    )

    rows = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )

    assert _nemo_gym_request_seeds(rows) == [91, 91]


def test_nemo_gym_inputs_create_seed_metadata_when_caller_metadata_is_null():
    input_sample = cast(
        DatumSpec,
        {
            "idx": 7,
            "extra_env_info": {"responses_create_params": {"metadata": None}},
        },
    )
    original = deepcopy(input_sample)
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    rows = impl._build_inputs(
        input_sample,
        rollout_group_id="00000000-0000-4000-8000-000000000001",
    )

    assert _nemo_gym_request_seeds(rows) == [8268443954876357215]
    assert input_sample == original


@pytest.mark.parametrize("flag", [None, 0, 1, "false", "true"])
def test_nemo_gym_inputs_reject_non_boolean_seed_opt_in(flag: object) -> None:
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=flag,
        seed_base=42,
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        impl._build_inputs(
            cast(
                DatumSpec,
                {"idx": 7, "extra_env_info": {"responses_create_params": {}}},
            ),
            rollout_group_id="00000000-0000-4000-8000-000000000001",
        )


@pytest.mark.parametrize(
    ("seed_base", "sample_idx", "error_match"),
    [
        (_UNSET, 7, "nemo_gym_per_rollout_seed_base=None"),
        (None, 7, "nemo_gym_per_rollout_seed_base=None"),
        (True, 7, "nemo_gym_per_rollout_seed_base=True"),
        ("42", 7, "nemo_gym_per_rollout_seed_base='42'"),
        (-1, 7, "nemo_gym_per_rollout_seed_base=-1"),
        (42, None, "DatumSpec.idx=None"),
        (42, True, "DatumSpec.idx=True"),
        (42, "7", "DatumSpec.idx='7'"),
        (42, -1, "DatumSpec.idx=-1"),
    ],
)
def test_nemo_gym_inputs_reject_missing_or_malformed_seed_identity(
    seed_base: object, sample_idx: object, error_match: str
) -> None:
    input_sample = cast(
        DatumSpec,
        {"idx": sample_idx, "extra_env_info": {"responses_create_params": {}}},
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=seed_base,
    )

    with pytest.raises(ValueError, match=error_match):
        impl._build_inputs(
            input_sample,
            rollout_group_id="00000000-0000-4000-8000-000000000001",
        )


@pytest.mark.parametrize(
    ("extra_body", "error_match"),
    [
        ("{not-json", "must be valid JSON"),
        ("[]", "must decode to a JSON object"),
        ("null", "must decode to a JSON object"),
        ({"guided_decoding": {}}, "must be a JSON-encoded object string"),
    ],
)
def test_nemo_gym_inputs_reject_malformed_seed_extra_body(
    extra_body: object, error_match: str
) -> None:
    input_sample = cast(
        DatumSpec,
        {
            "idx": 7,
            "extra_env_info": {
                "responses_create_params": {"metadata": {"extra_body": extra_body}}
            },
        },
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    with pytest.raises(ValueError, match=error_match):
        impl._build_inputs(
            input_sample,
            rollout_group_id="00000000-0000-4000-8000-000000000001",
        )


def test_nemo_gym_inputs_reject_non_object_seed_metadata():
    input_sample = cast(
        DatumSpec,
        {
            "idx": 7,
            "extra_env_info": {"responses_create_params": {"metadata": []}},
        },
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    with pytest.raises(ValueError, match="metadata must be an object"):
        impl._build_inputs(
            input_sample,
            rollout_group_id="00000000-0000-4000-8000-000000000001",
        )


def test_nemo_gym_inputs_reject_caller_seed_collision_when_enabled():
    input_sample = cast(
        DatumSpec,
        {
            "idx": 7,
            "extra_env_info": {
                "responses_create_params": {
                    "metadata": {"extra_body": json.dumps({"seed": 91})}
                }
            },
        },
    )
    impl = _nemo_gym_impl(
        mask_env_flagged_samples=True,
        add_seed_per_rollout=True,
        seed_base=42,
    )

    with pytest.raises(ValueError, match="extra_body.seed must be absent"):
        impl._build_inputs(
            input_sample,
            rollout_group_id="00000000-0000-4000-8000-000000000001",
        )


def _mask_gate_result():
    return {
        "message_log": [
            {
                "role": "assistant",
                "token_ids": [1, 2],
                "generation_logprobs": [0.0, 0.0],
            }
        ],
        "full_result": {
            "reward": 1.0,
            "instance_config": {"mask_sample": True, "other_key": "kept"},
        },
    }


def test_result_to_completion_keeps_mask_flag_when_gate_on():
    completion = _nemo_gym_impl(True)._result_to_completion(_mask_gate_result())
    assert completion.env_extras["instance_config"]["mask_sample"] is True
    assert completion.env_masked is True


def test_result_to_completion_preserves_raw_mask_flag_when_gate_off():
    completion = _nemo_gym_impl(False)._result_to_completion(_mask_gate_result())
    assert completion.env_extras["instance_config"]["mask_sample"] is True
    assert completion.env_extras["instance_config"]["other_key"] == "kept"
    assert completion.env_masked is False


@pytest.mark.parametrize("value", [1, "true", torch.tensor(True)])
def test_result_to_completion_rejects_non_bool_mask_flag(value):
    result = _mask_gate_result()
    result["full_result"]["instance_config"]["mask_sample"] = value

    with pytest.raises(TypeError, match="mask_sample must be a bool"):
        _nemo_gym_impl(True)._result_to_completion(result)


# ---------------------------------------------------------------------------
# Tests for AsyncRolloutManager (native async path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def single_multi_step_calculator_input_sample(rollout_tokenizer):  # noqa: F811
    """Returns a single DatumSpec prompt dict (problem 0) for AsyncRolloutManager tests."""
    problem_text = "(5 + 3) * 2"
    expected_answer = 16.0
    max_steps = 5

    tool_instructions = (
        "You have a calculator tool. To use it, respond with:\n"
        "'[operand1, operand2, operation_name]<call: calculator>'\n"
        "The valid 'operation_name' values are exactly: 'sum', 'diff', 'prod', 'div'.\n"
        "Example: [5, 3, sum]<call: calculator>\n"
        "You will receive the result of your calculation as <result>...</result>\n"
        "Use this result to make the next calculation if needed.\n"
        "IMPORTANT: Only perform one calculation step (one tool call) before waiting for a result and making a new tool call.\n"
        "IMPORTANT: Do not perform any other calculations or operations aside from the tool call and result. Doing so will result in failure.\n"
        "To give the final answer, just output the number. numbers inside of <result> don't count, so output just the final number yourself outside of this.\n"
        "Example full output: [2, 4, sum]<call: calculator>\n<result>6.0</result>\n[6, 6, diff]<call: calculator>\n<result>0.0</result> 0\n(note how you have to output the final 0 outside of the tags)"
        "------\n"
        f"Solve: {problem_text}"
    )

    initial_prompt_content = rollout_tokenizer.apply_chat_template(
        [{"role": "user", "content": tool_instructions}],
        tokenize=False,
        add_system_prompt=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    tokenized_prompt = rollout_tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    message_log = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]
    metadata = MultiStepCalcMetadata(
        problem=problem_text,
        expected_final_answer=expected_answer,
        max_steps=max_steps,
        current_step=0,
    )
    return {
        "message_log": message_log,
        "extra_env_info": metadata,
        "task_name": "multi_step_calculator_game",
        "stop_strings": ["<call: calculator>"],
        "idx": 0,
        "loss_multiplier": 1.0,
    }


@pytest.mark.vllm
def test_async_rollout_manager(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Standalone test for AsyncRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - rollout_metrics has the expected keys with correct types
    - completions hold independent (not aliased) message_log objects
    """
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )

    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == input_sample["idx"]

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) >= 4, (
            f"Completion {i}: expected >= 4 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant content
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert last_assistant["content"].strip() == "16", (
            f"Completion {i}: last assistant content {last_assistant['content']!r} != '16'"
        )

        # 3. reward
        assert completion.reward == 1.0, (
            f"Completion {i}: reward {completion.reward} != 1.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.vllm
def test_async_rollout_manager_truncation(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Small max_seq_len forces truncation and truncation_rate=1.0."""
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 290
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    vllm_generation.prepare_for_generation()
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    assert len(record.completions) == num_generations
    assert all(c.truncated for c in record.completions)
    assert record.rollout_metrics["truncation_rate"] == 1.0
    assert record.rollout_metrics["natural_termination_rate"] == 0.0


@pytest.mark.vllm
def test_async_rollout_manager_matches_original(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,
):
    """Comparison test: AsyncRolloutManager output is structurally equivalent to the original.

    Calls run_async_multi_turn_rollout with a batch of N identical prompts,
    then calls AsyncRolloutManager with 1 prompt and N generations.
    Asserts that both produce N results with matching message-log depth, rewards,
    and rollout_metrics numeric values.

    TODO: remove this test together with run_async_multi_turn_rollout when the legacy path is deleted.
    """
    vllm_generation, tokenizer, task_to_env, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample
    num_generations = 2
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    # Build a batch of N identical prompts for the original function
    batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_sample["message_log"]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_sample["extra_env_info"]) for _ in range(num_generations)
            ],
            "task_name": [input_sample["task_name"]] * num_generations,
            "stop_strings": [input_sample["stop_strings"]] * num_generations,
            "idx": list(range(num_generations)),
            "loss_multiplier": [1.0] * num_generations,
        }
    )

    vllm_generation.prepare_for_generation()
    original_batch, original_metrics = run_async_multi_turn_rollout(
        policy_generation=vllm_generation,
        input_batch=batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
    )

    manager = RolloutManager(
        use_nemo_gym=False,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
    )
    record = asyncio.run(manager.run_rollout(input_sample))
    vllm_generation.finish_generation()

    # Both should produce N results
    assert len(original_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant content matches
        def _last_assistant_content(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("content", "")
            return ""

        orig_last = _last_assistant_content(orig_msg_log)
        new_last = _last_assistant_content(new_msg_log)
        assert orig_last == new_last, (
            f"Completion {i}: last assistant content mismatch\n"
            f"  original:  {orig_last!r}\n"
            f"  manager:   {new_last!r}"
        )

        # 3. reward matches
        orig_reward = original_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and histogram fields are excluded).
    # The new impl emits slash-style keys (X/mean, X/max, X/min) via calculate_single_metric;
    # translate the legacy prefix-style keys before comparing.
    def _translate_legacy_key(key: str) -> str:
        if key == "avg_turns_per_sample":
            return "turns_per_sample/mean"
        if key == "max_turns_reached_rate":
            return key
        # Keys already in slash-style (e.g. turns_per_sample/p95, max_gen_tokens_per_turn/max)
        # are new-style and should not be re-translated by the prefix-strip logic.
        if "/" in key:
            return key
        for prefix, suffix in (("mean_", "/mean"), ("max_", "/max"), ("min_", "/min")):
            if key.startswith(prefix):
                return f"{key[len(prefix) :]}{suffix}"
        return key

    new_metrics = record.rollout_metrics
    for key in original_metrics.keys():
        if key.startswith("timing/") or key.startswith("histogram/"):
            continue

        new_key = _translate_legacy_key(key)
        assert new_key in new_metrics, (
            f"rollout_metrics[{new_key!r}] missing from manager"
        )

        orig_val = original_metrics[key]
        new_val = new_metrics[new_key]

        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )


# ---------------------------------------------------------------------------
# Tests for AsyncNemoGymRolloutManager
# ---------------------------------------------------------------------------


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Standalone test for AsyncNemoGymRolloutManager.

    Given 1 prompt with num_generations_per_prompt=N, asserts:
    - output is a PromptGroupRecord with N Completion objects
    - each Completion has a reward (float) and a non-empty message_log
    - completions hold independent message_log objects

    If the result here does not match, please check the following:
    1. Test data changed: re-run test_nemo_gym_sanity (tests/unit/environments/test_nemo_gym.py)
       and use _write_actual_test_data output to refresh test_nemo_gym_sanity.json.
    2. Logic changed: inspect recent changes to AsyncNemoGymRolloutManager or the gym env.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    # Use only the first prompt
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }
    num_generations = 2

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    assert isinstance(record, PromptGroupRecord)
    assert len(record.completions) == num_generations, (
        f"Expected {num_generations} completions, got {len(record.completions)}"
    )
    assert record.prompt_idx == 0

    for i, completion in enumerate(record.completions):
        assert isinstance(completion, Completion)

        # 1. message_log length
        assert len(completion.message_log) == 2, (
            f"Completion {i}: expected 2 messages, got {len(completion.message_log)}"
        )

        # 2. last assistant token_ids
        last_assistant = next(
            (m for m in reversed(completion.message_log) if m["role"] == "assistant"),
            None,
        )
        assert last_assistant is not None, f"Completion {i}: no assistant message found"
        assert torch.equal(
            last_assistant["token_ids"],
            torch.tensor([151667, 198, 32313, 11, 1077]),
        ), (
            f"Completion {i}: last assistant token_ids {last_assistant['token_ids'].tolist()} "
            f"!= [151667, 198, 32313, 11, 1077]"
        )

        # 3. reward
        assert completion.reward == 0.0, (
            f"Completion {i}: reward {completion.reward} != 0.0"
        )

    # completions must be independent objects
    assert record.completions[0].message_log is not record.completions[1].message_log


@pytest.mark.nemo_gym
def test_async_nemo_gym_rollout_manager_matches_original(
    nemo_gym,  # noqa: F811
    nemo_gym_vllm_generation,  # noqa: F811
    nemo_gym_sanity_test_data,  # noqa: F811
    nemo_gym_tokenizer,  # noqa: F811
):
    """Comparison test: AsyncNemoGymRolloutManager output is structurally equivalent to the original.

    Calls run_async_nemo_gym_rollout with a batch of N identical rows,
    then calls AsyncNemoGymRolloutManager with 1 prompt, N generations.
    Asserts that both produce N results and rewards are in the same numeric domain.

    TODO: remove this test together with run_async_nemo_gym_rollout when the legacy path is deleted.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for data in nemo_gym_sanity_test_data["input"]:
            f.write(json.dumps(data) + "\n")
        data_path = f.name

    dataset = NemoGymDataset(data_path)
    examples = [
        nemo_gym_data_processor(dataset.dataset[idx], None, None, None, idx)
        for idx in range(len(dataset.dataset))
    ]
    input_batch: BatchedDataDict[DatumSpec] = rl_collate_fn(examples)

    num_generations = 2
    single_prompt = {
        "message_log": input_batch["message_log"][0],
        "extra_env_info": input_batch["extra_env_info"][0],
        "task_name": "nemo_gym",
        "idx": 0,
        "loss_multiplier": float(input_batch["loss_multiplier"][0]),
    }

    # Build a batch of N identical rows for the original function
    repeated_batch = BatchedDataDict(
        {
            "message_log": [
                deepcopy(input_batch["message_log"][0]) for _ in range(num_generations)
            ],
            "extra_env_info": [
                deepcopy(input_batch["extra_env_info"][0])
                for _ in range(num_generations)
            ],
            "loss_multiplier": input_batch["loss_multiplier"][0:1].repeat(
                num_generations
            ),
            "idx": list(range(num_generations)),
            "task_name": ["nemo_gym"] * num_generations,
        }
    )

    async def _collect_original_results():
        return [
            result
            async for result in run_async_nemo_gym_rollout(
                policy_generation=nemo_gym_vllm_generation,
                input_batch=repeated_batch,
                tokenizer=nemo_gym_tokenizer,
                task_to_env={"nemo_gym": nemo_gym},
                generation_config=nemo_gym_vllm_generation.cfg,
                num_generations=num_generations,
                log_full_result_tables=False,
                max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
                max_rollout_turns=None,
            )
        ]

    original_results = asyncio.run(_collect_original_results())
    assert len(original_results) == 1
    original_result = original_results[0]

    manager = RolloutManager(
        use_nemo_gym=True,
        tokenizer=nemo_gym_tokenizer,
        task_to_env={"nemo_gym": nemo_gym},
        num_generations_per_prompt=num_generations,
        max_seq_len=nemo_gym_vllm_generation.cfg["vllm_cfg"]["max_model_len"],
        generation_config=nemo_gym_vllm_generation.cfg,
    )
    record = asyncio.run(manager.run_rollout(single_prompt))

    # Both should produce N completions
    assert len(original_result.final_batch["message_log"]) == num_generations
    assert len(record.completions) == num_generations

    for i in range(num_generations):
        orig_msg_log = original_result.final_batch["message_log"][i]
        new_msg_log = record.completions[i].message_log

        # 1. message_log length matches
        assert len(orig_msg_log) == len(new_msg_log), (
            f"Completion {i}: message_log length {len(new_msg_log)} != original {len(orig_msg_log)}"
        )

        # 2. last assistant token_ids match
        def _last_assistant_token_ids(msg_log):
            for m in reversed(msg_log):
                if m["role"] == "assistant":
                    return m.get("token_ids")
            return None

        orig_token_ids = _last_assistant_token_ids(orig_msg_log)
        new_token_ids = _last_assistant_token_ids(new_msg_log)
        assert orig_token_ids is not None, (
            f"Completion {i}: no assistant message in original"
        )
        assert new_token_ids is not None, (
            f"Completion {i}: no assistant message in manager"
        )
        assert torch.equal(orig_token_ids, new_token_ids), (
            f"Completion {i}: last assistant token_ids mismatch\n"
            f"  original:  {orig_token_ids.tolist()}\n"
            f"  manager:   {new_token_ids.tolist()}"
        )

        # 3. reward matches
        orig_reward = original_result.final_batch["total_reward"][i].item()
        new_reward = record.completions[i].reward
        assert orig_reward == new_reward, (
            f"Completion {i}: reward mismatch — original {orig_reward}, manager {new_reward}"
        )

    # 4. rollout_metrics numeric values match (timing and Table fields are excluded)
    orig_metrics = original_result.rollout_metrics
    new_metrics = record.rollout_metrics
    for key in orig_metrics.keys():
        # Skip timing and full_result fields
        if key.startswith("timing/") or key.endswith("/full_result"):
            continue

        # Check that the key is present in the new metrics
        assert key in new_metrics, f"rollout_metrics[{key!r}] missing from manager"

        orig_val = orig_metrics[key]
        new_val = new_metrics[key]

        # Skip non-numeric fields
        assert type(orig_val) == type(new_val), (
            f"rollout_metrics[{key!r}] type mismatch: {type(orig_val)} != {type(new_val)}"
        )
        if not isinstance(orig_val, (bool, int, float)):
            continue

        # Check equal
        assert orig_val == pytest.approx(new_val), (
            f"rollout_metrics[{key!r}] mismatch — original {orig_val}, manager {new_val}"
        )
