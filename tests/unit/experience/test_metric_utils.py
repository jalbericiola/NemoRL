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

"""Rollout metrics must never be able to fail a rollout."""

from __future__ import annotations

import math

import pytest

from nemo_rl.experience.interfaces import (
    NEMO_GYM_RESERVED_KEY_PREFIX,
    NEMO_GYM_TASK_INDEX_KEY,
    Completion,
)
from nemo_rl.experience.metric_utils import calculate_single_metric
from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl

# A 63-bit id, the width NeMo Gym task indices are folded into. Repeated across a
# cohort the range is zero, and at this magnitude float64 spacing swallows the
# widening numpy applies, so 64 equal bins collapse.
COHORT_ID = (1 << 62) + 12345


class TestCalculateSingleMetric:
    def test_reports_the_usual_statistics(self):
        metrics = calculate_single_metric([1.0, 2.0, 3.0], 3, "reward")

        assert metrics["reward/mean"] == pytest.approx(2.0)
        assert metrics["reward/max"] == 3.0
        assert metrics["reward/min"] == 1.0
        assert metrics["reward/median"] == 2.0
        assert "reward/histogram" in metrics

    def test_means_over_batch_size_not_sample_count(self):
        metrics = calculate_single_metric([1.0, 1.0], 4, "reward")

        assert metrics["reward/mean"] == pytest.approx(0.5)

    def test_stddev_is_nan_for_a_single_value(self):
        metrics = calculate_single_metric([7.0], 1, "reward")

        assert math.isnan(metrics["reward/stddev"])

    def test_survives_values_that_cannot_be_binned(self):
        metrics = calculate_single_metric([float(COHORT_ID)] * 16, 16, "task_index")

        assert metrics["task_index/max"] == float(COHORT_ID)
        assert "task_index/histogram" not in metrics

    def test_still_bins_a_repeated_small_value(self):
        metrics = calculate_single_metric([3.0] * 16, 16, "turns")

        assert "turns/histogram" in metrics


def _completion(env_extras: dict) -> Completion:
    return Completion(
        message_log=[
            {"role": "user", "token_ids": [1, 2]},
            {"role": "assistant", "token_ids": [3, 4, 5]},
        ],
        env_extras=env_extras,
        truncated=False,
        reward=1.0,
    )


class TestAgentMetricsSkipReservedKeys:
    """Gym's ``_ng_`` fields are identifiers, so they are not charted."""

    def test_a_cohort_id_does_not_fail_the_rollout(self):
        completions = [
            _completion({NEMO_GYM_TASK_INDEX_KEY: COHORT_ID, "score": 0.5})
            for _ in range(16)
        ]

        metrics = AsyncNemoGymRolloutImpl._compute_rollout_metrics(
            None, completions, "agent"
        )

        assert metrics["agent/score/mean"] == pytest.approx(0.5)

    def test_no_reserved_key_is_charted(self):
        completions = [
            _completion(
                {
                    NEMO_GYM_TASK_INDEX_KEY: COHORT_ID,
                    "_ng_rollout_index": idx,
                    "_ng_attempt_index": 0,
                    "score": 0.5,
                }
            )
            for idx in range(4)
        ]

        metrics = AsyncNemoGymRolloutImpl._compute_rollout_metrics(
            None, completions, "agent"
        )

        assert not [
            key
            for key in metrics
            if f"/{NEMO_GYM_RESERVED_KEY_PREFIX}" in key
            or key.startswith(NEMO_GYM_RESERVED_KEY_PREFIX)
        ]

    def test_ordinary_agent_keys_are_still_charted(self):
        completions = [_completion({"score": float(i)}) for i in range(4)]

        metrics = AsyncNemoGymRolloutImpl._compute_rollout_metrics(
            None, completions, "agent"
        )

        assert metrics["agent/score/max"] == 3.0
        assert "agent/score/histogram" in metrics
