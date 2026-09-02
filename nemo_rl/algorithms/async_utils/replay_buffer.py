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

import asyncio
import gc
import hashlib
import json
import math
import statistics
import threading as _threading
import uuid
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import ray
import torch
from tensordict import TensorDictBase

from nemo_rl.algorithms.async_utils.interfaces import ReplayBufferProtocol
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    INVALID_TOOL_CALL_TOKEN_MASK,
    MALFORMED_THINKING_MESSAGE_COUNT,
    MALFORMED_THINKING_TOKEN_MASK,
    NEMO_GYM_TASK_INDEX_KEY,
    NEXT_NEMO_GYM_TASK_INDEX_KEY,
    RESPONSE_TOKEN_LENGTHS,
    ROLLOUT_TRUNCATED,
    PromptGroupRecord,
)
from nemo_rl.experience.payload import pack_payload, record_to_train_batch
from nemo_rl.utils.r3_trace import trace_rollout_payload


RolloutScalarMetrics = dict[str, int | float]
# Bump this domain whenever the code-level meaning or ordering of the hashed
# effort, penalty, mask, or message-advantage stages changes.
_ROLLOUT_REWARD_SEMANTICS_DOMAIN = "single-controller-rollout-reward-v1"
_SHA256_FINGERPRINT_PREFIX = "sha256:"
_REQUIRED_ROLLOUT_RECEIPT_KEYS = {
    "cohort/samples",
    "cohort/generated_tokens",
    "cohort/total_tokens",
    "cohort/raw_environment_reward_sum",
    "cohort/pre_penalty_reward_sum",
    "cohort/effort_low_sample_count",
    "cohort/effort_reward_delta_sum",
    "cohort/env_masked_sample_count",
    "cohort/post_penalty_reward_sum",
    "cohort/duplicated_reasoning_count",
    "cohort/empty_final_answer_count",
    "cohort/unwanted_token_count",
    "cohort/malformed_think_tag_count",
    "cohort/rollout_started_at_s",
    "cohort/rollout_finished_at_s",
    "timing/rollout/total",
}


def build_rollout_reward_semantics_fingerprint(
    *,
    use_nemo_gym: bool,
    effort_config: Mapping[str, Any] | None,
    reward_penalty_config: Mapping[str, Any],
    mask_env_flagged_samples: bool,
    thinking_tags: Iterable[str],
    invalid_tool_call_advantage: float | None,
    malformed_thinking_advantage: float | None,
) -> str:
    """Return a canonical fingerprint for semantics embodied by replay rows.

    Buffered rewards and masks are computed before the group enters replay. A
    resumed run must therefore use the same rollout-time configuration as the
    run that wrote the checkpoint; otherwise restored and newly generated rows
    in one optimizer step can carry different semantics.

    Args:
        use_nemo_gym: Whether rewards come from the NeMo-Gym rollout path.
        effort_config: Normalized NeMo-Gym effort-shaping configuration, or None.
        reward_penalty_config: Normalized rollout reward-penalty configuration.
        mask_env_flagged_samples: Whether environment flags zero sample masks.
        thinking_tags: Tags used for malformed-thinking detection.
        invalid_tool_call_advantage: Invalid-tool message advantage override.
        malformed_thinking_advantage: Malformed-thinking advantage override.

    Returns:
        A domain-separated SHA-256 fingerprint.

    Raises:
        TypeError: If the supplied configuration cannot be encoded as canonical
            JSON.
        ValueError: If a float is non-finite.
    """
    payload = {
        "domain": _ROLLOUT_REWARD_SEMANTICS_DOMAIN,
        "use_nemo_gym": use_nemo_gym,
        "effort_config": effort_config,
        "reward_penalty_config": reward_penalty_config,
        "mask_env_flagged_samples": mask_env_flagged_samples,
        "thinking_tags": list(thinking_tags),
        "message_advantage_config": {
            "invalid_tool_call_advantage": invalid_tool_call_advantage,
            "malformed_thinking_advantage": malformed_thinking_advantage,
        },
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise type(error)(
            "rollout reward-semantics configuration must be canonical-JSON "
            f"serializable: {error}"
        ) from error
    return _SHA256_FINGERPRINT_PREFIX + hashlib.sha256(encoded).hexdigest()


def _validate_rollout_reward_semantics_fingerprint(fingerprint: str) -> None:
    """Reject absent or malformed fingerprints before replay is usable."""
    if not isinstance(fingerprint, str):
        raise TypeError(
            "rollout_reward_semantics_fingerprint must be a string, got "
            f"{type(fingerprint).__name__}"
        )
    if not fingerprint.startswith(_SHA256_FINGERPRINT_PREFIX):
        raise ValueError(
            "rollout_reward_semantics_fingerprint must use the 'sha256:' prefix"
        )
    digest = fingerprint.removeprefix(_SHA256_FINGERPRINT_PREFIX)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(
            "rollout_reward_semantics_fingerprint must contain a 64-character "
            "lowercase hexadecimal SHA-256 digest"
        )


def _validate_checkpoint_meta_fields(
    *,
    meta: KVBatchMeta,
    group_id: str,
) -> list[str]:
    """Validate and return a checkpoint group's declared field names."""
    if not isinstance(meta, KVBatchMeta):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} meta must be "
            f"KVBatchMeta, got {type(meta).__name__}"
        )
    if meta.fields is None:
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} has meta.fields=None"
        )
    if not isinstance(meta.fields, list):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} meta.fields must be a list"
        )
    meta_fields = list(meta.fields)
    if any(not isinstance(field, str) for field in meta_fields):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} has non-string meta.fields"
        )
    if len(meta_fields) != len(set(meta_fields)):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} has duplicate meta.fields"
        )
    return meta_fields


def _validate_checkpoint_group_fields(
    *,
    meta: KVBatchMeta,
    fields_data: Any,
    group_id: str,
) -> None:
    """Require checkpoint payload keys and batch size to match its metadata."""
    meta_fields = _validate_checkpoint_meta_fields(meta=meta, group_id=group_id)

    if not isinstance(fields_data, TensorDictBase):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} fields_data must be a "
            f"TensorDict, got {type(fields_data).__name__}"
        )
    expected_batch_size = (len(meta.sample_ids),)
    if tuple(fields_data.batch_size) != expected_batch_size:
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} fields_data batch_size "
            f"mismatch: checkpoint={tuple(fields_data.batch_size)}, "
            f"expected={expected_batch_size}"
        )
    # Top-level iteration intentionally includes NonTensorData / NonTensorStack
    # leaves, unlike keys(include_nested=True, leaves_only=True).
    payload_fields = list(fields_data.keys())
    if any(not isinstance(field, str) for field in payload_fields):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} has non-string "
            "fields_data keys"
        )
    if len(payload_fields) != len(set(payload_fields)):
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} has duplicate "
            "fields_data keys"
        )

    declared = set(meta_fields)
    materialized = set(payload_fields)
    if declared != materialized:
        raise ValueError(
            f"Replay buffer checkpoint group {group_id!r} field mismatch: "
            f"missing_from_fields_data={sorted(declared - materialized)}, "
            f"unexpected_in_fields_data={sorted(materialized - declared)}"
        )


def _scalar_rollout_metrics(metrics: Mapping[str, Any]) -> RolloutScalarMetrics:
    """Copy finite scalar rollout metrics suitable for checkpointing and W&B."""
    scalars: RolloutScalarMetrics = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError(
                "PromptGroupRecord.rollout_metrics keys must be strings; "
                f"got {type(key).__name__}"
            )
        # Tables and histograms remain rollout-local. The exact scalar summary is
        # what an optimizer step can reduce without serializing large result tables
        # into replay-buffer checkpoints.
        if isinstance(value, bool):
            scalars[key] = int(value)
        elif isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                if key in _REQUIRED_ROLLOUT_RECEIPT_KEYS:
                    raise ValueError(
                        f"required rollout metric {key!r} must be finite, got {value!r}"
                    )
                continue
            scalars[key] = value
    return scalars


def _validate_rollout_metric_receipt(
    metrics: RolloutScalarMetrics,
    *,
    expected_samples: Optional[int] = None,
) -> None:
    """Require the exact counters needed for a comparable optimizer-step history."""
    missing = _REQUIRED_ROLLOUT_RECEIPT_KEYS - set(metrics)
    if missing:
        raise ValueError(
            "rollout metric receipt is incomplete; missing exact cohort keys "
            f"{sorted(missing)}"
        )
    integer_keys = {
        "cohort/samples",
        "cohort/generated_tokens",
        "cohort/total_tokens",
        "cohort/effort_low_sample_count",
        "cohort/env_masked_sample_count",
        "cohort/duplicated_reasoning_count",
        "cohort/empty_final_answer_count",
        "cohort/unwanted_token_count",
        "cohort/malformed_think_tag_count",
    }
    for key in integer_keys:
        value = float(metrics[key])
        if value < 0 or not value.is_integer():
            raise ValueError(
                f"rollout metric {key!r} must be a nonnegative integer, got {value}"
            )
    samples = int(metrics["cohort/samples"])
    if samples < 1:
        raise ValueError("rollout metric 'cohort/samples' must be positive")
    if expected_samples is not None and samples != expected_samples:
        raise ValueError(
            "rollout metric receipt sample count does not match its payload: "
            f"metrics={samples}, payload={expected_samples}"
        )
    if metrics["cohort/total_tokens"] < metrics["cohort/generated_tokens"]:
        raise ValueError(
            "rollout metric 'cohort/total_tokens' cannot be smaller than "
            "'cohort/generated_tokens'"
        )
    if metrics["cohort/effort_low_sample_count"] > samples:
        raise ValueError(
            "rollout effort low-sample count exceeds cohort/samples: "
            f"{metrics['cohort/effort_low_sample_count']} > {samples}"
        )
    if metrics["cohort/env_masked_sample_count"] > samples:
        raise ValueError(
            "rollout env-masked sample count exceeds cohort/samples: "
            f"{metrics['cohort/env_masked_sample_count']} > {samples}"
        )
    observed_effort_delta = float(metrics["cohort/pre_penalty_reward_sum"]) - float(
        metrics["cohort/raw_environment_reward_sum"]
    )
    if not math.isclose(
        observed_effort_delta,
        float(metrics["cohort/effort_reward_delta_sum"]),
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise ValueError(
            "rollout effort delta does not match raw/pre-penalty reward boundaries"
        )
    for key in (
        "cohort/duplicated_reasoning_count",
        "cohort/empty_final_answer_count",
        "cohort/unwanted_token_count",
        "cohort/malformed_think_tag_count",
    ):
        if metrics[key] > samples:
            raise ValueError(
                f"rollout penalty metric {key!r} exceeds cohort/samples: "
                f"{metrics[key]} > {samples}"
            )
    if metrics["timing/rollout/total"] < 0:
        raise ValueError("rollout metric 'timing/rollout/total' must be nonnegative")
    started_at_s = metrics["cohort/rollout_started_at_s"]
    finished_at_s = metrics["cohort/rollout_finished_at_s"]
    if started_at_s < 0 or finished_at_s < started_at_s:
        raise ValueError(
            "rollout interval must satisfy 0 <= cohort/rollout_started_at_s <= "
            "cohort/rollout_finished_at_s"
        )


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
class ReplayBufferImpl(ReplayBufferProtocol):
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    the algorithm's ``num_generations_per_prompt`` setting.
    """

    def __init__(
        self,
        max_size: int,
        drop_incomplete_targets_on_restore: bool,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        # True discards partial restored rows. The dataloader is not rewound,
        # so replacement rollouts come from subsequent prompts.
        self._drop_incomplete_targets_on_restore = drop_incomplete_targets_on_restore
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    @staticmethod
    def _rollout_metrics_turn_count_for_diagnostics(
        rm: dict[str, Any],
    ) -> Optional[float]:
        """One scalar turn-depth per buffered trajectory for starvation diagnostics.

        Supports sync multi-turn rollouts (`max_turns_per_sample` / `avg_turns_per_sample`)
        and NeMo Gym (`turns_per_sample/max` / `turns_per_sample/mean`).
        """
        if "max_turns_per_sample" in rm:
            return float(rm["max_turns_per_sample"])
        if "avg_turns_per_sample" in rm:
            return float(rm["avg_turns_per_sample"])
        if "turns_per_sample/max" in rm:
            return float(rm["turns_per_sample/max"])
        if "turns_per_sample/mean" in rm:
            return float(rm["turns_per_sample/mean"])
        return None

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.add: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            # Do not advance last_target_weight_already_generated here. A target
            # is only safe to skip once training consumes a complete batch for it.
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        info: dict[str, Any] = {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }
        if self.trajectories:
            durations = []
            max_gen_tokens_per_turn_list = []
            turn_counts_list = []
            for t in self.trajectories:
                rm = t.get("rollout_metrics", {})
                if "trajectory_duration_s" in rm:
                    durations.append(rm["trajectory_duration_s"])
                if "max_gen_tokens_per_turn/max" in rm:
                    max_gen_tokens_per_turn_list.append(
                        rm["max_gen_tokens_per_turn/max"]
                    )
                elif "max_gen_tokens_per_turn" in rm:
                    max_gen_tokens_per_turn_list.append(rm["max_gen_tokens_per_turn"])
                tc = self._rollout_metrics_turn_count_for_diagnostics(rm)
                if tc is not None:
                    turn_counts_list.append(tc)

            def _pct(values: list[float], p: float) -> float:
                if not values:
                    return 0.0
                sorted_v = sorted(values)
                idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
                return float(sorted_v[idx])

            info["starvation_diagnostics"] = {
                "trajectory_duration_s": {
                    "mean": sum(durations) / len(durations) if durations else 0,
                    "median": statistics.median(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "p95": _pct(durations, 95),
                },
                "max_gen_tokens_per_turn_in_buffer": {
                    "mean": sum(max_gen_tokens_per_turn_list)
                    / len(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "median": statistics.median(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "max": max(max_gen_tokens_per_turn_list)
                    if max_gen_tokens_per_turn_list
                    else 0,
                    "p95": _pct(max_gen_tokens_per_turn_list, 95),
                },
                "turns_per_sample_in_buffer": {
                    "mean": sum(turn_counts_list) / len(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "median": statistics.median(turn_counts_list)
                    if turn_counts_list
                    else 0,
                    "max": max(turn_counts_list) if turn_counts_list else 0,
                    "p95": _pct(turn_counts_list, 95),
                },
                "num_trajectories_sampled": len(self.trajectories),
            }
        return info

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def _remove_indices(self, indices: Iterable[int]) -> None:
        """Remove trajectories at the given indices."""
        for idx in sorted(indices, reverse=True):
            self.trajectory_versions.pop(idx)
            self.target_weight_versions.pop(idx)
            self.trajectories.pop(idx)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # For debugging: check for unexpected old trajectories
            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Evict old trajectories that are beyond the age window. This can
            # happen after checkpoint restore when old trajectories remain.
            old_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if old_indices:
                print(
                    f"   Evicting {len(old_indices)} stale trajectories "
                    f"(version < {min_valid_version})"
                )
                self._remove_indices(old_indices)
                total_trajectories = len(self.trajectories)

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            # Remove selected items in reverse order to maintain correct indices
            sampled_items = [self.trajectories[i] for i in selected]
            self._remove_indices(selected)

            old_last_target = self.last_target_weight_already_generated
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated,
                current_weight_version,
            )
            if self.last_target_weight_already_generated > old_last_target:
                print(
                    "Advanced last_target_weight_already_generated: "
                    f"{old_last_target} -> "
                    f"{self.last_target_weight_already_generated} "
                    f"(consumed batch for step {current_weight_version})"
                )

            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        with self._lock:
            return {
                "trajectories": list(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "last_target_weight_already_generated": (
                    self.last_target_weight_already_generated
                ),
                "max_size": self.max_size,
            }

    def save_to_path(self, path: str) -> int:
        """Serialize inside the actor without materializing the buffer on the driver."""
        state = self.state_dict()
        torch.save(state, path)
        num_trajectories = len(state["trajectories"])
        del state
        gc.collect()
        return num_trajectories

    def load_from_path(
        self,
        path: str,
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> dict[str, int]:
        """Restore inside the actor and return only compact coordination metadata."""
        state = torch.load(path, weights_only=False)
        saved_task_indices = [
            int(trajectory[NEMO_GYM_TASK_INDEX_KEY])
            for trajectory in state.get("trajectories", [])
            if trajectory.get(NEMO_GYM_TASK_INDEX_KEY) is not None
        ]
        next_task_index = max(saved_task_indices, default=-1) + 1
        num_trajectories = len(state["trajectories"])
        self.load_state_dict(
            state,
            num_prompts_per_step=num_prompts_per_step,
            current_training_step=current_training_step,
            max_age_steps=max_age_steps,
        )
        del state
        gc.collect()
        return {
            "num_trajectories": num_trajectories,
            NEXT_NEMO_GYM_TASK_INDEX_KEY: next_task_index,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> None:
        """Restore replay buffer state from a checkpoint.

        Args:
            state: State returned by ``state_dict``.
            num_prompts_per_step: Number of prompt groups required for one
                training step. When provided, incomplete target steps can be
                removed or prepared for gap filling.
            current_training_step: Step being resumed. When provided with
                ``num_prompts_per_step``, past target steps are dropped and
                incomplete current/future target steps are kept for gap filling.
            max_age_steps: Maximum allowed age for restored trajectories. When
                provided, stale trajectories are removed during restore.

        Raises:
            ValueError: If the checkpoint is missing required fields or has
                inconsistent parallel list lengths.
        """
        with self._lock:
            required_keys = {
                "trajectories",
                "trajectory_versions",
                "target_weight_versions",
                "last_target_weight_already_generated",
            }
            missing_keys = required_keys - set(state)
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            trajectories = list(state["trajectories"])
            trajectory_versions = list(state["trajectory_versions"])
            target_weight_versions = list(state["target_weight_versions"])
            if not (
                len(trajectories)
                == len(trajectory_versions)
                == len(target_weight_versions)
            ):
                raise ValueError(
                    "Checkpoint has inconsistent replay buffer lengths: "
                    f"trajectories={len(trajectories)}, "
                    f"trajectory_versions={len(trajectory_versions)}, "
                    f"target_weight_versions={len(target_weight_versions)}"
                )

            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    "ReplayBuffer max_size changed: "
                    f"checkpoint={state['max_size']}, current={self.max_size}. "
                    "Using current config value."
                )

            self.trajectories = trajectories
            self.trajectory_versions = trajectory_versions
            self.target_weight_versions = target_weight_versions
            self.last_target_weight_already_generated = state[
                "last_target_weight_already_generated"
            ]

            # Filter stale rows before checking target completeness. Otherwise a
            # target can look complete, lose stale rows, and remain partially
            # restored even when incomplete targets should be dropped.
            if max_age_steps is not None and self.trajectories:
                self._remove_stale_trajectories(max_age_steps)

            if current_training_step is not None and num_prompts_per_step is not None:
                self._prepare_for_training_step(
                    current_step=current_training_step,
                    num_prompts_per_step=num_prompts_per_step,
                )
            elif num_prompts_per_step is not None and self.trajectories:
                self._remove_incomplete_target_steps(num_prompts_per_step)

            self._truncate_to_max_size(current_training_step)

            print(
                f"ReplayBuffer restored: {len(self.trajectories)} trajectories, "
                "last_target_weight_already_generated="
                f"{self.last_target_weight_already_generated}"
            )

    def _prepare_for_training_step(
        self, current_step: int, num_prompts_per_step: int
    ) -> None:
        """Prepare restored state so training can resume at ``current_step``."""
        print(f"   Preparing replay buffer for training step {current_step}...")

        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target >= current_step
        ]

        if len(indices_to_keep) < original_count:
            removed_past = original_count - len(indices_to_keep)
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [
                self.trajectory_versions[i] for i in indices_to_keep
            ]
            self.target_weight_versions = [
                self.target_weight_versions[i] for i in indices_to_keep
            ]
            print(
                f"   Removed {removed_past} trajectories for past steps "
                f"(target < {current_step})"
            )

        if not self.trajectories:
            self.last_target_weight_already_generated = current_step - 1
            print(
                "   No restored trajectories remain; collector will generate "
                f"from step {current_step}"
            )
            return

        target_counts = Counter(self.target_weight_versions)
        complete_targets = {
            target
            for target, count in target_counts.items()
            if count >= num_prompts_per_step
        }
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }

        print(
            "   Complete targets: "
            f"{sorted(complete_targets) if complete_targets else 'none'}"
        )
        if incomplete_targets and self._drop_incomplete_targets_on_restore:
            print(
                "   Dropping incomplete restored targets; replacements will use "
                "subsequent prompts: "
                + ", ".join(
                    f"{target}={target_counts[target]}/{num_prompts_per_step}"
                    for target in sorted(incomplete_targets)
                )
            )
            indices_to_keep = [
                i
                for i, target in enumerate(self.target_weight_versions)
                if target not in incomplete_targets
            ]
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [
                self.trajectory_versions[i] for i in indices_to_keep
            ]
            self.target_weight_versions = [
                self.target_weight_versions[i] for i in indices_to_keep
            ]
        else:
            for target in sorted(incomplete_targets):
                print(
                    f"   Incomplete target {target}: "
                    f"{target_counts[target]}/{num_prompts_per_step}"
                )

        # Let the collector ask each target from current_step onward how many
        # trajectories are still needed, so incomplete restored batches can be
        # gap-filled and complete batches can be skipped.
        self.last_target_weight_already_generated = current_step - 1

    @staticmethod
    def _is_valid_for_target(
        trajectory_version: int, target_step: int, max_age_steps: int | None
    ) -> bool:
        if max_age_steps is None:
            return True
        min_valid_version = max(0, target_step - max_age_steps)
        return min_valid_version <= trajectory_version <= target_step

    def _remove_stale_trajectories(self, max_age_steps: int) -> None:
        """Remove restored trajectories that are stale for their target step.

        Must be called while holding ``self._lock``.
        """
        indices_to_remove = [
            i
            for i, (trajectory_version, target) in enumerate(
                zip(self.trajectory_versions, self.target_weight_versions)
            )
            if not self._is_valid_for_target(trajectory_version, target, max_age_steps)
        ]
        if not indices_to_remove:
            return

        print(
            f"   Removing {len(indices_to_remove)} stale restored trajectories "
            f"(max_age_steps={max_age_steps})"
        )
        self._remove_indices(indices_to_remove)

    def _count_for_target(
        self, target_step: int, max_age_steps: int | None = None
    ) -> int:
        """Count trajectories usable for ``target_step``.

        Must be called while holding ``self._lock``.
        """
        return sum(
            1
            for trajectory_version, target in zip(
                self.trajectory_versions, self.target_weight_versions
            )
            if target == target_step
            and self._is_valid_for_target(
                trajectory_version, target_step, max_age_steps
            )
        )

    def _truncate_to_max_size(self, current_training_step: int | None = None) -> None:
        """Truncate restored state to ``max_size`` after resume cleanup.

        Must be called while holding ``self._lock``.
        """
        if len(self.trajectories) <= self.max_size:
            return

        print(
            f"Truncating restored buffer from {len(self.trajectories)} "
            f"to max_size={self.max_size}"
        )
        if current_training_step is None:
            indices_to_keep = list(
                range(len(self.trajectories) - self.max_size, len(self.trajectories))
            )
        else:
            prioritized_indices = sorted(
                range(len(self.trajectories)),
                key=lambda i: (self.target_weight_versions[i], i),
            )
            indices_to_keep = sorted(prioritized_indices[: self.max_size])

        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train."""
        with self._lock:
            current_count = self._count_for_target(target_step, max_age_steps)
            return current_count >= num_prompts_per_step

    def _remove_incomplete_target_steps(self, num_prompts_per_step: int) -> None:
        """Remove target steps without a complete batch.

        Must be called while holding ``self._lock``.
        """
        target_counts = Counter(self.target_weight_versions)
        incomplete_targets = {
            target
            for target, count in target_counts.items()
            if count < num_prompts_per_step
        }
        if not incomplete_targets:
            print(f"   All target steps have complete batches ({num_prompts_per_step})")
            return

        print(f"   Removing incomplete target steps: {sorted(incomplete_targets)}")
        original_count = len(self.trajectories)
        indices_to_keep = [
            i
            for i, target in enumerate(self.target_weight_versions)
            if target not in incomplete_targets
        ]
        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [
            self.trajectory_versions[i] for i in indices_to_keep
        ]
        self.target_weight_versions = [
            self.target_weight_versions[i] for i in indices_to_keep
        ]
        print(
            f"   Removed {original_count - len(self.trajectories)} trajectories "
            "from incomplete target steps"
        )

        if self.target_weight_versions:
            first_remaining_target = min(self.target_weight_versions)
            self.last_target_weight_already_generated = min(
                self.last_target_weight_already_generated,
                first_remaining_target - 1,
            )
        else:
            self.last_target_weight_already_generated = -1


@ray.remote  # pragma: no cover
class ReplayBuffer(ReplayBufferImpl):
    pass


class TQReplayBuffer:
    """Meta cache + TQ writer with reserve-then-commit slot semantics.

    meta_list, weight lists, rollout_metrics_list, ready_list, and _group_ids are
    parallel; a slot stays ready=False until commit fills it.
    """

    def __init__(
        self,
        dp_client: Any,
        partition_id: str,
        *,
        pad_value_dict: Mapping[str, int],
        require_routed_experts: bool = False,
        include_shared_prefix_metadata: bool = False,
        include_reward_processing_metadata: bool = False,
        include_message_advantage_metadata: bool = False,
        rollout_reward_semantics_fingerprint: str,
    ):
        """Create the replay buffer and configure its opt-in payload schema."""
        _validate_rollout_reward_semantics_fingerprint(
            rollout_reward_semantics_fingerprint
        )
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_value_dict = dict(pad_value_dict)
        self._require_routed_experts = require_routed_experts
        self._include_shared_prefix_metadata = include_shared_prefix_metadata
        self._include_reward_processing_metadata = include_reward_processing_metadata
        self._include_message_advantage_metadata = include_message_advantage_metadata
        self._rollout_reward_semantics_fingerprint = (
            rollout_reward_semantics_fingerprint
        )
        self.meta_list: list[Optional[KVBatchMeta]] = []
        self.start_weight_list: list[int] = []
        self.end_weight_list: list[int] = []
        # Per-slot target training step (set when force_in_order=True, else None).
        self.target_step_list: list[Optional[int]] = []
        self.rollout_metrics_list: list[Optional[RolloutScalarMetrics]] = []
        self.ready_list: list[bool] = []
        self._group_ids: list[str] = []

    def reserve(
        self,
        *,
        weight_version: int,
        target_step: Optional[int] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """Append an unready slot tagged with weight_version.

        Args:
            weight_version: Weight version stamped on the slot.
            target_step: Training step this slot targets; only consulted by StalenessSampler.force_in_order.
            group_id: Per-group sample_id prefix; defaults to a fresh uuid4.

        Returns:
            group_id used by the matching commit.
        """
        if group_id is None:
            group_id = str(uuid.uuid4())
        self.meta_list.append(None)
        self.start_weight_list.append(weight_version)
        self.end_weight_list.append(-1)
        self.target_step_list.append(target_step)
        self.rollout_metrics_list.append(None)
        self.ready_list.append(False)
        self._group_ids.append(group_id)
        return group_id

    async def commit(
        self,
        group_id: str,
        record: PromptGroupRecord,
        start_weight_version: int,
        end_weight_version: int,
    ) -> KVBatchMeta:
        """Tensorize record, write N rows to TQ, and mark the slot ready.

        Args:
            group_id: group_id returned by the matching reserve call.
            record: PromptGroupRecord to tensorize.
            start_weight_version: Weight version stamped on the slot before rollout.
                The same as the one from reserve, passed again to avoid race condition when lookup.
            end_weight_version: Weight version stamped on the slot after rollout.

        Returns:
            KVBatchMeta for the committed group.

        Raises:
            ValueError: group_id has no live slot (removed or never reserved).
            RuntimeError: router replay is enabled but the payload has no routes.
        """
        # Precondition: reserve() must have registered this group_id. Raise
        # before any side effects so a stray commit doesn't leak orphan DP rows.
        if group_id not in self._group_ids:
            raise ValueError(
                f"commit called with unknown group_id={group_id!r}; "
                f"reserve() must precede commit() (or the slot was already removed)"
            )
        scalar_rollout_metrics = _scalar_rollout_metrics(record.rollout_metrics)
        _validate_rollout_metric_receipt(scalar_rollout_metrics)
        record_kwargs: dict[str, bool] = {}
        if self._include_reward_processing_metadata:
            record_kwargs["include_reward_processing_metadata"] = True
        if self._include_message_advantage_metadata:
            record_kwargs["include_message_advantage_metadata"] = True
        shared_prefix_kwargs = (
            {"include_shared_prefix_metadata": True}
            if self._include_shared_prefix_metadata
            else {}
        )
        # Keep the disabled call shape identical to the legacy API. Besides
        # preserving the default path, this avoids breaking deployments that
        # wrap the converter/packer with the old keyword-only signature.
        train_batch = record_to_train_batch(
            record,
            pad_value_dict=self._pad_value_dict,
            **record_kwargs,
            **shared_prefix_kwargs,
        )
        sample_ids, fields, tags = pack_payload(
            train_batch,
            weight_version=start_weight_version,
            group_id=group_id,
            **shared_prefix_kwargs,
        )
        _validate_rollout_metric_receipt(
            scalar_rollout_metrics,
            expected_samples=len(sample_ids),
        )
        if self._require_routed_experts and ROUTED_EXPERTS_FIELD not in fields:
            raise RuntimeError(
                "policy.router_replay.enabled=true requires routed_experts in "
                "the SingleController rollout payload, but payload packing did "
                "not produce that field. Check vLLM routed-expert capture and "
                "the async message-log flattening path."
            )
        trace_rollout_payload(keys=sample_ids, data=train_batch)
        try:
            await self._call_dp(
                "put_samples",
                sample_ids=sample_ids,
                partition_id=self._partition_id,
                fields=fields,
                tags=tags,
            )

            # mirrors kv_first_write
            lengths = train_batch["input_lengths"]
            meta = KVBatchMeta(
                partition_id=self._partition_id,
                task_name="train",
                sample_ids=list(sample_ids),
                fields=list(fields.keys()),
                sequence_lengths=[int(s) for s in lengths.tolist()],
                tags=[dict(t) for t in tags],
            )

            idx = self._group_ids.index(group_id)
            self.meta_list[idx] = meta
            self.end_weight_list[idx] = end_weight_version
            self.rollout_metrics_list[idx] = scalar_rollout_metrics
            self.ready_list[idx] = True
            return meta
        except BaseException as commit_error:
            # put_samples may have written rows before raising. Roll back by the
            # deterministic IDs known here; the caller removes the reserved slot.
            try:
                await self._call_dp(
                    "clear_samples",
                    sample_ids=list(sample_ids),
                    partition_id=self._partition_id,
                )
            except BaseException as rollback_error:
                if isinstance(commit_error, asyncio.CancelledError):
                    raise commit_error from rollback_error
                raise BaseExceptionGroup(
                    f"commit and rollback both failed for group_id={group_id!r}",
                    [commit_error, rollback_error],
                )
            raise

    async def remove_group(self, group_id: str, *, remove_in_dp: bool = False) -> int:
        """Remove the live slot identified by ``group_id``.

        Args:
            group_id: Group identifier returned by :meth:`reserve`.
            remove_in_dp: Whether to clear rows referenced by a committed slot.

        Returns:
            Number of removed slots (always one on success).

        Raises:
            ValueError: ``group_id`` has no live slot.
        """
        try:
            idx = self._group_ids.index(group_id)
        except ValueError as error:
            raise ValueError(f"unknown group_id={group_id!r}") from error
        return await self.remove([idx], remove_in_dp=remove_in_dp)

    async def remove(self, idxs: list[int], remove_in_dp: bool) -> int:
        """Drop entries at the given indices and optionally clear them from DataPlane.

        Args:
            idxs: Entry indices to drop. Must be within [0, size).
            remove_in_dp: If True, also clear the dropped rows from DataPlane.

        Returns:
            Number of group entries removed from the buffer.
        """
        if len(idxs) == 0:
            return 0

        drop_idxs = sorted(idxs, reverse=True)
        if drop_idxs[0] >= len(self.meta_list):
            raise IndexError(
                f"TQReplayBuffer.remove: indices out of range: {drop_idxs[0]}; "
                f"size={len(self.meta_list)}"
            )

        dropped_sample_ids: list[str] = []
        for i in drop_idxs:
            meta = self.meta_list[i]
            if meta is not None:
                dropped_sample_ids.extend(meta.sample_ids)
            del self.meta_list[i]
            del self.start_weight_list[i]
            del self.end_weight_list[i]
            del self.target_step_list[i]
            del self.rollout_metrics_list[i]
            del self.ready_list[i]
            del self._group_ids[i]

        if remove_in_dp:
            await self._call_dp(
                "clear_samples",
                sample_ids=dropped_sample_ids,
                partition_id=self._partition_id,
            )

        return len(drop_idxs)

    async def remove_selected(
        self, idxs: list[int]
    ) -> tuple[int, list[RolloutScalarMetrics]]:
        """Remove ready groups locally and return their scalar rollout metrics.

        This is the sampler consumption path. Eviction and failed reservations use
        :meth:`remove` directly and therefore cannot leak their metrics into a later
        optimizer step.
        """
        if not idxs:
            return 0, []
        if len(set(idxs)) != len(idxs):
            raise ValueError(
                f"TQReplayBuffer.remove_selected indices must be unique: {idxs}"
            )
        selected_metrics: list[RolloutScalarMetrics] = []
        for idx in idxs:
            if idx < 0 or idx >= len(self.meta_list):
                raise IndexError(
                    f"TQReplayBuffer.remove_selected index out of range: {idx}; "
                    f"size={len(self.meta_list)}"
                )
            if not self.ready_list[idx] or self.meta_list[idx] is None:
                raise RuntimeError(
                    f"TQReplayBuffer.remove_selected requires ready group at index {idx}"
                )
            metrics = self.rollout_metrics_list[idx]
            if metrics is None:
                raise RuntimeError(
                    f"ready replay group at index {idx} has no rollout metrics"
                )
            selected_metrics.append(dict(metrics))

        removed = await self.remove(idxs, remove_in_dp=False)
        return removed, selected_metrics

    async def state_dict(self, *, saved_capacity: int) -> dict[str, Any]:
        """Serialize ready groups (meta + DataPlane payloads) for checkpointing.

        Snapshots the ready slots synchronously on the event loop first, then
        fetches each group's rows from the DataPlane. Unready reservations are
        in-flight rollouts and are dropped, matching legacy semantics. The
        snapshot stays consistent during the async fetch: concurrent commits
        only append/flip *other* slots, and the train pump — the only
        remover — is the caller itself; groups committed mid-save land in the
        next checkpoint.

        Args:
            saved_capacity: max_buffered_rollouts at save time, recorded so
                load_state_dict can report capacity changes across restarts.

        Returns:
            Envelope: ``{"partition_id": ..., "saved_capacity": ...,
            "rollout_reward_semantics_fingerprint": ..., "groups":
            [{"meta", "start_weight", "end_weight", "target_step", "group_id",
            "rollout_metrics", "fields_data"}, ...]}``.
        """
        snapshot: list[
            tuple[
                KVBatchMeta,
                int,
                int,
                Optional[int],
                str,
                RolloutScalarMetrics,
            ]
        ] = []
        for i, ready in enumerate(self.ready_list):
            if not ready:
                continue
            meta = self.meta_list[i]
            assert meta is not None  # commit sets meta before ready=True
            rollout_metrics = self.rollout_metrics_list[i]
            assert rollout_metrics is not None  # commit sets metrics before ready=True
            snapshot.append(
                (
                    meta,
                    self.start_weight_list[i],
                    self.end_weight_list[i],
                    self.target_step_list[i],
                    self._group_ids[i],
                    dict(rollout_metrics),
                )
            )

        groups: list[dict[str, Any]] = []
        for (
            meta,
            start_weight,
            end_weight,
            target_step,
            group_id,
            rollout_metrics,
        ) in snapshot:
            _validate_checkpoint_meta_fields(meta=meta, group_id=group_id)
            fields_data = await self._call_dp(
                "get_samples",
                sample_ids=meta.sample_ids,
                partition_id=self._partition_id,
                select_fields=meta.fields,
            )
            _validate_checkpoint_group_fields(
                meta=meta,
                fields_data=fields_data,
                group_id=group_id,
            )
            groups.append(
                {
                    "meta": meta,
                    "start_weight": start_weight,
                    "end_weight": end_weight,
                    "target_step": target_step,
                    "group_id": group_id,
                    "rollout_metrics": rollout_metrics,
                    "fields_data": fields_data,
                }
            )
        return {
            "partition_id": self._partition_id,
            "saved_capacity": saved_capacity,
            "rollout_reward_semantics_fingerprint": (
                self._rollout_reward_semantics_fingerprint
            ),
            "groups": groups,
        }

    async def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        max_groups: int,
        expected_partition_id: str,
        expected_group_size: int,
    ) -> int:
        """Validate and re-put checkpointed groups into the buffer.

        The preflight runs entirely before any DataPlane write (legacy
        precedent: validate, then truncate):
          1. Validate the envelope and raise ValueError on malformed state.
          2. Truncate to ``max_groups``, keeping the freshest groups, so the
             restored count can never exceed the buffer's capacity. Groups
             carrying a ``target_step`` are never truncated — an over-capacity
             in-order checkpoint raises instead (see Raises).

        Staleness is intentionally NOT handled here — load only loads. The
        train pump's first ``sampler.evict`` drops any restored group that is
        outside the staleness window and releases its capacity permit, keeping
        eviction in one place.

        Args:
            state: Envelope produced by ``state_dict``.
            max_groups: Current max_buffered_rollouts; the restored count
                never exceeds it.
            expected_partition_id: Partition this buffer writes to; must
                match the envelope.
            expected_group_size: num_generations_per_prompt; every group must
                hold exactly this many rows (a changed group size silently
                breaks the group-relative baseline).

        Returns:
            Number of groups restored into the buffer.

        Raises:
            ValueError: If the envelope is malformed (missing keys, partition
                mismatch, misaligned or wrongly sized groups, duplicate
                sample_ids), or if target-stamped groups exceed ``max_groups``.
        """
        required_keys = {
            "partition_id",
            "saved_capacity",
            "rollout_reward_semantics_fingerprint",
            "groups",
        }
        missing_keys = required_keys - set(state)
        if missing_keys:
            raise ValueError(
                f"Replay buffer checkpoint missing required keys: {missing_keys}"
            )
        if state["partition_id"] != expected_partition_id:
            raise ValueError(
                "Replay buffer checkpoint partition_id mismatch: "
                f"checkpoint={state['partition_id']!r}, "
                f"expected={expected_partition_id!r}"
            )
        checkpoint_fingerprint = state["rollout_reward_semantics_fingerprint"]
        _validate_rollout_reward_semantics_fingerprint(checkpoint_fingerprint)
        if checkpoint_fingerprint != self._rollout_reward_semantics_fingerprint:
            raise ValueError(
                "Replay buffer checkpoint rollout reward-semantics fingerprint "
                "mismatch: "
                f"checkpoint={checkpoint_fingerprint!r}, "
                f"expected={self._rollout_reward_semantics_fingerprint!r}. "
                "Resume with the same rollout path, effort, reward-penalty, "
                "environment-mask, thinking-tag, and message-advantage "
                "configuration, or delete replay_buffer.pt to start with an "
                "empty buffer."
            )

        groups = list(state["groups"])
        group_keys = {
            "meta",
            "start_weight",
            "end_weight",
            "target_step",
            "group_id",
            "rollout_metrics",
            "fields_data",
        }
        seen_sample_ids: set[str] = set()
        shared_prefix_fields = {
            SHARED_PREFIX_GROUP_ID,
            SHARED_PREFIX_PROMPT_LENGTHS,
        }
        for group in groups:
            missing_group_keys = group_keys - set(group)
            if missing_group_keys:
                raise ValueError(
                    f"Replay buffer checkpoint group missing keys: {missing_group_keys}"
                )
            restored_rollout_metrics = _scalar_rollout_metrics(group["rollout_metrics"])
            try:
                _validate_rollout_metric_receipt(restored_rollout_metrics)
            except ValueError as error:
                raise ValueError(
                    "Replay buffer checkpoint has no complete rollout metric "
                    "receipt. It predates exact SingleController rollout logging; "
                    "start with an empty replay buffer instead."
                ) from error
            meta = group["meta"]
            _validate_checkpoint_group_fields(
                meta=meta,
                fields_data=group["fields_data"],
                group_id=group["group_id"],
            )
            num_tags = len(meta.tags) if meta.tags is not None else -1
            num_lengths = (
                len(meta.sequence_lengths) if meta.sequence_lengths is not None else -1
            )
            if not (
                len(meta.sample_ids) == num_tags == num_lengths == expected_group_size
            ):
                raise ValueError(
                    "Replay buffer checkpoint group misaligned: "
                    f"sample_ids={len(meta.sample_ids)}, tags={num_tags}, "
                    f"sequence_lengths={num_lengths}, "
                    f"expected_group_size={expected_group_size}"
                )
            _validate_rollout_metric_receipt(
                restored_rollout_metrics,
                expected_samples=len(meta.sample_ids),
            )
            present_shared_prefix_fields = shared_prefix_fields.intersection(
                meta.fields or ()
            )
            expected_shared_prefix_fields = (
                shared_prefix_fields if self._include_shared_prefix_metadata else set()
            )
            if present_shared_prefix_fields != expected_shared_prefix_fields:
                raise ValueError(
                    "Replay buffer checkpoint shared-prefix schema mismatch: "
                    f"checkpoint_fields={sorted(present_shared_prefix_fields)}, "
                    f"expected_fields={sorted(expected_shared_prefix_fields)}. "
                    "Resume with the same policy.shared_prefix_training.mode or "
                    "delete replay_buffer.pt to start with an empty buffer."
                )
            reward_processing_fields = {
                ROLLOUT_TRUNCATED,
                RESPONSE_TOKEN_LENGTHS,
            }
            message_advantage_fields = {
                INVALID_TOOL_CALL_TOKEN_MASK,
                MALFORMED_THINKING_TOKEN_MASK,
                GENERATED_ASSISTANT_MESSAGE_COUNT,
                INVALID_TOOL_CALL_MESSAGE_COUNT,
                MALFORMED_THINKING_MESSAGE_COUNT,
                INVALID_AND_MALFORMED_MESSAGE_COUNT,
            }
            for schema_name, schema_fields, enabled in (
                (
                    "reward-processing",
                    reward_processing_fields,
                    self._include_reward_processing_metadata,
                ),
                (
                    "message-advantage",
                    message_advantage_fields,
                    self._include_message_advantage_metadata,
                ),
            ):
                present_fields = schema_fields.intersection(meta.fields or ())
                expected_fields = schema_fields if enabled else set()
                if present_fields != expected_fields:
                    raise ValueError(
                        f"Replay buffer checkpoint {schema_name} schema mismatch: "
                        f"checkpoint_fields={sorted(present_fields)}, "
                        f"expected_fields={sorted(expected_fields)}. Resume with "
                        "the same reward-processing configuration or delete "
                        "replay_buffer.pt to start with an empty buffer."
                    )
            for sid in meta.sample_ids:
                if sid in seen_sample_ids:
                    raise ValueError(
                        f"Replay buffer checkpoint has duplicate sample_id: {sid!r}"
                    )
                seen_sample_ids.add(sid)

        if state["saved_capacity"] != max_groups:
            print(
                "TQReplayBuffer capacity changed: "
                f"checkpoint={state['saved_capacity']}, current={max_groups}. "
                "Using current config value."
            )
        num_truncated = 0
        if len(groups) > max_groups:
            if any(group["target_step"] is not None for group in groups):
                raise ValueError(
                    f"Replay buffer checkpoint holds {len(groups)} group(s) "
                    f"but async_rl.max_buffered_rollouts is {max_groups}. "
                    "These groups carry target_step stamps (in-order "
                    "sampling) and are selected as whole per-step batches, so "
                    "dropping any of them would deadlock the resumed run. "
                    "Resume with async_rl.max_buffered_rollouts >= "
                    f"{len(groups)}, or delete replay_buffer.pt from the "
                    "checkpoint to resume with an empty buffer."
                )
            num_truncated = len(groups) - max_groups
            # Keep the freshest max_groups groups, preserving original order.
            prioritized = sorted(
                range(len(groups)),
                key=lambda i: (groups[i]["start_weight"], i),
            )
            indices_to_keep = sorted(prioritized[num_truncated:])
            groups = [groups[i] for i in indices_to_keep]

        for group in groups:
            meta = group["meta"]
            await self._call_dp(
                "put_samples",
                sample_ids=list(meta.sample_ids),
                partition_id=self._partition_id,
                fields=group["fields_data"],
                tags=[dict(t) for t in meta.tags],
            )
            self.meta_list.append(meta)
            self.start_weight_list.append(group["start_weight"])
            self.end_weight_list.append(group["end_weight"])
            self.target_step_list.append(group["target_step"])
            self.rollout_metrics_list.append(
                _scalar_rollout_metrics(group["rollout_metrics"])
            )
            self.ready_list.append(True)
            self._group_ids.append(group["group_id"])

        summary = f"📦 Restored {len(groups)} replay group(s) from checkpoint"
        if num_truncated:
            summary += f"; truncated {num_truncated} group(s) over capacity"
        print(summary, flush=True)
        return len(groups)

    def count_for_target_step(self, target_step: int) -> int:
        """Return how many slots are stamped with ``target_step``."""
        return sum(1 for target in self.target_step_list if target == target_step)

    def promote_ready_group(self, *, to_target_step: int) -> Optional[int]:
        """Re-stamp a finished group from a later step so it lands in this one.

        Fills a hole left by a dropped prompt with generation that is already done,
        which is the point: the step closes immediately instead of waiting out a fresh
        rollout. The step it was borrowed from is returned so the caller can repay it,
        and the caller must -- an unrepaid loan is the same hole one step later, carried
        forward until it reaches the last step, which has nobody to borrow from.

        The furthest future step is preferred because it is due last and so has the most
        slack to absorb the repayment. Only ready slots qualify: an unready one is a
        reservation whose rollout is still running, so moving its stamp would hand this
        step the same wait it is trying to avoid.

        Promotion can only make a step fresher, never staler. Slots are appended in
        dispatch order and the trainer version never decreases, so a group stamped for a
        later step was generated at a weight version at least as new as the ones already
        in this step.

        Synchronous on purpose. ``remove`` deletes its indices before its own first
        await, so as long as nothing here yields, the index picked below cannot be
        shifted out from under the write by a selection running concurrently.

        Args:
            to_target_step: Training step to re-stamp the borrowed group onto -- the
                step that lost a prompt. Must be at or ahead of the trainer version:
                a group re-stamped onto a step already trained is never selectable
                again and would only be evicted.

        Returns:
            The target step the group was taken from, or None when no later step has a
            ready group to lend.
        """
        lender_idx: Optional[int] = None
        lender_target: Optional[int] = None
        for i, target in enumerate(self.target_step_list):
            if target is None or target <= to_target_step or not self.ready_list[i]:
                continue
            if lender_target is None or target > lender_target:
                lender_idx, lender_target = i, target
        if lender_idx is None:
            return None
        self.target_step_list[lender_idx] = to_target_step
        return lender_target

    def size(self) -> int:
        """Return the number of prompt-group entries currently held."""
        return len(self.meta_list)

    def __len__(self) -> int:
        return len(self.meta_list)

    async def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        """Call a DataPlaneClient method, awaiting Ray remotes if needed."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await remote(**kwargs)
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
