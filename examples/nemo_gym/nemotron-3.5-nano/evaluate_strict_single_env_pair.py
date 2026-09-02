#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Evaluate a strict single-environment shared-prefix OFF/ON pair offline.

The evaluator consumes exports that were assembled from W&B and separately
sealed receipts.  It never imports W&B, contacts a service, or launches work.
Sparse W&B rows are joined by their exact integer ``_step``.  Missing and
conflicting observations remain unavailable evidence; they are never filled
from a summary value or a neighbouring step.

Exit status is 0 for GREEN, 1 for a complete but failed acceptance gate, and 2
when any required conclusion is unverifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "nemo-rl-strict-single-env-acceptance-contract-v1"
RUN_EXPORT_SCHEMA = "nemo-rl-offline-wandb-run-export-v1"
REPORT_SCHEMA = "nemo-rl-strict-single-env-acceptance-report-v1"
JOB_RECEIPT_SCHEMA = "nemo-rl-strict-pair-job-receipt-v1"
EXECUTION_MARKER_RECEIPT_SCHEMA = "nemo-rl-shared-prefix-physical-execution-receipt-v1"
EXECUTION_MARKER_SEMANTICS = "production_packed_fused_training_path"

STEPS = tuple(range(1, 101))
STEPS_PER_EPOCH = 5
EPOCHS = 20
PRIMARY_STEPS = tuple(range(11, 101))
PRIMARY_EPOCHS = tuple(
    tuple(range(first, first + STEPS_PER_EPOCH)) for first in range(11, 101, 5)
)
TAIL_STEPS = tuple(range(76, 101))
TAIL_EPOCHS = frozenset(range(16, 21))
K4_SAMPLES = 4
MIN_MATCHED_PRIMARY_EPOCHS = 15
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_828
REWARD_ABS_TOLERANCE = 1e-7
RATIO_ABS_TOLERANCE = 1e-9
MAX_EXACT_INTEGER = 1 << 53

HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

SOURCE_KEYS = frozenset(
    {
        "nemo_rl",
        "megatron_bridge",
        "megatron_lm",
        "nemo_gym",
        "nemo_automodel",
        "pipeclean",
    }
)
COMMON_PROVENANCE_KEYS = frozenset(
    {
        "pair_manifest_sha256",
        "source_bundle_manifest_sha256",
        "wave_manifest_sha256",
        "rendered_plan_sha256",
        "acceptance_contract_sha256",
        "fixture_sha256",
        "model_tree_sha256",
        "training_container_sha256",
        "sandbox_container_sha256",
        "verifier_source_sha256",
        "reward_liveness_contract_sha256",
        "gpu_numerical_parity_receipt_sha256",
        "base_recipe_sha256",
        "dataset_sha256",
        "gym_config_sha256",
        "environment_recipe_sha256",
        "prompt_schedule_sha256",
        "generation_seed_schedule_sha256",
        "launcher_sha256",
        "launch_package_manifest_sha256",
        "arm_normalized_runtime_environment_sha256",
        "arm_normalized_resolved_config_sha256",
        "paired_step1_ledger_sha256",
        "pinned_verifier_replay_receipt_sha256",
        "same_arm_reproducibility_receipt_sha256",
        "cross_arm_step1_parity_receipt_sha256",
        "reward_semantics_contract_sha256",
        "nemo_runnable_manifest_sha256",
        "bridge_runnable_manifest_sha256",
        "mcore_runnable_manifest_sha256",
        "deployment_ready_file_sha256",
    }
)
ARM_PROVENANCE_KEYS = frozenset(
    {
        "resolved_config_sha256",
        "runtime_environment_sha256",
        "code_snapshot_sha256",
        "allocation_receipt_sha256",
        "shared_prefix_runtime_trace_sha256",
        "runtime_direction_receipt_sha256",
        "reward_ledger_sha256",
        "train_data_index_sha256",
        "snapshot_manifest_sha256",
        "entrypoint_sha256",
        "wrapper_sha256",
        "inner_ray_sha256",
        "command_sha256",
        "mounts_sha256",
    }
)
TOPOLOGY_KEYS = frozenset(
    {
        "cluster_name",
        "slurm_partition",
        "gpu_model",
        "nvidia_driver_version",
        "allocated_nodes",
        "gpus_per_node",
        "trainer_nodes",
        "trainer_gpus_per_node",
        "generation_nodes",
        "generation_gpus_per_node",
        "tensor_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "expert_tensor_parallel_size",
        "mtp_num_layers",
    }
)
CONFIG_KEYS = frozenset(
    {
        "max_num_steps",
        "epochs",
        "steps_per_epoch",
        "fixture_rows",
        "num_prompts_per_step",
        "num_generations_per_prompt",
        "seed",
        "generation_seed_base",
        "data_shuffle",
        "reward_scaling_enabled",
        "reward_shaping_enabled",
        "shared_prefix_mode",
        "wandb_enabled",
        "tensorboard_enabled",
        "wandb_entity",
        "wandb_project",
        "wandb_group",
        "wandb_run_name",
        "tensor_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "expert_tensor_parallel_size",
        "mtp_num_layers",
        "mtp_use_repeated_layer",
        "mtp_detach_heads",
        "mtp_loss_scaling_factor",
        "slurm_partition",
        "slurm_account",
        "resolved_config_sha256",
    }
)

ACCEPTANCE = {
    "steps": 100,
    "epochs": 20,
    "steps_per_epoch": 5,
    "samples_per_step": 4,
    "warmup_steps": 10,
    "raw_environment_reward_distinct_values_min": 2,
    "effective_reward_distinct_values_min": 2,
    "mixed_K4_steps_min": 20,
    "active_epochs_min": 16,
    "tail_epoch_low": 16,
    "tail_epoch_high": 20,
    "tail_active_epochs_min": 1,
    "first_mixed_step_max": 10,
    "unique_step_reward_means_min": 3,
    "step_reward_mean_population_stddev_min": 0.05,
    "penalty_rate_max": 0.05,
    "nonzero_policy_gradient_epochs_min": 16,
    "nonzero_mtp_gradient_epochs_min": 16,
    "tail_policy_and_mtp_gradient_witness_required": True,
    "matched_primary_epochs_min": 15,
    "primary_epochs_total": 18,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "policy_training_speedup_ci_low_exclusive": 1.0,
}

REWARD_METRICS = frozenset(
    {
        "train/raw_environment_reward",
        "train/raw_environment_reward/min",
        "train/raw_environment_reward/max",
        "train/pre_penalty_environment_reward",
        "train/pre_penalty_environment_reward/min",
        "train/pre_penalty_environment_reward/max",
        "train/verifier_reward",
        "train/total_reward/mean",
        "train/total_reward/min",
        "train/total_reward/max",
        "train/reward",
        "train/reward_processing_delta",
        "train/effort_low_sample_count",
        "train/effort_low_sample_rate",
        "train/effort_reward_delta",
        "train/num_mask_sample_filtered",
        "train/mask_sample_rate",
        "train/reasoning_equal_to_final_answer_rate",
        "train/empty_final_answer_rate",
        "train/unwanted_token_rate",
        "train/malformed_think_tag_rate",
        "train/invalid_tool_call_rate",
        "train/malformed_thinking_rate",
        "train/raw_invalid_tool_call_rate",
        "train/raw_malformed_thinking_rate",
        "train/invalid_and_malformed_rate",
        "train/rollout/samples",
    }
)
PENALTY_METRICS = (
    "train/reasoning_equal_to_final_answer_rate",
    "train/empty_final_answer_rate",
    "train/unwanted_token_rate",
    "train/malformed_think_tag_rate",
    "train/invalid_tool_call_rate",
    "train/malformed_thinking_rate",
    "train/raw_invalid_tool_call_rate",
    "train/raw_malformed_thinking_rate",
    "train/invalid_and_malformed_rate",
)
ZERO_STEP_METRICS = (
    "train/num_masked_seqs_by_logprob_error",
    "train/shared_prefix/fallback_sequences",
    "train/shared_prefix/runtime_fallback_sequences",
    "train/dropped_prompt_groups",
    "train/replaced_prompt_groups",
    "train/promoted_prompt_groups",
    "train/evicted_stale_prompt_groups",
    "train/aborted_stale_inflight_groups",
)
CUMULATIVE_ZERO_METRICS = (
    "rollout/skipped_total",
    "rollout/redispatch_total",
    "rollout/data_retry_total",
    "rollout/data_failures_total",
    "rollout/gym_row_redispatch_total",
    "rollout/infra_drops_total",
    "rollout/max_consecutive_infra_drops",
)
TOKEN_METRICS = frozenset(
    {
        "train/num_valid_samples",
        "train/global_valid_seqs",
        "train/global_valid_toks",
        "train/total_num_tokens",
        "train/shared_prefix/total_sequences",
        "train/shared_prefix/eligible_sequences",
        "train/shared_prefix/complete_groups",
        "train/shared_prefix/total_tokens",
        "train/shared_prefix/prompt_tokens",
        "train/shared_prefix/valid_loss_tokens",
        "train/shared_prefix/non_loss_suffix_tokens",
        "train/shared_prefix/shareable_prompt_tokens",
        "train/shared_prefix/ideal_shared_token_work",
        "train/shared_prefix/ideal_token_reduction",
        "train/shared_prefix/ideal_token_work_speedup",
    }
)
LEARNING_METRICS = frozenset(
    {
        "train/raw_environment_reward",
        "train/raw_environment_reward/min",
        "train/raw_environment_reward/max",
        "train/total_reward/mean",
        "train/total_reward/min",
        "train/total_reward/max",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
        "train/mtp/grad_norm",
        *PENALTY_METRICS,
    }
)
WORK_METRICS = (
    "train/num_valid_samples",
    "train/global_valid_toks",
    "train/total_num_tokens",
    "train/shared_prefix/total_sequences",
    "train/shared_prefix/eligible_sequences",
    "train/shared_prefix/complete_groups",
    "train/shared_prefix/shareable_prompt_tokens",
    "train/shared_prefix/ideal_token_reduction",
)
SPEED_METRICS = frozenset(
    {
        *TOKEN_METRICS,
        "train/rollout/samples",
        "train/shared_prefix/fallback_sequences",
        "train/shared_prefix/runtime_fallback_sequences",
        "timing/train/policy_training",
    }
)
INTEGER_METRICS = frozenset(
    {
        "train/effort_low_sample_count",
        "train/num_mask_sample_filtered",
        "train/rollout/samples",
        "train/num_valid_samples",
        "train/global_valid_seqs",
        "train/global_valid_toks",
        "train/total_num_tokens",
        "train/shared_prefix/total_sequences",
        "train/shared_prefix/eligible_sequences",
        "train/shared_prefix/complete_groups",
        "train/shared_prefix/fallback_sequences",
        "train/shared_prefix/runtime_fallback_sequences",
        "train/shared_prefix/total_tokens",
        "train/shared_prefix/prompt_tokens",
        "train/shared_prefix/valid_loss_tokens",
        "train/shared_prefix/non_loss_suffix_tokens",
        "train/shared_prefix/shareable_prompt_tokens",
        "train/shared_prefix/ideal_shared_token_work",
        "train/num_masked_seqs_by_logprob_error",
        "train/dropped_prompt_groups",
        "train/replaced_prompt_groups",
        "train/promoted_prompt_groups",
        "train/evicted_stale_prompt_groups",
        "train/aborted_stale_inflight_groups",
        *CUMULATIVE_ZERO_METRICS,
    }
)
RATE_METRICS = frozenset(
    {
        "train/effort_low_sample_rate",
        "train/mask_sample_rate",
        *PENALTY_METRICS,
        "train/shared_prefix/ideal_token_reduction",
    }
)
CROSS_ARM_PARITY_FIELDS = (
    "completion_token_ids",
    "raw_environment_reward",
    "pre_penalty_environment_reward",
    "penalty_flags",
    "verifier_reward",
    "processed_reward",
    "sample_mask",
    "global_valid_toks",
    "total_num_tokens",
)


class EvidenceError(RuntimeError):
    """An input cannot establish the claimed acceptance result."""


@dataclass(frozen=True)
class Document:
    """A parsed JSON object and the SHA-256 of its exact source bytes."""

    value: dict[str, Any]
    sha256: str


@dataclass
class History:
    """Sparse W&B observations merged without imputing absent values."""

    values: dict[int, dict[str, int | float]] = field(default_factory=dict)
    metric_errors: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    global_errors: list[str] = field(default_factory=list)

    def require(
        self,
        metrics: Sequence[str] | set[str] | frozenset[str],
        *,
        steps: Sequence[int] = STEPS,
    ) -> None:
        """Raise when any requested step/metric observation is unavailable."""
        errors = list(self.global_errors)
        for metric in sorted(metrics):
            for step in steps:
                errors.extend(self.metric_errors.get(metric, {}).get(step, ()))
            missing = [
                step for step in steps if metric not in self.values.get(step, {})
            ]
            if missing:
                errors.append(f"{metric} missing at steps {_compact_steps(missing)}")
        if errors:
            raise EvidenceError("; ".join(errors[:20]))

    def number(self, step: int, metric: str) -> float:
        """Return one already-validated finite observation."""
        return float(self.values[step][metric])

    def integer(self, step: int, metric: str) -> int:
        """Return one observation as an exact integer."""
        return _integer(self.values[step][metric], f"step {step} {metric}")


@dataclass(frozen=True)
class Run:
    """One authenticated offline W&B run export."""

    arm: str
    identity: dict[str, Any]
    history: History


@dataclass
class Gate:
    """One independently reported acceptance decision."""

    name: str
    failures: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        """Record a complete observation that violates the acceptance contract."""
        if len(self.failures) < 50:
            self.failures.append(message)

    def unverifiable(self, message: str) -> None:
        """Record missing, contradictory, or unauthenticated evidence."""
        if len(self.unavailable) < 50:
            self.unavailable.append(message)

    @property
    def status(self) -> str:
        """Return PASS, FAIL, or UNVERIFIABLE without conflating them."""
        if self.unavailable:
            return "UNVERIFIABLE"
        if self.failures:
            return "FAIL"
        return "PASS"

    def report(self) -> dict[str, Any]:
        """Render the stable JSON form for this gate."""
        return {
            "status": self.status,
            "failures": self.failures,
            "unavailable": self.unavailable,
            "evidence": self.evidence,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON constant {value}")


def load_document(path: Path, label: str) -> Document:
    """Load one JSON object while rejecting duplicate keys and non-finite values."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise EvidenceError(f"invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must contain one JSON object")
    return Document(value=value, sha256=hashlib.sha256(raw).hexdigest())


def document_from_value(value: dict[str, Any]) -> Document:
    """Construct deterministic in-memory evidence for hermetic callers/tests."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Document(value=value, sha256=hashlib.sha256(raw).hexdigest())


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise EvidenceError(f"{label} fields differ: missing={missing}, extra={extra}")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a non-empty filesystem-safe identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or HEX64_RE.fullmatch(value) is None
        or set(value) == {"0"}
    ):
        raise EvidenceError(f"{label} must be a populated lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or HEX40_RE.fullmatch(value) is None
        or set(value) == {"0"}
    ):
        raise EvidenceError(f"{label} must be a populated 40-hex commit")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    number = _number(value, label)
    if not number.is_integer() or abs(number) > MAX_EXACT_INTEGER:
        raise EvidenceError(f"{label} must be an exactly representable integer")
    result = int(number)
    if result < minimum:
        raise EvidenceError(f"{label} must be >= {minimum}")
    return result


def _close(left: float, right: float, *, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _compact_steps(steps: Sequence[int]) -> str:
    if len(steps) <= 8:
        return str(list(steps))
    return f"{list(steps[:4])}...{list(steps[-4:])} ({len(steps)} total)"


def _validate_hash_mapping(
    value: Any, required: frozenset[str], label: str
) -> dict[str, Any]:
    mapping = _mapping(value, label)
    missing = sorted(required - set(mapping))
    if missing:
        raise EvidenceError(f"{label} lacks mandatory pins: {missing}")
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise EvidenceError(f"{label} has unsupported non-SHA pin {key!r}")
        _digest(item, f"{label}.{key}")
    return mapping


def _validate_topology(value: Any) -> dict[str, Any]:
    topology = _mapping(value, "contract topology")
    _exact_keys(topology, set(TOPOLOGY_KEYS), "contract topology")
    for key in (
        "cluster_name",
        "slurm_partition",
        "gpu_model",
        "nvidia_driver_version",
    ):
        if not isinstance(topology[key], str) or not topology[key]:
            raise EvidenceError(f"contract topology {key} must be a non-empty string")
    for key in TOPOLOGY_KEYS - {
        "cluster_name",
        "slurm_partition",
        "gpu_model",
        "nvidia_driver_version",
        "sequence_parallel",
    }:
        _integer(topology[key], f"contract topology {key}", minimum=1)
    if not isinstance(topology["sequence_parallel"], bool):
        raise EvidenceError("contract topology sequence_parallel must be boolean")
    expected = {
        "cluster_name": "HSG",
        "slurm_partition": "batch",
        "allocated_nodes": 1,
        "gpus_per_node": 4,
        "trainer_nodes": 1,
        "trainer_gpus_per_node": 4,
        # Generation is colocated on the same four GPUs, so it is not added to
        # allocated_nodes.
        "generation_nodes": 1,
        "generation_gpus_per_node": 4,
        "tensor_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "mtp_num_layers": 5,
    }
    for key, expected_value in expected.items():
        if topology[key] != expected_value:
            raise EvidenceError(
                f"contract topology {key} differs from {expected_value!r}"
            )
    return topology


def validate_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable acceptance trust anchor and frozen thresholds."""
    _exact_keys(
        value,
        {
            "schema",
            "pair",
            "acceptance",
            "provenance",
            "configs",
            "holdout",
            "receipts",
            "verifier_metric",
        },
        "acceptance contract",
    )
    if value["schema"] != CONTRACT_SCHEMA:
        raise EvidenceError("unexpected acceptance contract schema")
    if value["acceptance"] != ACCEPTANCE:
        raise EvidenceError("acceptance thresholds differ from the frozen contract")

    pair = _mapping(value["pair"], "contract pair")
    _exact_keys(
        pair,
        {"pair_id", "environment", "entity", "project", "group", "run_ids"},
        "contract pair",
    )
    for key in ("pair_id", "environment", "entity", "project", "group"):
        _safe_id(pair[key], f"contract pair {key}")
    if pair["group"] != pair["pair_id"]:
        raise EvidenceError("contract W&B group must equal pair_id")
    run_ids = _mapping(pair["run_ids"], "contract W&B run IDs")
    _exact_keys(run_ids, {"off", "on"}, "contract W&B run IDs")
    for arm in ("off", "on"):
        _safe_id(run_ids[arm], f"contract {arm} W&B run ID")
    if run_ids["off"] == run_ids["on"]:
        raise EvidenceError("OFF/ON must bind distinct W&B run IDs")

    verifier_metric = value["verifier_metric"]
    if (
        not isinstance(verifier_metric, str)
        or not verifier_metric.startswith("train/")
        or verifier_metric in REWARD_METRICS
    ):
        raise EvidenceError("verifier_metric must name one distinct train/* metric")

    provenance = _mapping(value["provenance"], "contract provenance")
    _exact_keys(
        provenance,
        {
            "common",
            "source_commits",
            "source_git_trees",
            "source_trees_sha256",
            "topology",
            "arms",
        },
        "contract provenance",
    )
    _validate_hash_mapping(
        provenance["common"], COMMON_PROVENANCE_KEYS, "common provenance"
    )
    commits = _mapping(provenance["source_commits"], "source commits")
    git_trees = _mapping(provenance["source_git_trees"], "source Git trees")
    trees = _mapping(provenance["source_trees_sha256"], "source trees")
    _exact_keys(commits, set(SOURCE_KEYS), "source commits")
    _exact_keys(git_trees, set(SOURCE_KEYS), "source Git trees")
    _exact_keys(trees, set(SOURCE_KEYS), "source trees")
    for key in SOURCE_KEYS:
        _commit(commits[key], f"source commit {key}")
        _commit(git_trees[key], f"source Git tree {key}")
        _digest(trees[key], f"source tree {key}")
    _validate_topology(provenance["topology"])
    arms = _mapping(provenance["arms"], "arm provenance")
    _exact_keys(arms, {"off", "on"}, "arm provenance")
    for arm in ("off", "on"):
        _validate_hash_mapping(arms[arm], ARM_PROVENANCE_KEYS, f"{arm} arm provenance")
    if arms["off"] == arms["on"]:
        raise EvidenceError("OFF/ON arm provenance must bind distinct artifacts")

    configs = _mapping(value["configs"], "contract configs")
    _exact_keys(configs, {"off", "on"}, "contract configs")
    for arm, mode in (("off", "observe"), ("on", "train")):
        config = _mapping(configs[arm], f"{arm} contract config")
        missing = sorted(CONFIG_KEYS - set(config))
        if missing:
            raise EvidenceError(f"{arm} contract config lacks pins: {missing}")
        expected = {
            "max_num_steps": 100,
            "epochs": 20,
            "steps_per_epoch": 5,
            "fixture_rows": 5,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 4,
            "seed": 42,
            "generation_seed_base": 42,
            "data_shuffle": False,
            "reward_scaling_enabled": False,
            "reward_shaping_enabled": False,
            "shared_prefix_mode": mode,
            "wandb_enabled": True,
            "tensorboard_enabled": False,
            "wandb_entity": pair["entity"],
            "wandb_project": pair["project"],
            "wandb_group": pair["group"],
            "wandb_run_name": f"{arm}-{pair['pair_id']}",
            "tensor_parallel_size": 2,
            "context_parallel_size": 2,
            "sequence_parallel": True,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "mtp_num_layers": 5,
            "mtp_use_repeated_layer": True,
            "mtp_detach_heads": True,
            "mtp_loss_scaling_factor": 0.3,
            "slurm_partition": "batch",
            "slurm_account": "nemotron_sw_post",
        }
        for key, expected_value in expected.items():
            if config.get(key) != expected_value:
                raise EvidenceError(
                    f"{arm} contract config {key} differs from {expected_value!r}"
                )
        resolved_config_sha256 = _digest(
            config["resolved_config_sha256"],
            f"{arm} config resolved_config_sha256",
        )
        if resolved_config_sha256 != provenance["arms"][arm]["resolved_config_sha256"]:
            raise EvidenceError(
                f"{arm} normalized config does not bind its resolved-config digest"
            )
    allowed_arm_config_differences = {
        "shared_prefix_mode",
        "wandb_run_name",
        "resolved_config_sha256",
    }
    normalized_configs = {
        arm: {
            key: item
            for key, item in configs[arm].items()
            if key not in allowed_arm_config_differences
        }
        for arm in ("off", "on")
    }
    if normalized_configs["off"] != normalized_configs["on"]:
        raise EvidenceError(
            "OFF/ON configs differ outside shared-prefix mode, W&B name, and digest"
        )

    holdout = _mapping(value["holdout"], "holdout contract")
    _exact_keys(
        holdout,
        {
            "receipt_sha256",
            "primary_reward_mean_min",
            "tail_reward_mean_min",
        },
        "holdout contract",
    )
    _digest(holdout["receipt_sha256"], "holdout receipt SHA-256")
    for key in ("primary_reward_mean_min", "tail_reward_mean_min"):
        floor = _number(holdout[key], f"holdout {key}")
        if not 0.05 <= floor <= 1.0:
            raise EvidenceError(f"holdout {key} must be in [0.05, 1]")

    receipts = _mapping(value["receipts"], "receipt pins")
    _exact_keys(
        receipts,
        {
            "external_step1_same_arm_reproducibility_receipt",
            "external_step1_off_on_parity_receipt",
            "shared_prefix_execution_marker_receipts",
            "strict_job_exit_receipts",
        },
        "receipt pins",
    )
    for name in (
        "external_step1_same_arm_reproducibility_receipt",
        "external_step1_off_on_parity_receipt",
    ):
        _validate_receipt_pin(receipts[name], name)
    for group_name in (
        "shared_prefix_execution_marker_receipts",
        "strict_job_exit_receipts",
    ):
        group = _mapping(receipts[group_name], group_name)
        _exact_keys(group, {"off", "on"}, group_name)
        for arm in ("off", "on"):
            _validate_receipt_pin(group[arm], f"{group_name}.{arm}")
    common = provenance["common"]
    receipt_bindings = {
        "external_step1_same_arm_reproducibility_receipt": (
            "same_arm_reproducibility_receipt_sha256"
        ),
        "external_step1_off_on_parity_receipt": (
            "cross_arm_step1_parity_receipt_sha256"
        ),
    }
    for receipt_name, provenance_name in receipt_bindings.items():
        if receipts[receipt_name]["sha256"] != common[provenance_name]:
            raise EvidenceError(f"{receipt_name} is not bound by common provenance")
    for arm in ("off", "on"):
        if (
            receipts["shared_prefix_execution_marker_receipts"][arm]["sha256"]
            != arms[arm]["runtime_direction_receipt_sha256"]
        ):
            raise EvidenceError(
                f"{arm} execution marker is not bound by arm provenance"
            )
    return value


def _validate_receipt_pin(value: Any, label: str) -> dict[str, Any]:
    pin = _mapping(value, label)
    _exact_keys(pin, {"sha256", "semantic_pins"}, label)
    _digest(pin["sha256"], f"{label}.sha256")
    semantic = _mapping(pin["semantic_pins"], f"{label}.semantic_pins")
    if not semantic:
        raise EvidenceError(f"{label}.semantic_pins cannot be empty")
    return pin


def _all_metrics(verifier_metric: str) -> frozenset[str]:
    return frozenset(
        {
            *REWARD_METRICS,
            *ZERO_STEP_METRICS,
            *CUMULATIVE_ZERO_METRICS,
            *TOKEN_METRICS,
            *LEARNING_METRICS,
            *SPEED_METRICS,
            verifier_metric,
        }
    )


def merge_sparse_history(rows: Any, *, label: str, verifier_metric: str) -> History:
    """Merge sparse W&B rows by exact integer step, retaining conflicts."""
    history = History()
    if not isinstance(rows, list):
        history.global_errors.append(f"{label} history must be a list")
        return history
    known_metrics = _all_metrics(verifier_metric)
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            history.global_errors.append(
                f"{label} history row {row_index} is not an object"
            )
            continue
        present = known_metrics & set(raw)
        if not present:
            continue
        step = raw.get("_step")
        if isinstance(step, bool) or not isinstance(step, int):
            history.global_errors.append(
                f"{label} history row {row_index} has non-integer _step {step!r}"
            )
            continue
        if step not in STEPS:
            history.global_errors.append(
                f"{label} history row {row_index} has out-of-contract _step {step}"
            )
            continue
        merged = history.values.setdefault(step, {})
        for metric in present:
            try:
                value: int | float
                if metric in INTEGER_METRICS:
                    value = _integer(raw[metric], f"{label} step {step} {metric}")
                else:
                    value = _number(raw[metric], f"{label} step {step} {metric}")
                    if metric in RATE_METRICS and not 0.0 <= value <= 1.0:
                        raise EvidenceError(
                            f"{label} step {step} {metric} is outside [0, 1]"
                        )
            except EvidenceError as error:
                history.metric_errors.setdefault(metric, {}).setdefault(
                    step, []
                ).append(str(error))
                continue
            if metric in merged and merged[metric] != value:
                history.metric_errors.setdefault(metric, {}).setdefault(
                    step, []
                ).append(
                    f"{label} step {step} has conflicting {metric}: "
                    f"{merged[metric]!r} versus {value!r}"
                )
                continue
            merged[metric] = value
    return history


def _validate_run(value: dict[str, Any], contract: dict[str, Any], arm: str) -> Run:
    _exact_keys(
        value,
        {"schema", "identity", "provenance", "config", "history"},
        f"{arm} run export",
    )
    if value["schema"] != RUN_EXPORT_SCHEMA:
        raise EvidenceError(f"unexpected {arm} run export schema")
    pair = contract["pair"]
    mode = "observe" if arm == "off" else "train"
    identity = _mapping(value["identity"], f"{arm} identity")
    expected_identity = {
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "arm": arm,
        "shared_prefix_mode": mode,
        "entity": pair["entity"],
        "project": pair["project"],
        "group": pair["group"],
        "run_id": pair["run_ids"][arm],
        "run_name": f"{arm}-{pair['pair_id']}",
        "state": "finished",
    }
    if identity != expected_identity:
        raise EvidenceError(f"{arm} W&B identity differs from exact contract pins")

    expected_provenance = {
        "common": contract["provenance"]["common"],
        "source_commits": contract["provenance"]["source_commits"],
        "source_git_trees": contract["provenance"]["source_git_trees"],
        "source_trees_sha256": contract["provenance"]["source_trees_sha256"],
        "topology": contract["provenance"]["topology"],
        "arm": contract["provenance"]["arms"][arm],
    }
    if value["provenance"] != expected_provenance:
        raise EvidenceError(f"{arm} W&B provenance differs from exact contract pins")
    if value["config"] != contract["configs"][arm]:
        raise EvidenceError(f"{arm} W&B config differs from exact contract pins")
    return Run(
        arm=arm,
        identity=identity,
        history=merge_sparse_history(
            value["history"],
            label=arm,
            verifier_metric=contract["verifier_metric"],
        ),
    )


def _semantic_subset(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise EvidenceError(f"{path} must be an object")
        for key, value in expected.items():
            if key not in actual:
                raise EvidenceError(f"{path} lacks semantic pin {key!r}")
            _semantic_subset(actual[key], value, f"{path}.{key}")
        return
    if actual != expected:
        raise EvidenceError(f"{path} differs from semantic pin {expected!r}")


def _authenticate_receipt(
    document: Document,
    pin: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if document.sha256 != pin["sha256"]:
        raise EvidenceError(f"{label} bytes differ from the pinned SHA-256")
    _semantic_subset(document.value, pin["semantic_pins"], label)
    return document.value


def _required_artifact(artifacts: Mapping[str, Any], key: str, label: str) -> Document:
    errors = artifacts.get("__errors__", {})
    if isinstance(errors, Mapping) and key in errors:
        raise EvidenceError(f"{label} is unavailable: {errors[key]}")
    document = artifacts.get(key)
    if not isinstance(document, Document):
        raise EvidenceError(f"missing {label}")
    return document


def _validate_holdout(
    document: Document, contract: Mapping[str, Any]
) -> dict[str, Any]:
    expected = contract["holdout"]
    if document.sha256 != expected["receipt_sha256"]:
        raise EvidenceError("holdout receipt bytes differ from the pinned SHA-256")
    receipt = document.value
    required = {
        "schema": "nemorl-single-env-reward-liveness-holdout-v1",
        "contract_sha256": contract["provenance"]["common"][
            "reward_liveness_contract_sha256"
        ],
        "environment": contract["pair"]["environment"],
        "selected_fixture_sha256": contract["provenance"]["common"]["fixture_sha256"],
        "frozen_reward_primary_mean_min": expected["primary_reward_mean_min"],
        "frozen_reward_tail_mean_min": expected["tail_reward_mean_min"],
        "eligible": True,
    }
    _semantic_subset(receipt, required, "holdout receipt")
    _digest(
        receipt.get("selection_receipt_sha256"), "holdout selection receipt SHA-256"
    )
    _digest(
        receipt.get("holdout_observation_sha256"),
        "holdout observation SHA-256",
    )
    return receipt


def _validate_step_receipts(
    contract: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> None:
    pair = contract["pair"]
    receipts = contract["receipts"]
    same = _authenticate_receipt(
        _required_artifact(
            artifacts, "same_arm", "same-arm fresh reproducibility receipt"
        ),
        receipts["external_step1_same_arm_reproducibility_receipt"],
        label="same-arm fresh reproducibility receipt",
    )
    same_required = {
        "scope": "same_arm_fresh_process_reproducibility",
        "status": "PASS",
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "step": 1,
        "sample_count": 4,
        "fresh_process": True,
        "reproducible": True,
    }
    _semantic_subset(same, same_required, "same-arm fresh reproducibility receipt")
    if "off_on" in str(same.get("scope", "")):
        raise EvidenceError(
            "same-arm reproducibility receipt improperly claims cross-arm parity"
        )

    cross = _authenticate_receipt(
        _required_artifact(artifacts, "cross_arm", "cross-arm step-1 parity receipt"),
        receipts["external_step1_off_on_parity_receipt"],
        label="cross-arm step-1 parity receipt",
    )
    cross_required = {
        "scope": "off_on_step1_row_parity",
        "status": "PASS",
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "step": 1,
        "sample_count": 4,
        "off_run_id": pair["run_ids"]["off"],
        "on_run_id": pair["run_ids"]["on"],
        "ordered_rows_equal": True,
        "compared_fields": list(CROSS_ARM_PARITY_FIELDS),
    }
    _semantic_subset(cross, cross_required, "cross-arm step-1 parity receipt")
    row_digest = _digest(cross.get("ordered_rows_sha256"), "cross-arm ordered rows")
    if row_digest != contract["provenance"]["common"]["paired_step1_ledger_sha256"]:
        raise EvidenceError(
            "cross-arm ordered rows are not bound by paired_step1_ledger_sha256"
        )
    if row_digest == contract["provenance"]["common"]["fixture_sha256"]:
        raise EvidenceError("cross-arm row digest aliases the input fixture digest")


def _validate_execution_receipts(
    contract: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> None:
    pair = contract["pair"]
    receipt_groups = contract["receipts"]
    for arm, mode in (("off", "observe"), ("on", "train")):
        marker = _authenticate_receipt(
            _required_artifact(
                artifacts,
                f"{arm}_execution",
                f"{arm} shared-prefix execution receipt",
            ),
            receipt_groups["shared_prefix_execution_marker_receipts"][arm],
            label=f"{arm} shared-prefix execution receipt",
        )
        _semantic_subset(
            marker,
            {
                "schema": EXECUTION_MARKER_RECEIPT_SCHEMA,
                "scope": "shared_prefix_physical_execution",
                "marker_semantics": EXECUTION_MARKER_SEMANTICS,
                "status": "PASS",
                "pair_id": pair["pair_id"],
                "environment": pair["environment"],
                "arm": arm,
                "shared_prefix_mode": mode,
                "shared_prefix_runtime_trace_sha256": contract["provenance"]["arms"][
                    arm
                ]["shared_prefix_runtime_trace_sha256"],
            },
            f"{arm} shared-prefix execution receipt",
        )
        count = _integer(
            marker.get("shared_prefix_execution_marker_count"),
            f"{arm} shared-prefix execution marker count",
        )
        if arm == "off" and count != 0:
            raise EvidenceError("OFF has a shared-prefix physical execution marker")
        if arm == "on" and count <= 0:
            raise EvidenceError("ON lacks a shared-prefix physical execution marker")

        job = _authenticate_receipt(
            _required_artifact(
                artifacts, f"{arm}_job_exit", f"{arm} strict job EXIT receipt"
            ),
            receipt_groups["strict_job_exit_receipts"][arm],
            label=f"{arm} strict job EXIT receipt",
        )
        _semantic_subset(
            job,
            {
                "schema": JOB_RECEIPT_SCHEMA,
                "phase": "EXIT",
                "post_verified": True,
                "driver_exit_code": 0,
                "pair_id": pair["pair_id"],
                "arm": arm,
                "runtime_attestation_expected_count": 4,
                "runtime_attestation_actual_count": 4,
                "pair_manifest_sha256": contract["provenance"]["common"][
                    "pair_manifest_sha256"
                ],
                "fixture_sha256": contract["provenance"]["common"]["fixture_sha256"],
                "fixture_rows": 5,
                "model_tree_sha256_v1": contract["provenance"]["common"][
                    "model_tree_sha256"
                ],
                "container_sha256": contract["provenance"]["common"][
                    "training_container_sha256"
                ],
                "sandbox_container_sha256": contract["provenance"]["common"][
                    "sandbox_container_sha256"
                ],
                "source_head": contract["provenance"]["source_commits"]["nemo_rl"],
                "source_tree": contract["provenance"]["source_git_trees"]["nemo_rl"],
                "config_sha256": contract["provenance"]["common"]["base_recipe_sha256"],
                "reward_semantics_config_sha256": contract["provenance"]["common"][
                    "base_recipe_sha256"
                ],
                "reward_semantics_contract_sha256": contract["provenance"]["common"][
                    "reward_semantics_contract_sha256"
                ],
                "nemo_runnable_manifest_sha256": contract["provenance"]["common"][
                    "nemo_runnable_manifest_sha256"
                ],
                "bridge_runnable_manifest_sha256": contract["provenance"]["common"][
                    "bridge_runnable_manifest_sha256"
                ],
                "mcore_runnable_manifest_sha256": contract["provenance"]["common"][
                    "mcore_runnable_manifest_sha256"
                ],
                "deployment_ready_file_sha256": contract["provenance"]["common"][
                    "deployment_ready_file_sha256"
                ],
                "snapshot_manifest_sha256": contract["provenance"]["arms"][arm][
                    "snapshot_manifest_sha256"
                ],
                "entrypoint_sha256": contract["provenance"]["arms"][arm][
                    "entrypoint_sha256"
                ],
                "wrapper_sha256": contract["provenance"]["arms"][arm]["wrapper_sha256"],
                "inner_ray_sha256": contract["provenance"]["arms"][arm][
                    "inner_ray_sha256"
                ],
                "command_sha256": contract["provenance"]["arms"][arm]["command_sha256"],
                "mounts_sha256": contract["provenance"]["arms"][arm]["mounts_sha256"],
            },
            f"{arm} strict job EXIT receipt",
        )
        pre_receipt_sha256 = _digest(
            job.get("pre_receipt_sha256"), f"{arm} PRE receipt hash"
        )
        if (
            pre_receipt_sha256
            == _required_artifact(
                artifacts, f"{arm}_job_exit", f"{arm} strict job EXIT receipt"
            ).sha256
        ):
            raise EvidenceError(f"{arm} PRE and EXIT receipt hashes alias")
        hashes = _mapping(
            job.get("runtime_attestation_receipts_sha256"),
            f"{arm} runtime attestation receipt hashes",
        )
        expected_names = {
            f"shared_prefix_determinism.{mode}.rank-{rank}.receipt" for rank in range(4)
        }
        _exact_keys(hashes, expected_names, f"{arm} runtime attestation receipts")
        for name, digest in hashes.items():
            _digest(digest, f"{arm} runtime receipt {name}")
        _digest(
            job.get("runtime_attestation_aggregate_sha256"),
            f"{arm} runtime attestation aggregate",
        )


def _require_histories(
    gate: Gate,
    runs: Mapping[str, Run],
    metrics: set[str] | frozenset[str] | Sequence[str],
    *,
    steps: Sequence[int] = STEPS,
) -> bool:
    ok = True
    for arm in ("off", "on"):
        try:
            runs[arm].history.require(metrics, steps=steps)
        except EvidenceError as error:
            gate.unverifiable(f"{arm}: {error}")
            ok = False
    return ok


def _validate_token_geometry(
    gate: Gate, run: Run, *, steps: Sequence[int] = STEPS
) -> None:
    history = run.history
    for step in steps:
        samples = history.integer(step, "train/rollout/samples")
        valid_samples = history.integer(step, "train/num_valid_samples")
        valid_sequences = history.integer(step, "train/global_valid_seqs")
        valid_tokens = history.integer(step, "train/global_valid_toks")
        total_tokens = history.integer(step, "train/total_num_tokens")
        shared_total = history.integer(step, "train/shared_prefix/total_tokens")
        shared_valid = history.integer(step, "train/shared_prefix/valid_loss_tokens")
        prompt = history.integer(step, "train/shared_prefix/prompt_tokens")
        suffix = history.integer(step, "train/shared_prefix/non_loss_suffix_tokens")
        shareable = history.integer(step, "train/shared_prefix/shareable_prompt_tokens")
        ideal_work = history.integer(
            step, "train/shared_prefix/ideal_shared_token_work"
        )
        exact_counts = {
            "train/rollout/samples": samples,
            "train/num_valid_samples": valid_samples,
            "train/global_valid_seqs": valid_sequences,
            "train/shared_prefix/total_sequences": history.integer(
                step, "train/shared_prefix/total_sequences"
            ),
            "train/shared_prefix/eligible_sequences": history.integer(
                step, "train/shared_prefix/eligible_sequences"
            ),
        }
        if any(value != K4_SAMPLES for value in exact_counts.values()):
            gate.fail(
                f"{run.arm} step {step} does not preserve the exact K=4 cohort: "
                f"{exact_counts}"
            )
        if history.integer(step, "train/shared_prefix/complete_groups") != 1:
            gate.fail(f"{run.arm} step {step} lacks one complete shared-prefix group")
        if (
            history.integer(step, "train/shared_prefix/fallback_sequences") != 0
            or history.integer(step, "train/shared_prefix/runtime_fallback_sequences")
            != 0
        ):
            gate.fail(f"{run.arm} step {step} used shared-prefix fallback")
        if not 0 < valid_tokens <= total_tokens:
            gate.fail(f"{run.arm} step {step} global_valid_toks is outside (0,total]")
        if total_tokens != shared_total or valid_tokens != shared_valid:
            gate.fail(
                f"{run.arm} step {step} global/shared-prefix token sources disagree"
            )
        if prompt + valid_tokens + suffix != total_tokens:
            gate.fail(f"{run.arm} step {step} prompt/valid/suffix tokens do not close")
        if not 0 < shareable <= prompt or ideal_work != total_tokens - shareable:
            gate.fail(f"{run.arm} step {step} ideal shared-token work does not close")
        reduction = history.number(step, "train/shared_prefix/ideal_token_reduction")
        speedup = history.number(step, "train/shared_prefix/ideal_token_work_speedup")
        if not _close(
            reduction, shareable / total_tokens, tolerance=RATIO_ABS_TOLERANCE
        ):
            gate.fail(f"{run.arm} step {step} ideal token reduction does not close")
        if not _close(
            speedup, total_tokens / ideal_work, tolerance=RATIO_ABS_TOLERANCE
        ):
            gate.fail(f"{run.arm} step {step} ideal token speedup does not close")


def _evaluate_reward(
    contract: Mapping[str, Any],
    runs: Mapping[str, Run],
    artifacts: Mapping[str, Any],
) -> Gate:
    gate = Gate("reward_correctness")
    verifier_metric = contract["verifier_metric"]
    required = {
        *REWARD_METRICS,
        *TOKEN_METRICS,
        *ZERO_STEP_METRICS,
        *CUMULATIVE_ZERO_METRICS,
        verifier_metric,
    }
    if not _require_histories(gate, runs, required):
        return gate
    try:
        _validate_step_receipts(contract, artifacts)
    except (EvidenceError, KeyError) as error:
        gate.unverifiable(str(error))

    for run in runs.values():
        _validate_token_geometry(gate, run)
        history = run.history
        for step in STEPS:
            raw = history.number(step, "train/raw_environment_reward")
            raw_min = history.number(step, "train/raw_environment_reward/min")
            raw_max = history.number(step, "train/raw_environment_reward/max")
            pre = history.number(step, "train/pre_penalty_environment_reward")
            pre_min = history.number(step, "train/pre_penalty_environment_reward/min")
            pre_max = history.number(step, "train/pre_penalty_environment_reward/max")
            effective = history.number(step, "train/verifier_reward")
            effective_alias = history.number(step, "train/total_reward/mean")
            effective_min = history.number(step, "train/total_reward/min")
            effective_max = history.number(step, "train/total_reward/max")
            processed = history.number(step, "train/reward")
            processing_delta = history.number(step, "train/reward_processing_delta")
            effort_delta = history.number(step, "train/effort_reward_delta")
            boundaries = (
                ("raw", raw, raw_min, raw_max),
                ("pre-penalty", pre, pre_min, pre_max),
                ("effective", effective, effective_min, effective_max),
            )
            for label, mean, minimum, maximum in boundaries:
                if minimum not in (0.0, 1.0) or maximum not in (0.0, 1.0):
                    gate.fail(f"{run.arm} step {step} {label} min/max is not binary")
                if minimum > mean or mean > maximum:
                    gate.fail(f"{run.arm} step {step} {label} mean is outside min/max")
                if not _close(
                    mean * K4_SAMPLES,
                    round(mean * K4_SAMPLES),
                    tolerance=REWARD_ABS_TOLERANCE,
                ):
                    gate.fail(
                        f"{run.arm} step {step} {label} mean is not a K=4 fraction"
                    )
            equations = {
                "raw verifier alias": (history.number(step, verifier_metric), raw),
                "raw to pre-penalty delta": (pre - raw, effort_delta),
                "effective alias": (effective, effective_alias),
                "effective to processed delta": (
                    processed - effective,
                    processing_delta,
                ),
                "disabled reward processing": (processing_delta, 0.0),
                "processed reward": (processed, effective),
            }
            for label, (observed, expected) in equations.items():
                if not _close(observed, expected, tolerance=REWARD_ABS_TOLERANCE):
                    gate.fail(f"{run.arm} step {step} violates {label}")
            exact_zero = {
                "train/effort_low_sample_count",
                "train/effort_low_sample_rate",
                "train/effort_reward_delta",
                "train/num_mask_sample_filtered",
                "train/mask_sample_rate",
                *PENALTY_METRICS,
                *ZERO_STEP_METRICS,
                *CUMULATIVE_ZERO_METRICS,
            }
            for metric in exact_zero:
                if history.number(step, metric) != 0.0:
                    gate.fail(f"{run.arm} step {step} {metric} is nonzero")
            if any(
                not _close(left, right, tolerance=REWARD_ABS_TOLERANCE)
                for left, right in (
                    (raw, pre),
                    (raw_min, pre_min),
                    (raw_max, pre_max),
                    (pre, effective),
                    (pre_min, effective_min),
                    (pre_max, effective_max),
                )
            ):
                gate.fail(f"{run.arm} step {step} zero-incidence reward stages differ")

    step_one_metrics = sorted(required - {"timing/train/policy_training"})
    parity_mismatches = [
        metric
        for metric in step_one_metrics
        if runs["off"].history.values[1][metric] != runs["on"].history.values[1][metric]
    ]
    if parity_mismatches:
        gate.fail(
            "W&B aggregate step-1 parity differs for "
            + ", ".join(parity_mismatches[:20])
        )
    gate.evidence = {
        "steps_checked_per_arm": len(STEPS),
        "external_same_arm_reproducibility": "required_separately",
        "external_cross_arm_step1_row_parity": "required_separately",
        "wandb_aggregate_step1_exact": not parity_mismatches,
        "token_key": "train/global_valid_toks",
    }
    return gate


def _epoch_numbers(steps: Sequence[int]) -> list[int]:
    return sorted({(step - 1) // STEPS_PER_EPOCH + 1 for step in steps})


def _evaluate_learning(
    contract: Mapping[str, Any],
    runs: Mapping[str, Run],
    artifacts: Mapping[str, Any],
) -> Gate:
    gate = Gate("learning_behavior")
    if not _require_histories(gate, runs, LEARNING_METRICS):
        return gate
    try:
        holdout = _required_artifact(artifacts, "holdout", "holdout receipt")
        _validate_holdout(holdout, contract)
    except EvidenceError as error:
        gate.unverifiable(str(error))
        return gate

    arm_evidence: dict[str, Any] = {}
    primary_floor = float(contract["holdout"]["primary_reward_mean_min"])
    tail_floor = float(contract["holdout"]["tail_reward_mean_min"])
    for run in runs.values():
        history = run.history
        raw_means = [
            history.number(step, "train/raw_environment_reward") for step in STEPS
        ]
        effective_means = [
            history.number(step, "train/total_reward/mean") for step in STEPS
        ]
        mixed_steps = [
            step
            for step in STEPS
            if history.number(step, "train/raw_environment_reward/min")
            < history.number(step, "train/raw_environment_reward/max")
            and history.number(step, "train/total_reward/min")
            < history.number(step, "train/total_reward/max")
            and history.number(step, "train/advantages/min") < 0.0
            and history.number(step, "train/advantages/max") > 0.0
        ]
        active_epochs = _epoch_numbers(mixed_steps)
        tail_active = [epoch for epoch in active_epochs if epoch in TAIL_EPOCHS]
        policy_epochs = _epoch_numbers(
            [step for step in STEPS if history.number(step, "train/grad_norm") > 0.0]
        )
        mtp_epochs = _epoch_numbers(
            [
                step
                for step in STEPS
                if history.number(step, "train/mtp/grad_norm") > 0.0
            ]
        )
        joint_tail = sorted(TAIL_EPOCHS & set(policy_epochs) & set(mtp_epochs))
        raw_distinct = len(set(raw_means))
        effective_distinct = len(set(effective_means))
        unique_means = len(set(effective_means))
        reward_stddev = statistics.pstdev(effective_means)
        primary_mean = statistics.fmean(
            history.number(step, "train/total_reward/mean") for step in PRIMARY_STEPS
        )
        tail_mean = statistics.fmean(
            history.number(step, "train/total_reward/mean") for step in TAIL_STEPS
        )
        checks = (
            (
                raw_distinct >= 2,
                f"raw reward has {raw_distinct} distinct values; requires >=2",
            ),
            (
                effective_distinct >= 2,
                f"effective reward has {effective_distinct} distinct values; requires >=2",
            ),
            (
                len(mixed_steps) >= 20,
                f"mixed K4 steps={len(mixed_steps)}; requires >=20",
            ),
            (
                len(active_epochs) >= 16,
                f"active epochs={len(active_epochs)}; requires >=16",
            ),
            (
                len(tail_active) >= 1,
                f"tail active epochs={tail_active}; requires at least one",
            ),
            (
                bool(mixed_steps) and mixed_steps[0] <= 10,
                f"first mixed step={mixed_steps[0] if mixed_steps else None}; requires <=10",
            ),
            (
                unique_means >= 3,
                f"unique effective step means={unique_means}; requires >=3",
            ),
            (
                reward_stddev >= 0.05,
                f"effective step-mean population stddev={reward_stddev}; requires >=0.05",
            ),
            (
                len(policy_epochs) >= 16,
                f"policy-gradient active epochs={len(policy_epochs)}; requires >=16",
            ),
            (
                len(mtp_epochs) >= 16,
                f"MTP-gradient active epochs={len(mtp_epochs)}; requires >=16",
            ),
            (
                bool(joint_tail),
                "tail lacks a joint policy/MTP gradient epoch",
            ),
            (
                primary_mean >= primary_floor,
                f"primary reward mean={primary_mean}; floor={primary_floor}",
            ),
            (
                tail_mean >= tail_floor,
                f"tail reward mean={tail_mean}; floor={tail_floor}",
            ),
        )
        for passed, message in checks:
            if not passed:
                gate.fail(f"{run.arm}: {message}")
        penalty_max = max(
            history.number(step, metric) for step in STEPS for metric in PENALTY_METRICS
        )
        if penalty_max > 0.05:
            gate.fail(f"{run.arm}: penalty-rate maximum={penalty_max}; requires <=0.05")
        arm_evidence[run.arm] = {
            "mixed_steps": mixed_steps,
            "active_epochs": active_epochs,
            "tail_active_epochs": tail_active,
            "policy_gradient_active_epochs": policy_epochs,
            "mtp_gradient_active_epochs": mtp_epochs,
            "joint_gradient_tail_epochs": joint_tail,
            "raw_reward_distinct_values": raw_distinct,
            "effective_reward_distinct_values": effective_distinct,
            "unique_effective_step_means": unique_means,
            "effective_step_mean_population_stddev": reward_stddev,
            "primary_effective_reward_mean": primary_mean,
            "tail_effective_reward_mean": tail_mean,
            "holdout_primary_floor": primary_floor,
            "holdout_tail_floor": tail_floor,
            "penalty_rate_max": penalty_max,
        }
    gate.evidence = {"arms": arm_evidence}
    return gate


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise EvidenceError("invalid percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_epoch_median_ci(
    blocks: Sequence[Sequence[float]], samples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    if samples != BOOTSTRAP_RESAMPLES:
        raise EvidenceError("bootstrap resamples differ from the frozen 10,000")
    if not blocks or any(len(block) != STEPS_PER_EPOCH for block in blocks):
        raise EvidenceError("bootstrap requires complete five-step epoch blocks")
    generator = random.Random(BOOTSTRAP_SEED)
    size = len(blocks)
    medians = []
    for _ in range(samples):
        draw = [blocks[generator.randrange(size)] for _ in range(size)]
        medians.append(statistics.median(value for block in draw for value in block))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _same_work(off: History, on: History, step: int) -> bool:
    return all(
        off.values[step][metric] == on.values[step][metric] for metric in WORK_METRICS
    )


def _evaluate_speed(
    contract: Mapping[str, Any],
    runs: Mapping[str, Run],
    artifacts: Mapping[str, Any],
) -> Gate:
    gate = Gate("speed_evidence")
    if not _require_histories(gate, runs, SPEED_METRICS, steps=PRIMARY_STEPS):
        return gate
    try:
        _validate_execution_receipts(contract, artifacts)
    except (EvidenceError, KeyError) as error:
        gate.unverifiable(str(error))
        return gate
    for run in runs.values():
        token_gate = Gate("speed_token_geometry")
        _validate_token_geometry(token_gate, run, steps=PRIMARY_STEPS)
        for message in token_gate.failures:
            gate.unverifiable(message)
        for step in PRIMARY_STEPS:
            if run.history.number(step, "timing/train/policy_training") <= 0.0:
                gate.unverifiable(
                    f"{run.arm} step {step} policy-training time is not positive"
                )
    if gate.unavailable:
        return gate

    matched_epochs = [
        epoch
        for epoch in PRIMARY_EPOCHS
        if all(
            _same_work(runs["off"].history, runs["on"].history, step) for step in epoch
        )
    ]
    excluded_epochs = [epoch for epoch in PRIMARY_EPOCHS if epoch not in matched_epochs]
    if len(matched_epochs) < MIN_MATCHED_PRIMARY_EPOCHS:
        gate.fail(
            f"complete matched primary epochs={len(matched_epochs)}; requires >=15/18"
        )
    ratios_by_step = {
        step: runs["off"].history.number(step, "timing/train/policy_training")
        / runs["on"].history.number(step, "timing/train/policy_training")
        for epoch in matched_epochs
        for step in epoch
    }
    if not ratios_by_step:
        gate.unverifiable("no complete matched primary epoch remains for speed")
        return gate
    blocks = [[ratios_by_step[step] for step in epoch] for epoch in matched_epochs]
    try:
        low, high = _bootstrap_epoch_median_ci(blocks)
    except EvidenceError as error:
        gate.unverifiable(str(error))
        return gate
    ratio_values = list(ratios_by_step.values())
    median = statistics.median(ratio_values)
    geomean = math.exp(statistics.fmean(math.log(value) for value in ratio_values))
    if not low > 1.0:
        gate.fail(f"bootstrap 95% speed CI lower={low} is not strictly >1.0")
    gate.evidence = {
        "primary_window_steps": [11, 100],
        "matched_epoch_numbers": [
            (epoch[0] - 1) // STEPS_PER_EPOCH + 1 for epoch in matched_epochs
        ],
        "excluded_epoch_numbers": [
            (epoch[0] - 1) // STEPS_PER_EPOCH + 1 for epoch in excluded_epochs
        ],
        "matched_steps": sorted(ratios_by_step),
        "ratio": "OFF policy-training seconds / ON policy-training seconds",
        "median": median,
        "geomean": geomean,
        "bootstrap_95_ci": {
            "low": low,
            "high": high,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "complete_matched_five_step_epoch",
            "statistic": "paired_step_median",
        },
        "decision": "PASS only when CI lower is strictly greater than 1.0",
    }
    return gate


def _unverifiable_report(
    contract: Mapping[str, Any] | None, message: str
) -> dict[str, Any]:
    pair = contract.get("pair", {}) if isinstance(contract, Mapping) else {}
    gate = {
        "status": "UNVERIFIABLE",
        "failures": [],
        "unavailable": [message],
        "evidence": {},
    }
    return {
        "schema": REPORT_SCHEMA,
        "pair_id": pair.get("pair_id"),
        "environment": pair.get("environment"),
        "reward_correctness": dict(gate),
        "learning_behavior": dict(gate),
        "speed_evidence": dict(gate),
        "overall": {
            "status": "UNVERIFIABLE",
            "independent_statuses": {
                "reward_correctness": "UNVERIFIABLE",
                "learning_behavior": "UNVERIFIABLE",
                "speed_evidence": "UNVERIFIABLE",
            },
        },
    }


def evaluate_pair(
    contract_value: dict[str, Any],
    off_export: dict[str, Any],
    on_export: dict[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all gates without allowing one result to mask another."""
    try:
        contract = validate_contract(contract_value)
        runs = {
            "off": _validate_run(off_export, contract, "off"),
            "on": _validate_run(on_export, contract, "on"),
        }
    except EvidenceError as error:
        return _unverifiable_report(contract_value, str(error))

    reward = _evaluate_reward(contract, runs, artifacts)
    learning = _evaluate_learning(contract, runs, artifacts)
    speed = _evaluate_speed(contract, runs, artifacts)
    statuses = {
        "reward_correctness": reward.status,
        "learning_behavior": learning.status,
        "speed_evidence": speed.status,
    }
    if all(status == "PASS" for status in statuses.values()):
        overall = "GREEN"
    elif any(status == "FAIL" for status in statuses.values()):
        overall = "RED"
    else:
        overall = "UNVERIFIABLE"
    return {
        "schema": REPORT_SCHEMA,
        "pair_id": contract["pair"]["pair_id"],
        "environment": contract["pair"]["environment"],
        "reward_correctness": reward.report(),
        "learning_behavior": learning.report(),
        "speed_evidence": speed.report(),
        "overall": {"status": overall, "independent_statuses": statuses},
    }


def _text_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"overall={report['overall']['status']}",
        f"pair_id={report.get('pair_id')}",
        f"environment={report.get('environment')}",
    ]
    for name in ("reward_correctness", "learning_behavior", "speed_evidence"):
        gate = report[name]
        lines.append(f"{name}={gate['status']}")
        for message in gate["unavailable"]:
            lines.append(f"  unavailable: {message}")
        for message in gate["failures"]:
            lines.append(f"  failure: {message}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--off-export", type=Path, required=True)
    parser.add_argument("--on-export", type=Path, required=True)
    parser.add_argument("--holdout-receipt", type=Path)
    parser.add_argument("--same-arm-repro-receipt", type=Path)
    parser.add_argument("--cross-arm-step1-receipt", type=Path)
    parser.add_argument("--off-execution-receipt", type=Path)
    parser.add_argument("--on-execution-receipt", type=Path)
    parser.add_argument("--off-job-exit-receipt", type=Path)
    parser.add_argument("--on-job-exit-receipt", type=Path)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    args = parser.parse_args(argv)
    try:
        contract = load_document(args.contract, "acceptance contract")
        off = load_document(args.off_export, "OFF W&B export")
        on = load_document(args.on_export, "ON W&B export")
        artifact_arguments = (
            ("holdout", args.holdout_receipt, "holdout receipt"),
            (
                "same_arm",
                args.same_arm_repro_receipt,
                "same-arm reproducibility receipt",
            ),
            ("cross_arm", args.cross_arm_step1_receipt, "cross-arm step-1 receipt"),
            ("off_execution", args.off_execution_receipt, "OFF execution receipt"),
            ("on_execution", args.on_execution_receipt, "ON execution receipt"),
            ("off_job_exit", args.off_job_exit_receipt, "OFF job EXIT receipt"),
            ("on_job_exit", args.on_job_exit_receipt, "ON job EXIT receipt"),
        )
        artifacts: dict[str, Any] = {}
        artifact_errors: dict[str, str] = {}
        for key, path, label in artifact_arguments:
            if path is None:
                artifact_errors[key] = "path was not provided"
                continue
            try:
                artifacts[key] = load_document(path, label)
            except EvidenceError as error:
                artifact_errors[key] = str(error)
        if artifact_errors:
            artifacts["__errors__"] = artifact_errors
        report = evaluate_pair(contract.value, off.value, on.value, artifacts)
    except EvidenceError as error:
        report = _unverifiable_report(None, str(error))
    if args.format == "json":
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(_text_report(report))
    status = report["overall"]["status"]
    return 0 if status == "GREEN" else 1 if status == "RED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
