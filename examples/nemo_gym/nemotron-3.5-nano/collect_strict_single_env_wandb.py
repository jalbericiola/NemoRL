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

"""Collect one authenticated W&B run for strict single-environment acceptance.

The collector deliberately has no W&B import at module import time.  A caller
must authenticate immutable Pair, submission, and acceptance documents first;
the W&B dependency and network are needed only after those local trust anchors
have selected one deterministic run ID.

The output is sorted compact ASCII JSON followed by exactly one LF.  Its
``export_receipt.canonical_sha256`` authenticates the seven non-receipt root
members, avoiding a self-referential digest.  The CLI also prints the SHA-256
of the complete file so it can be carried to the offline evaluator out of band.

The separate ``diagnostic-local-summary-pair`` command reads two local
``wandb-summary.json`` files.  That document contains one latest point, never a
history curve, and is marked non-acceptance at its schema, assurance, capture,
and boundary layers.  Acceptance-grade OFF/ON reward and speed curves continue
to come from two authenticated v2 history exports combined only by the offline
evaluator.  The ``diagnostic-wandb-history-pair`` command emits exploratory
live-rollout curves, but it remains explicitly ineligible for acceptance
because arbitrary diagnostic run IDs lack the strict Pair and scheduler
provenance chain.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_EXPORT_SCHEMA = "nemo-rl-offline-wandb-run-export-v2"
PAIR_MANIFEST_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
SUBMISSION_RECEIPT_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
ACCEPTANCE_CONTRACT_SCHEMA = "nemo-rl-strict-single-env-acceptance-contract-v2"
WANDB_API_BASE_URL = "https://api.wandb.ai"
WANDB_HISTORY_METHOD = "scan_history"
WANDB_SDK_VERSION = "0.28.1"
WANDB_EXPORT_CANONICALIZATION = "sorted-compact-ascii-json-plus-one-lf"
WANDB_RUN_ID_DERIVATION = "sha256-ascii:nemo-rl-strict-wandb-v1:{environment}:{pair_id}:{arm}"
DIAGNOSTIC_PAIR_SUMMARY_SCHEMA = "nemo-rl-diagnostic-local-wandb-pair-summary-v1"
DIAGNOSTIC_PAIR_SUMMARY_ASSURANCE = "non_acceptance_latest_local_summary_only_not_history_or_a1_evidence"
DIAGNOSTIC_PAIR_SUMMARY_METHOD = "local-wandb-summary-json"
DIAGNOSTIC_PAIR_HISTORY_SCHEMA = "nemo-rl-diagnostic-wandb-history-pair-v1"
DIAGNOSTIC_PAIR_HISTORY_ASSURANCE = "non_acceptance_exploratory_wandb_history_not_scheduler_or_pair_authenticated"
MAX_EXACT_INTEGER = 1 << 53
MAX_HISTORY_ROWS = 100_000
TRUSTED_OOB_DECLARATIONS_SCHEMA = "nemo-rl-strict-trusted-oob-declarations-v1"
TRUSTED_OOB_DECLARATIONS_ASSURANCE = "lineage_only_not_runtime_observation_or_correctness_evidence"
TRUSTED_DECLARED_SOURCE_KEYS = frozenset({"nemo_automodel", "pipeclean"})
TRUSTED_DECLARED_COMMON_KEYS = frozenset(
    {
        "arm_normalized_resolved_config_sha256",
        "arm_normalized_runtime_environment_sha256",
        "base_recipe_sha256",
        "dataset_sha256",
        "launch_package_manifest_sha256",
        "rendered_plan_sha256",
        "source_bundle_manifest_sha256",
        "wave_manifest_sha256",
    }
)
TRUSTED_DECLARED_ARM_KEYS = frozenset({"code_snapshot_sha256", "reward_ledger_sha256", "train_data_index_sha256"})
SOURCE_KEYS = frozenset({"megatron_bridge", "megatron_lm", "nemo_gym", "nemo_rl"})
ACCEPTANCE_CONFIG_KEYS = frozenset(
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
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
    }
)
COMMON_PROVENANCE_KEYS = frozenset(
    {
        "acceptance_contract_sha256",
        "bridge_runnable_manifest_sha256",
        "deployment_ready_file_sha256",
        "deployment_ready_sha256",
        "environment_recipe_sha256",
        "fixture_sha256",
        "gym_config_sha256",
        "launcher_sha256",
        "mcore_runnable_manifest_sha256",
        "model_tree_sha256",
        "nemo_runnable_manifest_sha256",
        "pair_campaign_reward_and_advantage_sha256",
        "pair_campaign_sha256",
        "pair_manifest_sha256",
        "reward_liveness_contract_sha256",
        "reward_semantics_contract_sha256",
        "runtime_tool_manifest_sha256",
        "sandbox_container_sha256",
        "strict_pair_arm_wrapper_sha256",
        "strict_pair_contract_sha256",
        "strict_pair_parent_wrapper_sha256",
        "submission_contract_sha256",
        "terminal_scheduler_collector_sha256",
        "training_container_sha256",
        "verifier_source_sha256",
        "wandb_exporter_sha256",
    }
)
ARM_PROVENANCE_KEYS = frozenset(
    {
        "command_sha256",
        "entrypoint_sha256",
        "inner_ray_sha256",
        "mounts_sha256",
        "runtime_direction_receipt_sha256",
        "runtime_environment_sha256",
        "shared_prefix_runtime_trace_sha256",
        "snapshot_manifest_sha256",
        "wrapper_sha256",
    }
)

HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

VERIFIER_METRIC_BY_ENVIRONMENT = {
    "citation": "train/citation_format_simple_agent/reward/mean",
    "freeform": "train/freeform_formatting_simple_agent/reward/mean",
    "reasoning_gym": "train/reasoning_gym_simple_agent/score/mean",
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
ZERO_STEP_METRICS = frozenset(
    {
        "train/num_masked_seqs_by_logprob_error",
        "train/shared_prefix/fallback_sequences",
        "train/shared_prefix/runtime_fallback_sequences",
        "train/dropped_prompt_groups",
        "train/replaced_prompt_groups",
        "train/promoted_prompt_groups",
        "train/evicted_stale_prompt_groups",
        "train/aborted_stale_inflight_groups",
    }
)
CUMULATIVE_ZERO_METRICS = frozenset(
    {
        "rollout/skipped_total",
        "rollout/redispatch_total",
        "rollout/data_retry_total",
        "rollout/data_failures_total",
        "rollout/gym_row_redispatch_total",
        "rollout/infra_drops_total",
        "rollout/max_consecutive_infra_drops",
    }
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
        "train/advantages/mean",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
        "train/mtp/grad_norm",
        "train/update_successful",
    }
)
SPEED_METRICS = frozenset({"timing/train/policy_training"})

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

# ``wandb-summary.json`` contains only the latest value of each metric.  This
# deliberately small inventory is useful for plotting a final diagnostic point
# while keeping the file completely outside the acceptance run-export schema.
DIAGNOSTIC_LOCAL_SUMMARY_METRICS = tuple(
    sorted(
        {
            "timing/train/policy_training",
            "train/global_valid_seqs",
            "train/global_valid_toks",
            "train/num_valid_samples",
            "train/raw_environment_reward",
            "train/raw_environment_reward/max",
            "train/raw_environment_reward/min",
            "train/rollout/samples",
            "train/shared_prefix/complete_groups",
            "train/shared_prefix/eligible_sequences",
            "train/shared_prefix/fallback_sequences",
            "train/shared_prefix/ideal_shared_token_work",
            "train/shared_prefix/ideal_token_reduction",
            "train/shared_prefix/ideal_token_work_speedup",
            "train/shared_prefix/shareable_prompt_tokens",
            "train/shared_prefix/total_tokens",
            "train/shared_prefix/total_sequences",
            "train/shared_prefix/valid_loss_tokens",
            "train/total_num_tokens",
        }
    )
)
DIAGNOSTIC_LOCAL_SUMMARY_INTEGER_METRICS = frozenset(set(DIAGNOSTIC_LOCAL_SUMMARY_METRICS) & set(INTEGER_METRICS))
DIAGNOSTIC_WORK_METRICS = (
    "train/num_valid_samples",
    "train/global_valid_toks",
    "train/total_num_tokens",
    "train/shared_prefix/total_sequences",
    "train/shared_prefix/eligible_sequences",
    "train/shared_prefix/complete_groups",
    "train/shared_prefix/shareable_prompt_tokens",
    "train/shared_prefix/ideal_token_reduction",
)
DIAGNOSTIC_TOKEN_COUNT_METRICS = (
    "train/num_valid_samples",
    "train/global_valid_seqs",
    "train/global_valid_toks",
    "train/total_num_tokens",
    "train/rollout/samples",
    "train/shared_prefix/total_sequences",
    "train/shared_prefix/eligible_sequences",
    "train/shared_prefix/complete_groups",
    "train/shared_prefix/fallback_sequences",
    "train/shared_prefix/total_tokens",
    "train/shared_prefix/valid_loss_tokens",
    "train/shared_prefix/shareable_prompt_tokens",
    "train/shared_prefix/ideal_shared_token_work",
)
DIAGNOSTIC_IGNORED_PROGRESS_FIELDS = ("rollout/train_steps",)
DIAGNOSTIC_LIMITATIONS = (
    "latest_values_only_no_history_or_curve_completeness",
    "per_metric_step_alignment_not_proven_by_wandb_summary",
    "run_identity_is_path_declared_not_scheduler_authenticated",
    "not_valid_for_reward_noninferiority_or_speed_acceptance",
)
DIAGNOSTIC_HISTORY_LIMITATIONS = (
    "arm_assignment_is_caller_declared",
    "remote_wandb_history_immutability_not_proven",
    "scan_iteration_completion_does_not_prove_remote_history_completeness",
    "run_identity_is_path_declared_not_scheduler_authenticated",
    "not_valid_for_reward_noninferiority_or_speed_acceptance",
)


class CollectionError(RuntimeError):
    """The requested run cannot produce authenticated acceptance evidence."""


@dataclass(frozen=True)
class Document:
    """One parsed JSON document together with its exact bytes and digest."""

    value: dict[str, Any]
    raw: bytes
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CollectionError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CollectionError(f"non-finite JSON constant {value}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise CollectionError(
            f"{label} fields differ: missing={sorted(expected - observed)}, " f"extra={sorted(observed - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectionError(f"{label} must be an object")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise CollectionError(f"{label} must be a non-empty safe identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise CollectionError(f"{label} must be a populated lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX40_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise CollectionError(f"{label} must be a populated lowercase Git object ID")
    return value


def _validate_hash_mapping(value: Any, required: frozenset[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    _exact_keys(mapping, set(required), label)
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise CollectionError(f"{label} has unsupported non-SHA pin {key!r}")
        _digest(item, f"{label}.{key}")
    return mapping


def _validate_trusted_oob_declarations(value: Any) -> Mapping[str, Any]:
    declarations = _mapping(value, "trusted OOB declarations")
    _exact_keys(
        declarations,
        {"arms", "assurance", "common", "schema", "sources"},
        "trusted OOB declarations",
    )
    if declarations["schema"] != TRUSTED_OOB_DECLARATIONS_SCHEMA:
        raise CollectionError("unexpected trusted OOB declaration schema")
    if declarations["assurance"] != TRUSTED_OOB_DECLARATIONS_ASSURANCE:
        raise CollectionError("unexpected trusted OOB declaration assurance")
    _validate_hash_mapping(
        declarations["common"],
        TRUSTED_DECLARED_COMMON_KEYS,
        "trusted OOB common declarations",
    )
    arms = _mapping(declarations["arms"], "trusted OOB arm declarations")
    _exact_keys(arms, {"off", "on"}, "trusted OOB arm declarations")
    for arm in ("off", "on"):
        _validate_hash_mapping(
            arms[arm],
            TRUSTED_DECLARED_ARM_KEYS,
            f"trusted OOB {arm} arm declarations",
        )
    sources = _mapping(declarations["sources"], "trusted OOB source declarations")
    _exact_keys(
        sources,
        set(TRUSTED_DECLARED_SOURCE_KEYS),
        "trusted OOB source declarations",
    )
    for name in TRUSTED_DECLARED_SOURCE_KEYS:
        record = _mapping(sources[name], f"trusted OOB {name} source declaration")
        _exact_keys(
            record,
            {"commit", "git_tree", "source_tree_sha256"},
            f"trusted OOB {name} source declaration",
        )
        _commit(record["commit"], f"trusted OOB {name} source commit")
        _commit(record["git_tree"], f"trusted OOB {name} source Git tree")
        _digest(
            record["source_tree_sha256"],
            f"trusted OOB {name} source-tree SHA-256",
        )
    return declarations


def _job_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or JOB_ID_RE.fullmatch(value) is None:
        raise CollectionError(f"{label} must be a positive ASCII-decimal job ID")
    return value


def _validate_json_numbers(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CollectionError(f"{label} contains a non-finite number")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise CollectionError(f"{label} contains forbidden negative zero")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CollectionError(f"{label} contains a non-string JSON key")
            _validate_json_numbers(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, f"{label}[{index}]")


def _normalize_json(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CollectionError(f"{label} contains a non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CollectionError(f"{label} contains a non-string JSON key")
            normalized[key] = _normalize_json(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise CollectionError(f"{label} contains a non-JSON value {type(value).__name__}")


def canonical_json_bytes(value: Any, label: str) -> bytes:
    """Return normalized sorted compact ASCII JSON without a trailing LF."""
    normalized = _normalize_json(value, label)
    try:
        raw = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CollectionError(f"{label} is not canonical ASCII JSON") from error
    if b"\n" in raw or b"\r" in raw:
        raise CollectionError(f"{label} canonical JSON contains a newline")
    return raw


def canonical_json_sha256(value: Any, label: str) -> str:
    """Hash normalized canonical JSON without a trailing LF."""
    return hashlib.sha256(canonical_json_bytes(value, label)).hexdigest()


def acceptance_contract_payload_sha256(contract: Mapping[str, Any]) -> str:
    """Hash the contract after omitting its non-self-referential payload pin."""
    payload = copy.deepcopy(dict(contract))
    provenance = _mapping(payload.get("provenance"), "acceptance provenance payload")
    common = _mapping(provenance.get("common"), "acceptance common payload")
    if "acceptance_contract_sha256" not in common:
        raise CollectionError("acceptance contract payload lacks acceptance_contract_sha256")
    del common["acceptance_contract_sha256"]
    return canonical_json_sha256(payload, "acceptance contract payload")


def load_document(path: Path, label: str) -> Document:
    """Load a JSON object, rejecting duplicate keys and non-finite numbers."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CollectionError(f"cannot read {label} {path}: {error}") from error
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise CollectionError(f"invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CollectionError(f"{label} must contain one JSON object")
    _validate_json_numbers(value, label)
    return Document(
        value=value,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def authenticate_document(
    document: Document,
    expected_sha256: str,
    *,
    label: str,
    canonical_lf: bool,
) -> dict[str, Any]:
    """Authenticate exact bytes and, where required, their canonical framing."""
    _digest(expected_sha256, f"trusted {label} SHA-256")
    if document.sha256 != expected_sha256:
        raise CollectionError(f"{label} differs from its trusted SHA-256")
    canonical = canonical_json_bytes(document.value, label)
    expected = canonical + b"\n" if canonical_lf else canonical
    if document.raw != expected:
        suffix = " plus exactly one LF" if canonical_lf else " without LF"
        raise CollectionError(f"{label} must be canonical ASCII JSON{suffix}")
    return document.value


def derive_wandb_run_id(environment: str, pair_id: str, arm: str) -> str:
    """Derive the immutable W&B ID named by the Pair manifest contract."""
    if arm not in {"off", "on"}:
        raise CollectionError(f"unsupported arm {arm!r}")
    payload = f"nemo-rl-strict-wandb-v1:{environment}:{pair_id}:{arm}"
    try:
        raw = payload.encode("ascii")
    except UnicodeEncodeError as error:
        raise CollectionError("W&B run-ID derivation inputs must be ASCII") from error
    return hashlib.sha256(raw).hexdigest()


def requested_metrics(environment: str) -> list[str]:
    """Return the frozen, sorted W&B history allowlist for one environment."""
    try:
        verifier_metric = VERIFIER_METRIC_BY_ENVIRONMENT[environment]
    except KeyError as error:
        raise CollectionError(f"unsupported strict environment {environment!r}") from error
    metrics = {
        "_step",
        *REWARD_METRICS,
        *ZERO_STEP_METRICS,
        *CUMULATIVE_ZERO_METRICS,
        *TOKEN_METRICS,
        *LEARNING_METRICS,
        *SPEED_METRICS,
        verifier_metric,
    }
    return sorted(metrics)


def authenticate_local_summary(document: Document, expected_sha256: str, *, label: str) -> Mapping[str, Any]:
    """Authenticate W&B-owned summary bytes without claiming canonical framing."""
    _digest(expected_sha256, f"trusted {label} SHA-256")
    actual_sha256 = hashlib.sha256(document.raw).hexdigest()
    if document.sha256 != actual_sha256:
        raise CollectionError(f"{label} Document digest differs from its exact bytes")
    if actual_sha256 != expected_sha256:
        raise CollectionError(f"{label} differs from its trusted SHA-256")
    try:
        parsed = json.loads(
            document.raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise CollectionError(f"invalid JSON in {label}: {error}") from error
    if not isinstance(parsed, dict):
        raise CollectionError(f"{label} must contain one JSON object")
    _validate_json_numbers(parsed, label)
    _exact_json_value(document.value, parsed, f"{label} parsed value")
    return parsed


def _validate_diagnostic_summary_path(
    path: Path,
    run_id: str,
    arm: str,
    *,
    pair_root: Path,
    require_existing_nonsymlink_path: bool,
) -> str:
    """Validate the diagnostic W&B path shape without claiming file provenance."""
    if not path.is_absolute() or not pair_root.is_absolute():
        raise CollectionError(f"{arm} local W&B summary and pair-root paths must be absolute")
    for candidate, label in ((pair_root, "pair root"), (path, "summary path")):
        rendered = candidate.as_posix()
        if rendered.startswith("//") or ".." in candidate.parts:
            raise CollectionError(f"diagnostic {arm} {label} must be a canonical absolute path")
    try:
        path.relative_to(pair_root / arm)
    except ValueError as error:
        raise CollectionError(f"{arm} local W&B summary is outside the declared arm root") from error
    if path.name != "wandb-summary.json" or path.parent.name != "files":
        raise CollectionError(f"{arm} local W&B summary path has an invalid suffix")
    expected_run_dir = re.compile(rf"run-[0-9]{{8}}_[0-9]{{6}}-{re.escape(run_id)}\Z")
    if expected_run_dir.fullmatch(path.parent.parent.name) is None:
        raise CollectionError(f"{arm} local W&B summary path does not name its declared run ID")
    if require_existing_nonsymlink_path:
        try:
            resolved_root = pair_root.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            raise CollectionError(f"cannot resolve diagnostic {arm} local W&B summary path") from error
        if resolved_root != pair_root or resolved_path != path:
            raise CollectionError(f"diagnostic {arm} local W&B summary path cannot contain symlinks")
        try:
            resolved_path.relative_to(resolved_root / arm)
        except ValueError as error:
            raise CollectionError(f"resolved diagnostic {arm} summary escapes the declared arm root") from error
    return path.as_posix()


def _diagnostic_latest_summary(value: Any, arm: str) -> dict[str, Any]:
    """Select one exact latest diagnostic point from a local W&B summary."""
    summary = _mapping(value, f"{arm} local W&B summary")
    step = summary.get("_step")
    if type(step) is not int or not 1 <= step <= 100:
        raise CollectionError(f"{arm} local W&B summary has invalid authoritative _step {step!r}")
    metrics: dict[str, int | float] = {}
    for metric in DIAGNOSTIC_LOCAL_SUMMARY_METRICS:
        if metric not in summary or summary[metric] is None:
            raise CollectionError(f"{arm} local W&B summary lacks diagnostic metric {metric}")
        normalized = _normalize_history_number(summary[metric], metric)
        if type(normalized) is bool:
            raise CollectionError(f"{arm} local W&B summary metric {metric} cannot be boolean")
        metrics[metric] = normalized

    _validate_diagnostic_metric_map(metrics, f"diagnostic {arm} latest metrics")
    return {"_step": step, "metrics": metrics}


def _diagnostic_latest_comparison(arms: Mapping[str, Any]) -> dict[str, Any]:
    off = arms["off"]["latest"]
    on = arms["on"]["latest"]
    if off["_step"] != on["_step"]:
        raise CollectionError("diagnostic OFF/ON local summaries do not share one latest _step")
    step = off["_step"]
    off_reward = off["metrics"]["train/raw_environment_reward"]
    on_reward = on["metrics"]["train/raw_environment_reward"]
    off_seconds = off["metrics"]["timing/train/policy_training"]
    on_seconds = on["metrics"]["timing/train/policy_training"]
    same_work = all(off["metrics"][metric] == on["metrics"][metric] for metric in DIAGNOSTIC_WORK_METRICS)
    ratio = off_seconds / on_seconds if same_work else None
    if ratio is not None and (not math.isfinite(ratio) or ratio <= 0.0):
        raise CollectionError("diagnostic OFF/ON policy-training ratio must be finite and positive")
    return {
        "step": step,
        "policy_training_seconds": {
            "off": off_seconds,
            "on": on_seconds,
            "same_work": same_work,
            "off_over_on": ratio,
        },
        "raw_environment_reward": {
            "off": off_reward,
            "on": on_reward,
            "on_minus_off": on_reward - off_reward,
        },
    }


def _diagnostic_pair_summary_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonicalization": WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": canonical_json_sha256(payload, "diagnostic local W&B pair summary payload"),
        "off_source_sha256": payload["arms"]["off"]["source"]["sha256"],
        "on_source_sha256": payload["arms"]["on"]["source"]["sha256"],
    }


def collect_diagnostic_local_summary_pair(
    *,
    pair_label: str,
    pair_root: Path,
    environment: str,
    run_ids: Mapping[str, str],
    summary_paths: Mapping[str, Path],
    expected_summary_sha256: Mapping[str, str],
    collected_at_unix_ns: int,
    collector_sha256: str,
) -> dict[str, Any]:
    """Build a non-acceptance paired diagnostic from two latest-value files."""
    pair_label = _safe_id(pair_label, "diagnostic pair label")
    if not isinstance(pair_root, Path) or not pair_root.is_absolute():
        raise CollectionError("diagnostic pair root must be an absolute Path")
    if pair_root.name != pair_label:
        raise CollectionError("diagnostic pair label must equal the pair-root basename")
    environment = _safe_id(environment, "diagnostic environment")
    if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise CollectionError(f"unsupported diagnostic environment {environment!r}")
    run_ids = _mapping(run_ids, "diagnostic run IDs")
    summary_paths = _mapping(summary_paths, "diagnostic summary paths")
    expected_summary_sha256 = _mapping(expected_summary_sha256, "diagnostic summary SHA-256 pins")
    for mapping, label in (
        (run_ids, "diagnostic run IDs"),
        (summary_paths, "diagnostic summary paths"),
        (expected_summary_sha256, "diagnostic summary SHA-256 pins"),
    ):
        _exact_keys(mapping, {"off", "on"}, label)
    normalized_run_ids = {arm: _safe_id(run_ids[arm], f"diagnostic {arm} W&B run ID") for arm in ("off", "on")}
    if normalized_run_ids["off"] == normalized_run_ids["on"]:
        raise CollectionError("diagnostic OFF/ON W&B run IDs must differ")
    _digest(collector_sha256, "diagnostic collector SHA-256")
    if type(collected_at_unix_ns) is not int or collected_at_unix_ns <= 0:
        raise CollectionError("diagnostic collected_at_unix_ns must be a positive JSON integer")

    arms: dict[str, Any] = {}
    for arm, mode in (("off", "observe"), ("on", "train")):
        source_path = summary_paths[arm]
        if not isinstance(source_path, Path):
            raise CollectionError(f"diagnostic {arm} summary path must be a Path")
        normalized_path = _validate_diagnostic_summary_path(
            source_path,
            normalized_run_ids[arm],
            arm,
            pair_root=pair_root,
            require_existing_nonsymlink_path=True,
        )
        document = load_document(source_path, f"diagnostic {arm} local W&B summary")
        expected_digest = _digest(
            expected_summary_sha256[arm],
            f"trusted diagnostic {arm} local W&B summary SHA-256",
        )
        summary = authenticate_local_summary(
            document,
            expected_digest,
            label=f"diagnostic {arm} local W&B summary",
        )
        arms[arm] = {
            "run_id": normalized_run_ids[arm],
            "shared_prefix_mode": mode,
            "source": {
                "format": "wandb-summary.json",
                "path": normalized_path,
                "sha256": document.sha256,
                "source_key_count": len(summary),
                "source_keys": sorted(summary),
                "source_keyset_sha256": canonical_json_sha256(sorted(summary), f"diagnostic {arm} source keyset"),
            },
            "latest": _diagnostic_latest_summary(summary, arm),
        }
    if arms["off"]["source"]["sha256"] == arms["on"]["source"]["sha256"]:
        raise CollectionError("diagnostic OFF/ON summaries must be distinct bytes")

    latest_comparison = _diagnostic_latest_comparison(arms)
    step = arms["off"]["latest"]["_step"]
    payload = _normalize_json(
        {
            "schema": DIAGNOSTIC_PAIR_SUMMARY_SCHEMA,
            "assurance": DIAGNOSTIC_PAIR_SUMMARY_ASSURANCE,
            "pair": {
                "label": pair_label,
                "root": pair_root.as_posix(),
                "environment": environment,
                "run_ids": normalized_run_ids,
            },
            "capture": {
                "acceptance_eligible": False,
                "summary_progress_field": "_step",
                "collected_at_unix_ns": collected_at_unix_ns,
                "collector_sha256": collector_sha256,
                "history_complete": False,
                "ignored_progress_fields": list(DIAGNOSTIC_IGNORED_PROGRESS_FIELDS),
                "limitations": list(DIAGNOSTIC_LIMITATIONS),
                "method": DIAGNOSTIC_PAIR_SUMMARY_METHOD,
                "selected_metrics": [
                    "_step",
                    *DIAGNOSTIC_LOCAL_SUMMARY_METRICS,
                ],
            },
            "arms": arms,
            "latest_comparison": latest_comparison,
            "acceptance_boundary": {
                "eligible": False,
                "equal_summary_cursors": [step],
                "reason": ("diagnostic_local_summary_is_not_complete_authenticated_history"),
                "required_complete_steps": {"first": 1, "last": 100, "count": 100},
                "reward_claim_scope": "none_diagnostic_only",
            },
        },
        "diagnostic local W&B pair summary payload",
    )
    return {
        **payload,
        "diagnostic_receipt": _diagnostic_pair_summary_receipt(payload),
    }


def _validate_diagnostic_metric_map(value: Any, label: str) -> Mapping[str, Any]:
    metrics = _mapping(value, label)
    _exact_keys(metrics, set(DIAGNOSTIC_LOCAL_SUMMARY_METRICS), label)
    for metric, item in metrics.items():
        if metric in DIAGNOSTIC_LOCAL_SUMMARY_INTEGER_METRICS:
            if type(item) is not int or not 0 <= item <= MAX_EXACT_INTEGER:
                raise CollectionError(f"{label}.{metric} must be an exact non-negative JSON integer")
        elif type(item) is not float or not math.isfinite(item):
            raise CollectionError(f"{label}.{metric} must be a finite JSON float")
    if metrics["timing/train/policy_training"] <= 0.0:
        raise CollectionError(f"{label} policy-training time must be positive")
    if metrics["train/shared_prefix/ideal_token_work_speedup"] <= 0.0:
        raise CollectionError(f"{label} ideal token-work speedup must be positive")
    for metric in (
        "train/total_num_tokens",
        "train/shared_prefix/total_tokens",
        "train/shared_prefix/ideal_shared_token_work",
    ):
        if metrics[metric] <= 0:
            raise CollectionError(f"{label}.{metric} must be positive")
    reward = metrics["train/raw_environment_reward"]
    reward_min = metrics["train/raw_environment_reward/min"]
    reward_max = metrics["train/raw_environment_reward/max"]
    if not reward_min <= reward <= reward_max:
        raise CollectionError(f"{label} raw reward lies outside its min/max")
    _validate_json_numbers(metrics, label)
    return metrics


def validate_diagnostic_pair_summary(value: Mapping[str, Any]) -> None:
    """Validate the exact closed grammar of a diagnostic pair document."""
    value = _mapping(value, "diagnostic pair summary")
    _exact_keys(
        value,
        {
            "schema",
            "assurance",
            "pair",
            "capture",
            "arms",
            "latest_comparison",
            "acceptance_boundary",
            "diagnostic_receipt",
        },
        "diagnostic pair summary",
    )
    _exact_json_value(value["schema"], DIAGNOSTIC_PAIR_SUMMARY_SCHEMA, "diagnostic schema")
    _exact_json_value(
        value["assurance"],
        DIAGNOSTIC_PAIR_SUMMARY_ASSURANCE,
        "diagnostic assurance",
    )
    pair = _mapping(value["pair"], "diagnostic pair")
    _exact_keys(pair, {"label", "root", "environment", "run_ids"}, "diagnostic pair")
    pair_label = _safe_id(pair["label"], "diagnostic pair label")
    if type(pair["root"]) is not str:
        raise CollectionError("diagnostic pair root must be a string")
    pair_root = Path(pair["root"])
    if not pair_root.is_absolute() or pair_root.name != pair_label:
        raise CollectionError("diagnostic pair root is not absolute and label-bound")
    environment = _safe_id(pair["environment"], "diagnostic pair environment")
    if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise CollectionError("diagnostic pair has an unsupported environment")
    run_ids = _mapping(pair["run_ids"], "diagnostic run IDs")
    _exact_keys(run_ids, {"off", "on"}, "diagnostic run IDs")
    for arm in ("off", "on"):
        _safe_id(run_ids[arm], f"diagnostic {arm} run ID")
    if run_ids["off"] == run_ids["on"]:
        raise CollectionError("diagnostic OFF/ON W&B run IDs must differ")

    capture = _mapping(value["capture"], "diagnostic capture")
    _exact_keys(
        capture,
        {
            "acceptance_eligible",
            "summary_progress_field",
            "collected_at_unix_ns",
            "collector_sha256",
            "history_complete",
            "ignored_progress_fields",
            "limitations",
            "method",
            "selected_metrics",
        },
        "diagnostic capture",
    )
    expected_capture = {
        "acceptance_eligible": False,
        "summary_progress_field": "_step",
        "history_complete": False,
        "ignored_progress_fields": list(DIAGNOSTIC_IGNORED_PROGRESS_FIELDS),
        "limitations": list(DIAGNOSTIC_LIMITATIONS),
        "method": DIAGNOSTIC_PAIR_SUMMARY_METHOD,
        "selected_metrics": ["_step", *DIAGNOSTIC_LOCAL_SUMMARY_METRICS],
    }
    for key, expected in expected_capture.items():
        _exact_json_value(capture[key], expected, f"diagnostic capture.{key}")
    _digest(capture["collector_sha256"], "diagnostic collector SHA-256")
    if type(capture["collected_at_unix_ns"]) is not int or capture["collected_at_unix_ns"] <= 0:
        raise CollectionError("diagnostic collected-at timestamp must be a positive JSON integer")

    arms = _mapping(value["arms"], "diagnostic arms")
    _exact_keys(arms, {"off", "on"}, "diagnostic arms")
    for arm, mode in (("off", "observe"), ("on", "train")):
        record = _mapping(arms[arm], f"diagnostic {arm} arm")
        _exact_keys(
            record,
            {"run_id", "shared_prefix_mode", "source", "latest"},
            f"diagnostic {arm} arm",
        )
        _exact_json_value(record["run_id"], run_ids[arm], f"diagnostic {arm} arm run ID")
        _exact_json_value(
            record["shared_prefix_mode"],
            mode,
            f"diagnostic {arm} shared-prefix mode",
        )
        source = _mapping(record["source"], f"diagnostic {arm} source")
        _exact_keys(
            source,
            {
                "format",
                "path",
                "sha256",
                "source_key_count",
                "source_keys",
                "source_keyset_sha256",
            },
            f"diagnostic {arm} source",
        )
        _exact_json_value(source["format"], "wandb-summary.json", f"diagnostic {arm} format")
        if type(source["path"]) is not str:
            raise CollectionError(f"diagnostic {arm} source path must be a string")
        _validate_diagnostic_summary_path(
            Path(source["path"]),
            run_ids[arm],
            arm,
            pair_root=pair_root,
            require_existing_nonsymlink_path=False,
        )
        _digest(source["sha256"], f"diagnostic {arm} source SHA-256")
        source_keys = source["source_keys"]
        if (
            type(source_keys) is not list
            or any(type(key) is not str for key in source_keys)
            or source_keys != sorted(set(source_keys))
        ):
            raise CollectionError(f"diagnostic {arm} source keys must be unique sorted strings")
        if (
            type(source["source_key_count"]) is not int
            or source["source_key_count"] != len(source_keys)
            or source["source_key_count"] < len(DIAGNOSTIC_LOCAL_SUMMARY_METRICS) + 1
        ):
            raise CollectionError(f"diagnostic {arm} source key count is not a sufficient exact integer")
        required_source_keys = {"_step", *DIAGNOSTIC_LOCAL_SUMMARY_METRICS}
        if not required_source_keys <= set(source_keys):
            raise CollectionError(f"diagnostic {arm} source keyset lacks selected metrics")
        expected_keyset_sha256 = canonical_json_sha256(source_keys, f"diagnostic {arm} source keyset")
        _exact_json_value(
            source["source_keyset_sha256"],
            expected_keyset_sha256,
            f"diagnostic {arm} source-keyset SHA-256",
        )
        latest = _mapping(record["latest"], f"diagnostic {arm} latest point")
        _exact_keys(latest, {"_step", "metrics"}, f"diagnostic {arm} latest point")
        if type(latest["_step"]) is not int or not 1 <= latest["_step"] <= 100:
            raise CollectionError(f"diagnostic {arm} latest _step is invalid")
        _validate_diagnostic_metric_map(latest["metrics"], f"diagnostic {arm} latest metrics")
    if arms["off"]["source"]["sha256"] == arms["on"]["source"]["sha256"]:
        raise CollectionError("diagnostic OFF/ON summaries must be distinct bytes")

    expected_comparison = _diagnostic_latest_comparison(arms)
    _exact_json_value(
        value["latest_comparison"],
        expected_comparison,
        "diagnostic latest comparison",
    )
    step = arms["off"]["latest"]["_step"]
    expected_boundary = {
        "eligible": False,
        "equal_summary_cursors": [step],
        "reason": "diagnostic_local_summary_is_not_complete_authenticated_history",
        "required_complete_steps": {"first": 1, "last": 100, "count": 100},
        "reward_claim_scope": "none_diagnostic_only",
    }
    _exact_json_value(
        value["acceptance_boundary"],
        expected_boundary,
        "diagnostic acceptance boundary",
    )
    payload = {key: item for key, item in value.items() if key != "diagnostic_receipt"}
    _exact_json_value(
        value["diagnostic_receipt"],
        _diagnostic_pair_summary_receipt(payload),
        "diagnostic receipt",
    )


def serialize_diagnostic_pair_summary(value: Mapping[str, Any]) -> bytes:
    """Serialize one exact non-acceptance diagnostic as canonical JSON plus LF."""
    validate_diagnostic_pair_summary(value)
    return canonical_json_bytes(value, "diagnostic local W&B pair summary") + b"\n"


def write_diagnostic_pair_summary(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a diagnostic pair document without overwriting an existing path."""
    raw = serialize_diagnostic_pair_summary(value)
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError as error:
        raise CollectionError(f"cannot publish diagnostic pair summary {path}: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def _diagnostic_history_receipt(
    history: list[dict[str, Any]],
    *,
    scan_row_count: int,
    scanned_steps: list[int],
) -> dict[str, Any]:
    return {
        "canonicalization": WANDB_EXPORT_CANONICALIZATION,
        "normalized_selected_history_sha256": canonical_json_sha256(history, "diagnostic W&B selected history"),
        "scan_iteration_completed": True,
        "scan_row_count": scan_row_count,
        "scanned_steps": scanned_steps,
        "selected_step_count": len(history),
        "training_steps": [row["_step"] for row in history],
    }


def collect_diagnostic_wandb_history(run: Any, arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect complete exploratory history while excluding W&B step zero."""
    try:
        rows: Iterable[Any] = run.scan_history(
            page_size=1000,
            min_step=0,
            max_step=101,
            use_cache=False,
        )
        merged: dict[int, dict[str, int | float]] = {}
        scanned_steps: set[int] = set()
        scan_row_count = 0
        for row_index, raw_row in enumerate(rows):
            if row_index >= MAX_HISTORY_ROWS:
                raise CollectionError("diagnostic W&B history exceeds the strict row-count limit")
            scan_row_count += 1
            row = _mapping(raw_row, f"diagnostic {arm} W&B row {row_index}")
            step = row.get("_step")
            if type(step) is not int or not 0 <= step <= 100:
                raise CollectionError(f"diagnostic {arm} W&B row {row_index} has invalid _step {step!r}")
            scanned_steps.add(step)
            present = [metric for metric in DIAGNOSTIC_LOCAL_SUMMARY_METRICS if row.get(metric) is not None]
            if not present:
                continue
            if step == 0:
                continue
            selected = merged.setdefault(step, {})
            for metric in present:
                normalized = _normalize_history_number(row[metric], metric)
                if type(normalized) is bool:
                    raise CollectionError(f"diagnostic {arm} W&B metric {metric} cannot be boolean")
                if metric in selected and selected[metric] != normalized:
                    raise CollectionError(f"diagnostic {arm} W&B step {step} has conflicting {metric}")
                selected[metric] = normalized
    except CollectionError:
        raise
    except Exception as error:
        raise CollectionError(f"diagnostic {arm} W&B history scan failed") from error

    if not merged:
        raise CollectionError(f"diagnostic {arm} W&B history has no training steps")
    history: list[dict[str, Any]] = []
    required = set(DIAGNOSTIC_LOCAL_SUMMARY_METRICS)
    for step in sorted(merged):
        metrics = merged[step]
        if set(metrics) != required:
            raise CollectionError(
                f"diagnostic {arm} W&B step {step} metric fields differ: "
                f"missing={sorted(required - set(metrics))}, "
                f"extra={sorted(set(metrics) - required)}"
            )
        _validate_diagnostic_metric_map(metrics, f"diagnostic {arm} W&B step {step} metrics")
        history.append({"_step": step, "metrics": metrics})
    receipt = _diagnostic_history_receipt(
        history,
        scan_row_count=scan_row_count,
        scanned_steps=sorted(scanned_steps),
    )
    return history, receipt


def _validate_diagnostic_wandb_run_identity(run: Any, expected: Mapping[str, Any], arm: str) -> dict[str, Any]:
    identity = {
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "group": getattr(run, "group", None),
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
    }
    _exact_json_value(identity, expected, f"diagnostic {arm} W&B identity")
    expected_path = [expected["entity"], expected["project"], expected["run_id"]]
    _exact_json_value(
        getattr(run, "path", None),
        expected_path,
        f"diagnostic {arm} W&B path",
    )
    return identity


def _diagnostic_paired_history(
    arms: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    by_arm = {arm: {row["_step"]: row["metrics"] for row in arms[arm]["history"]} for arm in ("off", "on")}
    off_step_set = set(by_arm["off"])
    on_step_set = set(by_arm["on"])
    paired_steps = sorted(off_step_set & on_step_set)
    if not paired_steps:
        raise CollectionError("diagnostic W&B histories have no paired training steps")
    coverage = {
        "off_training_steps": sorted(off_step_set),
        "on_training_steps": sorted(on_step_set),
        "paired_steps": paired_steps,
        "off_only_steps": sorted(off_step_set - on_step_set),
        "on_only_steps": sorted(on_step_set - off_step_set),
    }
    paired: list[dict[str, Any]] = []
    for step in paired_steps:
        off = by_arm["off"][step]
        on = by_arm["on"][step]
        same_work = all(off[metric] == on[metric] for metric in DIAGNOSTIC_WORK_METRICS)
        off_seconds = off["timing/train/policy_training"]
        on_seconds = on["timing/train/policy_training"]
        ratio = off_seconds / on_seconds if same_work else None
        if ratio is not None and (not math.isfinite(ratio) or ratio <= 0.0):
            raise CollectionError(f"diagnostic W&B step {step} speed ratio is not finite and positive")
        off_reward = off["train/raw_environment_reward"]
        on_reward = on["train/raw_environment_reward"]
        paired.append(
            {
                "step": step,
                "raw_environment_reward": {
                    "off": off_reward,
                    "on": on_reward,
                    "on_minus_off": on_reward - off_reward,
                },
                "policy_training_seconds": {
                    "off": off_seconds,
                    "on": on_seconds,
                    "same_work": same_work,
                    "off_over_on": ratio,
                },
                "token_counts": {
                    metric: {"off": off[metric], "on": on[metric]} for metric in DIAGNOSTIC_TOKEN_COUNT_METRICS
                },
            }
        )
    return paired, coverage


def _diagnostic_history_pair_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonicalization": WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": canonical_json_sha256(payload, "diagnostic W&B history pair payload"),
        "off_normalized_selected_history_sha256": payload["arms"]["off"]["history_receipt"][
            "normalized_selected_history_sha256"
        ],
        "on_normalized_selected_history_sha256": payload["arms"]["on"]["history_receipt"][
            "normalized_selected_history_sha256"
        ],
    }


def collect_diagnostic_wandb_history_pair(
    *,
    pair_label: str,
    environment: str,
    entity: str,
    project: str,
    group: str,
    run_ids: Mapping[str, str],
    run_names: Mapping[str, str],
    api: Any,
    fetched_at_unix_ns: int,
    collector_sha256: str,
) -> dict[str, Any]:
    """Collect a non-acceptance paired history for exploratory W&B runs."""
    pair_label = _safe_id(pair_label, "diagnostic pair label")
    environment = _safe_id(environment, "diagnostic environment")
    if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise CollectionError(f"unsupported diagnostic environment {environment!r}")
    entity = _safe_id(entity, "diagnostic W&B entity")
    project = _safe_id(project, "diagnostic W&B project")
    group = _safe_id(group, "diagnostic W&B group")
    if group != pair_label:
        raise CollectionError("diagnostic W&B group must equal the pair label")
    run_ids = _mapping(run_ids, "diagnostic run IDs")
    run_names = _mapping(run_names, "diagnostic run names")
    _exact_keys(run_ids, {"off", "on"}, "diagnostic run IDs")
    _exact_keys(run_names, {"off", "on"}, "diagnostic run names")
    normalized_run_ids = {arm: _safe_id(run_ids[arm], f"diagnostic {arm} W&B run ID") for arm in ("off", "on")}
    normalized_run_names = {arm: _safe_id(run_names[arm], f"diagnostic {arm} W&B run name") for arm in ("off", "on")}
    if normalized_run_ids["off"] == normalized_run_ids["on"]:
        raise CollectionError("diagnostic OFF/ON W&B run IDs must differ")
    if normalized_run_names["off"] == normalized_run_names["on"]:
        raise CollectionError("diagnostic OFF/ON W&B run names must differ")
    _digest(collector_sha256, "diagnostic collector SHA-256")
    if type(fetched_at_unix_ns) is not int or fetched_at_unix_ns <= 0:
        raise CollectionError("diagnostic fetched_at_unix_ns must be a positive JSON integer")
    validate_wandb_api_authentication(api)

    arms: dict[str, Any] = {}
    try:
        for arm in ("off", "on"):
            expected_identity = {
                "entity": entity,
                "project": project,
                "group": group,
                "run_id": normalized_run_ids[arm],
                "run_name": normalized_run_names[arm],
                "state": "finished",
            }
            run = api.run(f"{entity}/{project}/{normalized_run_ids[arm]}")
            identity = _validate_diagnostic_wandb_run_identity(run, expected_identity, arm)
            history, history_receipt = collect_diagnostic_wandb_history(run, arm)
            arms[arm] = {
                "identity": identity,
                "history": history,
                "history_receipt": history_receipt,
            }
    except CollectionError:
        raise
    except Exception as error:
        raise CollectionError("diagnostic W&B pair fetch failed") from error

    paired_history, coverage = _diagnostic_paired_history(arms)
    paired_steps = [row["step"] for row in paired_history]
    training_step_coverage_complete = paired_steps == list(range(1, 101))
    payload = _normalize_json(
        {
            "schema": DIAGNOSTIC_PAIR_HISTORY_SCHEMA,
            "assurance": DIAGNOSTIC_PAIR_HISTORY_ASSURANCE,
            "pair": {
                "label": pair_label,
                "environment": environment,
                "entity": entity,
                "project": project,
                "group": group,
                "run_ids": normalized_run_ids,
                "run_names": normalized_run_names,
            },
            "capture": {
                "acceptance_eligible": False,
                "api_base_url": WANDB_API_BASE_URL,
                "authoritative_progress_field": "_step",
                "collector_sha256": collector_sha256,
                "fetched_at_unix_ns": fetched_at_unix_ns,
                "history_method": WANDB_HISTORY_METHOD,
                "history_completeness_status": "not_proven",
                "identity_binding": "caller_supplied_wandb_path",
                "ignored_progress_fields": list(DIAGNOSTIC_IGNORED_PROGRESS_FIELDS),
                "limitations": list(DIAGNOSTIC_HISTORY_LIMITATIONS),
                "same_work_metrics": list(DIAGNOSTIC_WORK_METRICS),
                "scan_iteration_completed": True,
                "scan_query": {
                    "keys": None,
                    "min_step_inclusive": 0,
                    "max_step_exclusive": 101,
                    "page_size": 1000,
                    "use_cache": False,
                },
                "selected_metrics": [
                    "_step",
                    *DIAGNOSTIC_LOCAL_SUMMARY_METRICS,
                ],
                "step_zero_policy": "scanned_excluded_non_training",
                "summary_fallback_used": False,
                "wandb_api_viewer_resolved": True,
                "wandb_sdk_version": WANDB_SDK_VERSION,
            },
            "arms": arms,
            "coverage": coverage,
            "paired_history": paired_history,
            "acceptance_boundary": {
                "eligible": False,
                "observed_paired_steps": paired_steps,
                "reason": ("exploratory_identity_lacks_authenticated_pair_scheduler_provenance"),
                "required_complete_steps": {"first": 1, "last": 100, "count": 100},
                "reward_claim_scope": "none_diagnostic_live_rollouts_not_parity",
                "training_step_coverage_complete": training_step_coverage_complete,
            },
        },
        "diagnostic W&B history pair payload",
    )
    return {
        **payload,
        "diagnostic_receipt": _diagnostic_history_pair_receipt(payload),
    }


def _validate_diagnostic_history_points(value: Any, arm: str) -> list[dict[str, Any]]:
    if type(value) is not list or not value or len(value) > 100:
        raise CollectionError(f"diagnostic {arm} history must contain 1..100 selected steps")
    previous_step = 0
    for index, raw in enumerate(value):
        row = _mapping(raw, f"diagnostic {arm} history row {index}")
        _exact_keys(row, {"_step", "metrics"}, f"diagnostic {arm} history row")
        step = row["_step"]
        if type(step) is not int or not previous_step < step <= 100:
            raise CollectionError(f"diagnostic {arm} history steps must be unique increasing integers")
        previous_step = step
        _validate_diagnostic_metric_map(row["metrics"], f"diagnostic {arm} W&B step {step} metrics")
    return value


def validate_diagnostic_history_pair(value: Mapping[str, Any]) -> None:
    """Validate the exact closed grammar of an exploratory history pair."""
    value = _mapping(value, "diagnostic W&B history pair")
    _exact_keys(
        value,
        {
            "schema",
            "assurance",
            "pair",
            "capture",
            "arms",
            "coverage",
            "paired_history",
            "acceptance_boundary",
            "diagnostic_receipt",
        },
        "diagnostic W&B history pair",
    )
    _exact_json_value(value["schema"], DIAGNOSTIC_PAIR_HISTORY_SCHEMA, "diagnostic history schema")
    _exact_json_value(
        value["assurance"],
        DIAGNOSTIC_PAIR_HISTORY_ASSURANCE,
        "diagnostic history assurance",
    )
    pair = _mapping(value["pair"], "diagnostic history pair identity")
    _exact_keys(
        pair,
        {
            "label",
            "environment",
            "entity",
            "project",
            "group",
            "run_ids",
            "run_names",
        },
        "diagnostic history pair identity",
    )
    pair_label = _safe_id(pair["label"], "diagnostic pair label")
    environment = _safe_id(pair["environment"], "diagnostic environment")
    if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise CollectionError("diagnostic pair has an unsupported environment")
    entity = _safe_id(pair["entity"], "diagnostic entity")
    project = _safe_id(pair["project"], "diagnostic project")
    group = _safe_id(pair["group"], "diagnostic group")
    if group != pair_label:
        raise CollectionError("diagnostic group must equal the pair label")
    run_ids = _mapping(pair["run_ids"], "diagnostic run IDs")
    run_names = _mapping(pair["run_names"], "diagnostic run names")
    _exact_keys(run_ids, {"off", "on"}, "diagnostic run IDs")
    _exact_keys(run_names, {"off", "on"}, "diagnostic run names")
    for arm in ("off", "on"):
        _safe_id(run_ids[arm], f"diagnostic {arm} run ID")
        _safe_id(run_names[arm], f"diagnostic {arm} run name")
    if run_ids["off"] == run_ids["on"] or run_names["off"] == run_names["on"]:
        raise CollectionError("diagnostic OFF/ON identities must differ")

    capture = _mapping(value["capture"], "diagnostic history capture")
    _exact_keys(
        capture,
        {
            "acceptance_eligible",
            "api_base_url",
            "authoritative_progress_field",
            "collector_sha256",
            "fetched_at_unix_ns",
            "history_method",
            "history_completeness_status",
            "identity_binding",
            "ignored_progress_fields",
            "limitations",
            "same_work_metrics",
            "scan_iteration_completed",
            "scan_query",
            "selected_metrics",
            "step_zero_policy",
            "summary_fallback_used",
            "wandb_api_viewer_resolved",
            "wandb_sdk_version",
        },
        "diagnostic history capture",
    )
    expected_capture = {
        "acceptance_eligible": False,
        "api_base_url": WANDB_API_BASE_URL,
        "authoritative_progress_field": "_step",
        "history_method": WANDB_HISTORY_METHOD,
        "history_completeness_status": "not_proven",
        "identity_binding": "caller_supplied_wandb_path",
        "ignored_progress_fields": list(DIAGNOSTIC_IGNORED_PROGRESS_FIELDS),
        "limitations": list(DIAGNOSTIC_HISTORY_LIMITATIONS),
        "same_work_metrics": list(DIAGNOSTIC_WORK_METRICS),
        "scan_iteration_completed": True,
        "scan_query": {
            "keys": None,
            "min_step_inclusive": 0,
            "max_step_exclusive": 101,
            "page_size": 1000,
            "use_cache": False,
        },
        "selected_metrics": ["_step", *DIAGNOSTIC_LOCAL_SUMMARY_METRICS],
        "step_zero_policy": "scanned_excluded_non_training",
        "summary_fallback_used": False,
        "wandb_api_viewer_resolved": True,
        "wandb_sdk_version": WANDB_SDK_VERSION,
    }
    for key, expected in expected_capture.items():
        _exact_json_value(capture[key], expected, f"diagnostic history capture.{key}")
    _digest(capture["collector_sha256"], "diagnostic history collector SHA-256")
    if type(capture["fetched_at_unix_ns"]) is not int or capture["fetched_at_unix_ns"] <= 0:
        raise CollectionError("diagnostic history fetched-at timestamp must be a positive integer")

    arms = _mapping(value["arms"], "diagnostic history arms")
    _exact_keys(arms, {"off", "on"}, "diagnostic history arms")
    for arm in ("off", "on"):
        record = _mapping(arms[arm], f"diagnostic history {arm} arm")
        _exact_keys(
            record,
            {"identity", "history", "history_receipt"},
            f"diagnostic history {arm} arm",
        )
        expected_identity = {
            "entity": entity,
            "project": project,
            "group": group,
            "run_id": run_ids[arm],
            "run_name": run_names[arm],
            "state": "finished",
        }
        _exact_json_value(
            record["identity"],
            expected_identity,
            f"diagnostic history {arm} identity",
        )
        history = _validate_diagnostic_history_points(record["history"], arm)
        receipt = _mapping(record["history_receipt"], f"diagnostic {arm} history receipt")
        _exact_keys(
            receipt,
            {
                "canonicalization",
                "normalized_selected_history_sha256",
                "scan_iteration_completed",
                "scan_row_count",
                "scanned_steps",
                "selected_step_count",
                "training_steps",
            },
            f"diagnostic {arm} history receipt",
        )
        if (
            type(receipt["scan_row_count"]) is not int
            or receipt["scan_row_count"] < len(history)
            or receipt["scan_row_count"] > MAX_HISTORY_ROWS
        ):
            raise CollectionError(
                f"diagnostic {arm} scan row count must cover selected history " "within the strict row-count limit"
            )
        scanned_steps = receipt["scanned_steps"]
        if (
            type(scanned_steps) is not list
            or any(type(step) is not int or not 0 <= step <= 100 for step in scanned_steps)
            or scanned_steps != sorted(set(scanned_steps))
        ):
            raise CollectionError(f"diagnostic {arm} scanned steps must be unique sorted integers")
        training_steps = [row["_step"] for row in history]
        if not set(training_steps) <= set(scanned_steps):
            raise CollectionError(f"diagnostic {arm} scanned steps omit selected training steps")
        expected_receipt = _diagnostic_history_receipt(
            history,
            scan_row_count=receipt["scan_row_count"],
            scanned_steps=scanned_steps,
        )
        _exact_json_value(receipt, expected_receipt, f"diagnostic {arm} history receipt")

    paired_history, coverage = _diagnostic_paired_history(arms)
    _exact_json_value(value["coverage"], coverage, "diagnostic history coverage")
    _exact_json_value(value["paired_history"], paired_history, "diagnostic paired history")
    paired_steps = [row["step"] for row in paired_history]
    expected_boundary = {
        "eligible": False,
        "observed_paired_steps": paired_steps,
        "reason": "exploratory_identity_lacks_authenticated_pair_scheduler_provenance",
        "required_complete_steps": {"first": 1, "last": 100, "count": 100},
        "reward_claim_scope": "none_diagnostic_live_rollouts_not_parity",
        "training_step_coverage_complete": paired_steps == list(range(1, 101)),
    }
    _exact_json_value(
        value["acceptance_boundary"],
        expected_boundary,
        "diagnostic history acceptance boundary",
    )
    _validate_json_numbers(value, "diagnostic W&B history pair")
    payload = {key: item for key, item in value.items() if key != "diagnostic_receipt"}
    _exact_json_value(
        value["diagnostic_receipt"],
        _diagnostic_history_pair_receipt(payload),
        "diagnostic history receipt",
    )


def serialize_diagnostic_history_pair(value: Mapping[str, Any]) -> bytes:
    """Serialize a non-acceptance history pair as canonical ASCII JSON plus LF."""
    validate_diagnostic_history_pair(value)
    return canonical_json_bytes(value, "diagnostic W&B history pair") + b"\n"


def write_diagnostic_history_pair(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a diagnostic history pair without overwriting an existing path."""
    raw = serialize_diagnostic_history_pair(value)
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError as error:
        raise CollectionError(f"cannot publish diagnostic W&B history pair {path}: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def _exact_json_value(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected):
        raise CollectionError(f"{label} has the wrong JSON type")
    if isinstance(expected, dict):
        _exact_keys(actual, set(expected), label)
        for key, item in expected.items():
            _exact_json_value(actual[key], item, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise CollectionError(f"{label} differs from its exact ordered list")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _exact_json_value(actual_item, expected_item, f"{label}[{index}]")
        return
    if actual != expected:
        raise CollectionError(f"{label} differs from the exact expected value")
    if isinstance(actual, float) and actual == 0.0 and math.copysign(1.0, actual) != math.copysign(1.0, expected):
        raise CollectionError(f"{label} differs in signed-zero representation")


def validate_pair_wandb_identity(pair_manifest: Mapping[str, Any], arm: str) -> dict[str, str]:
    """Validate and expand the deterministic Pair W&B identity for one arm."""
    if pair_manifest.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise CollectionError("unexpected Pair manifest schema")
    if arm not in {"off", "on"}:
        raise CollectionError(f"unsupported arm {arm!r}")
    pair_id = _safe_id(pair_manifest.get("pair_id"), "Pair ID")
    selection = _mapping(pair_manifest.get("selection"), "Pair selection")
    environment = _safe_id(selection.get("environment"), "Pair environment")
    if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise CollectionError(f"unsupported strict environment {environment!r}")
    modes = _mapping(pair_manifest.get("arms"), "Pair arms")
    _exact_json_value(modes, {"off": "observe", "on": "train"}, "Pair arms")

    wandb = _mapping(pair_manifest.get("wandb"), "Pair W&B identity")
    _exact_keys(
        wandb,
        {"entity", "project", "group", "arms", "resume", "run_id_derivation"},
        "Pair W&B identity",
    )
    entity = _safe_id(wandb["entity"], "Pair W&B entity")
    project = _safe_id(wandb["project"], "Pair W&B project")
    group_value = f"{environment}-{pair_id}"
    expected = {
        "entity": entity,
        "project": project,
        "group": {"template": "{environment}-{pair_id}", "value": group_value},
        "resume": "never",
        "arms": {
            candidate_arm: {
                "name_template": f"{candidate_arm}-{{environment}}-{{pair_id}}",
                "name": f"{candidate_arm}-{environment}-{pair_id}",
                "run_id": derive_wandb_run_id(environment, pair_id, candidate_arm),
            }
            for candidate_arm in ("off", "on")
        },
        "run_id_derivation": WANDB_RUN_ID_DERIVATION,
    }
    _exact_json_value(wandb, expected, "Pair W&B identity")
    selected = wandb["arms"][arm]
    return {
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "shared_prefix_mode": modes[arm],
        "entity": entity,
        "project": project,
        "group": group_value,
        "run_id": selected["run_id"],
        "run_name": selected["name"],
        "state": "finished",
    }


def validate_submission_binding(
    submission_receipt: Mapping[str, Any],
    *,
    pair_manifest: Mapping[str, Any],
    pair_manifest_sha256: str,
    arm: str,
) -> str:
    """Return the scheduler-authenticated job ID bound to the selected arm."""
    if submission_receipt.get("schema") != SUBMISSION_RECEIPT_SCHEMA:
        raise CollectionError("unexpected submission receipt schema")
    if submission_receipt.get("outcome") != "released":
        raise CollectionError("submission receipt does not record a released pair")
    if submission_receipt.get("stage") != "complete":
        raise CollectionError("submission receipt is not complete")
    pair = _mapping(submission_receipt.get("pair"), "submission receipt pair")
    manifest = _mapping(pair.get("manifest"), "submission receipt Pair manifest")
    if pair.get("id") != pair_manifest.get("pair_id"):
        raise CollectionError("submission receipt Pair ID differs")
    if manifest.get("sha256") != pair_manifest_sha256:
        raise CollectionError("submission receipt does not bind the Pair manifest")

    pair_wandb = _mapping(pair_manifest.get("wandb"), "Pair W&B identity")
    receipt_wandb = _mapping(submission_receipt.get("wandb"), "submission receipt W&B identity")
    _exact_json_value(receipt_wandb, pair_wandb, "submission receipt W&B identity")

    held = _mapping(submission_receipt.get("held_submissions"), "held submissions")
    _exact_keys(held, {"off", "on"}, "held submissions")
    candidate_ids = {
        candidate_arm: _job_id(
            _mapping(held[candidate_arm], f"{candidate_arm} held submission").get("candidate_job_id"),
            f"{candidate_arm} candidate job ID",
        )
        for candidate_arm in ("off", "on")
    }
    if candidate_ids["off"] == candidate_ids["on"]:
        raise CollectionError("OFF/ON submissions bind the same scheduler job ID")

    authenticated = _mapping(submission_receipt.get("authenticated_jobs"), "authenticated jobs")
    _exact_keys(authenticated, {"off", "on"}, "authenticated jobs")
    for candidate_arm in ("off", "on"):
        records = authenticated[candidate_arm]
        if not isinstance(records, list) or len(records) != 1:
            raise CollectionError(f"{candidate_arm} must have exactly one authenticated scheduler record")
        record = _mapping(records[0], f"{candidate_arm} authenticated scheduler record")
        if record.get("job_id") != candidate_ids[candidate_arm]:
            raise CollectionError(f"{candidate_arm} authenticated scheduler job ID differs")
        expected_name = pair_wandb["arms"][candidate_arm]["name"]
        if record.get("job_name") != expected_name:
            raise CollectionError(f"{candidate_arm} authenticated scheduler job name differs")
    return candidate_ids[arm]


def _nested_value(value: Any, path: tuple[str, ...], label: str) -> Any:
    current = value
    for key in path:
        mapping = _mapping(current, label)
        if key not in mapping:
            raise CollectionError(f"{label} lacks {'.'.join(path)}")
        current = mapping[key]
    return current


def validate_wandb_run_config(run_config: Any, expected_config: Mapping[str, Any]) -> None:
    """Validate the acceptance-relevant subset of NeMo-RL's logged config."""
    _mapping(run_config, "W&B run config")
    paths = {
        "max_num_steps": ("grpo", "max_num_steps"),
        "epochs": ("grpo", "max_num_epochs"),
        "num_prompts_per_step": ("grpo", "num_prompts_per_step"),
        "num_generations_per_prompt": ("grpo", "num_generations_per_prompt"),
        "seed": ("grpo", "seed"),
        "generation_seed_base": (
            "policy",
            "generation",
            "nemo_gym_per_rollout_seed_base",
        ),
        "data_shuffle": ("data", "shuffle"),
        "reward_scaling_enabled": ("grpo", "reward_scaling", "enabled"),
        "reward_shaping_enabled": ("grpo", "reward_shaping", "enabled"),
        "shared_prefix_mode": ("policy", "shared_prefix_training", "mode"),
        "wandb_enabled": ("logger", "wandb_enabled"),
        "tensorboard_enabled": ("logger", "tensorboard_enabled"),
        "wandb_entity": ("logger", "wandb", "entity"),
        "wandb_project": ("logger", "wandb", "project"),
        "wandb_run_name": ("logger", "wandb", "name"),
        "tensor_parallel_size": (
            "policy",
            "megatron_cfg",
            "tensor_model_parallel_size",
        ),
        "context_parallel_size": (
            "policy",
            "megatron_cfg",
            "context_parallel_size",
        ),
        "sequence_parallel": (
            "policy",
            "megatron_cfg",
            "sequence_parallel",
        ),
        "pipeline_parallel_size": (
            "policy",
            "megatron_cfg",
            "pipeline_model_parallel_size",
        ),
        "expert_parallel_size": (
            "policy",
            "megatron_cfg",
            "expert_model_parallel_size",
        ),
        "expert_tensor_parallel_size": (
            "policy",
            "megatron_cfg",
            "expert_tensor_parallel_size",
        ),
        "mtp_num_layers": ("policy", "megatron_cfg", "mtp_num_layers"),
        "mtp_use_repeated_layer": (
            "policy",
            "megatron_cfg",
            "mtp_use_repeated_layer",
        ),
        "mtp_detach_heads": ("policy", "megatron_cfg", "mtp_detach_heads"),
        "mtp_loss_scaling_factor": (
            "policy",
            "megatron_cfg",
            "mtp_loss_scaling_factor",
        ),
        "max_new_tokens": ("policy", "generation", "max_new_tokens"),
        "temperature": ("policy", "generation", "temperature"),
        "top_k": ("policy", "generation", "top_k"),
        "top_p": ("policy", "generation", "top_p"),
    }
    for config_key, path in paths.items():
        if config_key not in expected_config:
            raise CollectionError(f"acceptance config lacks {config_key}")
        actual = _nested_value(run_config, path, "W&B run config")
        _exact_json_value(actual, expected_config[config_key], f"W&B {config_key}")
    for field in (
        "max_num_steps",
        "epochs",
        "steps_per_epoch",
        "fixture_rows",
        "num_prompts_per_step",
    ):
        value = expected_config.get(field)
        if type(value) is not int or value <= 0:
            raise CollectionError(f"acceptance config {field} must be a positive JSON integer")
    if expected_config["max_num_steps"] != (expected_config["epochs"] * expected_config["steps_per_epoch"]):
        raise CollectionError(
            "acceptance steps_per_epoch does not close against the W&B-observed " "max_num_steps and epochs"
        )
    if expected_config["fixture_rows"] != (
        expected_config["steps_per_epoch"] * expected_config["num_prompts_per_step"]
    ):
        raise CollectionError(
            "acceptance fixture_rows does not close against steps_per_epoch and " "the W&B-observed prompts per step"
        )
    require_deterministic = _nested_value(
        run_config,
        ("policy", "shared_prefix_training", "require_deterministic_execution"),
        "W&B run config",
    )
    _exact_json_value(
        require_deterministic,
        True,
        "W&B shared-prefix deterministic-execution requirement",
    )
    rollout_seed_opt_in = _nested_value(
        run_config,
        ("policy", "generation", "nemo_gym_add_seed_per_rollout"),
        "W&B run config",
    )
    _exact_json_value(rollout_seed_opt_in, True, "W&B rollout seed opt-in")


def validate_acceptance_contract(
    contract: Mapping[str, Any],
    *,
    identity: Mapping[str, str],
    pair_manifest_sha256: str,
    collector_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the exact arm config and provenance from the trust anchor."""
    if contract.get("schema") != ACCEPTANCE_CONTRACT_SCHEMA:
        raise CollectionError("unexpected acceptance contract schema")
    arm = identity["arm"]
    pair = _mapping(contract.get("pair"), "acceptance contract pair")
    expected_pair = {
        "pair_id": identity["pair_id"],
        "environment": identity["environment"],
        "entity": identity["entity"],
        "project": identity["project"],
        "group": identity["group"],
        "run_ids": {
            "off": derive_wandb_run_id(identity["environment"], identity["pair_id"], "off"),
            "on": derive_wandb_run_id(identity["environment"], identity["pair_id"], "on"),
        },
    }
    _exact_json_value(pair, expected_pair, "acceptance contract pair")

    configs = _mapping(contract.get("configs"), "acceptance configs")
    _exact_keys(configs, {"off", "on"}, "acceptance configs")
    config = copy.deepcopy(dict(_mapping(configs[arm], f"{arm} acceptance config")))
    _exact_keys(config, set(ACCEPTANCE_CONFIG_KEYS), f"{arm} acceptance config")

    provenance = _mapping(contract.get("provenance"), "acceptance provenance")
    required = {
        "common",
        "source_commits",
        "source_git_trees",
        "trusted_oob_declarations",
        "topology",
        "arms",
    }
    _exact_keys(provenance, required, "acceptance provenance")
    common = _validate_hash_mapping(
        provenance["common"],
        COMMON_PROVENANCE_KEYS,
        "acceptance common provenance",
    )
    if common.get("acceptance_contract_sha256") != (acceptance_contract_payload_sha256(contract)):
        raise CollectionError("acceptance_contract_sha256 differs from the canonical contract payload")
    if common.get("pair_manifest_sha256") != pair_manifest_sha256:
        raise CollectionError("acceptance provenance does not bind the Pair manifest")
    if common.get("wandb_exporter_sha256") != collector_sha256:
        raise CollectionError("acceptance provenance does not bind this W&B collector")
    arms = _mapping(provenance["arms"], "acceptance arm provenance")
    _exact_keys(arms, {"off", "on"}, "acceptance arm provenance")
    for selected_arm in ("off", "on"):
        _validate_hash_mapping(
            arms[selected_arm],
            ARM_PROVENANCE_KEYS,
            f"acceptance {selected_arm} arm provenance",
        )
    source_commits = _mapping(provenance["source_commits"], "source commits")
    source_git_trees = _mapping(provenance["source_git_trees"], "source Git trees")
    _exact_keys(source_commits, set(SOURCE_KEYS), "source commits")
    _exact_keys(source_git_trees, set(SOURCE_KEYS), "source Git trees")
    for name in SOURCE_KEYS:
        _commit(source_commits[name], f"source commit {name}")
        _commit(source_git_trees[name], f"source Git tree {name}")
    _validate_trusted_oob_declarations(provenance["trusted_oob_declarations"])
    selected_provenance = {
        "common": copy.deepcopy(dict(common)),
        "source_commits": copy.deepcopy(provenance["source_commits"]),
        "source_git_trees": copy.deepcopy(provenance["source_git_trees"]),
        "trusted_oob_declarations": copy.deepcopy(provenance["trusted_oob_declarations"]),
        "topology": copy.deepcopy(provenance["topology"]),
        "arm": copy.deepcopy(arms[arm]),
    }
    return config, selected_provenance


def validate_run_identity(run: Any, expected: Mapping[str, str]) -> None:
    """Require the fetched W&B object to be exactly the Pair-selected run."""
    observed = {
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "group": getattr(run, "group", None),
        "state": getattr(run, "state", None),
    }
    for key, actual in observed.items():
        if actual != expected[key]:
            raise CollectionError(f"fetched W&B {key} differs from the Pair identity")
    path = getattr(run, "path", None)
    expected_path = [expected["entity"], expected["project"], expected["run_id"]]
    if type(path) is not list or path != expected_path:
        raise CollectionError("fetched W&B run path differs from the Pair identity")


def validate_wandb_api_authentication(api: Any) -> None:
    """Resolve W&B's authenticated viewer before claiming API authentication."""
    try:
        viewer = api.viewer
    except Exception as error:
        raise CollectionError("W&B API did not authenticate a viewer") from error
    if viewer is None or isinstance(viewer, (bool, str, bytes, int, float)):
        raise CollectionError("W&B API did not authenticate a viewer")
    # ``Api.viewer`` is the authenticated User.  Its ``entity`` is the user's
    # default entity, not an authorization assertion for the team entity in a
    # three-part run path.  Access and attribution are closed below by fetching
    # that exact path and validating the returned Run identity byte-for-byte.


def _normalize_history_number(value: Any, metric: str) -> bool | int | float:
    if metric == "train/update_successful":
        if type(value) is not bool:
            raise CollectionError("W&B history metric train/update_successful must be an exact boolean")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionError(f"W&B history metric {metric} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CollectionError(f"W&B history metric {metric} must be finite")
    if metric in INTEGER_METRICS:
        if not number.is_integer() or abs(number) > MAX_EXACT_INTEGER:
            raise CollectionError(f"W&B history metric {metric} must be an exact integer")
        return int(number)
    number = 0.0 if number == 0.0 else number
    if metric == "timing/train/policy_training" and number <= 0.0:
        raise CollectionError("W&B timing/train/policy_training must be strictly positive")
    return number


def collect_history(run: Any, metrics: list[str]) -> list[dict[str, bool | int | float]]:
    """Scan complete unsampled history and retain only frozen acceptance metrics."""
    expected_metrics = set(metrics)
    if "_step" not in expected_metrics:
        raise CollectionError("requested metrics must include _step")
    # W&B treats multiple ``keys`` as an all-keys intersection.  NeMo-RL logs
    # this inventory sparsely, so scan the bounded history without ``keys`` and
    # apply the frozen allowlist locally.
    rows: Iterable[Any] = run.scan_history(
        page_size=1000,
        min_step=1,
        max_step=101,
        use_cache=False,
    )
    normalized: list[dict[str, bool | int | float]] = []
    for row_index, raw_row in enumerate(rows):
        if row_index >= MAX_HISTORY_ROWS:
            raise CollectionError("W&B history exceeds the strict row-count limit")
        row = _mapping(raw_row, f"W&B history row {row_index}")
        present = [metric for metric in metrics if metric != "_step" and row.get(metric) is not None]
        if not present:
            continue
        step = row.get("_step")
        if type(step) is not int or not 1 <= step <= 100:
            raise CollectionError(f"W&B history row {row_index} has invalid strict _step {step!r}")
        selected: dict[str, bool | int | float] = {"_step": step}
        for metric in present:
            selected[metric] = _normalize_history_number(row[metric], metric)
        normalized.append(selected)
    if not normalized:
        raise CollectionError("W&B scan_history returned no strict acceptance metrics")
    normalized.sort(
        key=lambda row: (
            row["_step"],
            canonical_json_bytes(row, "normalized W&B history row"),
        )
    )
    return normalized


def collect_run_export(
    *,
    pair_manifest: Mapping[str, Any],
    pair_manifest_sha256: str,
    submission_receipt: Mapping[str, Any],
    submission_receipt_sha256: str,
    acceptance_contract: Mapping[str, Any],
    arm: str,
    api: Any,
    fetched_at_unix_ns: int,
    collector_sha256: str,
) -> dict[str, Any]:
    """Fetch, validate, normalize, and receipt one strict W&B run."""
    _digest(pair_manifest_sha256, "Pair manifest SHA-256")
    _digest(submission_receipt_sha256, "submission receipt SHA-256")
    _digest(collector_sha256, "collector SHA-256")
    if type(fetched_at_unix_ns) is not int or fetched_at_unix_ns <= 0:
        raise CollectionError("fetched_at_unix_ns must be a positive JSON integer")

    identity = validate_pair_wandb_identity(pair_manifest, arm)
    job_id = validate_submission_binding(
        submission_receipt,
        pair_manifest=pair_manifest,
        pair_manifest_sha256=pair_manifest_sha256,
        arm=arm,
    )
    config, provenance = validate_acceptance_contract(
        acceptance_contract,
        identity=identity,
        pair_manifest_sha256=pair_manifest_sha256,
        collector_sha256=collector_sha256,
    )

    validate_wandb_api_authentication(api)
    run_path = f"{identity['entity']}/{identity['project']}/{identity['run_id']}"
    try:
        run = api.run(run_path)
        validate_run_identity(run, identity)
        validate_wandb_run_config(getattr(run, "config", None), config)
        metrics = requested_metrics(identity["environment"])
        history = collect_history(run, metrics)
    except CollectionError:
        raise
    except Exception as error:
        raise CollectionError("W&B run fetch or history scan failed") from error

    scheduler = {
        "job_id": job_id,
        "pair_manifest_sha256": pair_manifest_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
    }
    capture = {
        "api_base_url": WANDB_API_BASE_URL,
        "authenticated": True,
        "collector_sha256": collector_sha256,
        "complete": True,
        "fetched_at_unix_ns": fetched_at_unix_ns,
        "history_method": WANDB_HISTORY_METHOD,
        "requested_metrics": metrics,
        "summary_fallback_used": False,
        "wandb_sdk_version": WANDB_SDK_VERSION,
    }
    payload = _normalize_json(
        {
            "schema": RUN_EXPORT_SCHEMA,
            "identity": identity,
            "scheduler": scheduler,
            "capture": capture,
            "provenance": provenance,
            "config": config,
            "history": history,
        },
        "W&B export payload",
    )
    export_receipt = {
        "canonicalization": WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": canonical_json_sha256(payload, "W&B export non-receipt payload"),
        "config_sha256": canonical_json_sha256(payload["config"], "W&B config"),
        "history_row_count": len(payload["history"]),
        "history_sha256": canonical_json_sha256(payload["history"], "W&B history"),
        "provenance_sha256": canonical_json_sha256(payload["provenance"], "W&B provenance"),
    }
    return {**payload, "export_receipt": export_receipt}


def serialize_export(value: Mapping[str, Any]) -> bytes:
    """Serialize a v2 run export as canonical ASCII JSON plus exactly one LF."""
    _exact_keys(
        value,
        {
            "schema",
            "identity",
            "scheduler",
            "capture",
            "provenance",
            "config",
            "history",
            "export_receipt",
        },
        "W&B run export",
    )
    if value.get("schema") != RUN_EXPORT_SCHEMA:
        raise CollectionError("unexpected W&B run export schema")
    return canonical_json_bytes(value, "W&B run export") + b"\n"


def write_export(path: Path, value: Mapping[str, Any]) -> str:
    """Create an immutable export without overwriting an existing path."""
    raw = serialize_export(value)
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        path.chmod(0o400)
    except OSError as error:
        raise CollectionError(f"cannot publish W&B export {path}: {error}") from error
    return hashlib.sha256(raw).hexdigest()


def create_wandb_api(api_key: str) -> Any:
    """Create the W&B Public API client only after local evidence is valid."""
    if not isinstance(api_key, str) or len(api_key) < 20 or any(character.isspace() for character in api_key):
        raise CollectionError("WANDB_API_KEY must be one non-whitespace secret of at least 20 characters")
    # W&B is optional for offline evaluation; import it only in the collector CLI.
    try:
        import wandb

        if getattr(wandb, "__version__", None) != WANDB_SDK_VERSION:
            raise CollectionError("installed W&B SDK differs from the frozen supported version " f"{WANDB_SDK_VERSION}")

        return wandb.Api(
            api_key=api_key,
            overrides={"base_url": WANDB_API_BASE_URL},
        )
    except Exception as error:
        raise CollectionError("cannot create the authenticated W&B API client") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--submission-receipt", required=True, type=Path)
    parser.add_argument("--expected-submission-receipt-sha256", required=True)
    parser.add_argument("--acceptance-contract", required=True, type=Path)
    parser.add_argument("--expected-acceptance-contract-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=("off", "on"))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _diagnostic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build a NON_ACCEPTANCE latest-point diagnostic from paired local " "wandb-summary.json files.")
    )
    parser.add_argument("--pair-label", required=True)
    parser.add_argument("--pair-root", required=True, type=Path)
    parser.add_argument(
        "--environment",
        required=True,
        choices=tuple(sorted(VERIFIER_METRIC_BY_ENVIRONMENT)),
    )
    parser.add_argument("--off-run-id", required=True)
    parser.add_argument("--on-run-id", required=True)
    parser.add_argument("--off-summary", required=True, type=Path)
    parser.add_argument("--on-summary", required=True, type=Path)
    parser.add_argument("--expected-off-summary-sha256", required=True)
    parser.add_argument("--expected-on-summary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _diagnostic_history_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build a NON_ACCEPTANCE paired history diagnostic from two finished " "exploratory W&B runs.")
    )
    parser.add_argument("--pair-label", required=True)
    parser.add_argument(
        "--environment",
        required=True,
        choices=tuple(sorted(VERIFIER_METRIC_BY_ENVIRONMENT)),
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--off-run-id", required=True)
    parser.add_argument("--on-run-id", required=True)
    parser.add_argument("--off-run-name", required=True)
    parser.add_argument("--on-run-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _run_export_main(argv: list[str]) -> int:
    """Collect one run selected exclusively by authenticated local evidence."""
    args = _parser().parse_args(argv)
    try:
        pair_document = load_document(args.pair_manifest, "Pair manifest")
        pair_manifest = authenticate_document(
            pair_document,
            args.expected_pair_manifest_sha256,
            label="Pair manifest",
            canonical_lf=True,
        )
        submission_document = load_document(args.submission_receipt, "submission receipt")
        submission_receipt = authenticate_document(
            submission_document,
            args.expected_submission_receipt_sha256,
            label="submission receipt",
            canonical_lf=True,
        )
        contract_document = load_document(args.acceptance_contract, "acceptance contract")
        acceptance_contract = authenticate_document(
            contract_document,
            args.expected_acceptance_contract_sha256,
            label="acceptance contract",
            canonical_lf=False,
        )
        collector_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # Complete the semantic local preflight before importing W&B or creating
        # a client.  ``collect_run_export`` intentionally repeats these checks so
        # its programmatic interface has the same fail-closed boundary.
        identity = validate_pair_wandb_identity(pair_manifest, args.arm)
        validate_submission_binding(
            submission_receipt,
            pair_manifest=pair_manifest,
            pair_manifest_sha256=pair_document.sha256,
            arm=args.arm,
        )
        validate_acceptance_contract(
            acceptance_contract,
            identity=identity,
            pair_manifest_sha256=pair_document.sha256,
            collector_sha256=collector_sha256,
        )
        api_key = os.environ.get("WANDB_API_KEY", "")
        api = create_wandb_api(api_key)
        export = collect_run_export(
            pair_manifest=pair_manifest,
            pair_manifest_sha256=pair_document.sha256,
            submission_receipt=submission_receipt,
            submission_receipt_sha256=submission_document.sha256,
            acceptance_contract=acceptance_contract,
            arm=args.arm,
            api=api,
            fetched_at_unix_ns=time.time_ns(),
            collector_sha256=collector_sha256,
        )
        file_sha256 = write_export(args.output, export)
    except CollectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"RUN_EXPORT_V2 arm={args.arm} sha256={file_sha256} path={args.output}",
        flush=True,
    )
    return 0


def _diagnostic_local_summary_main(argv: list[str]) -> int:
    """Collect a digest-bound but explicitly non-acceptance paired diagnostic."""
    args = _diagnostic_parser().parse_args(argv)
    try:
        pair_root = args.pair_root.absolute()
        paths = {
            "off": args.off_summary.absolute(),
            "on": args.on_summary.absolute(),
        }
        collector_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        diagnostic = collect_diagnostic_local_summary_pair(
            pair_label=args.pair_label,
            pair_root=pair_root,
            environment=args.environment,
            run_ids={"off": args.off_run_id, "on": args.on_run_id},
            summary_paths=paths,
            expected_summary_sha256={
                "off": args.expected_off_summary_sha256,
                "on": args.expected_on_summary_sha256,
            },
            collected_at_unix_ns=time.time_ns(),
            collector_sha256=collector_sha256,
        )
        file_sha256 = write_diagnostic_pair_summary(args.output, diagnostic)
    except CollectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        "DIAGNOSTIC_LOCAL_WANDB_PAIR_SUMMARY_V1 " f"acceptance_eligible=false sha256={file_sha256} path={args.output}",
        flush=True,
    )
    return 0


def _diagnostic_wandb_history_main(argv: list[str]) -> int:
    """Collect paired exploratory history without creating acceptance evidence."""
    args = _diagnostic_history_parser().parse_args(argv)
    try:
        collector_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        api = create_wandb_api(os.environ.get("WANDB_API_KEY", ""))
        diagnostic = collect_diagnostic_wandb_history_pair(
            pair_label=args.pair_label,
            environment=args.environment,
            entity=args.entity,
            project=args.project,
            group=args.group,
            run_ids={"off": args.off_run_id, "on": args.on_run_id},
            run_names={"off": args.off_run_name, "on": args.on_run_name},
            api=api,
            fetched_at_unix_ns=time.time_ns(),
            collector_sha256=collector_sha256,
        )
        file_sha256 = write_diagnostic_history_pair(args.output, diagnostic)
    except CollectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        "DIAGNOSTIC_WANDB_HISTORY_PAIR_V1 " f"acceptance_eligible=false sha256={file_sha256} path={args.output}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch the acceptance exporter or the separate diagnostic mode."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv[:1] == ["diagnostic-local-summary-pair"]:
        return _diagnostic_local_summary_main(effective_argv[1:])
    if effective_argv[:1] == ["diagnostic-wandb-history-pair"]:
        return _diagnostic_wandb_history_main(effective_argv[1:])
    return _run_export_main(effective_argv)


if __name__ == "__main__":
    raise SystemExit(main())
