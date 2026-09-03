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

"""Evaluate one strict live W&B shared-prefix OFF/ON pair offline.

This evaluator authenticates the current Pair, submission, execution and W&B
export evidence, then evaluates logged reward consistency, live learning
behavior and paired policy-training speed for reasoning_gym, citation or
freeform. It intentionally makes no captured-output parity or trajectory-
equivalence claim; those are separate replay evaluations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

CONTRACT_SCHEMA = "nemo-rl-strict-single-env-acceptance-contract-v2"

PAIR_MANIFEST_SCHEMA = "nemo-rl-strict-single-env-pair-v2"

SLURM_EXPORT_BOUNDARY_SCHEMA = "nemo-rl-strict-slurm-export-file-v3"

RUNTIME_TOOL_MANIFEST_SCHEMA = "nemo-rl-strict-runtime-tools-v2"

RUN_EXPORT_SCHEMA = "nemo-rl-offline-wandb-run-export-v2"

REPORT_SCHEMA = "nemo-rl-strict-single-env-live-acceptance-report-v1"

ACCEPTANCE_SCOPE = (
    "live_wandb_reward_learning_speed_all_single_environments_" "not_captured_output_or_trajectory_equivalence"
)

JOB_RECEIPT_SCHEMA = "nemo-rl-strict-pair-job-receipt-v2"

SUBMISSION_RECEIPT_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"

TERMINAL_PAIR_RECEIPT_SCHEMA = "nemo-rl-strict-single-env-terminal-pair-receipt-v1"

TERMINAL_CAPTURE_METHOD = "exact-id-scontrol-show-job-json-poll-v1"

EXECUTION_MARKER_RECEIPT_SCHEMA = "nemo-rl-shared-prefix-physical-execution-receipt-v1"

EXECUTION_ENVIRONMENT_SCHEMA = "nemo-rl-strict-execution-environment-v1"

HARDWARE_OBSERVATION_SCHEMA = "nemo-rl-strict-hardware-observation-v1"

SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA = "nemo-rl-strict-scheduler-device-environment-v1"

STEP1_EVIDENCE_INDEX_SCHEMA = "nemo-rl-strict-step1-evidence-index-v4"

MAIN_STEP1_LEDGER_SCHEMA = "nemo-rl-strict-main-step1-ledger-v5"

STEP1_TRANSCRIPT_BUNDLE_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"

MODEL_TRANSPORT_POLICY_SCHEMA = "nemo-rl-strict-model-transport-policy-v1"

MODEL_TRANSPORT_CALL_SCHEMA = "nemo-rl-strict-model-transport-call-v1"

MODEL_TRANSPORT_BUNDLE_SCHEMA = "nemo-rl-strict-model-transport-bundle-v1"

MODEL_TRANSPORT_MANIFEST_SCHEMA = "nemo-rl-strict-model-transport-manifest-v1"

MODEL_TRANSPORT_EVIDENCE_INDEX_SCHEMA = "nemo-rl-strict-model-transport-evidence-index-v1"

STEP1_HASH_DOMAIN = "sha256-domain-nul-canonical-ascii-json-no-lf-v1"

STEP1_HASH_PREFIX = b"nemo-rl-strict-v2\0"

EXECUTION_MARKER_SEMANTICS = "production_packed_fused_training_path"

STEPS = tuple(range(1, 101))

STEPS_PER_EPOCH = 5

PRIMARY_STEPS = tuple(range(11, 101))

PRIMARY_EPOCHS = tuple(tuple(range(first, first + STEPS_PER_EPOCH)) for first in range(11, 101, 5))

TAIL_STEPS = tuple(range(76, 101))

TAIL_EPOCHS = frozenset(range(16, 21))

K4_SAMPLES = 4

MIN_MATCHED_PRIMARY_EPOCHS = 15

BOOTSTRAP_RESAMPLES = 10_000

BOOTSTRAP_SEED = 20_260_828

REWARD_ABS_TOLERANCE = 1e-7

RATIO_ABS_TOLERANCE = 1e-9

MAX_EXACT_INTEGER = 1 << 53

MAX_WANDB_HISTORY_ROWS = 100_000

WANDB_API_BASE_URL = "https://api.wandb.ai"

WANDB_HISTORY_METHOD = "scan_history"

WANDB_SDK_VERSION = "0.28.1"

WANDB_EXPORT_CANONICALIZATION = "sorted-compact-ascii-json-plus-one-lf"

WANDB_ENTITY = "nvidia"

WANDB_PROJECT = "nano35-rlvr-convergence"

WANDB_GROUP_TEMPLATE = "{environment}-{pair_id}"

WANDB_RUN_ID_DERIVATION = "sha256-ascii:nemo-rl-strict-wandb-v1:{environment}:{pair_id}:{arm}"

HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")

HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")

JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")

SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

ENVIRONMENT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

SAFE_POSIX_PATH_RE = re.compile(r"/[A-Za-z0-9._/+:-]+\Z")

FOUR_DECIMAL_IDS_RE = re.compile(r"[0-9]+(?:,[0-9]+){3}\Z")

GPU_UUID_RE = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")

ZE_DEVICE_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")

SLURM_EXPORT_ALLOWED_NAMES = (
    "BASE_LOG_DIR",
    "BATCH_SCRIPT",
    "COLOCATED_GENERATION",
    "COMMAND",
    "CONTAINER",
    "CPUS_PER_WORKER",
    "DEDICATED_RAY_HEAD",
    "DEPLOYMENT_ROOT",
    "EXPECTED_BRIDGE_HEAD",
    "EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256",
    "EXPECTED_BRIDGE_TREE",
    "EXPECTED_DEPLOYMENT_READY",
    "EXPECTED_DEPLOYMENT_READY_FILE_SHA256",
    "EXPECTED_GYM_GITLINK_COMMIT",
    "EXPECTED_GYM_TREE",
    "EXPECTED_MCORE_HEAD",
    "EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256",
    "EXPECTED_MCORE_TREE",
    "EXPECTED_NEMO_HEAD",
    "EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_COUNT",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SCHEMA",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SHA256",
    "EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256",
    "EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256",
    "EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256",
    "EXPECTED_STRICT_PAIR_CONTAINER_SHA256",
    "EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256",
    "EXPECTED_STRICT_PAIR_FIXTURE_SHA256",
    "EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256",
    "EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256",
    "EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256",
    "EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256",
    "EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256",
    "EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256",
    "EXPECTED_STRICT_PAIR_UV_SHIM_SHA256",
    "EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256",
    "EXP_NAME",
    "GPUS_PER_NODE",
    "HF_DATASETS_CACHE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_TOKEN",
    "MODEL_PATH",
    "MOUNTS",
    "NEMO_SKILLS_SANDBOX_PORT",
    "NUM_EXTERNAL_SERVICE_NODES",
    "NUM_GEN_NODES",
    "NUM_GYM_NODES",
    "NUM_TRAIN_NODES",
    "PAIR_ID",
    "PERSISTENT_CACHE",
    "RAY_LOG_SYNC_FREQUENCY",
    "RAY_SUB",
    "RESULTS_DIR",
    "SANDBOX_COMMAND",
    "SANDBOX_CONTAINER",
    "SEGMENT_SIZE",
    "SETUP_COMMAND",
    "STRICT_PAIR_CONTAINER_PYTHON",
    "STRICT_PAIR_CONTAINER_UV",
    "STRICT_PAIR_ENVIRONMENT",
    "STRICT_PAIR_HOST_PYTHON",
    "STRICT_PAIR_JOB_WRAPPER",
    "STRICT_PAIR_LAUNCH_MODE",
    "STRICT_PAIR_RUNTIME_TOOL_MANIFEST",
    "STRICT_PAIR_SHARED_PREFIX_MODE",
    "STRICT_PAIR_UV_SHIM",
    "STRICT_PREBUILT_SNAPSHOT_DIR",
    "TRAIN_PATH",
    "VAL_PATH",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_NAME",
    "WANDB_PROJ",
    "WANDB_RESUME",
    "WANDB_RUN_GROUP",
    "WANDB_RUN_ID",
)

HOST_RUNTIME_TOOL_NAMES = frozenset(
    {
        "awk",
        "bash",
        "cat",
        "chmod",
        "cmp",
        "date",
        "env",
        "find",
        "git",
        "grep",
        "ln",
        "mkdir",
        "mktemp",
        "nvidia_smi",
        "python",
        "readlink",
        "realpath",
        "rm",
        "rsync",
        "sbatch",
        "scancel",
        "scontrol",
        "sha256sum",
        "stat",
        "wc",
    }
)

EXPECTED_HOST_SCHEDULER_TOOLS = {
    "sbatch": {
        "path": "/cm/local/apps/slurm/25.11/bin/sbatch",
        "sha256": "ac1f483625d1005b60e0e650fab381ad55e1a52a42dc0c9bbf625fffcac789fc",
    },
    "scancel": {
        "path": "/cm/local/apps/slurm/25.11/bin/scancel",
        "sha256": "fb2ca904d41c954b993890f91f14bdd8b1ba85566c7006dbb66fd600dd9d6b96",
    },
    "scontrol": {
        "path": "/cm/local/apps/slurm/25.11/bin/scontrol",
        "sha256": "24333d205add15ce6a285ea81b7e55af7b0f35d26451472e6c3a011eba3b3594",
    },
}

CONTAINER_RUNTIME_TOOL_NAMES = frozenset({"python", "uv", "uv_shim"})

CONTAINER_ENTRY_BOUNDARY = {
    "bash_args": ["-p"],
    "bash_path": "/bin/bash",
    "env_path": "/usr/bin/env",
    "sha256sum": {
        "path": "/usr/bin/sha256sum",
        "sha256": ("f3d040161f5c29e4c7cd4e3d6bb513ce9a43b9d1bd06f456a6aab3d34d0f1e33"),
    },
    "unset_environment": ["BASH_ENV", "ENV"],
}

DETERMINISTIC_CONTROLS = {
    "environment": {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MAMBA_DETERMINISTIC": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    },
    "forbidden_triton_controls_absent": True,
    "model_overrides": {
        "cross_entropy_loss_fusion": False,
        "deterministic_mode": True,
        "tp_comm_overlap": False,
    },
    "require_deterministic_execution": True,
}

THREAT_MODEL = {
    "cooperative_exclusive_campaign_root": True,
    "malicious_same_uid_active_mutation": "out_of_scope",
    "trusted_operator_and_code": True,
}

HSG_SLURM_CONF = {
    "path": "/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf",
    "sha256": ("2f81094a7a631b921d33513e6a3d74b96360510b5f9766f75b0cf45ebd95a410"),
}

SECRET_VALUE_FIELD_NAMES = frozenset(
    {
        "env",
        "environment_values",
        "export_values",
        "payload",
        "raw_values",
        "secret",
        "secret_values",
        "secrets",
        "values",
    }
)

SLURM_EXPORT_JOB_ARGV = (
    "--pair-manifest",
    "{pair_manifest_path}",
    "--pair-manifest-sha256",
    "{pair_manifest_sha256}",
    "--arm",
    "{arm}",
)

SOURCE_KEYS = frozenset(
    {
        "nemo_rl",
        "megatron_bridge",
        "megatron_lm",
        "nemo_gym",
    }
)

LEGACY_UNVERIFIED_LINEAGE_SCHEMA = "nemo-rl-strict-trusted-oob-declarations-v1"

UNVERIFIED_LINEAGE_ASSURANCE = "lineage_only_not_runtime_observation_or_correctness_evidence"

UNVERIFIED_LINEAGE_SOURCE_KEYS = frozenset({"nemo_automodel", "pipeclean"})

UNVERIFIED_LINEAGE_SOURCE_RECORD_KEYS = frozenset({"commit", "git_tree", "source_tree_sha256"})

UNVERIFIED_LINEAGE_COMMON_KEYS = frozenset(
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

UNVERIFIED_LINEAGE_ARM_KEYS = frozenset({"code_snapshot_sha256", "reward_ledger_sha256", "train_data_index_sha256"})

COMMON_PROVENANCE_KEYS = frozenset(
    {
        "pair_manifest_sha256",
        "acceptance_contract_sha256",
        "fixture_sha256",
        "model_tree_sha256",
        "training_container_sha256",
        "sandbox_container_sha256",
        "verifier_source_sha256",
        "reward_liveness_contract_sha256",
        "gym_config_sha256",
        "environment_recipe_sha256",
        "launcher_sha256",
        "reward_semantics_contract_sha256",
        "nemo_runnable_manifest_sha256",
        "bridge_runnable_manifest_sha256",
        "mcore_runnable_manifest_sha256",
        "deployment_ready_file_sha256",
        "deployment_ready_sha256",
        "pair_campaign_reward_and_advantage_sha256",
        "pair_campaign_sha256",
        "runtime_tool_manifest_sha256",
        "strict_pair_arm_wrapper_sha256",
        "strict_pair_contract_sha256",
        "strict_pair_parent_wrapper_sha256",
        "submission_contract_sha256",
        "terminal_scheduler_collector_sha256",
        "wandb_exporter_sha256",
    }
)

ARM_PROVENANCE_KEYS = frozenset(
    {
        "runtime_environment_sha256",
        "shared_prefix_runtime_trace_sha256",
        "runtime_direction_receipt_sha256",
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
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
    }
)

JOB_EXIT_KEYS = frozenset(
    {
        "arm",
        "bridge_runnable_manifest_sha256",
        "command_sha256",
        "config_sha256",
        "container_entry_boundary",
        "container_entry_boundary_sha256",
        "container_sha256",
        "deployment_ready",
        "deployment_ready_file_sha256",
        "deployment_ready_sha256",
        "deterministic_controls",
        "driver_exit_code",
        "entrypoint_sha256",
        "environment",
        "execution_environment",
        "fixture_rows",
        "fixture_sha256",
        "gpus_per_node",
        "gym_gitlink_commit",
        "gym_tree",
        "hardware",
        "inner_ray_sha256",
        "job_account",
        "job_id",
        "job_name",
        "job_num_nodes",
        "job_partition",
        "job_qos",
        "mcore_runnable_manifest_sha256",
        "model_tree_sha256_v1",
        "mounts_sha256",
        "nemo_runnable_manifest_sha256",
        "pair_campaign_reward_and_advantage_sha256",
        "pair_campaign_sha256",
        "pair_id",
        "pair_manifest_sha256",
        "phase",
        "post_verified",
        "pre_receipt_sha256",
        "restart_count",
        "reward_semantics_config_sha256",
        "reward_semantics_contract_sha256",
        "runtime_attestation_actual_count",
        "runtime_attestation_aggregate_sha256",
        "runtime_attestation_expected_count",
        "runtime_attestation_marker_sha256",
        "runtime_attestation_receipt_dir",
        "runtime_attestation_receipt_dir_device",
        "runtime_attestation_receipt_dir_inode",
        "runtime_attestation_receipts_sha256",
        "runtime_tool_container_python_path",
        "runtime_tool_container_python_sha256",
        "runtime_tool_container_uv_path",
        "runtime_tool_container_uv_sha256",
        "runtime_tool_host_python_path",
        "runtime_tool_host_python_sha256",
        "runtime_tool_manifest_path",
        "runtime_tool_manifest_sha256",
        "runtime_tool_uv_shim_path",
        "runtime_tool_uv_shim_sha256",
        "sandbox_container_sha256",
        "scheduler_client_environment",
        "scheduler_device_environment",
        "schema",
        "selected_config_sha256",
        "selection",
        "slurm_export_boundary",
        "slurm_export_boundary_sha256",
        "snapshot_manifest_sha256",
        "source",
        "step1_evidence",
        "source_head",
        "source_tree",
        "strict_pair_arm_wrapper_sha256",
        "strict_pair_contract_sha256",
        "strict_pair_parent_wrapper_sha256",
        "submission_contract_path",
        "submission_contract_sha256",
        "submission_nonce",
        "submission_receipt_path",
        "submission_receipt_sha256",
        "wrapper_sha256",
        "wandb",
    }
)

LIVE_LEARNING_ACCEPTANCE_POLICY = {
    "schema": "nemo-rl-strict-live-learning-acceptance-policy-v1",
    "hash_domain": STEP1_HASH_DOMAIN,
    "policy_sha256": ("2424af50fde43b6bd13d2265720ed2415dd4de7138a6fed5535a2603ca0371e8"),
    "live_reward_noninferiority": {
        "metric": "train/raw_environment_reward",
        "burn_in_steps": 10,
        "evaluated_steps": {
            "first": 11,
            "last": 100,
            "count": 90,
            "require_complete_paired_steps": True,
        },
        "paired_step_bootstrap": {
            "statistic": "mean-on-minus-off",
            "sampling_unit": "paired-step",
            "resamples": 10_000,
            "seed": 20_260_828,
            "confidence_level": 0.95,
            "interval": "percentile",
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
        },
        "margin": 0.1,
        "primary_gate": ("lower-confidence-bound-strictly-greater-than-negative-margin"),
        "tail": {
            "steps": {"first": 76, "last": 100, "count": 25},
            "statistic": "mean-on-minus-mean-off",
            "margin": 0.1,
            "gate": "on-mean-greater-than-or-equal-to-off-mean-minus-margin",
        },
    },
    "optimizer_update_witness": {
        "ledger_field": "update_successful",
        "wandb_metric": "train/update_successful",
        "json_type": "boolean",
        "required_value": True,
        "required_steps": {"first": 1, "last": 100, "count": 100},
        "missing_or_non_boolean": "unverifiable",
        "false_value": "fail",
    },
}

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
    "live_learning_policy": LIVE_LEARNING_ACCEPTANCE_POLICY,
}

PAIR_CAMPAIGN_REWARD_AND_ADVANTAGE = {
    "advantage_clip": {"high": 20.0, "low": -20.0},
    "advantage_normalization": True,
    "dynamic_sampling": False,
    "effort_shaping": {
        "low_penalty": 1.0,
        "low_string": "{reasoning effort: efficient}",
        "low_ub": 15000,
        "low_weight": 0.1,
    },
    "invalid_tool_call": {
        "legacy_penalize_flag": True,
        "token_advantage": -5.0,
    },
    "leave_one_out_baseline": True,
    "loss": {
        "force_on_policy_ratio": True,
        "kl_input_clamp_value": None,
        "kl_output_clamp_value": None,
        "ratio_clip_c": None,
        "ratio_clip_max": 0.28,
        "ratio_clip_min": 0.2,
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "sequence_level_importance_ratios": False,
        "token_level": True,
        "truncated_importance_sampling_ratio": 5.0,
        "truncated_importance_sampling_ratio_min": 0.2,
        "truncated_importance_sampling_type": "tis",
        "use_kl_in_reward": False,
        "use_importance_sampling_correction": True,
        "use_on_policy_kl_approximation": True,
    },
    "malformed_thinking_token_advantage": None,
    "metrics": {
        "advantage_estimator_output_before_token_override_and_clip": [
            "train/advantages/mean",
            "train/advantages/min",
            "train/advantages/max",
        ],
        "advantage_estimator_reward_input_after_processing": "train/reward",
        "effective_reward": "train/verifier_reward",
        "effective_reward_legacy_alias": "train/total_reward/mean",
        "final_effective_advantage_after_override_and_clip": None,
        "pre_penalty_reward": "train/pre_penalty_environment_reward",
        "raw_task_score": "train/raw_environment_reward",
        "verifier_native_raw_score_alias": ("train/reasoning_gym_simple_agent/score/mean"),
    },
    "overlong_filtering": False,
    "required_step_relations": {
        "effort_low_sample_rate": "train/effort_low_sample_rate == 0",
        "effort_reward_delta": "train/effort_reward_delta == 0",
        "raw_equals_pre_penalty": ("train/raw_environment_reward == train/pre_penalty_environment_reward"),
        "reward_processing_delta": "train/reward_processing_delta == 0",
        "reward_equals_effective": ("train/reward == train/verifier_reward == train/total_reward/mean"),
    },
    "reward_scaling": {
        "enabled": False,
        "source_max": 1.0,
        "source_min": 0.0,
        "target_max": 1.0,
        "target_min": 0.0,
    },
    "reward_shaping": {
        "enabled": False,
        "max_response_length": 768,
        "overlong_buffer_length": 128,
        "overlong_buffer_penalty": 1.0,
        "stop_properly_penalty_coef": None,
    },
    "sample_mask": {
        "env_flagged_samples": True,
        "seq_logprob_error_threshold": 2.0,
    },
    "zeroing_penalties": {
        "duplicated_reasoning": True,
        "empty_final_answer": True,
        "malformed_think_tag": True,
        "thinking_tags": ["<think>", "</think>"],
        "token_ids": {"think_close": 13, "think_open": 12, "unwanted": [2]},
        "unwanted_token": True,
    },
}

VERIFIER_METRIC_BY_ENVIRONMENT = {
    "citation": "train/citation_format_simple_agent/reward/mean",
    "freeform": "train/freeform_formatting_simple_agent/reward/mean",
    "reasoning_gym": "train/reasoning_gym_simple_agent/score/mean",
}

ENVIRONMENT_SELECTIONS = {
    "reasoning_gym": {
        "config": {
            "path": ("examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml"),
            "sha256": ("f5517d8edabed2b4d77b493fa6a8a5f55fa8eb4b3da33d66f5f40b1afbf5d8c8"),
        },
        "fixture_sha256": ("da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"),
        "gym_resources": {
            "config": {
                "path": "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
                "sha256": ("bdbb459a4a920bc47cf84b1d7dc30aeaa9be35cf0dfac09c77879e45b62a52ab"),
            },
            "requirements": {
                "path": "resources_servers/reasoning_gym/requirements.txt",
                "sha256": ("b00b45db433d797d8a5c5c5602f24ab94d9d5620d83b4bef21fbee851287d411"),
            },
            "verifier_source": {
                "path": "resources_servers/reasoning_gym/app.py",
                "sha256": ("3a35c5d27392dae05499ceefac04e9c32ad963b51a54d77bb470ee59b1fe3127"),
            },
        },
    },
    "citation": {
        "config": {
            "path": "examples/nemo_gym/nemotron-3.5-nano/single_env_citation_sc.yaml",
            "sha256": ("02535ca5b16d7167b32952b7cabe66b224df47325b95d0f241e5872c272b7466"),
        },
        "fixture_sha256": ("d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc"),
        "gym_resources": {
            "config": {
                "path": ("resources_servers/format_verification/configs/citation_format.yaml"),
                "sha256": ("da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"),
            },
            "requirements": {
                "path": "resources_servers/format_verification/requirements.txt",
                "sha256": ("18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"),
            },
            "verifier_source": {
                "path": "resources_servers/format_verification/app.py",
                "sha256": ("6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"),
            },
        },
    },
    "freeform": {
        "config": {
            "path": "examples/nemo_gym/nemotron-3.5-nano/single_env_freeform_sc.yaml",
            "sha256": ("8dbfcb5866799aa4015080c9a615196cebff2fc36db82598f7d08d7c9f35fd8c"),
        },
        "fixture_sha256": ("8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399"),
        "gym_resources": {
            "config": {
                "path": ("resources_servers/format_verification/configs/" "freeform_formatting.yaml"),
                "sha256": ("92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa"),
            },
            "requirements": {
                "path": "resources_servers/format_verification/requirements.txt",
                "sha256": ("18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"),
            },
            "verifier_source": {
                "path": "resources_servers/format_verification/app.py",
                "sha256": ("6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"),
            },
        },
    },
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
        "train/advantages/mean",
        "train/advantages/min",
        "train/advantages/max",
        "train/grad_norm",
        "train/mtp/grad_norm",
        "train/update_successful",
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
    "train/shared_prefix/prompt_tokens",
    "train/shared_prefix/non_loss_suffix_tokens",
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


class EvidenceError(RuntimeError):
    """An input cannot establish the claimed acceptance result."""


@dataclass(frozen=True)
class Document:
    """A parsed JSON object and the SHA-256 of its exact source bytes."""

    value: dict[str, Any]
    sha256: str
    raw: bytes


@dataclass
class History:
    """Sparse W&B observations merged without imputing absent values."""

    values: dict[int, dict[str, bool | int | float]] = dataclass_field(default_factory=dict)
    metric_errors: dict[str, dict[int, list[str]]] = dataclass_field(default_factory=dict)
    global_errors: list[str] = dataclass_field(default_factory=list)

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
            missing = [step for step in steps if metric not in self.values.get(step, {})]
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

    def boolean(self, step: int, metric: str) -> bool:
        """Return one observation only when W&B preserved its JSON boolean type."""
        value = self.values[step][metric]
        if type(value) is not bool:
            raise EvidenceError(f"step {step} {metric} must be an exact JSON boolean")
        return value


@dataclass(frozen=True)
class Run:
    """One authenticated offline W&B run export."""

    arm: str
    document_sha256: str
    scheduler_job_id: str
    identity: dict[str, Any]
    history: History


@dataclass
class Gate:
    """One independently reported acceptance decision."""

    name: str
    failures: list[str] = dataclass_field(default_factory=list)
    unavailable: list[str] = dataclass_field(default_factory=list)
    evidence: dict[str, Any] = dataclass_field(default_factory=dict)

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


def _reject_negative_zero(value: Any, label: str) -> None:
    """Reject JSON negative zero; the v2 canonical policy represents zero as 0.0."""
    if isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise EvidenceError(f"{label} contains forbidden negative zero")
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_negative_zero(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_negative_zero(nested, f"{label}[{index}]")


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
    _reject_negative_zero(value, label)
    return Document(value=value, sha256=hashlib.sha256(raw).hexdigest(), raw=raw)


def document_from_value(value: dict[str, Any], *, trailing_lf: bool = False) -> Document:
    """Construct deterministic in-memory evidence for hermetic callers/tests."""
    raw = _canonical_json_bytes(value, "in-memory evidence")
    if trailing_lf:
        raw += b"\n"
    return Document(value=value, sha256=hashlib.sha256(raw).hexdigest(), raw=raw)


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


def _exact_json_value(actual: Any, expected: Any, label: str) -> None:
    """Compare JSON values without Python's bool/integer equality aliasing."""
    if isinstance(expected, dict):
        mapping = _mapping(actual, label)
        _exact_keys(mapping, set(expected), label)
        for key, value in expected.items():
            _exact_json_value(mapping[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise EvidenceError(f"{label} differs from the exact ordered JSON list")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _exact_json_value(actual_item, expected_item, f"{label}[{index}]")
        return
    if (
        type(actual) is not type(expected)
        or actual != expected
        or (isinstance(actual, float) and actual == 0.0 and math.copysign(1.0, actual) != math.copysign(1.0, expected))
    ):
        raise EvidenceError(f"{label} differs from the exact frozen JSON value {expected!r}")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a non-empty filesystem-safe identifier")
    return value


def _pair_id(value: Any, label: str) -> str:
    if type(value) is not str or PAIR_ID_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a 1..64 byte filesystem-safe identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise EvidenceError(f"{label} must be a populated lowercase SHA-256")
    return value


def _absolute_posix_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or SAFE_POSIX_PATH_RE.fullmatch(value) is None
        or value.startswith("//")
        or "\x00" in value
    ):
        raise EvidenceError(f"{label} must be a non-empty absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or value == "/" or ".." in path.parts:
        raise EvidenceError(f"{label} must be a canonical absolute POSIX path")
    return value


def _relative_posix_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvidenceError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise EvidenceError(f"{label} must be a canonical relative POSIX path")
    if SAFE_POSIX_PATH_RE.fullmatch("/" + value) is None:
        raise EvidenceError(f"{label} contains unsupported path characters")
    return value


def _canonical_json_bytes(value: Any, label: str) -> bytes:
    """Return sorted compact ASCII JSON, with no trailing LF or NaN values."""
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise EvidenceError(f"{label} is not canonical ASCII JSON") from error
    if b"\n" in raw or b"\r" in raw:
        raise EvidenceError(f"{label} canonical JSON unexpectedly contains a newline")
    return raw


def _canonical_json_sha256(value: Any, label: str) -> str:
    raw = _canonical_json_bytes(value, label)
    return hashlib.sha256(raw).hexdigest()


def _step1_projection_sha256(label: str, value: Any) -> str:
    """Hash one exact step-1 projection in the frozen domain."""
    if not isinstance(label, str) or SAFE_ID_RE.fullmatch(label) is None:
        raise EvidenceError("step-1 hash label must be a safe ASCII identifier")
    return hashlib.sha256(
        STEP1_HASH_PREFIX + label.encode("ascii") + b"\0" + _canonical_json_bytes(value, f"step-1 {label} projection")
    ).hexdigest()


def _validate_live_learning_acceptance_policy(value: Any) -> dict[str, Any]:
    """Validate the exact Pair-bound A1 live-learning policy and its digest."""
    policy = _mapping(value, "live-learning acceptance policy")
    _exact_json_value(
        policy,
        LIVE_LEARNING_ACCEPTANCE_POLICY,
        "live-learning acceptance policy",
    )
    payload = copy.deepcopy(policy)
    observed_digest = _digest(
        payload.pop("policy_sha256"),
        "live-learning acceptance policy SHA-256",
    )
    expected_digest = _step1_projection_sha256("live-learning-acceptance-policy", payload)
    if observed_digest != expected_digest:
        raise EvidenceError("live-learning acceptance policy digest does not close")
    return policy


def _acceptance_contract_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash the contract payload with its non-self-referential digest omitted."""
    payload = copy.deepcopy(dict(value))
    provenance = _mapping(payload.get("provenance"), "acceptance provenance payload")
    common = _mapping(provenance.get("common"), "acceptance common payload")
    if "acceptance_contract_sha256" not in common:
        raise EvidenceError("acceptance contract payload lacks acceptance_contract_sha256")
    del common["acceptance_contract_sha256"]
    return _canonical_json_sha256(payload, "acceptance contract payload")


def _expected_pair_campaign(contract: Mapping[str, Any]) -> dict[str, Any]:
    pair = contract["pair"]
    config = contract["configs"]["off"]
    topology = contract["provenance"]["topology"]
    environment = pair["environment"]
    try:
        verifier_metric = VERIFIER_METRIC_BY_ENVIRONMENT[environment]
    except KeyError as error:
        raise EvidenceError(f"unsupported strict single-environment label {environment!r}") from error
    reward_and_advantage = copy.deepcopy(PAIR_CAMPAIGN_REWARD_AND_ADVANTAGE)
    reward_and_advantage["metrics"]["verifier_native_raw_score_alias"] = verifier_metric
    topology_name = (
        f"TP{topology['tensor_parallel_size']}/"
        f"CP{topology['context_parallel_size']}/"
        f"PP{topology['pipeline_parallel_size']}/"
        f"EP{topology['expert_parallel_size']}/"
        f"ETP{topology['expert_tensor_parallel_size']}/"
        f"{'SP' if topology['sequence_parallel'] else 'NO-SP'}"
    )
    return {
        "async_grpo": None,
        "checkpointing_enabled": False,
        "data_plane_enabled": True,
        "data_shuffle": config["data_shuffle"],
        "epochs": config["epochs"],
        "generations_per_prompt": config["num_generations_per_prompt"],
        "generation": {
            "max_new_tokens": config["max_new_tokens"],
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"],
            "vllm_gpu_memory_utilization": 0.1,
        },
        "generation_seed_base": config["generation_seed_base"],
        "hardware": {"gpu_model": topology["gpu_model"]},
        "launch_mode": "submit",
        "logging": {
            "tensorboard_enabled": config["tensorboard_enabled"],
            "wandb_enabled": config["wandb_enabled"],
            "wandb_entity": pair["entity"],
            "wandb_group_template": "{environment}-{pair_id}",
            "wandb_project": pair["project"],
            "wandb_run_name_templates": {
                "off": "off-{environment}-{pair_id}",
                "on": "on-{environment}-{pair_id}",
            },
        },
        "nodes": topology["allocated_nodes"],
        "padding_multiple": 128,
        "prompts_per_step": config["num_prompts_per_step"],
        "require_deterministic_execution": True,
        "reward_and_advantage": reward_and_advantage,
        "rollout_seed_opt_in": True,
        "slurm": {
            "account": config["slurm_account"],
            "partition": config["slurm_partition"],
            "qos": "normal",
            "walltime": "04:00:00",
        },
        "steps": config["max_num_steps"],
        "threat_model": copy.deepcopy(THREAT_MODEL),
        "training_mtp": {
            "detach_heads": config["mtp_detach_heads"],
            "layers": config["mtp_num_layers"],
            "loss_scale": config["mtp_loss_scaling_factor"],
            "repeated_layer": config["mtp_use_repeated_layer"],
        },
        "training_topology": topology_name,
        "vllm_tp": topology["generation_gpus_per_node"],
    }


def _reject_secret_value_fields(value: Any, label: str) -> None:
    """Reject receipt members that could serialize environment secret values."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f"{label} contains a non-string field name")
            if key in SLURM_EXPORT_ALLOWED_NAMES or key.lower() in SECRET_VALUE_FIELD_NAMES:
                raise EvidenceError(f"{label} contains forbidden secret-value field {key!r}")
            _reject_secret_value_fields(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _reject_secret_value_fields(nested, label)


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX40_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise EvidenceError(f"{label} must be a populated 40-hex commit")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise EvidenceError(f"{label} must be finite and representable") from error
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    if result == 0.0 and math.copysign(1.0, result) < 0.0:
        raise EvidenceError(f"{label} must not be negative zero")
    return result


def _canonical_derived_float(value: Any, label: str) -> float:
    """Return a finite derived float with JSON-canonical positive zero."""
    if type(value) is not float or not math.isfinite(value):
        raise EvidenceError(f"{label} must be a finite derived float")
    return 0.0 if value == 0.0 else value


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


def _wandb_run_id(environment: str, pair_id: str, arm: str) -> str:
    return hashlib.sha256(f"nemo-rl-strict-wandb-v1:{environment}:{pair_id}:{arm}".encode("ascii")).hexdigest()


def _expected_pair_wandb(contract: Mapping[str, Any]) -> dict[str, Any]:
    pair = contract["pair"]
    environment = pair["environment"]
    pair_id = pair["pair_id"]
    return {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "group": {
            "template": WANDB_GROUP_TEMPLATE,
            "value": f"{environment}-{pair_id}",
        },
        "resume": "never",
        "arms": {
            arm: {
                "name_template": f"{arm}-{{environment}}-{{pair_id}}",
                "name": f"{arm}-{environment}-{pair_id}",
                "run_id": _wandb_run_id(environment, pair_id, arm),
            }
            for arm in ("off", "on")
        },
        "run_id_derivation": WANDB_RUN_ID_DERIVATION,
    }


def _compact_steps(steps: Sequence[int]) -> str:
    if len(steps) <= 8:
        return str(list(steps))
    return f"{list(steps[:4])}...{list(steps[-4:])} ({len(steps)} total)"


def _validate_hash_mapping(value: Any, required: frozenset[str], label: str) -> dict[str, Any]:
    mapping = _mapping(value, label)
    _exact_keys(mapping, set(required), label)
    for key, item in mapping.items():
        if not isinstance(key, str) or not key.endswith("_sha256"):
            raise EvidenceError(f"{label} has unsupported non-SHA pin {key!r}")
        _digest(item, f"{label}.{key}")
    return mapping


def _validate_unverified_lineage_metadata(value: Any) -> dict[str, Any]:
    """Validate opaque collector-v2 metadata without making it authority."""
    declarations = _mapping(value, "unverified lineage metadata")
    _exact_keys(
        declarations,
        {"arms", "assurance", "common", "schema", "sources"},
        "unverified lineage metadata",
    )
    _exact_json_value(
        declarations["schema"],
        LEGACY_UNVERIFIED_LINEAGE_SCHEMA,
        "legacy unverified lineage schema",
    )
    _exact_json_value(
        declarations["assurance"],
        UNVERIFIED_LINEAGE_ASSURANCE,
        "unverified lineage assurance",
    )
    _validate_hash_mapping(
        declarations["common"],
        UNVERIFIED_LINEAGE_COMMON_KEYS,
        "unverified lineage common metadata",
    )
    arms = _mapping(declarations["arms"], "unverified lineage arm metadata")
    _exact_keys(arms, {"off", "on"}, "unverified lineage arm metadata")
    for arm in ("off", "on"):
        _validate_hash_mapping(
            arms[arm],
            UNVERIFIED_LINEAGE_ARM_KEYS,
            f"unverified lineage {arm} arm metadata",
        )
    sources = _mapping(declarations["sources"], "unverified lineage source metadata")
    _exact_keys(
        sources,
        set(UNVERIFIED_LINEAGE_SOURCE_KEYS),
        "unverified lineage source metadata",
    )
    for name in UNVERIFIED_LINEAGE_SOURCE_KEYS:
        record = _mapping(sources[name], f"unverified lineage {name} source metadata")
        _exact_keys(
            record,
            set(UNVERIFIED_LINEAGE_SOURCE_RECORD_KEYS),
            f"unverified lineage {name} source metadata",
        )
        _commit(record["commit"], f"unverified lineage {name} source commit")
        _commit(record["git_tree"], f"unverified lineage {name} source Git tree")
        _digest(
            record["source_tree_sha256"],
            f"unverified lineage {name} source-tree SHA-256",
        )
    return declarations


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
        "gpu_model": "NVIDIA GB200",
        "nvidia_driver_version": "580.126.20",
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
        _exact_json_value(
            topology[key],
            expected_value,
            f"contract topology {key}",
        )
    return topology


def validate_contract(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the immutable acceptance trust anchor and frozen thresholds."""
    _exact_keys(
        value,
        {
            "schema",
            "pair",
            "campaign",
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
    _exact_json_value(
        value["acceptance"],
        ACCEPTANCE,
        "acceptance thresholds",
    )
    _validate_live_learning_acceptance_policy(
        _mapping(value["acceptance"], "acceptance thresholds")["live_learning_policy"]
    )

    pair = _mapping(value["pair"], "contract pair")
    _exact_keys(
        pair,
        {"pair_id", "environment", "entity", "project", "group", "run_ids"},
        "contract pair",
    )
    _pair_id(pair["pair_id"], "contract pair pair_id")
    for key in ("environment", "entity", "project", "group"):
        _safe_id(pair[key], f"contract pair {key}")
    if pair["environment"] not in VERIFIER_METRIC_BY_ENVIRONMENT:
        raise EvidenceError(f"unsupported strict single-environment label {pair['environment']!r}")
    expected_group = f"{pair['environment']}-{pair['pair_id']}"
    if pair["group"] != expected_group:
        raise EvidenceError("contract W&B group must equal <environment>-<pair_id>")
    _exact_json_value(pair["entity"], WANDB_ENTITY, "contract W&B entity")
    _exact_json_value(pair["project"], WANDB_PROJECT, "contract W&B project")
    run_ids = _mapping(pair["run_ids"], "contract W&B run IDs")
    _exact_keys(run_ids, {"off", "on"}, "contract W&B run IDs")
    for arm in ("off", "on"):
        _safe_id(run_ids[arm], f"contract {arm} W&B run ID")
        _exact_json_value(
            run_ids[arm],
            _wandb_run_id(pair["environment"], pair["pair_id"], arm),
            f"contract {arm} deterministic W&B run ID",
        )
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
            "topology",
            "arms",
            "trusted_oob_declarations",
        },
        "contract provenance",
    )
    common = _validate_hash_mapping(provenance["common"], COMMON_PROVENANCE_KEYS, "common provenance")
    if common["acceptance_contract_sha256"] != _acceptance_contract_payload_sha256(value):
        raise EvidenceError("acceptance_contract_sha256 differs from the canonical contract payload")
    digest_owners: dict[str, list[str]] = {}
    for key, digest in common.items():
        digest_owners.setdefault(digest, []).append(key)
    allowed_alias = {
        "pair_campaign_reward_and_advantage_sha256",
        "reward_semantics_contract_sha256",
    }
    for owners in digest_owners.values():
        if len(owners) > 1 and set(owners) != allowed_alias:
            raise EvidenceError(f"common provenance pins alias independent artifacts: {sorted(owners)}")
    commits = _mapping(provenance["source_commits"], "source commits")
    git_trees = _mapping(provenance["source_git_trees"], "source Git trees")
    _exact_keys(commits, set(SOURCE_KEYS), "source commits")
    _exact_keys(git_trees, set(SOURCE_KEYS), "source Git trees")
    for key in SOURCE_KEYS:
        _commit(commits[key], f"source commit {key}")
        _commit(git_trees[key], f"source Git tree {key}")
    # This legacy collector-v2 field is opaque lineage metadata. Exact export
    # propagation is checked below, but no acceptance gate consumes its values.
    _validate_unverified_lineage_metadata(provenance["trusted_oob_declarations"])
    _validate_topology(provenance["topology"])
    arms = _mapping(provenance["arms"], "arm provenance")
    _exact_keys(arms, {"off", "on"}, "arm provenance")
    for arm in ("off", "on"):
        _validate_hash_mapping(arms[arm], ARM_PROVENANCE_KEYS, f"{arm} arm provenance")
    if arms["off"] == arms["on"]:
        raise EvidenceError("OFF/ON arm provenance must bind distinct artifacts")
    # Both arms intentionally execute the same immutable code snapshot. Their
    # runtime environment, command, mounts, trace, and receipt establish the
    # OFF/ON distinction; inventing different snapshot bytes would weaken it.
    shared_arm_pins = {
        "entrypoint_sha256",
        "inner_ray_sha256",
        "snapshot_manifest_sha256",
        "wrapper_sha256",
    }
    for key in ARM_PROVENANCE_KEYS:
        if key in shared_arm_pins and arms["off"][key] != arms["on"][key]:
            raise EvidenceError(f"OFF/ON {key} must bind the same deployed artifact")
        if key not in shared_arm_pins and arms["off"][key] == arms["on"][key]:
            raise EvidenceError(f"OFF/ON {key} must bind distinct arm artifacts")

    configs = _mapping(value["configs"], "contract configs")
    _exact_keys(configs, {"off", "on"}, "contract configs")
    for arm, mode in (("off", "observe"), ("on", "train")):
        config = _mapping(configs[arm], f"{arm} contract config")
        _exact_keys(config, set(CONFIG_KEYS), f"{arm} contract config")
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
            "wandb_run_name": f"{arm}-{pair['environment']}-{pair['pair_id']}",
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
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
        }
        for key, expected_value in expected.items():
            _exact_json_value(
                config[key],
                expected_value,
                f"{arm} contract config {key}",
            )
    allowed_arm_config_differences = {
        "shared_prefix_mode",
        "wandb_run_name",
    }
    normalized_configs = {
        arm: {key: item for key, item in configs[arm].items() if key not in allowed_arm_config_differences}
        for arm in ("off", "on")
    }
    _exact_json_value(
        normalized_configs["off"],
        normalized_configs["on"],
        "OFF/ON configs outside shared-prefix mode, W&B name, and digest",
    )

    campaign = _mapping(value["campaign"], "acceptance campaign")
    expected_campaign = _expected_pair_campaign(value)
    _exact_json_value(campaign, expected_campaign, "acceptance campaign")
    campaign_sha256 = _canonical_json_sha256(campaign, "acceptance campaign")
    if campaign_sha256 != provenance["common"]["pair_campaign_sha256"]:
        raise EvidenceError("acceptance campaign differs from its provenance digest")
    reward_and_advantage_sha256 = _canonical_json_sha256(
        campaign["reward_and_advantage"],
        "acceptance campaign reward-and-advantage policy",
    )
    if reward_and_advantage_sha256 != provenance["common"]["pair_campaign_reward_and_advantage_sha256"]:
        raise EvidenceError("acceptance reward-and-advantage policy differs from its provenance digest")
    if verifier_metric != campaign["reward_and_advantage"]["metrics"]["verifier_native_raw_score_alias"]:
        raise EvidenceError("verifier_metric differs from the frozen campaign alias")

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
        if type(holdout[key]) is not float:
            raise EvidenceError(f"holdout {key} must be an exact JSON float")
        floor = _number(holdout[key], f"holdout {key}")
        if not 0.05 <= floor <= 1.0:
            raise EvidenceError(f"holdout {key} must be in [0.05, 1]")

    receipts = _mapping(value["receipts"], "receipt pins")
    _exact_keys(
        receipts,
        {
            "shared_prefix_execution_marker_receipts",
            "strict_job_exit_receipts",
            "terminal_scheduler_pair_receipt",
        },
        "receipt pins",
    )
    for group_name in (
        "shared_prefix_execution_marker_receipts",
        "strict_job_exit_receipts",
    ):
        group = _mapping(receipts[group_name], group_name)
        _exact_keys(group, {"off", "on"}, group_name)
        for arm in ("off", "on"):
            _validate_receipt_pin(group[arm], f"{group_name}.{arm}")
    for arm in ("off", "on"):
        if (
            receipts["shared_prefix_execution_marker_receipts"][arm]["sha256"]
            != arms[arm]["runtime_direction_receipt_sha256"]
        ):
            raise EvidenceError(f"{arm} execution marker is not bound by arm provenance")
    terminal_pin = _validate_receipt_pin(
        receipts["terminal_scheduler_pair_receipt"],
        "terminal_scheduler_pair_receipt",
    )
    terminal_semantic = _mapping(
        terminal_pin["semantic_pins"],
        "terminal_scheduler_pair_receipt.semantic_pins",
    )
    _exact_keys(
        terminal_semantic,
        {
            "schema",
            "capture_method",
            "collector_sha256",
            "pair_id",
            "environment",
            "pair_manifest_sha256",
            "submission_receipt_sha256",
            "job_exit_receipt_sha256s",
            "submission_contract_sha256",
            "runtime_tool_manifest_sha256",
            "capture_sha256s",
            "composition_sha256",
        },
        "terminal_scheduler_pair_receipt.semantic_pins",
    )
    _exact_json_value(
        terminal_semantic["schema"],
        TERMINAL_PAIR_RECEIPT_SCHEMA,
        "terminal scheduler receipt schema",
    )
    _exact_json_value(
        terminal_semantic["capture_method"],
        TERMINAL_CAPTURE_METHOD,
        "terminal scheduler capture method",
    )
    _exact_json_value(
        terminal_semantic["collector_sha256"],
        common["terminal_scheduler_collector_sha256"],
        "terminal scheduler collector provenance",
    )
    _exact_json_value(terminal_semantic["pair_id"], pair["pair_id"], "terminal scheduler Pair ID")
    _exact_json_value(
        terminal_semantic["environment"],
        pair["environment"],
        "terminal scheduler environment",
    )
    _exact_json_value(
        terminal_semantic["pair_manifest_sha256"],
        common["pair_manifest_sha256"],
        "terminal scheduler Pair manifest SHA-256",
    )
    _digest(
        terminal_semantic["submission_receipt_sha256"],
        "terminal scheduler submission receipt SHA-256",
    )
    _exact_json_value(
        terminal_semantic["submission_contract_sha256"],
        common["submission_contract_sha256"],
        "terminal scheduler submission contract SHA-256",
    )
    _exact_json_value(
        terminal_semantic["runtime_tool_manifest_sha256"],
        common["runtime_tool_manifest_sha256"],
        "terminal scheduler runtime-tool manifest SHA-256",
    )
    terminal_exit_hashes = _mapping(
        terminal_semantic["job_exit_receipt_sha256s"],
        "terminal scheduler EXIT receipt SHA-256s",
    )
    terminal_capture_hashes = _mapping(
        terminal_semantic["capture_sha256s"],
        "terminal scheduler arm-capture SHA-256s",
    )
    _exact_keys(terminal_exit_hashes, {"off", "on"}, "terminal scheduler EXIT receipt SHA-256s")
    _exact_keys(terminal_capture_hashes, {"off", "on"}, "terminal scheduler arm-capture SHA-256s")
    for arm in ("off", "on"):
        _exact_json_value(
            terminal_exit_hashes[arm],
            receipts["strict_job_exit_receipts"][arm]["sha256"],
            f"terminal scheduler {arm} EXIT receipt SHA-256",
        )
        _digest(terminal_capture_hashes[arm], f"terminal scheduler {arm} capture SHA-256")
    if terminal_capture_hashes["off"] == terminal_capture_hashes["on"]:
        raise EvidenceError("terminal scheduler OFF/ON capture SHA-256s alias")
    _digest(terminal_semantic["composition_sha256"], "terminal scheduler composition SHA-256")
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


def _requested_history_metrics(verifier_metric: str) -> list[str]:
    """Return the exact ordered W&B scan inventory for a v2 export."""
    return ["_step", *sorted(_all_metrics(verifier_metric))]


def _wandb_export_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the non-self-referential receipt for one exact v2 export payload."""
    _exact_keys(
        payload,
        {
            "schema",
            "identity",
            "scheduler",
            "capture",
            "provenance",
            "config",
            "history",
        },
        "W&B export receipt payload",
    )
    history = payload["history"]
    if not isinstance(history, list):
        raise EvidenceError("W&B export receipt history must be a list")
    return {
        "canonicalization": WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": _canonical_json_sha256(payload, "W&B export receipt payload"),
        "config_sha256": _canonical_json_sha256(payload["config"], "W&B export config"),
        "history_row_count": len(history),
        "history_sha256": _canonical_json_sha256(history, "W&B export history"),
        "provenance_sha256": _canonical_json_sha256(payload["provenance"], "W&B export provenance"),
    }


def wandb_export_document_from_payload(payload: dict[str, Any]) -> Document:
    """Seal one v2 export payload for hermetic producers and tests."""
    value = copy.deepcopy(payload)
    value["export_receipt"] = _wandb_export_receipt(value)
    return document_from_value(value, trailing_lf=True)


def _validate_history_rows(rows: Any, *, label: str, verifier_metric: str) -> list[dict[str, Any]]:
    """Validate the exact closed row grammar before sparse-history merging."""
    if not isinstance(rows, list):
        raise EvidenceError(f"{label} history must be a list")
    if len(rows) > MAX_WANDB_HISTORY_ROWS:
        raise EvidenceError(f"{label} history exceeds {MAX_WANDB_HISTORY_ROWS} selected rows")
    requested = set(_requested_history_metrics(verifier_metric))
    for row_index, raw in enumerate(rows):
        row = _mapping(raw, f"{label} history row {row_index}")
        unknown = set(row) - requested
        if unknown:
            raise EvidenceError(f"{label} history row {row_index} has unrequested fields " f"{sorted(unknown)}")
        if "_step" not in row or len(row) < 2:
            raise EvidenceError(
                f"{label} history row {row_index} must contain _step and at least " "one requested metric"
            )
        step = row["_step"]
        if type(step) is not int or step not in STEPS:
            raise EvidenceError(f"{label} history row {row_index} has invalid exact integer _step " f"{step!r}")
        for metric, raw_value in row.items():
            if metric == "_step":
                continue
            if metric == "train/update_successful":
                if type(raw_value) is not bool:
                    raise EvidenceError(f"{label} step {step} {metric} must be an exact JSON boolean")
            elif metric in INTEGER_METRICS:
                if type(raw_value) is not int:
                    raise EvidenceError(f"{label} step {step} {metric} must be an exact JSON integer")
                _integer(raw_value, f"{label} step {step} {metric}")
            else:
                if type(raw_value) is not float:
                    raise EvidenceError(f"{label} step {step} {metric} must be an exact JSON float")
                number = _number(raw_value, f"{label} step {step} {metric}")
                if metric in RATE_METRICS and not 0.0 <= number <= 1.0:
                    raise EvidenceError(f"{label} step {step} {metric} is outside [0, 1]")
                if metric == "timing/train/policy_training" and number <= 0.0:
                    raise EvidenceError(f"{label} step {step} {metric} must be strictly positive")
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["_step"],
            _canonical_json_bytes(row, f"{label} normalized history row"),
        ),
    )
    if rows != expected_order:
        raise EvidenceError(f"{label} history rows must be sorted by _step and canonical row bytes")
    return rows


def merge_sparse_history(rows: Any, *, label: str, verifier_metric: str) -> History:
    """Merge sparse W&B rows by exact integer step, retaining conflicts."""
    history = History()
    if not isinstance(rows, list):
        history.global_errors.append(f"{label} history must be a list")
        return history
    known_metrics = _all_metrics(verifier_metric)
    for row_index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            history.global_errors.append(f"{label} history row {row_index} is not an object")
            continue
        present = known_metrics & set(raw)
        step = raw.get("_step")
        if isinstance(step, bool) or not isinstance(step, int):
            history.global_errors.append(f"{label} history row {row_index} has non-integer _step {step!r}")
            continue
        if step not in STEPS:
            history.global_errors.append(f"{label} history row {row_index} has out-of-contract _step {step}")
            continue
        merged = history.values.setdefault(step, {})
        for metric in present:
            try:
                value: bool | int | float
                if metric == "train/update_successful":
                    if type(raw[metric]) is not bool:
                        raise EvidenceError(f"{label} step {step} {metric} must be an exact JSON boolean")
                    value = raw[metric]
                elif metric in INTEGER_METRICS:
                    value = _integer(raw[metric], f"{label} step {step} {metric}")
                else:
                    value = _number(raw[metric], f"{label} step {step} {metric}")
                    if metric in RATE_METRICS and not 0.0 <= value <= 1.0:
                        raise EvidenceError(f"{label} step {step} {metric} is outside [0, 1]")
            except EvidenceError as error:
                history.metric_errors.setdefault(metric, {}).setdefault(step, []).append(str(error))
                continue
            if metric in merged and merged[metric] != value:
                history.metric_errors.setdefault(metric, {}).setdefault(step, []).append(
                    f"{label} step {step} has conflicting {metric}: " f"{merged[metric]!r} versus {value!r}"
                )
                continue
            merged[metric] = value
    return history


def _validate_run(
    document: Document,
    expected_sha256: Any,
    contract: dict[str, Any],
    arm: str,
    *,
    pair_manifest_sha256: str,
    submission_receipt_sha256: str,
    scheduler_job_id: str,
) -> Run:
    if not isinstance(document, Document):
        raise EvidenceError(f"{arm} run export must be an authenticated Document")
    expected_digest = _digest(expected_sha256, f"trusted {arm} W&B export SHA-256")
    if document.sha256 != expected_digest:
        raise EvidenceError(f"{arm} W&B export bytes differ from the trusted OOB SHA-256")
    _require_canonical_document(document, f"{arm} W&B export", trailing_lf=True)
    _reject_negative_zero(document.value, f"{arm} W&B export")
    value = document.value
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
        "run_name": f"{arm}-{pair['environment']}-{pair['pair_id']}",
        "state": "finished",
    }
    _exact_json_value(identity, expected_identity, f"{arm} W&B identity")

    scheduler = _mapping(value["scheduler"], f"{arm} W&B scheduler binding")
    _exact_json_value(
        scheduler,
        {
            "job_id": scheduler_job_id,
            "pair_manifest_sha256": pair_manifest_sha256,
            "submission_receipt_sha256": submission_receipt_sha256,
        },
        f"{arm} W&B scheduler binding",
    )

    capture = _mapping(value["capture"], f"{arm} W&B capture")
    _exact_keys(
        capture,
        {
            "api_base_url",
            "authenticated",
            "collector_sha256",
            "complete",
            "fetched_at_unix_ns",
            "history_method",
            "requested_metrics",
            "summary_fallback_used",
            "wandb_sdk_version",
        },
        f"{arm} W&B capture",
    )
    _exact_json_value(capture["api_base_url"], WANDB_API_BASE_URL, f"{arm} W&B API")
    _exact_json_value(capture["authenticated"], True, f"{arm} W&B authentication")
    _exact_json_value(capture["complete"], True, f"{arm} W&B completion")
    _exact_json_value(capture["history_method"], WANDB_HISTORY_METHOD, f"{arm} W&B history method")
    _exact_json_value(
        capture["requested_metrics"],
        _requested_history_metrics(contract["verifier_metric"]),
        f"{arm} W&B requested metrics",
    )
    _exact_json_value(
        capture["summary_fallback_used"],
        False,
        f"{arm} W&B summary fallback",
    )
    _exact_json_value(
        capture["wandb_sdk_version"],
        WANDB_SDK_VERSION,
        f"{arm} W&B SDK version",
    )
    if type(capture["fetched_at_unix_ns"]) is not int or capture["fetched_at_unix_ns"] < 1:
        raise EvidenceError(f"{arm} W&B fetched-at timestamp must be a positive integer")
    collector_sha256 = _digest(capture["collector_sha256"], f"{arm} W&B collector SHA-256")
    if collector_sha256 != contract["provenance"]["common"]["wandb_exporter_sha256"]:
        raise EvidenceError(f"{arm} W&B collector differs from common provenance")

    expected_provenance = {
        "common": contract["provenance"]["common"],
        "source_commits": contract["provenance"]["source_commits"],
        "source_git_trees": contract["provenance"]["source_git_trees"],
        "trusted_oob_declarations": contract["provenance"]["trusted_oob_declarations"],
        "topology": contract["provenance"]["topology"],
        "arm": contract["provenance"]["arms"][arm],
    }
    _exact_json_value(value["provenance"], expected_provenance, f"{arm} W&B provenance")
    _exact_json_value(value["config"], contract["configs"][arm], f"{arm} W&B config")
    history_rows = _validate_history_rows(
        value["history"],
        label=arm,
        verifier_metric=contract["verifier_metric"],
    )
    receipt = _mapping(value["export_receipt"], f"{arm} W&B export receipt")
    expected_receipt = _wandb_export_receipt({key: item for key, item in value.items() if key != "export_receipt"})
    _exact_json_value(receipt, expected_receipt, f"{arm} W&B export receipt")
    return Run(
        arm=arm,
        document_sha256=document.sha256,
        scheduler_job_id=scheduler_job_id,
        identity=identity,
        history=merge_sparse_history(
            history_rows,
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
    _exact_json_value(actual, expected, path)


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


def _require_canonical_document(document: Document, label: str, *, trailing_lf: bool) -> None:
    expected = _canonical_json_bytes(document.value, label)
    if trailing_lf:
        expected += b"\n"
    if document.raw != expected:
        suffix = "plus one LF" if trailing_lf else "without LF"
        raise EvidenceError(f"{label} must be sorted compact canonical ASCII JSON {suffix}")


def _required_artifact(artifacts: Mapping[str, Any], key: str, label: str) -> Document:
    errors = artifacts.get("__errors__", {})
    if isinstance(errors, Mapping) and key in errors:
        raise EvidenceError(f"{label} is unavailable: {errors[key]}")
    document = artifacts.get(key)
    if not isinstance(document, Document):
        raise EvidenceError(f"missing {label}")
    return document


def _retained_adjacent_source(path: Path, label: str) -> bytes:
    """Read one adjacent source file once without following a final symlink."""
    if not path.is_absolute():
        raise EvidenceError(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            raise EvidenceError(f"{label} must be one nonempty regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise EvidenceError(f"{label} ended while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise EvidenceError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise EvidenceError(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise EvidenceError(f"cannot read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_terminal_collector(expected_sha256: str) -> ModuleType:
    path = Path(__file__).resolve(strict=True).with_name("collect_strict_single_env_terminal_jobs.py")
    raw = _retained_adjacent_source(path, "terminal scheduler collector")
    expected = _digest(expected_sha256, "terminal scheduler collector SHA-256")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise EvidenceError("terminal scheduler collector differs from contract provenance")
    name = "nemo_rl_strict_single_env_terminal_collector_for_live_evaluator"
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except Exception as error:
        raise EvidenceError(f"cannot load retained terminal scheduler collector: {error}") from error
    return module


def _validate_terminal_scheduler_receipt(
    contract: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    submitted_job_ids: Mapping[str, str],
    exit_job_ids: Mapping[str, str],
) -> dict[str, str]:
    document = _required_artifact(
        artifacts,
        "terminal_scheduler",
        "terminal scheduler Pair receipt",
    )
    _require_canonical_document(document, "terminal scheduler Pair receipt", trailing_lf=True)
    pin = contract["receipts"]["terminal_scheduler_pair_receipt"]
    value = _authenticate_receipt(document, pin, label="terminal scheduler Pair receipt")
    common = contract["provenance"]["common"]
    collector = _load_terminal_collector(common["terminal_scheduler_collector_sha256"])
    pair_document = _required_artifact(artifacts, "pair_manifest", "Pair manifest")
    submission_document = _required_artifact(
        artifacts,
        "submission_receipt",
        "pair submission receipt",
    )
    exit_documents = {
        arm: _required_artifact(artifacts, f"{arm}_job_exit", f"{arm} strict job EXIT receipt") for arm in ("off", "on")
    }
    expected_exit_sha256s = {
        arm: contract["receipts"]["strict_job_exit_receipts"][arm]["sha256"] for arm in ("off", "on")
    }
    collector_document = collector.Document(
        value=document.value,
        raw=document.raw,
        sha256=document.sha256,
    )
    collector_pair_document = collector.Document(
        value=pair_document.value,
        raw=pair_document.raw,
        sha256=pair_document.sha256,
    )
    collector_submission_document = collector.Document(
        value=submission_document.value,
        raw=submission_document.raw,
        sha256=submission_document.sha256,
    )
    collector_exit_documents = {
        arm: collector.Document(
            value=exit_documents[arm].value,
            raw=exit_documents[arm].raw,
            sha256=exit_documents[arm].sha256,
        )
        for arm in ("off", "on")
    }
    try:
        validated = collector.validate_pair_receipt(
            collector_document,
            pair_document=collector_pair_document,
            submission_document=collector_submission_document,
            exit_documents=collector_exit_documents,
            expected_pair_sha256=common["pair_manifest_sha256"],
            expected_submission_sha256=value["submission_receipt_sha256"],
            expected_exit_sha256s=expected_exit_sha256s,
            expected_collector_sha256=common["terminal_scheduler_collector_sha256"],
        )
        terminal_job_ids = {
            arm: _step1_job_id(
                validated["captures"][arm]["terminal_record"]["job_id"],
                f"terminal scheduler {arm} job ID",
            )
            for arm in ("off", "on")
        }
        _exact_json_value(terminal_job_ids, dict(submitted_job_ids), "terminal/submission scheduler job IDs")
        _exact_json_value(terminal_job_ids, dict(exit_job_ids), "terminal/EXIT scheduler job IDs")
        _exact_json_value(validated["pair_id"], pair_manifest["pair_id"], "terminal scheduler Pair ID")
        _exact_json_value(
            validated["environment"],
            pair_manifest["selection"]["environment"],
            "terminal scheduler environment",
        )
    except Exception as error:
        raise EvidenceError(f"terminal scheduler receipt rejected: {type(error).__name__}: {error}") from error
    return terminal_job_ids


def _exact_nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceError(f"{label} must be a nonnegative JSON integer")
    return value


def _validate_submission_state_record(
    value: Any,
    *,
    arm: str,
    phase: str,
    expected_label: Mapping[str, str],
    expected_work_dir: str,
    label: str,
) -> None:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "comment",
            "held",
            "job_id",
            "job_name",
            "job_state",
            "reason",
            "user_id",
            "work_dir",
        },
        label,
    )
    identity = {key: record[key] for key in ("comment", "job_id", "job_name", "user_id")}
    _exact_json_value(identity, dict(expected_label), f"{label} identity")
    _exact_json_value(record["work_dir"], expected_work_dir, f"{label} work directory")
    if type(record["job_state"]) is not str or type(record["reason"]) is not str:
        raise EvidenceError(f"{label} scheduler state must contain strings")
    for field in ("job_state", "reason"):
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in record[field]):
            raise EvidenceError(f"{label} scheduler {field} must be control-character-free")
    held = record["job_state"] == "PENDING" and record["reason"] == "JobHeldUser"
    _exact_json_value(record["held"], held, f"{label} held-state derivation")
    if phase == "pre" and not held:
        raise EvidenceError(f"{label} is not held before release")
    if phase == "recovery" and not held:
        raise EvidenceError(f"{label} is not held during scheduler recovery")
    if phase == "post" and (
        held or record["job_state"] not in {"PENDING", "CONFIGURING", "RUNNING"} or record["reason"] == "JobHeldUser"
    ):
        raise EvidenceError(f"{label} is not a valid released scheduler record")
    if phase not in {"pre", "post", "recovery"}:
        raise EvidenceError(f"{label} has an unsupported scheduler phase")


def _validate_submission_release_query(
    value: Any,
    *,
    phase: str,
    candidate_ids: Mapping[str, str],
    labels: Mapping[str, Mapping[str, str]],
    expected_work_dirs: Mapping[str, str],
    scontrol_path: str,
) -> None:
    label = f"submission receipt {phase}-release query"
    query = _mapping(value, label)
    _exact_keys(
        query,
        {
            "argv",
            "authenticated_job_ids",
            "byte_count",
            "candidate_job_ids",
            "complete",
            "line_count",
            "normalization_status",
            "output_sha256_raw",
            "phase",
            "records",
            "securely_unlinked",
            "status",
            "unterminated_final_line",
            "unresolved_job_ids",
        },
        label,
    )
    ordered_ids = [candidate_ids["off"], candidate_ids["on"]]
    expected = {
        "argv": [
            scontrol_path,
            "show",
            "job",
            "--json",
            ",".join(ordered_ids),
        ],
        "authenticated_job_ids": ordered_ids,
        "candidate_job_ids": {
            "off": [candidate_ids["off"]],
            "on": [candidate_ids["on"]],
            "unattributed": [],
        },
        "complete": True,
        "normalization_status": 0,
        "phase": phase,
        "securely_unlinked": True,
        "status": 0,
        "unterminated_final_line": False,
        "unresolved_job_ids": [],
    }
    for key, expected_value in expected.items():
        _exact_json_value(query[key], expected_value, f"{label}.{key}")
    byte_count = _exact_nonnegative_integer(query["byte_count"], f"{label}.byte_count")
    line_count = _exact_nonnegative_integer(query["line_count"], f"{label}.line_count")
    if byte_count <= 0 or line_count != len(ordered_ids) or byte_count < line_count:
        raise EvidenceError(f"{label} has impossible or incomplete raw scheduler-output counts")
    _digest(query["output_sha256_raw"], f"{label}.output_sha256_raw")
    records = _mapping(query["records"], f"{label}.records")
    _exact_keys(records, {"off", "on"}, f"{label}.records")
    for arm in ("off", "on"):
        arm_records = records[arm]
        if not isinstance(arm_records, list) or len(arm_records) != 1:
            raise EvidenceError(f"{label}.records.{arm} must contain exactly one row")
        _validate_submission_state_record(
            arm_records[0],
            arm=arm,
            phase=phase,
            expected_label=labels[arm],
            expected_work_dir=expected_work_dirs[arm],
            label=f"{label}.records.{arm}[0]",
        )


def _authenticate_submission_receipt_bytes(
    artifacts: Mapping[str, Any],
    expected_sha256: Any,
    pair_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, str]:
    if expected_sha256 is None:
        raise EvidenceError("missing trusted submission receipt SHA-256")
    expected = _digest(expected_sha256, "trusted expected submission receipt SHA-256")
    document = _required_artifact(artifacts, "submission_receipt", "pair submission receipt")
    if document.sha256 != expected:
        raise EvidenceError("pair submission receipt bytes differ from the trusted OOB SHA-256")
    canonical = _canonical_json_bytes(document.value, "pair submission receipt") + b"\n"
    if document.raw != canonical:
        raise EvidenceError("pair submission receipt must be sorted compact canonical ASCII JSON plus one LF")

    receipt = document.value
    _exact_keys(
        receipt,
        {
            "acceptance",
            "authenticated_jobs",
            "cancellations",
            "execution_environment",
            "held_submissions",
            "model_transport",
            "outcome",
            "pair",
            "post_cancel_queries",
            "post_release_query",
            "pre_cancel_queries",
            "pre_release_query",
            "receipt",
            "recovery_query",
            "release",
            "rollback_candidates",
            "rollback_confirmed",
            "runtime_tools",
            "scheduler_tools",
            "schema",
            "selection",
            "source",
            "stage",
            "submission_contract",
            "submission_nonce",
            "wandb",
        },
        "pair submission receipt",
    )
    _exact_json_value(receipt["schema"], SUBMISSION_RECEIPT_SCHEMA, "receipt schema")
    _exact_json_value(receipt["outcome"], "released", "receipt outcome")
    _exact_json_value(receipt["stage"], "complete", "receipt stage")
    _exact_json_value(receipt["rollback_confirmed"], None, "receipt rollback status")
    _exact_json_value(receipt["cancellations"], [], "receipt cancellations")
    _exact_json_value(
        receipt["acceptance"],
        pair_manifest["acceptance"],
        "submission receipt live-learning acceptance binding",
    )
    _validate_live_learning_acceptance_policy(receipt["acceptance"])
    _exact_json_value(
        receipt["execution_environment"],
        pair_manifest["execution_environment"],
        "submission receipt execution-environment binding",
    )
    _exact_json_value(
        receipt["model_transport"],
        pair_manifest["model_transport"],
        "submission receipt model-transport binding",
    )

    pair_id = contract["pair"]["pair_id"]
    results_root = pair_manifest["paths"]["results_root"]
    pair_manifest_sha256 = contract["provenance"]["common"]["pair_manifest_sha256"]
    pair_manifest_path = f"{results_root}/PAIR_MANIFEST.json"
    _exact_json_value(
        receipt["pair"],
        {
            "id": pair_id,
            "manifest": {
                "path": pair_manifest_path,
                "sha256": pair_manifest_sha256,
            },
        },
        "submission receipt Pair binding",
    )
    _exact_json_value(
        receipt["selection"],
        pair_manifest["selection"],
        "submission receipt selection binding",
    )
    _exact_json_value(
        receipt["wandb"],
        pair_manifest["wandb"],
        "submission receipt W&B identity binding",
    )
    _exact_json_value(
        receipt["source"],
        {
            "bridge": pair_manifest["source"]["bridge"],
            "mcore": pair_manifest["source"]["mcore"],
        },
        "submission receipt Bridge/MCore source binding",
    )
    scheduler_submission = pair_manifest["scheduler_submission"]
    _exact_json_value(
        receipt["receipt"],
        scheduler_submission["receipt"],
        "submission receipt self-binding",
    )
    _exact_json_value(
        receipt["submission_contract"],
        scheduler_submission["contract"],
        "submission receipt contract binding",
    )
    _exact_json_value(
        receipt["submission_nonce"],
        scheduler_submission["nonce"],
        "submission receipt nonce",
    )
    _exact_json_value(
        receipt["runtime_tools"],
        {
            "manifest": pair_manifest["runtime_tools"]["manifest"],
            "schema": RUNTIME_TOOL_MANIFEST_SCHEMA,
        },
        "submission receipt runtime tools",
    )

    host_tools = pair_manifest["runtime_tools"]["document"]["host"]
    scheduler_tools = _mapping(receipt["scheduler_tools"], "receipt scheduler tools")
    _exact_keys(
        scheduler_tools,
        {"client_environment", "sbatch", "scancel", "scontrol"},
        "receipt scheduler tools",
    )
    for name in ("sbatch", "scancel", "scontrol"):
        _exact_json_value(scheduler_tools[name], host_tools[name], f"receipt scheduler tool {name}")
    client_environment = _mapping(scheduler_tools["client_environment"], "receipt scheduler client environment")
    _exact_keys(
        client_environment,
        {"ambient_merge", "env", "variables"},
        "receipt scheduler client environment",
    )
    _exact_json_value(
        client_environment["ambient_merge"],
        False,
        "receipt scheduler ambient merge",
    )
    _exact_json_value(client_environment["env"], host_tools["env"], "receipt scheduler env tool")
    variables = _mapping(client_environment["variables"], "receipt scheduler environment variables")
    _exact_keys(variables, {"LC_ALL", "SLURM_CONF"}, "receipt scheduler environment variables")
    _exact_json_value(variables["LC_ALL"], "C", "receipt scheduler LC_ALL")
    slurm_conf = _mapping(variables["SLURM_CONF"], "receipt scheduler SLURM_CONF")
    _exact_json_value(slurm_conf, HSG_SLURM_CONF, "receipt scheduler HSG SLURM_CONF")

    held = _mapping(receipt["held_submissions"], "receipt held submissions")
    authenticated = _mapping(receipt["authenticated_jobs"], "receipt authenticated jobs")
    _exact_keys(held, {"off", "on"}, "receipt held submissions")
    _exact_keys(authenticated, {"off", "on"}, "receipt authenticated jobs")
    candidate_ids: dict[str, str] = {}
    labels: dict[str, dict[str, str]] = {}
    identity = pair_manifest["scheduler_submission"]["identity"]
    nonce = scheduler_submission["nonce"]
    for arm in ("off", "on"):
        held_record = _mapping(held[arm], f"receipt held_submissions.{arm}")
        _exact_keys(
            held_record,
            {
                "accepted_id_record",
                "candidate_job_id",
                "candidate_job_id_sha256_ascii_no_newline",
                "candidate_job_id_source",
                "submission_rpc",
                "wrapper_status",
            },
            f"receipt held_submissions.{arm}",
        )
        candidate = held_record["candidate_job_id"]
        candidate = _step1_job_id(candidate, f"receipt {arm} candidate job ID")
        candidate_ids[arm] = candidate
        _exact_json_value(
            held_record["candidate_job_id_sha256_ascii_no_newline"],
            hashlib.sha256(candidate.encode("ascii")).hexdigest(),
            f"receipt {arm} candidate job-ID digest",
        )
        _exact_json_value(
            held_record["candidate_job_id_source"],
            "accepted-id-record",
            f"receipt {arm} candidate authority",
        )
        _exact_json_value(held_record["wrapper_status"], 0, f"receipt {arm} wrapper status")
        accepted = _mapping(held_record["accepted_id_record"], f"receipt {arm} accepted-ID record")
        _exact_keys(
            accepted,
            {"parsed_job_id", "path", "sha256"},
            f"receipt {arm} accepted-ID record",
        )
        expected_accepted_path = scheduler_submission["accepted_id_records"][arm]["path"]
        _exact_json_value(accepted["path"], expected_accepted_path, f"receipt {arm} accepted-ID path")
        parsed_job_id = _step1_job_id(accepted["parsed_job_id"], f"receipt {arm} parsed accepted job ID")
        expected_accepted_sha256 = hashlib.sha256(f"{parsed_job_id}\n".encode("ascii")).hexdigest()
        _exact_json_value(
            accepted["sha256"],
            expected_accepted_sha256,
            f"receipt {arm} accepted-ID digest",
        )
        if parsed_job_id != candidate:
            raise EvidenceError(f"receipt {arm} durable job ID differs")
        rpc = _mapping(held_record["submission_rpc"], f"receipt {arm} submission RPC")
        _exact_keys(
            rpc,
            {
                "drained_unix_ns",
                "relay_status",
                "sbatch_status",
                "started_unix_ns",
                "writer_drained",
            },
            f"receipt {arm} submission RPC",
        )
        started = rpc["started_unix_ns"]
        drained = rpc["drained_unix_ns"]
        if type(started) is not int or type(drained) is not int or started <= 0 or drained < started:
            raise EvidenceError(f"receipt {arm} submission timestamps are invalid")
        _exact_json_value(rpc["sbatch_status"], 0, f"receipt {arm} sbatch status")
        _exact_json_value(rpc["relay_status"], 0, f"receipt {arm} relay status")
        _exact_json_value(rpc["writer_drained"], True, f"receipt {arm} writer drain")
        expected_label = {
            "comment": (f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair_manifest_sha256}"),
            "job_id": candidate,
            "job_name": identity["job_names"][arm],
            "user_id": str(identity["submitter_euid"]),
        }
        _exact_json_value(
            authenticated[arm],
            [expected_label],
            f"receipt authenticated_jobs.{arm}",
        )
        labels[arm] = expected_label
    if candidate_ids["off"] == candidate_ids["on"]:
        raise EvidenceError("submission receipt reuses one job ID for both arms")

    scontrol_path = host_tools["scontrol"]["path"]
    expected_work_dirs = {arm: pair_manifest["source"]["snapshots"][arm]["path"] for arm in ("off", "on")}
    _validate_submission_release_query(
        receipt["pre_release_query"],
        phase="pre",
        candidate_ids=candidate_ids,
        labels=labels,
        expected_work_dirs=expected_work_dirs,
        scontrol_path=scontrol_path,
    )
    _validate_submission_release_query(
        receipt["post_release_query"],
        phase="post",
        candidate_ids=candidate_ids,
        labels=labels,
        expected_work_dirs=expected_work_dirs,
        scontrol_path=scontrol_path,
    )
    if receipt["pre_release_query"]["output_sha256_raw"] == receipt["post_release_query"]["output_sha256_raw"]:
        raise EvidenceError("submission receipt PRE/POST scheduler queries reuse one raw-output digest")
    _exact_json_value(receipt["pre_cancel_queries"], [], "receipt pre-cancel queries")
    _exact_json_value(receipt["post_cancel_queries"], [], "receipt post-cancel queries")
    _exact_json_value(
        receipt["rollback_candidates"],
        {
            "off": [candidate_ids["off"]],
            "on": [candidate_ids["on"]],
            "unattributed": [],
        },
        "receipt rollback candidates",
    )
    release = _mapping(receipt["release"], "receipt release RPC")
    _exact_keys(
        release,
        {"argv", "output_sha256_ascii_no_newline", "status"},
        "receipt release RPC",
    )
    _exact_json_value(
        release["argv"],
        [
            scontrol_path,
            "release",
            f"{candidate_ids['off']},{candidate_ids['on']}",
        ],
        "receipt release argv",
    )
    _digest(release["output_sha256_ascii_no_newline"], "receipt release output digest")
    _exact_json_value(release["status"], 0, "receipt release status")

    # A successfully released pair has two distinct durable accepted IDs and
    # therefore must never use the unscoped rollback-only recovery scan.
    _exact_json_value(receipt["recovery_query"], None, "released receipt recovery query")
    return candidate_ids


def _validate_runtime_tool_record(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    _exact_keys(record, {"path", "sha256"}, label)
    _absolute_posix_path(record["path"], f"{label} path")
    _digest(record["sha256"], f"{label} SHA-256")
    return record


def _validate_pair_runtime_tools(manifest: Mapping[str, Any]) -> None:
    runtime_tools = _mapping(manifest.get("runtime_tools"), "Pair manifest runtime tools")
    _exact_keys(
        runtime_tools,
        {"bootstrap_sha256sum", "document", "manifest"},
        "Pair manifest runtime tools",
    )
    document = _mapping(runtime_tools["document"], "Pair manifest runtime-tool document")
    _exact_keys(
        document,
        {"schema", "host", "container"},
        "Pair manifest runtime-tool document",
    )
    if document["schema"] != RUNTIME_TOOL_MANIFEST_SCHEMA:
        raise EvidenceError("unexpected Pair manifest runtime-tool schema")

    host = _mapping(document["host"], "Pair manifest host runtime tools")
    container = _mapping(document["container"], "Pair manifest container runtime tools")
    _exact_keys(host, set(HOST_RUNTIME_TOOL_NAMES), "Pair manifest host runtime tools")
    _exact_keys(
        container,
        set(CONTAINER_RUNTIME_TOOL_NAMES),
        "Pair manifest container runtime tools",
    )
    for scope, records in (("host", host), ("container", container)):
        for name in sorted(records):
            _validate_runtime_tool_record(records[name], f"Pair manifest {scope} runtime tool {name}")
    _exact_json_value(
        host["nvidia_smi"]["path"],
        "/usr/bin/nvidia-smi",
        "Pair manifest host nvidia-smi path",
    )
    for name, expected in EXPECTED_HOST_SCHEDULER_TOOLS.items():
        _exact_json_value(host[name], expected, f"Pair manifest host {name} tool record")

    bootstrap = _validate_runtime_tool_record(
        runtime_tools["bootstrap_sha256sum"],
        "Pair manifest bootstrap sha256sum",
    )
    if bootstrap != host["sha256sum"]:
        raise EvidenceError("Pair manifest bootstrap sha256sum differs from the host inventory")

    manifest_record = _validate_runtime_tool_record(runtime_tools["manifest"], "Pair manifest runtime-tool manifest")
    expected_manifest_bytes = _canonical_json_bytes(document, "Pair runtime-tool document") + b"\n"
    if hashlib.sha256(expected_manifest_bytes).hexdigest() != manifest_record["sha256"]:
        raise EvidenceError("Pair runtime-tool manifest digest differs from its canonical document")
    deployment = _mapping(manifest.get("deployment"), "Pair manifest deployment")
    deployment_root = _absolute_posix_path(deployment.get("root"), "Pair manifest deployment root")
    expected_manifest_path = (PurePosixPath(deployment_root) / "strict_pair_runtime_tools.json").as_posix()
    if manifest_record["path"] != expected_manifest_path:
        raise EvidenceError("Pair manifest runtime-tool manifest is outside its canonical deployment path")


def _validate_pair_container_entry_boundary(manifest: Mapping[str, Any]) -> None:
    boundary = _mapping(
        manifest.get("container_entry_boundary"),
        "Pair manifest container-entry boundary",
    )
    _exact_keys(
        boundary,
        {"bash_args", "bash_path", "env_path", "sha256sum", "unset_environment"},
        "Pair manifest container-entry boundary",
    )
    sha256sum = _mapping(boundary["sha256sum"], "Pair manifest container-entry sha256sum")
    _exact_keys(
        sha256sum,
        {"path", "sha256"},
        "Pair manifest container-entry sha256sum",
    )
    if boundary != CONTAINER_ENTRY_BOUNDARY:
        raise EvidenceError("Pair manifest container-entry boundary differs from the fixed image contract")


def _determinism_marker(mode: str) -> str:
    return (
        "SHARED_PREFIX_DETERMINISM_ATTESTED "
        f"mode={mode} env_controls=5 triton_autotune=absent "
        "model_overrides=3 torch_deterministic=true mcore_backward=true "
        "total_controls=9"
    )


def _validate_pair_runtime_attestation(manifest: Mapping[str, Any]) -> None:
    attestation = _mapping(manifest.get("runtime_attestation"), "Pair manifest runtime attestation")
    _exact_keys(
        attestation,
        {
            "expected_count_per_fresh_process_group",
            "lines",
            "receipt_requires_line_count_and_hash",
            "schema",
        },
        "Pair manifest runtime attestation",
    )
    _exact_json_value(
        attestation["expected_count_per_fresh_process_group"],
        4,
        "Pair manifest runtime attestation expected count",
    )
    _exact_json_value(
        attestation["receipt_requires_line_count_and_hash"],
        True,
        "Pair manifest runtime attestation receipt requirement",
    )
    _exact_json_value(
        attestation["schema"],
        "nemo-rl-shared-prefix-determinism-attestation-v1",
        "Pair manifest runtime attestation schema",
    )
    lines = _mapping(attestation["lines"], "Pair manifest attestation lines")
    _exact_keys(lines, {"off", "on"}, "Pair manifest attestation lines")
    for arm, mode in (("off", "observe"), ("on", "train")):
        line = _mapping(lines[arm], f"Pair manifest {arm} attestation line")
        _exact_keys(
            line,
            {"mode", "sha256_ascii_no_newline", "text"},
            f"Pair manifest {arm} attestation line",
        )
        expected_text = _determinism_marker(mode)
        expected = {
            "mode": mode,
            "sha256_ascii_no_newline": hashlib.sha256(expected_text.encode("ascii")).hexdigest(),
            "text": expected_text,
        }
        _exact_json_value(line, expected, f"Pair manifest {arm} attestation line")


def _runtime_attestation_aggregate_sha256(names: Sequence[str], marker_text: str) -> str:
    marker_bytes = marker_text.encode("ascii")
    aggregate = hashlib.sha256()
    for name in sorted(names):
        aggregate.update(name.encode("ascii") + b"\0" + marker_bytes + b"\0")
    return aggregate.hexdigest()


def _validate_pair_common_provenance_bindings(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Bind independent acceptance pins to the Pair manifest that was executed."""
    common = contract["provenance"]["common"]

    deployment = _mapping(manifest.get("deployment"), "Pair manifest deployment")
    _exact_keys(
        deployment,
        {
            "bridge_runnable_manifest_sha256",
            "mcore_runnable_manifest_sha256",
            "nemo_runnable_manifest_sha256",
            "ready",
            "ready_file_sha256",
            "root",
        },
        "Pair manifest deployment",
    )
    deployment_bindings = {
        "bridge_runnable_manifest_sha256": "bridge_runnable_manifest_sha256",
        "mcore_runnable_manifest_sha256": "mcore_runnable_manifest_sha256",
        "nemo_runnable_manifest_sha256": "nemo_runnable_manifest_sha256",
        "ready": "deployment_ready_sha256",
        "ready_file_sha256": "deployment_ready_file_sha256",
    }
    for manifest_key, common_key in deployment_bindings.items():
        actual = _digest(deployment[manifest_key], f"Pair manifest deployment {manifest_key}")
        if actual != common[common_key]:
            raise EvidenceError(f"Pair manifest deployment {manifest_key} differs from common provenance")

    runtime_tools = _mapping(manifest.get("runtime_tools"), "Pair manifest runtime tools")
    runtime_manifest = _mapping(runtime_tools.get("manifest"), "Pair manifest runtime-tool manifest")
    if runtime_manifest["sha256"] != common["runtime_tool_manifest_sha256"]:
        raise EvidenceError("Pair manifest runtime-tool manifest differs from common provenance")

    scheduler = _mapping(manifest.get("scheduler_submission"), "Pair manifest scheduler submission")
    submission_contract = _mapping(scheduler.get("contract"), "Pair manifest submission contract")
    _exact_keys(
        submission_contract,
        {"path", "sha256"},
        "Pair manifest submission contract",
    )
    _absolute_posix_path(submission_contract["path"], "Pair manifest submission contract path")
    submission_contract_sha256 = _digest(
        submission_contract["sha256"],
        "Pair manifest submission contract SHA-256",
    )
    if submission_contract_sha256 != common["submission_contract_sha256"]:
        raise EvidenceError("Pair manifest submission contract differs from common provenance")

    source = _mapping(manifest.get("source"), "Pair manifest source")
    source_bindings = {
        "arm_wrapper_sha256": "strict_pair_arm_wrapper_sha256",
        "contract_sha256": "strict_pair_contract_sha256",
        "parent_wrapper_sha256": "strict_pair_parent_wrapper_sha256",
    }
    for manifest_key, common_key in source_bindings.items():
        actual = _digest(source.get(manifest_key), f"Pair manifest source {manifest_key}")
        if actual != common[common_key]:
            raise EvidenceError(f"Pair manifest source {manifest_key} differs from common provenance")


def _validate_pair_gym_source(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    source = _mapping(manifest.get("source"), "Pair manifest source")
    gym = _mapping(source.get("gym"), "Pair manifest Gym source")
    _exact_keys(gym, {"gitlink_commit", "path", "tree"}, "Pair manifest Gym source")
    commit = _commit(gym["gitlink_commit"], "Pair manifest Gym gitlink commit")
    tree = _commit(gym["tree"], "Pair manifest Gym Git tree")
    path = _absolute_posix_path(gym["path"], "Pair manifest Gym path")
    if commit != contract["provenance"]["source_commits"]["nemo_gym"]:
        raise EvidenceError("Pair manifest Gym commit differs from the acceptance pin")
    if tree != contract["provenance"]["source_git_trees"]["nemo_gym"]:
        raise EvidenceError("Pair manifest Gym tree differs from the acceptance pin")

    deployment = _mapping(manifest.get("deployment"), "Pair manifest deployment")
    deployment_root = _absolute_posix_path(deployment.get("root"), "Pair manifest deployment root")
    expected_path = (PurePosixPath(deployment_root) / "runnable/NemoRL/3rdparty/Gym-workspace/Gym").as_posix()
    if path != expected_path:
        raise EvidenceError("Pair manifest Gym path differs from the deployed gitlink")


def _validate_pair_selection(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(manifest.get("selection"), "Pair manifest selection")
    _exact_keys(
        selection,
        {"config", "environment", "fixture", "gym_resources"},
        "Pair manifest selection",
    )
    environment = selection["environment"]
    if type(environment) is not str or environment != contract["pair"]["environment"]:
        raise EvidenceError("Pair manifest selected environment differs from the acceptance contract")
    try:
        frozen = ENVIRONMENT_SELECTIONS[environment]
    except KeyError as error:
        raise EvidenceError(f"unsupported Pair environment {environment!r}") from error

    config = _mapping(selection["config"], "Pair manifest selected config")
    _exact_json_value(
        config,
        frozen["config"],
        "Pair manifest selected config",
    )
    _digest(config["sha256"], "Pair manifest selected config SHA-256")
    _relative_posix_path(config["path"], "Pair manifest selected config path")

    fixture = _mapping(selection["fixture"], "Pair manifest selected fixture")
    _exact_keys(
        fixture,
        {"path", "rows", "sha256"},
        "Pair manifest selected fixture",
    )
    _absolute_posix_path(fixture["path"], "Pair manifest selected fixture path")
    _exact_json_value(fixture["rows"], 5, "Pair manifest selected fixture row count")
    _exact_json_value(
        fixture["sha256"],
        frozen["fixture_sha256"],
        "Pair manifest selected fixture SHA-256",
    )

    gym_resources = _mapping(selection["gym_resources"], "Pair manifest selected Gym resources")
    _exact_json_value(
        gym_resources,
        frozen["gym_resources"],
        "Pair manifest selected Gym resources",
    )
    for name, record in gym_resources.items():
        _relative_posix_path(record["path"], f"Pair manifest selected Gym {name} path")

    common = contract["provenance"]["common"]
    bindings = {
        "environment_recipe_sha256": config["sha256"],
        "fixture_sha256": fixture["sha256"],
        "gym_config_sha256": gym_resources["config"]["sha256"],
        "verifier_source_sha256": gym_resources["verifier_source"]["sha256"],
    }
    for common_key, selected_sha256 in bindings.items():
        if common[common_key] != selected_sha256:
            raise EvidenceError(f"Pair selection differs from common provenance {common_key}")
    if manifest.get("artifacts", {}).get("fixture") != fixture:
        raise EvidenceError("Pair selection fixture differs from the artifact fixture")
    source = _mapping(manifest.get("source"), "Pair manifest source")
    if source.get("config_sha256") != config["sha256"]:
        raise EvidenceError("Pair selection config differs from source.config_sha256")
    return selection


def _validate_pair_artifacts_and_paths(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    common = contract["provenance"]["common"]
    artifacts = _mapping(manifest.get("artifacts"), "Pair manifest artifacts")
    _exact_keys(
        artifacts,
        {"container", "fixture", "model", "sandbox_container"},
        "Pair manifest artifacts",
    )
    for name in ("container", "sandbox_container"):
        record = _mapping(artifacts[name], f"Pair manifest {name} artifact")
        _exact_keys(record, {"path", "sha256"}, f"Pair manifest {name} artifact")
        _absolute_posix_path(record["path"], f"Pair manifest {name} path")
        _digest(record["sha256"], f"Pair manifest {name} SHA-256")
    fixture = _mapping(artifacts["fixture"], "Pair manifest fixture artifact")
    _exact_keys(
        fixture,
        {"path", "rows", "sha256"},
        "Pair manifest fixture artifact",
    )
    _absolute_posix_path(fixture["path"], "Pair manifest fixture path")
    _exact_json_value(fixture["rows"], 5, "Pair manifest fixture rows")
    _digest(fixture["sha256"], "Pair manifest fixture SHA-256")
    model = _mapping(artifacts["model"], "Pair manifest model artifact")
    _exact_keys(model, {"path", "tree_sha256_v1"}, "Pair manifest model artifact")
    _absolute_posix_path(model["path"], "Pair manifest model path")
    _digest(model["tree_sha256_v1"], "Pair manifest model tree SHA-256")
    artifact_bindings = (
        (artifacts["container"]["sha256"], common["training_container_sha256"]),
        (artifacts["fixture"]["sha256"], common["fixture_sha256"]),
        (artifacts["model"]["tree_sha256_v1"], common["model_tree_sha256"]),
        (
            artifacts["sandbox_container"]["sha256"],
            common["sandbox_container_sha256"],
        ),
    )
    if any(actual != expected for actual, expected in artifact_bindings):
        raise EvidenceError("Pair artifacts differ from common provenance pins")

    paths = _mapping(manifest.get("paths"), "Pair manifest paths")
    _exact_keys(
        paths,
        {"cache_root", "hf_home", "results_root", "snapshot_parent"},
        "Pair manifest paths",
    )
    for name, path in paths.items():
        _absolute_posix_path(path, f"Pair manifest {name}")
    expected_snapshot_parent = (
        PurePosixPath(paths["results_root"]) / "code_snapshots_strict_pairs" / manifest["pair_id"]
    ).as_posix()
    if paths["snapshot_parent"] != expected_snapshot_parent:
        raise EvidenceError("Pair snapshot parent differs from its canonical pair path")


def _validate_execution_environment(value: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    execution = _mapping(value, "Pair execution environment")
    _exact_keys(
        execution,
        {"arm_launcher", "arms", "fixed", "schema"},
        "Pair execution environment",
    )
    _exact_json_value(
        execution["schema"],
        EXECUTION_ENVIRONMENT_SCHEMA,
        "Pair execution-environment schema",
    )
    _exact_json_value(
        execution["arm_launcher"],
        {
            "ambient_merge": False,
            "argv_prefix": ["-i"],
            "forbidden_caller_names": [
                "BASE_LOG_DIR",
                "CPUS_PER_WORKER",
                "NEMO_SKILLS_SANDBOX_PORT",
                "RAY_LOG_SYNC_FREQUENCY",
                "SANDBOX_COMMAND",
                "SETUP_COMMAND",
                "SLURM_SUBMIT_DIR",
            ],
        },
        "Pair arm-launcher environment isolation",
    )
    paths = _mapping(manifest.get("paths"), "Pair paths for execution environment")
    selection = _mapping(manifest.get("selection"), "Pair selection for execution environment")
    fixture = _mapping(selection.get("fixture"), "Pair fixture for execution environment")
    source = _mapping(manifest.get("source"), "Pair source for execution environment")
    snapshots = _mapping(source.get("snapshots"), "Pair snapshots for execution environment")
    fixture_path = fixture.get("path")
    expected_fixed = {
        "cpus_per_worker": "144",
        "nemo_skills_sandbox_port": "6000",
        "ray_log_sync_frequency": "60",
        "sandbox_command": "/start-with-nginx.sh",
        "train_path": fixture_path,
        "val_path": fixture_path,
    }
    _exact_json_value(execution["fixed"], expected_fixed, "Pair fixed environment")
    arms = _mapping(execution["arms"], "Pair execution-environment arms")
    _exact_keys(arms, {"off", "on"}, "Pair execution-environment arms")
    for arm in ("off", "on"):
        record = _mapping(arms[arm], f"Pair {arm} execution environment")
        _exact_keys(
            record,
            {
                "base_log_dir",
                "cache_read",
                "hf_datasets_cache",
                "hf_home",
                "hf_hub_cache",
                "persistent_cache",
                "results_dir",
                "scheduler",
                "setup_command",
            },
            f"Pair {arm} execution environment",
        )
        arm_results = f"{paths['results_root']}/{arm}"
        arm_cache = f"{paths['cache_root']}/{arm}"
        arm_hf_home = f"{paths['hf_home']}/{arm}"
        snapshot = _mapping(snapshots.get(arm), f"Pair {arm} snapshot for execution environment").get("path")
        expected_values = {
            "base_log_dir": f"{arm_results}/ray_logs",
            "cache_read": {
                "entry_count": 0,
                "mode": "0700",
                "path": f"{arm_cache}/cache_read",
                "policy": "empty-at-publication-and-job-entry-no-read",
            },
            "hf_datasets_cache": f"{arm_hf_home}/hub",
            "hf_home": arm_hf_home,
            "hf_hub_cache": f"{arm_hf_home}/hub",
            "persistent_cache": arm_cache,
            "results_dir": arm_results,
            "scheduler": {
                "batch_working_directory": snapshot,
                "sbatch_chdir_argument": f"--chdir={snapshot}",
                "sbatch_client_cwd": snapshot,
                "slurm_submit_dir": snapshot,
            },
        }
        for key, expected in expected_values.items():
            _exact_json_value(record[key], expected, f"Pair {arm} execution environment {key}")
        setup = _mapping(record["setup_command"], f"Pair {arm} SETUP_COMMAND")
        expected_setup_bytes = _expected_setup_command_bytes()
        _exact_json_value(
            setup,
            {
                "byte_count": len(expected_setup_bytes),
                "sha256": hashlib.sha256(expected_setup_bytes).hexdigest(),
            },
            f"Pair {arm} SETUP_COMMAND",
        )
    for field in ("persistent_cache", "hf_home", "results_dir", "base_log_dir"):
        if arms["off"][field] == arms["on"][field]:
            raise EvidenceError(f"Pair OFF/ON {field} must be disjoint")
    if arms["off"]["cache_read"]["path"] == arms["on"]["cache_read"]["path"]:
        raise EvidenceError("Pair OFF/ON cache-read directories must be disjoint")
    if arms["off"]["setup_command"] != arms["on"]["setup_command"]:
        raise EvidenceError("Pair OFF/ON SETUP_COMMAND records must be exactly equal")
    return execution


def _expected_setup_command_bytes() -> bytes:
    """Return the reviewed cache-clearing, no-seed setup command exactly."""
    text = '''echo "[CACHE SETUP] Clearing stale node-local caches; strict pairs forbid cache seeds."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_* \\
  /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache
mkdir -p /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache
echo "[CACHE SETUP] Done."'''
    return text.encode("ascii")


def _validate_pair_scheduler_submission(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    pair_id = contract["pair"]["pair_id"]
    environment = contract["pair"]["environment"]
    results_root = manifest["paths"]["results_root"]
    submission = _mapping(manifest.get("scheduler_submission"), "Pair manifest scheduler submission")
    _exact_keys(
        submission,
        {"accepted_id_records", "contract", "identity", "nonce", "receipt", "schema"},
        "Pair manifest scheduler submission",
    )
    _exact_json_value(
        submission["schema"],
        "nemo-rl-strict-scheduler-submission-v1",
        "Pair manifest scheduler submission schema",
    )
    nonce = _digest(submission["nonce"], "Pair manifest submission nonce")
    contract_record = _mapping(submission["contract"], "Pair manifest submission contract")
    expected_contract = {
        "path": f"{results_root}/STRICT_PAIR_SUBMISSION_CONTRACT.json",
        "sha256": contract["provenance"]["common"]["submission_contract_sha256"],
    }
    _exact_json_value(contract_record, expected_contract, "Pair manifest submission contract")
    receipt = _mapping(submission["receipt"], "Pair manifest submission receipt")
    _exact_json_value(
        receipt,
        {
            "path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
            "schema": SUBMISSION_RECEIPT_SCHEMA,
        },
        "Pair manifest submission receipt",
    )
    identity = _mapping(submission["identity"], "Pair manifest scheduler identity")
    _exact_keys(
        identity,
        {"comment_template", "job_names", "submitter_euid"},
        "Pair manifest scheduler identity",
    )
    submitter_euid = identity["submitter_euid"]
    if type(submitter_euid) is not int or submitter_euid < 0:
        raise EvidenceError("Pair manifest scheduler submitter_euid must be an integer")
    _exact_json_value(
        identity["comment_template"],
        "nemo-rl-strict-pair-v1:{arm}:{submission_nonce}:{pair_manifest_sha256}",
        "Pair manifest scheduler comment template",
    )
    _exact_json_value(
        identity["job_names"],
        {
            "off": f"off-{environment}-{pair_id}",
            "on": f"on-{environment}-{pair_id}",
        },
        "Pair manifest scheduler job names",
    )
    accepted = _mapping(submission["accepted_id_records"], "Pair manifest accepted-ID records")
    _exact_keys(accepted, {"off", "on"}, "Pair manifest accepted-ID records")
    for arm in ("off", "on"):
        expected = {
            "accepted_format": "ascii-positive-decimal-lf",
            "capture_format": "opaque-sbatch-stdout",
            "initial_mode": "0600",
            "path": (f"{results_root}/strict_pair_submission_state/{pair_id}/{arm}.job-id"),
            "sealed_mode": "0400",
        }
        _exact_json_value(accepted[arm], expected, f"Pair manifest {arm} accepted-ID record")
    if nonce == contract["provenance"]["common"]["pair_manifest_sha256"]:
        raise EvidenceError("Pair submission nonce aliases the Pair manifest digest")


def _validate_pair_source(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    source = _mapping(manifest.get("source"), "Pair manifest source")
    _exact_keys(
        source,
        {
            "arm_wrapper_sha256",
            "bridge",
            "config_sha256",
            "contract_sha256",
            "entrypoint_sha256",
            "gym",
            "head",
            "job_wrapper",
            "launcher_sha256",
            "mcore",
            "parent_wrapper_sha256",
            "root",
            "snapshots",
            "tree",
        },
        "Pair manifest source",
    )
    deployment_root = manifest["deployment"]["root"]
    expected_source_root = (PurePosixPath(deployment_root) / "runnable/NemoRL").as_posix()
    _exact_json_value(source["root"], expected_source_root, "Pair source root")
    commits = contract["provenance"]["source_commits"]
    trees = contract["provenance"]["source_git_trees"]
    _exact_json_value(source["head"], commits["nemo_rl"], "Pair source HEAD")
    _exact_json_value(source["tree"], trees["nemo_rl"], "Pair source tree")
    _commit(source["head"], "Pair source HEAD")
    _commit(source["tree"], "Pair source tree")
    for name, source_key, directory in (
        ("bridge", "megatron_bridge", "Megatron-Bridge"),
        ("mcore", "megatron_lm", "Megatron-LM"),
    ):
        record = _mapping(source[name], f"Pair source {name}")
        expected_record = {
            "root": (PurePosixPath(deployment_root) / f"runnable/{directory}").as_posix(),
            "head": commits[source_key],
            "tree": trees[source_key],
        }
        _exact_json_value(record, expected_record, f"Pair source {name}")
    common = contract["provenance"]["common"]
    if source["launcher_sha256"] != common["launcher_sha256"]:
        raise EvidenceError("Pair source launcher differs from common provenance")
    for key in (
        "arm_wrapper_sha256",
        "config_sha256",
        "contract_sha256",
        "entrypoint_sha256",
        "launcher_sha256",
        "parent_wrapper_sha256",
    ):
        _digest(source[key], f"Pair source {key}")
    job_wrapper = _mapping(source["job_wrapper"], "Pair source job wrapper")
    _exact_keys(job_wrapper, {"path", "sha256"}, "Pair source job wrapper")
    job_wrapper_path = PurePosixPath(_absolute_posix_path(job_wrapper["path"], "Pair source job-wrapper path"))
    try:
        relative_job_wrapper_path = job_wrapper_path.relative_to(PurePosixPath(deployment_root))
    except ValueError as error:
        raise EvidenceError("Pair source job-wrapper path must be contained by the deployment root") from error
    if not relative_job_wrapper_path.parts:
        raise EvidenceError("Pair source job-wrapper path must name a file below the deployment root")
    _digest(job_wrapper["sha256"], "Pair source job-wrapper SHA-256")
    arms = contract["provenance"]["arms"]
    for arm in ("off", "on"):
        if job_wrapper["sha256"] != arms[arm]["wrapper_sha256"]:
            raise EvidenceError(f"Pair source job wrapper differs from {arm} arm provenance")

    snapshots = _mapping(source["snapshots"], "Pair source snapshots")
    _exact_keys(snapshots, {"off", "on"}, "Pair source snapshots")
    for arm in ("off", "on"):
        record = _mapping(snapshots[arm], f"Pair source {arm} snapshot")
        expected = {
            "config_sha256": source["config_sha256"],
            "entrypoint_sha256": source["entrypoint_sha256"],
            "manifest_sha256": arms[arm]["snapshot_manifest_sha256"],
            "path": f"{manifest['paths']['snapshot_parent']}/{arm}-{manifest['pair_id']}",
        }
        _exact_json_value(record, expected, f"Pair source {arm} snapshot")
        if source["entrypoint_sha256"] != arms[arm]["entrypoint_sha256"]:
            raise EvidenceError(f"Pair source entrypoint differs from {arm} arm provenance")


def _validate_pair_model_transport(value: Any) -> None:
    """Validate the closed, pre-run raw model-transport capture policy."""
    policy = _mapping(value, "Pair model transport policy")
    _exact_keys(
        policy,
        {
            "activation",
            "arms",
            "artifacts",
            "capture_window",
            "enabled",
            "hash_domain",
            "http",
            "policy_sha256",
            "schema",
            "sources",
        },
        "Pair model transport policy",
    )
    sources = _mapping(policy["sources"], "Pair model transport sources")
    _exact_keys(
        sources,
        {"collector", "rollout_finalizer", "vllm_route"},
        "Pair model transport sources",
    )
    expected_source_paths = {
        "collector": "nemo_rl/utils/strict_model_transport.py",
        "rollout_finalizer": "nemo_rl/experience/rollout_manager.py",
        "vllm_route": "nemo_rl/models/generation/vllm/vllm_worker_async.py",
    }
    normalized_sources: dict[str, dict[str, str]] = {}
    for name, expected_path in expected_source_paths.items():
        record = _mapping(sources[name], f"Pair model transport source {name}")
        _exact_keys(record, {"path", "sha256"}, f"Pair model transport source {name}")
        _exact_json_value(record["path"], expected_path, f"Pair model transport source {name} path")
        normalized_sources[name] = {
            "path": expected_path,
            "sha256": _digest(record["sha256"], f"Pair model transport source {name} SHA-256"),
        }
    expected = {
        "schema": MODEL_TRANSPORT_POLICY_SCHEMA,
        "hash_domain": STEP1_HASH_DOMAIN,
        "enabled": True,
        "arms": ["off", "on"],
        "sources": normalized_sources,
        "activation": {
            "config_key": "policy.generation.vllm_cfg.strict_model_transport",
            "main_mode": "capture",
            "replay_mode": "replay",
            "pair_id_environment": "PAIR_ID",
            "environment_environment": "STRICT_PAIR_ENVIRONMENT",
            "arm_environment": {"name": "STRICT_PAIR_ARM", "off": "off", "on": "on"},
            "results_dir_environment": "RESULTS_DIR",
        },
        "capture_window": {
            "step": 1,
            "fixture_row_index": 0,
            "sample_count": 4,
            "logical_rollout_indices": [0, 1, 2, 3],
            "seed_base": 42,
            "seed_derivation": "sha256-canonical-ascii-json-int63-v1",
            "concurrency": "arrival-independent",
            "duplicate_or_retry": "reject",
            "seal": "atomic-after-four-successes",
            "main_after_seal": ("reject-until-rollout-finalizer-attests-step1-complete-then-pass-through"),
            "replay_after_seal": "reject-terminal",
        },
        "http": {
            "body_boundary": "http-body-bytes-only",
            "headers": "excluded",
            "cookies": "excluded",
            "authorization": "excluded",
            "query": "forbidden",
            "encoding": "utf-8",
            "request_media_type": "application/json",
            "response_media_type": "application/json",
            "response_status_code": 200,
            "streaming": False,
            "max_request_body_bytes": 16_777_216,
            "max_response_body_bytes": 16_777_216,
            "max_bundle_bytes": 201_326_592,
            "endpoint_allowlist": [{"method": "POST", "path": "/v1/chat/completions", "logical_count": 4}],
            "probe_allowlist": [],
            "tokenize_count": 0,
            "unlisted_during_window": "reject",
            "unlisted_during_replay": "reject",
            "direct_python_generation_during_replay": "reject",
        },
        "artifacts": {
            "directory": {
                "relative_path": "strict_model_transport",
                "mode": "0700",
                "inventory": [
                    "model-transport.jsonl",
                    "model-transport-bundle.json",
                    "model-transport-manifest.json",
                ],
                "precondition": "absent-at-pre-runtime-creates-exclusively",
            },
            "log": {
                "relative_path": "strict_model_transport/model-transport.jsonl",
                "schema": MODEL_TRANSPORT_CALL_SCHEMA,
                "framing": "canonical-ascii-json-line-lf",
                "mode": "0400",
                "lines": 4,
            },
            "bundle": {
                "relative_path": "strict_model_transport/model-transport-bundle.json",
                "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
                "framing": "canonical-ascii-json-no-lf",
                "mode": "0400",
            },
            "manifest": {
                "relative_path": "strict_model_transport/model-transport-manifest.json",
                "schema": MODEL_TRANSPORT_MANIFEST_SCHEMA,
                "framing": "canonical-ascii-json-no-lf",
                "mode": "0400",
            },
            "owner": "effective-uid",
            "nlink": 1,
            "publication": "o-excl-fsync-atomic-seal",
        },
    }
    for key, expected_value in expected.items():
        _exact_json_value(policy[key], expected_value, f"Pair model transport {key}")
    payload = copy.deepcopy(policy)
    observed_digest = _digest(payload.pop("policy_sha256"), "Pair model transport policy SHA-256")
    if observed_digest != _step1_projection_sha256("model-transport-policy", payload):
        raise EvidenceError("Pair model transport policy digest does not close")


def _validate_pair_manifest(document: Document, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the execution Pair manifest and its sealed Slurm boundary."""
    expected_sha256 = contract["provenance"]["common"]["pair_manifest_sha256"]
    if document.sha256 != expected_sha256:
        raise EvidenceError("Pair manifest bytes differ from the pinned SHA-256")
    expected_bytes = _canonical_json_bytes(document.value, "Pair manifest") + b"\n"
    if document.raw != expected_bytes:
        raise EvidenceError("Pair manifest bytes must be sorted compact canonical ASCII JSON plus one LF")

    manifest = document.value
    _exact_keys(
        manifest,
        {
            "acceptance",
            "arms",
            "artifacts",
            "campaign",
            "container_entry_boundary",
            "deployment",
            "determinism_receipt_dir",
            "execution_environment",
            "model_transport",
            "pair_campaign_reward_and_advantage_sha256",
            "pair_campaign_sha256",
            "pair_id",
            "paths",
            "runtime_attestation",
            "runtime_tools",
            "scheduler_submission",
            "schema",
            "selection",
            "slurm_export_boundary",
            "source",
            "wandb",
        },
        "Pair manifest",
    )
    if manifest.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise EvidenceError("unexpected Pair manifest schema")
    if manifest.get("pair_id") != contract["pair"]["pair_id"]:
        raise EvidenceError("Pair manifest pair_id differs from the acceptance contract")
    if manifest.get("arms") != {"off": "observe", "on": "train"}:
        raise EvidenceError("Pair manifest arms differ from the strict OFF/ON modes")
    _exact_json_value(
        manifest.get("acceptance"),
        contract["acceptance"]["live_learning_policy"],
        "Pair manifest live-learning acceptance policy",
    )
    _validate_live_learning_acceptance_policy(manifest.get("acceptance"))
    _validate_pair_model_transport(manifest.get("model_transport"))
    _exact_json_value(
        manifest.get("wandb"),
        _expected_pair_wandb(contract),
        "Pair manifest W&B identity",
    )
    _exact_json_value(
        manifest.get("determinism_receipt_dir"),
        "shared_prefix_determinism_receipts",
        "Pair manifest determinism receipt directory",
    )
    campaign = _mapping(manifest.get("campaign"), "Pair manifest campaign")
    _exact_json_value(
        campaign,
        contract["campaign"],
        "Pair manifest campaign",
    )
    common = contract["provenance"]["common"]
    campaign_sha256 = _canonical_json_sha256(campaign, "Pair manifest campaign")
    if campaign_sha256 != common["pair_campaign_sha256"]:
        raise EvidenceError("Pair manifest campaign differs from its acceptance pin")
    reward_and_advantage_sha256 = _canonical_json_sha256(
        campaign["reward_and_advantage"],
        "Pair manifest reward-and-advantage policy",
    )
    if reward_and_advantage_sha256 != common["pair_campaign_reward_and_advantage_sha256"]:
        raise EvidenceError("Pair manifest reward-and-advantage policy differs from its acceptance pin")
    if common["reward_semantics_contract_sha256"] != common["pair_campaign_reward_and_advantage_sha256"]:
        raise EvidenceError("reward-semantics contract must be the canonical Pair reward-and-advantage policy")
    if manifest.get("pair_campaign_sha256") != campaign_sha256:
        raise EvidenceError("Pair manifest campaign digest differs from canonical campaign bytes")
    if manifest.get("pair_campaign_reward_and_advantage_sha256") != reward_and_advantage_sha256:
        raise EvidenceError("Pair manifest reward-and-advantage digest differs from canonical policy bytes")

    boundary = _mapping(
        manifest.get("slurm_export_boundary"),
        "Pair manifest Slurm export boundary",
    )
    _exact_keys(
        boundary,
        {
            "schema",
            "format",
            "allowed_names",
            "ambient_merge",
            "get_user_env",
            "arms",
            "job_argv",
        },
        "Pair manifest Slurm export boundary",
    )
    if boundary["schema"] != SLURM_EXPORT_BOUNDARY_SCHEMA:
        raise EvidenceError("unexpected Pair manifest Slurm export boundary schema")
    if boundary["format"] != "nul-separated-name-value":
        raise EvidenceError("unexpected Pair manifest Slurm export boundary format")
    if boundary["ambient_merge"] is not False:
        raise EvidenceError("Pair manifest Slurm export boundary permits ambient merge")
    if boundary["get_user_env"] is not False:
        raise EvidenceError("Pair manifest Slurm export boundary permits get-user-env")

    allowed_names = boundary["allowed_names"]
    if not isinstance(allowed_names, list) or not allowed_names:
        raise EvidenceError("Pair manifest Slurm export allowed_names must be a non-empty list")
    if any(not isinstance(name, str) or ENVIRONMENT_NAME_RE.fullmatch(name) is None for name in allowed_names):
        raise EvidenceError("Pair manifest Slurm export allowed_names must contain names only")
    if allowed_names != sorted(allowed_names) or len(allowed_names) != len(set(allowed_names)):
        raise EvidenceError("Pair manifest Slurm export allowed_names must be sorted and unique")
    if allowed_names != list(SLURM_EXPORT_ALLOWED_NAMES):
        raise EvidenceError("Pair manifest Slurm export allowed_names differ from the canonical user payload")

    arm_records = _mapping(boundary["arms"], "Pair manifest Slurm export arms")
    _exact_keys(arm_records, {"off", "on"}, "Pair manifest Slurm export arms")
    paths: dict[str, str] = {}
    digests: dict[str, str] = {}
    for arm in ("off", "on"):
        record = _mapping(arm_records[arm], f"Pair manifest {arm} Slurm export record")
        _exact_keys(
            record,
            {"path", "sha256"},
            f"Pair manifest {arm} Slurm export record",
        )
        paths[arm] = _absolute_posix_path(record["path"], f"Pair manifest {arm} Slurm export path")
        digests[arm] = _digest(record["sha256"], f"Pair manifest {arm} Slurm export SHA-256")
        if digests[arm] != contract["provenance"]["arms"][arm]["runtime_environment_sha256"]:
            raise EvidenceError(f"Pair manifest {arm} Slurm export differs from arm runtime provenance")
    if paths["off"] == paths["on"]:
        raise EvidenceError("OFF/ON Slurm export files must have distinct paths")
    if digests["off"] == digests["on"]:
        raise EvidenceError("OFF/ON Slurm export files must have distinct SHA-256s")

    if boundary["job_argv"] != list(SLURM_EXPORT_JOB_ARGV):
        raise EvidenceError("Pair manifest Slurm job argv differs from the positional boundary")

    pair_paths = _mapping(manifest.get("paths"), "Pair manifest paths")
    results_root = _absolute_posix_path(pair_paths.get("results_root"), "Pair manifest results root")
    export_parent = PurePosixPath(results_root) / "strict_pair_slurm_exports" / manifest["pair_id"]
    for arm in ("off", "on"):
        expected_export_path = (export_parent / f"{arm}.env").as_posix()
        if paths[arm] != expected_export_path:
            raise EvidenceError(f"Pair manifest {arm} Slurm export path differs from the canonical arm path")
    _validate_pair_artifacts_and_paths(manifest, contract)
    _validate_pair_selection(manifest, contract)
    _validate_pair_runtime_tools(manifest)
    _validate_pair_container_entry_boundary(manifest)
    _validate_pair_runtime_attestation(manifest)
    _validate_pair_scheduler_submission(manifest, contract)
    _validate_pair_common_provenance_bindings(manifest, contract)
    _validate_pair_gym_source(manifest, contract)
    _validate_pair_source(manifest, contract)
    _validate_execution_environment(manifest.get("execution_environment"), manifest)
    return manifest


def _validate_holdout(document: Document, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract["holdout"]
    if document.sha256 != expected["receipt_sha256"]:
        raise EvidenceError("holdout receipt bytes differ from the pinned SHA-256")
    _require_canonical_document(document, "holdout receipt", trailing_lf=True)
    receipt = document.value
    _exact_keys(
        receipt,
        {
            "contract_sha256",
            "eligible",
            "environment",
            "frozen_reward_primary_mean_min",
            "frozen_reward_tail_mean_min",
            "holdout_observation_sha256",
            "effective_reward_wilson95_lower",
            "effective_success_rate",
            "effective_successes",
            "execution_authorized",
            "per_candidate",
            "schema",
            "selected_fixture_sha256",
            "selection_receipt_sha256",
            "trials",
        },
        "holdout receipt",
    )
    required = {
        "schema": "nemorl-single-env-reward-liveness-holdout-v1",
        "contract_sha256": contract["provenance"]["common"]["reward_liveness_contract_sha256"],
        "environment": contract["pair"]["environment"],
        "selected_fixture_sha256": contract["provenance"]["common"]["fixture_sha256"],
        "frozen_reward_primary_mean_min": expected["primary_reward_mean_min"],
        "frozen_reward_tail_mean_min": expected["tail_reward_mean_min"],
        "eligible": True,
    }
    _semantic_subset(receipt, required, "holdout receipt")
    _digest(receipt.get("selection_receipt_sha256"), "holdout selection receipt SHA-256")
    _digest(
        receipt.get("holdout_observation_sha256"),
        "holdout observation SHA-256",
    )
    _exact_json_value(receipt["execution_authorized"], False, "holdout execution authorization")
    effective_successes = _exact_nonnegative_integer(receipt["effective_successes"], "holdout effective successes")
    trials = _exact_nonnegative_integer(receipt["trials"], "holdout trials")
    if trials != 80 or effective_successes > trials:
        raise EvidenceError("holdout effective successes/trials are invalid")
    if type(receipt["effective_success_rate"]) is not float:
        raise EvidenceError("holdout effective success rate must be an exact JSON float")
    effective_success_rate = _number(receipt["effective_success_rate"], "holdout effective success rate")
    if not _close(
        effective_success_rate,
        effective_successes / trials,
        tolerance=RATIO_ABS_TOLERANCE,
    ):
        raise EvidenceError("holdout effective success rate does not close")
    if type(receipt["effective_reward_wilson95_lower"]) is not float:
        raise EvidenceError("holdout Wilson lower bound must be an exact JSON float")
    wilson_lower = _number(receipt["effective_reward_wilson95_lower"], "holdout Wilson lower bound")
    if not 0.0 <= wilson_lower <= 1.0:
        raise EvidenceError("holdout Wilson lower bound is outside [0, 1]")
    z = 1.959963984540054
    probability = effective_successes / trials
    denominator = 1.0 + z * z / trials
    center = probability + z * z / (2.0 * trials)
    radius = z * math.sqrt((probability * (1.0 - probability) + z * z / (4.0 * trials)) / trials)
    expected_wilson_lower = (center - radius) / denominator
    if not _close(wilson_lower, expected_wilson_lower, tolerance=RATIO_ABS_TOLERANCE):
        raise EvidenceError("holdout Wilson lower bound does not close")
    expected_floor = max(0.05, math.floor(0.5 * expected_wilson_lower * 100.0) / 100.0)
    for key in ("frozen_reward_primary_mean_min", "frozen_reward_tail_mean_min"):
        if type(receipt[key]) is not float:
            raise EvidenceError(f"holdout {key} must be an exact JSON float")
        if not _close(receipt[key], expected_floor, tolerance=RATIO_ABS_TOLERANCE):
            raise EvidenceError(f"holdout {key} differs from the frozen Wilson floor")

    per_candidate = receipt["per_candidate"]
    if not isinstance(per_candidate, list) or len(per_candidate) != 5:
        raise EvidenceError("holdout per_candidate must contain exactly five rows")
    candidate_digests: set[str] = set()
    candidate_effective_successes = 0
    candidate_trials = 0
    for index, item in enumerate(per_candidate):
        record = _mapping(item, f"holdout per_candidate[{index}]")
        _exact_keys(
            record,
            {
                "candidate_sha256",
                "effective_successes",
                "environment_successes",
                "mixed_blocks",
                "trials",
            },
            f"holdout per_candidate[{index}]",
        )
        candidate_digest = _digest(
            record["candidate_sha256"],
            f"holdout per_candidate[{index}].candidate_sha256",
        )
        if candidate_digest in candidate_digests:
            raise EvidenceError("holdout repeats one candidate SHA-256")
        candidate_digests.add(candidate_digest)
        record_trials = _exact_nonnegative_integer(record["trials"], f"holdout per_candidate[{index}].trials")
        environment_successes = _exact_nonnegative_integer(
            record["environment_successes"],
            f"holdout per_candidate[{index}].environment_successes",
        )
        record_effective_successes = _exact_nonnegative_integer(
            record["effective_successes"],
            f"holdout per_candidate[{index}].effective_successes",
        )
        if (
            record_trials != 16
            or not 1 <= environment_successes <= 15
            or not 1 <= record_effective_successes <= 15
            or record_effective_successes > environment_successes
        ):
            raise EvidenceError(f"holdout per_candidate[{index}] counts are invalid")
        mixed_blocks = record["mixed_blocks"]
        frozen_blocks = [
            "100,101,102,103",
            "104,105,106,107",
            "108,109,110,111",
            "112,113,114,115",
        ]
        if (
            not isinstance(mixed_blocks, list)
            or not mixed_blocks
            or mixed_blocks != [block for block in frozen_blocks if block in mixed_blocks]
        ):
            raise EvidenceError(f"holdout per_candidate[{index}] lacks exact K=4 mixed blocks")
        mixed_count = len(mixed_blocks)
        if not (
            mixed_count <= environment_successes <= record_trials - mixed_count
            and mixed_count <= record_effective_successes <= record_trials - mixed_count
        ):
            raise EvidenceError(f"holdout per_candidate[{index}] counts cannot realize its mixed blocks")
        candidate_effective_successes += record_effective_successes
        candidate_trials += record_trials
    candidate_order = [record["candidate_sha256"] for record in per_candidate]
    if candidate_order != sorted(candidate_order):
        raise EvidenceError("holdout candidates must be sorted by SHA-256")
    if candidate_effective_successes != effective_successes or candidate_trials != trials:
        raise EvidenceError("holdout per-candidate counts do not close")
    return receipt


def _step1_job_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > 19 or JOB_ID_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a canonical positive decimal string")
    if int(value) > (1 << 63) - 1:
        raise EvidenceError(f"{label} exceeds the bounded scheduler ID range")
    return value


def _resolved_slurm_export_boundary(
    pair_manifest: Mapping[str, Any], arm: str, pair_manifest_sha256: str
) -> dict[str, Any]:
    pair_boundary = _mapping(
        pair_manifest["slurm_export_boundary"],
        "Pair manifest Slurm export boundary",
    )
    arm_records = _mapping(pair_boundary["arms"], "Pair manifest Slurm export arms")
    arm_record = _mapping(arm_records[arm], f"Pair manifest {arm} Slurm export record")
    paths = _mapping(pair_manifest["paths"], "Pair manifest paths")
    pair_manifest_path = (PurePosixPath(paths["results_root"]) / "PAIR_MANIFEST.json").as_posix()
    return {
        "schema": SLURM_EXPORT_BOUNDARY_SCHEMA,
        "format": "nul-separated-name-value",
        "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "get_user_env": False,
        "arm": arm,
        "path": arm_record["path"],
        "sha256": arm_record["sha256"],
        "job_argv": [
            "--pair-manifest",
            pair_manifest_path,
            "--pair-manifest-sha256",
            pair_manifest_sha256,
            "--arm",
            arm,
        ],
    }


def _runtime_tool_receipt_pins(pair_manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime_tools = pair_manifest["runtime_tools"]
    document = runtime_tools["document"]
    host = document["host"]
    container = document["container"]
    return {
        "runtime_tool_manifest_path": runtime_tools["manifest"]["path"],
        "runtime_tool_manifest_sha256": runtime_tools["manifest"]["sha256"],
        "runtime_tool_host_python_path": host["python"]["path"],
        "runtime_tool_host_python_sha256": host["python"]["sha256"],
        "runtime_tool_container_python_path": container["python"]["path"],
        "runtime_tool_container_python_sha256": container["python"]["sha256"],
        "runtime_tool_container_uv_path": container["uv"]["path"],
        "runtime_tool_container_uv_sha256": container["uv"]["sha256"],
        "runtime_tool_uv_shim_path": container["uv_shim"]["path"],
        "runtime_tool_uv_shim_sha256": container["uv_shim"]["sha256"],
    }


def _bounded_ascii_device_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("ascii", "ignore")) <= 255:
        raise EvidenceError(f"{label} must be a 1..255-byte ASCII string")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise EvidenceError(f"{label} must be a 1..255-byte ASCII string") from error
    if len(raw) != len(value):
        raise EvidenceError(f"{label} must be a 1..255-byte ASCII string")
    return value


def _four_distinct_decimal_ids(value: Any, label: str) -> str:
    text = _bounded_ascii_device_value(value, label)
    if FOUR_DECIMAL_IDS_RE.fullmatch(text) is None:
        raise EvidenceError(f"{label} must contain exactly four decimal IDs")
    tokens = text.split(",")
    if any(str(int(token, 10)) != token for token in tokens):
        raise EvidenceError(f"{label} decimal IDs must use canonical spelling")
    if len({int(token, 10) for token in tokens}) != 4:
        raise EvidenceError(f"{label} decimal IDs must be numerically distinct")
    return text


def _validate_scheduler_device_environment(value: Any, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "schema",
            "cuda_visible_devices",
            "gpu_device_ordinal",
            "nvidia_visible_devices",
            "rocr_visible_devices",
            "ze_affinity_mask",
        },
        label,
    )
    _exact_json_value(record["schema"], SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA, f"{label}.schema")
    cuda = _four_distinct_decimal_ids(record["cuda_visible_devices"], f"{label}.cuda_visible_devices")
    ordinal = record["gpu_device_ordinal"]
    if ordinal is not None:
        _bounded_ascii_device_value(ordinal, f"{label}.gpu_device_ordinal")
        if ordinal != cuda:
            raise EvidenceError(f"{label}.gpu_device_ordinal must equal cuda_visible_devices")

    nvidia = record["nvidia_visible_devices"]
    if nvidia is not None:
        text = _bounded_ascii_device_value(nvidia, f"{label}.nvidia_visible_devices")
        if text not in {cuda, "all", "none", "void"}:
            tokens = text.split(",")
            if (
                len(tokens) != 4
                or len(set(tokens)) != 4
                or any(GPU_UUID_RE.fullmatch(token) is None for token in tokens)
            ):
                raise EvidenceError(f"{label}.nvidia_visible_devices has invalid GPU identities")

    rocr = record["rocr_visible_devices"]
    if rocr is not None:
        _four_distinct_decimal_ids(rocr, f"{label}.rocr_visible_devices")

    ze = record["ze_affinity_mask"]
    if ze is not None:
        text = _bounded_ascii_device_value(ze, f"{label}.ze_affinity_mask")
        tokens = text.split(",")
        if not 1 <= len(tokens) <= 64 or any(ZE_DEVICE_RE.fullmatch(token) is None for token in tokens):
            raise EvidenceError(f"{label}.ze_affinity_mask has invalid device tokens")
        if any(any(str(int(part, 10)) != part for part in token.split(".")) for token in tokens):
            raise EvidenceError(f"{label}.ze_affinity_mask device tokens must use canonical spelling")
        identities = {tuple(int(part, 10) for part in token.split(".")) for token in tokens}
        if len(identities) != len(tokens):
            raise EvidenceError(f"{label}.ze_affinity_mask device identities must be distinct")
    return record


def _validate_execution_receipts(
    contract: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
) -> dict[str, str]:
    pair = contract["pair"]
    receipt_groups = contract["receipts"]
    job_ids: dict[str, str] = {}
    pre_receipt_hashes: dict[str, str] = {}
    runtime_receipt_directories: dict[str, tuple[int, int]] = {}
    hardware_observations: dict[str, dict[str, Any]] = {}
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
        _require_canonical_document(
            _required_artifact(
                artifacts,
                f"{arm}_execution",
                f"{arm} shared-prefix execution receipt",
            ),
            f"{arm} shared-prefix execution receipt",
            trailing_lf=False,
        )
        _exact_keys(
            marker,
            {
                "arm",
                "environment",
                "marker_semantics",
                "pair_id",
                "schema",
                "scope",
                "shared_prefix_execution_marker_count",
                "shared_prefix_mode",
                "shared_prefix_runtime_trace_sha256",
                "status",
            },
            f"{arm} shared-prefix execution receipt",
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
                "shared_prefix_runtime_trace_sha256": contract["provenance"]["arms"][arm][
                    "shared_prefix_runtime_trace_sha256"
                ],
            },
            f"{arm} shared-prefix execution receipt",
        )
        count = _exact_nonnegative_integer(
            marker.get("shared_prefix_execution_marker_count"),
            f"{arm} shared-prefix execution marker count",
        )
        if arm == "off" and count != 0:
            raise EvidenceError("OFF has a shared-prefix physical execution marker")
        if arm == "on" and count <= 0:
            raise EvidenceError("ON lacks a shared-prefix physical execution marker")

        job_document = _required_artifact(artifacts, f"{arm}_job_exit", f"{arm} strict job EXIT receipt")
        job = _authenticate_receipt(
            job_document,
            receipt_groups["strict_job_exit_receipts"][arm],
            label=f"{arm} strict job EXIT receipt",
        )
        expected_job_bytes = _canonical_json_bytes(job, f"{arm} strict job EXIT receipt") + b"\n"
        if job_document.raw != expected_job_bytes:
            raise EvidenceError(f"{arm} strict job EXIT receipt must be canonical ASCII JSON plus one LF")
        _exact_keys(job, set(JOB_EXIT_KEYS), f"{arm} strict job EXIT receipt")
        _reject_secret_value_fields(job, f"{arm} strict job EXIT receipt")
        job_id = job.get("job_id")
        job_id = _step1_job_id(job_id, f"{arm} EXIT job_id")
        job_ids[arm] = job_id
        restart_count = _exact_nonnegative_integer(job.get("restart_count"), f"{arm} EXIT restart count")
        pair_manifest_sha256 = contract["provenance"]["common"]["pair_manifest_sha256"]
        slurm_export_boundary = _resolved_slurm_export_boundary(pair_manifest, arm, pair_manifest_sha256)
        slurm_export_boundary_sha256 = _canonical_json_sha256(
            slurm_export_boundary, f"{arm} resolved Slurm export boundary"
        )
        runtime_tool_pins = _runtime_tool_receipt_pins(pair_manifest)
        expected_hardware = {
            "schema": HARDWARE_OBSERVATION_SCHEMA,
            "gpu_model": contract["provenance"]["topology"]["gpu_model"],
            "driver_version": contract["provenance"]["topology"]["nvidia_driver_version"],
            "nvidia_smi": pair_manifest["runtime_tools"]["document"]["host"]["nvidia_smi"],
        }
        container_entry_boundary = pair_manifest["container_entry_boundary"]
        container_entry_boundary_sha256 = _canonical_json_sha256(container_entry_boundary, "container-entry boundary")
        gym_source = pair_manifest["source"]["gym"]
        attestation_line = pair_manifest["runtime_attestation"]["lines"][arm]
        runtime_attestation_receipt_dir = (
            PurePosixPath(pair_manifest["paths"]["results_root"])
            / arm
            / pair_manifest["determinism_receipt_dir"]
            / f"{job_id}-{restart_count}"
        ).as_posix()
        _semantic_subset(
            job,
            {
                "schema": JOB_RECEIPT_SCHEMA,
                "phase": "EXIT",
                "post_verified": True,
                "driver_exit_code": 0,
                "pair_id": pair["pair_id"],
                "environment": pair["environment"],
                "arm": arm,
                "job_account": contract["configs"][arm]["slurm_account"],
                "job_name": pair_manifest["scheduler_submission"]["identity"]["job_names"][arm],
                "job_num_nodes": contract["provenance"]["topology"]["allocated_nodes"],
                "job_partition": contract["configs"][arm]["slurm_partition"],
                "job_qos": pair_manifest["campaign"]["slurm"]["qos"],
                "gpus_per_node": contract["provenance"]["topology"]["gpus_per_node"],
                "restart_count": restart_count,
                "deterministic_controls": DETERMINISTIC_CONTROLS,
                "runtime_attestation_expected_count": 4,
                "runtime_attestation_actual_count": 4,
                "runtime_attestation_marker_sha256": attestation_line["sha256_ascii_no_newline"],
                "runtime_attestation_receipt_dir": runtime_attestation_receipt_dir,
                "pair_manifest_sha256": pair_manifest_sha256,
                "fixture_sha256": contract["provenance"]["common"]["fixture_sha256"],
                "fixture_rows": 5,
                "model_tree_sha256_v1": contract["provenance"]["common"]["model_tree_sha256"],
                "container_sha256": contract["provenance"]["common"]["training_container_sha256"],
                "sandbox_container_sha256": contract["provenance"]["common"]["sandbox_container_sha256"],
                "source_head": contract["provenance"]["source_commits"]["nemo_rl"],
                "source_tree": contract["provenance"]["source_git_trees"]["nemo_rl"],
                "config_sha256": contract["provenance"]["common"]["environment_recipe_sha256"],
                "reward_semantics_config_sha256": contract["provenance"]["common"]["environment_recipe_sha256"],
                "selected_config_sha256": contract["provenance"]["common"]["environment_recipe_sha256"],
                "reward_semantics_contract_sha256": contract["provenance"]["common"][
                    "reward_semantics_contract_sha256"
                ],
                "nemo_runnable_manifest_sha256": contract["provenance"]["common"]["nemo_runnable_manifest_sha256"],
                "bridge_runnable_manifest_sha256": contract["provenance"]["common"]["bridge_runnable_manifest_sha256"],
                "mcore_runnable_manifest_sha256": contract["provenance"]["common"]["mcore_runnable_manifest_sha256"],
                "deployment_ready": contract["provenance"]["common"]["deployment_ready_sha256"],
                "deployment_ready_sha256": contract["provenance"]["common"]["deployment_ready_sha256"],
                "deployment_ready_file_sha256": contract["provenance"]["common"]["deployment_ready_file_sha256"],
                "pair_campaign_sha256": contract["provenance"]["common"]["pair_campaign_sha256"],
                "pair_campaign_reward_and_advantage_sha256": contract["provenance"]["common"][
                    "pair_campaign_reward_and_advantage_sha256"
                ],
                "submission_contract_sha256": contract["provenance"]["common"]["submission_contract_sha256"],
                "submission_contract_path": pair_manifest["scheduler_submission"]["contract"]["path"],
                "submission_nonce": pair_manifest["scheduler_submission"]["nonce"],
                "submission_receipt_path": pair_manifest["scheduler_submission"]["receipt"]["path"],
                "submission_receipt_sha256": _required_artifact(
                    artifacts, "submission_receipt", "pair submission receipt"
                ).sha256,
                "strict_pair_arm_wrapper_sha256": contract["provenance"]["common"]["strict_pair_arm_wrapper_sha256"],
                "strict_pair_contract_sha256": contract["provenance"]["common"]["strict_pair_contract_sha256"],
                "strict_pair_parent_wrapper_sha256": contract["provenance"]["common"][
                    "strict_pair_parent_wrapper_sha256"
                ],
                "snapshot_manifest_sha256": contract["provenance"]["arms"][arm]["snapshot_manifest_sha256"],
                "entrypoint_sha256": contract["provenance"]["arms"][arm]["entrypoint_sha256"],
                "wrapper_sha256": contract["provenance"]["arms"][arm]["wrapper_sha256"],
                "inner_ray_sha256": contract["provenance"]["arms"][arm]["inner_ray_sha256"],
                "command_sha256": contract["provenance"]["arms"][arm]["command_sha256"],
                "mounts_sha256": contract["provenance"]["arms"][arm]["mounts_sha256"],
                "container_entry_boundary": container_entry_boundary,
                "container_entry_boundary_sha256": container_entry_boundary_sha256,
                "gym_gitlink_commit": gym_source["gitlink_commit"],
                "gym_tree": gym_source["tree"],
                "hardware": expected_hardware,
                "selection": pair_manifest["selection"],
                "execution_environment": pair_manifest["execution_environment"],
                "source": pair_manifest["source"],
                "wandb": {
                    "entity": pair_manifest["wandb"]["entity"],
                    "project": pair_manifest["wandb"]["project"],
                    "group": pair_manifest["wandb"]["group"]["value"],
                    "name": pair_manifest["wandb"]["arms"][arm]["name"],
                    "name_template": pair_manifest["wandb"]["arms"][arm]["name_template"],
                    "run_id": pair_manifest["wandb"]["arms"][arm]["run_id"],
                    "run_id_derivation": pair_manifest["wandb"]["run_id_derivation"],
                    "resume": pair_manifest["wandb"]["resume"],
                },
                "scheduler_client_environment": {
                    "ambient_merge": False,
                    "SLURM_CONF": HSG_SLURM_CONF,
                    "propagated_to_inner_ray": True,
                },
                "slurm_export_boundary": slurm_export_boundary,
                "slurm_export_boundary_sha256": slurm_export_boundary_sha256,
                **runtime_tool_pins,
            },
            f"{arm} strict job EXIT receipt",
        )
        _exact_json_value(
            job["deterministic_controls"],
            DETERMINISTIC_CONTROLS,
            f"{arm} EXIT deterministic controls",
        )
        _exact_json_value(
            job["selection"],
            pair_manifest["selection"],
            f"{arm} EXIT selection",
        )
        _exact_json_value(
            job["execution_environment"],
            pair_manifest["execution_environment"],
            f"{arm} EXIT execution environment",
        )
        _exact_json_value(job["source"], pair_manifest["source"], f"{arm} EXIT source")
        step1_evidence = _mapping(job["step1_evidence"], f"{arm} EXIT step-1 evidence")
        _exact_keys(
            step1_evidence,
            {"main_ledger", "model_transport", "schema", "transcript_bundle"},
            f"{arm} EXIT step-1 evidence",
        )
        _exact_json_value(
            step1_evidence["schema"],
            STEP1_EVIDENCE_INDEX_SCHEMA,
            f"{arm} EXIT step-1 evidence schema",
        )
        transport_index = _mapping(
            step1_evidence["model_transport"],
            f"{arm} EXIT model transport evidence",
        )
        _exact_keys(
            transport_index,
            {"bundle", "manifest", "ordered_entries_sha256", "raw_log", "schema"},
            f"{arm} EXIT model transport evidence",
        )
        _exact_json_value(
            transport_index["schema"],
            MODEL_TRANSPORT_EVIDENCE_INDEX_SCHEMA,
            f"{arm} EXIT model transport evidence schema",
        )
        transport_root = (
            PurePosixPath(pair_manifest["execution_environment"]["arms"][arm]["results_dir"]) / "strict_model_transport"
        )
        for ref_name, ref_schema, filename in (
            ("bundle", MODEL_TRANSPORT_BUNDLE_SCHEMA, "model-transport-bundle.json"),
            (
                "manifest",
                MODEL_TRANSPORT_MANIFEST_SCHEMA,
                "model-transport-manifest.json",
            ),
        ):
            ref = _mapping(
                transport_index[ref_name],
                f"{arm} EXIT model transport {ref_name}",
            )
            _exact_json_value(
                ref,
                {
                    "path": (transport_root / filename).as_posix(),
                    "schema": ref_schema,
                    "sha256": _digest(
                        ref.get("sha256"),
                        f"{arm} EXIT model transport {ref_name} SHA-256",
                    ),
                },
                f"{arm} EXIT model transport {ref_name}",
            )
        raw_log = _mapping(transport_index["raw_log"], f"{arm} EXIT model transport raw log")
        _exact_json_value(
            raw_log,
            {
                "path": (transport_root / "model-transport.jsonl").as_posix(),
                "record_schema": MODEL_TRANSPORT_CALL_SCHEMA,
                "record_count": K4_SAMPLES,
                "sha256": _digest(
                    raw_log.get("sha256"),
                    f"{arm} EXIT model transport raw-log SHA-256",
                ),
            },
            f"{arm} EXIT model transport raw log",
        )
        _digest(
            transport_index["ordered_entries_sha256"],
            f"{arm} EXIT model transport ordered entries SHA-256",
        )
        main_ledger = _mapping(step1_evidence["main_ledger"], f"{arm} EXIT main step-1 ledger")
        _exact_keys(
            main_ledger,
            {"path", "schema", "sha256"},
            f"{arm} EXIT main step-1 ledger",
        )
        expected_ledger_path = (
            PurePosixPath(pair_manifest["execution_environment"]["arms"][arm]["results_dir"])
            / "strict_pair_step1_evidence/main-ledger.json"
        ).as_posix()
        _exact_json_value(main_ledger["path"], expected_ledger_path, f"{arm} EXIT ledger path")
        _exact_json_value(
            main_ledger["schema"],
            MAIN_STEP1_LEDGER_SCHEMA,
            f"{arm} EXIT ledger schema",
        )
        _digest(main_ledger["sha256"], f"{arm} EXIT ledger SHA-256")
        transcript_bundle = _mapping(
            step1_evidence["transcript_bundle"],
            f"{arm} EXIT main step-1 transcript bundle",
        )
        _exact_keys(
            transcript_bundle,
            {"path", "schema", "sha256"},
            f"{arm} EXIT main step-1 transcript bundle",
        )
        expected_transcript_path = (
            PurePosixPath(pair_manifest["execution_environment"]["arms"][arm]["results_dir"])
            / "strict_pair_step1_evidence/transcript-bundle.json"
        ).as_posix()
        _exact_json_value(
            transcript_bundle["path"],
            expected_transcript_path,
            f"{arm} EXIT transcript-bundle path",
        )
        _exact_json_value(
            transcript_bundle["schema"],
            STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
            f"{arm} EXIT transcript-bundle schema",
        )
        _digest(
            transcript_bundle["sha256"],
            f"{arm} EXIT transcript-bundle SHA-256",
        )
        _exact_json_value(job["hardware"], expected_hardware, f"{arm} EXIT hardware observation")
        hardware_observations[arm] = job["hardware"]
        _exact_json_value(
            job["wandb"],
            {
                "entity": pair_manifest["wandb"]["entity"],
                "project": pair_manifest["wandb"]["project"],
                "group": pair_manifest["wandb"]["group"]["value"],
                "name": pair_manifest["wandb"]["arms"][arm]["name"],
                "name_template": pair_manifest["wandb"]["arms"][arm]["name_template"],
                "run_id": pair_manifest["wandb"]["arms"][arm]["run_id"],
                "run_id_derivation": pair_manifest["wandb"]["run_id_derivation"],
                "resume": pair_manifest["wandb"]["resume"],
            },
            f"{arm} EXIT W&B identity",
        )
        expected_scheduler_client_environment = {
            "ambient_merge": False,
            "SLURM_CONF": HSG_SLURM_CONF,
            "propagated_to_inner_ray": True,
        }
        _exact_json_value(
            job["scheduler_client_environment"],
            expected_scheduler_client_environment,
            f"{arm} EXIT scheduler client environment",
        )
        _validate_scheduler_device_environment(
            job["scheduler_device_environment"],
            f"{arm} EXIT scheduler device environment",
        )
        for key in (
            "runtime_attestation_receipt_dir_device",
            "runtime_attestation_receipt_dir_inode",
        ):
            if _exact_nonnegative_integer(job[key], f"{arm} EXIT {key}") == 0:
                raise EvidenceError(f"{arm} EXIT {key} must be positive")
        runtime_receipt_directories[arm] = (
            job["runtime_attestation_receipt_dir_device"],
            job["runtime_attestation_receipt_dir_inode"],
        )
        if job["container_entry_boundary"] != container_entry_boundary:
            raise EvidenceError(f"{arm} EXIT container-entry boundary differs from the Pair manifest")
        resolved_boundary = _mapping(job["slurm_export_boundary"], f"{arm} EXIT Slurm export boundary")
        _exact_keys(
            resolved_boundary,
            {
                "schema",
                "format",
                "allowed_names",
                "ambient_merge",
                "get_user_env",
                "arm",
                "path",
                "sha256",
                "job_argv",
            },
            f"{arm} EXIT Slurm export boundary",
        )
        if resolved_boundary != slurm_export_boundary:
            raise EvidenceError(f"{arm} EXIT Slurm export boundary differs from its resolved Pair arm")
        if job["slurm_export_boundary_sha256"] != slurm_export_boundary_sha256:
            raise EvidenceError(f"{arm} EXIT Slurm export boundary digest differs from canonical JSON")
        pre_receipt_sha256 = _digest(job.get("pre_receipt_sha256"), f"{arm} PRE receipt hash")
        pre_receipt_hashes[arm] = pre_receipt_sha256
        if (
            pre_receipt_sha256
            == _required_artifact(artifacts, f"{arm}_job_exit", f"{arm} strict job EXIT receipt").sha256
        ):
            raise EvidenceError(f"{arm} PRE and EXIT receipt hashes alias")
        hashes = _mapping(
            job.get("runtime_attestation_receipts_sha256"),
            f"{arm} runtime attestation receipt hashes",
        )
        expected_names = {f"shared_prefix_determinism.{mode}.rank-{rank}.receipt" for rank in range(4)}
        _exact_keys(hashes, expected_names, f"{arm} runtime attestation receipts")
        expected_marker_sha256 = attestation_line["sha256_ascii_no_newline"]
        for name, digest in hashes.items():
            if _digest(digest, f"{arm} runtime receipt {name}") != expected_marker_sha256:
                raise EvidenceError(f"{arm} runtime receipt {name} differs from the exact Pair marker")
        aggregate_sha256 = _digest(
            job.get("runtime_attestation_aggregate_sha256"),
            f"{arm} runtime attestation aggregate",
        )
        expected_aggregate_sha256 = _runtime_attestation_aggregate_sha256(
            sorted(expected_names), attestation_line["text"]
        )
        if aggregate_sha256 != expected_aggregate_sha256:
            raise EvidenceError(f"{arm} runtime attestation aggregate differs from its sorted receipt bytes")
    if job_ids["off"] == job_ids["on"]:
        raise EvidenceError("OFF/ON EXIT receipts bind the same scheduler job ID")
    if pre_receipt_hashes["off"] == pre_receipt_hashes["on"]:
        raise EvidenceError("OFF/ON PRE receipt hashes must differ")
    if runtime_receipt_directories["off"] == runtime_receipt_directories["on"]:
        raise EvidenceError("OFF/ON runtime receipt directory identities must differ")
    if hardware_observations["off"] != hardware_observations["on"]:
        raise EvidenceError("OFF/ON hardware observations must be exactly equal")
    return job_ids


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


def _validate_token_geometry(gate: Gate, run: Run, *, steps: Sequence[int] = STEPS) -> None:
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
        ideal_work = history.integer(step, "train/shared_prefix/ideal_shared_token_work")
        exact_counts = {
            "train/rollout/samples": samples,
            "train/num_valid_samples": valid_samples,
            "train/global_valid_seqs": valid_sequences,
            "train/shared_prefix/total_sequences": history.integer(step, "train/shared_prefix/total_sequences"),
            "train/shared_prefix/eligible_sequences": history.integer(step, "train/shared_prefix/eligible_sequences"),
        }
        if any(value != K4_SAMPLES for value in exact_counts.values()):
            gate.fail(f"{run.arm} step {step} does not preserve the exact K=4 cohort: " f"{exact_counts}")
        if history.integer(step, "train/shared_prefix/complete_groups") != 1:
            gate.fail(f"{run.arm} step {step} lacks one complete shared-prefix group")
        if (
            history.integer(step, "train/shared_prefix/fallback_sequences") != 0
            or history.integer(step, "train/shared_prefix/runtime_fallback_sequences") != 0
        ):
            gate.fail(f"{run.arm} step {step} used shared-prefix fallback")
        if not 0 < valid_tokens <= total_tokens:
            gate.fail(f"{run.arm} step {step} global_valid_toks is outside (0,total]")
        if total_tokens != shared_total or valid_tokens != shared_valid:
            gate.fail(f"{run.arm} step {step} global/shared-prefix token sources disagree")
        if prompt + valid_tokens + suffix != total_tokens:
            gate.fail(f"{run.arm} step {step} prompt/valid/suffix tokens do not close")
        if not 0 < shareable <= prompt or ideal_work != total_tokens - shareable:
            gate.fail(f"{run.arm} step {step} ideal shared-token work does not close")
        reduction = history.number(step, "train/shared_prefix/ideal_token_reduction")
        speedup = history.number(step, "train/shared_prefix/ideal_token_work_speedup")
        if total_tokens <= 0:
            gate.fail(f"{run.arm} step {step} total token denominator is not positive")
        elif not _close(reduction, shareable / total_tokens, tolerance=RATIO_ABS_TOLERANCE):
            gate.fail(f"{run.arm} step {step} ideal token reduction does not close")
        if ideal_work <= 0:
            gate.fail(f"{run.arm} step {step} ideal token-work denominator is not positive")
        elif not _close(speedup, total_tokens / ideal_work, tolerance=RATIO_ABS_TOLERANCE):
            gate.fail(f"{run.arm} step {step} ideal token speedup does not close")


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
        failed_update_steps = [step for step in STEPS if not history.boolean(step, "train/update_successful")]
        if failed_update_steps:
            gate.fail(
                f"{run.arm} optimizer update was not successful at steps " f"{_compact_steps(failed_update_steps)}"
            )
        negative_policy_norm_steps = [step for step in STEPS if history.number(step, "train/grad_norm") < 0.0]
        negative_mtp_norm_steps = [step for step in STEPS if history.number(step, "train/mtp/grad_norm") < 0.0]
        invalid_advantage_steps = []
        for step in STEPS:
            advantage_min = history.number(step, "train/advantages/min")
            advantage_mean = history.number(step, "train/advantages/mean")
            advantage_max = history.number(step, "train/advantages/max")
            if not advantage_min <= advantage_mean <= advantage_max:
                invalid_advantage_steps.append(step)
        if negative_policy_norm_steps:
            gate.fail(
                f"{run.arm} policy grad_norm is negative at steps " f"{_compact_steps(negative_policy_norm_steps)}"
            )
        if negative_mtp_norm_steps:
            gate.fail(f"{run.arm} MTP grad_norm is negative at steps " f"{_compact_steps(negative_mtp_norm_steps)}")
        if invalid_advantage_steps:
            gate.fail(
                f"{run.arm} advantage min/mean/max ordering is invalid at steps "
                f"{_compact_steps(invalid_advantage_steps)}"
            )
        raw_means = [history.number(step, "train/raw_environment_reward") for step in STEPS]
        effective_means = [history.number(step, "train/total_reward/mean") for step in STEPS]
        mixed_steps = [
            step
            for step in STEPS
            if history.number(step, "train/raw_environment_reward/min")
            < history.number(step, "train/raw_environment_reward/max")
            and history.number(step, "train/total_reward/min") < history.number(step, "train/total_reward/max")
            and history.number(step, "train/advantages/min") < 0.0
            and history.number(step, "train/advantages/max") > 0.0
        ]
        active_epochs = _epoch_numbers(mixed_steps)
        tail_active = [epoch for epoch in active_epochs if epoch in TAIL_EPOCHS]
        policy_epochs = _epoch_numbers([step for step in STEPS if history.number(step, "train/grad_norm") > 0.0])
        mtp_epochs = _epoch_numbers([step for step in STEPS if history.number(step, "train/mtp/grad_norm") > 0.0])
        joint_tail = sorted(TAIL_EPOCHS & set(policy_epochs) & set(mtp_epochs))
        raw_distinct = len(set(raw_means))
        effective_distinct = len(set(effective_means))
        unique_means = len(set(effective_means))
        try:
            reward_stddev = _canonical_derived_float(
                statistics.pstdev(effective_means),
                f"{run.arm} effective reward population standard deviation",
            )
            primary_mean = _canonical_derived_float(
                statistics.fmean(history.number(step, "train/total_reward/mean") for step in PRIMARY_STEPS),
                f"{run.arm} primary effective reward mean",
            )
            tail_mean = _canonical_derived_float(
                statistics.fmean(history.number(step, "train/total_reward/mean") for step in TAIL_STEPS),
                f"{run.arm} tail effective reward mean",
            )
        except (OverflowError, ValueError) as error:
            gate.unverifiable(f"{run.arm} learning statistics cannot be computed: {error}")
            gate.evidence = {"arms": arm_evidence}
            return gate
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
        penalty_max = max(history.number(step, metric) for step in STEPS for metric in PENALTY_METRICS)
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
            "optimizer_update_successful_steps": [
                step for step in STEPS if history.boolean(step, "train/update_successful")
            ],
        }
    reward_policy = LIVE_LEARNING_ACCEPTANCE_POLICY["live_reward_noninferiority"]
    paired_differences = [
        runs["on"].history.number(step, reward_policy["metric"])
        - runs["off"].history.number(step, reward_policy["metric"])
        for step in PRIMARY_STEPS
    ]
    try:
        raw_reward_ci_low, raw_reward_ci_high = _bootstrap_paired_mean_difference_ci(paired_differences)
        reward_ci_low = _canonical_derived_float(raw_reward_ci_low, "live reward confidence lower bound")
        reward_ci_high = _canonical_derived_float(raw_reward_ci_high, "live reward confidence upper bound")
        off_tail_mean = _canonical_derived_float(
            statistics.fmean(runs["off"].history.number(step, reward_policy["metric"]) for step in TAIL_STEPS),
            "OFF live reward tail mean",
        )
        on_tail_mean = _canonical_derived_float(
            statistics.fmean(runs["on"].history.number(step, reward_policy["metric"]) for step in TAIL_STEPS),
            "ON live reward tail mean",
        )
        paired_mean = _canonical_derived_float(
            statistics.fmean(paired_differences),
            "paired live reward mean difference",
        )
    except (EvidenceError, OverflowError, ValueError) as error:
        gate.unverifiable(f"live reward noninferiority cannot be computed: {error}")
        gate.evidence = {"arms": arm_evidence}
        return gate
    margin = reward_policy["margin"]
    if not reward_ci_low > -margin:
        gate.fail(
            "paired live-reward bootstrap lower confidence bound "
            f"{reward_ci_low} is not strictly greater than {-margin}"
        )
    tail_margin = reward_policy["tail"]["margin"]
    if not on_tail_mean >= off_tail_mean - tail_margin:
        gate.fail(
            f"ON tail raw-reward mean {on_tail_mean} is below OFF tail mean "
            f"{off_tail_mean} minus margin {tail_margin}"
        )
    gate.evidence = {
        "arms": arm_evidence,
        "live_reward_noninferiority": {
            "claim_scope": "reward-noninferiority-only-not-trajectory-equivalence",
            "metric": reward_policy["metric"],
            "paired_steps": list(PRIMARY_STEPS),
            "paired_mean_on_minus_off": paired_mean,
            "bootstrap_95_ci": {
                "low": reward_ci_low,
                "high": reward_ci_high,
                "resamples": reward_policy["paired_step_bootstrap"]["resamples"],
                "seed": reward_policy["paired_step_bootstrap"]["seed"],
                "resampling_unit": "paired-step",
                "statistic": "mean-on-minus-off",
            },
            "margin": margin,
            "off_tail_mean": off_tail_mean,
            "on_tail_mean": on_tail_mean,
            "tail_margin": tail_margin,
        },
        "optimizer_update_witness": {
            "metric": "train/update_successful",
            "required_value": True,
            "steps": list(STEPS),
        },
    }
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


def _bootstrap_paired_mean_difference_ci(
    differences: Sequence[float],
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return the A1 percentile CI over complete paired reward steps."""
    policy = LIVE_LEARNING_ACCEPTANCE_POLICY["live_reward_noninferiority"]
    bootstrap = policy["paired_step_bootstrap"]
    if samples != bootstrap["resamples"]:
        raise EvidenceError("live-reward bootstrap resamples differ from A1")
    if len(differences) != policy["evaluated_steps"]["count"]:
        raise EvidenceError("live-reward bootstrap requires all 90 paired steps")
    values = [_number(value, "paired live-reward difference") for value in differences]
    generator = random.Random(bootstrap["seed"])
    size = len(values)
    replicates = [statistics.fmean(values[generator.randrange(size)] for _ in range(size)) for _ in range(samples)]
    return (
        _percentile(replicates, bootstrap["lower_quantile"]),
        _percentile(replicates, bootstrap["upper_quantile"]),
    )


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
    return all(off.values[step][metric] == on.values[step][metric] for metric in WORK_METRICS)


def _evaluate_speed(
    contract: Mapping[str, Any],
    runs: Mapping[str, Run],
    artifacts: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    submitted_job_ids: Mapping[str, str],
) -> Gate:
    gate = Gate("speed_evidence")
    if not _require_histories(gate, runs, SPEED_METRICS, steps=PRIMARY_STEPS):
        return gate
    try:
        exit_job_ids = _validate_execution_receipts(contract, artifacts, pair_manifest)
        _exact_json_value(
            exit_job_ids,
            dict(submitted_job_ids),
            "EXIT/submission authenticated scheduler job IDs",
        )
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
                gate.unverifiable(f"{run.arm} step {step} policy-training time is not positive")
    if gate.unavailable:
        return gate

    matched_epochs = [
        epoch
        for epoch in PRIMARY_EPOCHS
        if all(_same_work(runs["off"].history, runs["on"].history, step) for step in epoch)
    ]
    excluded_epochs = [epoch for epoch in PRIMARY_EPOCHS if epoch not in matched_epochs]
    if len(matched_epochs) < MIN_MATCHED_PRIMARY_EPOCHS:
        gate.fail(f"complete matched primary epochs={len(matched_epochs)}; requires >=15/18")
    ratios_by_step: dict[int, float] = {}
    for epoch in matched_epochs:
        for step in epoch:
            try:
                ratio = runs["off"].history.number(step, "timing/train/policy_training") / runs["on"].history.number(
                    step, "timing/train/policy_training"
                )
            except (OverflowError, ZeroDivisionError) as error:
                gate.unverifiable(f"step {step} speed ratio cannot be computed: {error}")
                return gate
            if not math.isfinite(ratio) or ratio <= 0.0:
                gate.unverifiable(f"step {step} speed ratio must be finite and positive, got {ratio!r}")
                return gate
            ratios_by_step[step] = ratio
    if not ratios_by_step:
        gate.unverifiable("no complete matched primary epoch remains for speed")
        return gate
    blocks = [[ratios_by_step[step] for step in epoch] for epoch in matched_epochs]
    try:
        raw_low, raw_high = _bootstrap_epoch_median_ci(blocks)
        ratio_values = list(ratios_by_step.values())
        low = _canonical_derived_float(raw_low, "speed confidence lower bound")
        high = _canonical_derived_float(raw_high, "speed confidence upper bound")
        median = _canonical_derived_float(float(statistics.median(ratio_values)), "speed ratio median")
        geomean = _canonical_derived_float(
            math.exp(statistics.fmean(math.log(value) for value in ratio_values)),
            "speed ratio geometric mean",
        )
    except (EvidenceError, OverflowError, ValueError, ZeroDivisionError) as error:
        gate.unverifiable(str(error))
        return gate
    derived = {
        "bootstrap low": low,
        "bootstrap high": high,
        "median": median,
        "geomean": geomean,
    }
    for name, value in derived.items():
        if not math.isfinite(value) or value <= 0.0:
            gate.unverifiable(f"derived {name} speed statistic must be finite and positive, got {value!r}")
    if gate.unavailable:
        return gate
    if not low > 1.0:
        gate.fail(f"bootstrap 95% speed CI lower={low} is not strictly >1.0")
    gate.evidence = {
        "primary_window_steps": [11, 100],
        "matched_epoch_numbers": [(epoch[0] - 1) // STEPS_PER_EPOCH + 1 for epoch in matched_epochs],
        "excluded_epoch_numbers": [(epoch[0] - 1) // STEPS_PER_EPOCH + 1 for epoch in excluded_epochs],
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


def _unverifiable_report(contract: Mapping[str, Any] | None, message: str) -> dict[str, Any]:
    pair_value = contract.get("pair", {}) if isinstance(contract, Mapping) else {}
    pair = pair_value if isinstance(pair_value, Mapping) else {}
    pair_id = pair.get("pair_id")
    if type(pair_id) is not str or PAIR_ID_RE.fullmatch(pair_id) is None:
        pair_id = None
    environment = pair.get("environment")
    if type(environment) is not str or environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
        environment = None
    gate = {
        "status": "UNVERIFIABLE",
        "failures": [],
        "unavailable": [message],
        "evidence": {},
    }
    return {
        "schema": REPORT_SCHEMA,
        "acceptance_scope": ACCEPTANCE_SCOPE,
        "threat_model": copy.deepcopy(THREAT_MODEL),
        "pair_id": pair_id,
        "environment": environment,
        "live_reward_consistency": dict(gate),
        "learning_behavior": dict(gate),
        "speed_evidence": dict(gate),
        "overall": {
            "status": "UNVERIFIABLE",
            "independent_statuses": {
                "live_reward_consistency": "UNVERIFIABLE",
                "learning_behavior": "UNVERIFIABLE",
                "speed_evidence": "UNVERIFIABLE",
            },
        },
    }


def _validate_exact_json_tree(value: Any, label: str) -> None:
    """Reject Python aliases that do not have an exact portable JSON type."""
    if value is None or type(value) in {bool, str}:
        return
    if type(value) is int:
        if not -(1 << 63) < value < (1 << 63):
            raise EvidenceError(f"{label} integer exceeds the bounded JSON range")
        return
    if type(value) is float:
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0.0):
            raise EvidenceError(f"{label} must be a finite non-negative-zero float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_exact_json_tree(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvidenceError(f"{label} contains a non-string JSON key")
            _validate_exact_json_tree(item, f"{label}.{key}")
        return
    raise EvidenceError(f"{label} contains a nonexact JSON value")


def _validate_acceptance_report(value: Any) -> dict[str, Any]:
    """Close the exact live-only report root before it is emitted."""
    report = _mapping(value, "acceptance report")
    _exact_keys(
        report,
        {
            "schema",
            "acceptance_scope",
            "threat_model",
            "pair_id",
            "environment",
            "live_reward_consistency",
            "learning_behavior",
            "speed_evidence",
            "overall",
        },
        "acceptance report",
    )
    _exact_json_value(report["schema"], REPORT_SCHEMA, "acceptance report schema")
    _exact_json_value(
        report["acceptance_scope"],
        ACCEPTANCE_SCOPE,
        "acceptance report scope",
    )
    _exact_json_value(report["threat_model"], THREAT_MODEL, "acceptance report threat model")
    if report["pair_id"] is not None:
        _pair_id(report["pair_id"], "acceptance report pair_id")
    if report["environment"] is not None:
        environment = _safe_id(report["environment"], "acceptance report environment")
        if environment not in VERIFIER_METRIC_BY_ENVIRONMENT:
            raise EvidenceError("acceptance report environment is unsupported")

    statuses: dict[str, str] = {}
    for name in ("live_reward_consistency", "learning_behavior", "speed_evidence"):
        gate = _mapping(report[name], f"acceptance report {name}")
        _exact_keys(
            gate,
            {"status", "failures", "unavailable", "evidence"},
            f"acceptance report {name}",
        )
        failures = gate["failures"]
        unavailable = gate["unavailable"]
        if not isinstance(failures, list) or any(type(item) is not str for item in failures):
            raise EvidenceError(f"acceptance report {name} failures must be strings")
        if not isinstance(unavailable, list) or any(type(item) is not str for item in unavailable):
            raise EvidenceError(f"acceptance report {name} unavailable entries must be strings")
        _mapping(gate["evidence"], f"acceptance report {name} evidence")
        expected_status = "UNVERIFIABLE" if unavailable else "FAIL" if failures else "PASS"
        _exact_json_value(gate["status"], expected_status, f"acceptance report {name} status")
        statuses[name] = expected_status

    overall = _mapping(report["overall"], "acceptance report overall")
    _exact_keys(overall, {"status", "independent_statuses"}, "acceptance report overall")
    _exact_json_value(
        overall["independent_statuses"],
        statuses,
        "acceptance report independent statuses",
    )
    expected_overall = (
        "UNVERIFIABLE" if "UNVERIFIABLE" in statuses.values() else "RED" if "FAIL" in statuses.values() else "GREEN"
    )
    _exact_json_value(overall["status"], expected_overall, "acceptance report overall status")
    _validate_exact_json_tree(report, "acceptance report")
    _canonical_json_bytes(report, "acceptance report")
    return dict(report)


def _text_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"overall={report['overall']['status']}",
        f"acceptance_scope={report['acceptance_scope']}",
        f"pair_id={report.get('pair_id')}",
        f"environment={report.get('environment')}",
    ]
    for name in ("live_reward_consistency", "learning_behavior", "speed_evidence"):
        gate = report[name]
        lines.append(f"{name}={gate['status']}")
        for message in gate["unavailable"]:
            lines.append(f"  unavailable: {message}")
        for message in gate["failures"]:
            lines.append(f"  failure: {message}")
    return "\n".join(lines)


def _report_exit_code(report: Mapping[str, Any]) -> int:
    """Map the aggregate tri-state verdict to the documented process status."""
    status = report["overall"]["status"]
    return 0 if status == "GREEN" else 1 if status == "RED" else 2


def _evaluate_live_reward_consistency(
    contract: Mapping[str, Any],
    runs: Mapping[str, Run],
    artifacts: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    submitted_job_ids: Mapping[str, str],
) -> Gate:
    """Check live logged reward equations without making a replay claim."""
    gate = Gate("live_reward_consistency")
    verifier_metric = contract["verifier_metric"]
    environment = contract["pair"]["environment"]
    required = {
        *REWARD_METRICS,
        *TOKEN_METRICS,
        *ZERO_STEP_METRICS,
        *CUMULATIVE_ZERO_METRICS,
        verifier_metric,
    }
    try:
        exit_job_ids = _validate_execution_receipts(contract, artifacts, pair_manifest)
        _exact_json_value(
            exit_job_ids,
            dict(submitted_job_ids),
            "EXIT/submission authenticated scheduler job IDs for live reward evidence",
        )
    except (EvidenceError, KeyError) as error:
        gate.unverifiable(str(error))

    if not _require_histories(gate, runs, required):
        gate.evidence = {
            "claim_scope": (
                "live_logged_reward_consistency_only_not_captured_output_" "parity_or_trajectory_equivalence"
            )
        }
        return gate

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
                if environment == "reasoning_gym":
                    if not 0.0 <= minimum <= maximum <= 1.0:
                        gate.fail(f"{run.arm} step {step} {label} min/max is outside [0,1]")
                else:
                    if minimum not in (0.0, 1.0) or maximum not in (0.0, 1.0):
                        gate.fail(f"{run.arm} step {step} {label} min/max is not binary")
                    if _close(
                        mean * K4_SAMPLES,
                        round(mean * K4_SAMPLES),
                        tolerance=REWARD_ABS_TOLERANCE,
                    ):
                        successes = round(mean * K4_SAMPLES)
                        expected_minimum = 1.0 if successes == K4_SAMPLES else 0.0
                        expected_maximum = 1.0 if successes > 0 else 0.0
                        if minimum != expected_minimum or maximum != expected_maximum:
                            gate.fail(
                                f"{run.arm} step {step} {label} min/max does not " "close to its K=4 success count"
                            )
                    else:
                        gate.fail(f"{run.arm} step {step} {label} mean is not a K=4 fraction")
                if minimum > mean or mean > maximum:
                    gate.fail(f"{run.arm} step {step} {label} mean is outside min/max")

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

    step_one_metrics = sorted(required)
    parity_mismatches = [
        metric
        for metric in step_one_metrics
        if runs["off"].history.values[1][metric] != runs["on"].history.values[1][metric]
    ]
    if parity_mismatches:
        gate.fail("W&B aggregate step-1 parity differs for " + ", ".join(parity_mismatches[:20]))
    gate.evidence = {
        "claim_scope": (
            "live_logged_reward_consistency_and_step1_aggregate_parity_only_"
            "not_captured_output_parity_or_trajectory_equivalence"
        ),
        "steps_checked_per_arm": len(STEPS),
        "wandb_aggregate_step1_exact": not parity_mismatches,
        "token_key": "train/global_valid_toks",
    }
    return gate


def evaluate_pair(
    contract_value: dict[str, Any],
    off_export: Document,
    on_export: Document,
    artifacts: Mapping[str, Any],
    *,
    expected_submission_receipt_sha256: str | None = None,
    expected_off_export_sha256: str | None = None,
    expected_on_export_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate the independent live reward, learning, and speed gates."""
    try:
        contract = validate_contract(contract_value)
        pair_manifest = _validate_pair_manifest(
            _required_artifact(artifacts, "pair_manifest", "Pair manifest"),
            contract,
        )
        submitted_job_ids = _authenticate_submission_receipt_bytes(
            artifacts,
            expected_submission_receipt_sha256,
            pair_manifest,
            contract,
        )
        exit_job_ids = _validate_execution_receipts(contract, artifacts, pair_manifest)
        _exact_json_value(
            exit_job_ids,
            dict(submitted_job_ids),
            "EXIT/submission authenticated scheduler job IDs",
        )
        _validate_terminal_scheduler_receipt(
            contract,
            artifacts,
            pair_manifest,
            submitted_job_ids,
            exit_job_ids,
        )
        off_expected = _digest(expected_off_export_sha256, "trusted OFF W&B export SHA-256")
        on_expected = _digest(expected_on_export_sha256, "trusted ON W&B export SHA-256")
        if off_expected == on_expected:
            raise EvidenceError("trusted OFF/ON W&B export SHA-256 pins must differ")
        pair_manifest_sha256 = contract["provenance"]["common"]["pair_manifest_sha256"]
        submission_receipt_sha256 = _required_artifact(
            artifacts, "submission_receipt", "pair submission receipt"
        ).sha256
        runs = {
            "off": _validate_run(
                off_export,
                off_expected,
                contract,
                "off",
                pair_manifest_sha256=pair_manifest_sha256,
                submission_receipt_sha256=submission_receipt_sha256,
                scheduler_job_id=submitted_job_ids["off"],
            ),
            "on": _validate_run(
                on_export,
                on_expected,
                contract,
                "on",
                pair_manifest_sha256=pair_manifest_sha256,
                submission_receipt_sha256=submission_receipt_sha256,
                scheduler_job_id=submitted_job_ids["on"],
            ),
        }
    except EvidenceError as error:
        return _unverifiable_report(contract_value, str(error))

    reward = _evaluate_live_reward_consistency(contract, runs, artifacts, pair_manifest, submitted_job_ids)
    learning = _evaluate_learning(contract, runs, artifacts)
    speed = _evaluate_speed(contract, runs, artifacts, pair_manifest, submitted_job_ids)
    statuses = {
        "live_reward_consistency": reward.status,
        "learning_behavior": learning.status,
        "speed_evidence": speed.status,
    }
    if any(status == "UNVERIFIABLE" for status in statuses.values()):
        overall = "UNVERIFIABLE"
    elif any(status == "FAIL" for status in statuses.values()):
        overall = "RED"
    else:
        overall = "GREEN"
    report = {
        "schema": REPORT_SCHEMA,
        "acceptance_scope": ACCEPTANCE_SCOPE,
        "threat_model": copy.deepcopy(THREAT_MODEL),
        "pair_id": contract["pair"]["pair_id"],
        "environment": contract["pair"]["environment"],
        "live_reward_consistency": reward.report(),
        "learning_behavior": learning.report(),
        "speed_evidence": speed.report(),
        "overall": {"status": overall, "independent_statuses": statuses},
    }
    try:
        return _validate_acceptance_report(report)
    except EvidenceError as error:
        return _unverifiable_report(contract, f"acceptance report failed exact validation: {error}")


def _load_live_artifacts(args: argparse.Namespace) -> dict[str, Document]:
    specifications = {
        "pair_manifest": (args.pair_manifest, "Pair manifest"),
        "submission_receipt": (
            args.submission_receipt,
            "pair submission receipt",
        ),
        "holdout": (args.holdout_receipt, "holdout receipt"),
        "off_execution": (
            args.off_execution_receipt,
            "OFF shared-prefix execution receipt",
        ),
        "on_execution": (
            args.on_execution_receipt,
            "ON shared-prefix execution receipt",
        ),
        "off_job_exit": (
            args.off_job_exit_receipt,
            "OFF strict job EXIT receipt",
        ),
        "on_job_exit": (
            args.on_job_exit_receipt,
            "ON strict job EXIT receipt",
        ),
        "terminal_scheduler": (
            args.terminal_scheduler_receipt,
            "terminal scheduler Pair receipt",
        ),
    }
    return {key: load_document(path, label) for key, (path, label) in specifications.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-acceptance-contract-sha256", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--expected-submission-receipt-sha256", required=True)
    parser.add_argument("--off-export", type=Path, required=True)
    parser.add_argument("--on-export", type=Path, required=True)
    parser.add_argument("--expected-off-export-sha256", required=True)
    parser.add_argument("--expected-on-export-sha256", required=True)
    parser.add_argument("--holdout-receipt", type=Path, required=True)
    parser.add_argument("--off-execution-receipt", type=Path, required=True)
    parser.add_argument("--on-execution-receipt", type=Path, required=True)
    parser.add_argument("--off-job-exit-receipt", type=Path, required=True)
    parser.add_argument("--on-job-exit-receipt", type=Path, required=True)
    parser.add_argument("--terminal-scheduler-receipt", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    args = parser.parse_args(argv)

    evaluator_path = Path(__file__).resolve(strict=True)
    evaluator_sha256 = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    contract_value: dict[str, Any] | None = None
    try:
        expected_evaluator_sha256 = _digest(
            args.expected_evaluator_sha256,
            "trusted evaluator source SHA-256",
        )
        if evaluator_sha256 != expected_evaluator_sha256:
            raise EvidenceError("evaluator source bytes differ from the trusted OOB SHA-256")
        contract = load_document(args.contract, "acceptance contract")
        expected_contract_sha256 = _digest(
            args.expected_acceptance_contract_sha256,
            "trusted acceptance contract SHA-256",
        )
        if contract.sha256 != expected_contract_sha256:
            raise EvidenceError("acceptance contract bytes differ from the trusted OOB SHA-256")
        _require_canonical_document(contract, "acceptance contract", trailing_lf=False)
        contract_value = contract.value
        artifacts = _load_live_artifacts(args)
        off_export = load_document(args.off_export, "OFF W&B export")
        on_export = load_document(args.on_export, "ON W&B export")
        report = evaluate_pair(
            contract.value,
            off_export,
            on_export,
            artifacts,
            expected_submission_receipt_sha256=(args.expected_submission_receipt_sha256),
            expected_off_export_sha256=args.expected_off_export_sha256,
            expected_on_export_sha256=args.expected_on_export_sha256,
        )
    except (EvidenceError, OSError) as error:
        report = _unverifiable_report(contract_value, str(error))
    report = _validate_acceptance_report(report)
    if args.format == "json":
        sys.stdout.write(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    else:
        sys.stdout.write(_text_report(report) + "\n")
    return _report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
