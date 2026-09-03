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

"""Produce profile-bound fail-closed evidence for strict captured replay V2.

This module deliberately contains no scheduler, NeMo, Ray, Torch, or W&B
dependency.  The login-side calibration launcher supplies the exact ``sbatch``
argv/stdout to :func:`build_captured_replay_submission_receipt_v2`; the independently
authenticated in-job wrapper supplies the successful EXIT observations to
:func:`build_captured_replay_exit_receipt_v2`.  A numeric ID returned by ``sbatch`` is
therefore only ever named ``candidate_job_id``.  It becomes useful authority
only after the in-job receipt binds the equal ``authenticated_job_id``.

The transcript bundle is also intentionally separate from the reward ledger.
``model_response_sha256`` (and the ledger's ``response_sha256``) hashes only
the exact model response object.  The inbound agent-run request, deterministically
reconstructed verifier request, verifier response, and raw reward are separate
fields and separate digest domains.  This prevents a reward change from being
misreported as a model-output change, without claiming the derived verifier
request was observed on the HTTP wire.

All bundles, ledgers, and indexes use sorted compact ASCII JSON with no final
LF.  Submission and EXIT receipts use the same encoding plus exactly one LF.
The publisher is exclusive and leaves an EUID-owned, single-link, mode-0400
regular file or fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import struct
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
    )


HASH_DOMAIN = "sha256-domain-nul-canonical-ascii-json-no-lf-v1"
_HASH_PREFIX = b"nemo-rl-strict-v2"

TRANSCRIPT_BUNDLE_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
MAIN_STEP1_LEDGER_SCHEMA = "nemo-rl-strict-main-step1-ledger-v5"
CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA = "nemo-rl-strict-captured-replay-step1-ledger-v5"
MODEL_TRANSPORT_BUNDLE_SCHEMA = "nemo-rl-strict-model-transport-bundle-v1"
REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v5"
REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-job-pre-receipt-v3"
REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-job-exit-receipt-v6"
REPLAY_POST_INDEX_V2_SCHEMA = "nemo-rl-strict-captured-replay-evidence-index-v4"
REPLAY_SCHEDULER_QUERY_SCHEMA = "nemo-rl-strict-captured-replay-scheduler-query-v3"
REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA = "nemo-rl-strict-captured-replay-runtime-attestation-v2"
REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-result-final-receipt-v1"
AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA = "nemo-rl-strict-captured-replay-authenticated-result-snapshot-v2"
_RESULT_INVENTORY_V2_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
_RESULT_INVENTORY_V2_FILENAME = "result-inventory-v2.json"
_RESULT_PROFILE_IDS_V2 = {
    "reasoning_gym": "reasoning-gym-exact-match-v1",
    "citation": "citation-string-match-v1",
    "freeform": "freeform-regex-v1",
}
_AUTHENTICATED_REPLAY_RESULT_V2_MINT_TOKEN = object()

__all__ = [
    "AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA",
    "AuthenticatedCapturedReplayResultV2",
    "REPLAY_EXIT_V2_ROOT_KEYS",
    "REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA",
    "REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA",
    "REPLAY_POST_INDEX_V2_ROOT_KEYS",
    "REPLAY_POST_INDEX_V2_SCHEMA",
    "REPLAY_PRE_V2_ROOT_KEYS",
    "REPLAY_RUNTIME_ATTESTATION_V2_KEYS",
    "REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA",
    "REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA",
    "REPLAY_SCHEDULER_QUERY_ROOT_KEYS",
    "REPLAY_SCHEDULER_QUERY_SCHEMA",
    "REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA",
    "REPLAY_SUBMISSION_V2_ROOT_KEYS",
    "build_captured_replay_evidence_index_v2",
    "build_captured_replay_exit_receipt_v2",
    "build_captured_replay_pre_receipt_v2",
    "build_captured_replay_scheduler_query_v2",
    "build_captured_replay_submission_receipt_v2",
    "build_captured_replay_result_final_receipt_v2",
    "decode_evidence_document_bytes",
    "load_authenticated_captured_replay_result_v2",
    "load_captured_replay_evidence_index_v2",
    "load_captured_replay_exit_receipt_v2",
    "load_captured_replay_pre_receipt_v2",
    "load_captured_replay_scheduler_query_v2",
    "load_captured_replay_submission_receipt_v2",
    "publish_captured_replay_evidence_index_v2",
    "publish_captured_replay_exit_receipt_v2",
    "publish_captured_replay_pre_receipt_v2",
    "publish_captured_replay_scheduler_query_v2",
    "publish_captured_replay_submission_receipt_v2",
    "publish_captured_replay_result_final_receipt_v2",
    "snapshot_authenticated_captured_replay_result_v2",
    "validate_captured_replay_evidence_index_v2",
    "validate_captured_replay_exit_receipt_v2",
    "validate_captured_replay_pre_receipt_v2",
    "validate_captured_replay_scheduler_query_v2",
    "validate_captured_replay_submission_receipt_v2",
]
REPLAY_EXECUTION_MANIFEST_V2_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
REPLAY_TRANSPORT_CONSUMPTION_V3_SCHEMA = "nemo-rl-strict-model-transport-replay-consumption-v3"
FORMAT_VERIFICATION_CALL_INDEX_SCHEMA = "nemo-rl-strict-format-verification-call-index-v1"
REASONING_SCORE_CALL_INDEX_SCHEMA = "nemo-rl-strict-reasoning-score-call-index-v1"
PAIR_MANIFEST_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
PAIR_SUBMISSION_RECEIPT_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
HARDWARE_OBSERVATION_SCHEMA = "nemo-rl-strict-hardware-observation-v2"
SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA = "nemo-rl-strict-scheduler-device-environment-v1"
HARDWARE_ORDERED_ROWS_HASH_LABEL = "captured-replay-nvidia-smi-ordered-rows-v1"

# This is the complete Python/shell execution closure admitted by manifest V4.
# Keep it local so no executable module is imported before its snapshot bytes
# have been authenticated against the manifest-carried digest.
REPLAY_PROGRAM_V2_PATHS = {
    "entrypoint": "examples/nemo_gym/run_strict_captured_replay_v2.py",
    "evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence_v2.py",
    "gym_child_bootstrap": "nemo_rl/environments/_strict_gym_child_bootstrap_v2/sitecustomize.py",
    "gym_child_runtime": "nemo_rl/environments/strict_gym_child_runtime_v2.py",
    "job_wrapper": "strict_pair_replay_job_wrapper_v2.sh",
    "legacy_evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence.py",
    "main_step_ledger": "nemo_rl/utils/strict_main_step_ledger.py",
    "manifest_utility": "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
    "model_transport_utility": "nemo_rl/utils/strict_model_transport.py",
    "profile_registry": "nemo_rl/utils/strict_captured_replay_profiles.py",
    "raw_transport_owner": "nemo_rl/utils/strict_model_transport_replay_v3.py",
    "result_sealer": "nemo_rl/utils/strict_captured_replay_seal_v2.py",
    "runtime": "nemo_rl/algorithms/strict_captured_replay_runtime_v2.py",
    "submission_launcher": "strict_pair_replay_launch_v2.sh",
}

K4_SAMPLES = 4
STRICT_ENVIRONMENTS = frozenset({"reasoning_gym", "citation", "freeform"})
AGENT_BY_ENVIRONMENT = {
    "reasoning_gym": "reasoning_gym_simple_agent",
    "citation": "citation_format_simple_agent",
    "freeform": "freeform_formatting_simple_agent",
}
FIXTURE_ROW_KEYS_BY_ENVIRONMENT = {
    "reasoning_gym": frozenset({"agent_ref", "answer", "metadata", "question", "responses_create_params"}),
    "citation": frozenset({"agent_ref", "responses_create_params", "verifier"}),
    "freeform": frozenset({"agent_ref", "responses_create_params", "verifier"}),
}
VERIFIER_RESPONSE_KEYS_BY_ENVIRONMENT = {
    "reasoning_gym": frozenset(
        {
            "responses_create_params",
            "response",
            "reward",
            "task_name",
            "score",
            "extracted_answer",
        }
    ),
    "citation": frozenset(
        {
            "responses_create_params",
            "response",
            "reward",
            "verifier",
            "match_details",
        }
    ),
    "freeform": frozenset(
        {
            "responses_create_params",
            "response",
            "reward",
            "verifier",
            "match_details",
        }
    ),
}
DERIVED_VERIFIER_REQUEST_SCHEMA = "nemo-rl-strict-derived-verifier-request-v1"
DERIVED_VERIFIER_REQUEST_SOURCE = {
    "base_resources": {
        "path": "nemo_gym/base_resources_server.py",
        "sha256": "b106a97397cdce8da2c1dbacd0b0b4b862ec03e664704e38044025fc9046693d",
    },
    "openai_utils": {
        "path": "nemo_gym/openai_utils.py",
        "sha256": "2e612f284de3cd290f76ccea8eccf577805127cfe3ea92d24f95b4ca4a068dce",
    },
    "simple_agent": {
        "path": "responses_api_agents/simple_agent/app.py",
        "sha256": "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a",
    },
}
DERIVED_VERIFIER_REQUEST_RUNTIME = {
    "openai_version": "2.6.1",
    "pydantic_version": "2.13.4",
}
VERIFIER_REQUEST_DERIVATION_KEYS = frozenset(
    {
        "schema",
        "assurance",
        "algorithm",
        "gym_gitlink_commit",
        "gym_tree",
        "runtime",
        "sources",
    }
)
EXPANDED_RESPONSES_CREATE_PARAMS_KEYS = frozenset(
    {
        "background",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "reasoning",
        "service_tier",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
)
REPLAY_ATTEMPTS = ("replay-1", "replay-2")
MAIN_EVIDENCE_DIRECTORY = "strict_pair_step1_evidence"
MAIN_TRANSCRIPT_FILENAME = "transcript-bundle.json"

TRANSCRIPT_BUNDLE_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "step",
        "sample_count",
        "generation",
        "bindings",
        "entries",
        "entries_sha256",
        "fixture_row",
        "model_transport_bundle",
        "verifier_request_derivation",
    }
)
TRANSCRIPT_BINDING_KEYS = frozenset(
    {
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "job_id",
        "run_id",
        "fixture_sha256",
        "verifier_source_sha256",
        "config_sha256",
        "snapshot_manifest_sha256",
    }
)
TRANSCRIPT_ENTRY_INPUT_KEYS = frozenset(
    {
        "sample_index",
        "fixture_row_index",
        "rollout_index",
        "generation_seed",
        "generation_request",
        "model_response",
        "agent_run_request",
        "derived_verifier_request",
        "verifier_response",
        "raw_environment_reward",
        "model_transport_entry_sha256",
        "model_transport_request_body_sha256",
        "model_transport_response_body_sha256",
    }
)
TRANSCRIPT_ENTRY_KEYS = TRANSCRIPT_ENTRY_INPUT_KEYS | frozenset(
    {
        "generation_request_sha256",
        "model_response_sha256",
        "agent_run_request_sha256",
        "derived_verifier_request_sha256",
        "verifier_response_sha256",
        "entry_sha256",
    }
)
GENERATION_KEYS = frozenset({"seed_base", "max_new_tokens", "temperature", "top_k", "top_p"})
ARTIFACT_REFERENCE_KEYS = frozenset({"path", "schema", "sha256"})

REPLAY_LEDGER_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "source_main_ledger_sha256",
        "source_transcript_bundle",
        "step",
        "sample_count",
        "compared_fields",
        "generation",
        "bindings",
        "transcript_bundle",
        "rows",
        "step_totals",
        "cohort_sha256",
        "outputs_sha256",
        "rewards_sha256",
        "ordered_rows_sha256",
    }
)
REPLAY_LEDGER_BINDING_KEYS = frozenset(
    {
        "config_sha256",
        "fixture_sha256",
        "job_id",
        "pair_campaign_reward_and_advantage_sha256",
        "pair_campaign_sha256",
        "pair_manifest_sha256",
        "process",
        "restart_count",
        "run_id",
        "snapshot_manifest_sha256",
        "submission_receipt_sha256",
        "verifier_source_sha256",
    }
)

REPLAY_SUBMISSION_V2_ROOT_KEYS = frozenset(
    {
        "schema",
        "scorer_profile",
        "phase",
        "status",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "replay_execution_manifest",
        "replay_source_snapshot",
        "submission_contract",
        "slurm_export_boundary",
        "submission_launcher",
        "job_wrapper",
        "scheduler_client_environment",
        "scheduler_tools",
        "submission_nonce",
        "job_name",
        "comment",
        "submitter_euid",
        "sbatch",
        "candidate_job_id",
        "accepted_id_record",
        "pre_release_scheduler_query",
        "submitted_at_unix_ns",
    }
)
SBATCH_KEYS = frozenset(
    {
        "path",
        "sha256",
        "argv",
        "argv_sha256",
        "parsable_stdout",
        "parsable_stdout_sha256",
    }
)
REPLAY_PRE_V2_ROOT_KEYS = frozenset(
    {
        "schema",
        "scorer_profile",
        "phase",
        "status",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "replay_execution_manifest",
        "submission_receipt",
        "candidate_job_id",
        "authenticated_job_id",
        "job",
        "static_boundary",
        "pre_scheduler_query",
        "output_precondition",
        "runtime_attestation_contract",
        "execution_source_root",
        "driver",
        "post_verified",
    }
)
REPLAY_EXIT_V2_ROOT_KEYS = frozenset(
    {
        "schema",
        "scorer_profile",
        "phase",
        "status",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "replay_execution_manifest",
        "submission_receipt",
        "candidate_job_id",
        "authenticated_job_id",
        "job",
        "static_boundary",
        "pre_receipt",
        "post_scheduler_query",
        "driver_exit_code",
        "hardware",
        "scheduler_device_environment",
        "driver_scheduler_device_environment",
        "driver_process",
        "runtime_attestation",
        "outputs",
        "post_verified",
    }
)
REPLAY_POST_INDEX_V2_ROOT_KEYS = frozenset(
    {
        "schema",
        "scorer_profile",
        "original_process_reaped",
        "profile_id",
        "hash_domain",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "replay_execution_manifest",
        "pair_submission_receipt",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "source_capture",
        "outputs",
        "identity",
    }
)
REPLAY_RESULT_FINAL_V2_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "phase",
        "status",
        "pair_id",
        "environment",
        "scorer_profile",
        "arm",
        "mode",
        "attempt_id",
        "candidate_job_id",
        "authenticated_job_id",
        "driver_process",
        "original_process_reaped",
        "replay_execution_manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "evidence_index",
        "result",
    }
)
REPLAY_STATIC_BOUNDARY_V2_KEYS = frozenset(
    {
        "scorer_profile",
        "pair",
        "source_capture",
        "replay_contract",
        "artifacts",
        "execution_environment",
        "wandb",
        "slurm_export_boundary",
        "deployment",
        "runtime_tools",
        "container_entry_boundary",
        "source",
    }
)
REPLAY_OUTPUT_V2_KEYS = frozenset(
    {
        "scorer_call_index",
        "transport_consumption",
        "transcript_bundle",
        "replay_ledger",
    }
)
REPLAY_POST_IDENTITY_KEYS = frozenset({"candidate_job_id", "authenticated_job_id", "driver_process", "run_id"})
_AUTHENTICATED_RESULT_SNAPSHOT_V2_KEYS = frozenset(
    {
        "schema",
        "pair_id",
        "environment",
        "profile_id",
        "attempt_id",
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_root",
        "result_inventory",
        "evidence_index",
        "outputs",
        "samples",
    }
)
_AUTHENTICATED_RESULT_SAMPLE_V2_KEYS = frozenset(
    {
        "sample_index",
        "fixture_row_index",
        "rollout_index",
        "generation_seed",
        "model_transport_entry_sha256",
        "model_transport_request_body_sha256",
        "model_transport_response_body_sha256",
        "model_response_sha256",
        "match_details",
        "raw_environment_reward",
    }
)
REPLAY_RUNTIME_ATTESTATION_V2_KEYS = frozenset(
    {
        "schema",
        "scorer_profile",
        "original_process_reaped",
        "environment",
        "profile_id",
        "requirements",
        "scorer_call_index",
        "transport_consumption",
        "transcript_bundle",
        "replay_ledger",
    }
)
REPLAY_OUTPUT_PRECONDITION_KEYS = frozenset({"path", "mode", "status"})
REPLAY_SCHEDULER_QUERY_ROOT_KEYS = frozenset(
    {
        "schema",
        "phase",
        "argv",
        "path",
        "sha256",
        "byte_count",
        "line_count",
        "status",
        "normalization",
        "records",
        "match_count",
    }
)
REPLAY_SCHEDULER_RECORD_KEYS = frozenset(
    {
        "job_id",
        "job_name",
        "comment",
        "user_id",
        "work_dir",
        "job_state",
        "reason",
        "held",
        "restart_count",
    }
)
REPLAY_SCHEDULER_QUERY_NORMALIZATION = {
    "algorithm": "scontrol-show-job-json-v1",
    "complete": True,
    "duplicate_keys_rejected": True,
    "nonfinite_numbers_rejected": True,
    "negative_zero_rejected": True,
}
REPLAY_ACCEPTED_ID_RECORD_KEYS = frozenset({"path", "sha256", "parsed_candidate_job_id", "format", "mode"})
REPLAY_SCHEDULER_CLIENT_ENVIRONMENT_KEYS = frozenset({"ambient_merge", "env", "variables"})
REPLAY_SCHEDULER_CLIENT_VARIABLE_KEYS = frozenset({"LC_ALL", "SLURM_CONF"})
REPLAY_SCHEDULER_TOOL_KEYS = frozenset({"sbatch", "scancel", "scontrol"})
JOB_KEYS = frozenset(
    {
        "account",
        "name",
        "num_nodes",
        "partition",
        "qos",
        "gpus_per_node",
        "restart_count",
    }
)
HARDWARE_KEYS = frozenset(
    {
        "schema",
        "gpu_model",
        "driver_version",
        "gpu_row_count",
        "ordered_rows",
        "raw_output_sha256",
        "ordered_rows_sha256",
        "nvidia_smi",
    }
)
HARDWARE_ROW_KEYS = frozenset({"index", "raw", "gpu_model", "driver_version"})
TOOL_REFERENCE_KEYS = frozenset({"path", "sha256"})
SCHEDULER_DEVICE_ENVIRONMENT_KEYS = frozenset(
    {
        "schema",
        "cuda_visible_devices",
        "gpu_device_ordinal",
        "nvidia_visible_devices",
        "rocr_visible_devices",
        "ze_affinity_mask",
    }
)
PROCESS_KEYS = frozenset({"boot_id_sha256", "pid", "start_time_ticks"})
RUNTIME_TOOL_KEYS = frozenset({"manifest_path", "manifest_sha256"})
REPLAY_RUNTIME_KEYS = frozenset(
    {
        "required",
        "bundle_sha256",
        "entries",
        "hits",
        "misses",
        "reuses",
        "pending",
        "streaming_rejections",
        "ready_marker_sha256",
        "hit_markers_sha256",
    }
)

CROSS_ARM_PARITY_FIELDS = (
    "sample_index",
    "fixture_row_index",
    "rollout_index",
    "prompt_sha256",
    "request_sha256",
    "agent_run_request_sha256",
    "generation_seed",
    "token_ids",
    "input_length",
    "prompt_token_ids",
    "completion_token_ids",
    "token_loss_mask",
    "raw_environment_reward",
    "pre_penalty_environment_reward",
    "penalty_flags",
    "verifier_reward",
    "processed_reward",
    "sample_mask",
    "advantages",
    "valid_loss_tokens",
    "total_tokens",
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_CANONICAL_DEVICE_COMPONENT = r"(?:0|[1-9][0-9]*)"
_ZE_DEVICE_RE = re.compile(rf"{_CANONICAL_DEVICE_COMPONENT}(?:\.{_CANONICAL_DEVICE_COMPONENT})?\Z")
_RESPONSE_UUID_RE = re.compile(r"(?:resp|rs|msg)_[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}\Z")
_GYM_RESPONSE_ROOT_KEYS = frozenset(
    {
        "background",
        "conversation",
        "created_at",
        "error",
        "id",
        "incomplete_details",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "object",
        "output",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "status",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "usage",
        "user",
    }
)
_GYM_REASONING_KEYS = frozenset({"content", "encrypted_content", "id", "summary", "type"})
_GYM_MESSAGE_KEYS = frozenset({"content", "id", "role", "status", "type"})
_GYM_TOKEN_KEYS = frozenset(
    {
        "generation_log_probs",
        "generation_token_ids",
        "prompt_token_ids",
        "routed_experts",
    }
)
_GYM_OUTPUT_TEXT_KEYS = frozenset({"annotations", "logprobs", "text", "type"})
_GYM_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "input_tokens_details",
        "output_tokens",
        "output_tokens_details",
        "total_tokens",
    }
)
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_DOCUMENT_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_MAX_STRICT_FIXTURE_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_DOCUMENT_BYTES = 256 * 1024 * 1024


class AuthenticatedCapturedReplayResultV2:
    """Opaque in-process authority for one fully authenticated replay result."""

    __slots__ = (
        "__authenticated_source",
        "__candidate_job_id",
        "__expected_environment",
        "__expected_profile_id",
        "__final_path",
        "__final_sha256",
        "__manifest_raw",
        "__pre_raw",
        "__result_capability",
        "__submission_raw",
        "__exit_raw",
        "__final_raw",
        "__mint_token",
        "__source_transcript_raw",
    )

    def __init__(
        self,
        *,
        _mint_token: object,
        authenticated_source: AuthenticatedOffSourceCapture,
        candidate_job_id: str,
        expected_environment: str,
        expected_profile_id: str,
        final_path: str,
        final_sha256: str,
        manifest_raw: bytes,
        submission_raw: bytes,
        pre_raw: bytes,
        exit_raw: bytes,
        final_raw: bytes,
        source_transcript_raw: bytes,
        result_capability: object,
    ) -> None:
        if _mint_token is not _AUTHENTICATED_REPLAY_RESULT_V2_MINT_TOKEN:
            raise ValueError("authenticated replay result may only be minted by the public loader")
        values = {
            "_AuthenticatedCapturedReplayResultV2__mint_token": _mint_token,
            "_AuthenticatedCapturedReplayResultV2__authenticated_source": authenticated_source,
            "_AuthenticatedCapturedReplayResultV2__candidate_job_id": candidate_job_id,
            "_AuthenticatedCapturedReplayResultV2__expected_environment": expected_environment,
            "_AuthenticatedCapturedReplayResultV2__expected_profile_id": expected_profile_id,
            "_AuthenticatedCapturedReplayResultV2__final_path": final_path,
            "_AuthenticatedCapturedReplayResultV2__final_sha256": final_sha256,
            "_AuthenticatedCapturedReplayResultV2__manifest_raw": manifest_raw,
            "_AuthenticatedCapturedReplayResultV2__submission_raw": submission_raw,
            "_AuthenticatedCapturedReplayResultV2__pre_raw": pre_raw,
            "_AuthenticatedCapturedReplayResultV2__exit_raw": exit_raw,
            "_AuthenticatedCapturedReplayResultV2__final_raw": final_raw,
            "_AuthenticatedCapturedReplayResultV2__source_transcript_raw": source_transcript_raw,
            "_AuthenticatedCapturedReplayResultV2__result_capability": result_capability,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("AuthenticatedCapturedReplayResultV2 is immutable")

    def __reduce__(self) -> object:
        raise TypeError("AuthenticatedCapturedReplayResultV2 cannot be pickled")


def canonical_ascii_json(value: Any) -> bytes:
    """Return sorted compact finite ASCII JSON with no trailing LF."""
    _validate_strict_json_types(value, path="$")
    _reject_negative_zero(value, path="$")
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("evidence value is not canonical finite ASCII JSON") from error
    if payload.endswith(b"\n"):
        raise AssertionError("canonical JSON unexpectedly ended in LF")
    return payload


def receipt_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the only accepted receipt framing: canonical JSON plus one LF."""
    return canonical_ascii_json(value) + b"\n"


def document_sha256(value: Mapping[str, Any], *, trailing_lf: bool) -> str:
    payload = receipt_bytes(value) if trailing_lf else canonical_ascii_json(value)
    return hashlib.sha256(payload).hexdigest()


def domain_sha256(label: str, value: Any) -> str:
    """Hash canonical JSON in the frozen strict-v2 NUL-separated domain."""
    _require_ascii(label, name="hash label", maximum=128)
    if "\x00" in label:
        raise ValueError("hash label must not contain NUL")
    preimage = _HASH_PREFIX + b"\x00" + label.encode("ascii") + b"\x00"
    return hashlib.sha256(preimage + canonical_ascii_json(value)).hexdigest()


def derive_nemo_gym_request_seed(*, seed_base: int, fixture_row_index: int, rollout_index: int) -> int:
    """Recompute the exact signed-int63 request seed used by NeMo-Gym."""
    _require_nonnegative_int(seed_base, name="seed_base")
    _require_nonnegative_int(fixture_row_index, name="fixture_row_index")
    _require_nonnegative_int(rollout_index, name="rollout_index")
    identity = canonical_ascii_json([seed_base, fixture_row_index, rollout_index])
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & ((1 << 63) - 1)


def replay_run_id(*, environment: str, pair_id: str, attempt_id: str) -> str:
    _validate_identity(pair_id=pair_id, environment=environment)
    _require_attempt(attempt_id)
    return hashlib.sha256(f"nemo-rl-strict-replay-v2:{environment}:{pair_id}:{attempt_id}".encode("ascii")).hexdigest()


def replay_job_name(*, pair_id: str, attempt_id: str) -> str:
    _require_safe_id(pair_id, name="pair_id", maximum=64)
    _require_attempt(attempt_id)
    value = f"strict-replay-{attempt_id}-{pair_id}"
    return _require_ascii(value, name="job_name", maximum=255)


def main_transcript_bundle_path(results_dir: str) -> Path:
    root = _canonical_absolute_path(results_dir, name="RESULTS_DIR")
    if root == Path("/"):
        raise ValueError("RESULTS_DIR must not be root")
    return root / MAIN_EVIDENCE_DIRECTORY / MAIN_TRANSCRIPT_FILENAME


def load_strict_fixture_row0(*, path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Stable-read the Pair-authenticated five-row fixture and return row zero."""
    fixture_path = _canonical_absolute_path(str(path), name="fixture path")
    expected = _require_digest(expected_sha256, name="fixture expected_sha256")
    parent_fd = _open_absolute_directory_without_symlinks(fixture_path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        pre_named = os.stat(
            fixture_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(pre_named.st_mode) or not 0 < pre_named.st_size <= _MAX_STRICT_FIXTURE_BYTES:
            raise RuntimeError("strict fixture must be a bounded regular file")
        descriptor = os.open(
            fixture_path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                _file_fingerprint(pre_named) != _file_fingerprint(before)
                or not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= _MAX_STRICT_FIXTURE_BYTES
            ):
                raise RuntimeError("strict fixture changed before stable read")
            raw = _read_all_bounded(descriptor, maximum=_MAX_STRICT_FIXTURE_BYTES)
            after = os.fstat(descriptor)
            named = os.stat(fixture_path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_without_symlinks(fixture_path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            fresh_named = os.stat(
                fixture_path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
    finally:
        os.close(parent_fd)
    if not (
        _directory_identity(parent_before) == _directory_identity(parent_after) == _directory_identity(fresh_parent)
    ):
        raise RuntimeError("strict fixture parent changed during stable read")
    if not (
        _file_fingerprint(pre_named)
        == _file_fingerprint(before)
        == _file_fingerprint(after)
        == _file_fingerprint(named)
        == _file_fingerprint(fresh_named)
    ):
        raise RuntimeError("strict fixture changed during stable read")
    if len(raw) != after.st_size:
        raise RuntimeError("strict fixture size changed during stable read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("strict fixture bytes differ from Pair SHA-256")
    if not raw.endswith(b"\n"):
        raise ValueError("strict fixture every row must be LF terminated")
    lines = raw[:-1].split(b"\n")
    if len(lines) != 5 or any(not line for line in lines):
        raise ValueError("strict fixture must contain exactly five nonblank rows")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if b"\r" in line:
            raise ValueError("strict fixture must use LF framing without CR")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"strict fixture row {index} is not UTF-8") from error
        try:
            row = json.loads(
                text,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
                parse_int=_parse_fixture_int,
                parse_float=_parse_fixture_float,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"strict fixture row {index} is not strict JSON") from error
        if type(row) is not dict:
            raise TypeError(f"strict fixture row {index} must be a JSON object")
        canonical_ascii_json(row)
        rows.append(row)
    return copy.deepcopy(rows[0])


def build_verifier_request_derivation(
    *,
    gym_gitlink_commit: str,
    gym_tree: str,
    openai_version: str,
    pydantic_version: str,
) -> dict[str, Any]:
    """Build the pinned deterministic verifier-request reconstruction contract."""
    for value, name in (
        (gym_gitlink_commit, "gym_gitlink_commit"),
        (gym_tree, "gym_tree"),
    ):
        if type(value) is not str or _HEX40_RE.fullmatch(value) is None or value == "0" * 40:
            raise ValueError(f"{name} must be a nonzero lowercase Git object ID")
    runtime = {
        "openai_version": openai_version,
        "pydantic_version": pydantic_version,
    }
    if not _exact_json_equal(runtime, DERIVED_VERIFIER_REQUEST_RUNTIME):
        raise ValueError("verifier request derivation runtime versions differ from the pinned contract")
    document: dict[str, Any] = {
        "schema": DERIVED_VERIFIER_REQUEST_SCHEMA,
        "assurance": "deterministic-reconstruction-not-wire-capture",
        "algorithm": "pinned-simple-agent-model-dump-v1",
        "gym_gitlink_commit": gym_gitlink_commit,
        "gym_tree": gym_tree,
        "runtime": runtime,
        "sources": copy.deepcopy(DERIVED_VERIFIER_REQUEST_SOURCE),
    }
    _require_exact_keys(
        document,
        VERIFIER_REQUEST_DERIVATION_KEYS,
        name="verifier_request_derivation",
    )
    canonical_ascii_json(document)
    return document


def validate_verifier_request_derivation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and rebuild the exact pinned reconstruction contract."""
    _require_exact_keys(
        value,
        VERIFIER_REQUEST_DERIVATION_KEYS,
        name="verifier_request_derivation",
    )
    _require_exact_keys(
        value["runtime"],
        frozenset({"openai_version", "pydantic_version"}),
        name="verifier_request_derivation.runtime",
    )
    rebuilt = build_verifier_request_derivation(
        gym_gitlink_commit=value["gym_gitlink_commit"],
        gym_tree=value["gym_tree"],
        openai_version=value["runtime"]["openai_version"],
        pydantic_version=value["runtime"]["pydantic_version"],
    )
    if not _exact_json_equal(value, rebuilt):
        raise ValueError("verifier_request_derivation differs from the pinned contract")
    return rebuilt


def build_transcript_bundle(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    mode: str,
    attempt_id: str | None,
    generation: Mapping[str, Any],
    bindings: Mapping[str, Any],
    fixture_row: Mapping[str, Any],
    model_transport_bundle: Mapping[str, Any],
    verifier_request_derivation: Mapping[str, Any],
    entry_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one exact K=4 main or replay transcript bundle."""
    _validate_identity(pair_id=pair_id, environment=environment)
    if arm not in {"off", "on"}:
        raise ValueError("transcript arm must be 'off' or 'on'")
    if attempt_id is None:
        expected_mode = "observe" if arm == "off" else "train"
        if mode != expected_mode:
            raise ValueError(f"main transcript arm={arm!r} requires mode={expected_mode!r}")
    else:
        _require_attempt(attempt_id)
        if arm != "on" or mode != "captured_replay":
            raise ValueError("replay transcript requires arm='on' and mode='captured_replay'")
    generation_document = _validate_generation(generation)
    binding_document = _validate_transcript_bindings(bindings)
    fixture_value = _validate_fixture_row(fixture_row, environment=environment)
    fixture_document: dict[str, Any] = {"index": 0, "value": fixture_value}
    fixture_document["sha256"] = domain_sha256("step1-fixture-row", fixture_document)
    transport_reference = _artifact_reference(
        model_transport_bundle,
        name="model_transport_bundle",
        expected_schema=MODEL_TRANSPORT_BUNDLE_SCHEMA,
    )
    derivation_document = validate_verifier_request_derivation(verifier_request_derivation)
    if not isinstance(entry_inputs, Sequence) or isinstance(entry_inputs, (str, bytes, bytearray)):
        raise TypeError("entry_inputs must be a sequence")
    if len(entry_inputs) != K4_SAMPLES:
        raise ValueError(f"transcript bundle requires exactly K={K4_SAMPLES} entries")

    entries = [
        _build_transcript_entry(
            raw,
            index=index,
            generation=generation_document,
            environment=environment,
            expected_agent=AGENT_BY_ENVIRONMENT[environment],
            fixture_row=fixture_value,
        )
        for index, raw in enumerate(entry_inputs)
    ]
    task_indices = {entry["agent_run_request"]["_ng_task_index"] for entry in entries}
    if len(task_indices) != 1:
        raise ValueError("transcript entries do not share one NeMo-Gym task index")
    document: dict[str, Any] = {
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "mode": mode,
        "attempt_id": attempt_id,
        "step": 1,
        "sample_count": K4_SAMPLES,
        "generation": generation_document,
        "bindings": binding_document,
        "fixture_row": fixture_document,
        "model_transport_bundle": transport_reference,
        "verifier_request_derivation": derivation_document,
        "entries": entries,
        "entries_sha256": domain_sha256("step1-transcript-entries", entries),
    }
    _require_exact_keys(document, TRANSCRIPT_BUNDLE_ROOT_KEYS, name="transcript bundle")
    canonical_ascii_json(document)
    return document


def validate_transcript_bundle(document: Mapping[str, Any]) -> None:
    """Rebuild a transcript bundle and reject every non-derived value."""
    _require_exact_keys(document, TRANSCRIPT_BUNDLE_ROOT_KEYS, name="transcript bundle")
    if document["schema"] != TRANSCRIPT_BUNDLE_SCHEMA:
        raise ValueError("unexpected transcript bundle schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected transcript hash domain")
    if (
        type(document["step"]) is not int
        or document["step"] != 1
        or type(document["sample_count"]) is not int
        or document["sample_count"] != K4_SAMPLES
    ):
        raise ValueError("transcript bundle must describe exactly step 1 K=4")
    entries = document["entries"]
    if not isinstance(entries, list):
        raise TypeError("transcript bundle entries must be a list")
    rebuilt = build_transcript_bundle(
        pair_id=document["pair_id"],
        environment=document["environment"],
        arm=document["arm"],
        mode=document["mode"],
        attempt_id=document["attempt_id"],
        generation=document["generation"],
        bindings=document["bindings"],
        fixture_row=document["fixture_row"]["value"],
        model_transport_bundle=document["model_transport_bundle"],
        verifier_request_derivation=document["verifier_request_derivation"],
        entry_inputs=[{key: entry[key] for key in TRANSCRIPT_ENTRY_INPUT_KEYS} for entry in entries],
    )
    if dict(document) != rebuilt:
        raise ValueError("transcript bundle contains a changed or non-derived value")
    canonical_ascii_json(document)


def validate_transcript_model_transport_join(
    *,
    transcript_bundle: Mapping[str, Any],
    model_transport_bundle: Mapping[str, Any],
    model_transport_policy: Mapping[str, Any],
    model_path: str,
) -> None:
    """Bind a transcript's flat raw-body digests to one validated K=4 bundle."""
    validate_transcript_bundle(transcript_bundle)

    # Call-time import avoids a module-initialization cycle: the transport
    # utility imports this module for the shared canonical hash/file helpers.
    from nemo_rl.utils.strict_model_transport import (
        validate_model_transport_bundle,
        validate_model_transport_generation_request_join,
        validate_model_transport_model_response_join,
    )

    validate_model_transport_bundle(
        model_transport_bundle,
        model_transport_policy=model_transport_policy,
        model_path=model_path,
        expected_generation_inputs=[entry["generation_request"]["input"] for entry in transcript_bundle["entries"]],
    )
    reference = transcript_bundle["model_transport_bundle"]
    actual_sha256 = document_sha256(model_transport_bundle, trailing_lf=False)
    if reference["sha256"] != actual_sha256:
        raise ValueError("transcript model transport bundle digest does not close")
    for key in ("pair_id", "environment"):
        if transcript_bundle[key] != model_transport_bundle[key]:
            raise ValueError(f"transcript/model transport {key} differs")
    expected_transport_arm = "off" if transcript_bundle["mode"] == "captured_replay" else transcript_bundle["arm"]
    if model_transport_bundle["arm"] != expected_transport_arm:
        raise ValueError("transcript/model transport authority arm differs")
    transport_entries = model_transport_bundle["entries"]
    for index, (transcript, transport) in enumerate(zip(transcript_bundle["entries"], transport_entries, strict=True)):
        expected = {
            "model_transport_entry_sha256": transport["entry_sha256"],
            "model_transport_request_body_sha256": transport["request_body_sha256"],
            "model_transport_response_body_sha256": transport["response_body_sha256"],
            "rollout_index": transport["rollout_index"],
            "generation_seed": transport["generation_seed"],
        }
        for name, value in expected.items():
            if transcript[name] != value or type(transcript[name]) is not type(value):
                raise ValueError(f"transcript/model transport entry {index} {name} differs")
    validate_model_transport_generation_request_join(
        model_transport_bundle,
        generation_requests=[entry["generation_request"] for entry in transcript_bundle["entries"]],
    )
    validate_model_transport_model_response_join(
        model_transport_bundle,
        generation_requests=[entry["generation_request"] for entry in transcript_bundle["entries"]],
        model_responses=[entry["model_response"] for entry in transcript_bundle["entries"]],
    )


def validate_captured_replay_source_join(
    *,
    source_transcript_bundle: Mapping[str, Any],
    replay_transcript_bundle: Mapping[str, Any],
) -> None:
    """Prove that replay consumes the OFF cohort while rerunning its verifier.

    The generation request and inbound agent-run request are immutable replay
    inputs.  The derived verifier request and full Gym response carry fresh UUID4
    identifiers, so they remain arm-local; each is instead checked independently
    against the exact raw OFF model transport by
    ``validate_transcript_model_transport_join``.
    """
    validate_transcript_bundle(source_transcript_bundle)
    validate_transcript_bundle(replay_transcript_bundle)
    if (
        source_transcript_bundle["arm"] != "off"
        or source_transcript_bundle["mode"] != "observe"
        or source_transcript_bundle["attempt_id"] is not None
    ):
        raise ValueError("captured replay source must be the OFF main transcript")
    if replay_transcript_bundle["arm"] != "on" or replay_transcript_bundle["mode"] != "captured_replay":
        raise ValueError("captured replay target must be an ON replay transcript")
    _require_attempt(replay_transcript_bundle["attempt_id"])
    for key in (
        "pair_id",
        "environment",
        "step",
        "sample_count",
        "generation",
        "fixture_row",
        "model_transport_bundle",
        "verifier_request_derivation",
    ):
        if not _exact_json_equal(source_transcript_bundle[key], replay_transcript_bundle[key]):
            raise ValueError(f"source/replay transcript {key} differs")
    for key in (
        "pair_manifest_sha256",
        "fixture_sha256",
        "verifier_source_sha256",
    ):
        if source_transcript_bundle["bindings"][key] != replay_transcript_bundle["bindings"][key]:
            raise ValueError(f"source/replay transcript binding {key} differs")
    for index, (source, replay) in enumerate(
        zip(
            source_transcript_bundle["entries"],
            replay_transcript_bundle["entries"],
            strict=True,
        )
    ):
        for key in (
            "sample_index",
            "fixture_row_index",
            "rollout_index",
            "generation_seed",
            "generation_request",
            "agent_run_request",
            "model_transport_entry_sha256",
            "model_transport_request_body_sha256",
            "model_transport_response_body_sha256",
        ):
            if not _exact_json_equal(source[key], replay[key]):
                raise ValueError(f"source/replay entry {index} {key} differs")


def build_captured_replay_step1_ledger(
    *,
    pair_id: str,
    environment: str,
    attempt_id: str,
    source_main_ledger_sha256: str,
    source_transcript_bundle: Mapping[str, Any],
    source_transcript_document: Mapping[str, Any],
    generation: Mapping[str, Any],
    bindings: Mapping[str, Any],
    transcript_bundle: Mapping[str, Any],
    transcript_document: Mapping[str, Any],
    row_inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a v5 ON replay ledger after authenticating both transcript docs."""
    _validate_identity(pair_id=pair_id, environment=environment)
    attempt = _require_attempt(attempt_id)
    source_main_sha = _require_digest(source_main_ledger_sha256, name="source_main_ledger_sha256")
    source_transcript_ref = _artifact_reference(
        source_transcript_bundle,
        name="source_transcript_bundle",
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
    )
    replay_transcript_ref = _artifact_reference(
        transcript_bundle,
        name="transcript_bundle",
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
    )
    validate_transcript_bundle(source_transcript_document)
    validate_transcript_bundle(transcript_document)
    if source_transcript_ref["sha256"] != document_sha256(source_transcript_document, trailing_lf=False):
        raise ValueError("source transcript reference does not bind document bytes")
    if replay_transcript_ref["sha256"] != document_sha256(transcript_document, trailing_lf=False):
        raise ValueError("replay transcript reference does not bind document bytes")
    validate_captured_replay_source_join(
        source_transcript_bundle=source_transcript_document,
        replay_transcript_bundle=transcript_document,
    )
    binding_document = _validate_replay_ledger_bindings(
        bindings,
        pair_id=pair_id,
        environment=environment,
        attempt_id=attempt,
    )

    # Import at call time.  The main runtime imports this common transcript
    # module, so a top-level reverse import would create a cycle.  At execution
    # time both modules are initialized, and the main builder supplies the one
    # frozen implementation of row geometry/reward/projection algebra.
    from nemo_rl.utils.strict_main_step_ledger import build_main_step1_ledger

    main_compatible_transcript_ref = dict(replay_transcript_ref)
    main_compatible_transcript_ref["path"] = str(
        Path(replay_transcript_ref["path"]).parent / MAIN_EVIDENCE_DIRECTORY / MAIN_TRANSCRIPT_FILENAME
    )
    main_shape = build_main_step1_ledger(
        pair_id=pair_id,
        environment=environment,
        arm="on",
        mode="train",
        generation=generation,
        bindings={key: value for key, value in binding_document.items() if key != "process"},
        transcript_bundle=main_compatible_transcript_ref,
        row_inputs=row_inputs,
        # Replay has no optimizer update.  This call reuses only the main
        # builder's row/reward/projection algebra; its root-only optimizer
        # witness is removed immediately below and is never replay evidence.
        update_successful=True,
    )
    document = dict(main_shape)
    document.pop("update_successful")
    document.update(
        {
            "schema": CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA,
            "mode": "captured_replay",
            "attempt_id": attempt,
            "source_main_ledger_sha256": source_main_sha,
            "source_transcript_bundle": source_transcript_ref,
            "bindings": binding_document,
            "transcript_bundle": replay_transcript_ref,
        }
    )
    _require_exact_keys(document, REPLAY_LEDGER_ROOT_KEYS, name="replay ledger")
    validate_ledger_transcript_join(ledger=document, transcript_bundle=transcript_document)
    canonical_ascii_json(document)
    return document


def validate_captured_replay_step1_ledger(
    document: Mapping[str, Any],
    *,
    source_transcript_document: Mapping[str, Any],
    transcript_document: Mapping[str, Any],
) -> None:
    """Rebuild a replay ledger and reject changed identities or derived fields."""
    _require_exact_keys(document, REPLAY_LEDGER_ROOT_KEYS, name="replay ledger")
    if document["schema"] != CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA:
        raise ValueError("unexpected captured replay ledger schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected replay ledger hash domain")
    if document["arm"] != "on" or document["mode"] != "captured_replay":
        raise ValueError("captured replay ledger requires on/captured_replay")
    if (
        type(document["step"]) is not int
        or document["step"] != 1
        or type(document["sample_count"]) is not int
        or document["sample_count"] != K4_SAMPLES
    ):
        raise ValueError("captured replay ledger must describe exactly step 1 K=4")
    if document["compared_fields"] != list(CROSS_ARM_PARITY_FIELDS):
        raise ValueError("captured replay compared_fields differ from frozen order")
    rows = document["rows"]
    if not isinstance(rows, list):
        raise TypeError("captured replay ledger rows must be a list")

    from nemo_rl.utils.strict_main_step_ledger import MAIN_STEP1_ROW_INPUT_KEYS

    rebuilt = build_captured_replay_step1_ledger(
        pair_id=document["pair_id"],
        environment=document["environment"],
        attempt_id=document["attempt_id"],
        source_main_ledger_sha256=document["source_main_ledger_sha256"],
        source_transcript_bundle=document["source_transcript_bundle"],
        source_transcript_document=source_transcript_document,
        generation=document["generation"],
        bindings=document["bindings"],
        transcript_bundle=document["transcript_bundle"],
        transcript_document=transcript_document,
        row_inputs=[{key: row[key] for key in MAIN_STEP1_ROW_INPUT_KEYS} for row in rows],
    )
    if dict(document) != rebuilt:
        raise ValueError("replay ledger contains a changed or non-derived value")
    canonical_ascii_json(document)


def validate_ledger_transcript_join(*, ledger: Mapping[str, Any], transcript_bundle: Mapping[str, Any]) -> None:
    """Close one ledger row-by-row to exact captured transport preimages."""
    validate_transcript_bundle(transcript_bundle)
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != K4_SAMPLES:
        raise ValueError("ledger must contain exactly K=4 rows")
    identity_fields = ("pair_id", "environment", "arm", "mode", "step", "sample_count")
    for key in identity_fields:
        if ledger.get(key) != transcript_bundle.get(key) or type(ledger.get(key)) is not type(
            transcript_bundle.get(key)
        ):
            raise ValueError(f"ledger/transcript {key} differs")
    if ledger.get("generation") != transcript_bundle["generation"]:
        raise ValueError("ledger/transcript generation policy differs")
    ledger_attempt = ledger.get("attempt_id")
    if ledger_attempt != transcript_bundle["attempt_id"]:
        raise ValueError("ledger/transcript attempt_id differs")
    transcript_ref = ledger.get("transcript_bundle")
    _artifact_reference(
        transcript_ref,
        name="ledger.transcript_bundle",
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
    )
    actual_bundle_sha = document_sha256(transcript_bundle, trailing_lf=False)
    if transcript_ref["sha256"] != actual_bundle_sha:
        raise ValueError("ledger transcript reference does not bind bundle bytes")
    ledger_bindings = ledger.get("bindings")
    if not isinstance(ledger_bindings, Mapping):
        raise TypeError("ledger bindings must be an object")
    for key, value in transcript_bundle["bindings"].items():
        if ledger_bindings.get(key) != value:
            raise ValueError(f"ledger/transcript binding {key} differs")
    group_ids = {row.get("shared_prefix_group_id") for row in rows}
    if len(group_ids) != 1:
        raise ValueError("ledger rows do not share one prompt-group UUID")
    group_id = next(iter(group_ids))
    if type(group_id) is not str:
        raise TypeError("ledger prompt-group UUID must be text")
    try:
        parsed_group_id = uuid.UUID(group_id)
    except (AttributeError, ValueError) as error:
        raise ValueError("ledger prompt-group identity is not a UUID") from error
    if parsed_group_id.version != 4 or str(parsed_group_id) != group_id:
        raise ValueError("ledger prompt-group identity is not canonical UUID4")
    expected_task_index = (parsed_group_id.int ^ (parsed_group_id.int >> 64)) & ((1 << 63) - 1)
    task_indices = {entry["agent_run_request"].get("_ng_task_index") for entry in transcript_bundle["entries"]}
    if task_indices != {expected_task_index}:
        raise ValueError("ledger prompt-group UUID does not close to transcript task index")
    for index, (row, entry) in enumerate(zip(rows, transcript_bundle["entries"], strict=True)):
        exact = {
            "sample_index": entry["sample_index"],
            "fixture_row_index": entry["fixture_row_index"],
            "rollout_index": entry["rollout_index"],
            "generation_seed": entry["generation_seed"],
            "request_sha256": entry["generation_request_sha256"],
            "response_sha256": entry["model_response_sha256"],
            "agent_run_request_sha256": entry["agent_run_request_sha256"],
            "derived_verifier_request_sha256": entry["derived_verifier_request_sha256"],
            "verifier_response_sha256": entry["verifier_response_sha256"],
        }
        for key, value in exact.items():
            if row.get(key) != value or type(row.get(key)) is not type(value):
                raise ValueError(f"ledger row {index} differs from transcript {key}")
        trainer_reward = _float32_round_trip(entry["raw_environment_reward"])
        if type(row.get("raw_environment_reward")) is not float or row["raw_environment_reward"] != trainer_reward:
            raise ValueError(f"ledger row {index} differs from float32 transcript raw_environment_reward")
        prompt, completion, all_tokens = model_response_token_geometry(
            entry["model_response"], name=f"transcript entry {index} model_response"
        )
        if row.get("prompt_token_ids") != prompt:
            raise ValueError(f"ledger row {index} prompt tokens differ from transcript")
        if row.get("completion_token_ids") != completion:
            raise ValueError(f"ledger row {index} completion tokens differ from transcript")
        if row.get("token_ids") != all_tokens:
            raise ValueError(f"ledger row {index} token IDs differ from transcript")
        if row.get("input_length") != len(all_tokens):
            raise ValueError(f"ledger row {index} input length differs from transcript")


def build_captured_replay_scheduler_query_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    phase: str,
    raw_output_path: str,
    raw_output_sha256: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one exact-ID ``scontrol --json`` result to normalized identity."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    query_phase = _replay_scheduler_query_phase(phase)
    raw_path = _canonical_absolute_path(raw_output_path, name="scheduler query raw output path")
    raw_sha256 = _require_digest(raw_output_sha256, name="scheduler query raw output SHA-256")
    raw = _load_lifecycle_raw_bytes(raw_path, expected_sha256=raw_sha256, maximum=1 << 20)
    normalized = _normalized_replay_scheduler_record(raw, phase=query_phase)
    if not _exact_json_equal(normalized, record):
        raise ValueError("caller scheduler record differs from normalized raw output")
    job_id = normalized["job_id"]
    scontrol = _authenticated_replay_scontrol(manifest)
    document = {
        "schema": REPLAY_SCHEDULER_QUERY_SCHEMA,
        "phase": query_phase,
        "argv": [scontrol["path"], "show", "job", "--json", job_id],
        "path": str(raw_path),
        "sha256": raw_sha256,
        "byte_count": len(raw),
        "line_count": raw.count(b"\n"),
        "status": 0,
        "normalization": copy.deepcopy(REPLAY_SCHEDULER_QUERY_NORMALIZATION),
        "records": [normalized],
        "match_count": 1,
    }
    validate_captured_replay_scheduler_query_v2(
        document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def validate_captured_replay_scheduler_query_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    """Validate and stable-load one normalized exact-ID scheduler query."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(
        document,
        REPLAY_SCHEDULER_QUERY_ROOT_KEYS,
        name="captured replay scheduler query",
    )
    if document["schema"] != REPLAY_SCHEDULER_QUERY_SCHEMA:
        raise ValueError("unexpected captured replay scheduler query schema")
    phase = _replay_scheduler_query_phase(document["phase"])
    records = document["records"]
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("scheduler query must normalize exactly one record")
    record = _validate_replay_scheduler_record(records[0], phase=phase)
    scontrol = _authenticated_replay_scontrol(manifest)
    expected_argv = [scontrol["path"], "show", "job", "--json", record["job_id"]]
    if document["argv"] != expected_argv:
        raise ValueError("scheduler query argv differs from exact-ID suffix")
    if (
        type(document["status"]) is not int
        or document["status"] != 0
        or not _exact_json_equal(document["normalization"], REPLAY_SCHEDULER_QUERY_NORMALIZATION)
        or type(document["line_count"]) is not int
        or document["line_count"] < 1
        or type(document["match_count"]) is not int
        or document["match_count"] != 1
        or type(document["byte_count"]) is not int
        or document["byte_count"] <= 1
    ):
        raise ValueError("scheduler query terminal normalization differs")
    raw_path = _canonical_absolute_path(document["path"], name="scheduler raw path")
    raw_sha256 = _require_digest(document["sha256"], name="scheduler raw SHA-256")
    raw = _load_lifecycle_raw_bytes(raw_path, expected_sha256=raw_sha256, maximum=1 << 20)
    if len(raw) != document["byte_count"] or raw.count(b"\n") != document["line_count"]:
        raise ValueError("scheduler query raw framing/count differs")
    normalized = _normalized_replay_scheduler_record(raw, phase=phase)
    if not _exact_json_equal(normalized, record):
        raise ValueError("scheduler query normalized record differs from raw output")
    canonical_ascii_json(document)


def publish_captured_replay_scheduler_query_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    validate_captured_replay_scheduler_query_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _scheduler_query_document_path(_canonical_absolute_path(document["path"], name="scheduler raw path"))
    actual = _canonical_absolute_path(str(output), name="scheduler query output")
    if actual != expected:
        raise ValueError("scheduler query document path differs from its raw output")
    return publish_evidence_document(output=actual, document=document, trailing_lf=True)


def load_captured_replay_scheduler_query_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    document, digest = load_evidence_document(path=path, expected_sha256=expected_sha256, trailing_lf=True)
    validate_captured_replay_scheduler_query_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _scheduler_query_document_path(_canonical_absolute_path(document["path"], name="scheduler raw path"))
    if _canonical_absolute_path(str(path), name="scheduler query path") != expected:
        raise ValueError("scheduler query document path differs from its raw output")
    return document, digest


def build_captured_replay_submission_receipt_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    replay_execution_manifest_path: str,
    replay_execution_manifest_sha256: str,
    scheduler_client_environment: Mapping[str, Any],
    scheduler_tools: Mapping[str, Any],
    sbatch_argv: Sequence[str],
    parsable_stdout: str,
    accepted_id_record: Mapping[str, Any],
    pre_release_scheduler_query: Mapping[str, Any],
    submitted_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Build a held login-side receipt that owns no authenticated job claim."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    manifest_ref = _artifact_reference(
        {
            "path": replay_execution_manifest_path,
            "schema": REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
            "sha256": replay_execution_manifest_sha256,
        },
        name="replay_execution_manifest",
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    _require_loaded_document_matches(manifest_ref, manifest, trailing_lf=False, name="replay execution manifest")
    tools = _validate_replay_scheduler_tools(scheduler_tools, replay_execution_manifest=manifest)
    client_environment = _validate_replay_scheduler_client_environment(
        scheduler_client_environment, replay_execution_manifest=manifest
    )
    accepted = _validate_replay_accepted_id_record(accepted_id_record, replay_execution_manifest=manifest)
    candidate = accepted["parsed_candidate_job_id"]
    _require_replay_job_disjoint_from_pair(manifest, candidate, name="candidate_job_id")
    _validate_submission_parent_precondition(manifest, candidate_job_id=candidate)
    query_ref, query = _load_replay_scheduler_query_reference(
        pre_release_scheduler_query,
        phase="PRE_RELEASE",
        expected_path=_submission_query_path(manifest, raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    nonce = _require_digest(manifest["scheduler_submission"]["nonce"], name="submission nonce")
    job_name = manifest["scheduler_submission"]["identity"]["job_name"]
    submitter_euid = manifest["scheduler_submission"]["identity"]["submitter_euid"]
    comment = f"nemo-rl-strict-captured-replay-v2:{manifest['attempt_id']}:" f"{nonce}:{manifest_ref['sha256']}"
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=candidate,
        comment=comment,
        submitter_euid=submitter_euid,
    )
    argv = _validate_replay_sbatch_argv(
        sbatch_argv,
        replay_execution_manifest=manifest,
        replay_execution_manifest_ref=manifest_ref,
        scheduler_tools=tools,
        comment=comment,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected_stdout = f"{candidate}\n"
    if parsable_stdout != expected_stdout:
        raise ValueError("sbatch stdout must be exactly candidate_job_id plus LF")
    _require_ascii(parsable_stdout, name="sbatch parsable stdout", maximum=64)
    timestamp = time.time_ns() if submitted_at_unix_ns is None else submitted_at_unix_ns
    _require_bounded_positive_int(timestamp, name="submitted_at_unix_ns", maximum=(1 << 63) - 1)
    snapshot = copy.deepcopy(manifest["replay_contract"]["source_snapshot"])
    snapshot_root = _canonical_absolute_path(snapshot["ref"]["path"], name="replay source snapshot path")
    program = manifest["replay_contract"]["program"]
    document = {
        "schema": REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        "scorer_profile": copy.deepcopy(manifest["scorer_profile"]),
        "phase": "SUBMISSION",
        "status": "held-candidate-not-in-job-authenticated",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
        "replay_execution_manifest": manifest_ref,
        "replay_source_snapshot": snapshot,
        "submission_contract": copy.deepcopy(manifest["scheduler_submission"]["contract"]),
        "slurm_export_boundary": copy.deepcopy(manifest["slurm_export_boundary"]),
        "submission_launcher": _absolute_program_reference(snapshot_root, program["submission_launcher"]),
        "job_wrapper": _absolute_program_reference(snapshot_root, program["job_wrapper"]),
        "scheduler_client_environment": client_environment,
        "scheduler_tools": tools,
        "submission_nonce": nonce,
        "job_name": job_name,
        "comment": comment,
        "submitter_euid": submitter_euid,
        "sbatch": {
            "path": tools["sbatch"]["path"],
            "sha256": tools["sbatch"]["sha256"],
            "argv": argv,
            "argv_sha256": domain_sha256("captured-replay-sbatch-argv", argv),
            "parsable_stdout": parsable_stdout,
            "parsable_stdout_sha256": hashlib.sha256(parsable_stdout.encode("ascii")).hexdigest(),
        },
        "candidate_job_id": candidate,
        "accepted_id_record": accepted,
        "pre_release_scheduler_query": query_ref,
        "submitted_at_unix_ns": timestamp,
    }
    validate_captured_replay_submission_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def validate_captured_replay_submission_receipt_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(document, REPLAY_SUBMISSION_V2_ROOT_KEYS, name="replay submission receipt")
    expected_envelope = {
        "schema": REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "phase": "SUBMISSION",
        "status": "held-candidate-not-in-job-authenticated",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(document, expected_envelope, name="submission envelope")
    manifest_ref = _artifact_reference(
        document["replay_execution_manifest"],
        name="replay_execution_manifest",
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    _require_loaded_document_matches(manifest_ref, manifest, trailing_lf=False, name="replay execution manifest")
    expected_snapshot = manifest["replay_contract"]["source_snapshot"]
    if not _exact_json_equal(document["replay_source_snapshot"], expected_snapshot):
        raise ValueError("submission replay source snapshot differs from manifest")
    if not _exact_json_equal(
        document["submission_contract"],
        manifest["scheduler_submission"]["contract"],
    ):
        raise ValueError("submission contract differs from manifest")
    if not _exact_json_equal(document["slurm_export_boundary"], manifest["slurm_export_boundary"]):
        raise ValueError("submission Slurm export boundary differs from manifest")
    snapshot_root = _canonical_absolute_path(expected_snapshot["ref"]["path"], name="replay source snapshot path")
    program = manifest["replay_contract"]["program"]
    for name in ("submission_launcher", "job_wrapper"):
        expected = _absolute_program_reference(snapshot_root, program[name])
        if not _exact_json_equal(document[name], expected):
            raise ValueError(f"submission {name} differs from manifest program")
    tools = _validate_replay_scheduler_tools(document["scheduler_tools"], replay_execution_manifest=manifest)
    _validate_replay_scheduler_client_environment(
        document["scheduler_client_environment"], replay_execution_manifest=manifest
    )
    nonce = _require_digest(document["submission_nonce"], name="submission nonce")
    if nonce != manifest["scheduler_submission"]["nonce"]:
        raise ValueError("submission nonce differs from manifest")
    expected_name = manifest["scheduler_submission"]["identity"]["job_name"]
    expected_euid = manifest["scheduler_submission"]["identity"]["submitter_euid"]
    expected_comment = (
        f"nemo-rl-strict-captured-replay-v2:{manifest['attempt_id']}:" f"{nonce}:{manifest_ref['sha256']}"
    )
    if (
        document["job_name"] != expected_name
        or document["comment"] != expected_comment
        or type(document["submitter_euid"]) is not int
        or document["submitter_euid"] != expected_euid
    ):
        raise ValueError("submission scheduler identity differs")
    accepted = _validate_replay_accepted_id_record(document["accepted_id_record"], replay_execution_manifest=manifest)
    candidate = _require_job_id(document["candidate_job_id"], name="candidate_job_id")
    _require_replay_job_disjoint_from_pair(manifest, candidate, name="candidate_job_id")
    if candidate != accepted["parsed_candidate_job_id"]:
        raise ValueError("candidate job ID differs from accepted-ID record")
    query_ref, query = _load_replay_scheduler_query_reference(
        document["pre_release_scheduler_query"],
        phase="PRE_RELEASE",
        expected_path=_submission_query_path(manifest, raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if query_ref != document["pre_release_scheduler_query"]:
        raise ValueError("pre-release scheduler query reference is noncanonical")
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=candidate,
        comment=expected_comment,
        submitter_euid=expected_euid,
    )
    sbatch = document["sbatch"]
    _require_exact_keys(sbatch, SBATCH_KEYS, name="submission sbatch")
    if sbatch["path"] != tools["sbatch"]["path"] or sbatch["sha256"] != tools["sbatch"]["sha256"]:
        raise ValueError("submission sbatch tool differs")
    argv = _validate_replay_sbatch_argv(
        sbatch["argv"],
        replay_execution_manifest=manifest,
        replay_execution_manifest_ref=manifest_ref,
        scheduler_tools=tools,
        comment=expected_comment,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if sbatch["argv_sha256"] != domain_sha256("captured-replay-sbatch-argv", argv):
        raise ValueError("submission sbatch argv digest does not close")
    stdout = sbatch["parsable_stdout"]
    if (
        stdout != f"{candidate}\n"
        or sbatch["parsable_stdout_sha256"] != hashlib.sha256(stdout.encode("ascii")).hexdigest()
    ):
        raise ValueError("submission sbatch stdout/candidate binding differs")
    _require_bounded_positive_int(
        document["submitted_at_unix_ns"],
        name="submitted_at_unix_ns",
        maximum=(1 << 63) - 1,
    )
    receipt_bytes(document)


def publish_captured_replay_submission_receipt_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_submission_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _canonical_absolute_path(
        manifest["scheduler_submission"]["receipt"]["path"],
        name="declared replay submission receipt path",
    )
    actual = _canonical_absolute_path(str(output), name="replay submission output")
    if actual != expected:
        raise ValueError("replay submission receipt output path differs from manifest")
    return publish_evidence_document(output=actual, document=document, trailing_lf=True)


def load_captured_replay_submission_receipt_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected_path = _canonical_absolute_path(
        manifest["scheduler_submission"]["receipt"]["path"],
        name="declared replay submission receipt path",
    )
    actual_path = _canonical_absolute_path(str(path), name="submission receipt path")
    if actual_path != expected_path:
        raise ValueError("replay submission receipt path differs from manifest")
    document, digest = load_evidence_document(path=actual_path, expected_sha256=expected_sha256, trailing_lf=True)
    validate_captured_replay_submission_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document, digest


def build_captured_replay_pre_receipt_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    submission_receipt: Mapping[str, Any],
    authenticated_job_id: str,
    job: Mapping[str, Any],
    pre_scheduler_query: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote the held candidate only after exact in-job scheduler authentication."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_submission_receipt_v2(
        submission_receipt,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    candidate = _require_job_id(submission_receipt["candidate_job_id"], name="candidate_job_id")
    authenticated = _require_job_id(authenticated_job_id, name="authenticated_job_id")
    _require_replay_job_disjoint_from_pair(manifest, candidate, name="candidate_job_id")
    _require_replay_job_disjoint_from_pair(manifest, authenticated, name="authenticated_job_id")
    if candidate != authenticated:
        raise ValueError("candidate and authenticated replay job IDs differ")
    submission_ref = _receipt_reference_for_document(
        path=manifest["scheduler_submission"]["receipt"]["path"],
        schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        document=submission_receipt,
    )
    _require_loaded_document_matches(
        submission_ref,
        submission_receipt,
        trailing_lf=True,
        name="replay submission receipt",
    )
    query_ref, query = _load_replay_scheduler_query_reference(
        pre_scheduler_query,
        phase="PRE",
        expected_path=_job_query_path(manifest, authenticated, "PRE", raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=authenticated,
        comment=submission_receipt["comment"],
        submitter_euid=submission_receipt["submitter_euid"],
    )
    job_document = _validate_job(job, replay_execution_manifest=manifest)
    pre_path = _replay_job_receipt_path(manifest, authenticated, phase="PRE")
    execution_source_root = _canonical_absolute_path(
        manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        name="authenticated replay execution source root",
    )
    document = {
        "schema": REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        "scorer_profile": copy.deepcopy(manifest["scorer_profile"]),
        "phase": "PRE",
        "status": "authenticated-pre",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
        "replay_execution_manifest": copy.deepcopy(submission_receipt["replay_execution_manifest"]),
        "submission_receipt": submission_ref,
        "candidate_job_id": candidate,
        "authenticated_job_id": authenticated,
        "job": job_document,
        "static_boundary": _replay_static_boundary(manifest),
        "pre_scheduler_query": query_ref,
        "output_precondition": {
            "path": manifest["artifacts"]["outputs"]["directory"]["path"],
            "mode": "0700",
            "status": "absent",
        },
        "runtime_attestation_contract": copy.deepcopy(manifest["runtime_attestation_requirements"]),
        "execution_source_root": str(execution_source_root),
        "driver": {
            "entrypoint": manifest["replay_contract"]["program"]["entrypoint"]["path"],
            "invocation": "python-isolated-no-bytecode",
            "pre_receipt_path": str(pre_path),
        },
        "post_verified": False,
    }
    validate_captured_replay_pre_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def validate_captured_replay_pre_receipt_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_submission_receipt_v2(
        submission_receipt,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(document, REPLAY_PRE_V2_ROOT_KEYS, name="replay PRE receipt")
    expected_envelope = {
        "schema": REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "phase": "PRE",
        "status": "authenticated-pre",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(document, expected_envelope, name="PRE envelope")
    if not _exact_json_equal(
        document["replay_execution_manifest"],
        submission_receipt["replay_execution_manifest"],
    ):
        raise ValueError("PRE replay execution manifest reference differs")
    submission_ref = _artifact_reference(
        document["submission_receipt"],
        name="PRE submission receipt",
        expected_schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    )
    expected_submission_ref = _receipt_reference_for_document(
        path=manifest["scheduler_submission"]["receipt"]["path"],
        schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        document=submission_receipt,
    )
    if submission_ref != expected_submission_ref:
        raise ValueError("PRE submission receipt reference differs")
    _require_loaded_document_matches(
        submission_ref,
        submission_receipt,
        trailing_lf=True,
        name="replay submission receipt",
    )
    candidate = _require_job_id(document["candidate_job_id"], name="candidate_job_id")
    authenticated = _require_job_id(document["authenticated_job_id"], name="authenticated_job_id")
    _require_replay_job_disjoint_from_pair(manifest, candidate, name="candidate_job_id")
    _require_replay_job_disjoint_from_pair(manifest, authenticated, name="authenticated_job_id")
    if candidate != authenticated or candidate != submission_receipt["candidate_job_id"]:
        raise ValueError("PRE candidate/authenticated job identity differs")
    _validate_job(document["job"], replay_execution_manifest=manifest)
    expected_static = _replay_static_boundary(manifest)
    _require_exact_keys(
        document["static_boundary"],
        REPLAY_STATIC_BOUNDARY_V2_KEYS,
        name="PRE static boundary",
    )
    if not _exact_json_equal(document["static_boundary"], expected_static):
        raise ValueError("PRE static boundary differs from replay manifest")
    query_ref, query = _load_replay_scheduler_query_reference(
        document["pre_scheduler_query"],
        phase="PRE",
        expected_path=_job_query_path(manifest, authenticated, "PRE", raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if query_ref != document["pre_scheduler_query"]:
        raise ValueError("PRE scheduler query reference is noncanonical")
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=authenticated,
        comment=submission_receipt["comment"],
        submitter_euid=submission_receipt["submitter_euid"],
    )
    expected_precondition = {
        "path": manifest["artifacts"]["outputs"]["directory"]["path"],
        "mode": "0700",
        "status": "absent",
    }
    _require_exact_keys(
        document["output_precondition"],
        REPLAY_OUTPUT_PRECONDITION_KEYS,
        name="PRE output precondition",
    )
    if document["output_precondition"] != expected_precondition:
        raise ValueError("PRE output precondition differs from manifest")
    if not _exact_json_equal(
        document["runtime_attestation_contract"],
        manifest["runtime_attestation_requirements"],
    ):
        raise ValueError("PRE runtime attestation contract differs from manifest")
    source_root = _canonical_absolute_path(
        manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        name="authenticated replay execution source root",
    )
    if document["execution_source_root"] != str(source_root):
        raise ValueError("PRE execution source root differs from manifest")
    pre_path = _replay_job_receipt_path(manifest, authenticated, phase="PRE")
    expected_driver = {
        "entrypoint": manifest["replay_contract"]["program"]["entrypoint"]["path"],
        "invocation": "python-isolated-no-bytecode",
        "pre_receipt_path": str(pre_path),
    }
    if not _exact_json_equal(document["driver"], expected_driver):
        raise ValueError("PRE driver boundary differs")
    if document["post_verified"] is not False:
        raise ValueError("PRE must not claim post verification")
    receipt_bytes(document)


def publish_captured_replay_pre_receipt_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    validate_captured_replay_pre_receipt_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        submission_receipt=submission_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _replay_job_receipt_path(
        replay_execution_manifest,
        document["authenticated_job_id"],
        phase="PRE",
    )
    actual = _canonical_absolute_path(str(output), name="replay PRE output")
    if actual != expected:
        raise ValueError("replay PRE receipt path differs")
    return publish_evidence_document(output=actual, document=document, trailing_lf=True)


def load_captured_replay_pre_receipt_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    candidate = _require_job_id(submission_receipt["candidate_job_id"], name="candidate_job_id")
    expected_path = _replay_job_receipt_path(manifest, candidate, phase="PRE")
    actual_path = _canonical_absolute_path(str(path), name="replay PRE path")
    if actual_path != expected_path:
        raise ValueError("replay PRE receipt path differs")
    document, digest = load_evidence_document(path=actual_path, expected_sha256=expected_sha256, trailing_lf=True)
    validate_captured_replay_pre_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document, digest


def build_captured_replay_exit_receipt_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    post_scheduler_query: Mapping[str, Any],
    driver_exit_code: int,
    hardware: Mapping[str, Any],
    scheduler_device_environment: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    driver_process: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build successful terminal replay evidence from authenticated descendants."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_pre_receipt_v2(
        pre_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if type(driver_exit_code) is not int or driver_exit_code != 0:
        raise ValueError("successful replay EXIT requires exact integer exit code zero")
    authenticated = _require_job_id(pre_receipt["authenticated_job_id"], name="authenticated_job_id")
    pre_path = _replay_job_receipt_path(manifest, authenticated, phase="PRE")
    pre_ref = _receipt_reference_for_document(
        path=str(pre_path),
        schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        document=pre_receipt,
    )
    _require_loaded_document_matches(pre_ref, pre_receipt, trailing_lf=True, name="replay PRE receipt")
    query_ref, query = _load_replay_scheduler_query_reference(
        post_scheduler_query,
        phase="POST",
        expected_path=_job_query_path(manifest, authenticated, "POST", raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=authenticated,
        comment=submission_receipt["comment"],
        submitter_euid=submission_receipt["submitter_euid"],
    )
    wrapper_device_environment = _validate_scheduler_device_environment(scheduler_device_environment)
    driver_device_environment = _validate_scheduler_device_environment(driver_scheduler_device_environment)
    if not _exact_json_equal(driver_device_environment, wrapper_device_environment):
        raise ValueError("driver scheduler device environment differs from wrapper observation")
    process = _validate_process(driver_process)
    output_refs = _validate_captured_replay_outputs(
        outputs,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_job_id=authenticated,
        driver_process=process,
        driver_scheduler_device_environment=driver_device_environment,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    runtime_attestation = _replay_runtime_attestation(manifest, output_refs)
    document = {
        "schema": REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        "scorer_profile": copy.deepcopy(manifest["scorer_profile"]),
        "phase": "EXIT",
        "status": "complete",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
        "replay_execution_manifest": copy.deepcopy(submission_receipt["replay_execution_manifest"]),
        "submission_receipt": copy.deepcopy(pre_receipt["submission_receipt"]),
        "candidate_job_id": pre_receipt["candidate_job_id"],
        "authenticated_job_id": authenticated,
        "job": copy.deepcopy(pre_receipt["job"]),
        "static_boundary": copy.deepcopy(pre_receipt["static_boundary"]),
        "pre_receipt": pre_ref,
        "post_scheduler_query": query_ref,
        "driver_exit_code": 0,
        "hardware": _validate_hardware(hardware, replay_execution_manifest=manifest),
        "scheduler_device_environment": wrapper_device_environment,
        "driver_scheduler_device_environment": driver_device_environment,
        "driver_process": process,
        "runtime_attestation": runtime_attestation,
        "outputs": output_refs,
        "post_verified": True,
    }
    validate_captured_replay_exit_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def validate_captured_replay_exit_receipt_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_pre_receipt_v2(
        pre_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(document, REPLAY_EXIT_V2_ROOT_KEYS, name="replay EXIT receipt")
    expected_envelope = {
        "schema": REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "phase": "EXIT",
        "status": "complete",
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(document, expected_envelope, name="EXIT envelope")
    for name in (
        "replay_execution_manifest",
        "submission_receipt",
        "candidate_job_id",
        "authenticated_job_id",
        "job",
        "static_boundary",
    ):
        if not _exact_json_equal(document[name], pre_receipt[name]):
            raise ValueError(f"EXIT {name} differs from authenticated PRE")
    authenticated = _require_job_id(document["authenticated_job_id"], name="authenticated_job_id")
    if document["candidate_job_id"] != authenticated:
        raise ValueError("EXIT candidate/authenticated job IDs differ")
    _validate_job(document["job"], replay_execution_manifest=manifest)
    pre_ref = _artifact_reference(
        document["pre_receipt"],
        name="EXIT PRE receipt",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    expected_pre_ref = _receipt_reference_for_document(
        path=str(_replay_job_receipt_path(manifest, authenticated, phase="PRE")),
        schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        document=pre_receipt,
    )
    if pre_ref != expected_pre_ref:
        raise ValueError("EXIT PRE receipt reference differs")
    _require_loaded_document_matches(pre_ref, pre_receipt, trailing_lf=True, name="replay PRE receipt")
    query_ref, query = _load_replay_scheduler_query_reference(
        document["post_scheduler_query"],
        phase="POST",
        expected_path=_job_query_path(manifest, authenticated, "POST", raw=False),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if query_ref != document["post_scheduler_query"]:
        raise ValueError("EXIT POST scheduler query reference is noncanonical")
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=authenticated,
        comment=submission_receipt["comment"],
        submitter_euid=submission_receipt["submitter_euid"],
    )
    if type(document["driver_exit_code"]) is not int or document["driver_exit_code"] != 0:
        raise ValueError("successful replay EXIT requires exact integer exit code zero")
    _validate_hardware(document["hardware"], replay_execution_manifest=manifest)
    wrapper_device_environment = _validate_scheduler_device_environment(document["scheduler_device_environment"])
    driver_device_environment = _validate_scheduler_device_environment(document["driver_scheduler_device_environment"])
    if not _exact_json_equal(driver_device_environment, wrapper_device_environment):
        raise ValueError("driver scheduler device environment differs from wrapper observation")
    process = _validate_process(document["driver_process"])
    output_refs = _validate_captured_replay_outputs(
        document["outputs"],
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_job_id=authenticated,
        driver_process=process,
        driver_scheduler_device_environment=driver_device_environment,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected_runtime = _replay_runtime_attestation(manifest, output_refs)
    _require_exact_keys(
        document["runtime_attestation"],
        REPLAY_RUNTIME_ATTESTATION_V2_KEYS,
        name="EXIT runtime attestation",
    )
    if not _exact_json_equal(document["runtime_attestation"], expected_runtime):
        raise ValueError("EXIT runtime attestation differs from validated outputs")
    if (
        type(document["runtime_attestation"]["original_process_reaped"]) is not bool
        or document["runtime_attestation"]["original_process_reaped"] is not True
    ):
        raise ValueError("EXIT runtime attestation must prove original process reaping")
    if document["post_verified"] is not True:
        raise ValueError("successful replay EXIT must be post-verified")
    receipt_bytes(document)


def publish_captured_replay_exit_receipt_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    validate_captured_replay_exit_receipt_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _replay_job_receipt_path(
        replay_execution_manifest,
        document["authenticated_job_id"],
        phase="EXIT",
    )
    actual = _canonical_absolute_path(str(output), name="replay EXIT output")
    if actual != expected:
        raise ValueError("replay EXIT receipt path differs")
    return publish_evidence_document(output=actual, document=document, trailing_lf=True)


def load_captured_replay_exit_receipt_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    authenticated = _require_job_id(pre_receipt["authenticated_job_id"], name="authenticated_job_id")
    expected_path = _replay_job_receipt_path(manifest, authenticated, phase="EXIT")
    actual_path = _canonical_absolute_path(str(path), name="replay EXIT path")
    if actual_path != expected_path:
        raise ValueError("replay EXIT receipt path differs")
    document, digest = load_evidence_document(path=actual_path, expected_sha256=expected_sha256, trailing_lf=True)
    validate_captured_replay_exit_receipt_v2(
        document,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document, digest


def build_captured_replay_evidence_index_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the non-authoritative terminal index from the validated receipt DAG."""
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_exit_receipt_v2(
        exit_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    authenticated = exit_receipt["authenticated_job_id"]
    document = {
        "schema": REPLAY_POST_INDEX_V2_SCHEMA,
        "scorer_profile": copy.deepcopy(manifest["scorer_profile"]),
        "original_process_reaped": True,
        "profile_id": manifest["scorer_profile"]["profile_id"],
        "hash_domain": HASH_DOMAIN,
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
        "replay_execution_manifest": copy.deepcopy(submission_receipt["replay_execution_manifest"]),
        "pair_submission_receipt": copy.deepcopy(manifest["pair"]["submission_receipt"]),
        "submission_receipt": copy.deepcopy(pre_receipt["submission_receipt"]),
        "pre_receipt": copy.deepcopy(exit_receipt["pre_receipt"]),
        "exit_receipt": _receipt_reference_for_document(
            path=str(_replay_job_receipt_path(manifest, authenticated, phase="EXIT")),
            schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
            document=exit_receipt,
        ),
        "source_capture": copy.deepcopy(manifest["source_capture"]),
        "outputs": copy.deepcopy(exit_receipt["outputs"]),
        "identity": {
            "candidate_job_id": exit_receipt["candidate_job_id"],
            "authenticated_job_id": authenticated,
            "driver_process": copy.deepcopy(exit_receipt["driver_process"]),
            "run_id": replay_run_id(
                environment=manifest["environment"],
                pair_id=manifest["pair_id"],
                attempt_id=manifest["attempt_id"],
            ),
        },
    }
    validate_captured_replay_evidence_index_v2(
        document,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        exit_receipt=exit_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def validate_captured_replay_evidence_index_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_exit_receipt_v2(
        exit_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(document, REPLAY_POST_INDEX_V2_ROOT_KEYS, name="replay evidence index")
    expected_envelope = {
        "schema": REPLAY_POST_INDEX_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "original_process_reaped": True,
        "profile_id": manifest["scorer_profile"]["profile_id"],
        "hash_domain": HASH_DOMAIN,
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(document, expected_envelope, name="post index envelope")
    if type(document["original_process_reaped"]) is not bool or document["original_process_reaped"] is not True:
        raise ValueError("post index must prove original process reaping")
    expected_refs = {
        "replay_execution_manifest": submission_receipt["replay_execution_manifest"],
        "pair_submission_receipt": manifest["pair"]["submission_receipt"],
        "submission_receipt": pre_receipt["submission_receipt"],
        "pre_receipt": exit_receipt["pre_receipt"],
        "exit_receipt": _receipt_reference_for_document(
            path=str(_replay_job_receipt_path(manifest, exit_receipt["authenticated_job_id"], phase="EXIT")),
            schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
            document=exit_receipt,
        ),
    }
    for name, expected in expected_refs.items():
        if not _exact_json_equal(document[name], expected):
            raise ValueError(f"post index {name} differs")
    _require_loaded_document_matches(
        document["exit_receipt"],
        exit_receipt,
        trailing_lf=True,
        name="replay EXIT receipt",
    )
    if not _exact_json_equal(document["source_capture"], manifest["source_capture"]):
        raise ValueError("post index source capture differs from replay manifest")
    if not _exact_json_equal(document["outputs"], exit_receipt["outputs"]):
        raise ValueError("post index outputs differ from EXIT")
    _validate_captured_replay_outputs(
        document["outputs"],
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_job_id=exit_receipt["authenticated_job_id"],
        driver_process=exit_receipt["driver_process"],
        driver_scheduler_device_environment=exit_receipt["driver_scheduler_device_environment"],
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(document["identity"], REPLAY_POST_IDENTITY_KEYS, name="post index identity")
    expected_identity = {
        "candidate_job_id": exit_receipt["candidate_job_id"],
        "authenticated_job_id": exit_receipt["authenticated_job_id"],
        "driver_process": _validate_process(exit_receipt["driver_process"]),
        "run_id": replay_run_id(
            environment=manifest["environment"],
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
    }
    if not _exact_json_equal(document["identity"], expected_identity):
        raise ValueError("post index identity differs")
    canonical_ascii_json(document)


def publish_captured_replay_evidence_index_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    validate_captured_replay_evidence_index_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        exit_receipt=exit_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected = _replay_post_index_path(replay_execution_manifest)
    actual = _canonical_absolute_path(str(output), name="replay evidence index output")
    if actual != expected:
        raise ValueError("replay evidence index path differs")
    return publish_evidence_document(output=actual, document=document, trailing_lf=False)


def load_captured_replay_evidence_index_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    expected_path = _replay_post_index_path(replay_execution_manifest)
    actual_path = _canonical_absolute_path(str(path), name="replay evidence index path")
    if actual_path != expected_path:
        raise ValueError("replay evidence index path differs")
    document, digest = load_evidence_document(path=actual_path, expected_sha256=expected_sha256, trailing_lf=False)
    validate_captured_replay_evidence_index_v2(
        document,
        replay_execution_manifest=replay_execution_manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        exit_receipt=exit_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document, digest


def _final_receipt_paths_v2(
    replay_execution_manifest: Mapping[str, Any],
    *,
    authenticated_job_id: str,
) -> dict[str, str]:
    """Derive PRE/EXIT/FINAL only from M4 and authenticated job identity."""
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    pair_id = _require_safe_id(manifest.get("pair_id"), name="pair_id", maximum=64)
    attempt_id = _require_attempt(manifest.get("attempt_id"))
    job_id = _require_job_id(authenticated_job_id, name="authenticated_job_id")
    pair = _strict_json_object(manifest.get("pair"), name="replay manifest pair")
    pair_manifest = _artifact_reference(
        pair.get("manifest"),
        name="replay Pair manifest",
        expected_schema=PAIR_MANIFEST_SCHEMA,
    )
    pair_manifest_path = Path(pair_manifest["path"])
    if pair_manifest_path.name != "PAIR_MANIFEST.json":
        raise ValueError("replay Pair manifest path is noncanonical")
    results_root = pair_manifest_path.parent
    receipt_root = (
        results_root / "captured_replay" / "replay_job_state" / pair_id / attempt_id / f"{job_id}-0" / "receipts"
    )
    return {
        "pre": str(receipt_root / "PRE.json"),
        "exit": str(receipt_root / "EXIT.json"),
        "final": str(receipt_root / "FINAL.json"),
    }


def _canonical_replay_manifest_path_v2(
    replay_execution_manifest: Mapping[str, Any],
) -> str:
    """Derive the sole admitted M4 publication path from its Pair anchor."""
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    pair_id = _require_safe_id(manifest.get("pair_id"), name="pair_id", maximum=64)
    attempt_id = _require_attempt(manifest.get("attempt_id"))
    pair = _strict_json_object(manifest.get("pair"), name="replay manifest pair")
    pair_manifest = _artifact_reference(
        pair.get("manifest"),
        name="replay Pair manifest",
        expected_schema=PAIR_MANIFEST_SCHEMA,
    )
    pair_manifest_path = _canonical_absolute_path(
        pair_manifest["path"],
        name="replay Pair manifest path",
    )
    if pair_manifest_path.name != "PAIR_MANIFEST.json":
        raise ValueError("replay Pair manifest path is noncanonical")
    return str(pair_manifest_path.parent / "captured_replay" / "manifests" / pair_id / f"{attempt_id}.json")


def _sealed_result_payloads_v2(
    verified_result: Any,
    *,
    replay_execution_manifest: Mapping[str, Any],
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Project verifier-retained inventory/member bytes without pathname I/O."""
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )
    from nemo_rl.utils.strict_captured_replay_seal_v2 import (
        RESULT_INVENTORY_V2_FILENAME,
        RESULT_INVENTORY_V2_SCHEMA,
        snapshot_verified_sealed_result_v2,
    )

    profile = get_strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    projection = snapshot_verified_sealed_result_v2(verified_result)
    if type(projection) is not dict or set(projection) != {
        "environment",
        "profile_id",
        "result_root",
        "inventory",
        "members",
    }:
        raise TypeError("sealed-result projection shape differs")
    result_root = str(
        _canonical_absolute_path(
            projection["result_root"],
            name="sealed result root",
        )
    )
    expected_root = str(
        _canonical_absolute_path(
            replay_execution_manifest["artifacts"]["outputs"]["directory"]["path"],
            name="manifest result root",
        )
    )
    if (
        projection["environment"] != expected_environment
        or type(projection["environment"]) is not str
        or projection["profile_id"] != expected_profile_id
        or type(projection["profile_id"]) is not str
        or result_root != expected_root
    ):
        raise ValueError("sealed-result projection identity differs from M4/profile")
    inventory = projection["inventory"]
    if type(inventory) is not dict or set(inventory) != {
        "path",
        "schema",
        "sha256",
        "raw",
    }:
        raise TypeError("sealed result inventory projection differs")
    expected_inventory_path = f"{result_root}/{RESULT_INVENTORY_V2_FILENAME}"
    inventory_sha256 = _require_digest(
        inventory.get("sha256"),
        name="sealed result inventory SHA-256",
    )
    inventory_raw = inventory.get("raw")
    if (
        inventory.get("path") != expected_inventory_path
        or type(inventory.get("path")) is not str
        or inventory.get("schema") != RESULT_INVENTORY_V2_SCHEMA
        or type(inventory_raw) is not bytes
    ):
        raise ValueError("sealed result inventory projection identity differs")
    inventory_document, _ = decode_evidence_document_bytes(
        raw=inventory_raw,
        expected_sha256=inventory_sha256,
        trailing_lf=False,
    )
    if (
        inventory_document.get("schema") != RESULT_INVENTORY_V2_SCHEMA
        or inventory_document.get("root") != result_root
        or inventory_document.get("environment") != expected_environment
        or inventory_document.get("profile_id") != expected_profile_id
    ):
        raise ValueError("sealed result inventory bytes differ from M4/profile")
    members = projection["members"]
    if type(members) is not tuple or len(members) != len(profile.result_files):
        raise TypeError("sealed result member projection differs")
    payloads: dict[str, bytes] = {}
    for item, relative, schema in zip(
        members,
        profile.result_files,
        profile.result_file_schemas,
        strict=True,
    ):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != relative
            or type(item[1]) is not bytes
            or relative in payloads
        ):
            raise TypeError(f"sealed result member projection differs for {relative}")
        document, _ = decode_evidence_document_bytes(
            raw=item[1],
            expected_sha256=hashlib.sha256(item[1]).hexdigest(),
            trailing_lf=False,
        )
        if document.get("schema") != schema:
            raise ValueError(f"sealed result member schema differs for {relative}")
        payloads[relative] = bytes(item[1])
    return projection, payloads, inventory_document


def _final_receipt_reference(
    *,
    path: str,
    schema: str,
    document: Mapping[str, Any],
    trailing_lf: bool,
) -> dict[str, str]:
    return {
        "path": str(_canonical_absolute_path(path, name="artifact path")),
        "schema": schema,
        "sha256": document_sha256(document, trailing_lf=trailing_lf),
    }


def _derive_result_final_receipt_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
    verified_result: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    submission = _strict_json_object(
        submission_receipt,
        name="replay submission receipt",
    )
    pre = _strict_json_object(pre_receipt, name="replay PRE receipt")
    exit_document = _strict_json_object(exit_receipt, name="replay EXIT receipt")
    supplied_index = _strict_json_object(
        evidence_index,
        name="replay evidence index",
    )
    _require_exact_keys(
        submission,
        REPLAY_SUBMISSION_V2_ROOT_KEYS,
        name="replay submission receipt",
    )
    _require_exact_keys(pre, REPLAY_PRE_V2_ROOT_KEYS, name="replay PRE receipt")
    _require_exact_keys(
        exit_document,
        REPLAY_EXIT_V2_ROOT_KEYS,
        name="replay EXIT receipt",
    )
    _require_exact_keys(
        supplied_index,
        REPLAY_POST_INDEX_V2_ROOT_KEYS,
        name="replay evidence index",
    )
    projection, payloads, _ = _sealed_result_payloads_v2(
        verified_result,
        replay_execution_manifest=manifest,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    captured_index_raw = payloads["evidence-index.json"]
    captured_index, captured_index_sha256 = decode_evidence_document_bytes(
        raw=captured_index_raw,
        expected_sha256=hashlib.sha256(captured_index_raw).hexdigest(),
        trailing_lf=False,
    )
    if not _exact_json_equal(captured_index, supplied_index):
        raise ValueError("in-memory evidence index differs from verifier-retained bytes")
    if captured_index.get("schema") != REPLAY_POST_INDEX_V2_SCHEMA:
        raise ValueError("verifier-retained evidence index schema differs")

    manifest_ref = _artifact_reference(
        submission.get("replay_execution_manifest"),
        name="submission replay execution manifest",
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    if manifest_ref["path"] != _canonical_replay_manifest_path_v2(manifest) or manifest_ref[
        "sha256"
    ] != document_sha256(manifest, trailing_lf=False):
        raise ValueError("submission manifest reference differs from M4 bytes")
    declared_submission = manifest["scheduler_submission"]["receipt"]
    submission_ref = _artifact_reference(
        pre.get("submission_receipt"),
        name="PRE submission receipt",
        expected_schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    )
    if submission_ref["path"] != declared_submission["path"] or submission_ref["sha256"] != document_sha256(
        submission, trailing_lf=True
    ):
        raise ValueError("PRE submission receipt reference differs from S5 bytes")
    candidate_job_id = _require_job_id(
        submission.get("candidate_job_id"),
        name="submission candidate_job_id",
    )
    authenticated_job_id = _require_job_id(
        pre.get("authenticated_job_id"),
        name="PRE authenticated_job_id",
    )
    if (
        pre.get("candidate_job_id") != candidate_job_id
        or authenticated_job_id != candidate_job_id
        or exit_document.get("candidate_job_id") != candidate_job_id
        or exit_document.get("authenticated_job_id") != authenticated_job_id
    ):
        raise ValueError("candidate/authenticated replay job identity differs")
    receipt_paths = _final_receipt_paths_v2(
        manifest,
        authenticated_job_id=authenticated_job_id,
    )
    pre_ref = _artifact_reference(
        exit_document.get("pre_receipt"),
        name="EXIT PRE receipt",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    if pre_ref["path"] != receipt_paths["pre"] or pre_ref["sha256"] != document_sha256(pre, trailing_lf=True):
        raise ValueError("EXIT PRE receipt reference differs from PRE3 bytes")
    exit_ref = _artifact_reference(
        captured_index.get("exit_receipt"),
        name="index EXIT receipt",
        expected_schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    )
    if exit_ref["path"] != receipt_paths["exit"] or exit_ref["sha256"] != document_sha256(
        exit_document, trailing_lf=True
    ):
        raise ValueError("index EXIT receipt reference differs from EXIT6 bytes")
    expected_envelope = {
        "pair_id": manifest["pair_id"],
        "environment": expected_environment,
        "scorer_profile": manifest["scorer_profile"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    for name, document in (
        ("submission", submission),
        ("PRE", pre),
        ("EXIT", exit_document),
        ("index", captured_index),
    ):
        _require_exact_projection(document, expected_envelope, name=name)
    if (
        submission.get("schema") != REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA
        or submission.get("phase") != "SUBMISSION"
        or pre.get("schema") != REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA
        or pre.get("phase") != "PRE"
        or exit_document.get("schema") != REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA
        or exit_document.get("phase") != "EXIT"
        or exit_document.get("post_verified") is not True
        or exit_document.get("driver_exit_code") != 0
        or type(exit_document.get("driver_exit_code")) is not int
    ):
        raise ValueError("replay lifecycle phase/status differs")
    driver_process = _validate_process(exit_document.get("driver_process"))
    index_identity = _strict_json_object(
        captured_index.get("identity"),
        name="replay index identity",
    )
    _require_exact_keys(
        index_identity,
        REPLAY_POST_IDENTITY_KEYS,
        name="replay index identity",
    )
    if not _exact_json_equal(
        index_identity,
        {
            "candidate_job_id": candidate_job_id,
            "authenticated_job_id": authenticated_job_id,
            "driver_process": driver_process,
            "run_id": replay_run_id(
                environment=expected_environment,
                pair_id=manifest["pair_id"],
                attempt_id=manifest["attempt_id"],
            ),
        },
    ):
        raise ValueError("replay index process/job/run identity differs")
    runtime_attestation = _strict_json_object(
        exit_document.get("runtime_attestation"),
        name="EXIT runtime attestation",
    )
    if (
        runtime_attestation.get("original_process_reaped") is not True
        or captured_index.get("original_process_reaped") is not True
    ):
        raise ValueError("terminal replay result does not prove scorer reaping")
    if not _exact_json_equal(captured_index.get("outputs"), exit_document.get("outputs")):
        raise ValueError("index outputs differ from EXIT outputs")
    result_root = projection["result_root"]
    inventory = projection["inventory"]
    declared_index = manifest["artifacts"]["outputs"]["evidence_index"]
    evidence_index_ref = {
        "path": declared_index["path"],
        "schema": REPLAY_POST_INDEX_V2_SCHEMA,
        "sha256": captured_index_sha256,
    }
    if evidence_index_ref["path"] != f"{result_root}/evidence-index.json":
        raise ValueError("evidence index path differs from sealed result root")
    inventory_ref = {
        "path": inventory["path"],
        "schema": inventory["schema"],
        "sha256": inventory["sha256"],
    }
    final_document = {
        "schema": REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "phase": "FINAL",
        "status": "complete",
        **copy.deepcopy(expected_envelope),
        "candidate_job_id": candidate_job_id,
        "authenticated_job_id": authenticated_job_id,
        "driver_process": driver_process,
        "original_process_reaped": True,
        "replay_execution_manifest": manifest_ref,
        "submission_receipt": submission_ref,
        "pre_receipt": pre_ref,
        "exit_receipt": exit_ref,
        "evidence_index": evidence_index_ref,
        "result": {"root": result_root, "inventory": inventory_ref},
    }
    _require_exact_keys(
        final_document,
        REPLAY_RESULT_FINAL_V2_ROOT_KEYS,
        name="result FINAL receipt",
    )
    receipt_bytes(final_document)
    return final_document, receipt_paths["final"]


def _validate_retained_scorer_call_index_v2(
    payload_roster: tuple[tuple[str, bytes], ...],
    *,
    expected_sha256: str,
    expected_receipt_root: Path,
    expected_bootstrap_root: Path,
    expected_bootstrap_sha256: str,
    expected_pair_id: str,
    expected_job_id: str,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    """Validate the profile-specific scorer graph from retained bytes only."""
    from nemo_rl.environments.strict_gym_child_runtime_v2 import (
        validate_finalized_format_verification_call_index_payloads,
        validate_finalized_reasoning_score_call_index_payloads,
    )

    if _RESULT_PROFILE_IDS_V2.get(expected_environment) != expected_profile_id:
        raise ValueError("retained scorer differs from outer profile authority")
    common_arguments = {
        "expected_sha256": expected_sha256,
        "expected_receipt_root": expected_receipt_root,
        "expected_bootstrap_root": expected_bootstrap_root,
        "expected_bootstrap_sha256": expected_bootstrap_sha256,
        "expected_pair_id": expected_pair_id,
        "expected_job_id": expected_job_id,
    }
    if expected_environment == "reasoning_gym":
        document, digest = validate_finalized_reasoning_score_call_index_payloads(
            payload_roster,
            **common_arguments,
        )
        terminal = _strict_json_object(document, name="retained reasoning scorer terminal")
        if (
            terminal.get("schema") != REASONING_SCORE_CALL_INDEX_SCHEMA
            or terminal.get("environment") != "reasoning_gym"
            or "profile_id" in terminal
        ):
            raise ValueError("reasoning scorer terminal differs from outer profile authority")
        return terminal, digest
    if expected_environment in {"citation", "freeform"}:
        document, digest = validate_finalized_format_verification_call_index_payloads(
            payload_roster,
            **common_arguments,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
        terminal = _strict_json_object(document, name="retained format scorer terminal")
        if (
            terminal.get("schema") != FORMAT_VERIFICATION_CALL_INDEX_SCHEMA
            or terminal.get("environment") != expected_environment
            or terminal.get("profile_id") != expected_profile_id
        ):
            raise ValueError("format scorer terminal differs from outer profile authority")
        return terminal, digest
    raise ValueError("retained scorer profile dispatch is unsupported")


def _project_authenticated_result_samples_v2(
    transcript: Mapping[str, Any],
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> list[dict[str, Any]]:
    """Project the fixed sample shape without weakening profile semantics."""
    if _RESULT_PROFILE_IDS_V2.get(expected_environment) != expected_profile_id:
        raise ValueError("sample projection differs from outer profile authority")
    entries = transcript.get("entries")
    if type(entries) is not list or len(entries) != K4_SAMPLES:
        raise ValueError("authenticated result semantic projection is not exact K=4")
    samples: list[dict[str, Any]] = []
    for index, entry_value in enumerate(entries):
        entry = _strict_json_object(entry_value, name=f"replay transcript entry {index}")
        verifier_response = _strict_json_object(
            entry.get("verifier_response"),
            name=f"replay verifier response {index}",
        )
        if expected_environment == "reasoning_gym":
            score = _require_exact_float(
                verifier_response.get("score"),
                name=f"replay reasoning verifier score {index}",
            )
            raw_reward = _require_exact_float(
                entry.get("raw_environment_reward"),
                name=f"replay reasoning reward {index}",
            )
            server_reward = _require_exact_float(
                verifier_response.get("reward"),
                name=f"replay reasoning server reward {index}",
            )
            extracted_answer = verifier_response.get("extracted_answer")
            if (
                verifier_response.get("task_name") != "knights_knaves"
                or type(verifier_response.get("task_name")) is not str
                or type(extracted_answer) is not str
                or not 0.0 <= score <= 1.0
                or server_reward != score
                or raw_reward != score
            ):
                raise ValueError("replay reasoning match projection differs")
            match_details = {
                "task_name": "knights_knaves",
                "score": score,
                "extracted_answer": extracted_answer,
            }
        else:
            match_details = _strict_json_object(
                verifier_response.get("match_details"),
                name=f"replay verifier match_details {index}",
            )
            raw_reward = entry.get("raw_environment_reward")
        sample = {
            "sample_index": entry.get("sample_index"),
            "fixture_row_index": entry.get("fixture_row_index"),
            "rollout_index": entry.get("rollout_index"),
            "generation_seed": entry.get("generation_seed"),
            "model_transport_entry_sha256": entry.get("model_transport_entry_sha256"),
            "model_transport_request_body_sha256": entry.get("model_transport_request_body_sha256"),
            "model_transport_response_body_sha256": entry.get("model_transport_response_body_sha256"),
            "model_response_sha256": entry.get("model_response_sha256"),
            "match_details": match_details,
            "raw_environment_reward": raw_reward,
        }
        _require_exact_keys(
            sample,
            _AUTHENTICATED_RESULT_SAMPLE_V2_KEYS,
            name=f"authenticated result sample {index}",
        )
        samples.append(sample)
    canonical_ascii_json(samples)
    return samples


def _validate_sealed_result_outputs_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    verified_result: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the exact 13 retained members and all K=4 semantic joins."""
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
    )
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )
    from nemo_rl.utils.strict_model_transport_replay_v3 import (
        validate_strict_model_transport_replay_consumption_v3,
    )

    if type(authenticated_source) is not AuthenticatedOffSourceCapture:
        raise TypeError("authenticated_source must be an exact OFF-source capability")
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    submission = _strict_json_object(
        submission_receipt,
        name="replay submission receipt",
    )
    exit_document = _strict_json_object(exit_receipt, name="replay EXIT receipt")
    profile = get_strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    projection, payloads, _ = _sealed_result_payloads_v2(
        verified_result,
        replay_execution_manifest=manifest,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    result_root = projection["result_root"]
    declared = manifest["artifacts"]["outputs"]
    _require_exact_keys(
        exit_document.get("outputs"),
        REPLAY_OUTPUT_V2_KEYS,
        name="EXIT outputs",
    )
    references: dict[str, dict[str, str]] = {}
    relative_by_output = {
        "scorer_call_index": profile.scorer_terminal_index_path,
        "transport_consumption": "model-transport-replay-consumption.json",
        "transcript_bundle": "transcript-bundle.json",
        "replay_ledger": "replay-ledger.json",
    }
    for name in sorted(REPLAY_OUTPUT_V2_KEYS):
        reference = _artifact_reference(
            exit_document["outputs"][name],
            name=f"EXIT output {name}",
            expected_schema=declared[name]["schema"],
        )
        relative = relative_by_output[name]
        raw = payloads[relative]
        if (
            reference["path"] != f"{result_root}/{relative}"
            or reference["path"] != declared[name]["path"]
            or reference["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError(f"EXIT output {name} differs from retained member bytes")
        references[name] = reference

    transcript_raw = payloads[relative_by_output["transcript_bundle"]]
    transcript, _ = decode_evidence_document_bytes(
        raw=transcript_raw,
        expected_sha256=references["transcript_bundle"]["sha256"],
        trailing_lf=False,
    )
    validate_transcript_bundle(transcript)
    consumption_raw = payloads[relative_by_output["transport_consumption"]]
    consumption, _ = decode_evidence_document_bytes(
        raw=consumption_raw,
        expected_sha256=references["transport_consumption"]["sha256"],
        trailing_lf=False,
    )
    validate_strict_model_transport_replay_consumption_v3(
        consumption,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    source_transcript = _strict_json_object(
        authenticated_source.transcript_bundle,
        name="authenticated OFF source transcript",
    )
    validate_transcript_bundle(source_transcript)
    source_transcript_ref = _artifact_reference(
        manifest["source_capture"]["step1_evidence"]["transcript_bundle"],
        name="source transcript bundle",
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
    )
    if source_transcript_ref["sha256"] != document_sha256(
        source_transcript,
        trailing_lf=False,
    ):
        raise ValueError("authenticated OFF source transcript digest differs")
    validate_captured_replay_source_join(
        source_transcript_bundle=source_transcript,
        replay_transcript_bundle=transcript,
    )
    ledger_raw = payloads[relative_by_output["replay_ledger"]]
    ledger, _ = decode_evidence_document_bytes(
        raw=ledger_raw,
        expected_sha256=references["replay_ledger"]["sha256"],
        trailing_lf=False,
    )
    validate_captured_replay_step1_ledger(
        ledger,
        source_transcript_document=source_transcript,
        transcript_document=transcript,
    )
    validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)

    scorer_relative = profile.scorer_terminal_index_path
    child_prefix = "strict_gym_child_runtime/"
    child_payloads = tuple(
        (relative, payloads[relative]) for relative in profile.result_files if relative.startswith(child_prefix)
    )
    bootstrap_program = manifest["replay_contract"]["program"]["gym_child_bootstrap"]
    bootstrap_relative = Path(bootstrap_program["path"])
    if (
        bootstrap_relative.is_absolute()
        or bootstrap_relative.name != "sitecustomize.py"
        or any(part in {"", ".", ".."} for part in bootstrap_relative.parts)
    ):
        raise ValueError("authenticated Gym bootstrap path is noncanonical")
    snapshot_root = _canonical_absolute_path(
        manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        name="authenticated replay source snapshot",
    )
    bootstrap_root = snapshot_root / bootstrap_relative.parent
    scorer_document, scorer_sha256 = _validate_retained_scorer_call_index_v2(
        child_payloads,
        expected_sha256=references["scorer_call_index"]["sha256"],
        expected_receipt_root=Path(result_root) / "strict_gym_child_runtime",
        expected_bootstrap_root=bootstrap_root,
        expected_bootstrap_sha256=bootstrap_program["sha256"],
        expected_pair_id=manifest["pair_id"],
        expected_job_id=exit_document["authenticated_job_id"],
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if scorer_sha256 != hashlib.sha256(payloads[scorer_relative]).hexdigest():
        raise ValueError("scorer validator returned a different terminal digest")
    scorer_resource, _ = decode_evidence_document_bytes(
        raw=payloads["strict_gym_child_runtime/resource.json"],
        expected_sha256=hashlib.sha256(payloads["strict_gym_child_runtime/resource.json"]).hexdigest(),
        trailing_lf=False,
    )
    driver_process, _ = _validate_scorer_resource_process_v2(
        replay_execution_manifest=manifest,
        driver_process=exit_document["driver_process"],
        resource_process=scorer_resource["process"],
    )

    documents = {
        "transcript_bundle": transcript,
        "transport_consumption": consumption,
        "replay_ledger": ledger,
        "scorer_call_index": scorer_document,
    }
    device_environment = _validate_scheduler_device_environment(exit_document["driver_scheduler_device_environment"])
    _close_replay_output_documents(
        documents,
        references=references,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        authenticated_job_id=exit_document["authenticated_job_id"],
        driver_process=driver_process,
        driver_scheduler_device_environment=device_environment,
        source_transcript_ref=source_transcript_ref,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected_runtime = _replay_runtime_attestation(manifest, references)
    if not _exact_json_equal(exit_document["runtime_attestation"], expected_runtime):
        raise ValueError("EXIT runtime attestation differs from retained outputs")
    samples = _project_authenticated_result_samples_v2(
        transcript,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return documents, samples


def _validate_lifecycle_before_result_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    """Validate all M4/S5/PRE3/EXIT6 authority before opening result_root."""
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    submission = _strict_json_object(
        submission_receipt,
        name="replay submission receipt",
    )
    pre = _strict_json_object(pre_receipt, name="replay PRE receipt")
    exit_document = _strict_json_object(exit_receipt, name="replay EXIT receipt")
    validate_captured_replay_submission_receipt_v2(
        submission,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_captured_replay_pre_receipt_v2(
        pre,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(
        exit_document,
        REPLAY_EXIT_V2_ROOT_KEYS,
        name="replay EXIT receipt",
    )
    expected_exit_envelope = {
        "schema": REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "phase": "EXIT",
        "status": "complete",
        "pair_id": manifest["pair_id"],
        "environment": expected_environment,
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(
        exit_document,
        expected_exit_envelope,
        name="EXIT envelope",
    )
    for name in (
        "replay_execution_manifest",
        "submission_receipt",
        "candidate_job_id",
        "authenticated_job_id",
        "job",
        "static_boundary",
    ):
        if not _exact_json_equal(exit_document[name], pre[name]):
            raise ValueError(f"EXIT {name} differs from authenticated PRE")
    authenticated_job_id = _require_job_id(
        exit_document["authenticated_job_id"],
        name="authenticated_job_id",
    )
    if exit_document["candidate_job_id"] != authenticated_job_id:
        raise ValueError("EXIT candidate/authenticated job IDs differ")
    pair = _strict_json_object(
        authenticated_source.pair_manifest,
        name="authenticated Pair manifest",
    )
    campaign = _strict_json_object(pair["campaign"], name="Pair campaign")
    expected_job = {
        "account": campaign["slurm"]["account"],
        "name": manifest["scheduler_submission"]["identity"]["job_name"],
        "num_nodes": campaign["nodes"],
        "partition": campaign["slurm"]["partition"],
        "qos": campaign["slurm"]["qos"],
        "gpus_per_node": 4,
        "restart_count": 0,
    }
    if not _exact_json_equal(exit_document["job"], expected_job):
        raise ValueError("EXIT job allocation differs from authenticated Pair")
    receipt_paths = _final_receipt_paths_v2(
        manifest,
        authenticated_job_id=authenticated_job_id,
    )
    expected_pre_ref = _final_receipt_reference(
        path=receipt_paths["pre"],
        schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        document=pre,
        trailing_lf=True,
    )
    if not _exact_json_equal(exit_document["pre_receipt"], expected_pre_ref):
        raise ValueError("EXIT PRE reference differs from owned PRE bytes")
    query_ref, query = _load_replay_scheduler_query_reference(
        exit_document["post_scheduler_query"],
        phase="POST",
        expected_path=_job_query_path(
            manifest,
            authenticated_job_id,
            "POST",
            raw=False,
        ),
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if not _exact_json_equal(query_ref, exit_document["post_scheduler_query"]):
        raise ValueError("EXIT POST scheduler query reference is noncanonical")
    _close_scheduler_query_to_lifecycle(
        query,
        replay_execution_manifest=manifest,
        candidate_job_id=authenticated_job_id,
        comment=submission["comment"],
        submitter_euid=submission["submitter_euid"],
    )
    if type(exit_document["driver_exit_code"]) is not int or exit_document["driver_exit_code"] != 0:
        raise ValueError("successful replay EXIT requires exact exit code zero")
    _validate_hardware(
        exit_document["hardware"],
        replay_execution_manifest=manifest,
    )
    wrapper_devices = _validate_scheduler_device_environment(exit_document["scheduler_device_environment"])
    driver_devices = _validate_scheduler_device_environment(exit_document["driver_scheduler_device_environment"])
    if not _exact_json_equal(wrapper_devices, driver_devices):
        raise ValueError("EXIT wrapper/driver scheduler device environments differ")
    _validate_process(exit_document["driver_process"])
    if exit_document["post_verified"] is not True:
        raise ValueError("successful replay EXIT must be post-verified")
    _require_exact_keys(
        exit_document["outputs"],
        REPLAY_OUTPUT_V2_KEYS,
        name="EXIT outputs",
    )
    declared_outputs = manifest["artifacts"]["outputs"]
    output_refs: dict[str, dict[str, str]] = {}
    for name in sorted(REPLAY_OUTPUT_V2_KEYS):
        declaration = declared_outputs[name]
        reference = _artifact_reference(
            exit_document["outputs"][name],
            name=f"EXIT output {name}",
            expected_schema=declaration["schema"],
        )
        if reference["path"] != declaration["path"]:
            raise ValueError(f"EXIT output {name} path differs from M4")
        output_refs[name] = reference
    expected_runtime = _replay_runtime_attestation(manifest, output_refs)
    _require_exact_keys(
        exit_document["runtime_attestation"],
        REPLAY_RUNTIME_ATTESTATION_V2_KEYS,
        name="EXIT runtime attestation",
    )
    if not _exact_json_equal(exit_document["runtime_attestation"], expected_runtime):
        raise ValueError("EXIT runtime attestation differs from declared outputs")
    if exit_document["runtime_attestation"]["original_process_reaped"] is not True:
        raise ValueError("EXIT runtime attestation does not prove scorer reaping")

    return manifest, submission, pre, exit_document, receipt_paths


def _validate_terminal_result_v2(
    *,
    validated_lifecycle: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, str],
    ],
    evidence_index: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    verified_result: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> list[dict[str, Any]]:
    """Close retained outputs/index using an already authenticated lifecycle."""
    if type(validated_lifecycle) is not tuple or len(validated_lifecycle) != 5:
        raise TypeError("validated lifecycle must be the exact private five-value tuple")
    manifest, submission, pre, exit_document = (
        _strict_json_object(document, name=name)
        for document, name in zip(
            validated_lifecycle[:4],
            (
                "replay execution manifest",
                "replay submission receipt",
                "replay PRE receipt",
                "replay EXIT receipt",
            ),
            strict=True,
        )
    )
    authenticated_job_id = _require_job_id(
        exit_document["authenticated_job_id"],
        name="authenticated_job_id",
    )
    receipt_paths = _final_receipt_paths_v2(
        manifest,
        authenticated_job_id=authenticated_job_id,
    )
    if not _exact_json_equal(validated_lifecycle[4], receipt_paths):
        raise ValueError("validated lifecycle receipt paths differ")
    index = _strict_json_object(evidence_index, name="replay evidence index")

    _, samples = _validate_sealed_result_outputs_v2(
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        exit_receipt=exit_document,
        authenticated_source=authenticated_source,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _require_exact_keys(index, REPLAY_POST_INDEX_V2_ROOT_KEYS, name="replay index")
    expected_index_envelope = {
        "schema": REPLAY_POST_INDEX_V2_SCHEMA,
        "scorer_profile": manifest["scorer_profile"],
        "original_process_reaped": True,
        "profile_id": expected_profile_id,
        "hash_domain": HASH_DOMAIN,
        "pair_id": manifest["pair_id"],
        "environment": expected_environment,
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(index, expected_index_envelope, name="index envelope")
    expected_index_refs = {
        "replay_execution_manifest": submission["replay_execution_manifest"],
        "pair_submission_receipt": manifest["pair"]["submission_receipt"],
        "submission_receipt": pre["submission_receipt"],
        "pre_receipt": exit_document["pre_receipt"],
        "exit_receipt": _final_receipt_reference(
            path=receipt_paths["exit"],
            schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
            document=exit_document,
            trailing_lf=True,
        ),
        "source_capture": manifest["source_capture"],
        "outputs": exit_document["outputs"],
    }
    for name, expected in expected_index_refs.items():
        if not _exact_json_equal(index[name], expected):
            raise ValueError(f"replay index {name} differs from lifecycle authority")
    expected_identity = {
        "candidate_job_id": authenticated_job_id,
        "authenticated_job_id": authenticated_job_id,
        "driver_process": _validate_process(exit_document["driver_process"]),
        "run_id": replay_run_id(
            environment=expected_environment,
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
    }
    if not _exact_json_equal(index["identity"], expected_identity):
        raise ValueError("replay index identity differs from lifecycle authority")
    canonical_ascii_json(index)
    return samples


def _validate_terminal_lifecycle_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    verified_result: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> list[dict[str, Any]]:
    """Validate the complete lifecycle and retained terminal result graph."""
    lifecycle = _validate_lifecycle_before_result_v2(
        replay_execution_manifest=replay_execution_manifest,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        exit_receipt=exit_receipt,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return _validate_terminal_result_v2(
        validated_lifecycle=lifecycle,
        evidence_index=evidence_index,
        authenticated_source=authenticated_source,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )


def build_captured_replay_result_final_receipt_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
    verified_result: Any,
) -> dict[str, Any]:
    """Build FINAL only from a validated lifecycle and verifier-retained bytes."""
    frozen_submission = _strict_json_object(
        submission_receipt,
        name="replay submission receipt",
    )
    frozen_pre = _strict_json_object(pre_receipt, name="replay PRE receipt")
    frozen_exit = _strict_json_object(exit_receipt, name="replay EXIT receipt")
    frozen_index = _strict_json_object(evidence_index, name="replay evidence index")
    manifest = _validated_lifecycle_manifest(
        replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _validate_terminal_lifecycle_v2(
        replay_execution_manifest=manifest,
        submission_receipt=frozen_submission,
        pre_receipt=frozen_pre,
        exit_receipt=frozen_exit,
        evidence_index=frozen_index,
        authenticated_source=authenticated_source,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    document, _ = _derive_result_final_receipt_v2(
        replay_execution_manifest=manifest,
        submission_receipt=frozen_submission,
        pre_receipt=frozen_pre,
        exit_receipt=frozen_exit,
        evidence_index=frozen_index,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return document


def publish_captured_replay_result_final_receipt_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
    submission_receipt: Mapping[str, Any],
    pre_receipt: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    evidence_index: Mapping[str, Any],
    verified_result: Any,
) -> tuple[Path, str]:
    """Exclusively publish the externally carried post-seal FINAL receipt."""
    frozen_manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    expected = build_captured_replay_result_final_receipt_v2(
        replay_execution_manifest=frozen_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
        submission_receipt=submission_receipt,
        pre_receipt=pre_receipt,
        exit_receipt=exit_receipt,
        evidence_index=evidence_index,
        verified_result=verified_result,
    )
    expected_path = _final_receipt_paths_v2(
        frozen_manifest,
        authenticated_job_id=expected["authenticated_job_id"],
    )["final"]
    supplied = _strict_json_object(document, name="result FINAL receipt")
    _require_exact_keys(
        supplied,
        REPLAY_RESULT_FINAL_V2_ROOT_KEYS,
        name="result FINAL receipt",
    )
    if not _exact_json_equal(supplied, expected):
        raise ValueError("result FINAL receipt differs from retained authority")
    actual_path = str(_canonical_absolute_path(str(output), name="FINAL output path"))
    if actual_path != expected_path:
        raise ValueError("FINAL output path differs from authenticated job receipt root")
    return publish_evidence_document(
        output=actual_path,
        document=supplied,
        trailing_lf=True,
    )


def _validate_result_final_receipt_v2(
    document: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    replay_execution_manifest_path: str,
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    candidate_job_id: str,
    result_final_receipt_path: str,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    """Validate the external FINAL envelope before following any of its refs."""
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    final = _strict_json_object(document, name="result FINAL receipt")
    _require_exact_keys(
        final,
        REPLAY_RESULT_FINAL_V2_ROOT_KEYS,
        name="result FINAL receipt",
    )
    expected_envelope = {
        "schema": REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "phase": "FINAL",
        "status": "complete",
        "pair_id": manifest["pair_id"],
        "environment": expected_environment,
        "scorer_profile": manifest["scorer_profile"],
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": manifest["attempt_id"],
        "original_process_reaped": True,
    }
    _require_exact_projection(final, expected_envelope, name="FINAL envelope")
    if final["original_process_reaped"] is not True:
        raise ValueError("FINAL must prove original scorer-process reaping")
    if manifest["scorer_profile"].get("profile_id") != expected_profile_id:
        raise ValueError("FINAL manifest profile differs from expected profile")

    candidate = _require_job_id(
        final["candidate_job_id"],
        name="FINAL candidate_job_id",
    )
    authenticated = _require_job_id(
        final["authenticated_job_id"],
        name="FINAL authenticated_job_id",
    )
    expected_candidate = _require_job_id(candidate_job_id, name="candidate_job_id")
    if candidate != expected_candidate or authenticated != expected_candidate:
        raise ValueError("FINAL candidate/authenticated job IDs differ from submit authority")
    _validate_process(final["driver_process"])

    manifest_ref = _artifact_reference(
        final["replay_execution_manifest"],
        name="FINAL replay execution manifest",
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    canonical_manifest_path = _canonical_replay_manifest_path_v2(manifest)
    if (
        replay_execution_manifest_path != canonical_manifest_path
        or manifest_ref["path"] != canonical_manifest_path
        or manifest_ref["sha256"] != replay_execution_manifest_sha256
    ):
        raise ValueError("FINAL manifest reference differs from OOB canonical M4 authority")
    submission_ref = _artifact_reference(
        final["submission_receipt"],
        name="FINAL submission receipt",
        expected_schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    )
    declared_submission = manifest["scheduler_submission"]["receipt"]
    if submission_ref["path"] != declared_submission["path"] or submission_ref["sha256"] != submission_receipt_sha256:
        raise ValueError("FINAL submission reference differs from OOB S5 authority")

    paths = _final_receipt_paths_v2(
        manifest,
        authenticated_job_id=authenticated,
    )
    actual_final_path = str(
        _canonical_absolute_path(
            result_final_receipt_path,
            name="result FINAL receipt path",
        )
    )
    if actual_final_path != paths["final"]:
        raise ValueError("FINAL is not at the authenticated job receipt root")
    pre_ref = _artifact_reference(
        final["pre_receipt"],
        name="FINAL PRE receipt",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    exit_ref = _artifact_reference(
        final["exit_receipt"],
        name="FINAL EXIT receipt",
        expected_schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    )
    if pre_ref["path"] != paths["pre"] or exit_ref["path"] != paths["exit"]:
        raise ValueError("FINAL PRE/EXIT paths differ from authenticated job identity")

    index_ref = _artifact_reference(
        final["evidence_index"],
        name="FINAL evidence index",
        expected_schema=REPLAY_POST_INDEX_V2_SCHEMA,
    )
    result = _strict_json_object(final["result"], name="FINAL result")
    _require_exact_keys(result, frozenset({"root", "inventory"}), name="FINAL result")
    result_root = str(_canonical_absolute_path(result["root"], name="FINAL result root"))
    declared_root = str(
        _canonical_absolute_path(
            manifest["artifacts"]["outputs"]["directory"]["path"],
            name="manifest result root",
        )
    )
    if result_root != declared_root:
        raise ValueError("FINAL result root differs from M4 output authority")
    declared_index = manifest["artifacts"]["outputs"]["evidence_index"]
    if index_ref["path"] != declared_index["path"]:
        raise ValueError("FINAL evidence index path differs from M4")
    inventory_ref = _artifact_reference(
        result["inventory"],
        name="FINAL result inventory",
        expected_schema=_RESULT_INVENTORY_V2_SCHEMA,
    )
    if inventory_ref["path"] != f"{result_root}/{_RESULT_INVENTORY_V2_FILENAME}":
        raise ValueError("FINAL inventory path differs from result root")
    receipt_bytes(final)
    return final


def load_authenticated_captured_replay_result_v2(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    replay_execution_manifest_path: str,
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    candidate_job_id: str,
    result_final_receipt_path: str,
    result_final_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> AuthenticatedCapturedReplayResultV2:
    """Authenticate one terminal replay from OOB M4/S5/FINAL authority.

    The sealed inventory and its exact thirteen members are captured once by the
    sealer verifier.  Every subsequent semantic check consumes only those retained
    bytes; no named result member is reopened after the verifier mints authority.
    The outer V2 manifest/inventory/FINAL graph is the sole profile authority.
    Reasoning Gym retains its environment-only V1 scorer-terminal bytes inside
    that profile-bound outer graph.
    """
    if type(expected_environment) is not str or type(expected_profile_id) is not str:
        raise TypeError("expected environment/profile must be exact strings")
    if _RESULT_PROFILE_IDS_V2.get(expected_environment) != expected_profile_id:
        raise ValueError("authenticated result V2 environment/profile pair is unsupported")
    manifest_path = str(
        _canonical_absolute_path(
            replay_execution_manifest_path,
            name="replay execution manifest path",
        )
    )
    final_path = str(
        _canonical_absolute_path(
            result_final_receipt_path,
            name="result FINAL receipt path",
        )
    )
    manifest_sha256 = _require_digest(
        replay_execution_manifest_sha256,
        name="replay execution manifest SHA-256",
    )
    submission_sha256 = _require_digest(
        submission_receipt_sha256,
        name="submission receipt SHA-256",
    )
    final_sha256 = _require_digest(
        result_final_receipt_sha256,
        name="result FINAL receipt SHA-256",
    )
    candidate = _require_job_id(candidate_job_id, name="candidate_job_id")

    # Freeze both independent OOB roots before importing or consulting any
    # manifest-selected executable, source, lifecycle, or result pathname.
    manifest_document, loaded_manifest_sha256, manifest_raw = _load_evidence_document_owned(
        path=manifest_path,
        expected_sha256=manifest_sha256,
        trailing_lf=False,
    )
    final_document, loaded_final_sha256, final_raw = _load_evidence_document_owned(
        path=final_path,
        expected_sha256=final_sha256,
        trailing_lf=True,
    )
    if loaded_manifest_sha256 != manifest_sha256 or loaded_final_sha256 != final_sha256:
        raise AssertionError("unreachable OOB root digest mismatch")
    final = _validate_result_final_receipt_v2(
        final_document,
        replay_execution_manifest=manifest_document,
        replay_execution_manifest_path=manifest_path,
        replay_execution_manifest_sha256=manifest_sha256,
        submission_receipt_sha256=submission_sha256,
        candidate_job_id=candidate,
        result_final_receipt_path=final_path,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )

    # Only an M4/FINAL pair that closes syntactically and cryptographically may
    # select the executable closure used for full source/static authentication.
    _authenticate_program_closure_v2(manifest_document)
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
        _reload_authenticated_off_source_capture,
    )
    from nemo_rl.utils.strict_captured_replay_seal_v2 import verify_sealed_result_v2

    if type(authenticated_source) is not AuthenticatedOffSourceCapture:
        raise TypeError("authenticated_source must be an exact OFF-source capability")
    # Refresh the public-constructible source capability before consulting any of
    # its detached fields.  The returned object is retained privately thereafter.
    source = _reload_authenticated_off_source_capture(authenticated_source)
    manifest = _validated_lifecycle_manifest(
        manifest_document,
        authenticated_source=source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if manifest_path != _canonical_replay_manifest_path_v2(manifest):
        raise ValueError("OOB M4 authority is not at its canonical publication path")
    # Repeat the envelope join after full M4 validation so no malformed field used
    # during early fail-closed screening escapes the authenticated manifest shape.
    final = _validate_result_final_receipt_v2(
        final,
        replay_execution_manifest=manifest,
        replay_execution_manifest_path=manifest_path,
        replay_execution_manifest_sha256=manifest_sha256,
        submission_receipt_sha256=submission_sha256,
        candidate_job_id=candidate,
        result_final_receipt_path=final_path,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )

    submission_path = manifest["scheduler_submission"]["receipt"]["path"]
    submission, loaded_submission_sha256, submission_raw = _load_evidence_document_owned(
        path=submission_path,
        expected_sha256=submission_sha256,
        trailing_lf=True,
    )
    if (
        loaded_submission_sha256 != submission_sha256
        or submission.get("schema") != REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA
        or _require_job_id(
            submission.get("candidate_job_id"),
            name="submission candidate_job_id",
        )
        != candidate
    ):
        raise ValueError("loaded S5 differs from OOB candidate/submission authority")
    expected_manifest_ref = {
        "path": manifest_path,
        "schema": REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
        "sha256": manifest_sha256,
    }
    if not _exact_json_equal(
        submission.get("replay_execution_manifest"),
        expected_manifest_ref,
    ):
        raise ValueError("S5 manifest reference differs from OOB M4 authority")

    pre_ref = _artifact_reference(
        final["pre_receipt"],
        name="FINAL PRE receipt",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    pre, pre_sha256, pre_raw = _load_evidence_document_owned(
        path=pre_ref["path"],
        expected_sha256=pre_ref["sha256"],
        trailing_lf=True,
    )
    if pre_sha256 != pre_ref["sha256"] or pre.get("schema") != REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA:
        raise ValueError("loaded PRE3 differs from FINAL authority")
    authenticated = _require_job_id(
        pre.get("authenticated_job_id"),
        name="PRE authenticated_job_id",
    )
    if (
        pre.get("candidate_job_id") != candidate
        or authenticated != candidate
        or final["authenticated_job_id"] != authenticated
    ):
        raise ValueError("PRE does not authenticate the submitted candidate job ID")

    exit_ref = _artifact_reference(
        final["exit_receipt"],
        name="FINAL EXIT receipt",
        expected_schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    )
    exit_document, exit_sha256, exit_raw = _load_evidence_document_owned(
        path=exit_ref["path"],
        expected_sha256=exit_ref["sha256"],
        trailing_lf=True,
    )
    if exit_sha256 != exit_ref["sha256"] or exit_document.get("schema") != REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA:
        raise ValueError("loaded EXIT6 differs from FINAL authority")

    # S5, PRE3, and every non-result EXIT6 field must authenticate before the
    # result root is opened.  A lifecycle poison therefore cannot select even a
    # schema-valid sealed tree for verification.
    validated_lifecycle = _validate_lifecycle_before_result_v2(
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_document,
        authenticated_source=source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _, validated_submission, validated_pre, validated_exit, receipt_paths = validated_lifecycle
    expected_exit_ref = _final_receipt_reference(
        path=receipt_paths["exit"],
        schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        document=validated_exit,
        trailing_lf=True,
    )
    pre_result_final_joins = {
        "replay_execution_manifest": validated_submission["replay_execution_manifest"],
        "submission_receipt": validated_pre["submission_receipt"],
        "pre_receipt": validated_exit["pre_receipt"],
        "exit_receipt": expected_exit_ref,
        "candidate_job_id": validated_exit["candidate_job_id"],
        "authenticated_job_id": validated_exit["authenticated_job_id"],
        "driver_process": validated_exit["driver_process"],
        "original_process_reaped": validated_exit["runtime_attestation"]["original_process_reaped"],
    }
    for name, expected in pre_result_final_joins.items():
        if not _exact_json_equal(final[name], expected):
            raise ValueError(f"FINAL {name} differs from authenticated EXIT lifecycle")

    result = _strict_json_object(final["result"], name="FINAL result")
    inventory_ref = _artifact_reference(
        result["inventory"],
        name="FINAL result inventory",
        expected_schema="nemo-rl-strict-captured-replay-result-inventory-v2",
    )
    verified_result = verify_sealed_result_v2(
        result_root=result["root"],
        expected_inventory_sha256=inventory_ref["sha256"],
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    projection, payloads, _ = _sealed_result_payloads_v2(
        verified_result,
        replay_execution_manifest=manifest,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    index_raw = payloads["evidence-index.json"]
    index, index_sha256 = decode_evidence_document_bytes(
        raw=index_raw,
        expected_sha256=final["evidence_index"]["sha256"],
        trailing_lf=False,
    )
    if index_sha256 != final["evidence_index"]["sha256"]:
        raise AssertionError("unreachable retained index digest mismatch")

    _validate_terminal_result_v2(
        validated_lifecycle=validated_lifecycle,
        evidence_index=index,
        authenticated_source=source,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    expected_final, expected_final_path = _derive_result_final_receipt_v2(
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_document,
        evidence_index=index,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if expected_final_path != final_path or not _exact_json_equal(final, expected_final):
        raise ValueError("FINAL differs from the fully authenticated lifecycle graph")
    if projection["inventory"]["sha256"] != inventory_ref["sha256"]:
        raise ValueError("retained inventory differs from FINAL authority")

    source_transcript_raw = canonical_ascii_json(source.transcript_bundle)
    source_transcript_ref = manifest["source_capture"]["step1_evidence"]["transcript_bundle"]
    decode_evidence_document_bytes(
        raw=source_transcript_raw,
        expected_sha256=source_transcript_ref["sha256"],
        trailing_lf=False,
    )
    return AuthenticatedCapturedReplayResultV2(
        _mint_token=_AUTHENTICATED_REPLAY_RESULT_V2_MINT_TOKEN,
        authenticated_source=source,
        candidate_job_id=candidate,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
        final_path=final_path,
        final_sha256=final_sha256,
        manifest_raw=bytes(manifest_raw),
        submission_raw=bytes(submission_raw),
        pre_raw=bytes(pre_raw),
        exit_raw=bytes(exit_raw),
        final_raw=bytes(final_raw),
        source_transcript_raw=bytes(source_transcript_raw),
        result_capability=verified_result,
    )


def snapshot_authenticated_captured_replay_result_v2(
    value: AuthenticatedCapturedReplayResultV2,
) -> dict[str, Any]:
    """Return a fresh detached semantic snapshot from owned authenticated bytes."""
    if type(value) is not AuthenticatedCapturedReplayResultV2:
        raise TypeError("authenticated result snapshot requires the exact V2 capability")
    try:
        token = object.__getattribute__(
            value,
            "_AuthenticatedCapturedReplayResultV2__mint_token",
        )
    except AttributeError as error:
        raise ValueError("authenticated result capability token differs") from error
    if token is not _AUTHENTICATED_REPLAY_RESULT_V2_MINT_TOKEN:
        raise ValueError("authenticated result capability token differs")

    def owned(name: str) -> Any:
        return object.__getattribute__(
            value,
            f"_AuthenticatedCapturedReplayResultV2__{name}",
        )

    candidate = _require_job_id(owned("candidate_job_id"), name="candidate_job_id")
    expected_environment = owned("expected_environment")
    expected_profile_id = owned("expected_profile_id")
    if type(expected_environment) is not str or type(expected_profile_id) is not str:
        raise TypeError("authenticated result environment/profile differs")
    final_path = str(_canonical_absolute_path(owned("final_path"), name="result FINAL receipt path"))
    final_sha256 = _require_digest(
        owned("final_sha256"),
        name="result FINAL receipt SHA-256",
    )

    final, _ = decode_evidence_document_bytes(
        raw=owned("final_raw"),
        expected_sha256=final_sha256,
        trailing_lf=True,
    )
    manifest_ref = _artifact_reference(
        final["replay_execution_manifest"],
        name="FINAL replay execution manifest",
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    manifest, _ = decode_evidence_document_bytes(
        raw=owned("manifest_raw"),
        expected_sha256=manifest_ref["sha256"],
        trailing_lf=False,
    )
    submission_ref = _artifact_reference(
        final["submission_receipt"],
        name="FINAL submission receipt",
        expected_schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    )
    submission, _ = decode_evidence_document_bytes(
        raw=owned("submission_raw"),
        expected_sha256=submission_ref["sha256"],
        trailing_lf=True,
    )
    pre_ref = _artifact_reference(
        final["pre_receipt"],
        name="FINAL PRE receipt",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    pre, _ = decode_evidence_document_bytes(
        raw=owned("pre_raw"),
        expected_sha256=pre_ref["sha256"],
        trailing_lf=True,
    )
    exit_ref = _artifact_reference(
        final["exit_receipt"],
        name="FINAL EXIT receipt",
        expected_schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    )
    exit_document, _ = decode_evidence_document_bytes(
        raw=owned("exit_raw"),
        expected_sha256=exit_ref["sha256"],
        trailing_lf=True,
    )
    source_transcript_ref = manifest["source_capture"]["step1_evidence"]["transcript_bundle"]
    source_transcript, _ = decode_evidence_document_bytes(
        raw=owned("source_transcript_raw"),
        expected_sha256=source_transcript_ref["sha256"],
        trailing_lf=False,
    )
    source = owned("authenticated_source")
    if not _exact_json_equal(source_transcript, source.transcript_bundle):
        raise ValueError("owned source transcript differs from source capability")

    verified_result = owned("result_capability")
    projection, payloads, _ = _sealed_result_payloads_v2(
        verified_result,
        replay_execution_manifest=manifest,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    index, _ = decode_evidence_document_bytes(
        raw=payloads["evidence-index.json"],
        expected_sha256=final["evidence_index"]["sha256"],
        trailing_lf=False,
    )
    expected_final, expected_final_path = _derive_result_final_receipt_v2(
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_document,
        evidence_index=index,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if (
        expected_final_path != final_path
        or final["candidate_job_id"] != candidate
        or not _exact_json_equal(final, expected_final)
    ):
        raise ValueError("owned FINAL graph identity differs")
    documents, samples = _validate_sealed_result_outputs_v2(
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        exit_receipt=exit_document,
        authenticated_source=source,
        verified_result=verified_result,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    scorer_resource, _ = decode_evidence_document_bytes(
        raw=payloads["strict_gym_child_runtime/resource.json"],
        expected_sha256=hashlib.sha256(payloads["strict_gym_child_runtime/resource.json"]).hexdigest(),
        trailing_lf=False,
    )
    resource_process = _strict_json_object(
        scorer_resource["process"],
        name="scorer resource process",
    )
    _, scorer_process_identity = _validate_scorer_resource_process_v2(
        replay_execution_manifest=manifest,
        driver_process=final["driver_process"],
        resource_process=resource_process,
    )
    # The scorer terminal was already joined to the resource process by the pure
    # nine-member validator.  Keep this access explicit so a future return-shape
    # change cannot silently drop that semantic validation.
    scorer_quiescence = _strict_json_object(
        documents["scorer_call_index"]["quiescence"],
        name="scorer terminal quiescence",
    )
    if scorer_quiescence.get("original_process_reaped") is not True:
        raise ValueError("scorer terminal does not prove process reaping")

    identity = _strict_json_object(index["identity"], name="replay index identity")
    snapshot = {
        "schema": AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA,
        "pair_id": manifest["pair_id"],
        "environment": expected_environment,
        "profile_id": expected_profile_id,
        "attempt_id": manifest["attempt_id"],
        "candidate_job_id": candidate,
        "authenticated_job_id": final["authenticated_job_id"],
        "run_id": identity["run_id"],
        "driver_process": copy.deepcopy(final["driver_process"]),
        "scorer_process_identity": scorer_process_identity,
        "manifest": copy.deepcopy(manifest_ref),
        "submission_receipt": copy.deepcopy(submission_ref),
        "pre_receipt": copy.deepcopy(pre_ref),
        "exit_receipt": copy.deepcopy(exit_ref),
        "result_final_receipt": {
            "path": final_path,
            "schema": REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
            "sha256": final_sha256,
        },
        "result_root": projection["result_root"],
        "result_inventory": copy.deepcopy(final["result"]["inventory"]),
        "evidence_index": copy.deepcopy(final["evidence_index"]),
        "outputs": copy.deepcopy(index["outputs"]),
        "samples": copy.deepcopy(samples),
    }
    _require_exact_keys(
        snapshot,
        _AUTHENTICATED_RESULT_SNAPSHOT_V2_KEYS,
        name="authenticated replay result snapshot",
    )
    canonical_ascii_json(snapshot)
    return copy.deepcopy(snapshot)


def publish_evidence_document(
    *, output: str | Path, document: Mapping[str, Any], trailing_lf: bool
) -> tuple[Path, str]:
    """Exclusively publish one canonical single-link mode-0400 document."""
    path = _canonical_absolute_path(str(output), name="output")
    if path == Path("/") or path.name in {"", ".", ".."}:
        raise ValueError("output must name a file below an absolute directory")
    payload = receipt_bytes(document) if trailing_lf else canonical_ascii_json(document)
    digest = hashlib.sha256(payload).hexdigest()
    parent_fd = _open_absolute_directory_without_symlinks(path.parent)
    candidate_fd: int | None = None
    candidate_created = False
    candidate_name = f".{path.name}.candidate"
    try:
        parent_initial = os.fstat(parent_fd)
        _require_owned_directory(parent_initial, name="output parent")
        candidate_fd = os.open(
            candidate_name,
            _DOCUMENT_CREATE_FLAGS,
            stat.S_IRUSR,
            dir_fd=parent_fd,
        )
        candidate_created = True
        _write_all(candidate_fd, payload)
        os.fchmod(candidate_fd, stat.S_IRUSR)
        os.fsync(candidate_fd)
        metadata = os.fstat(candidate_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise RuntimeError("evidence candidate inode validation failed")
        os.close(candidate_fd)
        candidate_fd = None
        try:
            os.link(
                candidate_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"evidence output already exists: {path}") from error
        os.unlink(candidate_name, dir_fd=parent_fd)
        candidate_created = False
        os.fsync(parent_fd)
        final_pre_named = os.stat(
            path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(final_pre_named.st_mode)
            or stat.S_IMODE(final_pre_named.st_mode) != 0o400
            or final_pre_named.st_uid != os.geteuid()
            or final_pre_named.st_nlink != 1
            or final_pre_named.st_size != len(payload)
        ):
            raise RuntimeError("published evidence pathname is not the candidate regular file")
        final_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            final_before = os.fstat(final_fd)
            if _file_fingerprint(final_pre_named) != _file_fingerprint(final_before) or not stat.S_ISREG(
                final_before.st_mode
            ):
                raise RuntimeError("published evidence changed before exact readback")
            actual = _read_all_bounded(final_fd, maximum=len(payload))
            final_after = os.fstat(final_fd)
            final_named = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(final_fd)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_without_symlinks(path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            _require_owned_directory(fresh_parent, name="fresh output parent")
            fresh_named = os.stat(
                path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
        if (
            _directory_identity(parent_initial) != _directory_identity(parent_after)
            or _directory_identity(parent_after) != _directory_identity(fresh_parent)
            or _file_fingerprint(final_pre_named) != _file_fingerprint(final_before)
            or _file_fingerprint(final_before) != _file_fingerprint(final_after)
            or _file_fingerprint(final_after) != _file_fingerprint(final_named)
            or _file_fingerprint(final_named) != _file_fingerprint(fresh_named)
            or not stat.S_ISREG(final_after.st_mode)
            or stat.S_IMODE(final_after.st_mode) != 0o400
            or final_after.st_uid != os.geteuid()
            or final_after.st_nlink != 1
            or final_after.st_size != len(payload)
            or actual != payload
            or hashlib.sha256(actual).hexdigest() != digest
        ):
            raise RuntimeError("published evidence failed exact verification")
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)
        if candidate_created:
            try:
                os.unlink(candidate_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return path, digest


def publish_main_transcript_bundle(*, results_dir: str, document: Mapping[str, Any]) -> tuple[Path, str]:
    """Create the strict evidence directory and publish its main transcript."""
    validate_transcript_bundle(document)
    results_path = _canonical_absolute_path(results_dir, name="RESULTS_DIR")
    output = main_transcript_bundle_path(results_dir)
    results_fd = _open_absolute_directory_without_symlinks(results_path)
    evidence_fd: int | None = None
    try:
        _require_owned_directory(os.fstat(results_fd), name="RESULTS_DIR")
        try:
            os.mkdir(MAIN_EVIDENCE_DIRECTORY, 0o700, dir_fd=results_fd)
            os.fsync(results_fd)
        except FileExistsError:
            pass
        evidence_fd = os.open(
            MAIN_EVIDENCE_DIRECTORY,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=results_fd,
        )
        _require_owned_directory(os.fstat(evidence_fd), name="main step1 evidence directory")
    finally:
        if evidence_fd is not None:
            os.close(evidence_fd)
        os.close(results_fd)
    return publish_evidence_document(output=output, document=document, trailing_lf=False)


def decode_evidence_document_bytes(
    *,
    raw: bytes,
    expected_sha256: str,
    trailing_lf: bool,
) -> tuple[dict[str, Any], str]:
    """Decode authenticated evidence bytes without consulting a pathname.

    ``raw`` must be exact ``bytes`` so callers cannot smuggle mutable buffers or
    subclasses across the authenticated snapshot boundary.  The parser keeps
    every framing and canonical-JSON check used by the filesystem loader.
    """
    if type(raw) is not bytes:
        raise TypeError("evidence payload must be exact immutable bytes")
    if type(trailing_lf) is not bool:
        raise TypeError("trailing_lf must be an exact bool")
    if not 0 < len(raw) <= _MAX_EVIDENCE_DOCUMENT_BYTES:
        raise ValueError("evidence size is outside the admitted range")
    expected = _require_digest(expected_sha256, name="expected_sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("evidence bytes differ from expected SHA-256")
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError("receipt must end in exactly one LF")
        json_payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            raise ValueError("bundle/ledger/index must not end in LF")
        json_payload = raw
    try:
        value = json.loads(
            json_payload.decode("ascii"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("evidence is not strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise TypeError("evidence root must be an object")
    expected_raw = receipt_bytes(value) if trailing_lf else canonical_ascii_json(value)
    if raw != expected_raw:
        raise ValueError("evidence bytes are not the exact canonical encoding")
    return value, actual


def _load_evidence_document_owned(
    *,
    path: str | Path,
    expected_sha256: str,
    trailing_lf: bool,
) -> tuple[dict[str, Any], str, bytes]:
    """Stable-load one exact evidence inode and retain its immutable bytes."""
    evidence_path = _canonical_absolute_path(str(path), name="evidence path")
    expected = _require_digest(expected_sha256, name="expected_sha256")
    parent_fd = _open_absolute_directory_without_symlinks(evidence_path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        _require_owned_directory(parent_before, name="evidence parent")
        pre_named = os.stat(
            evidence_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(pre_named.st_mode)
            or stat.S_IMODE(pre_named.st_mode) != 0o400
            or pre_named.st_uid != os.geteuid()
            or pre_named.st_nlink != 1
            or not 0 < pre_named.st_size <= _MAX_EVIDENCE_DOCUMENT_BYTES
        ):
            raise RuntimeError("evidence must be an EUID-owned single-link mode-0400 regular file")
        descriptor = os.open(
            evidence_path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                _file_fingerprint(pre_named) != _file_fingerprint(before)
                or not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= _MAX_EVIDENCE_DOCUMENT_BYTES
            ):
                raise RuntimeError("evidence changed before stable read")
            raw = _read_all_bounded(descriptor, maximum=_MAX_EVIDENCE_DOCUMENT_BYTES)
            after = os.fstat(descriptor)
            named = os.stat(evidence_path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_without_symlinks(evidence_path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            _require_owned_directory(fresh_parent, name="fresh evidence parent")
            fresh_named = os.stat(
                evidence_path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
    finally:
        os.close(parent_fd)
    if not (
        _directory_identity(parent_before) == _directory_identity(parent_after) == _directory_identity(fresh_parent)
    ):
        raise RuntimeError("evidence parent changed during stable read")
    if not (
        _file_fingerprint(pre_named)
        == _file_fingerprint(before)
        == _file_fingerprint(after)
        == _file_fingerprint(named)
        == _file_fingerprint(fresh_named)
    ):
        raise RuntimeError("evidence changed during stable read")
    if len(raw) != after.st_size:
        raise RuntimeError("evidence size changed during stable read")
    if (
        not stat.S_ISREG(after.st_mode)
        or stat.S_IMODE(after.st_mode) != 0o400
        or after.st_uid != os.geteuid()
        or after.st_nlink != 1
    ):
        raise RuntimeError("evidence must be an EUID-owned single-link mode-0400 regular file")
    document, actual = decode_evidence_document_bytes(
        raw=raw,
        expected_sha256=expected,
        trailing_lf=trailing_lf,
    )
    return document, actual, raw


def load_evidence_document(
    *,
    path: str | Path,
    expected_sha256: str,
    trailing_lf: bool,
) -> tuple[dict[str, Any], str]:
    """Load one exact immutable evidence inode without following symlinks."""
    document, digest, _ = _load_evidence_document_owned(
        path=path,
        expected_sha256=expected_sha256,
        trailing_lf=trailing_lf,
    )
    return document, digest


def _build_transcript_entry(
    raw: Mapping[str, Any],
    *,
    index: int,
    generation: Mapping[str, Any],
    environment: str,
    expected_agent: str,
    fixture_row: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(raw, TRANSCRIPT_ENTRY_INPUT_KEYS, name=f"entry[{index}]")
    if raw["sample_index"] != index or type(raw["sample_index"]) is not int:
        raise ValueError(f"entry[{index}].sample_index must equal its list index")
    if raw["fixture_row_index"] != 0 or type(raw["fixture_row_index"]) is not int:
        raise ValueError("strict no-shuffle capture requires fixture_row_index=0")
    if raw["rollout_index"] != index or type(raw["rollout_index"]) is not int:
        raise ValueError(f"entry[{index}].rollout_index must equal its list index")
    seed = _require_nonnegative_int(raw["generation_seed"], name=f"entry[{index}].generation_seed")
    expected_seed = derive_nemo_gym_request_seed(
        seed_base=generation["seed_base"], fixture_row_index=0, rollout_index=index
    )
    if seed != expected_seed:
        raise ValueError(f"entry[{index}] generation seed does not close")
    generation_request = _strict_json_object(raw["generation_request"], name=f"entry[{index}].generation_request")
    model_response = _strict_json_object(raw["model_response"], name=f"entry[{index}].model_response")
    if "reward" in model_response:
        raise ValueError(f"entry[{index}] model_response must not contain a root reward")
    agent_run_request = _strict_json_object(raw["agent_run_request"], name=f"entry[{index}].agent_run_request")
    derived_verifier_request = _strict_json_object(
        raw["derived_verifier_request"],
        name=f"entry[{index}].derived_verifier_request",
    )
    verifier_response = _strict_json_object(raw["verifier_response"], name=f"entry[{index}].verifier_response")
    _require_exact_keys(
        verifier_response,
        VERIFIER_RESPONSE_KEYS_BY_ENVIRONMENT[environment],
        name=f"entry[{index}].verifier_response",
    )
    agent_ref = agent_run_request.get("agent_ref")
    if not isinstance(agent_ref, Mapping) or set(agent_ref) != {"type", "name"}:
        raise ValueError(f"entry[{index}] agent_run_request.agent_ref is not exact")
    if agent_ref != {"type": "responses_api_agents", "name": expected_agent}:
        raise ValueError(f"entry[{index}] verifier agent differs from environment")
    task_index = _require_nonnegative_int(
        agent_run_request.get("_ng_task_index"),
        name=f"entry[{index}].agent_run_request._ng_task_index",
    )
    if task_index > (1 << 63) - 1:
        raise ValueError(f"entry[{index}] NeMo-Gym task index exceeds int63")
    expected_agent_run_request = _expected_transformed_fixture_request(
        fixture_row,
        rollout_index=index,
        generation_seed=seed,
        task_index=task_index,
    )
    if not _exact_json_equal(agent_run_request, expected_agent_run_request):
        raise ValueError(f"entry[{index}] fixture-to-agent-run transform differs")
    if not _exact_json_equal(generation_request, expected_agent_run_request["responses_create_params"]):
        raise ValueError(f"entry[{index}] fixture-to-generation transform differs")
    returned_seed = _seed_from_responses_create_params(generation_request, name=f"entry[{index}].generation_request")
    dispatched_params = agent_run_request.get("responses_create_params")
    dispatched_seed = _seed_from_responses_create_params(
        dispatched_params,
        name=f"entry[{index}].agent_run_request.responses_create_params",
    )
    returned_params = verifier_response.get("responses_create_params")
    verifier_returned_seed = _seed_from_responses_create_params(
        returned_params,
        name=f"entry[{index}].verifier_response.responses_create_params",
    )
    if {returned_seed, dispatched_seed, verifier_returned_seed, seed} != {seed}:
        raise ValueError(f"entry[{index}] returned/dispatched request seeds differ")
    if not _exact_json_equal(dispatched_params, generation_request):
        raise ValueError(f"entry[{index}] verifier request generation params differ from returned params")
    expected_returned_params = _expanded_responses_create_params(generation_request)
    if not _exact_json_equal(returned_params, expected_returned_params):
        raise ValueError(f"entry[{index}] returned response params differ from pinned Pydantic expansion")
    expected_derived_verifier_request = copy.deepcopy(agent_run_request)
    expected_derived_verifier_request["responses_create_params"] = copy.deepcopy(expected_returned_params)
    expected_derived_verifier_request["response"] = copy.deepcopy(model_response)
    if not _exact_json_equal(derived_verifier_request, expected_derived_verifier_request):
        raise ValueError(f"entry[{index}] derived verifier request differs from pinned model_dump")
    _validate_raw_generation_policy(
        generation_request,
        generation=generation,
        seed=seed,
        name=f"entry[{index}].generation_request",
    )
    validate_gym_model_response_r3(
        model_response,
        generation_request=generation_request,
        name=f"entry[{index}].model_response",
    )
    if not _exact_json_equal(verifier_response.get("response"), model_response):
        raise ValueError(f"entry[{index}] model_response differs from verifier_response.response")
    generation_input = generation_request.get("input")
    if not isinstance(generation_input, list) or not generation_input:
        raise ValueError(f"entry[{index}] generation_request.input must be nonempty")
    reward = _require_exact_float(
        raw["raw_environment_reward"],
        name=f"entry[{index}].raw_environment_reward",
    )
    if environment == "reasoning_gym":
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"entry[{index}] reasoning-gym raw reward must be in [0, 1]")
    elif reward not in (0.0, 1.0):
        raise ValueError(f"entry[{index}] format raw reward must be binary")
    returned_reward = _require_exact_float(
        verifier_response.get("reward"),
        name=f"entry[{index}].verifier_response.reward",
    )
    if returned_reward != reward:
        raise ValueError(f"entry[{index}] raw reward differs from verifier response")
    _validate_environment_verifier_response(
        verifier_response,
        agent_run_request=agent_run_request,
        model_response=model_response,
        environment=environment,
        reward=reward,
        name=f"entry[{index}].verifier_response",
    )
    entry: dict[str, Any] = {
        "sample_index": index,
        "fixture_row_index": 0,
        "rollout_index": index,
        "generation_seed": seed,
        "generation_request": generation_request,
        "generation_request_sha256": domain_sha256("step1-generation-request", generation_request),
        "model_response": model_response,
        "model_response_sha256": domain_sha256("step1-model-response", model_response),
        "agent_run_request": agent_run_request,
        "agent_run_request_sha256": domain_sha256("step1-agent-run-request", agent_run_request),
        "derived_verifier_request": derived_verifier_request,
        "derived_verifier_request_sha256": domain_sha256("step1-derived-verifier-request", derived_verifier_request),
        "verifier_response": verifier_response,
        "verifier_response_sha256": domain_sha256("step1-verifier-response", verifier_response),
        "raw_environment_reward": reward,
        "model_transport_entry_sha256": _require_digest(
            raw["model_transport_entry_sha256"],
            name=f"entry[{index}].model_transport_entry_sha256",
        ),
        "model_transport_request_body_sha256": _require_digest(
            raw["model_transport_request_body_sha256"],
            name=f"entry[{index}].model_transport_request_body_sha256",
        ),
        "model_transport_response_body_sha256": _require_digest(
            raw["model_transport_response_body_sha256"],
            name=f"entry[{index}].model_transport_response_body_sha256",
        ),
    }
    entry["entry_sha256"] = domain_sha256("step1-transcript-entry", entry)
    _require_exact_keys(entry, TRANSCRIPT_ENTRY_KEYS, name=f"entry[{index}]")
    return entry


def validate_model_response_token_ids(model_response: Mapping[str, Any], *, name: str = "model_response") -> None:
    """Validate bounded token arrays and logprobs in a raw Gym model response."""
    model_response_token_geometry(model_response, name=name)


def validate_fresh_verifier_response(
    *,
    environment: str,
    agent_run_request: Mapping[str, Any],
    derived_verifier_request: Mapping[str, Any],
    model_response: Mapping[str, Any],
    verifier_response: Mapping[str, Any],
) -> float:
    """Validate one fresh resource-server result against captured replay inputs.

    This is deliberately a scorer-only boundary.  The caller supplies the
    already-authenticated inbound ``/run`` request and captured model response;
    this helper deterministically reconstructs the pinned resource ``/verify``
    body and validates the fresh result without invoking a SimpleAgent or model.
    """
    if environment not in STRICT_ENVIRONMENTS:
        raise ValueError("fresh verifier environment is not admitted")
    agent_request = _strict_json_object(agent_run_request, name="fresh verifier agent_run_request")
    derived_request = _strict_json_object(derived_verifier_request, name="fresh verifier derived_verifier_request")
    captured_response = _strict_json_object(model_response, name="fresh verifier model_response")
    fresh_response = _strict_json_object(verifier_response, name="fresh verifier response")
    _require_exact_keys(
        fresh_response,
        VERIFIER_RESPONSE_KEYS_BY_ENVIRONMENT[environment],
        name="fresh verifier response",
    )

    compact_params = _strict_json_object(
        agent_request.get("responses_create_params"),
        name="fresh verifier compact responses_create_params",
    )
    expanded_params = _expanded_responses_create_params(compact_params)
    expected_derived = copy.deepcopy(dict(agent_request))
    expected_derived["responses_create_params"] = copy.deepcopy(expanded_params)
    expected_derived["response"] = copy.deepcopy(captured_response)
    if not _exact_json_equal(derived_request, expected_derived):
        raise ValueError("fresh verifier derived request differs from pinned reconstruction")
    if not _exact_json_equal(fresh_response.get("responses_create_params"), expanded_params):
        raise ValueError("fresh verifier returned response params differ")
    if not _exact_json_equal(fresh_response.get("response"), captured_response):
        raise ValueError("fresh verifier returned captured response differs")

    validate_gym_model_response_r3(
        captured_response,
        generation_request=compact_params,
        name="fresh verifier model_response",
    )
    reward = _require_exact_float(fresh_response.get("reward"), name="fresh verifier response.reward")
    if environment == "reasoning_gym":
        if not 0.0 <= reward <= 1.0:
            raise ValueError("fresh reasoning-gym reward must be in [0, 1]")
    elif reward not in (0.0, 1.0):
        raise ValueError("fresh format-verifier reward must be binary")
    _validate_environment_verifier_response(
        fresh_response,
        agent_run_request=agent_request,
        model_response=captured_response,
        environment=environment,
        reward=reward,
        name="fresh verifier response",
    )
    canonical_ascii_json(derived_request)
    canonical_ascii_json(fresh_response)
    return reward


def validate_gym_model_response_r3(
    model_response: Mapping[str, Any],
    *,
    generation_request: Mapping[str, Any],
    name: str = "model_response",
) -> None:
    """Validate the exact pinned Gym 354babf7 pre-verifier Response shape."""
    response = _strict_json_object(model_response, name=name)
    request = _strict_json_object(generation_request, name=f"{name}.generation_request")
    _require_exact_keys(response, _GYM_RESPONSE_ROOT_KEYS, name=name)
    for key in (
        "background",
        "conversation",
        "error",
        "instructions",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "text",
        "top_logprobs",
        "truncation",
        "user",
    ):
        if response[key] is not None:
            raise ValueError(f"{name}.{key} must be null")
    if not _response_uuid(response["id"], prefix="resp", name=f"{name}.id"):
        raise AssertionError("unreachable invalid response UUID")
    created_at = response["created_at"]
    if (
        type(created_at) is not float
        or not math.isfinite(created_at)
        or not 1.0 <= created_at <= float((1 << 63) - 1)
        or not created_at.is_integer()
    ):
        raise ValueError(f"{name}.created_at must be a positive integral float")
    if (
        type(response["model"]) is not str
        or not response["model"]
        or len(response["model"].encode("utf-8")) > 1_048_576
    ):
        raise ValueError(f"{name}.model must be bounded nonempty UTF-8")
    if response["object"] != "response" or type(response["object"]) is not str:
        raise ValueError(f"{name}.object must be response")
    if response["parallel_tool_calls"] is not True:
        raise ValueError(f"{name}.parallel_tool_calls must be true")
    if response["tool_choice"] != "auto" or type(response["tool_choice"]) is not str:
        raise ValueError(f"{name}.tool_choice must be auto")
    if response["tools"] != [] or type(response["tools"]) is not list:
        raise ValueError(f"{name}.tools must be empty")
    if response["metadata"] != request.get("metadata") or not _exact_json_equal(
        response["metadata"], request.get("metadata")
    ):
        raise ValueError(f"{name}.metadata differs from generation request")
    if type(response["max_output_tokens"]) is not int or response["max_output_tokens"] != 768:
        raise ValueError(f"{name}.max_output_tokens must be exact integer 768")
    for key in ("temperature", "top_p"):
        if type(response[key]) is not float or response[key] != 1.0:
            raise ValueError(f"{name}.{key} must be exact float 1.0")
        if type(request.get(key)) is not float or request[key] != response[key]:
            raise ValueError(f"{name}.{key} differs from generation request")

    status = response["status"]
    if status == "completed" and type(status) is str:
        if response["incomplete_details"] is not None:
            raise ValueError(f"{name}.incomplete_details must be null when completed")
    elif status == "incomplete" and type(status) is str:
        if not _exact_json_equal(response["incomplete_details"], {"reason": "max_output_tokens"}):
            raise ValueError(f"{name}.incomplete_details differs for length finish")
    else:
        raise ValueError(f"{name}.status must be completed or incomplete")

    output = response["output"]
    if not isinstance(output, list) or len(output) not in {1, 2}:
        raise ValueError(f"{name}.output must contain one or two converter output items")
    first = output[0]
    if not isinstance(first, Mapping):
        raise TypeError(f"{name}.output[0] must be an object")
    if first.get("type") == "reasoning":
        reasoning = first
        reasoning_keys = _GYM_REASONING_KEYS | (_GYM_TOKEN_KEYS if len(output) == 1 else set())
        _require_exact_keys(reasoning, reasoning_keys, name=f"{name}.output[0]")
        if reasoning["content"] is not None or reasoning["encrypted_content"] is not None:
            raise ValueError(f"{name}.output[0] reasoning payload must be null")
        _response_uuid(reasoning["id"], prefix="rs", name=f"{name}.output[0].id")
        if reasoning["type"] != "reasoning" or type(reasoning["type"]) is not str:
            raise ValueError(f"{name}.output[0].type must be reasoning")
        summaries = reasoning["summary"]
        if not isinstance(summaries, list) or not summaries:
            raise ValueError(f"{name}.output[0].summary must be nonempty")
        for summary_index, summary in enumerate(summaries):
            if not isinstance(summary, Mapping):
                raise TypeError(f"{name}.output[0].summary[{summary_index}] must be an object")
            _require_exact_keys(
                summary,
                frozenset({"text", "type"}),
                name=f"{name}.output[0].summary[{summary_index}]",
            )
            if summary["type"] != "summary_text" or type(summary["type"]) is not str:
                raise ValueError(f"{name}.output[0].summary[{summary_index}].type differs")
            _bounded_utf8_allow_empty(
                summary["text"],
                name=f"{name}.output[0].summary[{summary_index}].text",
                maximum=16 * 1024 * 1024,
            )
        message_index = 1 if len(output) == 2 else None
    elif first.get("type") == "message":
        if len(output) != 1:
            raise ValueError(f"{name}.output message-only branch must have one item")
        message_index = 0
    else:
        raise ValueError(f"{name}.output[0].type is not a converter output type")

    if message_index is not None:
        message = output[message_index]
        if not isinstance(message, Mapping):
            raise TypeError(f"{name}.output[{message_index}] must be an object")
        _require_exact_keys(
            message,
            _GYM_MESSAGE_KEYS | _GYM_TOKEN_KEYS,
            name=f"{name}.output[{message_index}]",
        )
        _response_uuid(
            message["id"],
            prefix="msg",
            name=f"{name}.output[{message_index}].id",
        )
        if (
            message["role"] != "assistant"
            or message["status"] != "completed"
            or message["type"] != "message"
            or any(type(message[key]) is not str for key in ("role", "status", "type"))
        ):
            raise ValueError(f"{name}.output[{message_index}] message identity differs")
        content = message["content"]
        if not isinstance(content, list) or len(content) != 1:
            raise ValueError(f"{name}.output[{message_index}].content must contain one output_text")
        output_text = content[0]
        if not isinstance(output_text, Mapping):
            raise TypeError(f"{name}.output[{message_index}].content[0] must be an object")
        _require_exact_keys(
            output_text,
            _GYM_OUTPUT_TEXT_KEYS,
            name=f"{name}.output[{message_index}].content[0]",
        )
        if (
            output_text["annotations"] != []
            or type(output_text["annotations"]) is not list
            or output_text["logprobs"] is not None
            or output_text["type"] != "output_text"
            or type(output_text["type"]) is not str
        ):
            raise ValueError(f"{name}.output[{message_index}].content[0] fixed fields differ")
        _bounded_utf8_allow_empty(
            output_text["text"],
            name=f"{name}.output[{message_index}].content[0].text",
            maximum=16 * 1024 * 1024,
        )
    last = output[-1]
    if last["routed_experts"] is not None:
        raise ValueError(f"{name}.output[-1].routed_experts must be null")

    usage = response["usage"]
    if not isinstance(usage, Mapping):
        raise TypeError(f"{name}.usage must be an object")
    _require_exact_keys(usage, _GYM_USAGE_KEYS, name=f"{name}.usage")
    if not _exact_json_equal(usage["input_tokens_details"], {"cached_tokens": None}):
        raise ValueError(f"{name}.usage.input_tokens_details differs")
    if not _exact_json_equal(usage["output_tokens_details"], {"reasoning_tokens": None}):
        raise ValueError(f"{name}.usage.output_tokens_details differs")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage[key]
        if type(value) is not int or not 0 <= value <= (1 << 63) - 1:
            raise ValueError(f"{name}.usage.{key} must be an int63 count")
    validate_model_response_token_ids(response, name=name)


def model_response_token_geometry(
    model_response: Mapping[str, Any], *, name: str = "model_response"
) -> tuple[list[int], list[int], list[int]]:
    """Return initial prompt, suffix, and full tokens using Gym's turn folding."""
    output = model_response.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError(f"{name}.output must be a nonempty list")
    seen: list[int] = []
    initial_prompt: list[int] | None = None
    trainable_items = 0
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}.output[{index}] must be an object")
        token_keys = {
            "prompt_token_ids",
            "generation_token_ids",
            "generation_log_probs",
        }
        present = token_keys.intersection(item)
        if not present:
            continue
        if present != token_keys:
            raise ValueError(f"{name}.output[{index}] has an incomplete token transport triple")
        prompt = _token_id_list(item["prompt_token_ids"], name=f"{name}.output[{index}].prompt_token_ids")
        generation = _token_id_list(
            item["generation_token_ids"],
            name=f"{name}.output[{index}].generation_token_ids",
        )
        if not prompt or not generation:
            raise ValueError(f"{name}.output[{index}] has an empty token array")
        logprobs = item["generation_log_probs"]
        if not isinstance(logprobs, list) or len(logprobs) != len(generation):
            raise ValueError(f"{name}.output[{index}] generation token/logprob lengths differ")
        for offset, value in enumerate(logprobs):
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(
                    f"{name}.output[{index}].generation_log_probs[{offset}] " "must be an exact finite float"
                )
            if value == 0.0 and math.copysign(1.0, value) < 0:
                raise ValueError(f"{name}.output[{index}].generation_log_probs[{offset}] " "must not be negative zero")
        if prompt[: len(seen)] != seen:
            raise ValueError(f"{name}.output[{index}] prompt history is non-contiguous")
        if initial_prompt is None:
            initial_prompt = list(prompt)
        seen.extend(prompt[len(seen) :])
        seen.extend(generation)
        if len(seen) > 131_072:
            raise ValueError(f"{name} accumulated token length exceeds 131072")
        trainable_items += 1
    if trainable_items == 0:
        raise ValueError(f"{name} has no transported generated-token item")
    assert initial_prompt is not None
    return initial_prompt, seen[len(initial_prompt) :], seen


def _validate_raw_generation_policy(
    value: Mapping[str, Any], *, generation: Mapping[str, Any], seed: int, name: str
) -> None:
    if generation != {
        "seed_base": generation["seed_base"],
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }:
        raise ValueError("strict captured replay generation policy is not frozen")
    if value.get("max_output_tokens") != 768 or type(value.get("max_output_tokens")) is not int:
        raise ValueError(f"{name}.max_output_tokens must be exactly 768")
    if value.get("temperature") != 1.0 or type(value.get("temperature")) is not float:
        raise ValueError(f"{name}.temperature must be exact float 1.0")
    if value.get("top_p") != 1.0 or type(value.get("top_p")) is not float:
        raise ValueError(f"{name}.top_p must be exact float 1.0")
    if "top_k" in value:
        raise ValueError(f"{name}.top_k must be absent")
    if _seed_from_responses_create_params(value, name=name) != seed:
        raise ValueError(f"{name} does not carry the derived request seed")


def _validate_fixture_row(value: Mapping[str, Any], *, environment: str) -> dict[str, Any]:
    row = _strict_json_object(value, name="fixture_row")
    _require_exact_keys(
        row,
        FIXTURE_ROW_KEYS_BY_ENVIRONMENT[environment],
        name=f"{environment} fixture_row",
    )
    expected_agent = {
        "type": "responses_api_agents",
        "name": AGENT_BY_ENVIRONMENT[environment],
    }
    if not _exact_json_equal(row.get("agent_ref"), expected_agent):
        raise ValueError("fixture_row.agent_ref differs from environment")
    if environment == "reasoning_gym":
        if type(row["question"]) is not str:
            raise TypeError("reasoning_gym fixture_row.question must be a string")
        if row["answer"] is not None and type(row["answer"]) is not str:
            raise TypeError("reasoning_gym fixture_row.answer must be null or a string")
    params = row.get("responses_create_params")
    if not isinstance(params, Mapping):
        raise TypeError("fixture_row.responses_create_params must be an object")
    _require_exact_keys(
        params,
        frozenset({"input"}),
        name="fixture_row.responses_create_params",
    )
    messages = params["input"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("fixture_row input must contain exactly one message")
    message = messages[0]
    if not isinstance(message, Mapping):
        raise TypeError("fixture_row input message must be an object")
    _require_exact_keys(message, frozenset({"content", "role"}), name="fixture_row input message")
    if message["role"] != "user" or type(message["role"]) is not str:
        raise ValueError("fixture_row input message role must be user")
    if (
        type(message["content"]) is not str
        or not message["content"]
        or len(message["content"].encode("utf-8")) > 1_048_576
    ):
        raise ValueError("fixture_row input message content must be bounded text")
    return row


def _expected_transformed_fixture_request(
    fixture_row: Mapping[str, Any],
    *,
    rollout_index: int,
    generation_seed: int,
    task_index: int,
) -> dict[str, Any]:
    transformed = copy.deepcopy(dict(fixture_row))
    transformed["_ng_task_index"] = task_index
    transformed["_rowidx"] = rollout_index
    transformed["_ng_rollout_index"] = rollout_index
    params = transformed["responses_create_params"]
    params["temperature"] = 1.0
    params["top_p"] = 1.0
    params["max_output_tokens"] = 768
    params["metadata"] = {"extra_body": json.dumps({"seed": generation_seed}, sort_keys=True, separators=(",", ":"))}
    return transformed


def _expanded_responses_create_params(
    compact: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce the pinned Gym 354babf7 Pydantic response-params dump."""
    _require_exact_keys(
        compact,
        frozenset({"input", "max_output_tokens", "metadata", "temperature", "top_p"}),
        name="compact responses_create_params",
    )
    messages = compact["input"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("compact responses_create_params input must have one message")
    message = messages[0]
    _require_exact_keys(
        message,
        frozenset({"content", "role"}),
        name="compact responses_create_params input message",
    )
    expanded_message = copy.deepcopy(dict(message))
    expanded_message["type"] = "message"
    expanded = {
        "background": None,
        "include": None,
        "input": [expanded_message],
        "instructions": None,
        "max_output_tokens": compact["max_output_tokens"],
        "max_tool_calls": None,
        "metadata": copy.deepcopy(compact["metadata"]),
        "model": None,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": None,
        "service_tier": None,
        "store": None,
        "stream": None,
        "temperature": compact["temperature"],
        "text": None,
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": None,
        "top_p": compact["top_p"],
        "truncation": None,
        "user": None,
    }
    _require_exact_keys(
        expanded,
        EXPANDED_RESPONSES_CREATE_PARAMS_KEYS,
        name="expanded responses_create_params",
    )
    return expanded


def _reward_facing_text(model_response: Mapping[str, Any], *, name: str) -> str:
    pieces: list[str] = []
    output = model_response.get("output")
    if not isinstance(output, list):
        raise TypeError(f"{name}.output must be a list")
    for output_index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}.output[{output_index}] must be an object")
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            raise TypeError(f"{name}.output[{output_index}].content must be a list")
        for content_index, part in enumerate(content):
            if not isinstance(part, Mapping):
                raise TypeError(f"{name}.output[{output_index}].content[{content_index}] " "must be an object")
            if part.get("type") == "output_text":
                pieces.append(
                    _bounded_utf8_allow_empty(
                        part.get("text"),
                        name=(f"{name}.output[{output_index}].content[{content_index}].text"),
                        maximum=16 * 1024 * 1024,
                    )
                )
    return "".join(pieces)


def _reasoning_gym_extracted_answer(text: str) -> str:
    matches = list(re.finditer(r"<answer>\s?(.*?)\s?</answer>", text, flags=re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()
    return text.strip() if text.strip() else ""


def _exact_string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    for index, item in enumerate(value):
        if type(item) is not str:
            raise TypeError(f"{name}[{index}] must be a string")
    return value


def _validate_environment_verifier_response(
    verifier_response: Mapping[str, Any],
    *,
    agent_run_request: Mapping[str, Any],
    model_response: Mapping[str, Any],
    environment: str,
    reward: float,
    name: str,
) -> None:
    """Close the exact pinned per-environment successful Gym `/run` result."""
    _require_exact_keys(
        verifier_response,
        VERIFIER_RESPONSE_KEYS_BY_ENVIRONMENT[environment],
        name=name,
    )
    text = _reward_facing_text(model_response, name=f"{name}.response")
    if environment == "reasoning_gym":
        metadata = agent_run_request.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{name} reasoning verifier metadata must be an object")
        task_name = metadata.get("source_dataset")
        if type(task_name) is not str or not task_name:
            raise ValueError(f"{name} metadata.source_dataset must be a nonempty string")
        if type(verifier_response["task_name"]) is not str or verifier_response["task_name"] != task_name:
            raise ValueError(f"{name}.task_name differs from metadata.source_dataset")
        score = _require_exact_float(verifier_response["score"], name=f"{name}.score")
        server_reward = _require_exact_float(
            verifier_response["reward"],
            name=f"{name}.reward",
        )
        if score != reward or server_reward != score:
            raise ValueError(f"{name}.score/reward differs from raw reward")
        extracted = _bounded_utf8_allow_empty(
            verifier_response["extracted_answer"],
            name=f"{name}.extracted_answer",
            maximum=16 * 1024 * 1024,
        )
        if extracted != _reasoning_gym_extracted_answer(text):
            raise ValueError(f"{name}.extracted_answer differs from pinned extraction")
        return

    verifier = agent_run_request.get("verifier")
    if not isinstance(verifier, Mapping):
        raise TypeError(f"{name} verifier request must carry an object")
    if not _exact_json_equal(verifier_response["verifier"], verifier):
        raise ValueError(f"{name}.verifier differs from verifier request")
    details = verifier_response["match_details"]
    if not isinstance(details, Mapping):
        raise TypeError(f"{name}.match_details must be an object")

    if environment == "citation":
        _require_exact_keys(
            verifier,
            frozenset({"expected_markers", "patterns", "type"}),
            name=f"{name}.verifier",
        )
        if verifier["type"] != "string_match" or type(verifier["type"]) is not str:
            raise ValueError(f"{name}.verifier.type must be string_match")
        expected = _exact_string_list(
            verifier["expected_markers"],
            name=f"{name}.verifier.expected_markers",
        )
        patterns = _exact_string_list(verifier["patterns"], name=f"{name}.verifier.patterns")
        missing = [marker for marker in expected if marker not in text]
        expected_set = set(expected)
        spurious: list[str] = []
        for pattern in patterns:
            spurious.extend(
                match.group(0) for match in re.finditer(pattern, text) if match.group(0) not in expected_set
            )
        expected_details = {
            "expected": list(expected),
            "missing": missing,
            "spurious": spurious,
            "passed": not missing and not spurious,
        }
    else:
        if environment != "freeform":
            raise AssertionError(f"unsupported strict environment {environment!r}")
        _require_exact_keys(
            verifier,
            frozenset({"pattern_id", "type", "verify_min_matches", "verify_regex"}),
            name=f"{name}.verifier",
        )
        if verifier["type"] != "regex" or type(verifier["type"]) is not str:
            raise ValueError(f"{name}.verifier.type must be regex")
        if type(verifier["pattern_id"]) is not str:
            raise TypeError(f"{name}.verifier.pattern_id must be a string")
        patterns = _exact_string_list(verifier["verify_regex"], name=f"{name}.verifier.verify_regex")
        minimum = _require_nonnegative_int(
            verifier["verify_min_matches"],
            name=f"{name}.verifier.verify_min_matches",
        )
        if minimum > (1 << 31) - 1:
            raise ValueError(f"{name}.verifier.verify_min_matches exceeds int31")
        compiled = [re.compile(pattern) for pattern in patterns]
        matching_lines = sum(1 for line in text.split("\n") if any(rx.search(line) for rx in compiled))
        if matching_lines > (1 << 31) - 1:
            raise ValueError(f"{name}.match_details.matching_lines exceeds int31")
        expected_details = {
            "matching_lines": matching_lines,
            "min_matches": minimum,
            "passed": matching_lines >= minimum,
        }

    _require_exact_keys(
        details,
        frozenset(expected_details),
        name=f"{name}.match_details",
    )
    if not _exact_json_equal(details, expected_details):
        raise ValueError(f"{name}.match_details differs from pinned verifier result")
    expected_reward = 1.0 if expected_details["passed"] else 0.0
    if reward != expected_reward:
        raise ValueError(f"{name}.reward differs from pinned verifier result")


def _token_id_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    if len(value) > 131_072:
        raise ValueError(f"{name} exceeds 131072 tokens")
    result: list[int] = []
    for index, token in enumerate(value):
        if type(token) is not int or not 0 <= token <= 2_147_483_647:
            raise ValueError(f"{name}[{index}] must be an int in [0, 2147483647]")
        result.append(token)
    return result


def _response_uuid(value: Any, *, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(f"{prefix}_") or _RESPONSE_UUID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical UUID4 {prefix}_ identifier")
    return value


def _bounded_utf8_allow_empty(value: Any, *, name: str, maximum: int) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be UTF-8 text of at most {maximum} bytes")
    return value


def _bounded_utf8_nonempty(value: Any, *, name: str, maximum: int) -> str:
    text = _bounded_utf8_allow_empty(value, name=name, maximum=maximum)
    if not text:
        raise ValueError(f"{name} must be nonempty")
    return text


def _validate_generation(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, GENERATION_KEYS, name="generation")
    seed_base = _require_nonnegative_int(value["seed_base"], name="generation.seed_base")
    max_new_tokens = _require_positive_int(value["max_new_tokens"], name="generation.max_new_tokens")
    temperature = _require_exact_float(value["temperature"], name="generation.temperature")
    top_p = _require_exact_float(value["top_p"], name="generation.top_p")
    top_k = value["top_k"]
    if top_k is not None:
        _require_nonnegative_int(top_k, name="generation.top_k")
    if temperature < 0.0 or not 0.0 <= top_p <= 1.0:
        raise ValueError("generation sampling values are out of range")
    result = {
        "seed_base": seed_base,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    if not _exact_json_equal(
        result,
        {
            "seed_base": 42,
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
        },
    ):
        raise ValueError("strict transcript generation policy is not frozen")
    return result


def _validate_transcript_bindings(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, TRANSCRIPT_BINDING_KEYS, name="transcript bindings")
    result = dict(value)
    for key in TRANSCRIPT_BINDING_KEYS - {"job_id"}:
        _require_digest(result[key], name=f"bindings.{key}")
    _require_job_id(result["job_id"], name="bindings.job_id")
    return result


def _validate_replay_ledger_bindings(
    value: Mapping[str, Any], *, pair_id: str, environment: str, attempt_id: str
) -> dict[str, Any]:
    _require_exact_keys(value, REPLAY_LEDGER_BINDING_KEYS, name="replay bindings")
    result = dict(value)
    for key in REPLAY_LEDGER_BINDING_KEYS - {
        "job_id",
        "process",
        "restart_count",
    }:
        _require_digest(result[key], name=f"replay bindings.{key}")
    _require_job_id(result["job_id"], name="replay bindings.job_id")
    if type(result["restart_count"]) is not int or result["restart_count"] != 0:
        raise ValueError("captured replay requires restart_count=0")
    result["process"] = _validate_process(result["process"])
    expected_run = replay_run_id(environment=environment, pair_id=pair_id, attempt_id=attempt_id)
    if result["run_id"] != expected_run:
        raise ValueError("replay ledger run_id is not deterministic")
    return result


def _validate_identity(*, pair_id: Any, environment: Any) -> None:
    _require_safe_id(pair_id, name="pair_id", maximum=64)
    if environment not in STRICT_ENVIRONMENTS:
        raise ValueError(f"environment must be one of {sorted(STRICT_ENVIRONMENTS)}, got {environment!r}")


def _require_attempt(value: Any) -> str:
    if value not in REPLAY_ATTEMPTS:
        raise ValueError(f"attempt_id must be one of {list(REPLAY_ATTEMPTS)}")
    return value


def _artifact_reference(value: Mapping[str, Any], *, name: str, expected_schema: str) -> dict[str, str]:
    _require_exact_keys(value, ARTIFACT_REFERENCE_KEYS, name=name)
    path = str(_canonical_absolute_path(value["path"], name=f"{name}.path"))
    if value["schema"] != expected_schema:
        raise ValueError(f"{name}.schema must be exactly {expected_schema!r}")
    return {
        "path": path,
        "schema": expected_schema,
        "sha256": _require_digest(value["sha256"], name=f"{name}.sha256"),
    }


def _validate_job(value: Mapping[str, Any], *, replay_execution_manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, JOB_KEYS, name="job")
    result = copy.deepcopy(dict(value))
    for key in ("account", "partition", "qos"):
        _require_ascii(result[key], name=f"job.{key}", maximum=128)
    _require_ascii(result["name"], name="job.name", maximum=255)
    pair = _load_lifecycle_pair_manifest(replay_execution_manifest)
    campaign = pair["campaign"]
    expected = {
        "account": campaign["slurm"]["account"],
        "name": replay_execution_manifest["scheduler_submission"]["identity"]["job_name"],
        "num_nodes": campaign["nodes"],
        "partition": campaign["slurm"]["partition"],
        "qos": campaign["slurm"]["qos"],
        "gpus_per_node": 4,
        "restart_count": 0,
    }
    if not _exact_json_equal(result, expected):
        raise ValueError("job allocation/identity differs from authenticated Pair")
    return result


def _validate_hardware(
    value: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(value, HARDWARE_KEYS, name="hardware")
    if value["schema"] != HARDWARE_OBSERVATION_SCHEMA:
        raise ValueError("unexpected hardware observation schema")
    if type(value["gpu_row_count"]) is not int or value["gpu_row_count"] != 4:
        raise ValueError("hardware GPU row count must be exact integer 4")
    rows = value["ordered_rows"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("hardware ordered rows must be exact K=4")
    normalized_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        _require_exact_keys(raw_row, HARDWARE_ROW_KEYS, name=f"hardware ordered row {index}")
        row = dict(raw_row)
        if type(row["index"]) is not int or row["index"] != index:
            raise ValueError("hardware ordered row indices must be exact 0..3")
        model = _require_ascii(row["gpu_model"], name=f"hardware ordered row {index} model", maximum=128)
        driver = _require_ascii(
            row["driver_version"],
            name=f"hardware ordered row {index} driver",
            maximum=64,
        )
        raw = _require_ascii(row["raw"], name=f"hardware ordered row {index} raw", maximum=255)
        if raw != f"{model}, {driver}":
            raise ValueError("hardware ordered row raw text differs from parsed fields")
        if model != "NVIDIA GB200":
            raise ValueError("captured replay requires NVIDIA GB200")
        if driver != "580.126.20":
            raise ValueError("captured replay requires NVIDIA driver 580.126.20")
        normalized_rows.append(row)
    models = {row["gpu_model"] for row in normalized_rows}
    drivers = {row["driver_version"] for row in normalized_rows}
    if models != {"NVIDIA GB200"} or value["gpu_model"] != next(iter(models)):
        raise ValueError("hardware GPU model summary differs from ordered rows")
    if drivers != {"580.126.20"} or value["driver_version"] != next(iter(drivers)):
        raise ValueError("hardware driver summary differs from ordered rows")
    raw_output = ("\n".join(row["raw"] for row in normalized_rows) + "\n").encode("ascii")
    if value["raw_output_sha256"] != hashlib.sha256(raw_output).hexdigest():
        raise ValueError("hardware raw nvidia-smi output digest differs")
    expected_rows_sha256 = domain_sha256(HARDWARE_ORDERED_ROWS_HASH_LABEL, normalized_rows)
    if value["ordered_rows_sha256"] != expected_rows_sha256:
        raise ValueError("hardware ordered-row digest differs")
    tool = _file_reference(value["nvidia_smi"], name="hardware.nvidia_smi")
    expected_tool = _file_reference(
        replay_execution_manifest["runtime_tools"]["document"]["host"]["nvidia_smi"],
        name="authenticated runtime_tools host nvidia_smi",
    )
    if tool != expected_tool:
        raise ValueError("hardware nvidia_smi differs from authenticated runtime tool")
    _load_bound_runtime_tool_bytes(Path(tool["path"]), expected_sha256=tool["sha256"])
    result = copy.deepcopy(dict(value))
    result["ordered_rows"] = normalized_rows
    result["nvidia_smi"] = tool
    return result


def _validate_scheduler_device_environment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        SCHEDULER_DEVICE_ENVIRONMENT_KEYS,
        name="scheduler_device_environment",
    )
    result = dict(value)
    if result["schema"] != SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA:
        raise ValueError("unexpected scheduler device environment schema")
    cuda = _four_distinct_decimal_ids(result["cuda_visible_devices"], name="cuda_visible_devices")
    ordinal = result["gpu_device_ordinal"]
    if ordinal is not None and ordinal != cuda:
        raise ValueError("gpu_device_ordinal must equal cuda_visible_devices or null")
    nvidia = result["nvidia_visible_devices"]
    if nvidia is not None:
        _require_ascii(nvidia, name="nvidia_visible_devices", maximum=255)
        if nvidia not in {cuda, "all", "none", "void"}:
            tokens = nvidia.split(",")
            if (
                len(tokens) != 4
                or len(set(tokens)) != 4
                or any(_GPU_UUID_RE.fullmatch(token) is None for token in tokens)
            ):
                raise ValueError("nvidia_visible_devices has invalid GPU identities")
    rocr = result["rocr_visible_devices"]
    if rocr is not None:
        _four_distinct_decimal_ids(rocr, name="rocr_visible_devices")
    ze = result["ze_affinity_mask"]
    if ze is not None:
        _require_ascii(ze, name="ze_affinity_mask", maximum=255)
        tokens = ze.split(",")
        if not 1 <= len(tokens) <= 64 or any(_ZE_DEVICE_RE.fullmatch(token) is None for token in tokens):
            raise ValueError("ze_affinity_mask has invalid device tokens")
        identities = {
            (
                int(token.split(".")[0], 10),
                int(token.split(".")[1], 10) if "." in token else 0,
            )
            for token in tokens
        }
        if len(identities) != len(tokens):
            raise ValueError("ze_affinity_mask device identities repeat")
    return result


def validate_scheduler_device_environment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the normalized exact six-key scheduler device observation."""
    return _validate_scheduler_device_environment(value)


def _validate_process(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, PROCESS_KEYS, name="process")
    result = dict(value)
    _require_digest(result["boot_id_sha256"], name="process.boot_id_sha256")
    _require_bounded_positive_int(result["pid"], name="process.pid", maximum=(1 << 31) - 1)
    _require_bounded_positive_int(
        result["start_time_ticks"],
        name="process.start_time_ticks",
        maximum=(1 << 63) - 1,
    )
    return result


def _validate_driver_scorer_process_join_v2(
    *,
    driver_process: Mapping[str, Any],
    scorer_process: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Close the independent driver/scorer identities onto one boot."""
    driver = _validate_process(driver_process)
    scorer = _strict_json_object(scorer_process, name="scorer process identity")
    _require_exact_keys(
        scorer,
        frozenset({"boot_id", "hostname", "pid", "start_ticks"}),
        name="scorer process identity",
    )
    boot_id = scorer["boot_id"]
    if (
        type(boot_id) is not str
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
        or type(scorer["hostname"]) is not str
        or not scorer["hostname"]
    ):
        raise ValueError("scorer process identity differs")
    _require_bounded_positive_int(
        scorer["pid"],
        name="scorer process identity.pid",
        maximum=(1 << 31) - 1,
    )
    _require_bounded_positive_int(
        scorer["start_ticks"],
        name="scorer process identity.start_ticks",
        maximum=(1 << 63) - 1,
    )
    boot_sha256 = hashlib.sha256(f"{boot_id}\n".encode("ascii")).hexdigest()
    if driver["boot_id_sha256"] != boot_sha256:
        raise ValueError("driver/scorer boot identities differ")
    if driver["pid"] == scorer["pid"]:
        raise ValueError("driver and scorer must be distinct processes")
    return driver, scorer


def _validate_scorer_resource_process_v2(
    *,
    replay_execution_manifest: Mapping[str, Any],
    driver_process: Mapping[str, Any],
    resource_process: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind retained scorer process evidence to M4 and the replay driver."""
    manifest = _strict_json_object(
        replay_execution_manifest,
        name="replay execution manifest",
    )
    resource = _strict_json_object(
        resource_process,
        name="scorer resource process",
    )
    container_python = _file_reference(
        manifest["runtime_tools"]["document"]["container"]["python"],
        name="runtime_tools container python",
    )
    expected_proc_exe = container_python["path"]
    expected_base_prefix = str(Path(expected_proc_exe).parent.parent)
    if resource.get("proc_exe") != expected_proc_exe or resource.get("sys_base_prefix") != expected_base_prefix:
        raise ValueError("scorer base interpreter differs from authenticated M4")
    driver = _validate_process(driver_process)
    if type(resource.get("ppid")) is not int or resource["ppid"] != driver["pid"]:
        raise ValueError("scorer parent process differs from replay driver")
    return _validate_driver_scorer_process_join_v2(
        driver_process=driver,
        scorer_process={name: resource[name] for name in ("boot_id", "hostname", "pid", "start_ticks")},
    )


def _validate_runtime_tools(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(value, RUNTIME_TOOL_KEYS, name="runtime_tools")
    return {
        "manifest_path": str(_canonical_absolute_path(value["manifest_path"], name="runtime_tools.manifest_path")),
        "manifest_sha256": _require_digest(value["manifest_sha256"], name="runtime_tools.manifest_sha256"),
    }


def _validate_replay_runtime(value: Mapping[str, Any], *, expected_bundle_sha256: str) -> dict[str, Any]:
    _require_exact_keys(value, REPLAY_RUNTIME_KEYS, name="replay_runtime")
    result = dict(value)
    expected = {
        "required": True,
        "bundle_sha256": expected_bundle_sha256,
        "entries": 4,
        "hits": 4,
        "misses": 0,
        "reuses": 0,
        "pending": 0,
        "streaming_rejections": 0,
    }
    for key, expected_value in expected.items():
        if result[key] != expected_value or type(result[key]) is not type(expected_value):
            raise ValueError(f"replay_runtime.{key} must be exactly {expected_value!r}")
    _require_digest(result["ready_marker_sha256"], name="replay_runtime.ready_marker_sha256")
    _require_digest(result["hit_markers_sha256"], name="replay_runtime.hit_markers_sha256")
    return result


def _validate_captured_replay_outputs(
    value: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    authenticated_job_id: str,
    driver_process: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, dict[str, str]]:
    _require_exact_keys(value, REPLAY_OUTPUT_V2_KEYS, name="replay outputs")
    manifest = replay_execution_manifest
    declared = manifest["artifacts"]["outputs"]
    references: dict[str, dict[str, str]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name in sorted(REPLAY_OUTPUT_V2_KEYS):
        declaration = declared[name]
        reference = _artifact_reference(
            value[name],
            name=f"replay output {name}",
            expected_schema=declaration["schema"],
        )
        if reference["path"] != declaration["path"]:
            raise ValueError(f"replay output {name} path differs from manifest")
        references[name] = reference

    transcript_ref = references["transcript_bundle"]
    transcript, _ = load_evidence_document(
        path=transcript_ref["path"],
        expected_sha256=transcript_ref["sha256"],
        trailing_lf=False,
    )
    validate_transcript_bundle(transcript)
    documents["transcript_bundle"] = transcript

    score_ref = references["scorer_call_index"]
    score = _load_finalized_scorer_call_index(
        score_ref,
        replay_execution_manifest=manifest,
        authenticated_job_id=authenticated_job_id,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    documents["scorer_call_index"] = score

    consumption_ref = references["transport_consumption"]
    consumption, _ = load_evidence_document(
        path=consumption_ref["path"],
        expected_sha256=consumption_ref["sha256"],
        trailing_lf=False,
    )
    from nemo_rl.utils.strict_model_transport_replay_v3 import (
        validate_strict_model_transport_replay_consumption_v3,
    )

    validate_strict_model_transport_replay_consumption_v3(
        consumption,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    documents["transport_consumption"] = consumption

    source_step1 = manifest["source_capture"]["step1_evidence"]
    source_transcript_ref = _artifact_reference(
        source_step1["transcript_bundle"],
        name="source transcript bundle",
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
    )
    source_transcript, _ = load_evidence_document(
        path=source_transcript_ref["path"],
        expected_sha256=source_transcript_ref["sha256"],
        trailing_lf=False,
    )
    validate_transcript_bundle(source_transcript)
    validate_captured_replay_source_join(
        source_transcript_bundle=source_transcript,
        replay_transcript_bundle=transcript,
    )

    ledger_ref = references["replay_ledger"]
    ledger, _ = load_evidence_document(
        path=ledger_ref["path"],
        expected_sha256=ledger_ref["sha256"],
        trailing_lf=False,
    )
    validate_captured_replay_step1_ledger(
        ledger,
        source_transcript_document=source_transcript,
        transcript_document=transcript,
    )
    validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)
    documents["replay_ledger"] = ledger

    _close_replay_output_documents(
        documents,
        references=references,
        replay_execution_manifest=manifest,
        submission_receipt=submission_receipt,
        authenticated_job_id=authenticated_job_id,
        driver_process=driver_process,
        driver_scheduler_device_environment=driver_scheduler_device_environment,
        source_transcript_ref=source_transcript_ref,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return references


def _load_finalized_scorer_call_index(
    reference: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_job_id: str,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    try:
        from nemo_rl.environments.strict_gym_child_runtime_v2 import (
            load_finalized_format_verification_call_index,
            load_finalized_reasoning_score_call_index,
        )
    except ImportError as error:
        raise RuntimeError("public finalized scorer call loader is unavailable") from error
    if _RESULT_PROFILE_IDS_V2.get(expected_environment) != expected_profile_id:
        raise ValueError("scorer loader differs from outer profile authority")
    receipt_root = Path(reference["path"]).parent
    common_arguments = {
        "expected_sha256": reference["sha256"],
        "expected_receipt_root": receipt_root,
        "expected_pair_id": replay_execution_manifest["pair_id"],
        "expected_job_id": authenticated_job_id,
    }
    if expected_environment == "reasoning_gym":
        if expected_profile_id != _RESULT_PROFILE_IDS_V2["reasoning_gym"]:
            raise ValueError("reasoning scorer differs from outer profile authority")
        loaded, digest = load_finalized_reasoning_score_call_index(
            Path(reference["path"]),
            **common_arguments,
        )
        if (
            loaded.get("schema") != REASONING_SCORE_CALL_INDEX_SCHEMA
            or loaded.get("environment") != "reasoning_gym"
            or "profile_id" in loaded
        ):
            raise ValueError("reasoning scorer terminal differs from outer profile authority")
    elif expected_environment in {"citation", "freeform"}:
        loaded, digest = load_finalized_format_verification_call_index(
            Path(reference["path"]),
            **common_arguments,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
    else:
        raise ValueError("scorer loader profile dispatch is unsupported")
    if digest != reference["sha256"]:
        raise ValueError("scorer call loader digest differs")
    return _strict_json_object(loaded, name="scorer call terminal index")


def _close_replay_output_documents(
    documents: Mapping[str, Mapping[str, Any]],
    *,
    references: Mapping[str, Mapping[str, str]],
    replay_execution_manifest: Mapping[str, Any],
    submission_receipt: Mapping[str, Any],
    authenticated_job_id: str,
    driver_process: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    source_transcript_ref: Mapping[str, Any],
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    manifest = replay_execution_manifest
    transcript = documents["transcript_bundle"]
    expected_transcript_envelope = {
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "captured_replay",
        "attempt_id": manifest["attempt_id"],
    }
    _require_exact_projection(transcript, expected_transcript_envelope, name="replay transcript")
    source_snapshot = manifest["replay_contract"]["source_snapshot"]["ref"]
    expected_transcript_bindings = {
        "pair_manifest_sha256": manifest["pair"]["manifest"]["sha256"],
        "submission_receipt_sha256": document_sha256(submission_receipt, trailing_lf=True),
        "job_id": authenticated_job_id,
        "run_id": replay_run_id(
            environment=manifest["environment"],
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
        "fixture_sha256": manifest["artifacts"]["fixture"]["sha256"],
        "verifier_source_sha256": manifest["replay_contract"]["gym_scorer"]["resources"]["verifier_source"]["sha256"],
        "config_sha256": manifest["replay_contract"]["selected_config"]["sha256"],
        "snapshot_manifest_sha256": source_snapshot["manifest_sha256"],
    }
    if not _exact_json_equal(transcript["bindings"], expected_transcript_bindings):
        raise ValueError("replay transcript bindings differ from lifecycle authority")

    ledger = documents["replay_ledger"]
    source_step1 = manifest["source_capture"]["step1_evidence"]
    expected_ledger_bindings = {
        **expected_transcript_bindings,
        "restart_count": 0,
        "pair_campaign_sha256": manifest["pair"]["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": manifest["pair"]["pair_campaign_reward_and_advantage_sha256"],
        "process": copy.deepcopy(dict(driver_process)),
    }
    expected_ledger = {
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "arm": "on",
        "mode": "captured_replay",
        "attempt_id": manifest["attempt_id"],
        "source_main_ledger_sha256": source_step1["main_ledger"]["sha256"],
        "source_transcript_bundle": source_transcript_ref,
        "transcript_bundle": references["transcript_bundle"],
        "bindings": expected_ledger_bindings,
    }
    _require_exact_projection(ledger, expected_ledger, name="replay ledger")

    consumption = documents["transport_consumption"]
    transport = source_step1["model_transport"]
    expected_source = {
        "arm": "off",
        "authenticated_job_id": manifest["source_capture"]["authenticated_job"]["job_id"],
        "pair_manifest_sha256": manifest["pair"]["manifest"]["sha256"],
        "submission_receipt_sha256": manifest["pair"]["submission_receipt"]["sha256"],
        "main_ledger": source_step1["main_ledger"],
        "transcript_bundle": source_step1["transcript_bundle"],
        "transport_bundle": transport["bundle"],
        "transport_manifest": transport["manifest"],
        "raw_log": transport["raw_log"],
        "ordered_entries_sha256": transport["ordered_entries_sha256"],
    }
    expected_consumption = {
        "pair_id": manifest["pair_id"],
        "environment": manifest["environment"],
        "source": expected_source,
    }
    _require_exact_projection(consumption, expected_consumption, name="transport consumption")
    replay = consumption["replay"]
    expected_replay = {
        "attempt_id": manifest["attempt_id"],
        "replay_execution_manifest_sha256": submission_receipt["replay_execution_manifest"]["sha256"],
        "authenticated_job_id": authenticated_job_id,
        "process": driver_process,
        "scheduler_device_environment": driver_scheduler_device_environment,
    }
    _require_exact_projection(replay, expected_replay, name="transport replay")
    scorer = replay["scorer_evidence"]
    if (
        scorer["terminal_index"] != references["scorer_call_index"]
        or scorer["status"] != "authenticated"
        or type(scorer.get("original_process_reaped")) is not bool
        or scorer["original_process_reaped"] is not True
    ):
        raise ValueError("transport consumption scorer terminal differs")

    score = documents["scorer_call_index"]
    score_quiescence = score.get("quiescence")
    if (
        type(score_quiescence) is not dict
        or type(score_quiescence.get("original_process_reaped")) is not bool
        or score_quiescence["original_process_reaped"] is not True
    ):
        raise ValueError("scorer call index does not prove original process reaping")
    _close_score_transcript_join(
        score,
        transcript,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )

    entries = consumption["entries"]
    transcript_entries = transcript["entries"]
    if len(entries) != K4_SAMPLES or len(transcript_entries) != K4_SAMPLES:
        raise ValueError("replay terminal outputs are not exact K=4")
    for index, (entry, transcript_entry) in enumerate(zip(entries, transcript_entries, strict=True)):
        expected_entry = {
            "rollout_index": index,
            "generation_seed": transcript_entry["generation_seed"],
            "source_model_transport_entry_sha256": transcript_entry["model_transport_entry_sha256"],
            "source_request_body_sha256": transcript_entry["model_transport_request_body_sha256"],
            "source_response_body_sha256": transcript_entry["model_transport_response_body_sha256"],
            "generation_request_sha256": transcript_entry["generation_request_sha256"],
            "model_response_sha256": transcript_entry["model_response_sha256"],
            "agent_run_request_sha256": transcript_entry["agent_run_request_sha256"],
            "derived_verifier_request_sha256": transcript_entry["derived_verifier_request_sha256"],
            "fresh_verifier_response_sha256": transcript_entry["verifier_response_sha256"],
            "fresh_native_reward": transcript_entry["raw_environment_reward"],
        }
        _require_exact_projection(entry, expected_entry, name=f"transport consumption entry {index}")


def _close_score_transcript_join(
    score_index: Mapping[str, Any],
    transcript: Mapping[str, Any],
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    if _RESULT_PROFILE_IDS_V2.get(expected_environment) != expected_profile_id:
        raise ValueError("scorer call index differs from outer profile authority")
    if expected_environment == "reasoning_gym":
        _close_reasoning_score_transcript_join(
            score_index,
            transcript,
            expected_profile_id=expected_profile_id,
        )
        return
    _close_format_score_transcript_join(
        score_index,
        transcript,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )


def _close_reasoning_score_transcript_join(
    score_index: Mapping[str, Any],
    transcript: Mapping[str, Any],
    *,
    expected_profile_id: str,
) -> None:
    """Bind environment-only V1 reasoning bytes to outer V2 profile authority."""
    from nemo_rl.environments.strict_gym_child_runtime_v2 import (
        reasoning_score_call_expectation,
    )

    if (
        expected_profile_id != _RESULT_PROFILE_IDS_V2["reasoning_gym"]
        or score_index.get("environment") != "reasoning_gym"
        or score_index.get("schema") != REASONING_SCORE_CALL_INDEX_SCHEMA
        or "profile_id" in score_index
    ):
        raise ValueError("reasoning scorer call index differs from outer profile authority")
    quiescence = score_index.get("quiescence")
    if type(quiescence) is not dict or quiescence.get("original_process_reaped") is not True:
        raise ValueError("scorer call index does not prove process reaping")
    calls = score_index.get("calls")
    entries = transcript.get("entries")
    if type(calls) is not list or len(calls) != K4_SAMPLES or type(entries) is not list or len(entries) != K4_SAMPLES:
        raise ValueError("scorer call index is not exact K=4")
    for index, (call, entry_value) in enumerate(zip(calls, entries, strict=True)):
        entry = _strict_json_object(entry_value, name=f"reasoning transcript entry {index}")
        request = _strict_json_object(
            entry.get("derived_verifier_request"),
            name=f"reasoning verifier request {index}",
        )
        response = _strict_json_object(
            entry.get("verifier_response"),
            name=f"reasoning verifier response {index}",
        )
        metadata = _strict_json_object(
            request.get("metadata"),
            name=f"reasoning verifier metadata {index}",
        )
        task_name = metadata.get("source_dataset")
        extracted_answer = response.get("extracted_answer")
        score = _require_exact_float(
            response.get("score"),
            name=f"reasoning verifier score {index}",
        )
        server_reward = _require_exact_float(
            response.get("reward"),
            name=f"reasoning verifier server reward {index}",
        )
        reward = _require_exact_float(
            entry.get("raw_environment_reward"),
            name=f"reasoning transcript reward {index}",
        )
        if (
            task_name != "knights_knaves"
            or type(task_name) is not str
            or response.get("task_name") != task_name
            or type(response.get("task_name")) is not str
            or type(extracted_answer) is not str
            or not 0.0 <= score <= 1.0
            or server_reward != score
            or score != reward
        ):
            raise ValueError(f"reasoning scorer call {index + 1} result differs from transcript")
        scorer_entry = {
            "question": copy.deepcopy(request.get("question")),
            "answer": copy.deepcopy(request.get("answer")),
            "metadata": copy.deepcopy(metadata),
        }
        expectation = reasoning_score_call_expectation(
            task_name=task_name,
            answer=extracted_answer,
            entry=scorer_entry,
            float_result=reward,
        )
        expected = {
            "sequence": index + 1,
            "task_name": expectation["task_name"],
            "input": {
                "answer_sha256": expectation["answer_sha256"],
                "entry_sha256": expectation["entry_sha256"],
            },
            "float_result": expectation["float_result"],
        }
        _require_exact_projection(call, expected, name=f"reasoning scorer call {index + 1}")


def _close_format_score_transcript_join(
    score_index: Mapping[str, Any],
    transcript: Mapping[str, Any],
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    from nemo_rl.environments.strict_gym_child_runtime_v2 import (
        format_verification_call_expectation,
    )

    if (
        expected_environment not in {"citation", "freeform"}
        or score_index.get("environment") != expected_environment
        or score_index.get("profile_id") != expected_profile_id
        or score_index.get("schema") != FORMAT_VERIFICATION_CALL_INDEX_SCHEMA
    ):
        raise ValueError("format scorer call index profile identity differs")
    quiescence = score_index.get("quiescence")
    if type(quiescence) is not dict or quiescence.get("original_process_reaped") is not True:
        raise ValueError("scorer call index does not prove process reaping")
    calls = score_index.get("calls")
    entries = transcript.get("entries")
    if type(calls) is not list or len(calls) != K4_SAMPLES or type(entries) is not list or len(entries) != K4_SAMPLES:
        raise ValueError("scorer call index is not exact K=4")
    for index, (call, entry) in enumerate(zip(calls, entries, strict=True)):
        request = entry["derived_verifier_request"]
        if type(request) is not dict:
            raise TypeError("derived format-verifier request must be an exact object")
        format_request = {
            name: copy.deepcopy(request[name]) for name in ("responses_create_params", "response", "verifier")
        }
        expectation = format_verification_call_expectation(
            environment=expected_environment,
            derived_verifier_request=format_request,
            verifier_response=entry["verifier_response"],
        )
        expected = {
            "sequence": index + 1,
            "method": expectation["method"],
            "input": {
                name: expectation[name]
                for name in (
                    "request_sha256",
                    "verifier_sha256",
                    "response_text_sha256",
                )
            },
            "outcome": {
                "kind": "returned",
                "response_sha256": expectation["response_sha256"],
                "match_details_sha256": expectation["match_details_sha256"],
                "float_result": expectation["float_result"],
            },
        }
        _require_exact_keys(
            call,
            frozenset({"sequence", "method", "input", "outcome", "receipt"}),
            name=f"scorer call {index + 1}",
        )
        _require_exact_projection(call, expected, name=f"scorer call {index + 1}")
        if entry["raw_environment_reward"] != expectation["float_result"]:
            raise ValueError(f"scorer call {index + 1} reward differs from transcript")


def _replay_runtime_attestation(
    replay_execution_manifest: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema": REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA,
        "scorer_profile": copy.deepcopy(replay_execution_manifest["scorer_profile"]),
        "original_process_reaped": True,
        "environment": replay_execution_manifest["environment"],
        "profile_id": replay_execution_manifest["scorer_profile"]["profile_id"],
        "requirements": copy.deepcopy(replay_execution_manifest["runtime_attestation_requirements"]),
        **{name: copy.deepcopy(outputs[name]) for name in REPLAY_OUTPUT_V2_KEYS},
    }


def _validated_lifecycle_manifest(
    value: Mapping[str, Any],
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    """Validate a profiled replay manifest after authenticating its program."""
    document = _strict_json_object(value, name="replay execution manifest")
    if type(expected_environment) is not str or type(expected_profile_id) is not str:
        raise TypeError("expected environment/profile must be exact strings")
    if document.get("schema") != REPLAY_EXECUTION_MANIFEST_V2_SCHEMA:
        raise ValueError("unexpected profiled replay execution manifest schema")
    if document.get("environment") != expected_environment:
        raise ValueError("profiled replay manifest environment differs")
    _authenticate_program_closure_v2(document)

    # The manifest validator and registry are executable program members.  They
    # are intentionally imported only after the entire admitted closure above
    # has been stable-read and matched to its caller-carried SHA-256 values.
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
        validate_replay_execution_manifest_v2,
    )

    if type(authenticated_source) is not AuthenticatedOffSourceCapture:
        raise TypeError("authenticated_source must be an authenticated OFF capability")
    validate_replay_execution_manifest_v2(
        document,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if not _exact_json_equal(document["source_capture"], authenticated_source.source_capture):
        raise ValueError("replay manifest source capture differs from capability")
    return document


def _authenticate_program_closure_v2(document: Mapping[str, Any]) -> None:
    """Stable-read every admitted executable before importing any V2 member."""
    replay_contract = document.get("replay_contract")
    if type(replay_contract) is not dict:
        raise TypeError("profiled replay contract must be an exact object")
    program = replay_contract.get("program")
    _require_exact_keys(
        program,
        frozenset(REPLAY_PROGRAM_V2_PATHS),
        name="profiled replay program closure",
    )
    source_snapshot = replay_contract.get("source_snapshot")
    if type(source_snapshot) is not dict or type(source_snapshot.get("ref")) is not dict:
        raise TypeError("profiled replay source snapshot reference must be an exact object")
    snapshot_root = _canonical_absolute_path(
        source_snapshot["ref"].get("path"),
        name="profiled replay source snapshot path",
    )
    if snapshot_root == Path("/"):
        raise ValueError("profiled replay source snapshot must not be root")
    for name, expected_relative in REPLAY_PROGRAM_V2_PATHS.items():
        reference = program[name]
        _require_exact_keys(
            reference,
            TOOL_REFERENCE_KEYS,
            name=f"profiled replay program {name}",
        )
        relative = _require_ascii(
            reference["path"],
            name=f"profiled replay program {name}.path",
            maximum=4096,
        )
        relative_path = Path(relative)
        if (
            relative != expected_relative
            or relative_path.is_absolute()
            or "." in relative_path.parts
            or ".." in relative_path.parts
            or str(relative_path) != relative
        ):
            raise ValueError(f"profiled replay program {name} path differs")
        expected_sha256 = _require_digest(
            reference["sha256"],
            name=f"profiled replay program {name}.sha256",
        )
        _load_bound_runtime_tool_bytes(
            snapshot_root / relative_path,
            expected_sha256=expected_sha256,
        )


def _load_lifecycle_pair_manifest(
    replay_execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    pair_binding = replay_execution_manifest.get("pair")
    if not isinstance(pair_binding, Mapping):
        raise TypeError("replay manifest pair binding must be an object")
    pair_ref = _artifact_reference(
        pair_binding.get("manifest"),
        name="replay Pair manifest",
        expected_schema=PAIR_MANIFEST_SCHEMA,
    )
    pair, _ = load_evidence_document(
        path=pair_ref["path"],
        expected_sha256=pair_ref["sha256"],
        trailing_lf=True,
    )
    if pair.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unexpected replay Pair manifest schema")
    return pair


def _load_lifecycle_pair_submission_receipt(
    replay_execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    pair_binding = replay_execution_manifest.get("pair")
    if not isinstance(pair_binding, Mapping):
        raise TypeError("replay manifest pair binding must be an object")
    receipt_ref = _artifact_reference(
        pair_binding.get("submission_receipt"),
        name="Pair submission receipt",
        expected_schema=PAIR_SUBMISSION_RECEIPT_SCHEMA,
    )
    receipt, _ = load_evidence_document(
        path=receipt_ref["path"],
        expected_sha256=receipt_ref["sha256"],
        trailing_lf=True,
    )
    if receipt.get("schema") != PAIR_SUBMISSION_RECEIPT_SCHEMA:
        raise ValueError("unexpected Pair submission receipt schema")
    return receipt


def _require_replay_job_disjoint_from_pair(
    replay_execution_manifest: Mapping[str, Any],
    replay_job_id: Any,
    *,
    name: str,
) -> str:
    """Reject reuse of either authenticated Pair cohort job identity."""
    job_id = _require_job_id(replay_job_id, name=name)
    receipt = _load_lifecycle_pair_submission_receipt(replay_execution_manifest)
    authenticated = receipt.get("authenticated_jobs")
    if not isinstance(authenticated, Mapping):
        raise TypeError("Pair authenticated_jobs must be an object")
    _require_exact_keys(authenticated, frozenset({"off", "on"}), name="Pair authenticated_jobs")
    pair_job_ids: dict[str, str] = {}
    for arm in ("off", "on"):
        identities = authenticated[arm]
        if not isinstance(identities, list) or len(identities) != 1 or not isinstance(identities[0], Mapping):
            raise ValueError(f"Pair authenticated_jobs.{arm} must contain one identity")
        pair_job_ids[arm] = _require_job_id(identities[0].get("job_id"), name=f"Pair authenticated {arm} job_id")
    if pair_job_ids["off"] == pair_job_ids["on"]:
        raise ValueError("Pair authenticated OFF and ON job IDs must differ")
    if job_id in pair_job_ids.values():
        raise ValueError(f"{name} reuses an authenticated Pair OFF/ON job ID")
    return job_id


def _require_exact_projection(actual: Mapping[str, Any], expected: Mapping[str, Any], *, name: str) -> None:
    for key, expected_value in expected.items():
        if key not in actual or not _exact_json_equal(actual[key], expected_value):
            raise ValueError(f"{name}.{key} differs")


def _require_loaded_document_matches(
    reference: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    trailing_lf: bool,
    name: str,
) -> None:
    schema = _require_ascii(reference.get("schema"), name=f"{name}.schema", maximum=255)
    canonical_ref = _artifact_reference(reference, name=name, expected_schema=schema)
    loaded, _ = load_evidence_document(
        path=canonical_ref["path"],
        expected_sha256=canonical_ref["sha256"],
        trailing_lf=trailing_lf,
    )
    if loaded.get("schema") != schema or not _exact_json_equal(loaded, expected):
        raise ValueError(f"{name} differs from immutable bytes")


def _file_reference(value: Mapping[str, Any], *, name: str) -> dict[str, str]:
    _require_exact_keys(value, TOOL_REFERENCE_KEYS, name=name)
    return {
        "path": str(_canonical_absolute_path(value["path"], name=f"{name}.path")),
        "sha256": _require_digest(value["sha256"], name=f"{name}.sha256"),
    }


def _validate_replay_scheduler_tools(
    value: Mapping[str, Any], *, replay_execution_manifest: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    _require_exact_keys(value, REPLAY_SCHEDULER_TOOL_KEYS, name="scheduler_tools")
    result = {
        name: _file_reference(value[name], name=f"scheduler_tools.{name}")
        for name in sorted(REPLAY_SCHEDULER_TOOL_KEYS)
    }
    runtime_document = replay_execution_manifest["runtime_tools"]["document"]
    host_tools = runtime_document["host"]
    pair_receipt = _load_lifecycle_pair_submission_receipt(replay_execution_manifest)
    authenticated_tools = pair_receipt["scheduler_tools"]
    for name in REPLAY_SCHEDULER_TOOL_KEYS:
        if not _exact_json_equal(result[name], host_tools[name]) or not _exact_json_equal(
            result[name], authenticated_tools[name]
        ):
            raise ValueError(f"scheduler_tools.{name} differs from authenticated runtime tools")
    return result


def _validate_replay_scheduler_client_environment(
    value: Mapping[str, Any], *, replay_execution_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(
        value,
        REPLAY_SCHEDULER_CLIENT_ENVIRONMENT_KEYS,
        name="scheduler_client_environment",
    )
    variables = value["variables"]
    _require_exact_keys(
        variables,
        REPLAY_SCHEDULER_CLIENT_VARIABLE_KEYS,
        name="scheduler_client_environment.variables",
    )
    if value["ambient_merge"] is not False or variables["LC_ALL"] != "C":
        raise ValueError("scheduler client environment is not closed")
    result = {
        "ambient_merge": False,
        "env": _file_reference(value["env"], name="scheduler_client_environment.env"),
        "variables": {
            "LC_ALL": "C",
            "SLURM_CONF": _file_reference(
                variables["SLURM_CONF"],
                name="scheduler_client_environment.variables.SLURM_CONF",
            ),
        },
    }
    pair_receipt = _load_lifecycle_pair_submission_receipt(replay_execution_manifest)
    expected = pair_receipt["scheduler_tools"]["client_environment"]
    if not _exact_json_equal(result, expected):
        raise ValueError("scheduler client environment differs from authenticated Pair receipt")
    host_env = replay_execution_manifest["runtime_tools"]["document"]["host"]["env"]
    if not _exact_json_equal(result["env"], host_env):
        raise ValueError("scheduler client env tool differs from replay runtime tools")
    return result


def _validate_replay_accepted_id_record(
    value: Mapping[str, Any], *, replay_execution_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(value, REPLAY_ACCEPTED_ID_RECORD_KEYS, name="accepted_id_record")
    contract = replay_execution_manifest["scheduler_submission"]["accepted_id_record"]
    path = _canonical_absolute_path(value["path"], name="accepted_id_record.path")
    expected_path = _canonical_absolute_path(contract["path"], name="declared accepted ID path")
    if path != expected_path:
        raise ValueError("accepted ID record path differs from replay manifest")
    if (
        value["format"] != contract["accepted_format"]
        or value["format"] != "ascii-positive-decimal-lf"
        or value["mode"] != contract["sealed_mode"]
        or value["mode"] != "0400"
    ):
        raise ValueError("accepted ID record framing/mode differs")
    sha256 = _require_digest(value["sha256"], name="accepted_id_record.sha256")
    candidate = _require_job_id(value["parsed_candidate_job_id"], name="parsed_candidate_job_id")
    raw = _load_lifecycle_raw_bytes(path, expected_sha256=sha256, maximum=64)
    if raw != f"{candidate}\n".encode("ascii"):
        raise ValueError("accepted ID bytes differ from parsed candidate job ID")
    return {
        "path": str(path),
        "sha256": sha256,
        "parsed_candidate_job_id": candidate,
        "format": "ascii-positive-decimal-lf",
        "mode": "0400",
    }


def _load_replay_scheduler_query_reference(
    value: Mapping[str, Any],
    *,
    phase: str,
    expected_path: Path,
    replay_execution_manifest: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    query_ref = _artifact_reference(
        value,
        name=f"{phase} scheduler query",
        expected_schema=REPLAY_SCHEDULER_QUERY_SCHEMA,
    )
    if Path(query_ref["path"]) != expected_path:
        raise ValueError(f"{phase} scheduler query path differs")
    query, digest = load_captured_replay_scheduler_query_v2(
        path=query_ref["path"],
        expected_sha256=query_ref["sha256"],
        replay_execution_manifest=replay_execution_manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if digest != query_ref["sha256"] or query["phase"] != phase:
        raise ValueError(f"{phase} scheduler query binding differs")
    return query_ref, query


def _lifecycle_results_root(replay_execution_manifest: Mapping[str, Any]) -> Path:
    pair = _load_lifecycle_pair_manifest(replay_execution_manifest)
    root = _canonical_absolute_path(pair["paths"]["results_root"], name="Pair results_root")
    attempt_root = _canonical_absolute_path(
        replay_execution_manifest["artifacts"]["outputs"]["directory"]["path"],
        name="replay attempt output root",
    )
    expected_attempt_root = root / "captured_replay" / replay_execution_manifest["attempt_id"]
    if attempt_root != expected_attempt_root:
        raise ValueError("replay output root differs from Pair results root")
    return root


def _submission_query_path(replay_execution_manifest: Mapping[str, Any], *, raw: bool) -> Path:
    accepted_path = _canonical_absolute_path(
        replay_execution_manifest["scheduler_submission"]["accepted_id_record"]["path"],
        name="accepted ID path",
    )
    expected_name = "accepted.job-id"
    if accepted_path.name != expected_name:
        raise ValueError("accepted ID filename differs from replay attempt")
    suffix = "scontrol.raw" if raw else "scontrol-query.json"
    return accepted_path.parent / f"PRE_RELEASE.{suffix}"


def _validate_submission_parent_precondition(
    replay_execution_manifest: Mapping[str, Any], *, candidate_job_id: str
) -> None:
    accepted_path = _canonical_absolute_path(
        replay_execution_manifest["scheduler_submission"]["accepted_id_record"]["path"],
        name="accepted ID path",
    )
    receipt_path = _canonical_absolute_path(
        replay_execution_manifest["scheduler_submission"]["receipt"]["path"],
        name="submission receipt path",
    )
    if accepted_path.parent != receipt_path.parent:
        raise ValueError("accepted-ID and submission receipt parents differ")
    operational = replay_execution_manifest["execution_environment"]["attempt"]["operational"]
    slurm_root = _canonical_absolute_path(operational["slurm"], name="operational Slurm root")
    parent_fd = _open_absolute_directory_without_symlinks(slurm_root)
    try:
        _require_owned_directory(os.fstat(parent_fd), name="submission state parent")
        job_id = _require_job_id(candidate_job_id, name="candidate_job_id")
        for name in (
            f"slurm-{job_id}.out",
            f"slurm-{job_id}.err",
        ):
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise FileExistsError(f"replay scheduler log target already exists: {name}")
    finally:
        os.close(parent_fd)


def _replay_job_parent(replay_execution_manifest: Mapping[str, Any], authenticated_job_id: str) -> Path:
    job_id = _require_job_id(authenticated_job_id, name="authenticated_job_id")
    return (
        _lifecycle_results_root(replay_execution_manifest)
        / "captured_replay"
        / "replay_job_state"
        / replay_execution_manifest["pair_id"]
        / replay_execution_manifest["attempt_id"]
        / f"{job_id}-0"
    )


def _job_query_path(
    replay_execution_manifest: Mapping[str, Any],
    authenticated_job_id: str,
    phase: str,
    *,
    raw: bool,
) -> Path:
    if phase not in {"PRE", "POST"}:
        raise ValueError("in-job scheduler query phase must be PRE or POST")
    suffix = "scontrol.raw" if raw else "scontrol-query.json"
    return _replay_job_parent(replay_execution_manifest, authenticated_job_id) / "queries" / f"{phase}.{suffix}"


def _replay_job_receipt_path(
    replay_execution_manifest: Mapping[str, Any],
    authenticated_job_id: str,
    *,
    phase: str,
) -> Path:
    if phase not in {"PRE", "EXIT"}:
        raise ValueError("replay job receipt phase must be PRE or EXIT")
    return _replay_job_parent(replay_execution_manifest, authenticated_job_id) / "receipts" / f"{phase}.json"


def _replay_post_index_path(replay_execution_manifest: Mapping[str, Any]) -> Path:
    output_root = _canonical_absolute_path(
        replay_execution_manifest["artifacts"]["outputs"]["directory"]["path"],
        name="replay output root",
    )
    expected = output_root / "evidence-index.json"
    declaration = replay_execution_manifest["artifacts"]["outputs"]["evidence_index"]
    _require_exact_keys(
        declaration,
        frozenset({"path", "schema", "framing", "mode"}),
        name="replay evidence index declaration",
    )
    if declaration != {
        "path": str(expected),
        "schema": REPLAY_POST_INDEX_V2_SCHEMA,
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
    }:
        raise ValueError("replay evidence index declaration differs")
    return expected


def _close_scheduler_query_to_lifecycle(
    query: Mapping[str, Any],
    *,
    replay_execution_manifest: Mapping[str, Any],
    candidate_job_id: str,
    comment: str,
    submitter_euid: int,
) -> None:
    record = query["records"][0]
    expected = {
        "job_id": _require_job_id(candidate_job_id, name="candidate_job_id"),
        "job_name": replay_execution_manifest["scheduler_submission"]["identity"]["job_name"],
        "comment": comment,
        "user_id": str(submitter_euid),
        "work_dir": replay_execution_manifest["execution_environment"]["attempt"]["scheduler"][
            "batch_working_directory"
        ],
        "restart_count": 0,
    }
    _require_exact_projection(record, expected, name=f"{query['phase']} scheduler record")


def _validate_replay_sbatch_argv(
    value: Sequence[str],
    *,
    replay_execution_manifest: Mapping[str, Any],
    replay_execution_manifest_ref: Mapping[str, Any],
    scheduler_tools: Mapping[str, Any],
    comment: str,
    expected_environment: str,
    expected_profile_id: str,
) -> list[str]:
    del scheduler_tools  # Executable identity is carried separately by sbatch.path/SHA.
    argv = _validate_argv(value)
    pair = _load_lifecycle_pair_manifest(replay_execution_manifest)
    campaign = pair["campaign"]
    slurm = campaign["slurm"]
    snapshot_root = _canonical_absolute_path(
        replay_execution_manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        name="replay source snapshot",
    )
    _absolute_program_reference(
        snapshot_root,
        replay_execution_manifest["replay_contract"]["program"]["job_wrapper"],
    )
    submission_parent = _canonical_absolute_path(
        replay_execution_manifest["scheduler_submission"]["accepted_id_record"]["path"],
        name="accepted ID record path",
    ).parent
    parent_fd = _open_absolute_directory_without_symlinks(submission_parent)
    try:
        _require_owned_directory(os.fstat(parent_fd), name="submission state parent")
    finally:
        os.close(parent_fd)
    slurm_root = _canonical_absolute_path(
        replay_execution_manifest["execution_environment"]["attempt"]["operational"]["slurm"],
        name="operational Slurm root",
    )
    slurm_fd = _open_absolute_directory_without_symlinks(slurm_root)
    try:
        _require_owned_directory(os.fstat(slurm_fd), name="operational Slurm root")
    finally:
        os.close(slurm_fd)
    if len(argv) != 39:
        raise ValueError("sbatch argv differs from authoritative replay ordering")
    export_match = re.fullmatch(r"--export-file=([1-9][0-9]*)", argv[17])
    wrapper_match = re.fullmatch(r"/proc/self/fd/([1-9][0-9]*)", argv[18])
    if export_match is None or wrapper_match is None:
        raise ValueError("sbatch retained descriptor arguments differ")
    export_fd = int(export_match.group(1), 10)
    wrapper_fd = int(wrapper_match.group(1), 10)
    if export_fd < 3 or wrapper_fd < 3 or export_fd == wrapper_fd:
        raise ValueError("sbatch retained descriptor identities differ")
    expected = [
        "--parsable",
        "--hold",
        f"--chdir={snapshot_root}",
        f"--nodes={campaign['nodes']}",
        f"--account={slurm['account']}",
        f"--job-name={replay_execution_manifest['scheduler_submission']['identity']['job_name']}",
        f"--partition={slurm['partition']}",
        "--time=04:00:00",
        "--gres=gpu:4",
        "--exclusive",
        "--mem=0",
        "--dependency=singleton",
        "--segment=1",
        f"--output={slurm_root}/slurm-%j.out",
        f"--error={slurm_root}/slurm-%j.err",
        f"--qos={slurm['qos']}",
        f"--comment={comment}",
        f"--export-file={export_fd}",
        f"/proc/self/fd/{wrapper_fd}",
        "--pair-manifest",
        replay_execution_manifest["pair"]["manifest"]["path"],
        "--pair-manifest-sha256",
        replay_execution_manifest["pair"]["manifest"]["sha256"],
        "--pair-submission-receipt",
        replay_execution_manifest["pair"]["submission_receipt"]["path"],
        "--pair-submission-receipt-sha256",
        replay_execution_manifest["pair"]["submission_receipt"]["sha256"],
        "--off-exit-receipt",
        replay_execution_manifest["source_capture"]["job_receipts"]["exit"]["path"],
        "--off-exit-receipt-sha256",
        replay_execution_manifest["source_capture"]["job_receipts"]["exit"]["sha256"],
        "--replay-manifest",
        replay_execution_manifest_ref["path"],
        "--replay-manifest-sha256",
        replay_execution_manifest_ref["sha256"],
        "--environment",
        expected_environment,
        "--profile-id",
        expected_profile_id,
    ]
    if argv != expected:
        raise ValueError("sbatch argv differs from authoritative replay ordering")
    return argv


def _absolute_program_reference(snapshot_root: Path, value: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_keys(value, TOOL_REFERENCE_KEYS, name="replay program reference")
    relative = _require_ascii(value["path"], name="replay program relative path", maximum=4096)
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or "." in relative_path.parts
        or ".." in relative_path.parts
        or str(relative_path) != relative
    ):
        raise ValueError("replay program path must be canonical and relative")
    return {
        "path": str(snapshot_root / relative_path),
        "sha256": _require_digest(value["sha256"], name="replay program reference SHA-256"),
    }


def _receipt_reference_for_document(*, path: str, schema: str, document: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(_canonical_absolute_path(path, name=f"{schema} path")),
        "schema": schema,
        "sha256": document_sha256(document, trailing_lf=True),
    }


def _replay_static_boundary(
    replay_execution_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {name: copy.deepcopy(replay_execution_manifest[name]) for name in REPLAY_STATIC_BOUNDARY_V2_KEYS}


def _replay_scheduler_query_phase(value: Any) -> str:
    if value not in {"PRE_RELEASE", "PRE", "POST"}:
        raise ValueError("scheduler query phase is not admitted")
    return value


def _validate_replay_scheduler_record(value: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    _require_exact_keys(value, REPLAY_SCHEDULER_RECORD_KEYS, name="scheduler record")
    result = copy.deepcopy(dict(value))
    _require_job_id(result["job_id"], name="scheduler record job_id")
    for key, maximum in (
        ("job_name", 255),
        ("comment", 4096),
        ("job_state", 64),
    ):
        _require_ascii(result[key], name=f"scheduler record {key}", maximum=maximum)
    _require_scheduler_string(result["reason"], name="scheduler record reason", maximum=4096)
    user_id = _require_ascii(result["user_id"], name="scheduler record user_id", maximum=32)
    if not user_id.isdecimal() or str(int(user_id)) != user_id or not 0 <= int(user_id) <= (1 << 31) - 1:
        raise ValueError("scheduler record user_id must be a canonical uint31 string")
    _canonical_absolute_path(result["work_dir"], name="scheduler record work_dir")
    if type(result["held"]) is not bool:
        raise TypeError("scheduler record held must be an exact bool")
    if type(result["restart_count"]) is not int or result["restart_count"] != 0:
        raise ValueError("scheduler record restart_count must be exact integer zero")
    if phase == "PRE_RELEASE":
        expected = {"job_state": "PENDING", "reason": "JobHeldUser", "held": True}
    else:
        expected = {"job_state": "RUNNING", "held": False}
    _require_exact_projection(result, expected, name=f"{phase} scheduler record")
    if phase != "PRE_RELEASE" and result["reason"] == "JobHeldUser":
        raise ValueError(f"{phase} scheduler record remains user-held")
    return result


def _normalized_replay_scheduler_record(raw: bytes, *, phase: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise ValueError("scheduler raw output framing differs")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("scheduler raw output must be UTF-8 JSON") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_int=_parse_fixture_int,
            parse_float=_parse_fixture_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"scheduler raw output is not strict JSON: {error}") from error
    _require_exact_keys(
        parsed,
        frozenset({"errors", "jobs", "last_backfill", "last_update", "meta", "warnings"}),
        name="scheduler raw JSON root",
    )
    if parsed["errors"] != [] or type(parsed["errors"]) is not list:
        raise ValueError("scheduler raw JSON errors must be the exact empty list")
    if parsed["warnings"] != [] or type(parsed["warnings"]) is not list:
        raise ValueError("scheduler raw JSON warnings must be the exact empty list")
    for name in ("last_backfill", "last_update", "meta"):
        if type(parsed[name]) is not dict:
            raise TypeError(f"scheduler raw JSON {name} must be an exact object")
    jobs = parsed["jobs"]
    if type(jobs) is not list or len(jobs) != 1 or type(jobs[0]) is not dict:
        raise ValueError("scheduler raw JSON must contain exactly one job object")
    source = jobs[0]
    required = {
        "job_id",
        "name",
        "comment",
        "current_working_directory",
        "state_reason",
        "user_id",
        "job_state",
        "restart_cnt",
        "hold",
    }
    missing = required - set(source)
    if missing:
        raise ValueError("scheduler raw job is missing required fields: " + ", ".join(sorted(missing)))
    job_id = source["job_id"]
    if type(job_id) is not int or not 1 <= job_id <= (1 << 63) - 1:
        raise ValueError("scheduler raw job_id must be a positive int63")
    user_id = source["user_id"]
    if type(user_id) is not int or not 0 <= user_id <= (1 << 31) - 1:
        raise ValueError("scheduler raw user_id must be a uint31")
    restart_count = source["restart_cnt"]
    if type(restart_count) is not int or restart_count != 0:
        raise ValueError("scheduler raw restart_cnt must be exact integer zero")
    if type(source["hold"]) is not bool:
        raise TypeError("scheduler raw hold must be an exact bool")
    states = source["job_state"]
    if type(states) is not list or len(states) != 1 or type(states[0]) is not str:
        raise ValueError("scheduler raw job_state must be an exact one-string list")
    job_name = _require_ascii(source["name"], name="scheduler raw name", maximum=255)
    comment = _require_ascii(source["comment"], name="scheduler raw comment", maximum=4096)
    work_dir = str(
        _canonical_absolute_path(
            _require_ascii(
                source["current_working_directory"],
                name="scheduler raw current_working_directory",
                maximum=4096,
            ),
            name="scheduler raw current_working_directory",
        )
    )
    job_state = _require_ascii(states[0], name="scheduler raw job_state", maximum=64)
    reason = _require_scheduler_string(source["state_reason"], name="scheduler raw state_reason", maximum=4096)
    record = {
        "job_id": str(job_id),
        "job_name": job_name,
        "comment": comment,
        "user_id": str(user_id),
        "work_dir": work_dir,
        "job_state": job_state,
        "reason": reason,
        "held": source["hold"],
        "restart_count": restart_count,
    }
    return _validate_replay_scheduler_record(record, phase=phase)


def _authenticated_replay_scontrol(
    replay_execution_manifest: Mapping[str, Any],
) -> dict[str, str]:
    host_ref = _file_reference(
        replay_execution_manifest["runtime_tools"]["document"]["host"]["scontrol"],
        name="runtime_tools host scontrol",
    )
    pair_receipt = _load_lifecycle_pair_submission_receipt(replay_execution_manifest)
    receipt_ref = _file_reference(
        pair_receipt["scheduler_tools"]["scontrol"],
        name="Pair scheduler scontrol",
    )
    if host_ref != receipt_ref:
        raise ValueError("replay scontrol differs from authenticated Pair receipt")
    _load_bound_runtime_tool_bytes(Path(host_ref["path"]), expected_sha256=host_ref["sha256"])
    return host_ref


def _load_bound_runtime_tool_bytes(path: Path, *, expected_sha256: str) -> bytes:
    expected = _require_digest(expected_sha256, name="runtime tool SHA-256")
    parent_fd = _open_absolute_directory_without_symlinks(path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        pre_named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(pre_named.st_mode) or not 0 < pre_named.st_size <= 64 * 1024 * 1024:
            raise ValueError("runtime tool must be a bounded regular file")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                _file_fingerprint(pre_named) != _file_fingerprint(before)
                or not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= 64 * 1024 * 1024
            ):
                raise RuntimeError("runtime tool changed before stable read")
            raw = _read_all_bounded(descriptor, maximum=64 * 1024 * 1024)
            after = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_without_symlinks(path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            fresh_named = os.stat(
                path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
    finally:
        os.close(parent_fd)
    if not (
        _directory_identity(parent_before) == _directory_identity(parent_after) == _directory_identity(fresh_parent)
    ):
        raise RuntimeError("runtime tool parent changed during stable read")
    if not (
        _file_fingerprint(pre_named)
        == _file_fingerprint(before)
        == _file_fingerprint(after)
        == _file_fingerprint(named)
        == _file_fingerprint(fresh_named)
    ):
        raise RuntimeError("runtime tool changed during stable read")
    if len(raw) != after.st_size or not stat.S_ISREG(after.st_mode) or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("runtime tool bytes differ from authenticated reference")
    return raw


def _scheduler_query_document_path(raw_path: Path) -> Path:
    suffix = ".scontrol.raw"
    if not raw_path.name.endswith(suffix):
        raise ValueError("scheduler raw output name must end in .scontrol.raw")
    return raw_path.with_name(f"{raw_path.name[: -len(suffix)]}.scontrol-query.json")


def _load_lifecycle_raw_bytes(path: Path, *, expected_sha256: str, maximum: int) -> bytes:
    expected = _require_digest(expected_sha256, name="raw expected SHA-256")
    parent_fd = _open_absolute_directory_without_symlinks(path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        _require_owned_directory(parent_before, name="raw evidence parent")
        pre_named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(pre_named.st_mode)
            or stat.S_IMODE(pre_named.st_mode) != 0o400
            or pre_named.st_uid != os.geteuid()
            or pre_named.st_nlink != 1
            or not 0 < pre_named.st_size <= maximum
        ):
            raise RuntimeError("raw evidence must be stable EUID-owned single-link mode-0400 bytes")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                _file_fingerprint(pre_named) != _file_fingerprint(before)
                or not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= maximum
            ):
                raise RuntimeError("raw evidence changed before stable read")
            raw = _read_all_bounded(descriptor, maximum=maximum)
            after = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_without_symlinks(path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            _require_owned_directory(fresh_parent, name="fresh raw evidence parent")
            fresh_named = os.stat(
                path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
    finally:
        os.close(parent_fd)
    if not (
        _directory_identity(parent_before) == _directory_identity(parent_after) == _directory_identity(fresh_parent)
    ):
        raise RuntimeError("raw evidence parent changed during stable read")
    if not (
        _file_fingerprint(pre_named)
        == _file_fingerprint(before)
        == _file_fingerprint(after)
        == _file_fingerprint(named)
        == _file_fingerprint(fresh_named)
    ):
        raise RuntimeError("raw evidence changed during stable read")
    if (
        len(raw) != after.st_size
        or not stat.S_ISREG(after.st_mode)
        or stat.S_IMODE(after.st_mode) != 0o400
        or after.st_uid != os.geteuid()
        or after.st_nlink != 1
    ):
        raise RuntimeError("raw evidence must be stable EUID-owned single-link mode-0400 bytes")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("raw evidence differs from expected SHA-256")
    return raw


def _strict_json_object(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    result = copy.deepcopy(value)
    canonical_ascii_json(result)
    return result


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare strict JSON including scalar types, not Python numeric aliases."""
    try:
        return canonical_ascii_json(left) == canonical_ascii_json(right)
    except (TypeError, ValueError):
        return False


def _seed_from_responses_create_params(value: Any, *, name: str) -> int:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name}.metadata must be an object")
    extra_body = metadata.get("extra_body")
    if type(extra_body) is not str:
        raise TypeError(f"{name}.metadata.extra_body must be a JSON object string")
    try:
        parsed = json.loads(
            extra_body,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_int=_parse_fixture_int,
            parse_float=_parse_fixture_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name}.metadata.extra_body is not strict JSON") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"{name}.metadata.extra_body must decode to an object")
    return _require_nonnegative_int(parsed.get("seed"), name=f"{name} seed")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _parse_fixture_int(value: str) -> int:
    if value == "-0":
        raise ValueError("negative zero is forbidden in strict fixture JSON")
    return int(value)


def _parse_fixture_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite float is forbidden in strict fixture JSON")
    if parsed == 0.0 and value.startswith("-"):
        raise ValueError("negative zero is forbidden in strict fixture JSON")
    return parsed


def _validate_argv(value: Sequence[str]) -> list[str]:
    if type(value) is not list:
        raise TypeError("sbatch.argv must be an exact JSON list of strings")
    if not value:
        raise ValueError("sbatch.argv must not be empty")
    result = []
    for index, item in enumerate(value):
        result.append(_require_ascii(item, name=f"sbatch.argv[{index}]", maximum=4096))
    return result


def _four_distinct_decimal_ids(value: Any, *, name: str) -> str:
    text = _require_ascii(value, name=name, maximum=255)
    tokens = text.split(",")
    if (
        len(tokens) != 4
        or any(re.fullmatch(r"0|[1-9][0-9]*", token) is None for token in tokens)
        or len({int(token) for token in tokens}) != 4
    ):
        raise ValueError(f"{name} must contain four canonical numerically distinct decimal IDs")
    return text


def _require_exact_keys(value: Any, expected: frozenset[str], *, name: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(f"{name} keyset mismatch: expected {sorted(expected)}, got {sorted(actual)}")


def _require_ascii(value: Any, *, name: str, maximum: int) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise TypeError(f"{name} must be a nonempty ASCII string")
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be ASCII") from error
    if len(payload) > maximum:
        raise ValueError(f"{name} exceeds {maximum} ASCII bytes")
    return value


def _require_scheduler_string(value: Any, *, name: str, maximum: int) -> str:
    text = _require_ascii(value, name=name, maximum=maximum)
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise ValueError(f"{name} contains a forbidden ASCII control character")
    return text


def _require_safe_id(value: Any, *, name: str, maximum: int) -> str:
    text = _require_ascii(value, name=name, maximum=maximum)
    if _SAFE_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} is not a safe identifier")
    return text


def _require_digest(value: Any, *, name: str) -> str:
    if type(value) is not str or _HEX64_RE.fullmatch(value) is None or value == "0" * 64:
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256 digest")
    return value


def _require_job_id(value: Any, *, name: str) -> str:
    text = _require_ascii(value, name=name, maximum=32)
    if not text.isdecimal() or str(int(text)) != text or not 1 <= int(text) <= (1 << 63) - 1:
        raise ValueError(f"{name} must be a canonical positive decimal string")
    return text


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be an exact nonnegative int")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{name} must be an exact positive int")
    return value


def _require_bounded_positive_int(value: Any, *, name: str, maximum: int) -> int:
    result = _require_positive_int(value, name=name)
    if result > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return result


def _require_exact_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be an exact finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{name} must not be negative zero")
    return value


def _float32_round_trip(value: float) -> float:
    """Return the exact Python-float representation of trainer IEEE float32."""
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _reject_negative_zero(value: Any, *, path: str) -> None:
    if type(value) is float:
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(f"negative zero is forbidden at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"canonical JSON key at {path} is not a string")
            _reject_negative_zero(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_negative_zero(item, path=f"{path}[{index}]")


def _validate_strict_json_types(value: Any, *, path: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is forbidden at {path}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"canonical JSON key at {path} is not a string")
            _validate_strict_json_types(item, path=f"{path}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_strict_json_types(item, path=f"{path}[{index}]")
        return
    raise TypeError(f"non-JSON value of type {type(value).__name__} at {path}")


def _canonical_absolute_path(value: Any, *, name: str) -> Path:
    if type(value) is not str or not value or "\x00" in value or not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"{name} must be a nonempty canonical absolute path")
    path = Path(value)
    if not path.is_absolute() or "." in path.parts or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a canonical absolute path, got {value!r}")
    return path


def _open_absolute_directory_without_symlinks(path: Path) -> int:
    directory_fd = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
    except OSError:
        os.close(directory_fd)
        raise
    return directory_fd


def _require_owned_directory(metadata: os.stat_result, *, name: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.geteuid():
        raise RuntimeError(f"{name} must be an EUID-owned mode-0700 directory")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("evidence write made no progress")
        offset += written


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_all_bounded(fd: int, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, maximum + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise ValueError(f"file exceeds maximum admitted size {maximum}")


def _file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Identity needed to prove a held directory is still at its canonical path."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )
