"""Closed pre-submit manifest for authenticated fresh verifier reward replay.

The manifest is deliberately pre-submit: it binds completed OFF evidence and every
static replay input, but it cannot contain the future candidate/authenticated replay
job identity, PRE/EXIT receipts, process observations, or replay outputs.  Those are
joined by the submission receipt, in-job wrapper receipts, and terminal indices.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import posixpath
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from nemo_rl.utils.strict_captured_replay_evidence import (
    canonical_ascii_json,
    domain_sha256,
    load_evidence_document,
    publish_evidence_document,
)

if TYPE_CHECKING:
    from nemo_rl.utils.strict_captured_replay_profiles import (
        StrictCapturedReplayProfile,
    )

REPLAY_EXECUTION_MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v3"
REPLAY_CONTRACT_SCHEMA = "nemo-rl-strict-captured-replay-contract-v1"
REPLAY_EXECUTION_ENVIRONMENT_SCHEMA = "nemo-rl-strict-captured-replay-execution-environment-v1"
REPLAY_SLURM_EXPORT_SCHEMA = "nemo-rl-strict-captured-replay-slurm-export-file-v2"
PAIR_SLURM_EXPORT_SCHEMA = "nemo-rl-strict-slurm-export-file-v3"
REPLAY_RUNTIME_REQUIREMENTS_SCHEMA = "nemo-rl-strict-captured-replay-runtime-requirements-v1"
REPLAY_SUBMISSION_RECEIPT_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v4"
REPLAY_SUBMISSION_CONTRACT_SCHEMA = "nemo-rl-strict-captured-replay-submission-contract-v1"
REPLAY_TRANSPORT_CONSUMPTION_SCHEMA = "nemo-rl-strict-model-transport-replay-consumption-v2"
REPLAY_EVIDENCE_INDEX_SCHEMA = "nemo-rl-strict-captured-replay-evidence-index-v3"
REPLAY_RESULT_INVENTORY_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v1"
REASONING_SCORE_CALL_SCHEMA = "nemo-rl-strict-reasoning-score-call-v1"
REASONING_SCORE_CALL_INDEX_SCHEMA = "nemo-rl-strict-reasoning-score-call-index-v1"
REPLAY_EXECUTION_MANIFEST_V2_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
REPLAY_CONTRACT_V2_SCHEMA = "nemo-rl-strict-captured-replay-contract-v2"
REPLAY_RUNTIME_REQUIREMENTS_V2_SCHEMA = "nemo-rl-strict-captured-replay-runtime-requirements-v2"
REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v5"
REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-job-pre-receipt-v3"
REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA = "nemo-rl-strict-captured-replay-job-exit-receipt-v6"
REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA = "nemo-rl-strict-captured-replay-runtime-attestation-v2"
REPLAY_EVIDENCE_INDEX_V2_SCHEMA = "nemo-rl-strict-captured-replay-evidence-index-v4"
REPLAY_RESULT_INVENTORY_V2_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
REPLAY_TRANSPORT_CONSUMPTION_V2_SCHEMA = "nemo-rl-strict-model-transport-replay-consumption-v3"
PAIR_MANIFEST_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
PAIR_SUBMISSION_RECEIPT_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
PAIR_JOB_RECEIPT_SCHEMA = "nemo-rl-strict-pair-job-receipt-v2"
STEP1_EVIDENCE_INDEX_SCHEMA = "nemo-rl-strict-step1-evidence-index-v4"
MAIN_LEDGER_SCHEMA = "nemo-rl-strict-main-step1-ledger-v5"
TRANSCRIPT_BUNDLE_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
TRANSPORT_EVIDENCE_INDEX_SCHEMA = "nemo-rl-strict-model-transport-evidence-index-v1"
TRANSPORT_BUNDLE_SCHEMA = "nemo-rl-strict-model-transport-bundle-v1"
TRANSPORT_MANIFEST_SCHEMA = "nemo-rl-strict-model-transport-manifest-v1"
TRANSPORT_CALL_SCHEMA = "nemo-rl-strict-model-transport-call-v1"
HASH_DOMAIN = "sha256-domain-nul-canonical-ascii-json-no-lf-v1"

ATTEMPTS = ("replay-1", "replay-2")
ENVIRONMENTS = ("citation", "freeform", "reasoning_gym")
ENVIRONMENT_AGENTS = {
    "citation": "citation_format_simple_agent",
    "freeform": "freeform_formatting_simple_agent",
    "reasoning_gym": "reasoning_gym_simple_agent",
}
REQUIRED_RUNTIME_VERSIONS = {
    "required_openai_version": "2.6.1",
    "required_pydantic_version": "2.13.4",
}
REPLAY_WANDB_POLICY = {
    "enabled": False,
    "mode": "disabled",
    "reason": "scorer-only-replay-no-wandb-credentials-or-output",
}
GYM_SCORER_DIRECTORY = {
    "citation": "format_verification",
    "freeform": "format_verification",
    "reasoning_gym": "reasoning_gym",
}
GYM_SCORER_CONFIG_PATH_NAME = {
    "citation": "citation_format",
    "freeform": "freeform_formatting",
    "reasoning_gym": "reasoning_gym",
}
GYM_RUN_HELPER_SOURCE = {
    "path": "nemo_gym/cli/env.py",
    "sha256": "e6f468072e6b6627624dbc9270f60ec792ff0fa2aed681f9ce338b714880188a",
}
GYM_SETUP_COMMAND_SOURCE = {
    "path": "nemo_gym/cli/setup_command.py",
    "sha256": "6e976fe8491e8ddc9770dc553f93ef46a5545760cf9f3ce5396369c0e9945a71",
}
GYM_PORT_ALLOCATOR_SOURCE = {
    "path": "nemo_gym/global_config.py",
    "sha256": "5e8e7457e6c3b9ae2cc5b124fb3b42f1920ba6c17707519da7b1d560e0a1ea70",
}
GYM_SERVER_SOURCE = {
    "path": "nemo_gym/server_utils.py",
    "sha256": "1c1fd497d51c87783ebd31c8e8526c53d0aae97c95a541a8d6b92e8901990b7b",
}
GYM_RESOURCE_APP = {
    "citation": {
        "path": "resources_servers/format_verification/app.py",
        "sha256": "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36",
    },
    "freeform": {
        "path": "resources_servers/format_verification/app.py",
        "sha256": "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36",
    },
    "reasoning_gym": {
        "path": "resources_servers/reasoning_gym/app.py",
        "sha256": "3a35c5d27392dae05499ceefac04e9c32ad963b51a54d77bb470ee59b1fe3127",
    },
}
GYM_REASONING_RESOURCE_ONLY_CONFIG = {
    "path": "resources_servers/reasoning_gym/configs/resources_only.yaml",
    "sha256": "e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68",
}
GYM_SNAPSHOT_RELATIVE_ROOT = "3rdparty/Gym-workspace/Gym"
GYM_CONTAINER_ROOT = "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
REPLAY_CONTAINER_OWNER_UID = 153493
REPLAY_CONTAINER_OWNER_GID = 30
REPLAY_JOB_ARGV_TEMPLATE_V2 = (
    "--pair-manifest",
    "{pair_manifest_path}",
    "--pair-manifest-sha256",
    "{pair_manifest_sha256}",
    "--pair-submission-receipt",
    "{pair_submission_receipt_path}",
    "--pair-submission-receipt-sha256",
    "{pair_submission_receipt_sha256}",
    "--off-exit-receipt",
    "{trusted_off_exit_receipt_path}",
    "--off-exit-receipt-sha256",
    "{trusted_off_exit_receipt_sha256}",
    "--replay-manifest",
    "{replay_manifest_path}",
    "--replay-manifest-sha256",
    "{replay_manifest_sha256}",
    "--environment",
    "{environment}",
    "--profile-id",
    "{profile_id}",
)

ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "attempt_id",
        "pair",
        "source_capture",
        "replay_contract",
        "artifacts",
        "execution_environment",
        "wandb",
        "scheduler_submission",
        "slurm_export_boundary",
        "deployment",
        "runtime_attestation_requirements",
        "runtime_tools",
        "container_entry_boundary",
        "source",
    }
)
ROOT_V2_KEYS = ROOT_KEYS | frozenset({"scorer_profile"})
PAIR_KEYS = frozenset(
    {
        "id",
        "environment",
        "manifest",
        "submission_receipt",
        "acceptance_policy_sha256",
        "model_transport_policy_sha256",
        "pair_campaign_sha256",
        "pair_campaign_reward_and_advantage_sha256",
    }
)
SOURCE_CAPTURE_KEYS = frozenset({"arm", "restart_count", "authenticated_job", "job_receipts", "step1_evidence"})
AUTHENTICATED_JOB_KEYS = frozenset({"comment", "job_id", "job_name", "user_id"})
JOB_RECEIPTS_KEYS = frozenset({"pre", "exit"})
STEP1_EVIDENCE_KEYS = frozenset({"schema", "main_ledger", "transcript_bundle", "model_transport"})
TRANSPORT_EVIDENCE_KEYS = frozenset({"schema", "bundle", "manifest", "raw_log", "ordered_entries_sha256"})
RAW_LOG_KEYS = frozenset({"path", "record_schema", "record_count", "sha256"})
REPLAY_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "claim",
        "execution_scope",
        "cohort",
        "policy_execution",
        "model_transport",
        "program",
        "selected_config",
        "source_generation",
        "source_snapshot",
        "gym_scorer",
    }
)
PROGRAM_KEYS = frozenset(
    {
        "entrypoint",
        "evidence_utility",
        "gym_child_bootstrap",
        "gym_child_runtime",
        "job_wrapper",
        "manifest_utility",
        "raw_transport_owner",
        "result_sealer",
        "runtime",
        "submission_launcher",
    }
)
REPLAY_PROGRAM_PATHS = {
    "entrypoint": "examples/nemo_gym/run_strict_captured_replay_v2.py",
    "evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence_v2.py",
    "gym_child_bootstrap": ("nemo_rl/environments/_strict_gym_child_bootstrap_v2/sitecustomize.py"),
    "gym_child_runtime": "nemo_rl/environments/strict_gym_child_runtime_v2.py",
    "job_wrapper": "strict_pair_replay_job_wrapper_v2.sh",
    "manifest_utility": "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
    "raw_transport_owner": "nemo_rl/utils/strict_model_transport_replay_v3.py",
    "result_sealer": "nemo_rl/utils/strict_captured_replay_seal_v2.py",
    "runtime": "nemo_rl/algorithms/strict_captured_replay_runtime_v2.py",
    "submission_launcher": "strict_pair_replay_launch_v2.sh",
}
PROGRAM_V2_KEYS = PROGRAM_KEYS | frozenset(
    {
        "legacy_evidence_utility",
        "main_step_ledger",
        "model_transport_utility",
        "profile_registry",
    }
)
REPLAY_PROGRAM_V2_PATHS = {
    **REPLAY_PROGRAM_PATHS,
    "legacy_evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence.py",
    "main_step_ledger": "nemo_rl/utils/strict_main_step_ledger.py",
    "model_transport_utility": "nemo_rl/utils/strict_model_transport.py",
    "profile_registry": "nemo_rl/utils/strict_captured_replay_profiles.py",
}
ARTIFACTS_KEYS = frozenset({"container", "fixture", "model", "sandbox_container", "outputs"})
OUTPUT_KEYS = frozenset(
    {
        "directory",
        "evidence_index",
        "reasoning_score_call_index",
        "transcript_bundle",
        "replay_ledger",
        "result_inventory",
        "transport_consumption",
    }
)
OUTPUT_V2_KEYS = (OUTPUT_KEYS - frozenset({"reasoning_score_call_index"})) | frozenset({"scorer_call_index"})
EXECUTION_ENVIRONMENT_KEYS = frozenset({"schema", "arm_launcher", "fixed", "attempt"})
ATTEMPT_ENVIRONMENT_KEYS = frozenset(
    {
        "base_log_dir",
        "cache_read",
        "hf_datasets_cache",
        "hf_home",
        "hf_hub_cache",
        "persistent_cache",
        "operational",
        "results_dir",
        "scheduler",
        "setup_command",
    }
)
SCHEDULER_SUBMISSION_KEYS = frozenset({"schema", "accepted_id_record", "contract", "identity", "nonce", "receipt"})
SLURM_EXPORT_KEYS = frozenset(
    {
        "schema",
        "allowed_names",
        "ambient_merge",
        "attempt_id",
        "format",
        "get_user_env",
        "job_argv_template",
        "path",
        "sha256",
    }
)
RUNTIME_REQUIREMENTS_KEYS = frozenset(
    {
        "schema",
        "shared_prefix_determinism",
        "model_transport_replay",
        "derived_request_runtime",
        "resource_scorer_child",
        "verifier",
    }
)
RUNTIME_REQUIREMENTS_V2_KEYS = RUNTIME_REQUIREMENTS_KEYS | frozenset({"lifecycle_schemas"})
ARTIFACT_REF_KEYS = frozenset({"path", "schema", "sha256"})
FILE_REF_KEYS = frozenset({"path", "sha256"})
GYM_SCORER_CONTAINER_KEYS = frozenset({"path", "sha256", "owner_uid", "owner_gid"})
REPLAY_SUBMISSION_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "attempt_id",
        "submission_nonce",
        "submitter_euid",
        "sbatch_program",
        "job_wrapper",
        "submission_launcher",
        "slurm_export",
    }
)

# This is the exact Pair Bootstrap-Trust R1 79-name export set. Replay reuses the closed export
# surface; replay-manifest identity is positional wrapper argv, never ambient state.
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

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_JOB_ID_RE = re.compile(r"[1-9][0-9]*")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

PAIR_SUBMISSION_RECEIPT_KEYS = frozenset(
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
    }
)
PAIR_PRE_RECEIPT_KEYS = frozenset(
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
        "entrypoint_sha256",
        "environment",
        "execution_environment",
        "fixture_rows",
        "fixture_sha256",
        "gpus_per_node",
        "gym_gitlink_commit",
        "gym_tree",
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
        "restart_count",
        "reward_semantics_config_sha256",
        "reward_semantics_contract_sha256",
        "runtime_attestation_expected_count",
        "runtime_attestation_marker_sha256",
        "runtime_attestation_receipt_dir",
        "runtime_attestation_receipt_dir_device",
        "runtime_attestation_receipt_dir_inode",
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
        "schema",
        "selected_config_sha256",
        "selection",
        "slurm_export_boundary",
        "slurm_export_boundary_sha256",
        "snapshot_manifest_sha256",
        "source",
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
        "wandb",
        "wrapper_sha256",
    }
)
PAIR_EXIT_RECEIPT_KEYS = PAIR_PRE_RECEIPT_KEYS | frozenset(
    {
        "driver_exit_code",
        "hardware",
        "pre_receipt_sha256",
        "runtime_attestation_actual_count",
        "runtime_attestation_aggregate_sha256",
        "runtime_attestation_receipts_sha256",
        "scheduler_device_environment",
        "step1_evidence",
    }
)


@dataclass(frozen=True)
class AuthenticatedOffSourceCapture:
    """Stable-loaded authority for one completed historical OFF source."""

    source_capture: dict[str, Any]
    pair_manifest: dict[str, Any]
    pair_manifest_sha256: str
    pair_submission_receipt: dict[str, Any]
    pair_submission_receipt_sha256: str
    trusted_off_exit_receipt_path: str
    trusted_off_exit_receipt_sha256: str
    pre_receipt: dict[str, Any]
    pre_receipt_sha256: str
    exit_receipt: dict[str, Any]
    exit_receipt_sha256: str
    main_ledger: dict[str, Any]
    transcript_bundle: dict[str, Any]
    transport_bundle: dict[str, Any]
    transport_manifest: dict[str, Any]
    transport_records: tuple[dict[str, Any], ...]

    @property
    def document(self) -> dict[str, Any]:
        """Return a detached source-capture document for manifest construction."""
        return copy.deepcopy(self.source_capture)


@dataclass(frozen=True)
class AuthenticatedReplayStaticInputs:
    """Stable-loaded pre-submit inputs that can safely precede job creation."""

    attempt_id: str
    container_asset: dict[str, Any]
    source_snapshot: dict[str, Any]
    gym_source_root: dict[str, str]
    replay_program: dict[str, dict[str, str]]
    slurm_export_path: str
    slurm_export_sha256: str
    slurm_export_values: tuple[tuple[str, bytes], ...]
    submission_contract_path: str
    submission_contract_sha256: str
    submission_contract: dict[str, Any]


def load_authenticated_off_source_capture(
    *,
    pair_manifest: Mapping[str, Any],
    pair_manifest_path: str,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: str,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: str,
    trusted_off_exit_receipt_sha256: str,
) -> AuthenticatedOffSourceCapture:
    """Derive the only admitted replay source from sealed historical evidence.

    The successful OFF EXIT path and digest are the independent OOB trust anchor.
    They are stable-loaded before any supporting Pair, PRE, or step-1 document.
    The OFF identity and expected EXIT path are then derived from the released
    Pair receipt; the trusted EXIT binds the PRE digest and every downstream
    step-1 reference.  No job identifier, PRE digest, or step-1 reference is
    accepted independently from the caller.
    """
    trusted_exit_path = _absolute_path(
        trusted_off_exit_receipt_path,
        name="trusted source OFF EXIT receipt path",
    )
    trusted_exit_sha = _digest(
        trusted_off_exit_receipt_sha256,
        name="trusted source OFF EXIT receipt SHA-256",
    )
    exit_receipt, actual_exit_sha = _load_anchored_evidence_document(
        path=trusted_exit_path,
        expected_sha256=trusted_exit_sha,
        trailing_lf=True,
        name="trusted source OFF EXIT receipt",
    )
    if actual_exit_sha != trusted_exit_sha:
        raise AssertionError("unreachable trusted OFF EXIT receipt digest mismatch")

    _require_exact_keys(pair_manifest, _pair_manifest_required_keys(), name="Pair")
    pair_path = _absolute_path(pair_manifest_path, name="Pair manifest path")
    pair_sha = _digest(pair_manifest_sha256, name="Pair manifest SHA-256")
    loaded_pair, loaded_pair_sha = load_evidence_document(
        path=pair_path,
        expected_sha256=pair_sha,
        trailing_lf=True,
    )
    if loaded_pair_sha != pair_sha or not _exact(loaded_pair, pair_manifest):
        raise ValueError("caller Pair manifest differs from sealed Pair bytes")
    _require_exact_keys(loaded_pair, _pair_manifest_required_keys(), name="Pair")
    if loaded_pair["schema"] != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unexpected Pair manifest schema")

    results_root = _absolute_path(loaded_pair["paths"]["results_root"], name="Pair results_root")
    if pair_path != f"{results_root}/PAIR_MANIFEST.json":
        raise ValueError("Pair manifest path differs from Pair results root")
    submission_path = _absolute_path(pair_submission_receipt_path, name="Pair submission receipt path")
    if submission_path != f"{results_root}/PAIR_SUBMISSION_RECEIPT.json":
        raise ValueError("Pair submission receipt path differs from Pair results root")
    submission_sha = _digest(
        pair_submission_receipt_sha256,
        name="Pair submission receipt SHA-256",
    )
    submission, actual_submission_sha = load_evidence_document(
        path=submission_path,
        expected_sha256=submission_sha,
        trailing_lf=True,
    )
    if actual_submission_sha != submission_sha:
        raise AssertionError("unreachable Pair submission receipt digest mismatch")
    identities = _validate_released_pair_submission_receipt(
        submission,
        pair=loaded_pair,
        pair_path=pair_path,
        pair_sha256=pair_sha,
        receipt_path=submission_path,
    )
    off_identity = identities["off"]
    source_job_id = off_identity["job_id"]
    receipt_root = f"{results_root}/off/strict_pair_job_state/{source_job_id}-0/receipts"
    pre_path = f"{receipt_root}/PRE.json"
    exit_path = f"{receipt_root}/EXIT.json"
    if trusted_exit_path != exit_path:
        raise ValueError("trusted source OFF EXIT path differs from released Pair authority")
    _require_exact_keys(
        exit_receipt,
        PAIR_EXIT_RECEIPT_KEYS,
        name="trusted source OFF EXIT receipt",
    )
    pre_expected_sha = _digest(
        exit_receipt["pre_receipt_sha256"],
        name="trusted source OFF EXIT PRE receipt SHA-256",
    )
    pre, pre_sha = _load_anchored_evidence_document(
        path=pre_path,
        expected_sha256=pre_expected_sha,
        trailing_lf=True,
        name="source OFF PRE receipt",
    )
    _validate_successful_off_job_receipts(
        pre,
        exit_receipt,
        pre_sha256=pre_sha,
        pair=loaded_pair,
        pair_path=pair_path,
        pair_sha256=pair_sha,
        submission=submission,
        submission_path=submission_path,
        submission_sha256=submission_sha,
        off_identity=off_identity,
    )

    evidence = copy.deepcopy(exit_receipt["step1_evidence"])
    _validate_source_step1_index_paths(evidence, results_root=results_root)
    ledger_ref = evidence["main_ledger"]
    transcript_ref = evidence["transcript_bundle"]
    transport_index = evidence["model_transport"]
    transport_bundle_ref = transport_index["bundle"]
    transport_manifest_ref = transport_index["manifest"]
    raw_log_ref = transport_index["raw_log"]

    main_ledger, _ = load_evidence_document(
        path=ledger_ref["path"],
        expected_sha256=ledger_ref["sha256"],
        trailing_lf=False,
    )
    transcript, _ = load_evidence_document(
        path=transcript_ref["path"],
        expected_sha256=transcript_ref["sha256"],
        trailing_lf=False,
    )
    transport_bundle, _ = load_evidence_document(
        path=transport_bundle_ref["path"],
        expected_sha256=transport_bundle_ref["sha256"],
        trailing_lf=False,
    )
    transport_manifest, _ = load_evidence_document(
        path=transport_manifest_ref["path"],
        expected_sha256=transport_manifest_ref["sha256"],
        trailing_lf=False,
    )
    transport_records = _load_transport_jsonl(path=raw_log_ref["path"], expected_sha256=raw_log_ref["sha256"])
    _validate_loaded_source_artifact_joins(
        pair=loaded_pair,
        pair_sha256=pair_sha,
        submission_sha256=submission_sha,
        off_identity=off_identity,
        exit_receipt=exit_receipt,
        evidence=evidence,
        main_ledger=main_ledger,
        transcript=transcript,
        transport_bundle=transport_bundle,
        transport_manifest=transport_manifest,
        transport_records=transport_records,
    )
    source_capture = {
        "arm": "off",
        "restart_count": 0,
        "authenticated_job": copy.deepcopy(off_identity),
        "job_receipts": {
            "pre": {
                "path": pre_path,
                "schema": PAIR_JOB_RECEIPT_SCHEMA,
                "sha256": pre_sha,
            },
            "exit": {
                "path": exit_path,
                "schema": PAIR_JOB_RECEIPT_SCHEMA,
                "sha256": trusted_exit_sha,
            },
        },
        "step1_evidence": evidence,
    }
    _validate_source_capture(source_capture)
    return AuthenticatedOffSourceCapture(
        source_capture=copy.deepcopy(source_capture),
        pair_manifest=copy.deepcopy(loaded_pair),
        pair_manifest_sha256=pair_sha,
        pair_submission_receipt=copy.deepcopy(submission),
        pair_submission_receipt_sha256=submission_sha,
        trusted_off_exit_receipt_path=trusted_exit_path,
        trusted_off_exit_receipt_sha256=trusted_exit_sha,
        pre_receipt=copy.deepcopy(pre),
        pre_receipt_sha256=pre_sha,
        exit_receipt=copy.deepcopy(exit_receipt),
        exit_receipt_sha256=trusted_exit_sha,
        main_ledger=copy.deepcopy(main_ledger),
        transcript_bundle=copy.deepcopy(transcript),
        transport_bundle=copy.deepcopy(transport_bundle),
        transport_manifest=copy.deepcopy(transport_manifest),
        transport_records=tuple(copy.deepcopy(transport_records)),
    )


def _reload_authenticated_off_source_capture(
    source: AuthenticatedOffSourceCapture,
) -> AuthenticatedOffSourceCapture:
    """Refresh a capability from its sealed Pair and receipt bytes.

    ``AuthenticatedOffSourceCapture`` contains detached mutable documents and has a
    public constructor, so its in-memory fields are never authority by themselves.
    Every production build/load/publish boundary reopens the historical roots and
    derives the complete OFF chain again.
    """
    if type(source) is not AuthenticatedOffSourceCapture:
        raise TypeError("authenticated_source must be AuthenticatedOffSourceCapture")
    pair = source.pair_manifest
    _require_exact_keys(pair, _pair_manifest_required_keys(), name="Pair")
    results_root = _absolute_path(pair["paths"]["results_root"], name="Pair results_root")
    refreshed = load_authenticated_off_source_capture(
        pair_manifest=pair,
        pair_manifest_path=f"{results_root}/PAIR_MANIFEST.json",
        pair_manifest_sha256=source.pair_manifest_sha256,
        pair_submission_receipt_path=f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
        pair_submission_receipt_sha256=source.pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=source.trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=source.trusted_off_exit_receipt_sha256,
    )
    return refreshed


def _validate_released_pair_submission_receipt(
    document: Mapping[str, Any],
    *,
    pair: Mapping[str, Any],
    pair_path: str,
    pair_sha256: str,
    receipt_path: str,
) -> dict[str, dict[str, str]]:
    _require_exact_keys(document, PAIR_SUBMISSION_RECEIPT_KEYS, name="Pair submission receipt")
    if (
        document["schema"] != PAIR_SUBMISSION_RECEIPT_SCHEMA
        or document["outcome"] != "released"
        or document["stage"] != "complete"
        or document["rollback_confirmed"] is not None
        or document["cancellations"] != []
        or document["pre_cancel_queries"] != []
        or document["post_cancel_queries"] != []
        or document["recovery_query"] is not None
    ):
        raise ValueError("Pair submission receipt is not one released complete cohort")
    expected_copies = {
        "acceptance": pair["acceptance"],
        "execution_environment": pair["execution_environment"],
        "model_transport": pair["model_transport"],
        "selection": pair["selection"],
        "wandb": pair["wandb"],
    }
    for name, expected in expected_copies.items():
        if not _exact(document[name], expected):
            raise ValueError(f"Pair submission receipt {name} binding differs")
    if not _exact(
        document["pair"],
        {"id": pair["pair_id"], "manifest": {"path": pair_path, "sha256": pair_sha256}},
    ):
        raise ValueError("Pair submission receipt Pair binding differs")
    if not _exact(
        document["receipt"],
        {"path": receipt_path, "schema": PAIR_SUBMISSION_RECEIPT_SCHEMA},
    ):
        raise ValueError("Pair submission receipt self-binding differs")
    expected_contract = pair["scheduler_submission"]["contract"]
    if not _exact(document["submission_contract"], expected_contract):
        raise ValueError("Pair submission receipt contract binding differs")
    if document["submission_nonce"] != pair["scheduler_submission"]["nonce"]:
        raise ValueError("Pair submission receipt nonce differs")
    expected_runtime_tools = {
        "manifest": pair["runtime_tools"]["manifest"],
        "schema": "nemo-rl-strict-runtime-tools-v2",
    }
    if not _exact(document["runtime_tools"], expected_runtime_tools):
        raise ValueError("Pair submission receipt runtime-tool binding differs")
    if not _exact(
        document["source"],
        {"bridge": pair["source"]["bridge"], "mcore": pair["source"]["mcore"]},
    ):
        raise ValueError("Pair submission receipt component source differs")

    scheduler_tools = document["scheduler_tools"]
    _require_exact_keys(
        scheduler_tools,
        frozenset({"client_environment", "sbatch", "scancel", "scontrol"}),
        name="Pair receipt scheduler_tools",
    )
    for name in ("sbatch", "scancel", "scontrol"):
        _file_ref(scheduler_tools[name], name=f"Pair receipt scheduler_tools.{name}")
    client = scheduler_tools["client_environment"]
    _require_exact_keys(
        client,
        frozenset({"ambient_merge", "env", "variables"}),
        name="Pair receipt scheduler client environment",
    )
    if client["ambient_merge"] is not False:
        raise ValueError("Pair receipt scheduler client ambient merge is not false")
    _file_ref(client["env"], name="Pair receipt scheduler env")
    if not isinstance(client["variables"], Mapping) or set(client["variables"]) != {
        "LC_ALL",
        "SLURM_CONF",
    }:
        raise ValueError("Pair receipt scheduler variable boundary differs")
    if client["variables"]["LC_ALL"] != "C":
        raise ValueError("Pair receipt scheduler locale differs")
    _file_ref(client["variables"]["SLURM_CONF"], name="Pair receipt scheduler SLURM_CONF")

    held = document["held_submissions"]
    authenticated = document["authenticated_jobs"]
    _require_exact_keys(held, frozenset({"off", "on"}), name="held_submissions")
    _require_exact_keys(authenticated, frozenset({"off", "on"}), name="authenticated_jobs")
    pair_id = _safe_id(pair["pair_id"], name="Pair pair_id", maximum=64)
    environment = pair["selection"]["environment"]
    nonce = pair["scheduler_submission"]["nonce"]
    submitter_euid = pair["scheduler_submission"]["identity"]["submitter_euid"]
    if type(submitter_euid) is not int or not 0 <= submitter_euid <= (1 << 31) - 1:
        raise ValueError("Pair submitter_euid is not an exact nonnegative int31")
    identities: dict[str, dict[str, str]] = {}
    candidate_ids: dict[str, str] = {}
    for arm in ("off", "on"):
        record = held[arm]
        _require_exact_keys(
            record,
            frozenset(
                {
                    "accepted_id_record",
                    "candidate_job_id",
                    "candidate_job_id_sha256_ascii_no_newline",
                    "candidate_job_id_source",
                    "submission_rpc",
                    "wrapper_status",
                }
            ),
            name=f"held_submissions.{arm}",
        )
        job_id = _job_id(record["candidate_job_id"], name=f"{arm} candidate job_id")
        if (
            record["candidate_job_id_source"] != "accepted-id-record"
            or record["candidate_job_id_sha256_ascii_no_newline"] != hashlib.sha256(job_id.encode("ascii")).hexdigest()
            or type(record["wrapper_status"]) is not int
            or record["wrapper_status"] != 0
        ):
            raise ValueError(f"Pair {arm} held-submission authority differs")
        accepted = record["accepted_id_record"]
        _require_exact_keys(
            accepted,
            frozenset({"parsed_job_id", "path", "sha256"}),
            name=f"held_submissions.{arm}.accepted_id_record",
        )
        expected_accepted_path = (
            f"{pair['paths']['results_root']}/strict_pair_submission_state/" f"{pair_id}/{arm}.job-id"
        )
        expected_accepted_bytes = f"{job_id}\n".encode("ascii")
        expected_accepted_sha = hashlib.sha256(expected_accepted_bytes).hexdigest()
        if (
            accepted["parsed_job_id"] != job_id
            or accepted["path"] != expected_accepted_path
            or accepted["sha256"] != expected_accepted_sha
        ):
            raise ValueError(f"Pair {arm} accepted-ID binding differs")
        if "accepted_id_records" in pair["scheduler_submission"]:
            contract = pair["scheduler_submission"]["accepted_id_records"][arm]
            if accepted["path"] != contract["path"]:
                raise ValueError(f"Pair {arm} accepted-ID path differs from Pair")
        accepted_bytes, accepted_sha = _load_stable_evidence_bytes(
            path=accepted["path"],
            expected_sha256=accepted["sha256"],
            name=f"Pair {arm} accepted-ID record",
            maximum=128,
        )
        if accepted_bytes != expected_accepted_bytes or accepted_sha != expected_accepted_sha:
            raise ValueError(f"Pair {arm} accepted-ID bytes differ")
        rpc = record["submission_rpc"]
        _require_exact_keys(
            rpc,
            frozenset(
                {
                    "drained_unix_ns",
                    "relay_status",
                    "sbatch_status",
                    "started_unix_ns",
                    "writer_drained",
                }
            ),
            name=f"held_submissions.{arm}.submission_rpc",
        )
        if (
            type(rpc["started_unix_ns"]) is not int
            or type(rpc["drained_unix_ns"]) is not int
            or rpc["started_unix_ns"] <= 0
            or rpc["drained_unix_ns"] < rpc["started_unix_ns"]
            or type(rpc["relay_status"]) is not int
            or rpc["relay_status"] != 0
            or type(rpc["sbatch_status"]) is not int
            or rpc["sbatch_status"] != 0
            or rpc["writer_drained"] is not True
        ):
            raise ValueError(f"Pair {arm} submission RPC did not drain successfully")
        expected_identity = {
            "comment": f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair_sha256}",
            "job_id": job_id,
            "job_name": f"{arm}-{environment}-{pair_id}",
            "user_id": str(submitter_euid),
        }
        if not _exact(authenticated[arm], [expected_identity]):
            raise ValueError(f"Pair receipt does not authenticate exactly one {arm} job")
        identities[arm] = expected_identity
        candidate_ids[arm] = job_id
    if candidate_ids["off"] == candidate_ids["on"]:
        raise ValueError("Pair receipt reuses one job ID for both arms")

    for phase in ("pre", "post"):
        _validate_pair_release_query(
            document[f"{phase}_release_query"],
            phase=phase,
            identities=identities,
            pair=pair,
            scontrol_path=scheduler_tools["scontrol"]["path"],
        )
    if document["pre_release_query"]["output_sha256_raw"] == document["post_release_query"]["output_sha256_raw"]:
        raise ValueError("Pair pre/post scheduler evidence digests are identical")
    ordered_csv = f"{candidate_ids['off']},{candidate_ids['on']}"
    release = document["release"]
    _require_exact_keys(
        release,
        frozenset({"argv", "output_sha256_ascii_no_newline", "status"}),
        name="Pair release evidence",
    )
    if (
        release["argv"] != [scheduler_tools["scontrol"]["path"], "release", ordered_csv]
        or type(release["status"]) is not int
        or release["status"] != 0
    ):
        raise ValueError("Pair release RPC evidence differs")
    _digest(
        release["output_sha256_ascii_no_newline"],
        name="Pair release output SHA-256",
    )
    expected_candidates = {
        "off": [candidate_ids["off"]],
        "on": [candidate_ids["on"]],
        "unattributed": [],
    }
    if not _exact(document["rollback_candidates"], expected_candidates):
        raise ValueError("Pair rollback candidate evidence differs")
    return identities


def _validate_pair_release_query(
    query: Any,
    *,
    phase: str,
    identities: Mapping[str, Mapping[str, str]],
    pair: Mapping[str, Any],
    scontrol_path: str,
) -> None:
    keys = frozenset(
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
        }
    )
    _require_exact_keys(query, keys, name=f"Pair {phase}-release query")
    ordered_ids = [identities["off"]["job_id"], identities["on"]["job_id"]]
    expected_candidates = {
        "off": [ordered_ids[0]],
        "on": [ordered_ids[1]],
        "unattributed": [],
    }
    if (
        query["argv"] != [scontrol_path, "show", "job", "--json", ",".join(ordered_ids)]
        or query["authenticated_job_ids"] != ordered_ids
        or not _exact(query["candidate_job_ids"], expected_candidates)
        or query["complete"] is not True
        or type(query["normalization_status"]) is not int
        or query["normalization_status"] != 0
        or query["phase"] != phase
        or query["securely_unlinked"] is not True
        or type(query["status"]) is not int
        or query["status"] != 0
        or query["unterminated_final_line"] is not False
        or query["unresolved_job_ids"] != []
        or type(query["byte_count"]) is not int
        or query["byte_count"] <= 0
        or type(query["line_count"]) is not int
        or query["line_count"] != 2
        or query["byte_count"] < query["line_count"]
    ):
        raise ValueError(f"Pair {phase}-release query lifecycle evidence differs")
    _digest(query["output_sha256_raw"], name=f"Pair {phase} query raw SHA-256")
    _require_exact_keys(query["records"], frozenset({"off", "on"}), name=f"Pair {phase} records")
    for arm in ("off", "on"):
        records = query["records"][arm]
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError(f"Pair {phase} query must contain one {arm} record")
        record = records[0]
        _require_exact_keys(
            record,
            frozenset(
                {
                    "comment",
                    "held",
                    "job_id",
                    "job_name",
                    "job_state",
                    "reason",
                    "user_id",
                    "work_dir",
                }
            ),
            name=f"Pair {phase} {arm} scheduler record",
        )
        identity = {name: record[name] for name in AUTHENTICATED_JOB_KEYS}
        if not _exact(identity, identities[arm]):
            raise ValueError(f"Pair {phase} {arm} scheduler identity differs")
        expected_work_dir = pair["execution_environment"]["arms"][arm]["scheduler"]["batch_working_directory"]
        if record["work_dir"] != expected_work_dir:
            raise ValueError(f"Pair {phase} {arm} scheduler work_dir differs")
        _absolute_path(record["work_dir"], name=f"Pair {phase} {arm} work_dir")
        for field in ("job_state", "reason"):
            value = record[field]
            if type(value) is not str or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
                raise ValueError(f"Pair {phase} {arm} scheduler state/reason differs")
        held = record["job_state"] == "PENDING" and record["reason"] == "JobHeldUser"
        if record["held"] is not held:
            raise ValueError(f"Pair {phase} {arm} held-state derivation differs")
        if phase == "pre" and not held:
            raise ValueError(f"Pair pre-release {arm} job was not held")
        if phase == "post" and (
            held
            or record["job_state"] not in {"PENDING", "CONFIGURING", "RUNNING"}
            or record["reason"] == "JobHeldUser"
        ):
            raise ValueError(f"Pair post-release {arm} job remains held")


def _validate_successful_off_job_receipts(
    pre: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
    *,
    pre_sha256: str,
    pair: Mapping[str, Any],
    pair_path: str,
    pair_sha256: str,
    submission: Mapping[str, Any],
    submission_path: str,
    submission_sha256: str,
    off_identity: Mapping[str, str],
) -> None:
    _require_exact_keys(pre, PAIR_PRE_RECEIPT_KEYS, name="source OFF PRE receipt")
    _require_exact_keys(exit_receipt, PAIR_EXIT_RECEIPT_KEYS, name="source OFF EXIT receipt")
    if (
        pre["schema"] != PAIR_JOB_RECEIPT_SCHEMA
        or pre["phase"] != "PRE"
        or pre["post_verified"] is not False
        or exit_receipt["schema"] != PAIR_JOB_RECEIPT_SCHEMA
        or exit_receipt["phase"] != "EXIT"
        or exit_receipt["post_verified"] is not True
        or type(exit_receipt["driver_exit_code"]) is not int
        or exit_receipt["driver_exit_code"] != 0
        or exit_receipt["pre_receipt_sha256"] != pre_sha256
    ):
        raise ValueError("source OFF PRE/EXIT is not one successful terminal lifecycle")
    for key in PAIR_PRE_RECEIPT_KEYS - {"phase", "post_verified"}:
        if not _exact(pre[key], exit_receipt[key]):
            raise ValueError(f"source OFF PRE/EXIT common field differs: {key}")

    environment = pair["selection"]["environment"]
    pair_id = pair["pair_id"]
    job_id = off_identity["job_id"]
    scalar_joins = {
        "arm": "off",
        "environment": environment,
        "job_id": job_id,
        "job_name": off_identity["job_name"],
        "pair_id": pair_id,
        "pair_manifest_sha256": pair_sha256,
        "restart_count": 0,
        "submission_nonce": pair["scheduler_submission"]["nonce"],
        "submission_receipt_path": submission_path,
        "submission_receipt_sha256": submission_sha256,
        "submission_contract_path": pair["scheduler_submission"]["contract"]["path"],
        "submission_contract_sha256": pair["scheduler_submission"]["contract"]["sha256"],
        "pair_campaign_sha256": pair["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        "reward_semantics_contract_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        "config_sha256": pair["selection"]["config"]["sha256"],
        "selected_config_sha256": pair["selection"]["config"]["sha256"],
        "reward_semantics_config_sha256": pair["selection"]["config"]["sha256"],
        "fixture_rows": pair["artifacts"]["fixture"]["rows"],
        "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
        "container_sha256": pair["artifacts"]["container"]["sha256"],
        "sandbox_container_sha256": pair["artifacts"]["sandbox_container"]["sha256"],
        "model_tree_sha256_v1": pair["artifacts"]["model"]["tree_sha256_v1"],
        "bridge_runnable_manifest_sha256": pair["deployment"]["bridge_runnable_manifest_sha256"],
        "mcore_runnable_manifest_sha256": pair["deployment"]["mcore_runnable_manifest_sha256"],
        "nemo_runnable_manifest_sha256": pair["deployment"]["nemo_runnable_manifest_sha256"],
        "deployment_ready": pair["deployment"]["ready"],
        "deployment_ready_sha256": pair["deployment"]["ready"],
        "deployment_ready_file_sha256": pair["deployment"]["ready_file_sha256"],
        "entrypoint_sha256": pair["source"]["entrypoint_sha256"],
        "gym_gitlink_commit": pair["source"]["gym"]["gitlink_commit"],
        "gym_tree": pair["source"]["gym"]["tree"],
        "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
        "source_head": pair["source"]["head"],
        "source_tree": pair["source"]["tree"],
        "strict_pair_arm_wrapper_sha256": pair["source"]["arm_wrapper_sha256"],
        "strict_pair_contract_sha256": pair["source"]["contract_sha256"],
        "strict_pair_parent_wrapper_sha256": pair["source"]["parent_wrapper_sha256"],
        "wrapper_sha256": pair["source"]["job_wrapper"]["sha256"],
        "runtime_tool_manifest_path": pair["runtime_tools"]["manifest"]["path"],
        "runtime_tool_manifest_sha256": pair["runtime_tools"]["manifest"]["sha256"],
    }
    for key, expected in scalar_joins.items():
        if not _exact(pre[key], expected):
            raise ValueError(f"source OFF receipt {key} differs from Pair authority")
    if not _exact(pre["selection"], pair["selection"]):
        raise ValueError("source OFF selection differs from Pair")
    if not _exact(pre["execution_environment"], pair["execution_environment"]):
        raise ValueError("source OFF execution environment differs from Pair")
    if not _exact(pre["source"], pair["source"]):
        raise ValueError("source OFF source tree differs from Pair")
    if not _exact(pre["container_entry_boundary"], pair["container_entry_boundary"]):
        raise ValueError("source OFF container-entry boundary differs from Pair")
    expected_entry_boundary_sha = hashlib.sha256(canonical_ascii_json(pair["container_entry_boundary"])).hexdigest()
    if pre["container_entry_boundary_sha256"] != expected_entry_boundary_sha:
        raise ValueError("source OFF container-entry boundary digest differs")

    selected_wandb = {
        "entity": pair["wandb"]["entity"],
        "group": pair["wandb"]["group"]["value"],
        "name": pair["wandb"]["arms"]["off"]["name"],
        "name_template": pair["wandb"]["arms"]["off"]["name_template"],
        "project": pair["wandb"]["project"],
        "resume": pair["wandb"]["resume"],
        "run_id": pair["wandb"]["arms"]["off"]["run_id"],
        "run_id_derivation": pair["wandb"]["run_id_derivation"],
    }
    if not _exact(pre["wandb"], selected_wandb):
        raise ValueError("source OFF W&B identity differs from Pair")
    resolved_export = _resolved_pair_slurm_export(pair=pair, pair_path=pair_path, pair_sha256=pair_sha256, arm="off")
    if not _exact(pre["slurm_export_boundary"], resolved_export):
        raise ValueError("source OFF resolved Slurm export differs from Pair")
    if pre["slurm_export_boundary_sha256"] != hashlib.sha256(canonical_ascii_json(resolved_export)).hexdigest():
        raise ValueError("source OFF resolved Slurm export digest differs")

    runtime_document = pair["runtime_tools"]["document"]
    tool_joins = {
        "runtime_tool_host_python": runtime_document["host"]["python"],
        "runtime_tool_container_python": runtime_document["container"]["python"],
        "runtime_tool_container_uv": runtime_document["container"]["uv"],
        "runtime_tool_uv_shim": runtime_document["container"]["uv_shim"],
    }
    for prefix, ref in tool_joins.items():
        if pre[f"{prefix}_path"] != ref["path"] or pre[f"{prefix}_sha256"] != ref["sha256"]:
            raise ValueError(f"source OFF {prefix} differs from Pair runtime tools")
    client = submission["scheduler_tools"]["client_environment"]
    expected_job_client = {
        "ambient_merge": False,
        "SLURM_CONF": client["variables"]["SLURM_CONF"],
        "propagated_to_inner_ray": True,
    }
    if not _exact(pre["scheduler_client_environment"], expected_job_client):
        raise ValueError("source OFF scheduler client environment differs")

    campaign_slurm = pair["campaign"]["slurm"]
    expected_scheduler = {
        "job_account": campaign_slurm["account"],
        "job_partition": campaign_slurm["partition"],
        "job_qos": campaign_slurm["qos"],
        "job_num_nodes": pair["campaign"]["nodes"],
        "gpus_per_node": 4,
    }
    for key, expected in expected_scheduler.items():
        if pre[key] != expected or type(pre[key]) is not type(expected):
            raise ValueError(f"source OFF scheduler allocation {key} differs")

    runtime = pair["runtime_attestation"]
    expected_receipt_dir = (
        f"{pair['paths']['results_root']}/off/{pair['determinism_receipt_dir']}/" f"{off_identity['job_id']}-0"
    )
    runtime_joins = {
        "runtime_attestation_expected_count": runtime["expected_count_per_fresh_process_group"],
        "runtime_attestation_marker_sha256": runtime["lines"]["off"]["sha256_ascii_no_newline"],
        "runtime_attestation_receipt_dir": expected_receipt_dir,
    }
    for key, expected in runtime_joins.items():
        if pre[key] != expected or type(pre[key]) is not type(expected):
            raise ValueError(f"source OFF {key} differs from Pair runtime attestation")
    for key in (
        "runtime_attestation_receipt_dir_device",
        "runtime_attestation_receipt_dir_inode",
    ):
        if type(pre[key]) is not int or pre[key] <= 0:
            raise ValueError(f"source OFF {key} must be one positive exact integer")
    if (
        type(exit_receipt["runtime_attestation_actual_count"]) is not int
        or exit_receipt["runtime_attestation_actual_count"] != runtime["expected_count_per_fresh_process_group"]
    ):
        raise ValueError("source OFF runtime attestation count differs")
    _digest(
        exit_receipt["runtime_attestation_aggregate_sha256"],
        name="source OFF runtime attestation aggregate SHA-256",
    )
    hashes = exit_receipt["runtime_attestation_receipts_sha256"]
    if not isinstance(hashes, Mapping) or len(hashes) != runtime_joins["runtime_attestation_expected_count"]:
        raise ValueError("source OFF runtime attestation receipt inventory differs")
    for name, digest in hashes.items():
        _ascii(name, name="runtime attestation receipt name", maximum=255)
        _digest(digest, name=f"runtime attestation receipt {name} SHA-256")
    if not isinstance(exit_receipt["hardware"], Mapping) or not isinstance(
        exit_receipt["scheduler_device_environment"], Mapping
    ):
        raise TypeError("source OFF EXIT lacks hardware/device observations")


def _resolved_pair_slurm_export(
    *, pair: Mapping[str, Any], pair_path: str, pair_sha256: str, arm: str
) -> dict[str, Any]:
    boundary = pair["slurm_export_boundary"]
    _require_exact_keys(
        boundary,
        frozenset(
            {
                "allowed_names",
                "ambient_merge",
                "arms",
                "format",
                "get_user_env",
                "job_argv",
                "schema",
            }
        ),
        name="Pair Slurm export boundary",
    )
    selected = boundary["arms"][arm]
    result = {key: copy.deepcopy(value) for key, value in boundary.items() if key != "arms"}
    result.update(
        {
            "arm": arm,
            "path": selected["path"],
            "sha256": selected["sha256"],
            "job_argv": [
                "--pair-manifest",
                pair_path,
                "--pair-manifest-sha256",
                pair_sha256,
                "--arm",
                arm,
            ],
        }
    )
    return result


def _validate_source_step1_index_paths(evidence: Any, *, results_root: str) -> None:
    _require_exact_keys(evidence, STEP1_EVIDENCE_KEYS, name="source step1 evidence")
    if evidence["schema"] != STEP1_EVIDENCE_INDEX_SCHEMA:
        raise ValueError("unexpected source step1 evidence schema")
    expected_refs = {
        "main_ledger": (
            f"{results_root}/off/strict_pair_step1_evidence/main-ledger.json",
            MAIN_LEDGER_SCHEMA,
        ),
        "transcript_bundle": (
            f"{results_root}/off/strict_pair_step1_evidence/transcript-bundle.json",
            TRANSCRIPT_BUNDLE_SCHEMA,
        ),
    }
    for name, (path, schema) in expected_refs.items():
        _artifact_ref(evidence[name], schema=schema, name=f"source {name}")
        if evidence[name]["path"] != path:
            raise ValueError(f"source {name} path differs from OFF results root")
    transport = evidence["model_transport"]
    _require_exact_keys(transport, TRANSPORT_EVIDENCE_KEYS, name="source model transport index")
    if transport["schema"] != TRANSPORT_EVIDENCE_INDEX_SCHEMA:
        raise ValueError("unexpected source model transport evidence schema")
    expected_transport = {
        "bundle": (
            f"{results_root}/off/strict_model_transport/model-transport-bundle.json",
            TRANSPORT_BUNDLE_SCHEMA,
        ),
        "manifest": (
            f"{results_root}/off/strict_model_transport/model-transport-manifest.json",
            TRANSPORT_MANIFEST_SCHEMA,
        ),
    }
    for name, (path, schema) in expected_transport.items():
        _artifact_ref(transport[name], schema=schema, name=f"source transport {name}")
        if transport[name]["path"] != path:
            raise ValueError(f"source transport {name} path differs")
    raw = transport["raw_log"]
    _require_exact_keys(raw, RAW_LOG_KEYS, name="source transport raw log")
    if (
        raw["path"] != f"{results_root}/off/strict_model_transport/model-transport.jsonl"
        or raw["record_schema"] != TRANSPORT_CALL_SCHEMA
        or type(raw["record_count"]) is not int
        or raw["record_count"] != 4
    ):
        raise ValueError("source transport raw-log contract differs")
    _digest(raw["sha256"], name="source transport raw log SHA-256")
    _digest(
        transport["ordered_entries_sha256"],
        name="source transport ordered entries SHA-256",
    )


def _validate_loaded_source_artifact_joins(
    *,
    pair: Mapping[str, Any],
    pair_sha256: str,
    submission_sha256: str,
    off_identity: Mapping[str, str],
    exit_receipt: Mapping[str, Any],
    evidence: Mapping[str, Any],
    main_ledger: Mapping[str, Any],
    transcript: Mapping[str, Any],
    transport_bundle: Mapping[str, Any],
    transport_manifest: Mapping[str, Any],
    transport_records: Sequence[Mapping[str, Any]],
) -> None:
    ledger_keys = frozenset(
        {
            "schema",
            "hash_domain",
            "pair_id",
            "environment",
            "arm",
            "mode",
            "step",
            "sample_count",
            "compared_fields",
            "generation",
            "bindings",
            "update_successful",
            "transcript_bundle",
            "rows",
            "step_totals",
            "cohort_sha256",
            "outputs_sha256",
            "rewards_sha256",
            "ordered_rows_sha256",
        }
    )
    transcript_keys = frozenset(
        {
            "schema",
            "hash_domain",
            "pair_id",
            "environment",
            "fixture_row",
            "arm",
            "mode",
            "attempt_id",
            "step",
            "sample_count",
            "generation",
            "bindings",
            "verifier_request_derivation",
            "model_transport_bundle",
            "entries",
            "entries_sha256",
        }
    )
    transport_bundle_keys = frozenset(
        {
            "arm",
            "capture_server",
            "capture_window",
            "endpoint",
            "entries",
            "entry_count",
            "environment",
            "hash_domain",
            "ordered_entries_sha256",
            "pair_id",
            "schema",
        }
    )
    transport_manifest_keys = frozenset(
        {
            "schema",
            "hash_domain",
            "pair_id",
            "environment",
            "arm",
            "pair_manifest_sha256",
            "authenticated_job_id",
            "submission_receipt_sha256",
            "capture_server",
            "main_transcript_bundle",
            "main_ledger",
            "transport_bundle",
            "transport_capture",
            "model_transport_policy_sha256",
            "entry_count",
            "ordered_entries_sha256",
        }
    )
    _require_exact_keys(main_ledger, ledger_keys, name="source main ledger")
    _require_exact_keys(transcript, transcript_keys, name="source transcript bundle")
    _require_exact_keys(transport_bundle, transport_bundle_keys, name="source transport bundle")
    _require_exact_keys(
        transport_manifest,
        transport_manifest_keys,
        name="source transport manifest",
    )
    pair_id = pair["pair_id"]
    environment = pair["selection"]["environment"]
    common = {
        "pair_id": pair_id,
        "environment": environment,
        "arm": "off",
        "mode": "observe",
        "step": 1,
        "sample_count": 4,
    }
    for document_name, document in (
        ("main ledger", main_ledger),
        ("transcript bundle", transcript),
    ):
        for key, expected in common.items():
            if document[key] != expected or type(document[key]) is not type(expected):
                raise ValueError(f"source {document_name} {key} differs")
        if document["hash_domain"] != HASH_DOMAIN:
            raise ValueError(f"source {document_name} hash domain differs")
    if main_ledger["schema"] != MAIN_LEDGER_SCHEMA:
        raise ValueError("unexpected source main ledger schema")
    if transcript["schema"] != TRANSCRIPT_BUNDLE_SCHEMA:
        raise ValueError("unexpected source transcript schema")
    if transcript["attempt_id"] is not None:
        raise ValueError("source main transcript attempt_id must be null")
    if (
        main_ledger["update_successful"] is not True
        or not isinstance(main_ledger["rows"], list)
        or len(main_ledger["rows"]) != 4
        or not isinstance(transcript["entries"], list)
        or len(transcript["entries"]) != 4
    ):
        raise ValueError("source step1 artifacts lack the exact successful K4 witness")
    if not _exact(main_ledger["transcript_bundle"], evidence["transcript_bundle"]):
        raise ValueError("source ledger transcript reference differs from EXIT")
    if not _exact(transcript["model_transport_bundle"], evidence["model_transport"]["bundle"]):
        raise ValueError("source transcript transport reference differs from EXIT")

    # Pair campaign generation also carries deployment-only controls such as
    # vLLM memory utilization.  Transcript-v4/main-ledger-v5 intentionally bind
    # only the request-facing generation controls plus their seed base.
    pair_generation = pair["campaign"]["generation"]
    generation = {
        "seed_base": pair["campaign"]["generation_seed_base"],
        **{name: pair_generation[name] for name in ("max_new_tokens", "temperature", "top_k", "top_p")},
    }
    if not _exact(main_ledger["generation"], generation) or not _exact(transcript["generation"], generation):
        raise ValueError("source step1 generation contract differs from Pair")
    transcript_bindings = {
        "pair_manifest_sha256": pair_sha256,
        "submission_receipt_sha256": submission_sha256,
        "job_id": off_identity["job_id"],
        "run_id": pair["wandb"]["arms"]["off"]["run_id"],
        "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
        "verifier_source_sha256": pair["selection"]["gym_resources"]["verifier_source"]["sha256"],
        "config_sha256": pair["selection"]["config"]["sha256"],
        "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
    }
    if not _exact(transcript["bindings"], transcript_bindings):
        raise ValueError("source transcript bindings differ from Pair/OFF authority")
    ledger_bindings = {
        **transcript_bindings,
        "restart_count": 0,
        "pair_campaign_sha256": pair["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
    }
    if not _exact(main_ledger["bindings"], ledger_bindings):
        raise ValueError("source main ledger bindings differ from Pair/OFF authority")

    if (
        transport_bundle["schema"] != TRANSPORT_BUNDLE_SCHEMA
        or transport_bundle["hash_domain"] != HASH_DOMAIN
        or transport_bundle["pair_id"] != pair_id
        or transport_bundle["environment"] != environment
        or transport_bundle["arm"] != "off"
        or type(transport_bundle["entry_count"]) is not int
        or transport_bundle["entry_count"] != 4
    ):
        raise ValueError("source model transport bundle identity differs")
    entries = transport_bundle["entries"]
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("source model transport bundle is not K4")
    if not _exact(entries, transport_records):
        raise ValueError("source transport raw log differs from sealed bundle")
    ordered_sha = domain_sha256("model-transport-ordered-entries", entries)
    if any(
        candidate != ordered_sha
        for candidate in (
            transport_bundle["ordered_entries_sha256"],
            evidence["model_transport"]["ordered_entries_sha256"],
        )
    ):
        raise ValueError("source model transport ordered digest does not close")
    entry_keys = frozenset(
        {
            "arrival_index",
            "entry_sha256",
            "generation_seed",
            "request_body_base64",
            "request_body_sha256",
            "request_payload",
            "request_payload_sha256",
            "response_body_base64",
            "response_body_sha256",
            "response_payload",
            "response_payload_sha256",
            "rollout_index",
            "schema",
        }
    )
    for index, entry in enumerate(entries):
        _require_exact_keys(entry, entry_keys, name=f"source transport entry {index}")
        if entry["schema"] != TRANSPORT_CALL_SCHEMA:
            raise ValueError("source transport call schema differs")
        if type(entry["rollout_index"]) is not int or entry["rollout_index"] != index:
            raise ValueError("source transport entries are not in rollout order")
        entry_preimage = {
            "pair_id": pair_id,
            "environment": environment,
            "arm": "off",
            "endpoint": transport_bundle["endpoint"],
            "capture_server": transport_bundle["capture_server"],
            "entry": {key: value for key, value in entry.items() if key != "entry_sha256"},
        }
        if entry["entry_sha256"] != domain_sha256("model-transport-entry", entry_preimage):
            raise ValueError(f"source transport entry {index} digest does not close")

    expected_manifest = {
        "schema": TRANSPORT_MANIFEST_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": "off",
        "pair_manifest_sha256": pair_sha256,
        "authenticated_job_id": off_identity["job_id"],
        "submission_receipt_sha256": submission_sha256,
        "capture_server": transport_bundle["capture_server"],
        "main_transcript_bundle": evidence["transcript_bundle"],
        "main_ledger": evidence["main_ledger"],
        "transport_bundle": evidence["model_transport"]["bundle"],
        "transport_capture": evidence["model_transport"]["raw_log"],
        "model_transport_policy_sha256": pair["model_transport"]["policy_sha256"],
        "entry_count": 4,
        "ordered_entries_sha256": ordered_sha,
    }
    if not _exact(transport_manifest, expected_manifest):
        raise ValueError("source transport manifest differs from acyclic authority")
    if not _exact(exit_receipt["step1_evidence"], evidence):
        raise AssertionError("unreachable detached EXIT evidence mismatch")


def _load_transport_jsonl(*, path: str, expected_sha256: str) -> list[dict[str, Any]]:
    raw, _ = _load_stable_evidence_bytes(
        path=path,
        expected_sha256=expected_sha256,
        name="source model transport raw log",
        maximum=256 * 1024 * 1024,
    )
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("source transport log must use exact LF-only framing")
    lines = raw[:-1].split(b"\n")
    if len(lines) != 4 or any(not line for line in lines):
        raise ValueError("source transport log must contain four nonblank records")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        value = _parse_strict_json_object(line, name=f"source transport record {index}")
        if line != canonical_ascii_json(value):
            raise ValueError(f"source transport record {index} is not canonical")
        records.append(value)
    return records


def _load_unanchored_evidence_document(*, path: str, trailing_lf: bool, name: str) -> tuple[dict[str, Any], str]:
    raw, digest = _load_stable_evidence_bytes(
        path=path,
        expected_sha256=None,
        name=name,
        maximum=256 * 1024 * 1024,
    )
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError(f"{name} must end in exactly one LF")
        payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            raise ValueError(f"{name} must not end in LF")
        payload = raw
    value = _parse_strict_json_object(payload, name=name)
    expected = canonical_ascii_json(value) + (b"\n" if trailing_lf else b"")
    if raw != expected:
        raise ValueError(f"{name} is not exact canonical ASCII JSON")
    return value, digest


def _load_anchored_evidence_document(
    *,
    path: str,
    expected_sha256: str,
    trailing_lf: bool,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Stable no-follow load one canonical document against external authority."""
    raw, digest = _load_stable_evidence_bytes(
        path=path,
        expected_sha256=expected_sha256,
        name=name,
        maximum=256 * 1024 * 1024,
    )
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError(f"{name} must end in exactly one LF")
        payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            raise ValueError(f"{name} must not end in LF")
        payload = raw
    value = _parse_strict_json_object(payload, name=name)
    expected = canonical_ascii_json(value) + (b"\n" if trailing_lf else b"")
    if raw != expected:
        raise ValueError(f"{name} is not exact canonical ASCII JSON")
    return value, digest


def _parse_strict_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON member {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} root must be an object")
    return value


def _load_stable_evidence_bytes(
    *,
    path: str,
    expected_sha256: str | None,
    name: str,
    maximum: int,
    required_mode: int | None = 0o400,
    allowed_parent_modes: frozenset[int] | None = frozenset({0o700}),
    allow_empty: bool = False,
    required_executable: bool | None = None,
) -> tuple[bytes, str]:
    evidence_path = Path(_absolute_path(path, name=f"{name} path"))
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("stable evidence maximum must be positive")
    minimum = 0 if allow_empty else 1
    parent_fd = _open_absolute_directory_no_symlinks(evidence_path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        parent_mode = stat.S_IMODE(parent_before.st_mode)
        if not stat.S_ISDIR(parent_before.st_mode) or parent_before.st_uid != os.geteuid():
            raise RuntimeError(f"{name} parent must be an EUID-owned directory")
        if allowed_parent_modes is None:
            if parent_mode & 0o222:
                raise RuntimeError(f"{name} parent must be nonwritable")
        elif parent_mode not in allowed_parent_modes:
            modes = ",".join(f"{mode:04o}" for mode in sorted(allowed_parent_modes))
            raise RuntimeError(f"{name} parent must be EUID-owned with mode in {{{modes}}}")
        pre_named = os.stat(
            evidence_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        pre_mode = stat.S_IMODE(pre_named.st_mode)
        if (
            not stat.S_ISREG(pre_named.st_mode)
            or pre_named.st_uid != os.geteuid()
            or pre_named.st_nlink != 1
            or not minimum <= pre_named.st_size <= maximum
            or (required_mode is not None and pre_mode != required_mode)
            or (required_mode is None and bool(pre_mode & 0o222))
            or (required_executable is not None and bool(pre_mode & 0o111) != required_executable)
        ):
            raise RuntimeError(f"{name} differs from the stable regular-file contract")
        descriptor = os.open(
            evidence_path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                _stable_file_fingerprint(pre_named) != _stable_file_fingerprint(before)
                or not stat.S_ISREG(before.st_mode)
                or not minimum <= before.st_size <= maximum
            ):
                raise RuntimeError(f"{name} changed before stable read")
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            named = os.stat(evidence_path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_no_symlinks(evidence_path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
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
        _stable_directory_identity(parent_before)
        == _stable_directory_identity(parent_after)
        == _stable_directory_identity(fresh_parent)
    ):
        raise RuntimeError(f"{name} parent changed during stable read")
    fingerprints = {
        _stable_file_fingerprint(pre_named),
        _stable_file_fingerprint(before),
        _stable_file_fingerprint(after),
        _stable_file_fingerprint(named),
        _stable_file_fingerprint(fresh_named),
    }
    if len(fingerprints) != 1 or len(raw) != after.st_size:
        raise RuntimeError(f"{name} changed during stable read")
    file_mode = stat.S_IMODE(after.st_mode)
    if not stat.S_ISREG(after.st_mode) or after.st_uid != os.geteuid() or after.st_nlink != 1:
        raise RuntimeError(f"{name} must be an EUID-owned single-link regular file")
    if required_mode is not None and file_mode != required_mode:
        raise RuntimeError(f"{name} must have exact mode {required_mode:04o}")
    if required_mode is None and file_mode & 0o222:
        raise RuntimeError(f"{name} must be nonwritable")
    if required_executable is not None and bool(file_mode & 0o111) != required_executable:
        raise RuntimeError(f"{name} executable mode differs from snapshot manifest")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        expected = _digest(expected_sha256, name=f"{name} expected SHA-256")
        if actual_sha != expected:
            raise ValueError(f"{name} bytes differ from expected SHA-256")
    return raw, actual_sha


def _verify_stable_system_executable(*, path: str, expected_sha256: str, name: str) -> None:
    """Hash one root- or EUID-owned immutable executable without symlink traversal."""
    executable_path = Path(_absolute_path(path, name=f"{name} path"))
    parent_fd = _open_absolute_directory_no_symlinks(executable_path.parent)
    try:
        parent_before = os.fstat(parent_fd)
        pre_named = os.stat(
            executable_path.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        pre_mode = stat.S_IMODE(pre_named.st_mode)
        if (
            not stat.S_ISREG(pre_named.st_mode)
            or not 0 < pre_named.st_size <= 256 * 1024 * 1024
            or pre_named.st_nlink != 1
            or pre_named.st_uid not in {0, os.geteuid()}
            or not pre_mode & 0o111
            or pre_mode & 0o022
            or (pre_named.st_uid == os.geteuid() and pre_mode & 0o200)
        ):
            raise RuntimeError(f"{name} is not an admitted immutable executable")
        descriptor = os.open(
            executable_path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            if _stable_file_fingerprint(pre_named) != _stable_file_fingerprint(before) or not stat.S_ISREG(
                before.st_mode
            ):
                raise RuntimeError(f"{name} changed before stable read")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise RuntimeError(f"{name} truncated during stable read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise RuntimeError(f"{name} grew during stable read")
            after = os.fstat(descriptor)
            named = os.stat(executable_path.name, dir_fd=parent_fd, follow_symlinks=False)
        finally:
            os.close(descriptor)
        parent_after = os.fstat(parent_fd)
        fresh_parent_fd = _open_absolute_directory_no_symlinks(executable_path.parent)
        try:
            fresh_parent = os.fstat(fresh_parent_fd)
            fresh_named = os.stat(
                executable_path.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(fresh_parent_fd)
    finally:
        os.close(parent_fd)
    if not (
        _stable_directory_identity(parent_before)
        == _stable_directory_identity(parent_after)
        == _stable_directory_identity(fresh_parent)
    ):
        raise RuntimeError(f"{name} parent changed during stable read")
    if (
        len(
            {
                _stable_file_fingerprint(pre_named),
                _stable_file_fingerprint(before),
                _stable_file_fingerprint(after),
                _stable_file_fingerprint(named),
                _stable_file_fingerprint(fresh_named),
            }
        )
        != 1
    ):
        raise RuntimeError(f"{name} changed during stable read")
    mode = stat.S_IMODE(after.st_mode)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or after.st_uid not in {0, os.geteuid()}
        or not mode & 0o111
        or mode & 0o022
        or (after.st_uid == os.geteuid() and mode & 0o200)
    ):
        raise RuntimeError(f"{name} is not an admitted immutable executable")
    actual_sha256 = hashlib.sha256(b"".join(chunks)).hexdigest()
    if actual_sha256 != _digest(expected_sha256, name=f"{name} SHA-256"):
        raise ValueError(f"{name} bytes differ from authenticated SHA-256")


def _open_absolute_directory_no_symlinks(path: Path) -> int:
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise ValueError("stable evidence parent must be a canonical absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _stable_file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _stable_directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


_SNAPSHOT_SHA_MANIFEST = "strict-pair-snapshot-manifest.sha256"
_SNAPSHOT_SYMLINK_MANIFEST = "strict-pair-snapshot-symlinks.json"
_SNAPSHOT_MODE_MANIFEST = "strict-pair-snapshot-modes.json"


def _snapshot_relative_path(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or "\n" in value
        or "\r" in value
        or posixpath.normpath(value) != value
        or ".." in value.split("/")
        or value == "."
    ):
        raise ValueError(f"{name} must be a canonical POSIX relative path")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be ASCII") from error
    return value


def _parse_snapshot_sha_manifest(raw: bytes, *, name: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or raw.endswith(b"\n\n"):
        raise ValueError(f"{name} must use exact complete LF framing")
    result: dict[str, str] = {}
    for index, line in enumerate(raw[:-1].split(b"\n")):
        digest_raw, separator, path_raw = line.partition(b"  ")
        if not separator or not path_raw:
            raise ValueError(f"{name} line {index} is malformed")
        try:
            digest = digest_raw.decode("ascii")
            relative = path_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} line {index} must be ASCII") from error
        _digest(digest, name=f"{name} line {index} SHA-256")
        _snapshot_relative_path(relative, name=f"{name} line {index} path")
        if relative in result:
            raise ValueError(f"{name} repeats snapshot path {relative}")
        result[relative] = digest
    if not result:
        raise ValueError(f"{name} must contain at least one file")
    return result


def _load_snapshot_json(
    *,
    snapshot_root: str,
    relative_path: str,
    expected_sha256: str,
    name: str,
    trailing_lf: bool,
    parent_mode: int = 0o700,
) -> dict[str, Any]:
    raw, _ = _load_stable_evidence_bytes(
        path=f"{snapshot_root}/{relative_path}",
        expected_sha256=expected_sha256,
        name=name,
        maximum=16 * 1024 * 1024,
        required_mode=0o400,
        allowed_parent_modes=frozenset({parent_mode}),
    )
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            raise ValueError(f"{name} must end in exactly one LF")
        payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            raise ValueError(f"{name} must not end in LF")
        payload = raw
    document = _parse_strict_json_object(payload, name=name)
    expected = canonical_ascii_json(document) + (b"\n" if trailing_lf else b"")
    if raw != expected:
        raise ValueError(f"{name} is not canonical ASCII JSON")
    return document


def _snapshot_manifest_documents(
    *, snapshot_root: str, manifest: Mapping[str, str], parent_mode: int
) -> tuple[dict[str, str], dict[str, bool]]:
    for relative in (_SNAPSHOT_SYMLINK_MANIFEST, _SNAPSHOT_MODE_MANIFEST):
        if relative not in manifest:
            raise ValueError(f"snapshot SHA manifest omits {relative}")
    symlink_document = _load_snapshot_json(
        snapshot_root=snapshot_root,
        relative_path=_SNAPSHOT_SYMLINK_MANIFEST,
        expected_sha256=manifest[_SNAPSHOT_SYMLINK_MANIFEST],
        name="snapshot symlink manifest",
        trailing_lf=True,
        parent_mode=parent_mode,
    )
    _require_exact_keys(
        symlink_document,
        frozenset({"schema", "symlinks"}),
        name="snapshot symlink manifest",
    )
    if symlink_document["schema"] != "nemo-rl-strict-snapshot-symlinks-v1":
        raise ValueError("unexpected snapshot symlink manifest schema")
    symlinks = symlink_document["symlinks"]
    if not isinstance(symlinks, Mapping):
        raise TypeError("snapshot symlinks must be an object")
    parsed_symlinks: dict[str, str] = {}
    for raw_path, target in symlinks.items():
        relative = _snapshot_relative_path(raw_path, name="snapshot symlink path")
        if (
            type(target) is not str
            or not target
            or target.startswith("/")
            or "\x00" in target
            or "\n" in target
            or "\r" in target
        ):
            raise ValueError("snapshot symlink target is invalid")
        parsed_symlinks[relative] = target

    mode_document = _load_snapshot_json(
        snapshot_root=snapshot_root,
        relative_path=_SNAPSHOT_MODE_MANIFEST,
        expected_sha256=manifest[_SNAPSHOT_MODE_MANIFEST],
        name="snapshot mode manifest",
        trailing_lf=True,
        parent_mode=parent_mode,
    )
    _require_exact_keys(
        mode_document,
        frozenset({"schema", "regular_file_executable"}),
        name="snapshot mode manifest",
    )
    if mode_document["schema"] != "nemo-rl-strict-snapshot-modes-v1":
        raise ValueError("unexpected snapshot mode manifest schema")
    modes = mode_document["regular_file_executable"]
    if not isinstance(modes, Mapping):
        raise TypeError("snapshot executable modes must be an object")
    parsed_modes: dict[str, bool] = {}
    for raw_path, executable in modes.items():
        relative = _snapshot_relative_path(raw_path, name="snapshot mode path")
        if type(executable) is not bool:
            raise TypeError("snapshot executable flags must be exact booleans")
        parsed_modes[relative] = executable
    if set(parsed_modes) != set(manifest):
        raise ValueError("snapshot mode inventory differs from SHA manifest")
    return parsed_symlinks, parsed_modes


def _verify_pair_snapshot_tree(
    *,
    snapshot_root: str,
    manifest: Mapping[str, str],
    symlinks: Mapping[str, str],
    executable_modes: Mapping[str, bool],
) -> None:
    root = Path(snapshot_root)
    metadata = os.lstat(root)
    root_mode = stat.S_IMODE(metadata.st_mode)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or root_mode & 0o222
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError("Pair ON snapshot root must be EUID-owned and nonwritable")

    def walk_inventory() -> tuple[set[str], dict[str, str], set[str], dict[str, tuple[int, ...]]]:
        actual_regular: set[str] = set()
        actual_symlinks: dict[str, str] = {}
        actual_directories: set[str] = {""}
        fingerprints: dict[str, tuple[int, ...]] = {"": _stable_file_fingerprint(os.lstat(root))}

        def raise_walk_error(error: OSError) -> None:
            raise error

        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False, onerror=raise_walk_error
        ):
            directory_names.sort()
            file_names.sort()
            for entry in list(directory_names):
                path = Path(directory) / entry
                relative = path.relative_to(root).as_posix()
                child = os.lstat(path)
                fingerprints[relative] = _stable_file_fingerprint(child)
                if stat.S_ISLNK(child.st_mode):
                    if child.st_uid != os.geteuid():
                        raise RuntimeError(f"Pair ON snapshot symlink owner differs: {relative}")
                    actual_symlinks[relative] = os.readlink(path)
                    directory_names.remove(entry)
                    continue
                child_mode = stat.S_IMODE(child.st_mode)
                if not stat.S_ISDIR(child.st_mode) or child_mode & 0o222 or child.st_uid != os.geteuid():
                    raise RuntimeError(f"Pair ON snapshot directory mode/owner differs: {relative}")
                actual_directories.add(relative)
            for entry in file_names:
                path = Path(directory) / entry
                relative = path.relative_to(root).as_posix()
                child = os.lstat(path)
                fingerprints[relative] = _stable_file_fingerprint(child)
                if stat.S_ISLNK(child.st_mode):
                    if child.st_uid != os.geteuid():
                        raise RuntimeError(f"Pair ON snapshot symlink owner differs: {relative}")
                    actual_symlinks[relative] = os.readlink(path)
                elif stat.S_ISREG(child.st_mode):
                    actual_regular.add(relative)
                else:
                    raise RuntimeError(f"Pair ON snapshot contains special file: {relative}")
        return actual_regular, actual_symlinks, actual_directories, fingerprints

    actual_regular, actual_symlinks, actual_directories, before_fingerprints = walk_inventory()
    if actual_regular != set(manifest) | {_SNAPSHOT_SHA_MANIFEST}:
        raise ValueError("Pair ON snapshot regular-file inventory differs")
    if actual_symlinks != dict(symlinks):
        raise ValueError("Pair ON snapshot symlink inventory differs")
    expected_directories = {""}
    for relative in actual_regular | set(actual_symlinks):
        parts = relative.split("/")[:-1]
        for length in range(1, len(parts) + 1):
            expected_directories.add("/".join(parts[:length]))
    if actual_directories != expected_directories:
        raise ValueError("Pair ON snapshot directory inventory differs")

    # Resolve each authenticated symlink one component at a time without following
    # it while reading bytes.  In particular, never collapse ``x/..`` until ``x``
    # has itself been expanded: the kernel follows an embedded symlink before
    # applying a later ``..``.  Targets may traverse within the root, but may not
    # escape it, form a cycle, or traverse an unauthenticated/non-directory entry.
    inventory = actual_regular | actual_directories | set(actual_symlinks)

    def resolve_components(
        components: Sequence[str],
        resolved: list[str],
        active_symlinks: frozenset[str],
        *,
        source_relative: str,
    ) -> list[str]:
        pending = list(components)
        while pending:
            component = pending.pop(0)
            if component in {"", "."}:
                continue
            if component == "..":
                if not resolved:
                    raise ValueError(f"Pair ON snapshot symlink escapes root: {source_relative}")
                resolved.pop()
                continue
            candidate = "/".join([*resolved, component])
            if candidate in actual_symlinks:
                if candidate in active_symlinks:
                    raise ValueError(f"Pair ON snapshot symlink cycle: {source_relative}")
                resolved = resolve_components(
                    actual_symlinks[candidate].split("/"),
                    resolved,
                    active_symlinks | {candidate},
                    source_relative=source_relative,
                )
                if pending and "/".join(resolved) not in actual_directories:
                    raise ValueError(f"Pair ON snapshot symlink traverses non-directory: " f"{source_relative}")
                continue
            if candidate not in inventory:
                raise ValueError(f"Pair ON snapshot symlink target is absent: {source_relative}")
            resolved.append(component)
            if pending and candidate not in actual_directories:
                raise ValueError(f"Pair ON snapshot symlink traverses non-directory: " f"{source_relative}")
        return resolved

    for source_relative in sorted(actual_symlinks):
        resolved = resolve_components(
            source_relative.split("/"),
            [],
            frozenset(),
            source_relative=source_relative,
        )
        if "/".join(resolved) not in inventory:
            raise AssertionError("resolved snapshot symlink left authenticated inventory")
    for relative, expected_sha256 in manifest.items():
        _, actual_sha256 = _load_stable_evidence_bytes(
            path=f"{snapshot_root}/{relative}",
            expected_sha256=expected_sha256,
            name=f"Pair ON snapshot member {relative}",
            maximum=256 * 1024 * 1024,
            required_mode=None,
            allowed_parent_modes=None,
            allow_empty=True,
            required_executable=executable_modes[relative],
        )
        if actual_sha256 != expected_sha256:
            raise AssertionError("unreachable Pair ON snapshot digest mismatch")
    post_regular, post_symlinks, post_directories, after_fingerprints = walk_inventory()
    if (
        post_regular != actual_regular
        or post_symlinks != actual_symlinks
        or post_directories != actual_directories
        or after_fingerprints != before_fingerprints
    ):
        raise RuntimeError("Pair ON snapshot changed during verification")


def _load_authenticated_replay_snapshot(
    *,
    source: AuthenticatedOffSourceCapture,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
    pair = source.pair_manifest
    on_snapshot = copy.deepcopy(pair["source"]["snapshots"]["on"])
    _require_exact_keys(
        on_snapshot,
        frozenset({"config_sha256", "entrypoint_sha256", "manifest_sha256", "path"}),
        name="Pair ON snapshot ref",
    )
    root = _absolute_path(on_snapshot["path"], name="Pair ON snapshot path")
    manifest_raw, manifest_sha256 = _load_stable_evidence_bytes(
        path=f"{root}/{_SNAPSHOT_SHA_MANIFEST}",
        expected_sha256=on_snapshot["manifest_sha256"],
        name="Pair ON snapshot SHA manifest",
        maximum=64 * 1024 * 1024,
        required_mode=0o400,
        allowed_parent_modes=None,
    )
    if manifest_sha256 != on_snapshot["manifest_sha256"]:
        raise AssertionError("unreachable Pair ON snapshot manifest mismatch")
    snapshot_manifest = _parse_snapshot_sha_manifest(manifest_raw, name="Pair ON snapshot SHA manifest")
    config_relative = _snapshot_relative_path(pair["selection"]["config"]["path"], name="Pair selected config path")
    expected_config_sha256 = pair["selection"]["config"]["sha256"]
    if (
        on_snapshot["config_sha256"] != expected_config_sha256
        or pair["source"]["config_sha256"] != expected_config_sha256
        or snapshot_manifest.get(config_relative) != expected_config_sha256
    ):
        raise ValueError("Pair ON snapshot config digest does not close")
    entrypoint_relative = "examples/run_grpo_single_controller.py"
    expected_entrypoint_sha256 = pair["source"]["entrypoint_sha256"]
    if (
        on_snapshot["entrypoint_sha256"] != expected_entrypoint_sha256
        or snapshot_manifest.get(entrypoint_relative) != expected_entrypoint_sha256
    ):
        raise ValueError("Pair ON snapshot entrypoint digest does not close")
    gym_prefix = f"{GYM_SNAPSHOT_RELATIVE_ROOT}/"
    for name in ("config", "requirements", "verifier_source"):
        reference = pair["selection"]["gym_resources"][name]
        relative = gym_prefix + _snapshot_relative_path(reference["path"], name=f"Pair selected Gym {name} path")
        if snapshot_manifest.get(relative) != reference["sha256"]:
            raise ValueError(f"Pair ON snapshot selected Gym {name} digest does not close")
    resource_only_relative = gym_prefix + GYM_REASONING_RESOURCE_ONLY_CONFIG["path"]
    if (
        pair["selection"]["environment"] == "reasoning_gym"
        and snapshot_manifest.get(resource_only_relative) != GYM_REASONING_RESOURCE_ONLY_CONFIG["sha256"]
    ):
        raise ValueError("Pair ON snapshot resource-only config digest does not close")
    root_mode = stat.S_IMODE(os.lstat(root).st_mode)
    symlinks, executable_modes = _snapshot_manifest_documents(
        snapshot_root=root, manifest=snapshot_manifest, parent_mode=root_mode
    )
    _verify_pair_snapshot_tree(
        snapshot_root=root,
        manifest=snapshot_manifest,
        symlinks=symlinks,
        executable_modes=executable_modes,
    )
    # Gym must be imported from the authenticated nested checkout.  Requiring
    # every source-root component to be a real directory prevents import
    # traversal through even an otherwise in-root authenticated symlink.
    current = root
    for component in GYM_SNAPSHOT_RELATIVE_ROOT.split("/"):
        current = f"{current}/{component}"
        if not stat.S_ISDIR(os.lstat(current).st_mode):
            raise ValueError("Pair ON snapshot Gym source root is not a directory")
    gym_source_root = {
        "snapshot_relative_path": GYM_SNAPSHOT_RELATIVE_ROOT,
        "host_path": f"{root}/{GYM_SNAPSHOT_RELATIVE_ROOT}",
        "container_path": GYM_CONTAINER_ROOT,
    }
    program: dict[str, dict[str, str]] = {}
    for name, relative in REPLAY_PROGRAM_PATHS.items():
        digest = snapshot_manifest.get(relative)
        if digest is None:
            raise ValueError(f"Pair ON snapshot omits replay program {name}")
        program[name] = {"path": relative, "sha256": digest}
    _validate_program(program)
    return on_snapshot, program, gym_source_root


def _replay_control_root(pair: Mapping[str, Any], *, attempt_id: str) -> str:
    pair_id = _safe_id(pair["pair_id"], name="Pair pair_id", maximum=64)
    attempt = _attempt(attempt_id)
    results_root = _absolute_path(pair["paths"]["results_root"], name="Pair results_root")
    return f"{results_root}/captured_replay/replay_submission_state/{pair_id}/{attempt}"


def _load_replay_slurm_export(
    *, source: AuthenticatedOffSourceCapture, attempt_id: str
) -> tuple[str, str, tuple[tuple[str, bytes], ...]]:
    pair = source.pair_manifest
    attempt = _attempt(attempt_id)
    results_root = _absolute_path(pair["paths"]["results_root"], name="Pair results_root")
    path = f"{results_root}/captured_replay/slurm_exports/{pair['pair_id']}/{attempt}.env"
    raw, digest = _load_stable_evidence_bytes(
        path=path,
        expected_sha256=None,
        name="replay Slurm export",
        maximum=16 * 1024 * 1024,
    )
    if not raw.endswith(b"\0") or raw.endswith(b"\0\0"):
        raise ValueError("replay Slurm export must have exact terminal NUL framing")
    records: list[tuple[str, bytes]] = []
    for index, record in enumerate(raw[:-1].split(b"\0")):
        raw_name, separator, value = record.partition(b"=")
        if not separator:
            raise ValueError(f"replay Slurm export record {index} is malformed")
        try:
            name = raw_name.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("replay Slurm export name must be ASCII") from error
        if name not in SLURM_EXPORT_ALLOWED_NAMES:
            raise ValueError(f"replay Slurm export name is not admitted: {name}")
        records.append((name, value))
    names = [name for name, _ in records]
    if names != list(SLURM_EXPORT_ALLOWED_NAMES):
        raise ValueError("replay Slurm export records must be exact Pair79 order")
    values = dict(records)
    expected_values = {name: b"" for name in SLURM_EXPORT_ALLOWED_NAMES}
    exact_ascii = {
        "EXPECTED_GYM_GITLINK_COMMIT": pair["source"]["gym"]["gitlink_commit"],
        "EXPECTED_GYM_TREE": pair["source"]["gym"]["tree"],
        "PAIR_ID": pair["pair_id"],
        "RESULTS_DIR": f"{results_root}/captured_replay/{attempt}",
        "STRICT_PAIR_ENVIRONMENT": pair["selection"]["environment"],
        "STRICT_PREBUILT_SNAPSHOT_DIR": pair["source"]["snapshots"]["on"]["path"],
    }
    for name, expected in exact_ascii.items():
        expected_values[name] = expected.encode("ascii")
    for name in SLURM_EXPORT_ALLOWED_NAMES:
        if values[name] != expected_values[name]:
            raise ValueError(f"replay Slurm export {name} value differs")
    return path, digest, tuple(records)


def _submission_contract_path(pair: Mapping[str, Any], *, attempt_id: str) -> str:
    return f"{_replay_control_root(pair, attempt_id=attempt_id)}.submission-contract.json"


def build_replay_submission_contract(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    submission_nonce: str,
) -> dict[str, Any]:
    """Build the acyclic login-side contract from authenticated static bytes."""
    source = _reload_authenticated_off_source_capture(authenticated_source)
    pair = source.pair_manifest
    attempt = _attempt(attempt_id)
    nonce = _digest(submission_nonce, name="replay submission nonce")
    source_snapshot, program, _ = _load_authenticated_replay_snapshot(source=source)
    export_path, export_sha256, _ = _load_replay_slurm_export(source=source, attempt_id=attempt)
    snapshot_root = source_snapshot["path"]
    host_tools = pair["runtime_tools"]["document"]["host"]
    _verify_stable_system_executable(
        path=host_tools["sbatch"]["path"],
        expected_sha256=host_tools["sbatch"]["sha256"],
        name="replay sbatch program",
    )
    if pair["scheduler_submission"]["identity"]["submitter_euid"] != os.geteuid():
        raise RuntimeError("replay submitter EUID differs from the current process")
    document = {
        "schema": REPLAY_SUBMISSION_CONTRACT_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair["pair_id"],
        "environment": pair["selection"]["environment"],
        "attempt_id": attempt,
        "submission_nonce": nonce,
        "submitter_euid": pair["scheduler_submission"]["identity"]["submitter_euid"],
        "sbatch_program": copy.deepcopy(host_tools["sbatch"]),
        "job_wrapper": {
            "path": f"{snapshot_root}/{program['job_wrapper']['path']}",
            "sha256": program["job_wrapper"]["sha256"],
        },
        "submission_launcher": {
            "path": f"{snapshot_root}/{program['submission_launcher']['path']}",
            "sha256": program["submission_launcher"]["sha256"],
        },
        "slurm_export": {"path": export_path, "sha256": export_sha256},
    }
    _validate_replay_submission_contract(
        document,
        authenticated_source=source,
        expected_attempt_id=attempt,
        source_snapshot=source_snapshot,
        replay_program=program,
        slurm_export_path=export_path,
        slurm_export_sha256=export_sha256,
    )
    return document


def _validate_replay_submission_contract(
    document: Mapping[str, Any],
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_attempt_id: str,
    source_snapshot: Mapping[str, Any],
    replay_program: Mapping[str, Any],
    slurm_export_path: str,
    slurm_export_sha256: str,
) -> None:
    source = _reload_authenticated_off_source_capture(authenticated_source)
    pair = source.pair_manifest
    _require_exact_keys(document, REPLAY_SUBMISSION_CONTRACT_KEYS, name="replay submission contract")
    attempt = _attempt(document["attempt_id"])
    if attempt != _attempt(expected_attempt_id):
        raise ValueError("replay submission contract attempt differs")
    if (
        document["schema"] != REPLAY_SUBMISSION_CONTRACT_SCHEMA
        or document["hash_domain"] != HASH_DOMAIN
        or document["pair_id"] != pair["pair_id"]
        or document["environment"] != pair["selection"]["environment"]
    ):
        raise ValueError("replay submission contract identity differs")
    _digest(document["submission_nonce"], name="replay submission nonce")
    if (
        type(document["submitter_euid"]) is not int
        or document["submitter_euid"] != pair["scheduler_submission"]["identity"]["submitter_euid"]
        or document["submitter_euid"] != os.geteuid()
    ):
        raise ValueError("replay submission contract submitter EUID differs")
    authenticated_sbatch = pair["runtime_tools"]["document"]["host"]["sbatch"]
    _verify_stable_system_executable(
        path=authenticated_sbatch["path"],
        expected_sha256=authenticated_sbatch["sha256"],
        name="replay sbatch program",
    )
    expected_program = _validate_program(replay_program)
    snapshot_root = source_snapshot["path"]
    expected = {
        "sbatch_program": authenticated_sbatch,
        "job_wrapper": {
            "path": f"{snapshot_root}/{expected_program['job_wrapper']['path']}",
            "sha256": expected_program["job_wrapper"]["sha256"],
        },
        "submission_launcher": {
            "path": f"{snapshot_root}/{expected_program['submission_launcher']['path']}",
            "sha256": expected_program["submission_launcher"]["sha256"],
        },
        "slurm_export": {
            "path": slurm_export_path,
            "sha256": slurm_export_sha256,
        },
    }
    for name, expected_value in expected.items():
        _file_ref(document[name], name=f"replay submission contract {name}")
        if not _exact(document[name], expected_value):
            raise ValueError(f"replay submission contract {name} differs")
    canonical_ascii_json(document)


def validate_replay_submission_contract(
    document: Mapping[str, Any],
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_attempt_id: str,
) -> None:
    """Validate a contract only against freshly loaded static authority."""
    source = _reload_authenticated_off_source_capture(authenticated_source)
    attempt = _attempt(expected_attempt_id)
    source_snapshot, program, _ = _load_authenticated_replay_snapshot(source=source)
    export_path, export_sha256, _ = _load_replay_slurm_export(source=source, attempt_id=attempt)
    _validate_replay_submission_contract(
        document,
        authenticated_source=source,
        expected_attempt_id=attempt,
        source_snapshot=source_snapshot,
        replay_program=program,
        slurm_export_path=export_path,
        slurm_export_sha256=export_sha256,
    )


def publish_replay_submission_contract(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    document: Mapping[str, Any],
) -> tuple[Path, str]:
    source = _reload_authenticated_off_source_capture(authenticated_source)
    validate_replay_submission_contract(
        document,
        authenticated_source=source,
        expected_attempt_id=attempt_id,
    )
    output = _submission_contract_path(source.pair_manifest, attempt_id=attempt_id)
    return publish_evidence_document(output=output, document=document, trailing_lf=True)


def _stable_container_asset_identity(
    pair: Mapping[str, Any],
    *,
    lstat: Any = os.lstat,
    access: Any = os.access,
    geteuid: Any = os.geteuid,
) -> dict[str, Any]:
    """Bind the one explicitly trusted foreign-owned replay container asset."""
    reference = pair["artifacts"]["container"]
    _file_ref(reference, name="Pair container artifact")
    path = _absolute_path(reference["path"], name="Pair container artifact path")
    try:
        before = lstat(path)
    except OSError as error:
        raise ValueError(f"cannot stat Pair container artifact: {error}") from error
    fingerprint = lambda metadata: (
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
    mode = stat.S_IMODE(before.st_mode)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Pair container artifact is not one regular single-link file")
    if before.st_uid != REPLAY_CONTAINER_OWNER_UID or before.st_gid != REPLAY_CONTAINER_OWNER_GID:
        raise ValueError("Pair container artifact named publisher identity differs")
    effective_uid = geteuid()
    if type(effective_uid) is not int or effective_uid in {
        0,
        REPLAY_CONTAINER_OWNER_UID,
    }:
        raise ValueError("replay submitter must be distinct from the container publisher")
    if mode & 0o022:
        raise ValueError("Pair container artifact is group/other writable")
    try:
        effectively_writable = access(path, os.W_OK, effective_ids=True)
    except (OSError, TypeError) as error:
        raise ValueError(f"cannot determine effective Pair container write access: {error}") from error
    if effectively_writable is not False:
        raise ValueError("Pair container artifact is writable by the replay submitter")
    try:
        after = lstat(path)
    except OSError as error:
        raise ValueError(f"cannot restat Pair container artifact: {error}") from error
    if fingerprint(before) != fingerprint(after):
        raise RuntimeError("Pair container artifact changed during identity validation")
    return {
        "path": path,
        "sha256": reference["sha256"],
        "owner_uid": REPLAY_CONTAINER_OWNER_UID,
        "owner_gid": REPLAY_CONTAINER_OWNER_GID,
    }


def load_authenticated_replay_static_inputs(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
) -> AuthenticatedReplayStaticInputs:
    """Stable-load every static artifact before a replay manifest is built."""
    source = _reload_authenticated_off_source_capture(authenticated_source)
    attempt = _attempt(attempt_id)
    container_asset = _stable_container_asset_identity(source.pair_manifest)
    source_snapshot, program, gym_source_root = _load_authenticated_replay_snapshot(source=source)
    export_path, export_sha256, export_values = _load_replay_slurm_export(source=source, attempt_id=attempt)
    contract_path = _submission_contract_path(source.pair_manifest, attempt_id=attempt)
    contract, contract_sha256 = _load_unanchored_evidence_document(
        path=contract_path,
        trailing_lf=True,
        name="replay submission contract",
    )
    _validate_replay_submission_contract(
        contract,
        authenticated_source=source,
        expected_attempt_id=attempt,
        source_snapshot=source_snapshot,
        replay_program=program,
        slurm_export_path=export_path,
        slurm_export_sha256=export_sha256,
    )
    return AuthenticatedReplayStaticInputs(
        attempt_id=attempt,
        container_asset=copy.deepcopy(container_asset),
        source_snapshot=copy.deepcopy(source_snapshot),
        gym_source_root=copy.deepcopy(gym_source_root),
        replay_program=copy.deepcopy(program),
        slurm_export_path=export_path,
        slurm_export_sha256=export_sha256,
        slurm_export_values=tuple(export_values),
        submission_contract_path=contract_path,
        submission_contract_sha256=contract_sha256,
        submission_contract=copy.deepcopy(contract),
    )


def build_replay_execution_manifest(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
) -> dict[str, Any]:
    """Build one immutable per-attempt manifest without future replay facts."""
    authenticated_source = _reload_authenticated_off_source_capture(authenticated_source)
    pair_manifest = authenticated_source.pair_manifest
    pair_manifest_sha256 = authenticated_source.pair_manifest_sha256
    pair_submission_receipt_sha256 = authenticated_source.pair_submission_receipt_sha256
    source_capture = authenticated_source.document
    _require_exact_keys(pair_manifest, _pair_manifest_required_keys(), name="Pair")
    if pair_manifest["schema"] != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unexpected Pair manifest schema")
    pair_id = _safe_id(pair_manifest["pair_id"], name="Pair pair_id", maximum=64)
    environment = pair_manifest["selection"]["environment"]
    if environment not in ENVIRONMENTS:
        raise ValueError("Pair environment is not admitted")
    attempt_id = _attempt(attempt_id)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=authenticated_source, attempt_id=attempt_id
    )
    replay_submission_contract_path = static_inputs.submission_contract_path
    replay_submission_contract_sha256 = static_inputs.submission_contract_sha256
    slurm_export_path = static_inputs.slurm_export_path
    slurm_export_sha256 = static_inputs.slurm_export_sha256
    submission_nonce = static_inputs.submission_contract["submission_nonce"]
    submitter_euid = static_inputs.submission_contract["submitter_euid"]
    results_root = _absolute_path(pair_manifest["paths"]["results_root"], name="results_root")
    pair_manifest_path = f"{results_root}/PAIR_MANIFEST.json"
    cache_root = _absolute_path(pair_manifest["paths"]["cache_root"], name="cache_root")
    hf_root = _absolute_path(pair_manifest["paths"]["hf_home"], name="hf_home")
    attempt_root = f"{results_root}/captured_replay/{attempt_id}"
    operational_root = f"{results_root}/captured_replay/operational/{pair_id}/{attempt_id}"
    submission_state_root = f"{results_root}/captured_replay/replay_submission_state/{pair_id}/{attempt_id}"
    persistent_cache = f"{cache_root}/captured_replay/{attempt_id}"
    hf_home = f"{hf_root}/captured_replay/{attempt_id}"
    on_snapshot = static_inputs.source_snapshot
    programs = static_inputs.replay_program
    replay_snapshot_path = on_snapshot["path"]
    _digest(pair_manifest_sha256, name="pair_manifest_sha256")
    actual_pair_sha256 = hashlib.sha256(canonical_ascii_json(pair_manifest) + b"\n").hexdigest()
    if pair_manifest_sha256 != actual_pair_sha256:
        raise ValueError("Pair manifest raw SHA-256 does not close to canonical Pair bytes")
    _digest(pair_submission_receipt_sha256, name="pair_submission_receipt_sha256")
    _digest(replay_submission_contract_sha256, name="replay submission contract SHA")
    _digest(slurm_export_sha256, name="Slurm export SHA")
    _digest(submission_nonce, name="submission_nonce")
    if type(submitter_euid) is not int or not 0 <= submitter_euid <= (1 << 31) - 1:
        raise ValueError("submitter_euid must be an exact nonnegative int31")

    pair_binding = {
        "id": pair_id,
        "environment": environment,
        "manifest": {
            "path": _absolute_path(pair_manifest_path, name="Pair manifest path"),
            "schema": PAIR_MANIFEST_SCHEMA,
            "sha256": pair_manifest_sha256,
        },
        "submission_receipt": {
            "path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
            "schema": PAIR_SUBMISSION_RECEIPT_SCHEMA,
            "sha256": pair_submission_receipt_sha256,
        },
        "acceptance_policy_sha256": pair_manifest["acceptance"]["policy_sha256"],
        "model_transport_policy_sha256": pair_manifest["model_transport"]["policy_sha256"],
        "pair_campaign_sha256": pair_manifest["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": pair_manifest["pair_campaign_reward_and_advantage_sha256"],
    }
    outputs = {
        "directory": {
            "path": attempt_root,
            "mode": "0700",
            "precondition": "absent-at-pre-runtime-creates-exclusively",
        },
        "evidence_index": {
            "path": f"{attempt_root}/evidence-index.json",
            "schema": REPLAY_EVIDENCE_INDEX_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "reasoning_score_call_index": {
            "path": (f"{attempt_root}/strict_gym_child_runtime/" "reasoning-score-call-index.json"),
            "schema": REASONING_SCORE_CALL_INDEX_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "transcript_bundle": {
            "path": f"{attempt_root}/transcript-bundle.json",
            "schema": TRANSCRIPT_BUNDLE_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "replay_ledger": {
            "path": f"{attempt_root}/replay-ledger.json",
            "schema": "nemo-rl-strict-captured-replay-step1-ledger-v5",
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "result_inventory": {
            "path": f"{attempt_root}/result-inventory-v1.json",
            "schema": REPLAY_RESULT_INVENTORY_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
            "self_excluded": True,
            "terminal_directory_mode": "0555",
        },
        "transport_consumption": {
            "path": f"{attempt_root}/model-transport-replay-consumption.json",
            "schema": REPLAY_TRANSPORT_CONSUMPTION_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
    }
    selected_config = pair_manifest["selection"]["config"]
    gym_resources = pair_manifest["selection"]["gym_resources"]
    contract = {
        "schema": REPLAY_CONTRACT_SCHEMA,
        "claim": "fresh_verifier_reward_replay",
        "execution_scope": "scorer-only",
        "cohort": {
            "fixture_row_index": 0,
            "logical_rollout_indices": [0, 1, 2, 3],
            "sample_count": 4,
            "step": 1,
        },
        "policy_execution": {
            "backward": False,
            "forward": False,
            "optimizer": False,
            "violation": "fail-closed",
        },
        "model_transport": {
            "direct_python_generation": "forbidden",
            "expected_count": 4,
            "mode": "replay",
            "policy_sha256": pair_manifest["model_transport"]["policy_sha256"],
            "source_arm": "off",
            "streaming": "forbidden",
            "terminal_status": "complete-terminal",
        },
        "program": programs,
        "selected_config": copy.deepcopy(selected_config),
        "source_generation": copy.deepcopy(pair_manifest["campaign"]["generation"]),
        "source_snapshot": {"arm": "on", "ref": copy.deepcopy(on_snapshot)},
        "gym_scorer": {
            "nonexecuted_derivation_source_agent": ENVIRONMENT_AGENTS[environment],
            "mode": "fresh-pinned-resource-scorer",
            "container": copy.deepcopy(static_inputs.container_asset),
            "launcher": _gym_scorer_launcher(environment),
            "resources": copy.deepcopy(gym_resources),
            "source_root": copy.deepcopy(static_inputs.gym_source_root),
            "runtime": {
                **_gym_scorer_runtime(environment),
            },
            "source": copy.deepcopy(pair_manifest["source"]["gym"]),
        },
    }
    document: dict[str, Any] = {
        "schema": REPLAY_EXECUTION_MANIFEST_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": attempt_id,
        "pair": pair_binding,
        "source_capture": copy.deepcopy(dict(source_capture)),
        "replay_contract": contract,
        "artifacts": {
            "container": copy.deepcopy(pair_manifest["artifacts"]["container"]),
            "fixture": copy.deepcopy(pair_manifest["artifacts"]["fixture"]),
            "model": copy.deepcopy(pair_manifest["artifacts"]["model"]),
            "sandbox_container": copy.deepcopy(pair_manifest["artifacts"]["sandbox_container"]),
            "outputs": outputs,
        },
        "execution_environment": {
            "schema": REPLAY_EXECUTION_ENVIRONMENT_SCHEMA,
            "arm_launcher": copy.deepcopy(pair_manifest["execution_environment"]["arm_launcher"]),
            "fixed": copy.deepcopy(pair_manifest["execution_environment"]["fixed"]),
            "attempt": {
                "base_log_dir": f"{operational_root}/ray_logs",
                "cache_read": {
                    "entry_count": 0,
                    "mode": "0700",
                    "path": f"{persistent_cache}/cache_read",
                    "policy": "empty-at-publication-and-job-entry-no-read",
                },
                "hf_datasets_cache": f"{hf_home}/hub",
                "hf_home": hf_home,
                "hf_hub_cache": f"{hf_home}/hub",
                "persistent_cache": persistent_cache,
                "operational": {
                    "boundary": "outside-sealed-result",
                    "root": operational_root,
                    "slurm": f"{operational_root}/slurm",
                    "ray_logs": f"{operational_root}/ray_logs",
                    "persistent_cache": persistent_cache,
                    "hf_home": hf_home,
                },
                "results_dir": attempt_root,
                "scheduler": {
                    "batch_working_directory": replay_snapshot_path,
                    "sbatch_chdir_argument": f"--chdir={replay_snapshot_path}",
                    "sbatch_client_cwd": replay_snapshot_path,
                    "slurm_submit_dir": replay_snapshot_path,
                },
                "setup_command": copy.deepcopy(pair_manifest["execution_environment"]["arms"]["on"]["setup_command"]),
            },
        },
        "wandb": copy.deepcopy(REPLAY_WANDB_POLICY),
        "scheduler_submission": {
            "schema": "nemo-rl-strict-scheduler-submission-v1",
            "accepted_id_record": {
                "accepted_format": "ascii-positive-decimal-lf",
                "capture_format": "opaque-sbatch-stdout",
                "initial_mode": "0600",
                "path": (f"{submission_state_root}/accepted.job-id"),
                "sealed_mode": "0400",
            },
            "contract": {
                "path": _absolute_path(
                    replay_submission_contract_path,
                    name="replay submission contract path",
                ),
                "sha256": replay_submission_contract_sha256,
            },
            "identity": {
                "comment_template": (
                    "nemo-rl-strict-captured-replay-v1:{attempt_id}:" "{submission_nonce}:{replay_manifest_sha256}"
                ),
                "job_name": f"strict-replay-{attempt_id}-{pair_id}",
                "submitter_euid": submitter_euid,
            },
            "nonce": submission_nonce,
            "receipt": {
                "path": (f"{submission_state_root}/submission-receipt.json"),
                "schema": REPLAY_SUBMISSION_RECEIPT_SCHEMA,
            },
        },
        "slurm_export_boundary": {
            "schema": REPLAY_SLURM_EXPORT_SCHEMA,
            "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
            "ambient_merge": False,
            "attempt_id": attempt_id,
            "format": "nul-separated-name-value",
            "get_user_env": False,
            "job_argv_template": [
                "--pair-manifest",
                "{pair_manifest_path}",
                "--pair-manifest-sha256",
                "{pair_manifest_sha256}",
                "--pair-submission-receipt",
                "{pair_submission_receipt_path}",
                "--pair-submission-receipt-sha256",
                "{pair_submission_receipt_sha256}",
                "--off-exit-receipt",
                "{trusted_off_exit_receipt_path}",
                "--off-exit-receipt-sha256",
                "{trusted_off_exit_receipt_sha256}",
                "--replay-manifest",
                "{replay_manifest_path}",
                "--replay-manifest-sha256",
                "{replay_manifest_sha256}",
            ],
            "path": _absolute_path(slurm_export_path, name="Slurm export path"),
            "sha256": slurm_export_sha256,
        },
        "deployment": copy.deepcopy(pair_manifest["deployment"]),
        "runtime_attestation_requirements": {
            "schema": REPLAY_RUNTIME_REQUIREMENTS_SCHEMA,
            "shared_prefix_determinism": {
                "applicable": False,
                "reason": "verifier-only-no-policy-forward-backward-or-optimizer",
                "status": "not_applicable",
            },
            "model_transport_replay": {
                "schema": REPLAY_TRANSPORT_CONSUMPTION_SCHEMA,
                "expected_count": 4,
                "required_status": "complete-terminal",
            },
            "derived_request_runtime": {
                **REQUIRED_RUNTIME_VERSIONS,
                "algorithm": "pinned-simple-agent-model-dump-v1",
                "forbidden_endpoints": ["/run"],
                "required_attestation": ("replay-driver-importlib-metadata-before-first-verifier-request"),
                "required": True,
            },
            "resource_scorer_child": _gym_scorer_runtime(environment),
            "verifier": {
                "required_mode": "fresh-pinned-resource-scorer",
                "required_request_evidence": ("derived-from-pinned-simple-agent-source-not-wire-captured"),
                "required_response_evidence": "fresh-pinned-gym-result",
            },
        },
        "runtime_tools": copy.deepcopy(pair_manifest["runtime_tools"]),
        "container_entry_boundary": copy.deepcopy(pair_manifest["container_entry_boundary"]),
        "source": copy.deepcopy(pair_manifest["source"]),
    }
    _validate_replay_execution_manifest_against_source(
        document, source=authenticated_source, static_inputs=static_inputs
    )
    return document


def _validate_replay_execution_manifest_shape(document: Mapping[str, Any], *, pair_manifest: Mapping[str, Any]) -> None:
    """Validate exact pre-submit shape and its non-training replay semantics."""
    _require_exact_keys(document, ROOT_KEYS, name="replay execution manifest")
    if document["schema"] != REPLAY_EXECUTION_MANIFEST_SCHEMA:
        raise ValueError("unexpected replay execution manifest schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected replay execution manifest hash domain")
    pair_id = _safe_id(document["pair_id"], name="pair_id", maximum=64)
    environment = document["environment"]
    if environment not in ENVIRONMENTS:
        raise ValueError("replay environment is not admitted")
    if document["arm"] != "on" or document["mode"] != "fresh_verifier_reward_replay":
        raise ValueError("replay must be on-arm fresh_verifier_reward_replay")
    attempt_id = _attempt(document["attempt_id"])
    _validate_pair_binding(document["pair"], pair_id=pair_id, environment=environment)
    _validate_source_capture(document["source_capture"])
    _validate_replay_contract(
        document["replay_contract"],
        environment=environment,
        model_transport_policy_sha256=document["pair"]["model_transport_policy_sha256"],
    )
    _require_exact_keys(document["artifacts"], ARTIFACTS_KEYS, name="artifacts")
    _validate_outputs(
        document["artifacts"]["outputs"],
        attempt_id=attempt_id,
        environment=environment,
        pair_id=pair_id,
    )
    _require_exact_keys(
        document["execution_environment"],
        EXECUTION_ENVIRONMENT_KEYS,
        name="execution_environment",
    )
    if document["execution_environment"]["schema"] != REPLAY_EXECUTION_ENVIRONMENT_SCHEMA:
        raise ValueError("unexpected replay execution-environment schema")
    _require_exact_keys(
        document["execution_environment"]["attempt"],
        ATTEMPT_ENVIRONMENT_KEYS,
        name="execution_environment.attempt",
    )
    if not _exact(document["wandb"], REPLAY_WANDB_POLICY):
        raise ValueError("replay W&B disabled policy differs")
    _validate_scheduler_submission(document["scheduler_submission"], pair_id=pair_id, attempt_id=attempt_id)
    _validate_slurm_export(document["slurm_export_boundary"], attempt_id=attempt_id)
    _validate_runtime_requirements(document["runtime_attestation_requirements"], environment=environment)
    canonical_ascii_json(document)
    _validate_against_pair(document, pair_manifest)


def validate_replay_execution_manifest(
    document: Mapping[str, Any], *, authenticated_source: AuthenticatedOffSourceCapture
) -> None:
    """Validate shape and freshly reload every historical/static authority."""
    source = _reload_authenticated_off_source_capture(authenticated_source)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
    )
    _validate_replay_execution_manifest_against_source(document, source=source, static_inputs=static_inputs)


def _validate_manifest_static_inputs(
    document: Mapping[str, Any], *, static_inputs: AuthenticatedReplayStaticInputs
) -> None:
    """Join a manifest to freshly stable-loaded pre-submit artifacts."""
    if document["attempt_id"] != static_inputs.attempt_id:
        raise ValueError("replay manifest attempt differs from static inputs")
    if not _exact(
        document["replay_contract"]["gym_scorer"]["container"],
        static_inputs.container_asset,
    ):
        raise ValueError("replay manifest container identity differs from authenticated host asset")
    expected_snapshot = {"arm": "on", "ref": static_inputs.source_snapshot}
    if not _exact(document["replay_contract"]["source_snapshot"], expected_snapshot):
        raise ValueError("replay manifest source snapshot differs from authenticated bytes")
    if not _exact(document["replay_contract"]["program"], static_inputs.replay_program):
        raise ValueError("replay manifest program differs from authenticated ON snapshot")
    if not _exact(
        document["replay_contract"]["gym_scorer"]["source_root"],
        static_inputs.gym_source_root,
    ):
        raise ValueError("replay manifest Gym source root differs from authenticated ON snapshot")
    submission = document["scheduler_submission"]
    expected_submission = {
        "contract": {
            "path": static_inputs.submission_contract_path,
            "sha256": static_inputs.submission_contract_sha256,
        },
        "nonce": static_inputs.submission_contract["submission_nonce"],
        "submitter_euid": static_inputs.submission_contract["submitter_euid"],
    }
    if not _exact(submission["contract"], expected_submission["contract"]):
        raise ValueError("replay manifest submission contract differs from sealed bytes")
    if submission["nonce"] != expected_submission["nonce"]:
        raise ValueError("replay manifest submission nonce differs from sealed contract")
    if submission["identity"]["submitter_euid"] != expected_submission["submitter_euid"]:
        raise ValueError("replay manifest submitter EUID differs from sealed contract")
    export = document["slurm_export_boundary"]
    if export["path"] != static_inputs.slurm_export_path or export["sha256"] != static_inputs.slurm_export_sha256:
        raise ValueError("replay manifest Slurm export differs from sealed bytes")


def _validate_replay_execution_manifest_against_source(
    document: Mapping[str, Any],
    *,
    source: AuthenticatedOffSourceCapture,
    static_inputs: AuthenticatedReplayStaticInputs,
) -> None:
    _validate_replay_execution_manifest_shape(document, pair_manifest=source.pair_manifest)
    _validate_manifest_static_inputs(document, static_inputs=static_inputs)
    if not _exact(document["source_capture"], source.source_capture):
        raise ValueError("replay source capture differs from authenticated OFF bytes")
    if document["pair"]["manifest"]["sha256"] != source.pair_manifest_sha256:
        raise ValueError("replay Pair manifest anchor differs from authenticated bytes")
    if document["pair"]["submission_receipt"]["sha256"] != source.pair_submission_receipt_sha256:
        raise ValueError("replay Pair receipt anchor differs from authenticated bytes")


def publish_replay_execution_manifest(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
) -> tuple[Path, str]:
    source = _reload_authenticated_off_source_capture(authenticated_source)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
    )
    _validate_replay_execution_manifest_against_source(document, source=source, static_inputs=static_inputs)
    return publish_evidence_document(output=output, document=document, trailing_lf=False)


def load_replay_execution_manifest(
    *,
    path: str | Path,
    expected_sha256: str,
    authenticated_source: AuthenticatedOffSourceCapture,
) -> tuple[dict[str, Any], str]:
    source = _reload_authenticated_off_source_capture(authenticated_source)
    document, digest = load_evidence_document(path=path, expected_sha256=expected_sha256, trailing_lf=False)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
    )
    _validate_replay_execution_manifest_against_source(document, source=source, static_inputs=static_inputs)
    return document, digest


def _gym_scorer_launcher(environment: str) -> dict[str, Any]:
    if environment not in ENVIRONMENTS:
        raise ValueError("Gym scorer environment is not admitted")
    directory = GYM_SCORER_DIRECTORY[environment]
    return {
        "scope": "selected-resource-scorer-only-not-full-main-run-helper-process-set",
        "mechanism": "nemo-gym-run-helper-shell-subprocess",
        "executable": "/bin/bash",
        "argv_shape": ["-c", "<runtime-generated-command>"],
        "core_command_template": (
            "cd {working-directory} && source {venv-directory}/bin/activate && "
            "NEMO_GYM_CONFIG_DICT={shlex-quoted-resolved-yaml} "
            "NEMO_GYM_CONFIG_PATH={shlex-quoted-config-path-name} python app.py"
        ),
        "log_wrapper": "forbidden",
        "forbidden_resolved_config_keys": ["nemo_gym_log_dir"],
        "generator": copy.deepcopy(GYM_RUN_HELPER_SOURCE),
        "setup": copy.deepcopy(GYM_SETUP_COMMAND_SOURCE),
        "allocator": copy.deepcopy(GYM_PORT_ALLOCATOR_SOURCE),
        "server": copy.deepcopy(GYM_SERVER_SOURCE),
        "port_policy": {
            "first": 5000,
            "last": 5999,
            "last_inclusive": True,
            "head_port_excluded": True,
            "selection": "python-random-randint-inclusive",
        },
        "host_policy": "loopback-127.0.0.1",
        "working_directory": (f"/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/{directory}"),
        "venv_directory": f"/opt/gym_venvs/resources_servers/{directory}/.venv",
        "entrypoint": "app.py",
        "config_path_name": (
            "resources_only" if environment == "reasoning_gym" else GYM_SCORER_CONFIG_PATH_NAME[environment]
        ),
        "resource_only_config": (
            copy.deepcopy(GYM_REASONING_RESOURCE_ONLY_CONFIG) if environment == "reasoning_gym" else None
        ),
        "resolved_evidence_requirement": (
            "authenticated-replay-pre-and-exit-command-workdir-interpreter-host-port-process"
        ),
    }


def _gym_scorer_runtime(environment: str) -> dict[str, Any]:
    if environment not in ENVIRONMENTS:
        raise ValueError("Gym scorer environment is not admitted")
    if environment != "reasoning_gym":
        raise ValueError("only reasoning_gym has authenticated selected-child call evidence")
    scorer_pin = {
        "distribution": "reasoning-gym",
        "required_distribution_version": "0.1.25",
        "module": "reasoning_gym.logic.knights_knaves",
        "module_internal_version_literal": "0.1.19",
        "module_relative_path": "reasoning_gym/logic/knights_knaves.py",
        "module_sha256": ("8837a3c6dfc72bb40db168b82ad6b3da45a08a4000a006fc306368b77b622705"),
        "score_function": "KnightsKnavesDataset.score_answer",
    }
    return {
        "required_common_distributions": {
            "nemo-gym": "0.5.0rc0",
            "openai": "2.6.1",
            "pydantic": "2.13.4",
            "pydantic-core": "2.46.4",
            "ray": "2.56.1",
        },
        "required_module_versions": {
            "nemo_gym": "0.5.1",
            "ray": "2.56.1",
        },
        "required_attestation": ("selected-resource-child-interpreter-process-and-port-before-first-request"),
        "required_per_call_success_evidence": {
            "call_schema": REASONING_SCORE_CALL_SCHEMA,
            "expected_count": 4,
            "required_outcome_kind": "returned",
            "terminal_index_schema": REASONING_SCORE_CALL_INDEX_SCHEMA,
        },
        "required_python_version": "3.13.14",
        "required": True,
        "selected_resource_app": copy.deepcopy(GYM_RESOURCE_APP[environment]),
        "scorer_pin": scorer_pin,
    }


def _validate_pair_binding(value: Any, *, pair_id: str, environment: str) -> None:
    _require_exact_keys(value, PAIR_KEYS, name="pair")
    if value["id"] != pair_id or value["environment"] != environment:
        raise ValueError("nested Pair identity differs from manifest identity")
    _artifact_ref(value["manifest"], schema=PAIR_MANIFEST_SCHEMA, name="pair.manifest")
    _artifact_ref(
        value["submission_receipt"],
        schema=PAIR_SUBMISSION_RECEIPT_SCHEMA,
        name="pair.submission_receipt",
    )
    for name in (
        "acceptance_policy_sha256",
        "model_transport_policy_sha256",
        "pair_campaign_sha256",
        "pair_campaign_reward_and_advantage_sha256",
    ):
        _digest(value[name], name=f"pair.{name}")


def _validate_source_capture(value: Any) -> None:
    _require_exact_keys(value, SOURCE_CAPTURE_KEYS, name="source_capture")
    if value["arm"] != "off" or value["restart_count"] != 0 or type(value["restart_count"]) is not int:
        raise ValueError("source capture must be authenticated OFF restart zero")
    _require_exact_keys(
        value["authenticated_job"],
        AUTHENTICATED_JOB_KEYS,
        name="source_capture.authenticated_job",
    )
    _job_id(value["authenticated_job"]["job_id"], name="source OFF job_id")
    _ascii(
        value["authenticated_job"]["comment"],
        name="source job comment",
        maximum=512,
    )
    _safe_id(
        value["authenticated_job"]["job_name"],
        name="source job job_name",
        maximum=256,
    )
    user_id = value["authenticated_job"]["user_id"]
    if type(user_id) is not str or re.fullmatch(r"0|[1-9][0-9]*", user_id) is None or int(user_id) > (1 << 31) - 1:
        raise ValueError("source job user_id must be a canonical nonnegative int31 string")
    _require_exact_keys(value["job_receipts"], JOB_RECEIPTS_KEYS, name="source_capture.job_receipts")
    for phase in ("pre", "exit"):
        _artifact_ref(
            value["job_receipts"][phase],
            schema=PAIR_JOB_RECEIPT_SCHEMA,
            name=f"source {phase} receipt",
        )
    evidence = value["step1_evidence"]
    _require_exact_keys(evidence, STEP1_EVIDENCE_KEYS, name="source step1_evidence")
    if evidence["schema"] != STEP1_EVIDENCE_INDEX_SCHEMA:
        raise ValueError("unexpected source step1 evidence schema")
    _artifact_ref(evidence["main_ledger"], schema=MAIN_LEDGER_SCHEMA, name="source main ledger")
    _artifact_ref(
        evidence["transcript_bundle"],
        schema=TRANSCRIPT_BUNDLE_SCHEMA,
        name="source transcript",
    )
    transport = evidence["model_transport"]
    _require_exact_keys(transport, TRANSPORT_EVIDENCE_KEYS, name="source transport evidence")
    if transport["schema"] != TRANSPORT_EVIDENCE_INDEX_SCHEMA:
        raise ValueError("unexpected source transport evidence schema")
    _artifact_ref(
        transport["bundle"],
        schema=TRANSPORT_BUNDLE_SCHEMA,
        name="source transport bundle",
    )
    _artifact_ref(
        transport["manifest"],
        schema=TRANSPORT_MANIFEST_SCHEMA,
        name="source transport manifest",
    )
    _require_exact_keys(transport["raw_log"], RAW_LOG_KEYS, name="source raw log")
    if (
        transport["raw_log"]["record_schema"] != TRANSPORT_CALL_SCHEMA
        or transport["raw_log"]["record_count"] != 4
        or type(transport["raw_log"]["record_count"]) is not int
    ):
        raise ValueError("source transport raw log must contain exact K4 call records")
    _absolute_path(transport["raw_log"]["path"], name="source raw log path")
    _digest(transport["raw_log"]["sha256"], name="source raw log SHA")
    _digest(transport["ordered_entries_sha256"], name="source ordered entries SHA")


def _validate_replay_contract(value: Any, *, environment: str, model_transport_policy_sha256: str) -> None:
    _require_exact_keys(value, REPLAY_CONTRACT_KEYS, name="replay_contract")
    if (
        value["schema"] != REPLAY_CONTRACT_SCHEMA
        or value["claim"] != "fresh_verifier_reward_replay"
        or value["execution_scope"] != "scorer-only"
    ):
        raise ValueError("replay contract claim/scope differs")
    if not _exact(
        value["cohort"],
        {
            "fixture_row_index": 0,
            "logical_rollout_indices": [0, 1, 2, 3],
            "sample_count": 4,
            "step": 1,
        },
    ):
        raise ValueError("replay contract cohort differs from K4 step1")
    if not _exact(
        value["policy_execution"],
        {
            "backward": False,
            "forward": False,
            "optimizer": False,
            "violation": "fail-closed",
        },
    ):
        raise ValueError("replay contract must forbid all policy execution")
    transport = value["model_transport"]
    expected_transport = {
        "direct_python_generation": "forbidden",
        "expected_count": 4,
        "mode": "replay",
        "policy_sha256": model_transport_policy_sha256,
        "source_arm": "off",
        "streaming": "forbidden",
        "terminal_status": "complete-terminal",
    }
    _digest(model_transport_policy_sha256, name="replay transport policy SHA")
    if not _exact(transport, expected_transport):
        raise ValueError("replay model-transport contract differs")
    _validate_program(value["program"])
    scorer = value["gym_scorer"]
    _require_exact_keys(
        scorer,
        frozenset(
            {
                "mode",
                "container",
                "launcher",
                "nonexecuted_derivation_source_agent",
                "resources",
                "runtime",
                "source",
                "source_root",
            }
        ),
        name="gym_scorer",
    )
    if (
        scorer["nonexecuted_derivation_source_agent"] != ENVIRONMENT_AGENTS[environment]
        or scorer["mode"] != "fresh-pinned-resource-scorer"
    ):
        raise ValueError("resource scorer provenance/mode differs")
    runtime = scorer["runtime"]
    expected_runtime = _gym_scorer_runtime(environment)
    if not _exact(runtime, expected_runtime):
        raise ValueError("Gym scorer runtime versions/attestation differ")
    container = scorer["container"]
    _require_exact_keys(
        container,
        GYM_SCORER_CONTAINER_KEYS,
        name="gym_scorer.container",
    )
    _absolute_path(container["path"], name="gym scorer container path")
    _digest(container["sha256"], name="gym scorer container SHA")
    if (
        type(container["owner_uid"]) is not int
        or container["owner_uid"] != REPLAY_CONTAINER_OWNER_UID
        or type(container["owner_gid"]) is not int
        or container["owner_gid"] != REPLAY_CONTAINER_OWNER_GID
    ):
        raise ValueError("Gym scorer container named publisher identity differs")
    if not _exact(scorer["launcher"], _gym_scorer_launcher(environment)):
        raise ValueError("Gym scorer static launcher policy differs")
    _require_exact_keys(
        scorer["source_root"],
        frozenset({"snapshot_relative_path", "host_path", "container_path"}),
        name="gym_scorer.source_root",
    )
    if scorer["source_root"]["snapshot_relative_path"] != GYM_SNAPSHOT_RELATIVE_ROOT:
        raise ValueError("Gym scorer snapshot-relative source root differs")
    _absolute_path(scorer["source_root"]["host_path"], name="Gym host source root")
    if scorer["source_root"]["container_path"] != GYM_CONTAINER_ROOT:
        raise ValueError("Gym container source root differs")


def _validate_outputs(value: Any, *, attempt_id: str, environment: str, pair_id: str) -> None:
    _require_exact_keys(value, OUTPUT_KEYS, name="artifacts.outputs")
    if environment != "reasoning_gym":
        raise ValueError("only reasoning_gym has authenticated selected-child call evidence")
    directory = value["directory"]
    _require_exact_keys(
        directory,
        frozenset({"path", "mode", "precondition"}),
        name="replay output directory",
    )
    root = _absolute_path(directory["path"], name="replay output directory path")
    suffix = f"/captured_replay/{attempt_id}"
    if (
        not root.endswith(suffix)
        or directory["mode"] != "0700"
        or directory["precondition"] != "absent-at-pre-runtime-creates-exclusively"
    ):
        raise ValueError("replay output directory contract differs")
    evidence_index = value["evidence_index"]
    _require_exact_keys(
        evidence_index,
        frozenset({"path", "schema", "framing", "mode"}),
        name="replay evidence index declaration",
    )
    if evidence_index != {
        "path": f"{root}/evidence-index.json",
        "schema": REPLAY_EVIDENCE_INDEX_SCHEMA,
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
    }:
        raise ValueError("replay evidence index contract differs")
    result_inventory = value["result_inventory"]
    _require_exact_keys(
        result_inventory,
        frozenset(
            {
                "path",
                "schema",
                "framing",
                "mode",
                "self_excluded",
                "terminal_directory_mode",
            }
        ),
        name="replay result inventory declaration",
    )
    if result_inventory != {
        "path": f"{root}/result-inventory-v1.json",
        "schema": REPLAY_RESULT_INVENTORY_SCHEMA,
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
        "self_excluded": True,
        "terminal_directory_mode": "0555",
    }:
        raise ValueError("replay result inventory contract differs")
    expected = {
        "reasoning_score_call_index": (
            "strict_gym_child_runtime/reasoning-score-call-index.json",
            REASONING_SCORE_CALL_INDEX_SCHEMA,
        ),
        "transcript_bundle": ("transcript-bundle.json", TRANSCRIPT_BUNDLE_SCHEMA),
        "replay_ledger": (
            "replay-ledger.json",
            "nemo-rl-strict-captured-replay-step1-ledger-v5",
        ),
        "transport_consumption": (
            "model-transport-replay-consumption.json",
            REPLAY_TRANSPORT_CONSUMPTION_SCHEMA,
        ),
    }
    for name, (filename, schema) in expected.items():
        item = value[name]
        _require_exact_keys(
            item,
            frozenset({"path", "schema", "framing", "mode"}),
            name=f"replay output {name}",
        )
        if item != {
            "path": f"{root}/{filename}",
            "schema": schema,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        }:
            raise ValueError(f"replay output {name} contract differs")


def _validate_scheduler_submission(value: Any, *, pair_id: str, attempt_id: str) -> None:
    _require_exact_keys(value, SCHEDULER_SUBMISSION_KEYS, name="scheduler_submission")
    if value["schema"] != "nemo-rl-strict-scheduler-submission-v1":
        raise ValueError("unexpected replay scheduler submission schema")
    _require_exact_keys(
        value["accepted_id_record"],
        frozenset({"accepted_format", "capture_format", "initial_mode", "path", "sealed_mode"}),
        name="accepted_id_record",
    )
    if (
        value["accepted_id_record"]["accepted_format"] != "ascii-positive-decimal-lf"
        or value["accepted_id_record"]["capture_format"] != "opaque-sbatch-stdout"
        or value["accepted_id_record"]["initial_mode"] != "0600"
        or value["accepted_id_record"]["sealed_mode"] != "0400"
    ):
        raise ValueError("accepted replay job-ID record contract differs")
    _absolute_path(value["accepted_id_record"]["path"], name="accepted ID path")
    _file_ref(value["contract"], name="replay submission contract")
    _require_exact_keys(
        value["identity"],
        frozenset({"comment_template", "job_name", "submitter_euid"}),
        name="replay scheduler identity",
    )
    if (
        value["identity"]["comment_template"]
        != "nemo-rl-strict-captured-replay-v1:{attempt_id}:{submission_nonce}:{replay_manifest_sha256}"
        or value["identity"]["job_name"] != f"strict-replay-{attempt_id}-{pair_id}"
    ):
        raise ValueError("replay scheduler identity contract differs")
    if (
        type(value["identity"]["submitter_euid"]) is not int
        or not 0 <= value["identity"]["submitter_euid"] <= (1 << 31) - 1
    ):
        raise ValueError("replay submitter EUID is invalid")
    _digest(value["nonce"], name="replay submission nonce")
    _require_exact_keys(
        value["receipt"],
        frozenset({"path", "schema"}),
        name="replay submission receipt declaration",
    )
    if value["receipt"]["schema"] != REPLAY_SUBMISSION_RECEIPT_SCHEMA:
        raise ValueError("unexpected replay submission receipt declaration")
    _absolute_path(value["receipt"]["path"], name="replay submission receipt path")


def _validate_slurm_export(value: Any, *, attempt_id: str) -> None:
    _require_exact_keys(value, SLURM_EXPORT_KEYS, name="slurm_export_boundary")
    if value["schema"] != REPLAY_SLURM_EXPORT_SCHEMA or value["attempt_id"] != attempt_id:
        raise ValueError("replay Slurm export schema/attempt differs")
    expected = {
        "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "format": "nul-separated-name-value",
        "get_user_env": False,
        "job_argv_template": [
            "--pair-manifest",
            "{pair_manifest_path}",
            "--pair-manifest-sha256",
            "{pair_manifest_sha256}",
            "--pair-submission-receipt",
            "{pair_submission_receipt_path}",
            "--pair-submission-receipt-sha256",
            "{pair_submission_receipt_sha256}",
            "--off-exit-receipt",
            "{trusted_off_exit_receipt_path}",
            "--off-exit-receipt-sha256",
            "{trusted_off_exit_receipt_sha256}",
            "--replay-manifest",
            "{replay_manifest_path}",
            "--replay-manifest-sha256",
            "{replay_manifest_sha256}",
        ],
    }
    for name, expected_value in expected.items():
        if not _exact(value[name], expected_value):
            raise ValueError(f"replay Slurm export {name} differs")
    _absolute_path(value["path"], name="replay Slurm export path")
    _digest(value["sha256"], name="replay Slurm export SHA")


def _validate_runtime_requirements(value: Any, *, environment: str) -> None:
    _require_exact_keys(value, RUNTIME_REQUIREMENTS_KEYS, name="runtime_attestation_requirements")
    expected = {
        "schema": REPLAY_RUNTIME_REQUIREMENTS_SCHEMA,
        "shared_prefix_determinism": {
            "applicable": False,
            "reason": "verifier-only-no-policy-forward-backward-or-optimizer",
            "status": "not_applicable",
        },
        "model_transport_replay": {
            "schema": REPLAY_TRANSPORT_CONSUMPTION_SCHEMA,
            "expected_count": 4,
            "required_status": "complete-terminal",
        },
        "derived_request_runtime": {
            **REQUIRED_RUNTIME_VERSIONS,
            "algorithm": "pinned-simple-agent-model-dump-v1",
            "forbidden_endpoints": ["/run"],
            "required_attestation": ("replay-driver-importlib-metadata-before-first-verifier-request"),
            "required": True,
        },
        "resource_scorer_child": _gym_scorer_runtime(environment),
        "verifier": {
            "required_mode": "fresh-pinned-resource-scorer",
            "required_request_evidence": ("derived-from-pinned-simple-agent-source-not-wire-captured"),
            "required_response_evidence": "fresh-pinned-gym-result",
        },
    }
    if not _exact(value, expected):
        raise ValueError("replay runtime requirements contract differs")


def _validate_program(value: Any) -> dict[str, dict[str, str]]:
    _require_exact_keys(value, PROGRAM_KEYS, name="replay program")
    result = copy.deepcopy(dict(value))
    for name, expected_path in REPLAY_PROGRAM_PATHS.items():
        _file_ref(result[name], name=f"replay program {name}", absolute=False)
        if result[name]["path"] != expected_path:
            raise ValueError(f"replay program {name} path differs")
    return result


def _validate_program_v2(value: Any) -> dict[str, dict[str, str]]:
    """Validate the profiled family without widening the legacy program set."""
    _require_exact_keys(value, PROGRAM_V2_KEYS, name="profiled replay program")
    result = copy.deepcopy(dict(value))
    for name, expected_path in REPLAY_PROGRAM_V2_PATHS.items():
        _file_ref(result[name], name=f"profiled replay program {name}", absolute=False)
        if result[name]["path"] != expected_path:
            raise ValueError(f"profiled replay program {name} path differs")
    return result


def _load_authenticated_replay_static_inputs_v2(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    profile: StrictCapturedReplayProfile,
) -> AuthenticatedReplayStaticInputs:
    """Add every authenticated V2-only/transitive program for manifest V4."""
    source = _reload_authenticated_off_source_capture(authenticated_source)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=attempt_id,
    )
    snapshot = static_inputs.source_snapshot
    snapshot_root = _absolute_path(snapshot["path"], name="Pair ON snapshot path")
    manifest_raw, manifest_sha256 = _load_stable_evidence_bytes(
        path=f"{snapshot_root}/{_SNAPSHOT_SHA_MANIFEST}",
        expected_sha256=snapshot["manifest_sha256"],
        name="Pair ON snapshot SHA manifest for profiled replay",
        maximum=64 * 1024 * 1024,
        required_mode=0o400,
        allowed_parent_modes=None,
    )
    if manifest_sha256 != snapshot["manifest_sha256"]:
        raise AssertionError("unreachable profiled replay snapshot manifest mismatch")
    snapshot_manifest = _parse_snapshot_sha_manifest(
        manifest_raw,
        name="Pair ON snapshot SHA manifest for profiled replay",
    )
    fixture_sha256 = snapshot_manifest.get(profile.fixture_path)
    if fixture_sha256 != profile.fixture_sha256:
        raise ValueError("Pair ON snapshot fixture differs from the closed scorer profile")
    _, loaded_fixture_sha256 = _load_stable_evidence_bytes(
        path=f"{snapshot_root}/{profile.fixture_path}",
        expected_sha256=profile.fixture_sha256,
        name="Pair ON snapshot fixture for profiled replay",
        maximum=64 * 1024 * 1024,
        required_mode=None,
        allowed_parent_modes=None,
        allow_empty=False,
    )
    if loaded_fixture_sha256 != profile.fixture_sha256:
        raise AssertionError("unreachable replay fixture digest mismatch")
    program = copy.deepcopy(static_inputs.replay_program)
    for name, relative in REPLAY_PROGRAM_V2_PATHS.items():
        if name in program:
            continue
        expected_sha256 = snapshot_manifest.get(relative)
        if expected_sha256 is None:
            raise ValueError(f"Pair ON snapshot omits replay program {name}")
        _, loaded_sha256 = _load_stable_evidence_bytes(
            path=f"{snapshot_root}/{relative}",
            expected_sha256=expected_sha256,
            name=f"Pair ON snapshot replay program {name}",
            maximum=16 * 1024 * 1024,
            required_mode=None,
            allowed_parent_modes=None,
            allow_empty=False,
        )
        if loaded_sha256 != expected_sha256:
            raise AssertionError(f"unreachable replay program {name} digest mismatch")
        program[name] = {
            "path": relative,
            "sha256": expected_sha256,
        }
    _validate_program_v2(program)
    return replace(static_inputs, replay_program=program)


def _validate_against_pair(document: Mapping[str, Any], pair: Mapping[str, Any]) -> None:
    _require_exact_keys(pair, _pair_manifest_required_keys(), name="Pair")
    if (
        pair["schema"] != PAIR_MANIFEST_SCHEMA
        or pair["pair_id"] != document["pair_id"]
        or pair["selection"]["environment"] != document["environment"]
    ):
        raise ValueError("replay manifest identity differs from authenticated Pair")
    pair_boundary = pair["slurm_export_boundary"]
    if pair_boundary["schema"] != PAIR_SLURM_EXPORT_SCHEMA:
        raise ValueError("source Pair Slurm export schema differs from authoritative Pair79")
    if not _exact(pair_boundary["allowed_names"], list(SLURM_EXPORT_ALLOWED_NAMES)):
        raise ValueError("source Pair Slurm export names differ from exact Pair79")
    pair_sha256 = hashlib.sha256(canonical_ascii_json(pair) + b"\n").hexdigest()
    if document["pair"]["manifest"]["sha256"] != pair_sha256:
        raise ValueError("replay Pair anchor does not hash authenticated Pair bytes")
    results_root = pair["paths"]["results_root"]
    cache_root = pair["paths"]["cache_root"]
    hf_root = pair["paths"]["hf_home"]
    attempt_id = document["attempt_id"]
    attempt_root = f"{results_root}/captured_replay/{attempt_id}"
    if document["pair"]["manifest"]["path"] != f"{results_root}/PAIR_MANIFEST.json":
        raise ValueError("replay Pair manifest path differs from Pair results root")
    if document["pair"]["submission_receipt"]["path"] != f"{results_root}/PAIR_SUBMISSION_RECEIPT.json":
        raise ValueError("replay Pair receipt path differs from Pair results root")
    expected_pair_scalars = {
        "acceptance_policy_sha256": pair["acceptance"]["policy_sha256"],
        "model_transport_policy_sha256": pair["model_transport"]["policy_sha256"],
        "pair_campaign_sha256": pair["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
    }
    for name, expected in expected_pair_scalars.items():
        if document["pair"][name] != expected:
            raise ValueError(f"replay manifest Pair {name} differs")
    for name in ("container", "fixture", "model", "sandbox_container"):
        if not _exact(document["artifacts"][name], pair["artifacts"][name]):
            raise ValueError(f"replay artifact {name} differs from Pair")
    for name in ("deployment", "runtime_tools", "container_entry_boundary", "source"):
        if not _exact(document[name], pair[name]):
            raise ValueError(f"replay {name} differs from Pair")
    if not _exact(
        document["execution_environment"]["arm_launcher"],
        pair["execution_environment"]["arm_launcher"],
    ) or not _exact(
        document["execution_environment"]["fixed"],
        pair["execution_environment"]["fixed"],
    ):
        raise ValueError("replay execution common boundary differs from Pair")
    contract = document["replay_contract"]
    if not _exact(contract["selected_config"], pair["selection"]["config"]):
        raise ValueError("replay selected config differs from Pair")
    if not _exact(contract["source_generation"], pair["campaign"]["generation"]):
        raise ValueError("replay source generation controls differ from Pair")
    source_snapshot = contract["source_snapshot"]
    _require_exact_keys(
        source_snapshot,
        frozenset({"arm", "ref"}),
        name="replay source snapshot",
    )
    if source_snapshot["arm"] != "on" or not _exact(source_snapshot["ref"], pair["source"]["snapshots"]["on"]):
        raise ValueError("replay source snapshot differs from authenticated Pair ON")
    scorer = contract["gym_scorer"]
    if not _exact(scorer["resources"], pair["selection"]["gym_resources"]):
        raise ValueError("replay Gym resources differ from Pair")
    if not _exact(scorer["source"], pair["source"]["gym"]):
        raise ValueError("replay Gym source differs from Pair")
    expected_container = {
        **pair["artifacts"]["container"],
        "owner_uid": REPLAY_CONTAINER_OWNER_UID,
        "owner_gid": REPLAY_CONTAINER_OWNER_GID,
    }
    if not _exact(scorer["container"], expected_container):
        raise ValueError("replay Gym container differs from Pair")
    expected_gym_source_root = {
        "snapshot_relative_path": GYM_SNAPSHOT_RELATIVE_ROOT,
        "host_path": (f"{source_snapshot['ref']['path']}/{GYM_SNAPSHOT_RELATIVE_ROOT}"),
        "container_path": GYM_CONTAINER_ROOT,
    }
    if not _exact(scorer["source_root"], expected_gym_source_root):
        raise ValueError("replay Gym source root differs from Pair ON snapshot")
    attempt = document["execution_environment"]["attempt"]
    on_environment = pair["execution_environment"]["arms"]["on"]
    if not _exact(attempt["setup_command"], on_environment["setup_command"]):
        raise ValueError("replay setup command differs from Pair ON arm")
    snapshot_path = source_snapshot["ref"]["path"]
    expected_scheduler = {
        "batch_working_directory": snapshot_path,
        "sbatch_chdir_argument": f"--chdir={snapshot_path}",
        "sbatch_client_cwd": snapshot_path,
        "slurm_submit_dir": snapshot_path,
    }
    if not _exact(attempt["scheduler"], expected_scheduler):
        raise ValueError("replay scheduler paths differ from replay source snapshot")
    pair_export_names = pair_boundary["allowed_names"]
    if not _exact(pair_export_names, list(SLURM_EXPORT_ALLOWED_NAMES)) or not _exact(
        document["slurm_export_boundary"]["allowed_names"], pair_export_names
    ):
        raise ValueError("replay Slurm export names differ from exact Pair79")
    expected_off_identity = {
        "comment": (
            f"nemo-rl-strict-pair-v1:off:{pair['scheduler_submission']['nonce']}:"
            f"{document['pair']['manifest']['sha256']}"
        ),
        "job_id": document["source_capture"]["authenticated_job"]["job_id"],
        "job_name": f"off-{document['environment']}-{document['pair_id']}",
        "user_id": str(pair["scheduler_submission"]["identity"]["submitter_euid"]),
    }
    if not _exact(document["source_capture"]["authenticated_job"], expected_off_identity):
        raise ValueError("source OFF scheduler identity differs from Pair")
    source = document["source_capture"]
    source_job_id = source["authenticated_job"]["job_id"]
    receipt_root = f"{results_root}/off/strict_pair_job_state/{source_job_id}-0/receipts"
    if source["job_receipts"]["pre"]["path"] != f"{receipt_root}/PRE.json":
        raise ValueError("source OFF PRE receipt path differs")
    if source["job_receipts"]["exit"]["path"] != f"{receipt_root}/EXIT.json":
        raise ValueError("source OFF EXIT receipt path differs")
    source_step1 = source["step1_evidence"]
    expected_source_paths = {
        "main_ledger": f"{results_root}/off/strict_pair_step1_evidence/main-ledger.json",
        "transcript_bundle": (f"{results_root}/off/strict_pair_step1_evidence/transcript-bundle.json"),
    }
    for name, expected_path in expected_source_paths.items():
        if source_step1[name]["path"] != expected_path:
            raise ValueError(f"source OFF {name} path differs")
    source_transport = source_step1["model_transport"]
    expected_transport_paths = {
        "bundle": (f"{results_root}/off/strict_model_transport/model-transport-bundle.json"),
        "manifest": (f"{results_root}/off/strict_model_transport/model-transport-manifest.json"),
    }
    for name, expected_path in expected_transport_paths.items():
        if source_transport[name]["path"] != expected_path:
            raise ValueError(f"source OFF transport {name} path differs")
    if source_transport["raw_log"]["path"] != f"{results_root}/off/strict_model_transport/model-transport.jsonl":
        raise ValueError("source OFF transport raw-log path differs")
    outputs = document["artifacts"]["outputs"]
    if outputs["directory"]["path"] != attempt_root:
        raise ValueError("replay output root differs from Pair results root")
    expected_attempt = {
        "base_log_dir": (f"{results_root}/captured_replay/operational/" f"{document['pair_id']}/{attempt_id}/ray_logs"),
        "cache_read": {
            "entry_count": 0,
            "mode": "0700",
            "path": f"{cache_root}/captured_replay/{attempt_id}/cache_read",
            "policy": "empty-at-publication-and-job-entry-no-read",
        },
        "hf_datasets_cache": f"{hf_root}/captured_replay/{attempt_id}/hub",
        "hf_home": f"{hf_root}/captured_replay/{attempt_id}",
        "hf_hub_cache": f"{hf_root}/captured_replay/{attempt_id}/hub",
        "persistent_cache": f"{cache_root}/captured_replay/{attempt_id}",
        "operational": {
            "boundary": "outside-sealed-result",
            "root": (f"{results_root}/captured_replay/operational/" f"{document['pair_id']}/{attempt_id}"),
            "slurm": (f"{results_root}/captured_replay/operational/" f"{document['pair_id']}/{attempt_id}/slurm"),
            "ray_logs": (f"{results_root}/captured_replay/operational/" f"{document['pair_id']}/{attempt_id}/ray_logs"),
            "persistent_cache": f"{cache_root}/captured_replay/{attempt_id}",
            "hf_home": f"{hf_root}/captured_replay/{attempt_id}",
        },
        "results_dir": attempt_root,
        "scheduler": expected_scheduler,
        "setup_command": on_environment["setup_command"],
    }
    if not _exact(attempt, expected_attempt):
        raise ValueError("replay attempt environment differs from Pair-derived paths")
    state_root = f"{results_root}/captured_replay/replay_submission_state/" f"{document['pair_id']}/{attempt_id}"
    scheduler_submission = document["scheduler_submission"]
    if (
        scheduler_submission["accepted_id_record"]["path"] != f"{state_root}/accepted.job-id"
        or scheduler_submission["receipt"]["path"] != f"{state_root}/submission-receipt.json"
    ):
        raise ValueError("replay login-side state paths differ from Pair results root")
    if (
        document["slurm_export_boundary"]["path"]
        != f"{results_root}/captured_replay/slurm_exports/{document['pair_id']}/{attempt_id}.env"
    ):
        raise ValueError("replay Slurm export path differs from Pair results root")


def _pair_manifest_required_keys() -> frozenset[str]:
    return frozenset(
        {
            "acceptance",
            "arms",
            "artifacts",
            "campaign",
            "container_entry_boundary",
            "determinism_receipt_dir",
            "deployment",
            "execution_environment",
            "model_transport",
            "pair_campaign_reward_and_advantage_sha256",
            "pair_campaign_sha256",
            "pair_id",
            "paths",
            "schema",
            "selection",
            "slurm_export_boundary",
            "scheduler_submission",
            "source",
            "runtime_attestation",
            "runtime_tools",
            "wandb",
        }
    )


def _artifact_ref(value: Any, *, schema: str, name: str) -> None:
    _require_exact_keys(value, ARTIFACT_REF_KEYS, name=name)
    _absolute_path(value["path"], name=f"{name}.path")
    if value["schema"] != schema:
        raise ValueError(f"{name} schema differs")
    _digest(value["sha256"], name=f"{name}.sha256")


def _file_ref(value: Any, *, name: str, absolute: bool = True) -> None:
    _require_exact_keys(value, FILE_REF_KEYS, name=name)
    if absolute:
        _absolute_path(value["path"], name=f"{name}.path")
    elif type(value["path"]) is not str or value["path"].startswith("/") or ".." in Path(value["path"]).parts:
        raise ValueError(f"{name}.path must be a canonical relative path")
    _digest(value["sha256"], name=f"{name}.sha256")


def _digest(value: Any, *, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None or value == "0" * 64:
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256")
    return value


def _absolute_path(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value.startswith("//")
        or "\x00" in value
        or os.path.normpath(value) != value
    ):
        raise ValueError(f"{name} must be a canonical absolute path")
    return value


def _safe_id(value: Any, *, name: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("ascii", errors="ignore")) > maximum
        or _SAFE_ID_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a bounded safe identifier")
    return value


def _ascii(value: Any, *, name: str, maximum: int) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty ASCII string")
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be ASCII") from error
    if len(payload) > maximum:
        raise ValueError(f"{name} exceeds {maximum} ASCII bytes")
    return value


def _job_id(value: Any, *, name: str) -> str:
    if type(value) is not str or _JOB_ID_RE.fullmatch(value) is None or int(value) > (1 << 63) - 1:
        raise ValueError(f"{name} must be a canonical positive decimal int63 string")
    return value


def _attempt(value: Any) -> str:
    if value not in ATTEMPTS or type(value) is not str:
        raise ValueError("attempt_id must be replay-1 or replay-2")
    return value


def _argv(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        raise TypeError(f"{name} must be a nonempty string list")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or not item or "\x00" in item or len(item.encode("utf-8")) > 65536:
            raise ValueError(f"{name}[{index}] is invalid")
        result.append(item)
    return result


def _require_exact_keys(value: Any, expected: frozenset[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(f"{name} keyset mismatch: expected {sorted(expected)}, got {sorted(actual)}")


def _exact(left: Any, right: Any) -> bool:
    try:
        return canonical_ascii_json(left) == canonical_ascii_json(right)
    except (TypeError, ValueError):
        return False


def _profile_v2(
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> StrictCapturedReplayProfile:
    """Resolve only one exact caller-selected citation/freeform profile."""
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )

    return get_strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )


def _scorer_profile_v2(
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    """Project one frozen registry record into the manifest wire shape."""
    return {
        "environment": profile.environment,
        "profile_id": profile.profile_id,
        "verifier_type": profile.verifier_type,
        "method": profile.method,
        "resource_config_path_name": profile.resource_config_path_name,
        "disabled_config_path_name": profile.disabled_config_path_name,
        "resource_app": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
        "resource_config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "fixture": {
            "path": profile.fixture_path,
            "sha256": profile.fixture_sha256,
            "rows": profile.fixture_rows,
        },
        "call_schema": profile.call_schema,
        "closed_schema": profile.closed_schema,
        "call_index_schema": profile.call_index_schema,
    }


def _validate_profile_fixture_binding_v2(
    pair_manifest: Mapping[str, Any],
    *,
    profile: StrictCapturedReplayProfile,
) -> None:
    """Bind the selected Pair dataset to one immutable scorer profile."""
    source = pair_manifest.get("source")
    artifacts = pair_manifest.get("artifacts")
    if type(source) is not dict or type(artifacts) is not dict:
        raise ValueError("Pair source/artifacts must be exact dictionaries")
    source_root = source.get("root")
    fixture = artifacts.get("fixture")
    if type(source_root) is not str or type(fixture) is not dict:
        raise ValueError("Pair source root/fixture types differ")
    _require_exact_keys(
        fixture,
        frozenset({"path", "rows", "sha256"}),
        name="profiled Pair fixture",
    )
    root = _absolute_path(source_root, name="Pair source root")
    relative = PurePosixPath(profile.fixture_path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("closed profile fixture path is not canonical relative")
    expected_path = str(PurePosixPath(root) / relative)
    if (
        type(fixture["path"]) is not str
        or fixture["path"] != expected_path
        or type(fixture["sha256"]) is not str
        or fixture["sha256"] != profile.fixture_sha256
        or type(fixture["rows"]) is not int
        or fixture["rows"] != profile.fixture_rows
        or type(profile.fixture_rows) is not int
        or profile.fixture_rows != 5
    ):
        raise ValueError("Pair fixture differs from the closed scorer profile")
    _digest(fixture["sha256"], name="profiled Pair fixture SHA-256")


def _result_inventory_contract_v2(
    *,
    root: str,
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    files = [
        {
            "path": relative,
            "schema": schema,
            "mode": "0400",
        }
        for relative, schema in zip(
            profile.result_files,
            profile.result_file_schemas,
            strict=True,
        )
    ]
    return {
        "path": f"{root}/result-inventory-v2.json",
        "schema": REPLAY_RESULT_INVENTORY_V2_SCHEMA,
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
        "self_excluded": True,
        "terminal_directory_mode": "0555",
        "environment": profile.environment,
        "profile_id": profile.profile_id,
        "directories": [{"path": relative, "mode": "0555"} for relative in profile.result_directories],
        "files": files,
        "anchors": [copy.deepcopy(record) for record in files if record["path"] in profile.result_anchor_paths],
    }


def _gym_scorer_launcher_v2(
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    return {
        "scope": "selected-resource-scorer-only-not-full-main-run-helper-process-set",
        "mechanism": "nemo-gym-run-helper-shell-subprocess",
        "executable": "/bin/bash",
        "argv_shape": ["-c", "<runtime-generated-command>"],
        "core_command_template": (
            "cd {working-directory} && source {venv-directory}/bin/activate && "
            "NEMO_GYM_CONFIG_DICT={shlex-quoted-resolved-yaml} "
            "NEMO_GYM_CONFIG_PATH={shlex-quoted-config-path-name} python app.py"
        ),
        "log_wrapper": "forbidden",
        "forbidden_resolved_config_keys": ["nemo_gym_log_dir"],
        "generator": copy.deepcopy(GYM_RUN_HELPER_SOURCE),
        "setup": copy.deepcopy(GYM_SETUP_COMMAND_SOURCE),
        "allocator": copy.deepcopy(GYM_PORT_ALLOCATOR_SOURCE),
        "server": copy.deepcopy(GYM_SERVER_SOURCE),
        "port_policy": {
            "first": 5000,
            "last": 5999,
            "last_inclusive": True,
            "head_port_excluded": True,
            "selection": "python-random-randint-inclusive",
        },
        "host_policy": "loopback-127.0.0.1",
        "working_directory": (f"{GYM_CONTAINER_ROOT}/resources_servers/format_verification"),
        "venv_directory": ("/opt/gym_venvs/resources_servers/format_verification/.venv"),
        "entrypoint": "app.py",
        "config_path_name": profile.resource_config_path_name,
        "resource_only_config": None,
        "resolved_evidence_requirement": (
            "authenticated-replay-pre-and-exit-command-workdir-interpreter-host-port-process"
        ),
    }


def _gym_scorer_runtime_v2(
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    return {
        "required_common_distributions": {
            "nemo-gym": "0.5.0rc0",
            "openai": "2.6.1",
            "pydantic": "2.13.4",
            "pydantic-core": "2.46.4",
            "ray": "2.56.1",
        },
        "required_module_versions": {
            "nemo_gym": "0.5.1",
            "ray": "2.56.1",
        },
        "required_attestation": ("selected-resource-child-interpreter-process-and-port-before-first-request"),
        "required_per_call_success_evidence": {
            "call_schema": profile.call_schema,
            "closed_schema": profile.closed_schema,
            "expected_count": 4,
            "method": profile.method,
            "profile_id": profile.profile_id,
            "required_outcome_kind": "returned",
            "terminal_index_schema": profile.call_index_schema,
        },
        "required_python_version": "3.13.14",
        "required": True,
        "selected_resource_app": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
        "selected_resource_config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "selected_resource_requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "verifier_pin": {
            "environment": profile.environment,
            "profile_id": profile.profile_id,
            "type": profile.verifier_type,
            "method": profile.method,
        },
    }


def _runtime_requirements_v2(
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    return {
        "schema": REPLAY_RUNTIME_REQUIREMENTS_V2_SCHEMA,
        "shared_prefix_determinism": {
            "applicable": False,
            "reason": "verifier-only-no-policy-forward-backward-or-optimizer",
            "status": "not_applicable",
        },
        "model_transport_replay": {
            "schema": REPLAY_TRANSPORT_CONSUMPTION_V2_SCHEMA,
            "expected_count": 4,
            "required_status": "complete-terminal",
        },
        "derived_request_runtime": {
            **REQUIRED_RUNTIME_VERSIONS,
            "algorithm": "pinned-simple-agent-model-dump-v1",
            "forbidden_endpoints": ["/run"],
            "required_attestation": ("replay-driver-importlib-metadata-before-first-verifier-request"),
            "required": True,
        },
        "resource_scorer_child": _gym_scorer_runtime_v2(profile),
        "verifier": {
            "required_mode": "fresh-pinned-resource-scorer",
            "required_profile": {
                "environment": profile.environment,
                "profile_id": profile.profile_id,
            },
            "required_request_evidence": ("derived-from-pinned-simple-agent-source-not-wire-captured"),
            "required_response_evidence": "fresh-pinned-gym-result",
        },
        "lifecycle_schemas": {
            "submission_receipt": REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
            "pre_receipt": REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
            "exit_receipt": REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
            "runtime_attestation": REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA,
        },
    }


def _outputs_v2(
    *,
    attempt_root: str,
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    return {
        "directory": {
            "path": attempt_root,
            "mode": "0700",
            "precondition": "absent-at-pre-runtime-creates-exclusively",
        },
        "evidence_index": {
            "path": f"{attempt_root}/evidence-index.json",
            "schema": REPLAY_EVIDENCE_INDEX_V2_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "scorer_call_index": {
            "path": (f"{attempt_root}/{profile.scorer_terminal_index_path}"),
            "schema": profile.call_index_schema,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "transcript_bundle": {
            "path": f"{attempt_root}/transcript-bundle.json",
            "schema": TRANSCRIPT_BUNDLE_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "replay_ledger": {
            "path": f"{attempt_root}/replay-ledger.json",
            "schema": "nemo-rl-strict-captured-replay-step1-ledger-v5",
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
        "result_inventory": _result_inventory_contract_v2(
            root=attempt_root,
            profile=profile,
        ),
        "transport_consumption": {
            "path": f"{attempt_root}/model-transport-replay-consumption.json",
            "schema": REPLAY_TRANSPORT_CONSUMPTION_V2_SCHEMA,
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
        },
    }


def _replay_contract_v2(
    *,
    pair_manifest: Mapping[str, Any],
    static_inputs: AuthenticatedReplayStaticInputs,
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    return {
        "schema": REPLAY_CONTRACT_V2_SCHEMA,
        "claim": "fresh_verifier_reward_replay",
        "execution_scope": "scorer-only",
        "cohort": {
            "fixture_row_index": 0,
            "logical_rollout_indices": [0, 1, 2, 3],
            "sample_count": 4,
            "step": 1,
        },
        "policy_execution": {
            "backward": False,
            "forward": False,
            "optimizer": False,
            "violation": "fail-closed",
        },
        "model_transport": {
            "direct_python_generation": "forbidden",
            "expected_count": 4,
            "mode": "replay",
            "policy_sha256": pair_manifest["model_transport"]["policy_sha256"],
            "source_arm": "off",
            "streaming": "forbidden",
            "terminal_status": "complete-terminal",
        },
        "program": copy.deepcopy(static_inputs.replay_program),
        "selected_config": copy.deepcopy(pair_manifest["selection"]["config"]),
        "source_generation": copy.deepcopy(pair_manifest["campaign"]["generation"]),
        "source_snapshot": {
            "arm": "on",
            "ref": copy.deepcopy(static_inputs.source_snapshot),
        },
        "gym_scorer": {
            "nonexecuted_derivation_source_agent": (profile.disabled_config_path_name),
            "mode": "fresh-pinned-resource-scorer",
            "container": copy.deepcopy(static_inputs.container_asset),
            "launcher": _gym_scorer_launcher_v2(profile),
            "resources": copy.deepcopy(pair_manifest["selection"]["gym_resources"]),
            "source_root": copy.deepcopy(static_inputs.gym_source_root),
            "runtime": _gym_scorer_runtime_v2(profile),
            "source": copy.deepcopy(pair_manifest["source"]["gym"]),
        },
    }


def build_replay_execution_manifest_v2(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    """Build a profile-bound citation/freeform execution manifest."""
    profile = _profile_v2(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    source = _reload_authenticated_off_source_capture(authenticated_source)
    pair_manifest = source.pair_manifest
    _require_exact_keys(pair_manifest, _pair_manifest_required_keys(), name="Pair")
    if pair_manifest["schema"] != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unexpected Pair manifest schema")
    if pair_manifest["selection"]["environment"] != profile.environment:
        raise ValueError("Pair environment differs from expected scorer profile")
    _validate_profile_fixture_binding_v2(pair_manifest, profile=profile)
    pair_id = _safe_id(pair_manifest["pair_id"], name="Pair pair_id", maximum=64)
    attempt = _attempt(attempt_id)
    static_inputs = _load_authenticated_replay_static_inputs_v2(
        authenticated_source=source,
        attempt_id=attempt,
        profile=profile,
    )
    results_root = _absolute_path(
        pair_manifest["paths"]["results_root"],
        name="results_root",
    )
    cache_root = _absolute_path(
        pair_manifest["paths"]["cache_root"],
        name="cache_root",
    )
    hf_root = _absolute_path(
        pair_manifest["paths"]["hf_home"],
        name="hf_home",
    )
    attempt_root = f"{results_root}/captured_replay/{attempt}"
    operational_root = f"{results_root}/captured_replay/operational/{pair_id}/{attempt}"
    submission_state_root = f"{results_root}/captured_replay/replay_submission_state/{pair_id}/{attempt}"
    persistent_cache = f"{cache_root}/captured_replay/{attempt}"
    hf_home = f"{hf_root}/captured_replay/{attempt}"
    pair_manifest_sha256 = _digest(
        source.pair_manifest_sha256,
        name="pair_manifest_sha256",
    )
    actual_pair_sha256 = hashlib.sha256(canonical_ascii_json(pair_manifest) + b"\n").hexdigest()
    if pair_manifest_sha256 != actual_pair_sha256:
        raise ValueError("Pair manifest raw SHA-256 does not close to canonical Pair bytes")
    pair_submission_receipt_sha256 = _digest(
        source.pair_submission_receipt_sha256,
        name="pair_submission_receipt_sha256",
    )
    replay_submission_contract_sha256 = _digest(
        static_inputs.submission_contract_sha256,
        name="replay submission contract SHA",
    )
    slurm_export_sha256 = _digest(
        static_inputs.slurm_export_sha256,
        name="Slurm export SHA",
    )
    submission_nonce = _digest(
        static_inputs.submission_contract["submission_nonce"],
        name="submission_nonce",
    )
    submitter_euid = static_inputs.submission_contract["submitter_euid"]
    if type(submitter_euid) is not int or not 0 <= submitter_euid <= (1 << 31) - 1:
        raise ValueError("submitter_euid must be an exact nonnegative int31")

    pair_binding = {
        "id": pair_id,
        "environment": profile.environment,
        "manifest": {
            "path": f"{results_root}/PAIR_MANIFEST.json",
            "schema": PAIR_MANIFEST_SCHEMA,
            "sha256": pair_manifest_sha256,
        },
        "submission_receipt": {
            "path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
            "schema": PAIR_SUBMISSION_RECEIPT_SCHEMA,
            "sha256": pair_submission_receipt_sha256,
        },
        "acceptance_policy_sha256": pair_manifest["acceptance"]["policy_sha256"],
        "model_transport_policy_sha256": pair_manifest["model_transport"]["policy_sha256"],
        "pair_campaign_sha256": pair_manifest["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": pair_manifest["pair_campaign_reward_and_advantage_sha256"],
    }
    document: dict[str, Any] = {
        "schema": REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": profile.environment,
        "scorer_profile": _scorer_profile_v2(profile),
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": attempt,
        "pair": pair_binding,
        "source_capture": source.document,
        "replay_contract": _replay_contract_v2(
            pair_manifest=pair_manifest,
            static_inputs=static_inputs,
            profile=profile,
        ),
        "artifacts": {
            "container": copy.deepcopy(pair_manifest["artifacts"]["container"]),
            "fixture": copy.deepcopy(pair_manifest["artifacts"]["fixture"]),
            "model": copy.deepcopy(pair_manifest["artifacts"]["model"]),
            "sandbox_container": copy.deepcopy(pair_manifest["artifacts"]["sandbox_container"]),
            "outputs": _outputs_v2(
                attempt_root=attempt_root,
                profile=profile,
            ),
        },
        "execution_environment": {
            "schema": REPLAY_EXECUTION_ENVIRONMENT_SCHEMA,
            "arm_launcher": copy.deepcopy(pair_manifest["execution_environment"]["arm_launcher"]),
            "fixed": copy.deepcopy(pair_manifest["execution_environment"]["fixed"]),
            "attempt": {
                "base_log_dir": f"{operational_root}/ray_logs",
                "cache_read": {
                    "entry_count": 0,
                    "mode": "0700",
                    "path": f"{persistent_cache}/cache_read",
                    "policy": "empty-at-publication-and-job-entry-no-read",
                },
                "hf_datasets_cache": f"{hf_home}/hub",
                "hf_home": hf_home,
                "hf_hub_cache": f"{hf_home}/hub",
                "persistent_cache": persistent_cache,
                "operational": {
                    "boundary": "outside-sealed-result",
                    "root": operational_root,
                    "slurm": f"{operational_root}/slurm",
                    "ray_logs": f"{operational_root}/ray_logs",
                    "persistent_cache": persistent_cache,
                    "hf_home": hf_home,
                },
                "results_dir": attempt_root,
                "scheduler": {
                    "batch_working_directory": static_inputs.source_snapshot["path"],
                    "sbatch_chdir_argument": (f"--chdir={static_inputs.source_snapshot['path']}"),
                    "sbatch_client_cwd": static_inputs.source_snapshot["path"],
                    "slurm_submit_dir": static_inputs.source_snapshot["path"],
                },
                "setup_command": copy.deepcopy(pair_manifest["execution_environment"]["arms"]["on"]["setup_command"]),
            },
        },
        "wandb": copy.deepcopy(REPLAY_WANDB_POLICY),
        "scheduler_submission": {
            "schema": "nemo-rl-strict-scheduler-submission-v1",
            "accepted_id_record": {
                "accepted_format": "ascii-positive-decimal-lf",
                "capture_format": "opaque-sbatch-stdout",
                "initial_mode": "0600",
                "path": f"{submission_state_root}/accepted.job-id",
                "sealed_mode": "0400",
            },
            "contract": {
                "path": _absolute_path(
                    static_inputs.submission_contract_path,
                    name="replay submission contract path",
                ),
                "sha256": replay_submission_contract_sha256,
            },
            "identity": {
                "comment_template": (
                    "nemo-rl-strict-captured-replay-v2:{attempt_id}:" "{submission_nonce}:{replay_manifest_sha256}"
                ),
                "job_name": f"strict-replay-{attempt}-{pair_id}",
                "submitter_euid": submitter_euid,
            },
            "nonce": submission_nonce,
            "receipt": {
                "path": f"{submission_state_root}/submission-receipt.json",
                "schema": REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
            },
        },
        "slurm_export_boundary": {
            "schema": REPLAY_SLURM_EXPORT_SCHEMA,
            "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
            "ambient_merge": False,
            "attempt_id": attempt,
            "format": "nul-separated-name-value",
            "get_user_env": False,
            "job_argv_template": list(REPLAY_JOB_ARGV_TEMPLATE_V2),
            "path": _absolute_path(
                static_inputs.slurm_export_path,
                name="Slurm export path",
            ),
            "sha256": slurm_export_sha256,
        },
        "deployment": copy.deepcopy(pair_manifest["deployment"]),
        "runtime_attestation_requirements": _runtime_requirements_v2(profile),
        "runtime_tools": copy.deepcopy(pair_manifest["runtime_tools"]),
        "container_entry_boundary": copy.deepcopy(pair_manifest["container_entry_boundary"]),
        "source": copy.deepcopy(pair_manifest["source"]),
    }
    _validate_replay_execution_manifest_v2_against_source(
        document,
        source=source,
        static_inputs=static_inputs,
        profile=profile,
    )
    return document


def _validate_replay_contract_v2(
    value: Any,
    *,
    profile: StrictCapturedReplayProfile,
    model_transport_policy_sha256: str,
) -> None:
    _require_exact_keys(value, REPLAY_CONTRACT_KEYS, name="profiled replay_contract")
    if (
        value["schema"] != REPLAY_CONTRACT_V2_SCHEMA
        or value["claim"] != "fresh_verifier_reward_replay"
        or value["execution_scope"] != "scorer-only"
    ):
        raise ValueError("profiled replay contract claim/scope differs")
    if not _exact(
        value["cohort"],
        {
            "fixture_row_index": 0,
            "logical_rollout_indices": [0, 1, 2, 3],
            "sample_count": 4,
            "step": 1,
        },
    ):
        raise ValueError("profiled replay contract cohort differs from K4 step1")
    if not _exact(
        value["policy_execution"],
        {
            "backward": False,
            "forward": False,
            "optimizer": False,
            "violation": "fail-closed",
        },
    ):
        raise ValueError("profiled replay contract must forbid all policy execution")
    _digest(
        model_transport_policy_sha256,
        name="profiled replay transport policy SHA",
    )
    expected_transport = {
        "direct_python_generation": "forbidden",
        "expected_count": 4,
        "mode": "replay",
        "policy_sha256": model_transport_policy_sha256,
        "source_arm": "off",
        "streaming": "forbidden",
        "terminal_status": "complete-terminal",
    }
    if not _exact(value["model_transport"], expected_transport):
        raise ValueError("profiled replay model-transport contract differs")
    _validate_program_v2(value["program"])
    scorer = value["gym_scorer"]
    _require_exact_keys(
        scorer,
        frozenset(
            {
                "mode",
                "container",
                "launcher",
                "nonexecuted_derivation_source_agent",
                "resources",
                "runtime",
                "source",
                "source_root",
            }
        ),
        name="profiled gym_scorer",
    )
    if (
        scorer["nonexecuted_derivation_source_agent"] != profile.disabled_config_path_name
        or scorer["mode"] != "fresh-pinned-resource-scorer"
    ):
        raise ValueError("profiled resource scorer provenance/mode differs")
    if not _exact(scorer["launcher"], _gym_scorer_launcher_v2(profile)):
        raise ValueError("profiled Gym scorer static launcher policy differs")
    if not _exact(scorer["runtime"], _gym_scorer_runtime_v2(profile)):
        raise ValueError("profiled Gym scorer runtime requirements differ")
    expected_resources = {
        "config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "verifier_source": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
    }
    if not _exact(scorer["resources"], expected_resources):
        raise ValueError("profiled Gym resource pins differ from closed registry")
    container = scorer["container"]
    _require_exact_keys(
        container,
        GYM_SCORER_CONTAINER_KEYS,
        name="profiled gym_scorer.container",
    )
    _absolute_path(container["path"], name="profiled gym scorer container path")
    _digest(container["sha256"], name="profiled gym scorer container SHA")
    if (
        type(container["owner_uid"]) is not int
        or container["owner_uid"] != REPLAY_CONTAINER_OWNER_UID
        or type(container["owner_gid"]) is not int
        or container["owner_gid"] != REPLAY_CONTAINER_OWNER_GID
    ):
        raise ValueError("profiled Gym scorer publisher identity differs")
    _require_exact_keys(
        scorer["source_root"],
        frozenset({"snapshot_relative_path", "host_path", "container_path"}),
        name="profiled gym_scorer.source_root",
    )
    if scorer["source_root"]["snapshot_relative_path"] != GYM_SNAPSHOT_RELATIVE_ROOT:
        raise ValueError("profiled Gym scorer snapshot-relative root differs")
    _absolute_path(
        scorer["source_root"]["host_path"],
        name="profiled Gym host source root",
    )
    if scorer["source_root"]["container_path"] != GYM_CONTAINER_ROOT:
        raise ValueError("profiled Gym container source root differs")


def _validate_outputs_v2(
    value: Any,
    *,
    attempt_id: str,
    profile: StrictCapturedReplayProfile,
) -> None:
    _require_exact_keys(value, OUTPUT_V2_KEYS, name="profiled artifacts.outputs")
    directory = value["directory"]
    _require_exact_keys(
        directory,
        frozenset({"path", "mode", "precondition"}),
        name="profiled replay output directory",
    )
    root = _absolute_path(
        directory["path"],
        name="profiled replay output directory path",
    )
    if (
        not root.endswith(f"/captured_replay/{attempt_id}")
        or directory["mode"] != "0700"
        or directory["precondition"] != "absent-at-pre-runtime-creates-exclusively"
    ):
        raise ValueError("profiled replay output directory contract differs")
    expected = _outputs_v2(attempt_root=root, profile=profile)
    if not _exact(value, expected):
        raise ValueError("profiled replay output contract differs")


def _validate_scheduler_submission_v2(
    value: Any,
    *,
    pair_id: str,
    attempt_id: str,
) -> None:
    _require_exact_keys(
        value,
        SCHEDULER_SUBMISSION_KEYS,
        name="profiled scheduler_submission",
    )
    _require_exact_keys(
        value["identity"],
        frozenset({"comment_template", "job_name", "submitter_euid"}),
        name="profiled replay scheduler identity",
    )
    _require_exact_keys(
        value["receipt"],
        frozenset({"path", "schema"}),
        name="profiled replay submission receipt declaration",
    )
    if (
        value["identity"]["comment_template"]
        != ("nemo-rl-strict-captured-replay-v2:{attempt_id}:" "{submission_nonce}:{replay_manifest_sha256}")
        or value["receipt"]["schema"] != REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA
    ):
        raise ValueError("profiled replay scheduler version binding differs")
    legacy = copy.deepcopy(dict(value))
    legacy["identity"]["comment_template"] = (
        "nemo-rl-strict-captured-replay-v1:{attempt_id}:" "{submission_nonce}:{replay_manifest_sha256}"
    )
    legacy["receipt"]["schema"] = REPLAY_SUBMISSION_RECEIPT_SCHEMA
    _validate_scheduler_submission(
        legacy,
        pair_id=pair_id,
        attempt_id=attempt_id,
    )


def _validate_slurm_export_v2(value: Any, *, attempt_id: str) -> None:
    """Validate exact environment/profile argv without widening V1."""
    _require_exact_keys(
        value,
        SLURM_EXPORT_KEYS,
        name="profiled slurm_export_boundary",
    )
    if value["job_argv_template"] != list(REPLAY_JOB_ARGV_TEMPLATE_V2):
        raise ValueError("profiled replay Slurm job argv template differs")
    legacy = copy.deepcopy(dict(value))
    legacy["job_argv_template"] = list(REPLAY_JOB_ARGV_TEMPLATE_V2[:-4])
    _validate_slurm_export(legacy, attempt_id=attempt_id)


def _validate_runtime_requirements_v2(
    value: Any,
    *,
    profile: StrictCapturedReplayProfile,
) -> None:
    _require_exact_keys(
        value,
        RUNTIME_REQUIREMENTS_V2_KEYS,
        name="profiled runtime_attestation_requirements",
    )
    if not _exact(value, _runtime_requirements_v2(profile)):
        raise ValueError("profiled replay runtime requirements contract differs")


def _validate_replay_execution_manifest_v2_shape(
    document: Mapping[str, Any],
    *,
    pair_manifest: Mapping[str, Any],
    profile: StrictCapturedReplayProfile,
) -> None:
    _require_exact_keys(
        document,
        ROOT_V2_KEYS,
        name="profiled replay execution manifest",
    )
    if (
        document["schema"] != REPLAY_EXECUTION_MANIFEST_V2_SCHEMA
        or document["hash_domain"] != HASH_DOMAIN
        or document["environment"] != profile.environment
    ):
        raise ValueError("profiled replay execution manifest identity differs")
    if not _exact(document["scorer_profile"], _scorer_profile_v2(profile)):
        raise ValueError("profiled replay scorer_profile differs from registry")
    pair_id = _safe_id(document["pair_id"], name="pair_id", maximum=64)
    if document["arm"] != "on" or document["mode"] != "fresh_verifier_reward_replay":
        raise ValueError("profiled replay must be on-arm fresh_verifier_reward_replay")
    attempt_id = _attempt(document["attempt_id"])
    _validate_pair_binding(
        document["pair"],
        pair_id=pair_id,
        environment=profile.environment,
    )
    _validate_source_capture(document["source_capture"])
    _validate_replay_contract_v2(
        document["replay_contract"],
        profile=profile,
        model_transport_policy_sha256=document["pair"]["model_transport_policy_sha256"],
    )
    _require_exact_keys(document["artifacts"], ARTIFACTS_KEYS, name="artifacts")
    _validate_outputs_v2(
        document["artifacts"]["outputs"],
        attempt_id=attempt_id,
        profile=profile,
    )
    _require_exact_keys(
        document["execution_environment"],
        EXECUTION_ENVIRONMENT_KEYS,
        name="execution_environment",
    )
    if document["execution_environment"]["schema"] != REPLAY_EXECUTION_ENVIRONMENT_SCHEMA:
        raise ValueError("unexpected profiled replay execution-environment schema")
    _require_exact_keys(
        document["execution_environment"]["attempt"],
        ATTEMPT_ENVIRONMENT_KEYS,
        name="execution_environment.attempt",
    )
    if not _exact(document["wandb"], REPLAY_WANDB_POLICY):
        raise ValueError("profiled replay W&B disabled policy differs")
    _validate_scheduler_submission_v2(
        document["scheduler_submission"],
        pair_id=pair_id,
        attempt_id=attempt_id,
    )
    _validate_slurm_export_v2(
        document["slurm_export_boundary"],
        attempt_id=attempt_id,
    )
    _validate_runtime_requirements_v2(
        document["runtime_attestation_requirements"],
        profile=profile,
    )
    canonical_ascii_json(document)
    if pair_manifest["selection"]["environment"] != profile.environment:
        raise ValueError("authenticated Pair differs from expected scorer profile")
    _validate_profile_fixture_binding_v2(pair_manifest, profile=profile)
    _validate_against_pair(document, pair_manifest)


def _validate_replay_execution_manifest_v2_against_source(
    document: Mapping[str, Any],
    *,
    source: AuthenticatedOffSourceCapture,
    static_inputs: AuthenticatedReplayStaticInputs,
    profile: StrictCapturedReplayProfile,
) -> None:
    _validate_replay_execution_manifest_v2_shape(
        document,
        pair_manifest=source.pair_manifest,
        profile=profile,
    )
    _validate_manifest_static_inputs(document, static_inputs=static_inputs)
    if not _exact(document["source_capture"], source.source_capture):
        raise ValueError("profiled replay source capture differs from authenticated OFF bytes")
    if document["pair"]["manifest"]["sha256"] != source.pair_manifest_sha256:
        raise ValueError("profiled replay Pair manifest anchor differs")
    if document["pair"]["submission_receipt"]["sha256"] != source.pair_submission_receipt_sha256:
        raise ValueError("profiled replay Pair receipt anchor differs")


def validate_replay_execution_manifest_v2(
    document: Mapping[str, Any],
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    """Validate a V2 manifest against explicit profile and fresh authority."""
    profile = _profile_v2(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    source = _reload_authenticated_off_source_capture(authenticated_source)
    static_inputs = _load_authenticated_replay_static_inputs_v2(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
        profile=profile,
    )
    _validate_replay_execution_manifest_v2_against_source(
        document,
        source=source,
        static_inputs=static_inputs,
        profile=profile,
    )


def publish_replay_execution_manifest_v2(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    """Validate and exclusively publish one profile-bound V2 manifest."""
    profile = _profile_v2(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    source = _reload_authenticated_off_source_capture(authenticated_source)
    static_inputs = _load_authenticated_replay_static_inputs_v2(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
        profile=profile,
    )
    _validate_replay_execution_manifest_v2_against_source(
        document,
        source=source,
        static_inputs=static_inputs,
        profile=profile,
    )
    return publish_evidence_document(
        output=output,
        document=document,
        trailing_lf=False,
    )


def load_replay_execution_manifest_v2(
    *,
    path: str | Path,
    expected_sha256: str,
    authenticated_source: AuthenticatedOffSourceCapture,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], str]:
    """Load only the explicitly selected profile-bound V2 manifest family."""
    profile = _profile_v2(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    source = _reload_authenticated_off_source_capture(authenticated_source)
    document, digest = load_evidence_document(
        path=path,
        expected_sha256=expected_sha256,
        trailing_lf=False,
    )
    static_inputs = _load_authenticated_replay_static_inputs_v2(
        authenticated_source=source,
        attempt_id=_attempt(document.get("attempt_id")),
        profile=profile,
    )
    _validate_replay_execution_manifest_v2_against_source(
        document,
        source=source,
        static_inputs=static_inputs,
        profile=profile,
    )
    return document, digest
