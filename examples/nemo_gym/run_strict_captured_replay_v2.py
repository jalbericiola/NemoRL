#!/usr/bin/env python3
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

"""Authenticated executable entrypoint for citation/freeform replay V2.

Only the external replay wrapper may invoke the driver phase.  This module is
kept stdlib-only at import time: it authenticates the immutable replay manifest,
PRE receipt, source snapshot, and every replay-program member before adding the
authenticated source root to ``sys.path`` or importing NeMo-RL/NeMo-Gym.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import math
import os
import posixpath
import re
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.algorithms.strict_captured_replay_runtime_v2 import (
        FinalizeFormatCallEvidence,
        IndependentFormatCheck,
        ReplayDocumentsV2,
        StrictModelTransportReplaySourceV3,
        VerifierPost,
    )


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_MAX_INT63 = (1 << 63) - 1
_RUN_HELPER_REAP_TIMEOUT_SECONDS = 5.0
_RUN_HELPER_HANDLE_UNSET = object()
_MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
_MANIFEST_HASH_DOMAIN = "sha256-domain-nul-canonical-ascii-json-no-lf-v1"
_PRE_SCHEMA = "nemo-rl-strict-captured-replay-job-pre-receipt-v3"
_PAIR_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
_PAIR_SUBMISSION_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
_REPLAY_SUBMISSION_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v5"
_SCHEDULER_QUERY_SCHEMA = "nemo-rl-strict-captured-replay-scheduler-query-v3"
_REPLAY_SLURM_EXPORT_SCHEMA = "nemo-rl-strict-captured-replay-slurm-export-file-v2"
_PAIR_SLURM_EXPORT_SCHEMA = "nemo-rl-strict-slurm-export-file-v3"
_SNAPSHOT_SHA_MANIFEST = "strict-pair-snapshot-manifest.sha256"
_SNAPSHOT_SYMLINK_MANIFEST = "strict-pair-snapshot-symlinks.json"
_SNAPSHOT_MODE_MANIFEST = "strict-pair-snapshot-modes.json"
_GYM_SOURCE_RELATIVE = "3rdparty/Gym-workspace/Gym"
_GYM_CONTAINER_ROOT = "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
_GYM_PACKAGE_INIT_RELATIVE = f"{_GYM_SOURCE_RELATIVE}/nemo_gym/__init__.py"
_FORMAT_PROFILES = {
    "citation": {
        "environment": "citation",
        "profile_id": "citation-string-match-v1",
        "verifier_type": "string_match",
        "method": "_verify_string_match",
        "resource_config_path_name": "citation_format",
        "disabled_config_path_name": "citation_format_simple_agent",
        "resource_app": {
            "path": "resources_servers/format_verification/app.py",
            "sha256": "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36",
        },
        "resource_config": {
            "path": "resources_servers/format_verification/configs/citation_format.yaml",
            "sha256": "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796",
        },
        "requirements": {
            "path": "resources_servers/format_verification/requirements.txt",
            "sha256": "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d",
        },
        "fixture": {
            "path": "tests/unit/tools/data/citation_example.jsonl",
            "sha256": "d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc",
            "rows": 5,
        },
        "call_schema": "nemo-rl-strict-format-verification-call-v1",
        "closed_schema": "nemo-rl-strict-format-verification-closed-v1",
        "call_index_schema": "nemo-rl-strict-format-verification-call-index-v1",
    },
    "freeform": {
        "environment": "freeform",
        "profile_id": "freeform-regex-v1",
        "verifier_type": "regex",
        "method": "_verify_regex",
        "resource_config_path_name": "freeform_formatting",
        "disabled_config_path_name": "freeform_formatting_simple_agent",
        "resource_app": {
            "path": "resources_servers/format_verification/app.py",
            "sha256": "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36",
        },
        "resource_config": {
            "path": "resources_servers/format_verification/configs/freeform_formatting.yaml",
            "sha256": "92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa",
        },
        "requirements": {
            "path": "resources_servers/format_verification/requirements.txt",
            "sha256": "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d",
        },
        "fixture": {
            "path": "tests/unit/tools/data/freeform_example.jsonl",
            "sha256": "8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399",
            "rows": 5,
        },
        "call_schema": "nemo-rl-strict-format-verification-call-v1",
        "closed_schema": "nemo-rl-strict-format-verification-closed-v1",
        "call_index_schema": "nemo-rl-strict-format-verification-call-index-v1",
    },
}
_DISABLED_WANDB_POLICY = {
    "enabled": False,
    "mode": "disabled",
    "reason": "scorer-only-replay-no-wandb-credentials-or-output",
}
_SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA = "nemo-rl-strict-scheduler-device-environment-v1"
_RUNTIME_DEVICE_ENVIRONMENT_FIELDS = (
    ("CUDA_VISIBLE_DEVICES", "cuda_visible_devices"),
    ("GPU_DEVICE_ORDINAL", "gpu_device_ordinal"),
    ("NVIDIA_VISIBLE_DEVICES", "nvidia_visible_devices"),
    ("ROCR_VISIBLE_DEVICES", "rocr_visible_devices"),
    ("ZE_AFFINITY_MASK", "ze_affinity_mask"),
)
_SCHEDULER_DEVICE_ENVIRONMENT_KEYS = frozenset(
    {"schema", *(field for _, field in _RUNTIME_DEVICE_ENVIRONMENT_FIELDS)}
)
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_CANONICAL_DEVICE_COMPONENT = r"(?:0|[1-9][0-9]*)"
_ZE_DEVICE_RE = re.compile(
    rf"{_CANONICAL_DEVICE_COMPONENT}(?:\.{_CANONICAL_DEVICE_COMPONENT})?\Z"
)
_WANDB_ENV_NAMES = frozenset(
    {
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_NAME",
        "WANDB_PROJ",
        "WANDB_RESUME",
        "WANDB_RUN_GROUP",
        "WANDB_RUN_ID",
    }
)
_SLURM_EXPORT_ALLOWED_NAMES = (
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
_SLURM_EXPORT_BOUNDARY_KEYS = frozenset(
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
_PROGRAM_PATHS = {
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
_PROGRAM_MODULE_NAMES = {
    "nemo_rl.algorithms.strict_captured_replay_runtime_v2": "runtime",
    "nemo_rl.environments.strict_gym_child_runtime_v2": "gym_child_runtime",
    "nemo_rl.utils.strict_captured_replay_evidence": "legacy_evidence_utility",
    "nemo_rl.utils.strict_captured_replay_evidence_v2": "evidence_utility",
    "nemo_rl.utils.strict_captured_replay_manifest_v2": "manifest_utility",
    "nemo_rl.utils.strict_captured_replay_profiles": "profile_registry",
    "nemo_rl.utils.strict_captured_replay_seal_v2": "result_sealer",
    "nemo_rl.utils.strict_main_step_ledger": "main_step_ledger",
    "nemo_rl.utils.strict_model_transport": "model_transport_utility",
    "nemo_rl.utils.strict_model_transport_replay_v3": "raw_transport_owner",
}
_BOOTSTRAP_REQUIRED_PROGRAM_NAMES = frozenset(
    {
        "evidence_utility",
        "legacy_evidence_utility",
        "manifest_utility",
        "profile_registry",
    }
)
_DRIVER_LOADER_REQUIRED_PROGRAM_NAMES = frozenset(
    {
        "evidence_utility",
        "legacy_evidence_utility",
        "main_step_ledger",
        "manifest_utility",
        "model_transport_utility",
        "raw_transport_owner",
    }
)
_DRIVER_PROFILE_REQUIRED_PROGRAM_NAMES = frozenset(
    {*_DRIVER_LOADER_REQUIRED_PROGRAM_NAMES, "profile_registry"}
)
_DRIVER_RUNTIME_REQUIRED_PROGRAM_NAMES = frozenset(
    {
        *_DRIVER_PROFILE_REQUIRED_PROGRAM_NAMES,
        "gym_child_runtime",
        "runtime",
    }
)
_MANIFEST_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "scorer_profile",
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
_PRE_ROOT_KEYS = frozenset(
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
_PRE_STATIC_BOUNDARY_KEYS = frozenset(
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
_PAIR_ROOT_KEYS = frozenset(
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
_PAIR_SUBMISSION_ROOT_KEYS = frozenset(
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
_REPLAY_SUBMISSION_ROOT_KEYS = frozenset(
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
_SCHEDULER_QUERY_KEYS = frozenset(
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
_SCHEDULER_RECORD_KEYS = frozenset(
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


class StrictCapturedReplayEntrypointError(RuntimeError):
    """The external wrapper/driver authentication boundary was not closed."""


@dataclass(frozen=True)
class _BootstrapAuthority:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    pre_receipt: dict[str, Any]
    pre_receipt_path: Path
    pre_receipt_sha256: str
    pair_manifest: dict[str, Any]
    execution_source_root: Path
    authenticated_job_id: str
    scheduler_device_environment: dict[str, Any]
    snapshot_authentication: dict[str, str]


def _closed_profile(
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    """Return one explicitly selected stdlib-only bootstrap profile."""
    if type(expected_environment) is not str or type(expected_profile_id) is not str:
        raise StrictCapturedReplayEntrypointError(
            "format environment/profile must be exact strings"
        )
    profile = _FORMAT_PROFILES.get(expected_environment)
    if profile is None or profile.get("profile_id") != expected_profile_id:
        raise StrictCapturedReplayEntrypointError(
            "format environment/profile pair is not admitted"
        )
    return json.loads(_canonical_ascii_json(profile).decode("ascii"))


def _contains_forbidden_server_type(value: Any) -> bool:
    """Return whether a resolved config retains any agent/model server key."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"responses_api_agents", "responses_api_models"}:
                return True
            if _contains_forbidden_server_type(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_server_type(item) for item in value)
    return False


def run_from_authenticated_wrapper(
    *,
    manifest: Mapping[str, Any],
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    authenticated_job_id: str,
    driver_process: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    source_transcript_document: Mapping[str, Any],
    source_main_ledger_document: Mapping[str, Any],
    transport_source: "StrictModelTransportReplaySourceV3",
    post_verifier: "VerifierPost",
    independent_format_check: "IndependentFormatCheck | None",
    finalize_format_call_evidence: "FinalizeFormatCallEvidence",
    expected_environment: str,
    expected_profile_id: str,
) -> "ReplayDocumentsV2":
    """Run only with objects already authenticated by the replay wrapper PRE."""
    from nemo_rl.algorithms.strict_captured_replay_runtime_v2 import (
        execute_profiled_captured_replay_cohort,
    )

    return execute_profiled_captured_replay_cohort(
        manifest=manifest,
        replay_execution_manifest_sha256=replay_execution_manifest_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        authenticated_job_id=authenticated_job_id,
        driver_process=driver_process,
        driver_scheduler_device_environment=driver_scheduler_device_environment,
        source_transcript_document=source_transcript_document,
        source_main_ledger_document=source_main_ledger_document,
        transport_source=transport_source,
        post_verifier=post_verifier,
        independent_format_check=independent_format_check,
        finalize_format_call_evidence=finalize_format_call_evidence,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )


def _force_reap_profiled_resource_child(
    *,
    run_helper: Any,
    resource_process: Any,
    head_server: Any,
    head_server_thread: Any,
    child_runtime: Any,
    authenticated_process_identity: tuple[int, int] | None,
) -> None:
    """Reap and verify the exact scorer after ``RunHelper.shutdown`` returns.

    In scorer-only mode the authenticated launch hook returns the direct Python
    resource child as RunHelper's ``Popen`` object.  Once attestation succeeds,
    use the receipt-bound ``(pid, start_ticks)`` with the child runtime's pidfd
    terminator.  A failure during ``RunHelper.start`` has no receipt-bound
    identity yet, so only its launch-hook-owned ``Popen`` may be killed.  In
    both cases a successful return means the Popen was reaped and the head
    server thread is no longer live.
    """
    failures: list[BaseException] = []
    if resource_process is not None:
        pid = getattr(resource_process, "pid", None)
        if type(pid) is not int or pid <= 1:
            failures.append(
                StrictCapturedReplayEntrypointError(
                    "strict scorer RunHelper process has an invalid PID"
                )
            )
        else:
            try:
                returncode = resource_process.poll()
                if returncode is None:
                    if authenticated_process_identity is None:
                        resource_process.kill()
                    else:
                        authenticated_pid, authenticated_start_ticks = (
                            authenticated_process_identity
                        )
                        if authenticated_pid != pid:
                            raise StrictCapturedReplayEntrypointError(
                                "strict scorer Popen PID differs from its attestation"
                            )
                        child_runtime._terminate_authenticated_process(
                            authenticated_pid,
                            authenticated_start_ticks,
                        )
                    returncode = resource_process.wait(
                        timeout=_RUN_HELPER_REAP_TIMEOUT_SECONDS
                    )
                else:
                    # poll() has already waitpid-reaped the direct child, but a
                    # second bounded wait also exercises the exact Popen handle.
                    returncode = resource_process.wait(timeout=0)
                if (
                    type(returncode) is not int
                    or type(resource_process.poll()) is not int
                ):
                    raise StrictCapturedReplayEntrypointError(
                        "strict scorer Popen did not reach a reaped terminal state"
                    )
                if authenticated_process_identity is not None:
                    authenticated_pid, authenticated_start_ticks = (
                        authenticated_process_identity
                    )
                    try:
                        _, post_reap_start_ticks = child_runtime._process_stat(
                            authenticated_pid
                        )
                    except (FileNotFoundError, ProcessLookupError):
                        post_reap_start_ticks = None
                    if post_reap_start_ticks == authenticated_start_ticks:
                        raise StrictCapturedReplayEntrypointError(
                            "strict scorer remained live after authenticated reap"
                        )
            except BaseException as error:
                failures.append(error)

    if head_server is not None:
        try:
            head_server.should_exit = True
        except BaseException as error:
            failures.append(error)
    if head_server_thread is not None:
        try:
            head_server_thread.join(timeout=_RUN_HELPER_REAP_TIMEOUT_SECONDS)
            alive = head_server_thread.is_alive()
            if type(alive) is not bool or alive:
                raise StrictCapturedReplayEntrypointError(
                    "strict scorer RunHelper head server remained live"
                )
        except BaseException as error:
            failures.append(error)
    elif head_server is not None:
        failures.append(
            StrictCapturedReplayEntrypointError(
                "strict scorer RunHelper head server has no joinable thread"
            )
        )

    if not failures:
        run_helper._processes = {}
        run_helper._head_server = None
        run_helper._head_server_thread = None
        return
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(
        "strict scorer forced cleanup had multiple failures",
        failures,
    )


def _shutdown_profiled_resource_child(
    *,
    run_helper: Any,
    session: Any,
    child_runtime: Any,
    authenticated_process_identity: tuple[int, int] | None,
    primary_failure: BaseException | None,
    captured_resource_process: Any = _RUN_HELPER_HANDLE_UNSET,
    captured_head_server: Any = _RUN_HELPER_HANDLE_UNSET,
    captured_head_server_thread: Any = _RUN_HELPER_HANDLE_UNSET,
) -> None:
    """Run normal shutdown, then force/verify cleanup without hiding a primary error."""
    targets = session.spec.get("targets")
    if (
        not isinstance(targets, list)
        or len(targets) != 1
        or not isinstance(targets[0], Mapping)
        or type(targets[0].get("config_path")) is not str
    ):
        raise StrictCapturedReplayEntrypointError(
            "strict scorer cleanup requires one authenticated target"
        )
    config_path = targets[0]["config_path"]
    processes = getattr(run_helper, "_processes", None)
    state_failure: BaseException | None = None
    if type(processes) is not dict or set(processes) - {config_path}:
        state_failure = StrictCapturedReplayEntrypointError(
            "strict scorer cleanup found an unexpected RunHelper process table"
        )
        current_resource_process = None
    else:
        current_resource_process = processes.get(config_path)
    current_head_server = getattr(run_helper, "_head_server", None)
    current_head_server_thread = getattr(run_helper, "_head_server_thread", None)
    resource_process = (
        current_resource_process
        if captured_resource_process is _RUN_HELPER_HANDLE_UNSET
        else captured_resource_process
    )
    head_server = (
        current_head_server
        if captured_head_server is _RUN_HELPER_HANDLE_UNSET
        else captured_head_server
    )
    head_server_thread = (
        current_head_server_thread
        if captured_head_server_thread is _RUN_HELPER_HANDLE_UNSET
        else captured_head_server_thread
    )
    if state_failure is None and (
        (
            current_resource_process is not None
            and current_resource_process is not resource_process
        )
        or (current_head_server is not None and current_head_server is not head_server)
        or (
            current_head_server_thread is not None
            and current_head_server_thread is not head_server_thread
        )
    ):
        state_failure = StrictCapturedReplayEntrypointError(
            "strict scorer cleanup handles differ from their captured identities"
        )
    shutdown_failure: BaseException | None = None
    try:
        run_helper.shutdown()
    except BaseException as error:
        shutdown_failure = error
    try:
        _force_reap_profiled_resource_child(
            run_helper=run_helper,
            resource_process=resource_process,
            head_server=head_server,
            head_server_thread=head_server_thread,
            child_runtime=child_runtime,
            authenticated_process_identity=authenticated_process_identity,
        )
    except BaseException as reap_failure:
        failures: list[BaseException] = []
        if primary_failure is not None:
            failures.append(primary_failure)
        if state_failure is not None:
            failures.append(state_failure)
        if shutdown_failure is not None:
            failures.append(shutdown_failure)
        failures.append(reap_failure)
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup(
            "strict scorer primary/shutdown/reap failures",
            failures,
        ) from None
    if state_failure is not None:
        if primary_failure is None:
            raise state_failure
        raise BaseExceptionGroup(
            "strict scorer primary/cleanup-state failures",
            [primary_failure, state_failure],
        ) from None
    if shutdown_failure is not None:
        if primary_failure is None:
            raise shutdown_failure
        primary_failure.add_note(
            "RunHelper.shutdown also failed; the authenticated forced reap "
            f"completed: {shutdown_failure!r}"
        )


def run_profiled_replay_with_authenticated_resource_child(
    *,
    snapshot_authentication: Mapping[str, Any],
    resource_parser_config: Any,
    manifest: Mapping[str, Any],
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    authenticated_job_id: str,
    driver_process: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    source_transcript_document: Mapping[str, Any],
    source_main_ledger_document: Mapping[str, Any],
    transport_source: "StrictModelTransportReplaySourceV3",
    expected_environment: str,
    expected_profile_id: str,
) -> "ReplayDocumentsV2":
    """Run a profile-selected format-verifier lifecycle after authentication.

    Imports that can execute snapshot or Gym code occur only after the external
    wrapper's closed snapshot/program attestation is checked.  Ray must already
    be initialized so the scoped child hook can wrap only ``RunHelper.start``.
    """
    contract = _required_mapping(manifest.get("replay_contract"), "replay_contract")
    source_snapshot = _required_mapping(
        contract.get("source_snapshot"), "replay_contract.source_snapshot"
    )
    source_snapshot_ref = _required_mapping(
        source_snapshot.get("ref"), "replay_contract.source_snapshot.ref"
    )
    program = _required_mapping(contract.get("program"), "replay_contract.program")
    expected_program_sha256 = _domain_sha256("captured-replay-program", program)
    expected_authentication = {
        "status": "authenticated",
        "replay_execution_manifest_sha256": replay_execution_manifest_sha256,
        "source_snapshot_manifest_sha256": source_snapshot_ref.get("manifest_sha256"),
        "program_sha256": expected_program_sha256,
    }
    profile = _closed_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if (
        manifest.get("environment") != expected_environment
        or manifest.get("scorer_profile") != profile
    ):
        raise StrictCapturedReplayEntrypointError(
            "manifest differs from the explicit format-verifier profile"
        )
    from nemo_rl.algorithms.strict_captured_replay_runtime_v2 import (
        StrictCapturedReplayError,
        post_resource_verify,
    )

    execution_source_root = _canonical_absolute_path(
        source_snapshot_ref.get("path"), "authenticated execution source root"
    )
    runtime_module = sys.modules["nemo_rl.algorithms.strict_captured_replay_runtime_v2"]
    runtime_path = Path(runtime_module.__file__).resolve(strict=True)
    expected_runtime = (
        execution_source_root
        / manifest["replay_contract"]["program"]["runtime"]["path"]
    )
    if (
        runtime_path != expected_runtime
        or _stable_regular_sha256(runtime_path, name="imported replay runtime")
        != manifest["replay_contract"]["program"]["runtime"]["sha256"]
    ):
        raise StrictCapturedReplayError(
            "imported replay runtime differs from authenticated program"
        )

    if dict(snapshot_authentication) != expected_authentication:
        raise StrictCapturedReplayError(
            "replay snapshot/program authentication is incomplete"
        )

    # These imports are deliberately below the authenticated snapshot gate.
    import ray
    from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper

    from nemo_rl.environments import strict_gym_child_runtime_v2 as child_runtime

    _verify_imported_source_module_origins(
        execution_source_root=execution_source_root,
        gym_source_root=execution_source_root / _GYM_SOURCE_RELATIVE,
        expected_snapshot_manifest_sha256=source_snapshot_ref["manifest_sha256"],
    )

    child_runtime_path = Path(child_runtime.__file__).resolve(strict=True)
    expected_child_runtime = (
        execution_source_root
        / manifest["replay_contract"]["program"]["gym_child_runtime"]["path"]
    )
    if (
        child_runtime_path != expected_child_runtime
        or _stable_regular_sha256(
            child_runtime_path, name="imported strict Gym child runtime"
        )
        != manifest["replay_contract"]["program"]["gym_child_runtime"]["sha256"]
    ):
        raise StrictCapturedReplayError(
            "imported strict Gym child runtime differs from authenticated program"
        )
    _verify_imported_program_modules(
        execution_source_root=execution_source_root,
        program=program,
        required_program_names=_DRIVER_RUNTIME_REQUIRED_PROGRAM_NAMES,
    )

    if not ray.is_initialized():
        raise StrictCapturedReplayError(
            "Ray must be initialized before scorer-only child injection"
        )
    if type(resource_parser_config) is not GlobalConfigDictParserConfig:
        raise StrictCapturedReplayError(
            "resource parser config is not the pinned Gym parser type"
        )
    if getattr(child_runtime, "STRICT_GYM_SCORE_FINALIZER_SAFE", False) is not True:
        raise StrictCapturedReplayError(
            "scorer callable freeze/quiesce finalizer is not authenticated SAFE"
        )

    session = child_runtime.prepare_strict_gym_child_runtime(scope="scorer-only")
    scorer = _required_mapping(contract.get("gym_scorer"), "gym_scorer")
    scorer_source = _required_mapping(scorer.get("source"), "gym_scorer.source")
    scorer_source_root = _required_mapping(
        scorer.get("source_root"), "gym_scorer.source_root"
    )
    child_gym = _required_mapping(session.spec.get("gym"), "strict Gym child spec.gym")
    if (
        session.environment != expected_environment
        or session.scope != "scorer-only"
        or child_gym.get("root") != scorer_source_root.get("container_path")
        or child_gym.get("git_commit") != scorer_source.get("gitlink_commit")
        or child_gym.get("tree") != scorer_source.get("tree")
    ):
        raise StrictCapturedReplayError(
            "strict Gym child spec differs from manifest container source root"
        )
    run_helper = RunHelper()
    expected_calls: list[dict[str, Any]] = []
    run_helper_closed = False
    authenticated_process_identity: tuple[int, int] | None = None
    captured_resource_process: Any = _RUN_HELPER_HANDLE_UNSET
    captured_head_server: Any = _RUN_HELPER_HANDLE_UNSET
    captured_head_server_thread: Any = _RUN_HELPER_HANDLE_UNSET
    try:
        with session.launch_environment():
            run_helper.start(global_config_dict_parser_config=resource_parser_config)
        targets = session.spec.get("targets")
        processes = getattr(run_helper, "_processes", None)
        if (
            type(targets) is not list
            or len(targets) != 1
            or type(targets[0]) is not dict
            or type(targets[0].get("config_path")) is not str
            or type(processes) is not dict
            or set(processes) != {targets[0]["config_path"]}
        ):
            raise StrictCapturedReplayError(
                "started scorer has no exact RunHelper process authority"
            )
        captured_resource_process = processes[targets[0]["config_path"]]
        captured_head_server = getattr(run_helper, "_head_server", None)
        captured_head_server_thread = getattr(run_helper, "_head_server_thread", None)
        child_index, child_index_sha256 = session.attest_started(run_helper)
        if (
            child_index.get("scope") != "scorer-only"
            or child_index.get("environment") != expected_environment
            or child_index.get("job_id") != authenticated_job_id
            or type(child_index_sha256) is not str
            or _DIGEST_RE.fullmatch(child_index_sha256) is None
        ):
            raise StrictCapturedReplayError(
                "scorer-only child index does not bind the replay job"
            )
        children = child_index.get("children")
        if (
            not isinstance(children, list)
            or len(children) != 1
            or children[0].get("role") != "resource"
        ):
            raise StrictCapturedReplayError(
                "scorer-only replay launched a non-resource Gym child"
            )
        child = children[0]
        observation = child.get("observation")
        receipt_ref = child.get("receipt")
        if not isinstance(observation, Mapping) or not isinstance(receipt_ref, Mapping):
            raise StrictCapturedReplayError("resource child index is malformed")
        process_pid = observation.get("pid")
        process_start_ticks = observation.get("start_ticks")
        if (
            type(process_pid) is not int
            or process_pid <= 1
            or type(process_start_ticks) is not int
            or process_start_ticks <= 0
            or observation.get("wrapper_pid") != process_pid
        ):
            raise StrictCapturedReplayError(
                "resource child index has no exact direct-process identity"
            )
        authenticated_process_identity = (process_pid, process_start_ticks)
        _, resource_receipt_bytes = child_runtime._load_canonical_document(
            Path(receipt_ref["path"])
        )

        if (
            set(receipt_ref) != {"path", "schema", "sha256"}
            or receipt_ref.get("schema") != "nemo-rl-strict-gym-child-receipt-v1"
            or receipt_ref.get("sha256")
            != hashlib.sha256(resource_receipt_bytes).hexdigest()
        ):
            raise StrictCapturedReplayError("resource child receipt reference differs")

        def post_verifier(
            rollout_index: int,
            generation_seed: int,
            request: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del rollout_index, generation_seed
            return post_resource_verify(
                host=observation["host"],
                port=observation["port"],
                derived_verifier_request=request,
            )

        def independent_format_check(
            rollout_index: int,
            request: Mapping[str, Any],
            response: Mapping[str, Any],
        ) -> None:
            if rollout_index != len(expected_calls):
                raise StrictCapturedReplayError(
                    "format-verifier calls are not in logical rollout order"
                )
            expectation = child_runtime.format_verification_call_expectation(
                environment=expected_environment,
                derived_verifier_request=request,
                verifier_response=response,
            )
            if expectation.get("profile_id") != expected_profile_id:
                raise StrictCapturedReplayError(
                    "independent format expectation differs from selected profile"
                )
            expected_calls.append(expectation)

        def finalize_format_call_evidence() -> Mapping[str, Any]:
            nonlocal run_helper_closed
            terminal, digest = session.finalize_format_verification_calls(
                expected_calls,
                run_helper=run_helper,
            )
            run_helper_closed = True
            declared = _required_mapping(
                _required_mapping(
                    _required_mapping(manifest.get("artifacts"), "artifacts").get(
                        "outputs"
                    ),
                    "artifacts.outputs",
                ).get("scorer_call_index"),
                "artifacts.outputs.scorer_call_index",
            )
            expected_path = session.receipt_root / "format-verification-call-index.json"
            if (
                set(declared) != {"path", "schema", "framing", "mode"}
                or declared.get("path") != str(expected_path)
                or declared.get("schema") != profile["call_index_schema"]
                or declared.get("framing") != "canonical-ascii-json-no-lf"
                or declared.get("mode") != "0400"
            ):
                raise StrictCapturedReplayError(
                    "format call terminal path is not manifest-declared"
                )
            if (
                terminal.get("environment") != expected_environment
                or terminal.get("profile_id") != expected_profile_id
                or type(terminal.get("call_count")) is not int
                or terminal.get("call_count") != 4
            ):
                raise StrictCapturedReplayError(
                    "format call terminal does not close the selected K=4 profile"
                )
            return {
                "path": str(expected_path),
                "schema": profile["call_index_schema"],
                "sha256": digest,
            }

        return run_from_authenticated_wrapper(
            manifest=manifest,
            replay_execution_manifest_sha256=replay_execution_manifest_sha256,
            submission_receipt_sha256=submission_receipt_sha256,
            authenticated_job_id=authenticated_job_id,
            driver_process=driver_process,
            driver_scheduler_device_environment=driver_scheduler_device_environment,
            source_transcript_document=source_transcript_document,
            source_main_ledger_document=source_main_ledger_document,
            transport_source=transport_source,
            post_verifier=post_verifier,
            independent_format_check=independent_format_check,
            finalize_format_call_evidence=finalize_format_call_evidence,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
    finally:
        if not run_helper_closed:
            _shutdown_profiled_resource_child(
                run_helper=run_helper,
                session=session,
                child_runtime=child_runtime,
                authenticated_process_identity=authenticated_process_identity,
                primary_failure=sys.exc_info()[1],
                captured_resource_process=captured_resource_process,
                captured_head_server=captured_head_server,
                captured_head_server_thread=captured_head_server_thread,
            )


def _run_authenticated_driver(authority: _BootstrapAuthority) -> None:
    """Load the authenticated source, execute K=4, and publish declared outputs."""
    _require_isolated_driver_process()
    refreshed_authentication = _authenticate_snapshot_program(
        manifest=authority.manifest,
        manifest_sha256=authority.manifest_sha256,
        pair_manifest=authority.pair_manifest,
        execution_source_root=authority.execution_source_root,
    )
    if refreshed_authentication != authority.snapshot_authentication:
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot changed before repository activation"
        )
    _assert_wandb_disabled(authority.manifest)
    gym_source_root = _activate_authenticated_source_roots(
        authority.execution_source_root
    )

    # Every repository import is intentionally below the stdlib-only gate.
    import ray

    from nemo_rl.utils.strict_captured_replay_evidence_v2 import (
        canonical_ascii_json,
        load_captured_replay_submission_receipt_v2,
        publish_evidence_document,
        validate_captured_replay_step1_ledger,
        validate_transcript_bundle,
    )
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        load_authenticated_off_source_capture,
        load_replay_execution_manifest_v2,
    )
    from nemo_rl.utils.strict_model_transport_replay_v3 import (
        load_strict_model_transport_replay_source_v3,
        publish_strict_model_transport_replay_consumption_v3,
        validate_strict_model_transport_replay_consumption_v3,
    )

    _assert_wandb_disabled(authority.manifest)
    _verify_imported_source_module_origins(
        execution_source_root=authority.execution_source_root,
        gym_source_root=gym_source_root,
        expected_snapshot_manifest_sha256=authority.snapshot_authentication[
            "source_snapshot_manifest_sha256"
        ],
    )
    _verify_imported_program_modules(
        execution_source_root=authority.execution_source_root,
        program=authority.manifest["replay_contract"]["program"],
        required_program_names=_DRIVER_LOADER_REQUIRED_PROGRAM_NAMES,
    )
    pair_ref = authority.manifest["pair"]["manifest"]
    pair_submission_ref = authority.manifest["pair"]["submission_receipt"]
    trusted_off_exit_ref = authority.manifest["source_capture"]["job_receipts"]["exit"]
    replay_submission_ref = authority.pre_receipt["submission_receipt"]
    expected_environment = authority.manifest["environment"]
    expected_profile_id = authority.manifest["scorer_profile"]["profile_id"]
    source = load_authenticated_off_source_capture(
        pair_manifest=authority.pair_manifest,
        pair_manifest_path=pair_ref["path"],
        pair_manifest_sha256=pair_ref["sha256"],
        pair_submission_receipt_path=pair_submission_ref["path"],
        pair_submission_receipt_sha256=pair_submission_ref["sha256"],
        trusted_off_exit_receipt_path=trusted_off_exit_ref["path"],
        trusted_off_exit_receipt_sha256=trusted_off_exit_ref["sha256"],
    )
    replay_submission, replay_submission_sha256 = (
        load_captured_replay_submission_receipt_v2(
            path=replay_submission_ref["path"],
            expected_sha256=replay_submission_ref["sha256"],
            replay_execution_manifest=authority.manifest,
            authenticated_source=source,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
    )
    if (
        replay_submission_sha256 != replay_submission_ref["sha256"]
        or replay_submission.get("candidate_job_id") != authority.authenticated_job_id
    ):
        raise StrictCapturedReplayEntrypointError(
            "repository submission validation changed the PRE authority"
        )
    manifest, manifest_sha256 = load_replay_execution_manifest_v2(
        path=authority.manifest_path,
        expected_sha256=authority.manifest_sha256,
        authenticated_source=source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if manifest != authority.manifest or manifest_sha256 != authority.manifest_sha256:
        raise StrictCapturedReplayEntrypointError(
            "repository manifest validation changed the bootstrap authority"
        )
    _verify_imported_program_modules(
        execution_source_root=authority.execution_source_root,
        program=authority.manifest["replay_contract"]["program"],
        required_program_names=_DRIVER_PROFILE_REQUIRED_PROGRAM_NAMES,
    )
    transport_source = load_strict_model_transport_replay_source_v3(
        source=source,
        replay_execution_manifest=manifest,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    output_root = Path(manifest["artifacts"]["outputs"]["directory"]["path"])
    _create_exclusive_output_root(output_root)
    driver_process = _observe_driver_process()
    driver_scheduler_device_environment = _live_scheduler_device_environment()
    if driver_scheduler_device_environment != authority.scheduler_device_environment:
        raise StrictCapturedReplayEntrypointError(
            "driver scheduler device environment changed after repository activation"
        )
    ray_started = False
    try:
        if ray.is_initialized():
            raise StrictCapturedReplayEntrypointError(
                "replay driver inherited an initialized Ray runtime"
            )
        ray.init(ignore_reinit_error=False, include_dashboard=False)
        ray_started = True
        parser_config = _build_format_resource_only_parser_config(
            manifest=manifest,
            execution_source_root=authority.execution_source_root,
            ray_module=ray,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
        _verify_imported_source_module_origins(
            execution_source_root=authority.execution_source_root,
            gym_source_root=gym_source_root,
            expected_snapshot_manifest_sha256=authority.snapshot_authentication[
                "source_snapshot_manifest_sha256"
            ],
        )
        documents = run_profiled_replay_with_authenticated_resource_child(
            snapshot_authentication=authority.snapshot_authentication,
            resource_parser_config=parser_config,
            manifest=manifest,
            replay_execution_manifest_sha256=authority.manifest_sha256,
            submission_receipt_sha256=_submission_receipt_sha256(authority.pre_receipt),
            authenticated_job_id=authority.authenticated_job_id,
            driver_process=driver_process,
            driver_scheduler_device_environment=driver_scheduler_device_environment,
            source_transcript_document=source.transcript_bundle,
            source_main_ledger_document=source.main_ledger,
            transport_source=transport_source,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
        _verify_imported_program_modules(
            execution_source_root=authority.execution_source_root,
            program=manifest["replay_contract"]["program"],
            required_program_names=_DRIVER_RUNTIME_REQUIRED_PROGRAM_NAMES,
        )
        _assert_wandb_disabled(manifest)
        if _live_scheduler_device_environment() != driver_scheduler_device_environment:
            raise StrictCapturedReplayEntrypointError(
                "driver scheduler device environment changed during replay"
            )

        # Validate the entire terminal set before publishing any runner-owned
        # document.  The child-owned score index is already immutable and
        # offline at this point; transport finalization has reloaded it.
        validate_transcript_bundle(documents.transcript_bundle)
        validate_captured_replay_step1_ledger(
            documents.replay_ledger,
            source_transcript_document=source.transcript_bundle,
            transcript_document=documents.transcript_bundle,
        )
        validate_strict_model_transport_replay_consumption_v3(
            documents.transport_consumption,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
        canonical_ascii_json(documents.scorer_call_index)
        _publish_runtime_documents(
            manifest=manifest,
            documents=documents,
            publish_evidence_document=publish_evidence_document,
            publish_transport=publish_strict_model_transport_replay_consumption_v3,
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
    finally:
        if ray_started and ray.is_initialized():
            ray.shutdown()


def _build_format_resource_only_parser_config(
    *,
    manifest: Mapping[str, Any],
    execution_source_root: Path,
    ray_module: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> Any:
    """Reduce one authenticated format YAML to one resource and no agents."""
    from nemo_gym import PARENT_DIR
    from nemo_gym.cli import GlobalConfigDictParserConfig
    from nemo_gym.global_config import (
        NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
        get_global_config_dict,
        maybe_get_global_config_dict,
    )
    from omegaconf import DictConfig

    profile = _closed_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if manifest.get("scorer_profile") != profile:
        raise StrictCapturedReplayEntrypointError(
            "resource parser profile differs from authenticated manifest"
        )
    contract = _required_mapping(manifest.get("replay_contract"), "replay_contract")
    scorer = _required_mapping(contract.get("gym_scorer"), "gym_scorer")
    launcher = _required_mapping(scorer.get("launcher"), "gym_scorer.launcher")
    if (
        launcher.get("log_wrapper") != "forbidden"
        or "log_wrapper_template" in launcher
        or launcher.get("resource_only_config") is not None
        or launcher.get("config_path_name") != profile["resource_config_path_name"]
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay scorer must disable the Gym Bash/tee log wrapper"
        )
    resources = _required_mapping(scorer.get("resources"), "gym_scorer.resources")
    config_reference = _required_mapping(
        resources.get("config"),
        "gym_scorer.resources.config",
    )
    if (
        set(config_reference) != {"path", "sha256"}
        or dict(config_reference) != profile["resource_config"]
    ):
        raise StrictCapturedReplayEntrypointError(
            "selected format Gym config differs from closed profile"
        )
    relative = profile["resource_config"]["path"]
    expected_digest = _digest(
        profile["resource_config"]["sha256"],
        "selected format Gym config SHA-256",
    )
    gym_root = execution_source_root / _GYM_SOURCE_RELATIVE
    imported_gym_root = Path(PARENT_DIR).resolve(strict=True)
    if imported_gym_root != gym_root:
        raise StrictCapturedReplayEntrypointError(
            "imported NeMo-Gym root differs from authenticated execution source"
        )
    config_path = gym_root / relative
    if (
        _stable_regular_sha256(config_path, name="selected format Gym config")
        != expected_digest
    ):
        raise StrictCapturedReplayEntrypointError(
            "selected format Gym config differs from replay manifest"
        )
    if maybe_get_global_config_dict() is not None:
        raise StrictCapturedReplayEntrypointError(
            "NeMo-Gym config was initialized before format-profile reduction"
        )

    context = ray_module.get_runtime_context()
    gcs_address = getattr(context, "gcs_address", None)
    if type(gcs_address) is not str or not gcs_address:
        raise StrictCapturedReplayEntrypointError(
            "initialized Ray runtime has no GCS address"
        )
    attempt_environment = manifest["execution_environment"]["attempt"]
    output_root = manifest["artifacts"]["outputs"]["directory"]["path"]
    initial = DictConfig(
        {
            "config_paths": [str(config_path)],
            profile["disabled_config_path_name"]: {
                "_delete_key": "responses_api_agents",
            },
            "default_host": "127.0.0.1",
            "head_server": {"host": "127.0.0.1", "port": 11000},
            "port_range_low": 5000,
            "port_range_high": 5999,
            "ray_head_node_address": gcs_address,
            "skip_venv_if_present": True,
            "model_endpoint_readiness_timeout_seconds": 0,
            "results_dir": output_root,
            "cache_dir": attempt_environment["persistent_cache"],
            "uv_cache_dir": f"{attempt_environment['persistent_cache']}/uv",
            "uv_venv_dir": "/opt/gym_venvs",
            "observability_enabled": False,
        }
    )
    parser_config = GlobalConfigDictParserConfig(
        initial_global_config_dict=initial,
        skip_load_from_cli=True,
        skip_load_from_dotenv=True,
    )
    resolved = get_global_config_dict(global_config_dict_parser_config=parser_config)
    if (
        _stable_regular_sha256(
            config_path, name="post-parse selected format Gym config"
        )
        != expected_digest
    ):
        raise StrictCapturedReplayEntrypointError(
            "selected format Gym config changed while Gym parsed it"
        )
    if "nemo_gym_log_dir" in resolved:
        raise StrictCapturedReplayEntrypointError(
            "resource-only replay unexpectedly enabled the Gym log wrapper"
        )
    if list(resolved.get("config_paths", [])) != [str(config_path)]:
        raise StrictCapturedReplayEntrypointError(
            "resolved Gym config path roster differs from authenticated YAML"
        )
    nonreserved = {
        str(name): value
        for name, value in resolved.items()
        if name not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    }
    admitted_names = {
        profile["resource_config_path_name"],
        profile["disabled_config_path_name"],
    }
    if set(nonreserved) != admitted_names:
        raise StrictCapturedReplayEntrypointError(
            "reduced Gym config top-level roster differs"
        )
    disabled = nonreserved[profile["disabled_config_path_name"]]
    resource = nonreserved[profile["resource_config_path_name"]]
    if not isinstance(disabled, Mapping) or dict(disabled) != {}:
        raise StrictCapturedReplayEntrypointError(
            "selected simple-agent branch survived _delete_key reduction"
        )
    if not isinstance(resource, Mapping) or set(resource) != {"resources_servers"}:
        raise StrictCapturedReplayEntrypointError(
            "reduced Gym config contains a non-resource server type"
        )
    by_name = resource["resources_servers"]
    if (
        not isinstance(by_name, Mapping)
        or set(by_name) != {"format_verification"}
        or not isinstance(by_name["format_verification"], Mapping)
        or by_name["format_verification"].get("entrypoint") != "app.py"
    ):
        raise StrictCapturedReplayEntrypointError(
            "reduced Gym config resource roster differs"
        )
    if _contains_forbidden_server_type(resolved):
        raise StrictCapturedReplayEntrypointError(
            "reduced Gym config contains an agent or model server"
        )
    return parser_config


def _publish_runtime_documents(
    *,
    manifest: Mapping[str, Any],
    documents: Any,
    publish_evidence_document: Any,
    publish_transport: Any,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    profile = _closed_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    outputs = manifest["artifacts"]["outputs"]
    score_ref = documents.scorer_call_index
    declared_score = outputs["scorer_call_index"]
    if (
        set(declared_score) != {"path", "schema", "framing", "mode"}
        or declared_score.get("schema") != profile["call_index_schema"]
        or declared_score.get("framing") != "canonical-ascii-json-no-lf"
        or declared_score.get("mode") != "0400"
        or score_ref
        != {
            "path": declared_score["path"],
            "schema": declared_score["schema"],
            "sha256": score_ref.get("sha256"),
        }
        or _DIGEST_RE.fullmatch(str(score_ref.get("sha256", ""))) is None
    ):
        raise StrictCapturedReplayEntrypointError(
            "child-owned scorer-call terminal differs from declared output"
        )
    plan = (
        ("transcript_bundle", documents.transcript_bundle),
        ("replay_ledger", documents.replay_ledger),
    )
    for name, document in plan:
        declaration = outputs[name]
        if document.get("schema") != declaration["schema"]:
            raise StrictCapturedReplayEntrypointError(
                f"{name} schema differs from manifest declaration"
            )
        publish_evidence_document(
            output=declaration["path"], document=document, trailing_lf=False
        )
    consumption = outputs["transport_consumption"]
    if documents.transport_consumption.get("schema") != consumption["schema"]:
        raise StrictCapturedReplayEntrypointError(
            "transport consumption schema differs from manifest declaration"
        )
    publish_transport(
        output=consumption["path"],
        document=documents.transport_consumption,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )


def _observe_driver_process() -> dict[str, Any]:
    boot_raw = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    try:
        boot_id = boot_raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise StrictCapturedReplayEntrypointError(
            "driver boot identity is not ASCII"
        ) from error
    if (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
    ):
        raise StrictCapturedReplayEntrypointError("driver boot identity is malformed")
    stat_text = Path("/proc/self/stat").read_text(encoding="ascii")
    closing = stat_text.rfind(") ")
    fields = stat_text[closing + 2 :].split() if closing >= 0 else []
    if len(fields) <= 19:
        raise StrictCapturedReplayEntrypointError("driver process stat is malformed")
    start_ticks = int(fields[19], 10)
    if not 0 < os.getpid() <= (1 << 31) - 1 or not 0 < start_ticks <= _MAX_INT63:
        raise StrictCapturedReplayEntrypointError("driver process identity is invalid")
    return {
        "boot_id_sha256": hashlib.sha256(boot_raw).hexdigest(),
        "pid": os.getpid(),
        "start_time_ticks": start_ticks,
    }


def _authenticate_before_repo_imports(
    *,
    manifest_path: str,
    manifest_sha256: str,
    pre_receipt_path: str,
    pre_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> _BootstrapAuthority:
    """Close wrapper authority using only stdlib and authenticated bytes."""
    if any(name == "nemo_rl" or name.startswith("nemo_rl.") for name in sys.modules):
        raise StrictCapturedReplayEntrypointError(
            "NeMo-RL was imported before replay snapshot authentication"
        )
    if any(name == "nemo_gym" or name.startswith("nemo_gym.") for name in sys.modules):
        raise StrictCapturedReplayEntrypointError(
            "NeMo-Gym was imported before replay snapshot authentication"
        )
    manifest_digest = _digest(manifest_sha256, "replay manifest SHA-256")
    pre_digest = _digest(pre_receipt_sha256, "PRE receipt SHA-256")
    manifest, _ = _load_canonical_document(
        manifest_path,
        expected_sha256=manifest_digest,
        trailing_lf=False,
        name="replay execution manifest",
    )
    if set(manifest) != _MANIFEST_ROOT_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "replay execution manifest root keyset differs"
        )
    profile = _closed_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("hash_domain") != _MANIFEST_HASH_DOMAIN
        or manifest.get("environment") != expected_environment
        or manifest.get("scorer_profile") != profile
        or manifest.get("arm") != "on"
        or manifest.get("mode") != "fresh_verifier_reward_replay"
        or manifest.get("attempt_id") not in {"replay-1", "replay-2"}
        or type(manifest.get("pair_id")) is not str
        or _PAIR_ID_RE.fullmatch(manifest["pair_id"]) is None
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay execution manifest is not the selected format-profile envelope"
        )
    if manifest.get("wandb") != _DISABLED_WANDB_POLICY:
        raise StrictCapturedReplayEntrypointError(
            "replay execution manifest does not disable W&B"
        )
    contract = _required_mapping(manifest.get("replay_contract"), "replay_contract")
    if contract.get("execution_scope") != "scorer-only" or contract.get(
        "policy_execution"
    ) != {
        "backward": False,
        "forward": False,
        "optimizer": False,
        "violation": "fail-closed",
    }:
        raise StrictCapturedReplayEntrypointError(
            "replay manifest permits policy/model execution"
        )

    pair_manifest = _authenticate_pair_authority(manifest=manifest)
    pre_receipt, _ = _load_canonical_document(
        pre_receipt_path,
        expected_sha256=pre_digest,
        trailing_lf=True,
        name="authenticated replay PRE receipt",
    )
    authenticated_job_id, execution_source_root = _authenticate_pre_receipt(
        pre_receipt=pre_receipt,
        pre_receipt_path=_canonical_absolute_path(pre_receipt_path, "PRE receipt path"),
        manifest=manifest,
        manifest_path=_canonical_absolute_path(manifest_path, "manifest path"),
        manifest_sha256=manifest_digest,
        pair_manifest=pair_manifest,
    )
    scheduler_device_environment = _validate_runtime_environment(
        manifest=manifest,
        pair_manifest=pair_manifest,
        authenticated_job_id=authenticated_job_id,
    )
    snapshot_authentication = _authenticate_snapshot_program(
        manifest=manifest,
        manifest_sha256=manifest_digest,
        pair_manifest=pair_manifest,
        execution_source_root=execution_source_root,
    )
    return _BootstrapAuthority(
        manifest=manifest,
        manifest_path=_canonical_absolute_path(manifest_path, "manifest path"),
        manifest_sha256=manifest_digest,
        pre_receipt=pre_receipt,
        pre_receipt_path=_canonical_absolute_path(pre_receipt_path, "PRE receipt path"),
        pre_receipt_sha256=pre_digest,
        pair_manifest=pair_manifest,
        execution_source_root=execution_source_root,
        authenticated_job_id=authenticated_job_id,
        scheduler_device_environment=scheduler_device_environment,
        snapshot_authentication=snapshot_authentication,
    )


def _authenticate_pair_authority(*, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Stable-load the released Pair roots before trusting its ON snapshot."""
    pair_binding = _required_mapping(manifest.get("pair"), "pair")
    pair_ref = _required_mapping(pair_binding.get("manifest"), "pair.manifest")
    submission_ref = _required_mapping(
        pair_binding.get("submission_receipt"), "pair.submission_receipt"
    )
    if (
        set(pair_ref) != {"path", "schema", "sha256"}
        or pair_ref.get("schema") != _PAIR_SCHEMA
    ):
        raise StrictCapturedReplayEntrypointError("Pair manifest reference differs")
    if (
        set(submission_ref) != {"path", "schema", "sha256"}
        or submission_ref.get("schema") != _PAIR_SUBMISSION_SCHEMA
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair submission receipt reference differs"
        )
    pair, _ = _load_canonical_document(
        pair_ref["path"],
        expected_sha256=pair_ref["sha256"],
        trailing_lf=True,
        name="Pair manifest",
    )
    if set(pair) != _PAIR_ROOT_KEYS:
        raise StrictCapturedReplayEntrypointError("Pair manifest root keyset differs")
    selection = _required_mapping(pair.get("selection"), "Pair selection")
    paths = _required_mapping(pair.get("paths"), "Pair paths")
    results_root = _canonical_absolute_path(
        paths.get("results_root"), "Pair results root"
    )
    if (
        pair.get("schema") != _PAIR_SCHEMA
        or pair.get("pair_id") != manifest["pair_id"]
        or pair_binding.get("id") != manifest["pair_id"]
        or pair_binding.get("environment") != manifest["environment"]
        or selection.get("environment") != manifest["environment"]
        or Path(pair_ref["path"]) != results_root / "PAIR_MANIFEST.json"
        or Path(submission_ref["path"]) != results_root / "PAIR_SUBMISSION_RECEIPT.json"
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair manifest identity/path differs before repository import"
        )
    released, _ = _load_canonical_document(
        submission_ref["path"],
        expected_sha256=submission_ref["sha256"],
        trailing_lf=True,
        name="released Pair submission receipt",
    )
    if set(released) != _PAIR_SUBMISSION_ROOT_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "released Pair submission receipt root keyset differs"
        )
    if (
        released.get("schema") != _PAIR_SUBMISSION_SCHEMA
        or released.get("outcome") != "released"
        or released.get("stage") != "complete"
        or released.get("rollback_confirmed") is not None
        or released.get("cancellations") != []
        or released.get("pre_cancel_queries") != []
        or released.get("post_cancel_queries") != []
        or released.get("recovery_query") is not None
        or released.get("pair")
        != {
            "id": manifest["pair_id"],
            "manifest": {
                "path": pair_ref["path"],
                "sha256": pair_ref["sha256"],
            },
        }
        or released.get("receipt")
        != {"path": submission_ref["path"], "schema": _PAIR_SUBMISSION_SCHEMA}
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair submission receipt is not one released authenticated cohort"
        )
    return pair


def _authenticate_pre_receipt(
    *,
    pre_receipt: Mapping[str, Any],
    pre_receipt_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    pair_manifest: Mapping[str, Any],
) -> tuple[str, Path]:
    """Validate the frozen wrapper PRE projection consumed by the driver.

    The wrapper owns the full scheduler query.  The driver independently binds
    the resulting authenticated identity, exact source mount, submission
    receipt, and manifest.  Exact root keys are frozen with the shell schema.
    """
    if set(pre_receipt) != _PRE_ROOT_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "replay PRE receipt root keyset differs"
        )
    authenticated_job_id = _job_id(
        pre_receipt.get("authenticated_job_id"), "PRE authenticated job ID"
    )
    candidate_job_id = _job_id(
        pre_receipt.get("candidate_job_id"), "PRE candidate job ID"
    )
    if (
        pre_receipt.get("schema") != _PRE_SCHEMA
        or pre_receipt.get("scorer_profile") != manifest["scorer_profile"]
        or pre_receipt.get("phase") != "PRE"
        or pre_receipt.get("status") != "authenticated-pre"
        or pre_receipt.get("pair_id") != manifest["pair_id"]
        or pre_receipt.get("environment") != manifest["environment"]
        or pre_receipt.get("arm") != "on"
        or pre_receipt.get("mode") != "fresh_verifier_reward_replay"
        or pre_receipt.get("attempt_id") != manifest["attempt_id"]
        or candidate_job_id != authenticated_job_id
        or pre_receipt.get("post_verified") is not False
    ):
        raise StrictCapturedReplayEntrypointError("replay PRE receipt identity differs")
    if pre_receipt.get("replay_execution_manifest") != {
        "path": str(manifest_path),
        "schema": _MANIFEST_SCHEMA,
        "sha256": manifest_sha256,
    }:
        raise StrictCapturedReplayEntrypointError(
            "replay PRE receipt does not bind the execution manifest"
        )
    submission = _required_mapping(
        pre_receipt.get("submission_receipt"), "PRE submission receipt"
    )
    declared_submission = manifest["scheduler_submission"]["receipt"]
    if (
        set(submission) != {"path", "schema", "sha256"}
        or submission.get("path") != declared_submission["path"]
        or submission.get("schema") != _REPLAY_SUBMISSION_SCHEMA
        or submission.get("schema") != declared_submission["schema"]
        or _DIGEST_RE.fullmatch(str(submission.get("sha256", ""))) is None
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay PRE submission receipt reference differs"
        )
    execution_source_root = _canonical_absolute_path(
        pre_receipt.get("execution_source_root"), "PRE execution source root"
    )
    source_snapshot = _required_mapping(
        _required_mapping(manifest.get("replay_contract"), "replay_contract").get(
            "source_snapshot"
        ),
        "replay_contract.source_snapshot",
    )
    source_ref = _required_mapping(
        source_snapshot.get("ref"), "replay_contract.source_snapshot.ref"
    )
    if source_snapshot.get(
        "arm"
    ) != "on" or execution_source_root != _canonical_absolute_path(
        source_ref.get("path"), "Pair ON snapshot path"
    ):
        raise StrictCapturedReplayEntrypointError(
            "PRE execution source root differs from Pair ON snapshot"
        )
    static_boundary = _required_mapping(
        pre_receipt.get("static_boundary"), "PRE static boundary"
    )
    if set(static_boundary) != _PRE_STATIC_BOUNDARY_KEYS or dict(static_boundary) != {
        name: manifest[name] for name in _PRE_STATIC_BOUNDARY_KEYS
    }:
        raise StrictCapturedReplayEntrypointError(
            "PRE static boundary differs from replay manifest"
        )
    if pre_receipt.get("runtime_attestation_contract") != manifest.get(
        "runtime_attestation_requirements"
    ):
        raise StrictCapturedReplayEntrypointError(
            "PRE runtime attestation contract differs from replay manifest"
        )
    output_root = _canonical_absolute_path(
        manifest["artifacts"]["outputs"]["directory"]["path"],
        "replay output root",
    )
    if pre_receipt.get("output_precondition") != {
        "path": str(output_root),
        "mode": "0700",
        "status": "absent",
    }:
        raise StrictCapturedReplayEntrypointError(
            "PRE output precondition differs from manifest"
        )
    if output_root.exists() or output_root.is_symlink():
        raise StrictCapturedReplayEntrypointError(
            "replay output root violates authenticated absent precondition"
        )
    driver = _required_mapping(pre_receipt.get("driver"), "PRE driver")
    expected_driver = {
        "entrypoint": _PROGRAM_PATHS["entrypoint"],
        "invocation": "python-isolated-no-bytecode",
        "pre_receipt_path": str(pre_receipt_path),
    }
    if driver != expected_driver:
        raise StrictCapturedReplayEntrypointError(
            "replay PRE driver invocation differs"
        )
    job = _required_mapping(pre_receipt.get("job"), "PRE job")
    slurm = _required_mapping(
        _required_mapping(pair_manifest.get("campaign"), "Pair campaign").get("slurm"),
        "Pair campaign.slurm",
    )
    expected_job = {
        "account": slurm.get("account"),
        "name": manifest["scheduler_submission"]["identity"]["job_name"],
        "num_nodes": pair_manifest["campaign"].get("nodes"),
        "partition": slurm.get("partition"),
        "qos": slurm.get("qos"),
        "gpus_per_node": 4,
        "restart_count": 0,
    }
    if (
        set(job) != set(expected_job)
        or dict(job) != expected_job
        or any(
            type(job[name]) is not expected_type
            for name, expected_type in {
                "account": str,
                "name": str,
                "num_nodes": int,
                "partition": str,
                "qos": str,
                "gpus_per_node": int,
                "restart_count": int,
            }.items()
        )
    ):
        raise StrictCapturedReplayEntrypointError("PRE job contract differs")
    pre_query = _authenticate_pre_scheduler_query(
        reference=pre_receipt.get("pre_scheduler_query"),
        manifest=manifest,
        pair_manifest=pair_manifest,
        authenticated_job_id=authenticated_job_id,
        execution_source_root=execution_source_root,
        pre_receipt_path=pre_receipt_path,
    )
    submission_comment = _authenticate_replay_submission_receipt(
        reference=submission,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        candidate_job_id=candidate_job_id,
    )
    if pre_query["records"][0]["comment"] != submission_comment:
        raise StrictCapturedReplayEntrypointError(
            "PRE scheduler comment differs from authenticated submission"
        )
    return authenticated_job_id, execution_source_root


def _submission_receipt_sha256(pre_receipt: Mapping[str, Any]) -> str:
    submission = _required_mapping(
        pre_receipt.get("submission_receipt"), "PRE submission receipt"
    )
    return _digest(submission.get("sha256"), "submission receipt SHA-256")


def _authenticate_pre_scheduler_query(
    *,
    reference: Any,
    manifest: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    authenticated_job_id: str,
    execution_source_root: Path,
    pre_receipt_path: Path,
) -> dict[str, Any]:
    results_root = _canonical_absolute_path(
        pair_manifest["paths"]["results_root"], "Pair results root"
    )
    job_root = (
        results_root
        / "captured_replay/replay_job_state"
        / manifest["pair_id"]
        / manifest["attempt_id"]
        / f"{authenticated_job_id}-0"
    )
    if pre_receipt_path != job_root / "receipts/PRE.json":
        raise StrictCapturedReplayEntrypointError(
            "PRE receipt path differs from authenticated job root"
        )
    identity = manifest["scheduler_submission"]["identity"]
    # The manifest cannot contain its own SHA.  The exact expanded comment is
    # recovered from the replay submission receipt below; here all independent
    # scheduler fields are closed and that final comment join is checked there.
    return _authenticate_scheduler_query(
        reference=reference,
        phase="PRE",
        expected_document_path=job_root / "queries/PRE.scontrol-query.json",
        expected_raw_path=job_root / "queries/PRE.scontrol.raw",
        job_id=authenticated_job_id,
        job_name=identity["job_name"],
        comment=None,
        user_id=str(identity["submitter_euid"]),
        work_dir=str(execution_source_root),
        job_state="RUNNING",
        held=False,
        reason=None,
        scontrol_path=manifest["runtime_tools"]["document"]["host"]["scontrol"]["path"],
    )


def _authenticate_scheduler_query(
    *,
    reference: Any,
    phase: str,
    expected_document_path: Path,
    expected_raw_path: Path,
    job_id: str,
    job_name: str,
    comment: str | None,
    user_id: str,
    work_dir: str,
    job_state: str,
    held: bool,
    reason: str | None,
    scontrol_path: str,
) -> dict[str, Any]:
    ref = _required_mapping(reference, f"{phase} scheduler query reference")
    if (
        set(ref) != {"path", "schema", "sha256"}
        or ref.get("path") != str(expected_document_path)
        or ref.get("schema") != _SCHEDULER_QUERY_SCHEMA
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler query reference differs"
        )
    query, _ = _load_canonical_document(
        ref["path"],
        expected_sha256=ref["sha256"],
        trailing_lf=True,
        name=f"{phase} scheduler query",
    )
    if set(query) != _SCHEDULER_QUERY_KEYS:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler query keyset differs"
        )
    records = query.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler query is not singleton"
        )
    record = _required_mapping(records[0], f"{phase} scheduler record")
    if set(record) != _SCHEDULER_RECORD_KEYS:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler record keyset differs"
        )
    authenticated_scontrol_path = _canonical_absolute_path(
        scontrol_path, "authenticated scontrol path"
    )
    if (
        query.get("schema") != _SCHEDULER_QUERY_SCHEMA
        or query.get("phase") != phase
        or query.get("argv")
        != [str(authenticated_scontrol_path), "show", "job", "--json", job_id]
        or query.get("path") != str(expected_raw_path)
        or type(query.get("status")) is not int
        or query.get("status") != 0
        or query.get("normalization")
        != {
            "algorithm": "scontrol-show-job-json-v1",
            "complete": True,
            "duplicate_keys_rejected": True,
            "nonfinite_numbers_rejected": True,
            "negative_zero_rejected": True,
        }
        or type(query.get("line_count")) is not int
        or query.get("line_count") < 1
        or type(query.get("match_count")) is not int
        or query.get("match_count") != 1
        or type(query.get("byte_count")) is not int
        or query.get("byte_count") <= 1
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler query normalization differs"
        )
    raw = _stable_regular_bytes(
        expected_raw_path,
        name=f"{phase} scheduler raw output",
        expected_mode=0o400,
        maximum=1 << 20,
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != _digest(query.get("sha256"), f"{phase} scheduler raw SHA-256")
        or len(raw) != query["byte_count"]
        or not raw.endswith(b"\n")
        or b"\r" in raw
        or b"\x00" in raw
        or raw.count(b"\n") != query["line_count"]
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw framing/digest differs"
        )
    source_job = _strict_scheduler_source_job(raw, phase=phase)
    normalized_record = _normalize_scheduler_source_job(source_job, phase=phase)
    expected_record = {
        "job_id": job_id,
        "job_name": job_name,
        "comment": normalized_record["comment"] if comment is None else comment,
        "user_id": user_id,
        "work_dir": work_dir,
        "job_state": job_state,
        "reason": normalized_record["reason"] if reason is None else reason,
        "held": held,
        "restart_count": 0,
    }
    if normalized_record != expected_record or dict(record) != expected_record:
        raise StrictCapturedReplayEntrypointError(f"{phase} scheduler identity differs")
    if (
        not normalized_record["comment"]
        or not normalized_record["reason"]
        or (phase in {"PRE", "POST"} and normalized_record["reason"] == "JobHeldUser")
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler state/reason differs"
        )
    return query


def _strict_scheduler_source_job(raw: bytes, *, phase: str) -> Mapping[str, Any]:
    """Parse exact Slurm JSON without trusting its normalized projection."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw output is not UTF-8"
        ) from error

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictCapturedReplayEntrypointError(
                    f"{phase} scheduler raw JSON contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON contains non-finite constant {value!r}"
        )

    def parse_integer(value: str) -> int:
        if value == "-0":
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw JSON contains negative zero"
            )
        return int(value)

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or (parsed == 0.0 and value.startswith("-")):
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw JSON contains non-finite/negative-zero float"
            )
        return parsed

    try:
        document = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_int=parse_integer,
            parse_float=parse_float,
        )
    except json.JSONDecodeError as error:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw output is not strict JSON"
        ) from error
    if not isinstance(document, Mapping) or set(document) != {
        "errors",
        "jobs",
        "last_backfill",
        "last_update",
        "meta",
        "warnings",
    }:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON root differs"
        )
    if any(
        type(document[field]) is not dict
        for field in ("last_backfill", "last_update", "meta")
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON metadata fields are not exact objects"
        )
    if type(document["errors"]) is not list or document["errors"] != []:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON errors are not empty"
        )
    if type(document["warnings"]) is not list or document["warnings"] != []:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON warnings are not empty"
        )
    jobs = document["jobs"]
    if type(jobs) is not list or len(jobs) != 1 or type(jobs[0]) is not dict:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw JSON is not singleton"
        )
    return jobs[0]


def _normalize_scheduler_source_job(
    source: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
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
    if required - set(source):
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw job omits required fields"
        )
    source_job_id = source["job_id"]
    source_user_id = source["user_id"]
    restart_count = source["restart_cnt"]
    states = source["job_state"]
    if type(source_job_id) is not int or not 1 <= source_job_id <= _MAX_INT63:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw job_id differs"
        )
    if type(source_user_id) is not int or not 0 <= source_user_id <= (1 << 31) - 1:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw user_id differs"
        )
    if type(restart_count) is not int or restart_count != 0:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw restart_cnt differs"
        )
    if type(source["hold"]) is not bool:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw hold is not an exact bool"
        )
    if type(states) is not list or len(states) != 1 or type(states[0]) is not str:
        raise StrictCapturedReplayEntrypointError(
            f"{phase} scheduler raw job_state differs"
        )

    def ascii_value(value: Any, *, name: str, maximum: int) -> str:
        if type(value) is not str or not value:
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw {name} is not a nonempty string"
            )
        try:
            payload = value.encode("ascii")
        except UnicodeEncodeError as error:
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw {name} is not ASCII"
            ) from error
        if len(payload) > maximum:
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw {name} exceeds its bound"
            )
        if any(byte < 0x20 or byte == 0x7F for byte in payload):
            raise StrictCapturedReplayEntrypointError(
                f"{phase} scheduler raw {name} contains a control character"
            )
        return value

    work_dir = ascii_value(
        source["current_working_directory"],
        name="current_working_directory",
        maximum=4096,
    )
    return {
        "job_id": str(source_job_id),
        "job_name": ascii_value(source["name"], name="name", maximum=255),
        "comment": ascii_value(source["comment"], name="comment", maximum=4096),
        "user_id": str(source_user_id),
        "work_dir": str(
            _canonical_absolute_path(
                work_dir, "scheduler raw current_working_directory"
            )
        ),
        "job_state": ascii_value(states[0], name="job_state", maximum=64),
        "reason": ascii_value(
            source["state_reason"], name="state_reason", maximum=4096
        ),
        "held": source["hold"],
        "restart_count": restart_count,
    }


def _authenticate_replay_submission_receipt(
    *,
    reference: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    candidate_job_id: str,
) -> str:
    """Stable-load the candidate-only receipt before repository imports."""
    submission, _ = _load_canonical_document(
        reference["path"],
        expected_sha256=reference["sha256"],
        trailing_lf=True,
        name="replay submission receipt",
    )
    if set(submission) != _REPLAY_SUBMISSION_ROOT_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "replay submission receipt root keyset differs"
        )
    expected_manifest_ref = {
        "path": str(manifest_path),
        "schema": _MANIFEST_SCHEMA,
        "sha256": manifest_sha256,
    }
    nonce = _digest(manifest["scheduler_submission"]["nonce"], "submission nonce")
    identity = manifest["scheduler_submission"]["identity"]
    expected_comment = (
        f"nemo-rl-strict-captured-replay-v2:{manifest['attempt_id']}:"
        f"{nonce}:{manifest_sha256}"
    )
    if (
        submission.get("schema") != _REPLAY_SUBMISSION_SCHEMA
        or submission.get("scorer_profile") != manifest["scorer_profile"]
        or submission.get("phase") != "SUBMISSION"
        or submission.get("status") != "held-candidate-not-in-job-authenticated"
        or submission.get("pair_id") != manifest["pair_id"]
        or submission.get("environment") != manifest["environment"]
        or submission.get("arm") != "on"
        or submission.get("mode") != "fresh_verifier_reward_replay"
        or submission.get("attempt_id") != manifest["attempt_id"]
        or submission.get("replay_execution_manifest") != expected_manifest_ref
        or submission.get("replay_source_snapshot")
        != manifest["replay_contract"]["source_snapshot"]
        or submission.get("submission_contract")
        != manifest["scheduler_submission"]["contract"]
        or submission.get("slurm_export_boundary") != manifest["slurm_export_boundary"]
        or submission.get("submission_nonce") != nonce
        or submission.get("job_name") != identity["job_name"]
        or submission.get("comment") != expected_comment
        or type(submission.get("submitter_euid")) is not int
        or submission.get("submitter_euid") != identity["submitter_euid"]
        or _job_id(submission.get("candidate_job_id"), "submission candidate ID")
        != candidate_job_id
        or type(submission.get("submitted_at_unix_ns")) is not int
        or not 0 < submission["submitted_at_unix_ns"] <= _MAX_INT63
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay submission receipt identity differs"
        )
    snapshot_root = _canonical_absolute_path(
        manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        "Pair ON snapshot root",
    )
    program = manifest["replay_contract"]["program"]
    for name in ("submission_launcher", "job_wrapper"):
        expected = {
            "path": str(snapshot_root / program[name]["path"]),
            "sha256": program[name]["sha256"],
        }
        if submission.get(name) != expected:
            raise StrictCapturedReplayEntrypointError(
                f"replay submission {name} differs from authenticated program"
            )
    host_tools = manifest["runtime_tools"]["document"]["host"]
    expected_tools = {
        name: host_tools[name] for name in ("sbatch", "scancel", "scontrol")
    }
    if submission.get("scheduler_tools") != expected_tools:
        raise StrictCapturedReplayEntrypointError(
            "replay submission scheduler tools differ"
        )
    client = _required_mapping(
        submission.get("scheduler_client_environment"),
        "submission scheduler client environment",
    )
    variables = _required_mapping(
        client.get("variables"), "submission scheduler client variables"
    )
    if (
        set(client) != {"ambient_merge", "env", "variables"}
        or client.get("ambient_merge") is not False
        or client.get("env") != host_tools["env"]
        or set(variables) != {"LC_ALL", "SLURM_CONF"}
        or variables.get("LC_ALL") != "C"
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay submission scheduler client environment differs"
        )
    slurm_conf = _required_mapping(variables.get("SLURM_CONF"), "submission SLURM_CONF")
    if set(slurm_conf) != {"path", "sha256"}:
        raise StrictCapturedReplayEntrypointError(
            "submission SLURM_CONF reference differs"
        )
    _canonical_absolute_path(slurm_conf.get("path"), "submission SLURM_CONF path")
    _digest(slurm_conf.get("sha256"), "submission SLURM_CONF SHA-256")

    accepted = _required_mapping(
        submission.get("accepted_id_record"), "accepted-ID record"
    )
    results_root = Path(manifest["artifacts"]["outputs"]["directory"]["path"]).parents[
        1
    ]
    expected_accepted_path = (
        results_root
        / "captured_replay/replay_submission_state"
        / manifest["pair_id"]
        / manifest["attempt_id"]
        / "accepted.job-id"
    )
    expected_candidate_bytes = f"{candidate_job_id}\n".encode("ascii")
    if accepted != {
        "path": str(expected_accepted_path),
        "sha256": hashlib.sha256(expected_candidate_bytes).hexdigest(),
        "parsed_candidate_job_id": candidate_job_id,
        "format": "ascii-positive-decimal-lf",
        "mode": "0400",
    }:
        raise StrictCapturedReplayEntrypointError("accepted-ID record differs")
    accepted_raw = _stable_regular_bytes(
        expected_accepted_path,
        name="accepted-ID record",
        expected_mode=0o400,
        maximum=128,
    )
    if accepted_raw != expected_candidate_bytes:
        raise StrictCapturedReplayEntrypointError("accepted-ID record bytes differ")

    sbatch = _required_mapping(submission.get("sbatch"), "submission sbatch")
    if set(sbatch) != {
        "path",
        "sha256",
        "argv",
        "argv_sha256",
        "parsable_stdout",
        "parsable_stdout_sha256",
    }:
        raise StrictCapturedReplayEntrypointError("submission sbatch keyset differs")
    argv = sbatch.get("argv")
    stdout = sbatch.get("parsable_stdout")
    if (
        sbatch.get("path") != host_tools["sbatch"]["path"]
        or sbatch.get("sha256") != host_tools["sbatch"]["sha256"]
        or not isinstance(argv, list)
        or not argv
        or any(type(member) is not str for member in argv)
        or sbatch.get("argv_sha256")
        != _domain_sha256("captured-replay-sbatch-argv", argv)
        or stdout != f"{candidate_job_id}\n"
        or sbatch.get("parsable_stdout_sha256")
        != hashlib.sha256(stdout.encode("ascii")).hexdigest()
    ):
        raise StrictCapturedReplayEntrypointError("submission sbatch authority differs")
    query_root = (
        results_root
        / "captured_replay/replay_submission_state"
        / manifest["pair_id"]
        / manifest["attempt_id"]
    )
    _authenticate_scheduler_query(
        reference=submission.get("pre_release_scheduler_query"),
        phase="PRE_RELEASE",
        expected_document_path=(query_root / "PRE_RELEASE.scontrol-query.json"),
        expected_raw_path=(query_root / "PRE_RELEASE.scontrol.raw"),
        job_id=candidate_job_id,
        job_name=identity["job_name"],
        comment=expected_comment,
        user_id=str(identity["submitter_euid"]),
        work_dir=str(snapshot_root),
        job_state="PENDING",
        held=True,
        reason="JobHeldUser",
        scontrol_path=host_tools["scontrol"]["path"],
    )
    return expected_comment


def _validate_runtime_environment(
    *,
    manifest: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    authenticated_job_id: str,
) -> dict[str, Any]:
    export_values = _load_authenticated_slurm_export(manifest)
    pair_boundary = _required_mapping(
        pair_manifest.get("slurm_export_boundary"),
        "Pair slurm_export_boundary",
    )
    if pair_boundary.get("schema") != _PAIR_SLURM_EXPORT_SCHEMA:
        raise StrictCapturedReplayEntrypointError(
            "source Pair Slurm export schema differs from authoritative Pair79"
        )
    if pair_boundary.get("allowed_names") != list(_SLURM_EXPORT_ALLOWED_NAMES):
        raise StrictCapturedReplayEntrypointError(
            "replay Slurm export names differ from exact Pair79"
        )
    expected_export_values = {name: "" for name in _SLURM_EXPORT_ALLOWED_NAMES}
    expected_export_values.update(
        {
            "EXPECTED_GYM_GITLINK_COMMIT": manifest["source"]["gym"]["gitlink_commit"],
            "EXPECTED_GYM_TREE": manifest["source"]["gym"]["tree"],
            "PAIR_ID": manifest["pair_id"],
            "RESULTS_DIR": manifest["artifacts"]["outputs"]["directory"]["path"],
            "STRICT_PAIR_ENVIRONMENT": manifest["environment"],
            "STRICT_PREBUILT_SNAPSHOT_DIR": manifest["replay_contract"][
                "source_snapshot"
            ]["ref"]["path"],
        }
    )
    if export_values != expected_export_values:
        raise StrictCapturedReplayEntrypointError(
            "replay Slurm export values differ from exact scorer-only Pair79"
        )
    for name in _SLURM_EXPORT_ALLOWED_NAMES:
        if os.environ.get(name) != export_values[name]:
            raise StrictCapturedReplayEntrypointError(
                f"authenticated replay Slurm export environment differs: {name}"
            )
    expected = {"STRICT_PAIR_BOUND_JOB_ID": authenticated_job_id}
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise StrictCapturedReplayEntrypointError(
                f"authenticated wrapper environment differs: {name}"
            )
    scheduler_device_environment = _live_scheduler_device_environment()
    for name in _WANDB_ENV_NAMES:
        if os.environ.get(name) != "":
            raise StrictCapturedReplayEntrypointError(
                f"scorer-only replay W&B boundary differs: {name}"
            )
    return scheduler_device_environment


def _live_scheduler_device_environment() -> dict[str, Any]:
    """Reconstruct and validate the exact wrapper-derived device boundary."""
    document: dict[str, Any] = {
        "schema": _SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA,
    }
    for environment_name, field_name in _RUNTIME_DEVICE_ENVIRONMENT_FIELDS:
        document[field_name] = os.environ.get(environment_name)
    return _validate_scheduler_device_environment(document)


def _validate_scheduler_device_environment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != _SCHEDULER_DEVICE_ENVIRONMENT_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "scheduler device environment keyset differs"
        )
    result = dict(value)
    if result["schema"] != _SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA:
        raise StrictCapturedReplayEntrypointError(
            "scheduler device environment schema differs"
        )
    cuda = _four_distinct_device_ids(
        result["cuda_visible_devices"], name="cuda_visible_devices"
    )
    ordinal = result["gpu_device_ordinal"]
    if ordinal is not None and ordinal != cuda:
        raise StrictCapturedReplayEntrypointError(
            "gpu_device_ordinal must equal cuda_visible_devices or null"
        )
    nvidia = result["nvidia_visible_devices"]
    if nvidia is not None:
        _nonempty_ascii_device_value(nvidia, name="nvidia_visible_devices")
        if nvidia not in {cuda, "all", "none", "void"}:
            tokens = nvidia.split(",")
            if (
                len(tokens) != 4
                or len(set(tokens)) != 4
                or any(_GPU_UUID_RE.fullmatch(token) is None for token in tokens)
            ):
                raise StrictCapturedReplayEntrypointError(
                    "nvidia_visible_devices has invalid GPU identities"
                )
    rocr = result["rocr_visible_devices"]
    if rocr is not None:
        _four_distinct_device_ids(rocr, name="rocr_visible_devices")
    ze = result["ze_affinity_mask"]
    if ze is not None:
        _nonempty_ascii_device_value(ze, name="ze_affinity_mask")
        tokens = ze.split(",")
        if not 1 <= len(tokens) <= 64 or any(
            _ZE_DEVICE_RE.fullmatch(token) is None for token in tokens
        ):
            raise StrictCapturedReplayEntrypointError(
                "ze_affinity_mask has invalid device tokens"
            )
        identities = {
            (
                int(token.split(".")[0], 10),
                int(token.split(".")[1], 10) if "." in token else 0,
            )
            for token in tokens
        }
        if len(identities) != len(tokens):
            raise StrictCapturedReplayEntrypointError(
                "ze_affinity_mask device identities repeat"
            )
    return result


def _four_distinct_device_ids(value: Any, *, name: str) -> str:
    text = _nonempty_ascii_device_value(value, name=name)
    tokens = text.split(",")
    if (
        len(tokens) != 4
        or any(re.fullmatch(r"0|[1-9][0-9]*", token) is None for token in tokens)
        or len({int(token) for token in tokens}) != 4
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{name} must contain four canonical numerically distinct decimal IDs"
        )
    return text


def _nonempty_ascii_device_value(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise StrictCapturedReplayEntrypointError(
            f"{name} must be a nonempty ASCII string"
        )
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise StrictCapturedReplayEntrypointError(f"{name} must be ASCII") from error
    if len(payload) > 255:
        raise StrictCapturedReplayEntrypointError(f"{name} exceeds 255 ASCII bytes")
    return value


def _load_authenticated_slurm_export(manifest: Mapping[str, Any]) -> dict[str, str]:
    boundary = _required_mapping(
        manifest.get("slurm_export_boundary"), "slurm_export_boundary"
    )
    if set(boundary) != _SLURM_EXPORT_BOUNDARY_KEYS:
        raise StrictCapturedReplayEntrypointError(
            "replay Slurm export boundary keyset differs"
        )
    expected_boundary = {
        "schema": _REPLAY_SLURM_EXPORT_SCHEMA,
        "allowed_names": list(_SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "attempt_id": manifest["attempt_id"],
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
            "--environment",
            "{environment}",
            "--profile-id",
            "{profile_id}",
        ],
    }
    for name, expected in expected_boundary.items():
        if boundary.get(name) != expected or type(boundary.get(name)) is not type(
            expected
        ):
            raise StrictCapturedReplayEntrypointError(
                f"replay Slurm export boundary differs: {name}"
            )
    path = _canonical_absolute_path(boundary.get("path"), "replay Slurm export path")
    expected_sha256 = _digest(boundary.get("sha256"), "replay Slurm export SHA-256")
    raw = _stable_regular_bytes(
        path,
        name="replay Slurm export",
        expected_mode=0o400,
        maximum=16 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise StrictCapturedReplayEntrypointError(
            "replay Slurm export bytes differ from manifest authority"
        )
    if not raw.endswith(b"\0") or raw.endswith(b"\0\0"):
        raise StrictCapturedReplayEntrypointError("replay Slurm export framing differs")
    records = raw[:-1].split(b"\0")
    if len(records) != len(_SLURM_EXPORT_ALLOWED_NAMES):
        raise StrictCapturedReplayEntrypointError(
            "replay Slurm export record count differs"
        )
    result: dict[str, str] = {}
    for expected_name, record in zip(_SLURM_EXPORT_ALLOWED_NAMES, records, strict=True):
        raw_name, separator, raw_value = record.partition(b"=")
        if separator != b"=" or raw_name != expected_name.encode("ascii"):
            raise StrictCapturedReplayEntrypointError(
                "replay Slurm export record order/name differs"
            )
        try:
            result[expected_name] = raw_value.decode("ascii")
        except UnicodeDecodeError as error:
            raise StrictCapturedReplayEntrypointError(
                f"replay Slurm export value is not ASCII: {expected_name}"
            ) from error
    return result


def _assert_wandb_disabled(manifest: Mapping[str, Any]) -> None:
    if manifest.get("wandb") != _DISABLED_WANDB_POLICY:
        raise StrictCapturedReplayEntrypointError(
            "scorer-only replay W&B policy differs"
        )
    if any(name == "wandb" or name.startswith("wandb.") for name in sys.modules):
        raise StrictCapturedReplayEntrypointError(
            "scorer-only replay imported W&B despite disabled policy"
        )


def _authenticate_snapshot_program(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    pair_manifest: Mapping[str, Any],
    execution_source_root: Path,
) -> dict[str, str]:
    """Authenticate the complete Pair ON tree and derive program9 from it."""
    contract = _required_mapping(manifest.get("replay_contract"), "replay_contract")
    source_snapshot = _required_mapping(
        contract.get("source_snapshot"), "source_snapshot"
    )
    source_ref = _required_mapping(source_snapshot.get("ref"), "source_snapshot.ref")
    pair_snapshots = _required_mapping(
        _required_mapping(pair_manifest.get("source"), "Pair source").get("snapshots"),
        "Pair source.snapshots",
    )
    pair_on = _required_mapping(pair_snapshots.get("on"), "Pair ON snapshot")
    if (
        set(source_snapshot) != {"arm", "ref"}
        or source_snapshot.get("arm") != "on"
        or set(source_ref)
        != {"config_sha256", "entrypoint_sha256", "manifest_sha256", "path"}
        or dict(source_ref) != dict(pair_on)
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay source snapshot differs from authenticated Pair ON"
        )
    snapshot_root = _canonical_absolute_path(
        source_ref.get("path"), "Pair ON snapshot root"
    )
    if snapshot_root != execution_source_root:
        raise StrictCapturedReplayEntrypointError(
            "PRE execution root differs from authenticated Pair ON snapshot"
        )
    snapshot_manifest_path = snapshot_root / _SNAPSHOT_SHA_MANIFEST
    snapshot_manifest_raw = _stable_regular_bytes(
        snapshot_manifest_path,
        name="Pair ON snapshot SHA manifest",
        expected_mode=0o400,
        maximum=64 * 1024 * 1024,
    )
    snapshot_manifest_sha256 = hashlib.sha256(snapshot_manifest_raw).hexdigest()
    if snapshot_manifest_sha256 != _digest(
        source_ref.get("manifest_sha256"), "Pair ON snapshot manifest SHA-256"
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot SHA manifest differs from Pair"
        )
    snapshot_manifest = _parse_snapshot_sha_manifest(snapshot_manifest_raw)
    for required in (_SNAPSHOT_SYMLINK_MANIFEST, _SNAPSHOT_MODE_MANIFEST):
        if required not in snapshot_manifest:
            raise StrictCapturedReplayEntrypointError(
                f"Pair ON snapshot SHA manifest omits {required}"
            )
    symlink_document, _ = _load_canonical_document(
        snapshot_root / _SNAPSHOT_SYMLINK_MANIFEST,
        expected_sha256=snapshot_manifest[_SNAPSHOT_SYMLINK_MANIFEST],
        trailing_lf=True,
        name="Pair ON snapshot symlink manifest",
    )
    if (
        set(symlink_document) != {"schema", "symlinks"}
        or symlink_document.get("schema") != "nemo-rl-strict-snapshot-symlinks-v1"
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot symlink manifest differs"
        )
    symlinks_raw = _required_mapping(
        symlink_document.get("symlinks"), "snapshot symlinks"
    )
    symlinks: dict[str, str] = {}
    for raw_relative, target in symlinks_raw.items():
        relative = _snapshot_relative_path(raw_relative, "snapshot symlink path")
        if (
            type(target) is not str
            or not target
            or target.startswith("/")
            or posixpath.normpath(target) != target
            or any(character in target for character in ("\x00", "\n", "\r"))
        ):
            raise StrictCapturedReplayEntrypointError(
                f"snapshot symlink target is unsafe: {relative}"
            )
        symlinks[relative] = target
    mode_document, _ = _load_canonical_document(
        snapshot_root / _SNAPSHOT_MODE_MANIFEST,
        expected_sha256=snapshot_manifest[_SNAPSHOT_MODE_MANIFEST],
        trailing_lf=True,
        name="Pair ON snapshot mode manifest",
    )
    if (
        set(mode_document) != {"schema", "regular_file_executable"}
        or mode_document.get("schema") != "nemo-rl-strict-snapshot-modes-v1"
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot mode manifest differs"
        )
    modes_raw = _required_mapping(
        mode_document.get("regular_file_executable"), "snapshot executable modes"
    )
    executable_modes: dict[str, bool] = {}
    for raw_relative, executable in modes_raw.items():
        relative = _snapshot_relative_path(raw_relative, "snapshot mode path")
        if type(executable) is not bool:
            raise StrictCapturedReplayEntrypointError(
                "snapshot executable flag is not an exact boolean"
            )
        executable_modes[relative] = executable
    if set(executable_modes) != set(snapshot_manifest):
        raise StrictCapturedReplayEntrypointError(
            "snapshot executable-mode inventory differs from SHA manifest"
        )

    actual_regular, actual_symlinks, actual_directories, before_fingerprints = (
        _snapshot_tree_inventory(snapshot_root)
    )
    if actual_regular != set(snapshot_manifest) | {_SNAPSHOT_SHA_MANIFEST}:
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot regular-file inventory differs"
        )
    if actual_symlinks != symlinks:
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot symlink inventory differs"
        )
    expected_directories = {""}
    for relative in actual_regular | set(actual_symlinks):
        parents = relative.split("/")[:-1]
        for length in range(1, len(parents) + 1):
            expected_directories.add("/".join(parents[:length]))
    if actual_directories != expected_directories:
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot directory inventory differs"
        )
    _validate_snapshot_symlink_targets(
        symlinks=symlinks,
        regular_paths=actual_regular,
        directory_paths=actual_directories,
    )
    for relative, expected_digest in snapshot_manifest.items():
        path = snapshot_root / relative
        raw = _stable_regular_bytes(
            path,
            name=f"Pair ON snapshot member {relative}",
            expected_mode=None,
            maximum=256 * 1024 * 1024,
            allow_empty=True,
        )
        metadata = path.lstat()
        if (
            hashlib.sha256(raw).hexdigest() != expected_digest
            or bool(stat.S_IMODE(metadata.st_mode) & 0o111)
            != executable_modes[relative]
        ):
            raise StrictCapturedReplayEntrypointError(
                f"Pair ON snapshot member differs: {relative}"
            )

    selected_config = _required_mapping(
        contract.get("selected_config"), "replay selected config"
    )
    config_path = _snapshot_relative_path(
        selected_config.get("path"), "selected config path"
    )
    original_entrypoint = "examples/run_grpo_single_controller.py"
    if (
        snapshot_manifest.get(config_path)
        != _digest(source_ref.get("config_sha256"), "Pair ON config SHA-256")
        or snapshot_manifest.get(config_path) != selected_config.get("sha256")
        or snapshot_manifest.get(original_entrypoint)
        != _digest(
            source_ref.get("entrypoint_sha256"),
            "Pair ON training entrypoint SHA-256",
        )
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON config/training entrypoint does not join its SHA manifest"
        )

    _authenticate_nested_gym_source(
        manifest=manifest,
        pair_manifest=pair_manifest,
        snapshot_manifest=snapshot_manifest,
        directory_paths=actual_directories,
        symlinks=actual_symlinks,
    )

    program = _required_mapping(contract.get("program"), "replay program")
    if set(program) != set(_PROGRAM_PATHS):
        raise StrictCapturedReplayEntrypointError("replay program keyset differs")
    normalized_program: dict[str, dict[str, str]] = {}
    for name, relative in _PROGRAM_PATHS.items():
        reference = _required_mapping(program.get(name), f"program.{name}")
        expected_digest = snapshot_manifest.get(relative)
        if (
            set(reference) != {"path", "sha256"}
            or reference.get("path") != relative
            or reference.get("sha256") != expected_digest
            or expected_digest is None
        ):
            raise StrictCapturedReplayEntrypointError(
                f"replay program {name} was not derived from Pair ON bytes"
            )
        normalized_program[name] = {
            "path": relative,
            "sha256": _digest(expected_digest, f"program.{name} SHA-256"),
        }
    post_regular, post_symlinks, post_directories, after_fingerprints = (
        _snapshot_tree_inventory(snapshot_root)
    )
    if (
        post_regular != actual_regular
        or post_symlinks != actual_symlinks
        or post_directories != actual_directories
        or after_fingerprints != before_fingerprints
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot changed during authentication"
        )
    expected_program_sha = _domain_sha256("captured-replay-program", normalized_program)
    invoked = Path(__file__)
    if not invoked.is_absolute():
        invoked = invoked.absolute()
    if (
        invoked.resolve(strict=True)
        != execution_source_root / _PROGRAM_PATHS["entrypoint"]
    ):
        raise StrictCapturedReplayEntrypointError(
            "executed replay entrypoint differs from PRE-bound source root"
        )
    return {
        "status": "authenticated",
        "replay_execution_manifest_sha256": _digest(
            manifest_sha256, "replay execution manifest SHA-256"
        ),
        "source_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "program_sha256": expected_program_sha,
    }


def _parse_snapshot_sha_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot SHA manifest framing differs"
        )
    result: dict[str, str] = {}
    for index, line in enumerate(raw[:-1].split(b"\n")):
        digest_raw, separator, relative_raw = line.partition(b"  ")
        if not separator or not relative_raw:
            raise StrictCapturedReplayEntrypointError(
                f"snapshot SHA manifest line {index} is malformed"
            )
        try:
            digest = digest_raw.decode("ascii")
            relative = relative_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise StrictCapturedReplayEntrypointError(
                f"snapshot SHA manifest line {index} is not ASCII"
            ) from error
        relative = _snapshot_relative_path(
            relative, f"snapshot SHA manifest line {index} path"
        )
        if relative in result:
            raise StrictCapturedReplayEntrypointError(
                f"snapshot SHA manifest repeats {relative}"
            )
        result[relative] = _digest(
            digest, f"snapshot SHA manifest line {index} SHA-256"
        )
    if not result or _SNAPSHOT_SHA_MANIFEST in result:
        raise StrictCapturedReplayEntrypointError(
            "snapshot SHA manifest inventory is empty or self-referential"
        )
    return result


def _authenticate_nested_gym_source(
    *,
    manifest: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
    snapshot_manifest: Mapping[str, str],
    directory_paths: set[str],
    symlinks: Mapping[str, str],
) -> None:
    """Bind imports to the Gym checkout copied into the authenticated ON tree."""
    manifest_profile = _required_mapping(
        manifest.get("scorer_profile"), "scorer_profile"
    )
    profile = _closed_profile(
        expected_environment=manifest.get("environment"),
        expected_profile_id=manifest_profile.get("profile_id"),
    )
    if dict(manifest_profile) != profile:
        raise StrictCapturedReplayEntrypointError(
            "nested Gym source profile differs from closed registry"
        )
    pair_source = _required_mapping(pair_manifest.get("source"), "Pair source")
    pair_artifacts = pair_manifest.get("artifacts")
    if type(pair_source) is not dict or type(pair_artifacts) is not dict:
        raise StrictCapturedReplayEntrypointError(
            "Pair source/artifacts must be exact dictionaries"
        )
    pair_fixture = pair_artifacts.get("fixture")
    source_root = pair_source.get("root")
    if type(pair_fixture) is not dict or type(source_root) is not str:
        raise StrictCapturedReplayEntrypointError(
            "Pair source root/fixture types differ"
        )
    source_root_path = _canonical_absolute_path(source_root, "Pair source root")
    profile_fixture = profile["fixture"]
    fixture_relative = _snapshot_relative_path(
        profile_fixture["path"], "profile fixture path"
    )
    expected_fixture_path = str(PurePosixPath(str(source_root_path)) / fixture_relative)
    if (
        set(pair_fixture) != {"path", "rows", "sha256"}
        or type(pair_fixture.get("path")) is not str
        or pair_fixture["path"] != expected_fixture_path
        or type(pair_fixture.get("sha256")) is not str
        or pair_fixture["sha256"] != profile_fixture["sha256"]
        or type(pair_fixture.get("rows")) is not int
        or pair_fixture["rows"] != profile_fixture["rows"]
        or snapshot_manifest.get(fixture_relative)
        != _digest(profile_fixture["sha256"], "profile fixture SHA-256")
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair fixture differs from the closed scorer profile"
        )
    replay_source = _required_mapping(manifest.get("source"), "replay source")
    pair_gym = _required_mapping(pair_source.get("gym"), "Pair source.gym")
    if (
        dict(replay_source) != dict(pair_source)
        or set(pair_gym) != {"gitlink_commit", "path", "tree"}
        or type(pair_gym.get("gitlink_commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", pair_gym["gitlink_commit"]) is None
        or type(pair_gym.get("tree")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", pair_gym["tree"]) is None
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay Gym source differs from authenticated Pair"
        )
    _canonical_absolute_path(pair_gym.get("path"), "Pair Gym source path")

    contract = _required_mapping(manifest.get("replay_contract"), "replay_contract")
    scorer = _required_mapping(contract.get("gym_scorer"), "gym_scorer")
    resources = _required_mapping(scorer.get("resources"), "gym_scorer.resources")
    pair_resources = _required_mapping(
        _required_mapping(pair_manifest.get("selection"), "Pair selection").get(
            "gym_resources"
        ),
        "Pair selection.gym_resources",
    )
    snapshot_root = _canonical_absolute_path(
        _required_mapping(contract.get("source_snapshot"), "source_snapshot")["ref"][
            "path"
        ],
        "Pair ON snapshot root",
    )
    expected_source_root = {
        "snapshot_relative_path": _GYM_SOURCE_RELATIVE,
        "host_path": str(snapshot_root / _GYM_SOURCE_RELATIVE),
        "container_path": _GYM_CONTAINER_ROOT,
    }
    if (
        scorer.get("source") != pair_gym
        or dict(resources) != dict(pair_resources)
        or scorer.get("source_root") != expected_source_root
    ):
        raise StrictCapturedReplayEntrypointError(
            "replay Gym import root/resources do not bind the Pair ON Gym"
        )

    if (
        _GYM_SOURCE_RELATIVE not in directory_paths
        or f"{_GYM_SOURCE_RELATIVE}/nemo_gym" not in directory_paths
        or _GYM_PACKAGE_INIT_RELATIVE not in snapshot_manifest
    ):
        raise StrictCapturedReplayEntrypointError(
            "authenticated Pair ON snapshot omits the nested Gym package"
        )
    shadow_paths = set(snapshot_manifest) | set(symlinks) | directory_paths
    if (
        "nemo_gym" in shadow_paths
        or "nemo_gym.py" in shadow_paths
        or any(relative.startswith("nemo_gym/") for relative in shadow_paths)
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON root contains a Gym package shadow outside nested Gym"
        )
    nested_nemo_rl = f"{_GYM_SOURCE_RELATIVE}/nemo_rl"
    if (
        nested_nemo_rl in shadow_paths
        or f"{nested_nemo_rl}.py" in shadow_paths
        or any(relative.startswith(f"{nested_nemo_rl}/") for relative in shadow_paths)
    ):
        raise StrictCapturedReplayEntrypointError(
            "nested Gym root contains a NeMo-RL package shadow"
        )

    expected_resource_paths = {
        "config": profile["resource_config"]["path"],
        "requirements": profile["requirements"]["path"],
        "verifier_source": profile["resource_app"]["path"],
    }
    expected_resource_sha256 = {
        "config": profile["resource_config"]["sha256"],
        "requirements": profile["requirements"]["sha256"],
        "verifier_source": profile["resource_app"]["sha256"],
    }
    if set(pair_resources) != set(expected_resource_paths):
        raise StrictCapturedReplayEntrypointError(
            "Pair format-verifier resource keyset differs"
        )
    for name, expected_path in expected_resource_paths.items():
        reference = _required_mapping(
            pair_resources.get(name), f"Pair Gym resource {name}"
        )
        relative = f"{_GYM_SOURCE_RELATIVE}/{expected_path}"
        if (
            set(reference) != {"path", "sha256"}
            or reference.get("path") != expected_path
            or reference.get("sha256") != expected_resource_sha256[name]
            or snapshot_manifest.get(relative)
            != _digest(reference.get("sha256"), f"Pair Gym resource {name} SHA-256")
        ):
            raise StrictCapturedReplayEntrypointError(
                f"Pair ON snapshot Gym resource differs: {name}"
            )

    launcher = _required_mapping(scorer.get("launcher"), "gym_scorer.launcher")
    if (
        launcher.get("resource_only_config") is not None
        or launcher.get("config_path_name") != profile["resource_config_path_name"]
    ):
        raise StrictCapturedReplayEntrypointError(
            "format replay launcher/config reduction contract differs"
        )


def _snapshot_relative_path(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(character in value for character in ("\x00", "\n", "\r"))
        or posixpath.normpath(value) != value
        or ".." in value.split("/")
        or value == "."
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{name} must be a canonical POSIX relative path"
        )
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise StrictCapturedReplayEntrypointError(f"{name} must be ASCII") from error
    return value


def _snapshot_tree_inventory(
    root: Path,
) -> tuple[set[str], dict[str, str], set[str], dict[str, tuple[int, ...]]]:
    root_metadata = root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o222
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot root must be EUID-owned and nonwritable"
        )
    regular: set[str] = set()
    symlinks: dict[str, str] = {}
    directories: set[str] = {""}
    fingerprints: dict[str, tuple[int, ...]] = {"": _stat_fingerprint(root_metadata)}

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=raise_walk_error, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        for name in list(directory_names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            _reject_snapshot_bytecode_path(relative)
            metadata = path.lstat()
            fingerprints[relative] = _stat_fingerprint(metadata)
            if stat.S_ISLNK(metadata.st_mode):
                if metadata.st_uid != os.geteuid():
                    raise StrictCapturedReplayEntrypointError(
                        f"snapshot symlink owner differs: {relative}"
                    )
                symlinks[relative] = os.readlink(path)
                directory_names.remove(name)
            elif (
                stat.S_ISDIR(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and not stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                directories.add(relative)
            else:
                raise StrictCapturedReplayEntrypointError(
                    f"snapshot directory is writable/special: {relative}"
                )
        for name in file_names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            _reject_snapshot_bytecode_path(relative)
            metadata = path.lstat()
            fingerprints[relative] = _stat_fingerprint(metadata)
            if stat.S_ISLNK(metadata.st_mode):
                if metadata.st_uid != os.geteuid():
                    raise StrictCapturedReplayEntrypointError(
                        f"snapshot symlink owner differs: {relative}"
                    )
                symlinks[relative] = os.readlink(path)
            elif (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
                and metadata.st_nlink == 1
                and not stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                regular.add(relative)
            else:
                raise StrictCapturedReplayEntrypointError(
                    f"snapshot file is writable/multilink/special: {relative}"
                )
    return regular, symlinks, directories, fingerprints


def _reject_snapshot_bytecode_path(relative: str) -> None:
    parts = relative.split("/")
    if "__pycache__" in parts or relative.endswith((".pyc", ".pyo")):
        raise StrictCapturedReplayEntrypointError(
            f"Pair ON snapshot contains executable cached bytecode: {relative}"
        )


def _validate_snapshot_symlink_targets(
    *,
    symlinks: Mapping[str, str],
    regular_paths: set[str],
    directory_paths: set[str],
) -> None:
    authenticated_targets = regular_paths | directory_paths | set(symlinks)

    def resolve_components(
        components: list[str],
        resolved: list[str],
        active_symlinks: frozenset[str],
        *,
        source: str,
    ) -> list[str]:
        pending = list(components)
        while pending:
            component = pending.pop(0)
            if component in {"", "."}:
                continue
            if component == "..":
                if not resolved:
                    raise StrictCapturedReplayEntrypointError(
                        f"snapshot symlink escapes authenticated root: {source}"
                    )
                resolved.pop()
                continue
            candidate = "/".join([*resolved, component])
            if candidate in symlinks:
                if candidate in active_symlinks:
                    raise StrictCapturedReplayEntrypointError(
                        f"snapshot symlink cycle detected: {source}"
                    )
                resolved = resolve_components(
                    symlinks[candidate].split("/"),
                    resolved,
                    active_symlinks | {candidate},
                    source=source,
                )
                continue
            if candidate not in authenticated_targets:
                raise StrictCapturedReplayEntrypointError(
                    f"snapshot symlink target is absent from authenticated inventory: {source}"
                )
            resolved.append(component)
            if pending and candidate not in directory_paths:
                raise StrictCapturedReplayEntrypointError(
                    f"snapshot symlink traverses an authenticated non-directory: {source}"
                )
        return resolved

    for link in sorted(symlinks):
        resolved = resolve_components(link.split("/"), [], frozenset(), source=link)
        if "/".join(resolved) not in authenticated_targets:
            raise StrictCapturedReplayEntrypointError(
                f"snapshot symlink target is absent from authenticated inventory: {link}"
            )


def _activate_authenticated_source_roots(root: Path) -> Path:
    """Put only the authenticated NeMo-RL and nested Gym roots first."""
    gym_root = root / _GYM_SOURCE_RELATIVE
    for candidate, name in ((root, "NeMo-RL"), (gym_root, "NeMo-Gym")):
        metadata = candidate.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise StrictCapturedReplayEntrypointError(
                f"authenticated {name} import root is not an EUID-owned nonwritable directory"
            )
    package_init = gym_root / "nemo_gym/__init__.py"
    specification = importlib.machinery.PathFinder.find_spec(
        "nemo_gym", [str(gym_root)]
    )
    if (
        specification is None
        or type(specification.origin) is not str
        or Path(specification.origin).resolve(strict=True) != package_init
    ):
        raise StrictCapturedReplayEntrypointError(
            "authenticated nested Gym root does not resolve the pinned nemo_gym package"
        )
    values = [str(gym_root), str(root)]
    sys.path[:] = [entry for entry in sys.path if entry not in values]
    sys.path[0:0] = values
    return gym_root


def _verify_imported_source_module_origins(
    *,
    execution_source_root: Path,
    gym_source_root: Path,
    expected_snapshot_manifest_sha256: str,
) -> None:
    """Reject mixed/ambient NeMo-RL or NeMo-Gym module graphs after import."""
    expected_gym_root = execution_source_root / _GYM_SOURCE_RELATIVE
    if gym_source_root != expected_gym_root:
        raise StrictCapturedReplayEntrypointError(
            "authenticated NeMo-Gym import root changed"
        )
    snapshot_manifest_raw = _stable_regular_bytes(
        execution_source_root / _SNAPSHOT_SHA_MANIFEST,
        name="post-import Pair ON snapshot SHA manifest",
        expected_mode=0o400,
        maximum=64 * 1024 * 1024,
    )
    if hashlib.sha256(snapshot_manifest_raw).hexdigest() != _digest(
        expected_snapshot_manifest_sha256,
        "post-import Pair ON snapshot manifest SHA-256",
    ):
        raise StrictCapturedReplayEntrypointError(
            "Pair ON snapshot manifest changed after source activation"
        )
    snapshot_manifest = _parse_snapshot_sha_manifest(snapshot_manifest_raw)
    roots = {
        "nemo_rl": execution_source_root,
        "nemo_gym": gym_source_root,
    }
    observed = 0
    for module_name, module in sorted(sys.modules.items()):
        package = next(
            (
                prefix
                for prefix in roots
                if module_name == prefix or module_name.startswith(f"{prefix}.")
            ),
            None,
        )
        if package is None:
            continue
        observed += 1
        if module is None:
            raise StrictCapturedReplayEntrypointError(
                f"imported module is an unresolved sentinel: {module_name}"
            )
        raw_file = getattr(module, "__file__", None)
        specification = getattr(module, "__spec__", None)
        specification_origin = getattr(specification, "origin", None)
        if type(raw_file) is not str or not raw_file:
            raise StrictCapturedReplayEntrypointError(
                f"imported module has no concrete origin: {module_name}"
            )
        module_path = Path(raw_file).resolve(strict=True)
        expected_root = roots[package].resolve(strict=True)
        try:
            module_path.relative_to(expected_root)
            snapshot_relative = module_path.relative_to(
                execution_source_root.resolve(strict=True)
            ).as_posix()
        except ValueError as error:
            raise StrictCapturedReplayEntrypointError(
                f"imported module escaped authenticated {package} root: {module_name}"
            ) from error
        expected_digest = snapshot_manifest.get(snapshot_relative)
        imported_raw = _stable_regular_bytes(
            module_path,
            name=f"imported module {module_name}",
            expected_mode=None,
            allow_empty=True,
        )
        if (
            expected_digest is None
            or hashlib.sha256(imported_raw).hexdigest() != expected_digest
        ):
            raise StrictCapturedReplayEntrypointError(
                f"imported module bytes are absent from authenticated Pair ON inventory: {module_name}"
            )
        if specification_origin not in {None, raw_file}:
            try:
                same_origin = (
                    Path(specification_origin).resolve(strict=True) == module_path
                )
            except (OSError, TypeError, ValueError):
                same_origin = False
            if not same_origin:
                raise StrictCapturedReplayEntrypointError(
                    f"imported module spec origin differs: {module_name}"
                )
        package_paths = getattr(module, "__path__", None)
        if package_paths is not None:
            for package_path in package_paths:
                try:
                    Path(package_path).resolve(strict=True).relative_to(expected_root)
                except (OSError, TypeError, ValueError) as error:
                    raise StrictCapturedReplayEntrypointError(
                        f"imported package search path escaped authenticated root: {module_name}"
                    ) from error
    if observed == 0:
        raise StrictCapturedReplayEntrypointError(
            "no authenticated NeMo-RL/Gym modules were imported"
        )


def _verify_imported_program_modules(
    *,
    execution_source_root: Path,
    program: Mapping[str, Any],
    required_program_names: frozenset[str] | None = None,
) -> None:
    """Close every required imported module to its manifest path and bytes.

    The external launch/wrapper bootstraps predate the explicit staged argument.
    Their default stage requires the four modules both scripts import, plus the
    result sealer when the wrapper has imported it.  Driver call sites always
    pass an exact stage set and never infer a missing runtime dependency.
    """
    if required_program_names is None:
        required = set(_BOOTSTRAP_REQUIRED_PROGRAM_NAMES)
        if "nemo_rl.utils.strict_captured_replay_seal_v2" in sys.modules:
            required.add("result_sealer")
    else:
        if type(required_program_names) is not frozenset or any(
            type(name) is not str for name in required_program_names
        ):
            raise StrictCapturedReplayEntrypointError(
                "required program module stage must be an exact frozenset of strings"
            )
        required = set(required_program_names)
    admitted = set(_PROGRAM_MODULE_NAMES.values())
    if not required or not required <= admitted:
        raise StrictCapturedReplayEntrypointError(
            "required program module stage contains an unadmitted name"
        )

    # Any additional program module already resident in this interpreter is
    # also authority-bearing and therefore receives the same origin check.
    observed = {
        program_name
        for module_name, program_name in _PROGRAM_MODULE_NAMES.items()
        if module_name in sys.modules
    }
    for module_name, program_name in _PROGRAM_MODULE_NAMES.items():
        if program_name not in required | observed:
            continue
        module = sys.modules.get(module_name)
        if module is None:
            raise StrictCapturedReplayEntrypointError(
                f"required authenticated program module is not imported: {module_name}"
            )
        entry = _required_mapping(
            program.get(program_name),
            f"replay program.{program_name}",
        )
        if set(entry) != {"path", "sha256"}:
            raise StrictCapturedReplayEntrypointError(
                f"replay program entry keyset differs: {program_name}"
            )
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
        expected_path = execution_source_root / entry["path"]
        if (
            module_path != expected_path
            or _stable_regular_sha256(
                module_path, name=f"imported module {module_name}"
            )
            != entry["sha256"]
        ):
            raise StrictCapturedReplayEntrypointError(
                f"imported module differs from authenticated program: {module_name}"
            )


def _create_exclusive_output_root(path: Path) -> None:
    canonical = _canonical_absolute_path(str(path), "replay output root")
    parent = canonical.parent
    directory_fd = _open_absolute_directory_no_symlinks(parent)
    try:
        parent_info = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_IMODE(parent_info.st_mode) != 0o700
            or parent_info.st_uid != os.geteuid()
        ):
            raise StrictCapturedReplayEntrypointError(
                "replay output parent is not EUID-owned mode-0700"
            )
        try:
            os.stat(canonical.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise StrictCapturedReplayEntrypointError(
                "replay output root already exists at driver entry"
            )
        try:
            os.mkdir(canonical.name, 0o700, dir_fd=directory_fd)
        except FileExistsError as error:
            raise StrictCapturedReplayEntrypointError(
                "replay output root appeared during exclusive creation"
            ) from error
        created_fd = os.open(
            canonical.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            created = os.fstat(created_fd)
            named = os.stat(canonical.name, dir_fd=directory_fd, follow_symlinks=False)
            os.fsync(created_fd)
        finally:
            os.close(created_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if (
        _stat_fingerprint(created) != _stat_fingerprint(named)
        or not stat.S_ISDIR(created.st_mode)
        or stat.S_IMODE(created.st_mode) != 0o700
        or created.st_uid != os.geteuid()
    ):
        raise StrictCapturedReplayEntrypointError(
            "created replay output root failed validation"
        )


def _load_canonical_document(
    path: str | Path,
    *,
    expected_sha256: str,
    trailing_lf: bool,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    canonical_path = _canonical_absolute_path(str(path), f"{name} path")
    digest = _digest(expected_sha256, f"{name} expected SHA-256")
    raw = _stable_regular_bytes(canonical_path, name=name, expected_mode=0o400)

    def reject_constant(value: str) -> Any:
        raise StrictCapturedReplayEntrypointError(
            f"{name} contains non-finite JSON constant {value}"
        )

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictCapturedReplayEntrypointError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StrictCapturedReplayEntrypointError(
            f"{name} is not strict ASCII JSON"
        ) from error
    if not isinstance(document, dict):
        raise StrictCapturedReplayEntrypointError(f"{name} root is not an object")
    _reject_negative_zero(document, name=name)
    encoded = _canonical_ascii_json(document) + (b"\n" if trailing_lf else b"")
    if raw != encoded or hashlib.sha256(raw).hexdigest() != digest:
        raise StrictCapturedReplayEntrypointError(
            f"{name} bytes/digest differ from canonical authority"
        )
    return document, raw


def _stable_regular_bytes(
    path: Path,
    *,
    name: str,
    expected_mode: int | None = None,
    maximum: int = 256 * 1024 * 1024,
    allow_empty: bool = False,
) -> bytes:
    if not path.is_absolute() or posixpath.normpath(str(path)) != str(path):
        raise StrictCapturedReplayEntrypointError(f"{name} path is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = _open_absolute_directory_no_symlinks(path.parent)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid not in {0, os.geteuid()}
            or not (0 if allow_empty else 1) <= before.st_size <= maximum
            or (expected_mode is not None and mode != expected_mode)
            or (expected_mode is None and mode & 0o222)
        ):
            raise StrictCapturedReplayEntrypointError(
                f"{name} metadata differs from the immutable file contract"
            )
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(descriptor)
            if _stat_fingerprint(opened) != _stat_fingerprint(before):
                raise StrictCapturedReplayEntrypointError(
                    f"{name} inode changed while opening"
                )
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise StrictCapturedReplayEntrypointError(
                        f"{name} truncated while reading"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise StrictCapturedReplayEntrypointError(f"{name} grew while reading")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)
    if not (
        _stat_fingerprint(opened)
        == _stat_fingerprint(after)
        == _stat_fingerprint(named)
    ):
        raise StrictCapturedReplayEntrypointError(f"{name} changed during stable read")
    return b"".join(chunks)


def _stable_regular_sha256(path: Path, *, name: str) -> str:
    raw = _stable_regular_bytes(path, name=name, expected_mode=None)
    return hashlib.sha256(raw).hexdigest()


def _open_absolute_directory_no_symlinks(path: Path) -> int:
    if not path.is_absolute() or posixpath.normpath(str(path)) != str(path):
        raise StrictCapturedReplayEntrypointError(
            "stable-read parent must be a canonical absolute path"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
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


def _reject_symlink_components(path: Path) -> None:
    if not path.is_absolute() or str(path) != posixpath.normpath(str(path)):
        raise StrictCapturedReplayEntrypointError("secure path is not canonical")
    cursor = Path("/")
    for component in path.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            raise StrictCapturedReplayEntrypointError(
                f"secure path contains a symlink: {cursor}"
            )


def _canonical_absolute_path(value: Any, name: str) -> Path:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or posixpath.normpath(value) != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{name} must be one canonical absolute path"
        )
    path = Path(value)
    _reject_symlink_components(path)
    return path


def _canonical_ascii_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise StrictCapturedReplayEntrypointError(
            "value is not canonical finite ASCII JSON"
        ) from error


def _domain_sha256(label: str, value: Any) -> str:
    prefix = b"nemo-rl-strict-v2\x00" + label.encode("ascii") + b"\x00"
    return hashlib.sha256(prefix + _canonical_ascii_json(value)).hexdigest()


def _reject_negative_zero(value: Any, *, name: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            raise StrictCapturedReplayEntrypointError(
                f"{name} contains invalid numeric value"
            )
    elif isinstance(value, Mapping):
        for key, member in value.items():
            _reject_negative_zero(member, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            _reject_negative_zero(member, name=f"{name}[{index}]")


def _stat_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictCapturedReplayEntrypointError(f"{name} must be an object")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or _DIGEST_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{name} must be a nonzero lowercase SHA-256"
        )
    return value


def _job_id(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or _JOB_ID_RE.fullmatch(value) is None
        or not 0 < int(value) <= _MAX_INT63
    ):
        raise StrictCapturedReplayEntrypointError(
            f"{name} must be a canonical positive int63 string"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--replay-driver-phase", action="store_true")
    parser.add_argument("--replay-manifest", required=True)
    parser.add_argument("--replay-manifest-sha256", required=True)
    parser.add_argument("--pre-receipt", required=True)
    parser.add_argument("--pre-receipt-sha256", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--profile-id", required=True)
    return parser


def _require_isolated_driver_process() -> None:
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise StrictCapturedReplayEntrypointError(
            "replay driver requires live Python -I -B isolation"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.replay_driver_phase:
        raise StrictCapturedReplayEntrypointError(
            "direct replay execution is forbidden; authenticated wrapper driver phase required"
        )
    _require_isolated_driver_process()
    authority = _authenticate_before_repo_imports(
        manifest_path=args.replay_manifest,
        manifest_sha256=args.replay_manifest_sha256,
        pre_receipt_path=args.pre_receipt,
        pre_receipt_sha256=args.pre_receipt_sha256,
        expected_environment=args.environment,
        expected_profile_id=args.profile_id,
    )
    _run_authenticated_driver(authority)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StrictCapturedReplayEntrypointError, ValueError, RuntimeError) as error:
        raise SystemExit(f"strict captured replay failed: {error}") from error


__all__ = [
    "StrictCapturedReplayEntrypointError",
    "main",
    "run_from_authenticated_wrapper",
    "run_profiled_replay_with_authenticated_resource_child",
]
