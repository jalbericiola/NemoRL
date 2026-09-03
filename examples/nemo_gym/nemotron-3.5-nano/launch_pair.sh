#!/bin/bash -p
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

set -euo pipefail

# Threat boundary: direct use requires a caller that has already excluded
# dynamic-loader injection. A shell script cannot undo loader actions that
# happen before Bash starts. Submitted jobs instead cross the authenticated
# mode-0400 Slurm export-file boundary rendered below. In particular, a direct
# caller must leave POSIXLY_CORRECT unset because it changes parsing before this
# guard can execute.
# This pre-source guard is intentionally self-contained. It authenticates this
# entrypoint and its helper from the already SHA-anchored deployment manifest;
# sourcing unauthenticated shell code and validating it afterward is too late.
if [[ "$-" != *p* ]]; then
  echo "ERROR: launch_pair.sh must be executed directly through its privileged Bash shebang." >&2
  exit 2
fi
while IFS= read -r _strict_pair_environment_name; do
  case "${_strict_pair_environment_name}" in
    BASH_ENV|ENV|BASHOPTS|SHELLOPTS|BASH_COMPAT|CDPATH|GLOBIGNORE|BASH_XTRACEFD|\
    PYTHON*|GIT_*|LD_*|DYLD_*|BASH_FUNC_*%%)
      echo "ERROR: hostile startup environment variable must be unset: ${_strict_pair_environment_name}" >&2
      exit 2
      ;;
  esac
done < <(builtin compgen -e)
PATH=/usr/bin:/bin:/usr/sbin:/sbin
IFS=$' \t\n'
LC_ALL=C
export PATH LC_ALL
unset CDPATH GLOBIGNORE BASH_XTRACEFD
hash -r
umask 077

: "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
: "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256:?EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256 is required}"
: "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256:?EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256 is required}"
if [[ ! "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ || \
      ! "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: pre-source deployment anchors must be lowercase SHA-256 values." >&2
  exit 2
fi
if [[ -f /usr/bin/sha256sum && ! -L /usr/bin/sha256sum && \
      -x /usr/bin/sha256sum && ! -w /usr/bin/sha256sum ]]; then
  STRICT_PAIR_BOOTSTRAP_SHA256SUM=/usr/bin/sha256sum
elif [[ -f /sbin/sha256sum && ! -L /sbin/sha256sum && \
        -x /sbin/sha256sum && ! -w /sbin/sha256sum ]]; then
  STRICT_PAIR_BOOTSTRAP_SHA256SUM=/sbin/sha256sum
else
  echo "ERROR: no supported fixed bootstrap sha256sum is available." >&2
  exit 2
fi
if ! _strict_pair_hash_output="$(
  "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}"
)"; then
  echo "ERROR: bootstrap sha256sum failed while authenticating itself." >&2
  exit 2
fi
_strict_pair_actual_sha256="${_strict_pair_hash_output%% *}"
if [[ "${_strict_pair_actual_sha256}" != \
      "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" ]]; then
  echo "ERROR: bootstrap sha256sum SHA-256 mismatch." >&2
  exit 2
fi

_strict_pair_expected_source="${DEPLOYMENT_ROOT}/runnable/NemoRL/examples/nemo_gym/nemotron-3.5-nano/launch_pair.sh"
if [[ "${BASH_SOURCE[0]}" != "${_strict_pair_expected_source}" || \
      -L "${BASH_SOURCE[0]}" || ! -f "${BASH_SOURCE[0]}" || \
      ! -x "${BASH_SOURCE[0]}" || -w "${BASH_SOURCE[0]}" ]]; then
  echo "ERROR: launch_pair.sh must be the sealed deployment entrypoint at ${_strict_pair_expected_source}." >&2
  exit 2
fi
SCRIPT_DIR="${_strict_pair_expected_source%/*}"
PROJECT_ROOT="${DEPLOYMENT_ROOT}/runnable/NemoRL"
_strict_pair_contract="${SCRIPT_DIR}/strict_pair_contract.sh"
_strict_pair_nemo_manifest="${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256"
if [[ -L "${_strict_pair_nemo_manifest}" || ! -f "${_strict_pair_nemo_manifest}" ]]; then
  echo "ERROR: pre-source NeMo-RL runnable manifest must be a regular non-symlink file." >&2
  exit 2
fi
if ! _strict_pair_hash_output="$(
  "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- "${_strict_pair_nemo_manifest}"
)"; then
  echo "ERROR: bootstrap sha256sum failed for NeMo-RL runnable manifest." >&2
  exit 2
fi
_strict_pair_actual_sha256="${_strict_pair_hash_output%% *}"
if [[ "${_strict_pair_actual_sha256}" != \
      "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" ]]; then
  echo "ERROR: pre-source NeMo-RL runnable manifest SHA-256 mismatch." >&2
  exit 2
fi
_strict_pair_source_manifest_sha256=""
_strict_pair_contract_manifest_sha256=""
while IFS= read -r _strict_pair_manifest_line; do
  _strict_pair_manifest_sha256="${_strict_pair_manifest_line%%  *}"
  _strict_pair_manifest_path="${_strict_pair_manifest_line#*  }"
  case "${_strict_pair_manifest_path}" in
    "${BASH_SOURCE[0]}")
      [[ -z "${_strict_pair_source_manifest_sha256}" ]] || {
        echo "ERROR: duplicate launch_pair.sh runnable-manifest entry." >&2
        exit 2
      }
      _strict_pair_source_manifest_sha256="${_strict_pair_manifest_sha256}"
      ;;
    "${_strict_pair_contract}")
      [[ -z "${_strict_pair_contract_manifest_sha256}" ]] || {
        echo "ERROR: duplicate strict_pair_contract.sh runnable-manifest entry." >&2
        exit 2
      }
      _strict_pair_contract_manifest_sha256="${_strict_pair_manifest_sha256}"
      ;;
  esac
done < "${_strict_pair_nemo_manifest}"
for _strict_pair_guard_path in "${BASH_SOURCE[0]}" "${_strict_pair_contract}"; do
  if [[ -L "${_strict_pair_guard_path}" || ! -f "${_strict_pair_guard_path}" || \
        ! -x "${_strict_pair_guard_path}" || -w "${_strict_pair_guard_path}" ]]; then
    echo "ERROR: pre-source strict-pair code must be regular, executable, and sealed: ${_strict_pair_guard_path}" >&2
    exit 2
  fi
done
if ! _strict_pair_hash_output="$(
  "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- "${BASH_SOURCE[0]}"
)"; then
  echo "ERROR: bootstrap sha256sum failed for launch_pair.sh." >&2
  exit 2
fi
_strict_pair_actual_sha256="${_strict_pair_hash_output%% *}"
if [[ -z "${_strict_pair_source_manifest_sha256}" || \
      "${_strict_pair_actual_sha256}" != "${_strict_pair_source_manifest_sha256}" ]]; then
  echo "ERROR: launch_pair.sh is absent or drifted in the authenticated runnable manifest." >&2
  exit 2
fi
if ! _strict_pair_hash_output="$(
  "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- "${_strict_pair_contract}"
)"; then
  echo "ERROR: bootstrap sha256sum failed for strict_pair_contract.sh." >&2
  exit 2
fi
_strict_pair_actual_sha256="${_strict_pair_hash_output%% *}"
if [[ -z "${_strict_pair_contract_manifest_sha256}" || \
      "${_strict_pair_actual_sha256}" != "${_strict_pair_contract_manifest_sha256}" ]]; then
  echo "ERROR: strict_pair_contract.sh is absent or drifted in the authenticated runnable manifest." >&2
  exit 2
fi
export STRICT_PAIR_BOOTSTRAP_SHA256SUM

ARM_WRAPPER="${SCRIPT_DIR}/nano35_single_env_pair.sh"

# shellcheck source=strict_pair_contract.sh
source "${_strict_pair_contract}"
strict_pair_select_environment
strict_pair_load_runtime_tools
SCRIPT_DIR="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${SCRIPT_DIR}")"
PROJECT_ROOT="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${PROJECT_ROOT}")"
unset _strict_pair_actual_sha256 _strict_pair_contract \
  _strict_pair_contract_manifest_sha256 _strict_pair_environment_name \
  _strict_pair_expected_source _strict_pair_hash_output \
  _strict_pair_guard_path _strict_pair_manifest_line \
  _strict_pair_manifest_path _strict_pair_manifest_sha256 \
  _strict_pair_nemo_manifest _strict_pair_source_manifest_sha256

if (( $# != 1 )) || [[ "$1" != "--dry-run" && "$1" != "--submit" ]]; then
  echo "Usage: STRICT_PAIR_ENVIRONMENT=<reasoning_gym|citation|freeform> PAIR_ID=<id> ${BASH_SOURCE[0]} <--dry-run|--submit>" >&2
  exit 2
fi
case "$1" in
  --dry-run)
    STRICT_PAIR_LAUNCH_MODE="dry-run"
    ;;
  --submit)
    STRICT_PAIR_LAUNCH_MODE="submit"
    ;;
esac

: "${PAIR_ID:?PAIR_ID is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required; strict pair runs must log to W&B}"
: "${TRAIN_PATH:?TRAIN_PATH is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${HF_HOME:?HF_HOME is required; the strict pair derives disjoint per-arm Hugging Face cache roots}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
: "${STRICT_PAIR_JOB_WRAPPER:?STRICT_PAIR_JOB_WRAPPER is required}"
: "${STRICT_PAIR_SLURM_CONF:?STRICT_PAIR_SLURM_CONF is required}"
: "${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256:?EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256 is required}"
: "${EXPECTED_NEMO_HEAD:?EXPECTED_NEMO_HEAD is required}"
: "${EXPECTED_GYM_GITLINK_COMMIT:?EXPECTED_GYM_GITLINK_COMMIT is required}"
: "${EXPECTED_BRIDGE_HEAD:?EXPECTED_BRIDGE_HEAD is required}"
: "${EXPECTED_BRIDGE_TREE:?EXPECTED_BRIDGE_TREE is required}"
: "${EXPECTED_MCORE_HEAD:?EXPECTED_MCORE_HEAD is required}"
: "${EXPECTED_MCORE_TREE:?EXPECTED_MCORE_TREE is required}"

if [[ ! "${PAIR_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || (( ${#PAIR_ID} > 64 )); then
  strict_pair_error "PAIR_ID must be a 1-64 character filesystem-safe identifier."
fi
if ! "${STRICT_PAIR_TOOL_PYTHON}" -I -B -c '
import sys
import re

value = sys.argv[1]
if re.fullmatch(r"/[A-Za-z0-9._/+:-]+", value) is None:
    raise SystemExit(1)
' "${RESULTS_DIR}"; then
  strict_pair_error "RESULTS_DIR must be an absolute machine-record-safe ASCII path."
fi
if [[ -n "${EXPECTED_PAIR_MANIFEST_SHA256+x}" ]]; then
  strict_pair_error "EXPECTED_PAIR_MANIFEST_SHA256 is parent-owned output and must be unset."
fi
if [[ -n "${CODE_SNAPSHOT_DIRNAME+x}" ]]; then
  strict_pair_error "CODE_SNAPSHOT_DIRNAME must be unset; the strict pair owns its fresh snapshot parent."
fi
if [[ -n "${USE_SNAPSHOT+x}" && "${USE_SNAPSHOT}" != "1" ]]; then
  strict_pair_error "USE_SNAPSHOT must be unset or 1; live-tree/container fallback is forbidden."
fi
if [[ "${ENABLE_MTP_INFERENCE:-0}" != "0" ]]; then
  strict_pair_error "ENABLE_MTP_INFERENCE must be unset or 0; generation speculative decoding is off."
fi
if [[ -n "${NRL_MAX_STEPS+x}" ]]; then
  strict_pair_error "NRL_MAX_STEPS must be unset; the strict pair pins exactly 100 optimizer steps."
fi
if [[ -n "${TRITON_CACHE_AUTOTUNING+x}" ]]; then
  strict_pair_error "TRITON_CACHE_AUTOTUNING must be unset for deterministic execution."
fi
while IFS= read -r forbidden_name; do
  strict_pair_error "${forbidden_name} must be unset for deterministic execution."
done < <(builtin compgen -A variable TRITON_AUTOTUNE_BLOCK || true)
if [[ -n "${WANDB_DISABLED+x}" ]]; then
  strict_pair_error "WANDB_DISABLED must be unset; strict pair evidence requires online W&B."
fi
if [[ -n "${WANDB_MODE+x}" && "${WANDB_MODE}" != "online" ]]; then
  strict_pair_error "WANDB_MODE must be unset or 'online'; got '${WANDB_MODE}'."
fi
if [[ -n "${WANDB_RUN_ID+x}" ]]; then
  strict_pair_error "WANDB_RUN_ID must be unset; each arm derives a fresh isolated W&B identity."
fi
for forbidden_name in \
  BASE_LOG_DIR BATCH_SCRIPT CPUS_PER_WORKER RAY_SUB MOUNTS EXTRA_MOUNTS \
  USE_CUSTOM_VLLM INTERACTIVE NEMO_SKILLS_SANDBOX_PORT \
  RAY_LOG_SYNC_FREQUENCY SANDBOX_COMMAND SETUP_COMMAND SLURM_SUBMIT_DIR \
  CHECKPOINTING_SAVE_BY WALLTIME SLURM_QOS SLURM_ACCOUNT DRY_RUN \
  WANDB_ENTITY WANDB_NAME WANDB_PROJ WANDB_RESUME WANDB_RUN_GROUP \
  HF_HUB_CACHE HF_DATASETS_CACHE; do
  if [[ -n "${!forbidden_name+x}" ]]; then
    strict_pair_error "${forbidden_name} must be unset; the strict pair owns this launcher input."
  fi
done
unset forbidden_name

strict_pair_prepare_contract "${PROJECT_ROOT}" "${SCRIPT_DIR}"

if [[ -e "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" || \
      -L "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" ]]; then
  strict_pair_error "STRICT_PAIR_SUBMISSION_CONTRACT.json must be absent before strict pair preparation."
fi
if [[ -e "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" || \
      -L "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" ]]; then
  strict_pair_error "PAIR_SUBMISSION_RECEIPT.json must be absent before any strict pair submission."
fi
STRICT_PAIR_SUBMISSION_NONCE="$(strict_pair_generate_submission_nonce)"
if [[ ! "${STRICT_PAIR_SUBMISSION_NONCE}" =~ ^[0-9a-f]{64}$ ]]; then
  strict_pair_error "generated strict pair submission nonce is malformed."
fi
strict_pair_publish_submission_contract

strict_pair_build_snapshot off
strict_pair_build_snapshot on
"${STRICT_PAIR_TOOL_MKDIR}" -p -- "${STRICT_PAIR_SLURM_EXPORT_PARENT}"
"${STRICT_PAIR_TOOL_CHMOD}" 700 "${STRICT_PAIR_SLURM_EXPORT_PARENT}"
strict_pair_require_canonical_dir \
  "${STRICT_PAIR_SLURM_EXPORT_PARENT}" "strict Slurm export parent"
strict_pair_require_mode \
  "${STRICT_PAIR_SLURM_EXPORT_PARENT}" "700" "strict Slurm export parent"

off_stdout=""
off_stderr=""
on_stdout=""
on_stderr=""
STRICT_PAIR_RECOVERY_SCAN_PATH=""
STRICT_PAIR_RECOVERY_SCAN_OWNED=0
STRICT_PAIR_RECOVERY_SCAN_FD_OPEN=0
STRICT_PAIR_ACCEPTED_ID_FDS_OPEN=0
STRICT_PAIR_OFF_ACCEPTED_ID_FD=8
STRICT_PAIR_ON_ACCEPTED_ID_FD=9
close_strict_pair_accepted_id_fds() {
  if [[ "${STRICT_PAIR_ACCEPTED_ID_FDS_OPEN}" == "1" ]]; then
    exec 8>&-
    exec 9>&-
    STRICT_PAIR_ACCEPTED_ID_FDS_OPEN=0
  fi
}
cleanup_pair_temporary_files() {
  close_strict_pair_accepted_id_fds
  "${STRICT_PAIR_TOOL_RM}" -f -- \
    "${STRICT_PAIR_OFF_SLURM_EXPORT}" "${STRICT_PAIR_ON_SLURM_EXPORT}"
  [[ -z "${off_stdout}" ]] || "${STRICT_PAIR_TOOL_RM}" -f -- "${off_stdout}"
  [[ -z "${off_stderr}" ]] || "${STRICT_PAIR_TOOL_RM}" -f -- "${off_stderr}"
  [[ -z "${on_stdout}" ]] || "${STRICT_PAIR_TOOL_RM}" -f -- "${on_stdout}"
  [[ -z "${on_stderr}" ]] || "${STRICT_PAIR_TOOL_RM}" -f -- "${on_stderr}"
  if [[ "${STRICT_PAIR_RECOVERY_SCAN_OWNED}" == "1" && \
        -n "${STRICT_PAIR_RECOVERY_SCAN_PATH}" ]]; then
    strict_pair_cleanup_recovery_scan 2>/dev/null || true
  fi
}
trap cleanup_pair_temporary_files EXIT
off_stdout="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.strict-pair-off.stdout.XXXXXX")"
off_stderr="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.strict-pair-off.stderr.XXXXXX")"
on_stdout="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.strict-pair-on.stdout.XXXXXX")"
on_stderr="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.strict-pair-on.stderr.XXXXXX")"

run_arm_phase() {
  local arm="$1"
  local prepare_only="$2"
  local stdout_path="$3"
  local stderr_path="$4"
  # Bash 3 treats an empty-array expansion as unbound under `set -u`.
  local pair_anchors=("STRICT_PAIR_PARENT_PHASE=${prepare_only}")

  if [[ "${prepare_only}" == "0" ]]; then
    pair_anchors+=(
      "EXPECTED_PAIR_MANIFEST_SHA256=${STRICT_PAIR_MANIFEST_SHA256}"
      "EXPECTED_STRICT_PAIR_OFF_SLURM_EXPORT_SHA256=${STRICT_PAIR_OFF_SLURM_EXPORT_SHA256}"
      "EXPECTED_STRICT_PAIR_ON_SLURM_EXPORT_SHA256=${STRICT_PAIR_ON_SLURM_EXPORT_SHA256}"
    )
    if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "submit" ]]; then
      if [[ "${arm}" == "off" ]]; then
        exec 9>&-
        pair_anchors+=(
          "STRICT_PAIR_ACCEPTED_ID_FD=${STRICT_PAIR_OFF_ACCEPTED_ID_FD}"
          "STRICT_PAIR_ACCEPTED_ID_RECORD=${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}"
        )
      else
        exec 8>&-
        pair_anchors+=(
          "STRICT_PAIR_ACCEPTED_ID_FD=${STRICT_PAIR_ON_ACCEPTED_ID_FD}"
          "STRICT_PAIR_ACCEPTED_ID_RECORD=${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}"
        )
      fi
    fi
  fi
  exec "${STRICT_PAIR_TOOL_ENV}" -i \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    LC_ALL=C \
    PAIR_ID="${PAIR_ID}" \
    WANDB_API_KEY="${WANDB_API_KEY}" \
    HF_TOKEN="${HF_TOKEN:-}" \
    TRAIN_PATH="${TRAIN_PATH}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    PERSISTENT_CACHE="${PERSISTENT_CACHE}" \
    HF_HOME="${HF_HOME}" \
    MODEL_PATH="${MODEL_PATH}" \
    CONTAINER="${CONTAINER}" \
    SANDBOX_CONTAINER="${SANDBOX_CONTAINER}" \
    DEPLOYMENT_ROOT="${DEPLOYMENT_ROOT}" \
    STRICT_PAIR_JOB_WRAPPER="${STRICT_PAIR_JOB_WRAPPER}" \
    STRICT_PAIR_SLURM_CONF="${STRICT_PAIR_SLURM_CONF}" \
    EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256="${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256}" \
    STRICT_PAIR_SUBMISSION_NONCE="${STRICT_PAIR_SUBMISSION_NONCE}" \
    STRICT_PAIR_SUBMISSION_CONTRACT_SHA256="${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" \
    EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256="${EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256}" \
    EXPECTED_NEMO_HEAD="${EXPECTED_NEMO_HEAD}" \
    EXPECTED_GYM_GITLINK_COMMIT="${EXPECTED_GYM_GITLINK_COMMIT}" \
    EXPECTED_GYM_TREE="${EXPECTED_GYM_TREE}" \
    EXPECTED_BRIDGE_HEAD="${EXPECTED_BRIDGE_HEAD}" \
    EXPECTED_BRIDGE_TREE="${EXPECTED_BRIDGE_TREE}" \
    EXPECTED_MCORE_HEAD="${EXPECTED_MCORE_HEAD}" \
    EXPECTED_MCORE_TREE="${EXPECTED_MCORE_TREE}" \
    EXPECTED_DEPLOYMENT_READY="${EXPECTED_DEPLOYMENT_READY}" \
    EXPECTED_DEPLOYMENT_READY_FILE_SHA256="${EXPECTED_DEPLOYMENT_READY_FILE_SHA256}" \
    EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256="${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" \
    EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256="${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256}" \
    EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256="${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256}" \
    EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256="${STRICT_PAIR_MODEL_TREE_SHA256}" \
    EXPECTED_STRICT_PAIR_CONTAINER_SHA256="${STRICT_PAIR_CONTAINER_SHA256}" \
    EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256="${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" \
    EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256="${STRICT_PAIR_ACCEPTANCE_POLICY_SHA256}" \
    STRICT_PAIR_RUNTIME_TOOL_MANIFEST="${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
    EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256="${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}" \
    STRICT_PAIR_HOST_PYTHON="${STRICT_PAIR_HOST_PYTHON}" \
    EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256="${EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256}" \
    EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256="${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" \
    STRICT_PAIR_CONTAINER_PYTHON="${STRICT_PAIR_CONTAINER_PYTHON}" \
    EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256="${EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256}" \
    STRICT_PAIR_CONTAINER_UV="${STRICT_PAIR_CONTAINER_UV}" \
    EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256="${EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256}" \
    STRICT_PAIR_UV_SHIM="${STRICT_PAIR_UV_SHIM}" \
    EXPECTED_STRICT_PAIR_UV_SHIM_SHA256="${EXPECTED_STRICT_PAIR_UV_SHIM_SHA256}" \
    STRICT_PAIR_ENVIRONMENT="${STRICT_PAIR_ENVIRONMENT}" \
    STRICT_PAIR_LAUNCH_MODE="${STRICT_PAIR_LAUNCH_MODE}" \
    STRICT_PAIR_EXPORT_PREPARE_ONLY="${prepare_only}" \
    "${pair_anchors[@]}" \
    "${STRICT_PAIR_TOOL_BASH}" -p "${ARM_WRAPPER}" "${arm}" \
    >"${stdout_path}" 2>"${stderr_path}"
}

run_arm_phase off 1 "${off_stdout}" "${off_stderr}" &
off_pid=$!
run_arm_phase on 1 "${on_stdout}" "${on_stderr}" &
on_pid=$!

off_status=0
on_status=0
wait "${off_pid}" || off_status=$?
wait "${on_pid}" || on_status=$?
if (( off_status != 0 || on_status != 0 )); then
  "${STRICT_PAIR_TOOL_CAT}" -- "${off_stdout}" "${on_stdout}"
  "${STRICT_PAIR_TOOL_CAT}" -- "${off_stderr}" "${on_stderr}" >&2
  strict_pair_error "strict pair export preparation failed: off_status=${off_status} on_status=${on_status}"
fi
strict_pair_require_canonical_file \
  "${STRICT_PAIR_OFF_SLURM_EXPORT}" "off strict Slurm export file"
strict_pair_require_canonical_file \
  "${STRICT_PAIR_ON_SLURM_EXPORT}" "on strict Slurm export file"
strict_pair_require_mode \
  "${STRICT_PAIR_OFF_SLURM_EXPORT}" "400" "off strict Slurm export file"
strict_pair_require_mode \
  "${STRICT_PAIR_ON_SLURM_EXPORT}" "400" "on strict Slurm export file"
STRICT_PAIR_OFF_SLURM_EXPORT_SHA256="$(
  strict_pair_sha256_file "${STRICT_PAIR_OFF_SLURM_EXPORT}"
)"
STRICT_PAIR_ON_SLURM_EXPORT_SHA256="$(
  strict_pair_sha256_file "${STRICT_PAIR_ON_SLURM_EXPORT}"
)"

strict_pair_publish_manifest
echo "STRICT_PAIR_MANIFEST_SHA256=${STRICT_PAIR_MANIFEST_SHA256} path=${STRICT_PAIR_MANIFEST_PATH}"
echo "STRICT_PAIR_PLAN pair_id=${PAIR_ID} environment=${STRICT_PAIR_ENVIRONMENT} config=${STRICT_PAIR_CONFIG_RELATIVE} arms=off:observe,on:train mode=${STRICT_PAIR_LAUNCH_MODE} submissions=parallel partition=batch export_boundary=v3"

prepare_strict_pair_accepted_id_records() {
  local state_root="${RESULTS_DIR}/strict_pair_submission_state"

  if [[ -L "${state_root}" || \
        ( -e "${state_root}" && ! -d "${state_root}" ) ]]; then
    strict_pair_error "strict pair submission-state root must be a real directory."
  fi
  "${STRICT_PAIR_TOOL_MKDIR}" -p -- "${state_root}"
  "${STRICT_PAIR_TOOL_CHMOD}" 700 "${state_root}"
  strict_pair_require_canonical_dir \
    "${state_root}" "strict pair submission-state root"
  strict_pair_require_mode \
    "${state_root}" "700" "strict pair submission-state root"
  if [[ -e "${STRICT_PAIR_SUBMISSION_STATE_PARENT}" || \
        -L "${STRICT_PAIR_SUBMISSION_STATE_PARENT}" ]]; then
    strict_pair_error "strict pair submission-state directory must be fresh."
  fi
  "${STRICT_PAIR_TOOL_MKDIR}" -- "${STRICT_PAIR_SUBMISSION_STATE_PARENT}"
  "${STRICT_PAIR_TOOL_CHMOD}" 700 "${STRICT_PAIR_SUBMISSION_STATE_PARENT}"
  strict_pair_require_canonical_dir \
    "${STRICT_PAIR_SUBMISSION_STATE_PARENT}" \
    "strict pair submission-state directory"
  strict_pair_require_mode \
    "${STRICT_PAIR_SUBMISSION_STATE_PARENT}" "700" \
    "strict pair submission-state directory"

  set -o noclobber
  if ! exec 8>"${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}"; then
    set +o noclobber
    strict_pair_error "failed to exclusively create OFF accepted-ID record."
  fi
  if ! exec 9>"${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}"; then
    set +o noclobber
    exec 8>&-
    strict_pair_error "failed to exclusively create ON accepted-ID record."
  fi
  set +o noclobber
  STRICT_PAIR_ACCEPTED_ID_FDS_OPEN=1
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${STRICT_PAIR_SUBMISSION_STATE_PARENT}" \
    "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" \
    "${STRICT_PAIR_OFF_ACCEPTED_ID_FD}" \
    "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" \
    "${STRICT_PAIR_ON_ACCEPTED_ID_FD}" <<'PY'
import os
import pathlib
import stat
import sys

parent = pathlib.Path(sys.argv[1])
records = ((pathlib.Path(sys.argv[2]), int(sys.argv[3])),
           (pathlib.Path(sys.argv[4]), int(sys.argv[5])))
for path, fd in records:
    path_stat = os.lstat(path)
    fd_stat = os.fstat(fd)
    if (
        not path.is_absolute()
        or os.path.realpath(path) != str(path)
        or stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)
        or path_stat.st_nlink != 1
        or stat.S_IMODE(path_stat.st_mode) != 0o600
    ):
        raise SystemExit("accepted-ID record did not preserve its exclusive inode")
    os.fsync(fd)
directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

off_job_id=""
on_job_id=""
off_status=0
on_status=0
off_held_output_sha256=""
on_held_output_sha256=""
off_accepted_id_record_sha256=""
on_accepted_id_record_sha256=""
off_job_id_source="none"
on_job_id_source="none"
off_record_job_id=""
on_record_job_id=""
off_sbatch_status="-"
on_sbatch_status="-"
off_relay_status="-"
on_relay_status="-"
off_writer_drained="false"
on_writer_drained="false"
off_started_unix_ns="-"
on_started_unix_ns="-"
off_drained_unix_ns="-"
on_drained_unix_ns="-"
pre_release_query="null"
release_status="-"
release_output_sha256="-"
post_release_query="null"
cancel_status="-"
cancel_output_sha256="-"
cancel_job_ids_csv="-"
cancel_query_status="-"
cancel_query_output_sha256="-"
cancel_query_records="null"
post_cancel_queries="[]"
pre_cancel_query_status="-"
pre_cancel_query_output_sha256="-"
pre_cancel_query_records="null"
pre_cancel_job_ids_csv="-"
pre_cancel_queries="[]"
pre_cancel_authenticated_job_ids_csv="-"
pre_cancel_unresolved_job_ids_csv="-"
cancel_query_authenticated_job_ids_csv="-"
cancel_query_unresolved_job_ids_csv="-"
recovery_scan_status="-"
recovery_scan_normalization_status="-"
recovery_scan_output_sha256="-"
recovery_scan_byte_count="-"
recovery_scan_line_count="-"
recovery_scan_parsed_record_count="-"
recovery_scan_identity_match_counts="null"
recovery_scan_records="null"
recovery_scan_unterminated_final_line="-"
recovery_scan_unterminated_candidate_job_ids="null"
unterminated_candidate_job_ids_csv="-"
recovery_scan_securely_unlinked="-"
recovery_scan_authenticated_job_ids_csv=""
recovery_scan_matched_job_ids_csv=""
off_recovery_authenticated_job_ids_csv="-"
on_recovery_authenticated_job_ids_csv="-"
off_cleanup_candidate_job_ids_csv=""
on_cleanup_candidate_job_ids_csv=""

extract_strict_held_record() {
  local arm="$1"
  local output="$2"
  local prefix="STRICT_PAIR_HELD arm=${arm} job_id="
  local digest_prefix="job_id_sha256_ascii_no_newline="
  local record_digest_prefix="accepted_id_record_sha256="
  local sbatch_status_prefix="sbatch_status="
  local relay_status_prefix="relay_status="
  local writer_drained_prefix="writer_drained="
  local started_prefix="started_unix_ns="
  local drained_prefix="drained_unix_ns="
  local line
  local found=""
  local job_id
  local remainder
  local output_sha256
  local record_sha256
  local sbatch_status
  local relay_status
  local writer_drained
  local started_unix_ns
  local drained_unix_ns

  while IFS= read -r line; do
    case "${line}" in
      "${prefix}"*)
        [[ -z "${found}" ]] || return 2
        remainder="${line#${prefix}}"
        job_id="${remainder%% *}"
        remainder="${remainder#* }"
        case "${remainder}" in
          "${digest_prefix}"*)
            output_sha256="${remainder#${digest_prefix}}"
            output_sha256="${output_sha256%% *}"
            remainder="${remainder#* }"
            ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${record_digest_prefix}"*)
            record_sha256="${remainder#${record_digest_prefix}}"
            record_sha256="${record_sha256%% *}"
            remainder="${remainder#* }"
          ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${sbatch_status_prefix}"*)
            sbatch_status="${remainder#${sbatch_status_prefix}}"
            sbatch_status="${sbatch_status%% *}"
            remainder="${remainder#* }"
            ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${relay_status_prefix}"*)
            relay_status="${remainder#${relay_status_prefix}}"
            relay_status="${relay_status%% *}"
            remainder="${remainder#* }"
            ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${writer_drained_prefix}"*)
            writer_drained="${remainder#${writer_drained_prefix}}"
            writer_drained="${writer_drained%% *}"
            remainder="${remainder#* }"
            ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${started_prefix}"*)
            started_unix_ns="${remainder#${started_prefix}}"
            started_unix_ns="${started_unix_ns%% *}"
            remainder="${remainder#* }"
            ;;
          *) return 2 ;;
        esac
        case "${remainder}" in
          "${drained_prefix}"*)
            drained_unix_ns="${remainder#${drained_prefix}}"
            ;;
          *) return 2 ;;
        esac
        [[ "${job_id}" =~ ^[0-9]+$ ]] || return 2
        [[ "${output_sha256}" =~ ^[0-9a-f]{64}$ ]] || return 2
        [[ "${record_sha256}" =~ ^[0-9a-f]{64}$ ]] || return 2
        [[ "${sbatch_status}" =~ ^[0-9]+$ ]] || return 2
        [[ "${relay_status}" =~ ^[0-9]+$ ]] || return 2
        [[ "${writer_drained}" == "true" ]] || return 2
        [[ "${started_unix_ns}" =~ ^[1-9][0-9]*$ ]] || return 2
        [[ "${drained_unix_ns}" =~ ^[1-9][0-9]*$ ]] || return 2
        (( drained_unix_ns >= started_unix_ns )) || return 2
        found="${job_id} ${output_sha256} ${record_sha256} ${sbatch_status} ${relay_status} ${writer_drained} ${started_unix_ns} ${drained_unix_ns}"
        ;;
    esac
  done < "${output}"
  [[ -n "${found}" ]] || return 2
  printf '%s\n' "${found}"
}

strict_pair_seal_and_read_accepted_id_record() {
  local record_path="$1"

  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "${record_path}" <<'PY'
import hashlib
import os
import pathlib
import re
import stat
import sys

path = pathlib.Path(sys.argv[1])
metadata = os.lstat(path)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
):
    raise SystemExit("accepted-ID record is not one canonical sealed/in-progress file")
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("accepted-ID record inode changed while opening")
    chunks = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks)
    if stat.S_IMODE(opened.st_mode) == 0o600:
        os.fchmod(fd, 0o400)
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
if re.fullmatch(rb"[1-9][0-9]*\n", payload) is None:
    raise SystemExit("accepted-ID record is not one positive ASCII decimal plus LF")
job_id = payload[:-1].decode("ascii")
print(
    job_id,
    hashlib.sha256(job_id.encode("ascii")).hexdigest(),
    hashlib.sha256(payload).hexdigest(),
)
PY
}

strict_pair_capture_accepted_id_records() {
  local record

  close_strict_pair_accepted_id_fds
  record="$(
    strict_pair_seal_and_read_accepted_id_record \
      "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" 2>/dev/null
  )" || record=""
  if [[ -n "${record}" ]]; then
    read -r off_job_id off_held_output_sha256 \
      off_accepted_id_record_sha256 <<< "${record}"
    off_record_job_id="${off_job_id}"
    off_job_id_source="accepted-id-record"
  elif [[ -f "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" && \
          ! -L "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" ]]; then
    off_accepted_id_record_sha256="$(
      strict_pair_sha256_file "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}"
    )" || off_accepted_id_record_sha256=""
  fi
  record="$(
    strict_pair_seal_and_read_accepted_id_record \
      "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" 2>/dev/null
  )" || record=""
  if [[ -n "${record}" ]]; then
    read -r on_job_id on_held_output_sha256 \
      on_accepted_id_record_sha256 <<< "${record}"
    on_record_job_id="${on_job_id}"
    on_job_id_source="accepted-id-record"
  elif [[ -f "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" && \
          ! -L "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" ]]; then
    on_accepted_id_record_sha256="$(
      strict_pair_sha256_file "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}"
    )" || on_accepted_id_record_sha256=""
  fi
}

strict_pair_canonical_job_id_union() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B -c '
import re
import sys

first = None
values = []
for index, raw in enumerate(sys.argv[1:]):
    if raw in {"", "-"}:
        continue
    split = raw.split(",")
    if index == 0 and len(split) == 1:
        first = split[0]
    values.extend(split)
if any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values):
    raise SystemExit("candidate job-ID union contains malformed values")
result = [] if first is None else [first]
result.extend(sorted(set(values) - set(result), key=int))
print(",".join(result))
' "$@"
}

strict_pair_recover_missing_accepted_ids() {
  local scan_result=""
  local off_recovered_job_ids_csv="-"
  local on_recovered_job_ids_csv="-"
  local recovered_job_id

  STRICT_PAIR_RECOVERY_SCAN_PATH="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/scheduler-recovery.scan"
  if [[ -e "${STRICT_PAIR_RECOVERY_SCAN_PATH}" || \
        -L "${STRICT_PAIR_RECOVERY_SCAN_PATH}" ]]; then
    recovery_scan_status="-"
    return
  fi
  set -o noclobber
  if exec 7> "${STRICT_PAIR_RECOVERY_SCAN_PATH}"; then
    STRICT_PAIR_RECOVERY_SCAN_FD_OPEN=1
    STRICT_PAIR_RECOVERY_SCAN_OWNED=1
  else
    set +o noclobber
    recovery_scan_status="-"
    return
  fi
  set +o noclobber
  if strict_pair_scheduler_client \
      "${STRICT_PAIR_TOOL_SCONTROL}" show job --oneliner \
      >&7 2>/dev/null; then
    recovery_scan_status=0
  else
    recovery_scan_status=$?
  fi
  if [[ ! -e "${STRICT_PAIR_RECOVERY_SCAN_PATH}" || \
        -L "${STRICT_PAIR_RECOVERY_SCAN_PATH}" ]]; then
    recovery_scan_status="-"
    return
  fi
  if scan_result="$(
    "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
      "${STRICT_PAIR_RECOVERY_SCAN_PATH}" "7" "${PAIR_ID}" \
      "${STRICT_PAIR_ENVIRONMENT}" \
      "${STRICT_PAIR_SUBMISSION_NONCE}" \
      "${STRICT_PAIR_MANIFEST_SHA256}" "${EUID}" \
      "${STRICT_PAIR_OFF_SNAPSHOT}" \
      "${STRICT_PAIR_ON_SNAPSHOT}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

path = pathlib.Path(sys.argv[1])
writer_fd = int(sys.argv[2])
(
    pair_id,
    environment,
    nonce,
    pair_manifest_sha256,
    expected_user_id,
    off_snapshot_dir,
    on_snapshot_dir,
) = sys.argv[3:]
expected_work_dirs = {"off": off_snapshot_dir, "on": on_snapshot_dir}
metadata = os.lstat(path)
writer_metadata = os.fstat(writer_fd)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or stat.S_IMODE(metadata.st_mode) != 0o600
    or metadata.st_uid != int(expected_user_id)
    or (metadata.st_dev, metadata.st_ino)
    != (writer_metadata.st_dev, writer_metadata.st_ino)
):
    raise SystemExit("scheduler recovery scan is not one private regular file")
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("scheduler recovery scan inode changed while opening")
    os.fsync(writer_fd)
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    parsed_record_count = 0
    unterminated_final_line = False
    matches = {"off": [], "on": []}
    unterminated_candidate_job_ids = []
    required_keys = {
        "Comment",
        "JobId",
        "JobName",
        "JobState",
        "Reason",
        "UserId",
        "WorkDir",
    }
    with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
        for raw_line in stream:
            digest.update(raw_line)
            byte_count += len(raw_line)
            line_count += 1
            if not raw_line.endswith(b"\n"):
                unterminated_final_line = True
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError:
                continue
            fields = {}
            duplicate_keys = set()
            for token in line.split():
                key, separator, value = token.partition("=")
                if separator:
                    if key in fields:
                        duplicate_keys.add(key)
                    fields[key] = value
            if not raw_line.endswith(b"\n"):
                candidate_job_id = fields.get("JobId", "")
                if (
                    "JobId" not in duplicate_keys
                    and re.fullmatch(r"[1-9][0-9]*", candidate_job_id)
                    is not None
                    and candidate_job_id not in unterminated_candidate_job_ids
                ):
                    unterminated_candidate_job_ids.append(candidate_job_id)
                continue
            user_match = re.fullmatch(
                r"[^()\s]+\(([0-9]+)\)", fields.get("UserId", "")
            )
            if (
                required_keys <= set(fields)
                and not (required_keys & duplicate_keys)
                and re.fullmatch(r"[1-9][0-9]*", fields.get("JobId", ""))
                is not None
                and bool(fields.get("JobName"))
                and bool(fields.get("Comment"))
                and bool(fields.get("JobState"))
                and bool(fields.get("Reason"))
                and user_match is not None
            ):
                parsed_record_count += 1
            for arm in ("off", "on"):
                expected_name = f"{arm}-{environment}-{pair_id}"
                expected_comment = (
                    f"nemo-rl-strict-pair-v1:{arm}:{nonce}:"
                    f"{pair_manifest_sha256}"
                )
                if (
                    fields.get("JobName") == expected_name
                    and fields.get("Comment") == expected_comment
                    and user_match is not None
                    and user_match.group(1) == expected_user_id
                ):
                    matches[arm].append(
                        (
                            fields,
                            user_match.group(1),
                            bool(required_keys & duplicate_keys),
                        )
                    )
    os.fchmod(writer_fd, 0o400)
    os.fsync(writer_fd)
finally:
    os.close(fd)
directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("scheduler recovery scan path changed before unlink")
    os.unlink(path.name, dir_fd=directory_fd)
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)

records = {"off": [], "on": []}
job_ids = {"off": [], "on": []}
match_counts = {"off": 0, "on": 0}
for arm, values in matches.items():
    for fields, user_id, has_duplicate_required_key in values:
        job_id = fields.get("JobId", "")
        state = fields.get("JobState")
        reason = fields.get("Reason")
        if (
            re.fullmatch(r"[1-9][0-9]*", job_id) is not None
            and job_id not in job_ids[arm]
        ):
            # This remains a candidate until a fresh, completely framed query
            # authenticates it.  Preserve it even when the recovery row has
            # malformed state fields so cleanup does not strand a pair job.
            job_ids[arm].append(job_id)
        if (
            has_duplicate_required_key
            or re.fullmatch(r"[1-9][0-9]*", job_id) is None
            or not state
            or reason is None
            or fields.get("WorkDir") != expected_work_dirs[arm]
            or state != "PENDING"
            or reason != "JobHeldUser"
        ):
            continue
        match_counts[arm] += 1
        record = {
            "comment": fields["Comment"],
            "held": state == "PENDING" and reason == "JobHeldUser",
            "job_id": job_id,
            "job_name": fields["JobName"],
            "job_state": state,
            "reason": reason,
            "user_id": user_id,
            "work_dir": fields["WorkDir"],
        }
        records[arm].append(record)
    records[arm].sort(
        key=lambda record: (
            int(record["job_id"]),
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )
    )
    job_ids[arm].sort(key=int)
unterminated_candidate_job_ids.sort(key=int)
print(
    "\t".join(
        (
            digest.hexdigest(),
            "true" if unterminated_final_line else "false",
            str(byte_count),
            str(line_count),
            str(parsed_record_count),
            json.dumps(match_counts, sort_keys=True, separators=(",", ":")),
            ",".join(job_ids["off"]) or "-",
            ",".join(job_ids["on"]) or "-",
            json.dumps(records, sort_keys=True, separators=(",", ":")),
            ",".join(unterminated_candidate_job_ids) or "-",
            json.dumps(
                unterminated_candidate_job_ids,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "true",
        )
    )
)
PY
  )"; then
    recovery_scan_normalization_status=0
    IFS=$'\t' read -r \
      recovery_scan_output_sha256 recovery_scan_unterminated_final_line \
      recovery_scan_byte_count \
      recovery_scan_line_count recovery_scan_parsed_record_count \
      recovery_scan_identity_match_counts off_recovered_job_ids_csv \
      on_recovered_job_ids_csv recovery_scan_records \
      unterminated_candidate_job_ids_csv \
      recovery_scan_unterminated_candidate_job_ids \
      recovery_scan_securely_unlinked <<< "${scan_result}"
    STRICT_PAIR_RECOVERY_SCAN_OWNED=0
  else
    recovery_scan_normalization_status=1
    recovery_scan_output_sha256="$(
      strict_pair_sha256_file "${STRICT_PAIR_RECOVERY_SCAN_PATH}"
    )" || recovery_scan_output_sha256="-"
    recovery_scan_byte_count="-"
    recovery_scan_line_count="-"
    recovery_scan_parsed_record_count="-"
    recovery_scan_identity_match_counts="null"
    recovery_scan_records="null"
    recovery_scan_unterminated_final_line="-"
    recovery_scan_unterminated_candidate_job_ids="null"
    unterminated_candidate_job_ids_csv="-"
    recovery_scan_securely_unlinked="false"
  fi
  if [[ "${STRICT_PAIR_RECOVERY_SCAN_FD_OPEN}" == "1" ]]; then
    exec 7>&-
    STRICT_PAIR_RECOVERY_SCAN_FD_OPEN=0
  fi

  recovery_scan_authenticated_job_ids_csv=""
  recovery_scan_matched_job_ids_csv=""
  off_recovery_authenticated_job_ids_csv="${off_recovered_job_ids_csv}"
  on_recovery_authenticated_job_ids_csv="${on_recovered_job_ids_csv}"
  off_cleanup_candidate_job_ids_csv="$(
    strict_pair_canonical_job_id_union \
      "${off_record_job_id:-}" "${off_recovered_job_ids_csv}"
  )" || off_cleanup_candidate_job_ids_csv=""
  on_cleanup_candidate_job_ids_csv="$(
    strict_pair_canonical_job_id_union \
      "${on_record_job_id:-}" "${on_recovered_job_ids_csv}"
  )" || on_cleanup_candidate_job_ids_csv=""
  for recovered_job_id in \
      ${off_cleanup_candidate_job_ids_csv//,/ } \
      ${on_cleanup_candidate_job_ids_csv//,/ }; do
    if [[ "${recovered_job_id}" =~ ^[1-9][0-9]*$ && \
          ( ",${off_recovered_job_ids_csv}," == *",${recovered_job_id},"* || \
            ",${on_recovered_job_ids_csv}," == *",${recovered_job_id},"* ) && \
          ",${recovery_scan_matched_job_ids_csv}," != \
            *",${recovered_job_id},"* ]]; then
      recovery_scan_matched_job_ids_csv="${recovery_scan_matched_job_ids_csv:+${recovery_scan_matched_job_ids_csv},}${recovered_job_id}"
    fi
  done
  recovery_scan_authenticated_job_ids_csv="${recovery_scan_matched_job_ids_csv}"

  if [[ "${off_recovered_job_ids_csv}" =~ ^[1-9][0-9]*$ ]]; then
    off_job_id="${off_recovered_job_ids_csv}"
    if [[ "${off_record_job_id}" == "${off_recovered_job_ids_csv}" ]]; then
      off_job_id_source="accepted-id-record"
    else
      off_job_id_source="scheduler-recovery"
    fi
    off_held_output_sha256="$(strict_pair_sha256_text "${off_recovered_job_ids_csv}")"
  elif [[ "${off_record_job_id}" =~ ^[1-9][0-9]*$ ]]; then
    off_job_id="${off_record_job_id}"
    off_job_id_source="accepted-id-record"
    off_held_output_sha256="$(strict_pair_sha256_text "${off_record_job_id}")"
  else
    off_job_id=""
    off_job_id_source="none"
    off_held_output_sha256=""
  fi
  if [[ "${on_recovered_job_ids_csv}" =~ ^[1-9][0-9]*$ ]]; then
    on_job_id="${on_recovered_job_ids_csv}"
    if [[ "${on_record_job_id}" == "${on_recovered_job_ids_csv}" ]]; then
      on_job_id_source="accepted-id-record"
    else
      on_job_id_source="scheduler-recovery"
    fi
    on_held_output_sha256="$(strict_pair_sha256_text "${on_recovered_job_ids_csv}")"
  elif [[ "${on_record_job_id}" =~ ^[1-9][0-9]*$ ]]; then
    on_job_id="${on_record_job_id}"
    on_job_id_source="accepted-id-record"
    on_held_output_sha256="$(strict_pair_sha256_text "${on_record_job_id}")"
  else
    on_job_id=""
    on_job_id_source="none"
    on_held_output_sha256=""
  fi
}

strict_pair_cleanup_recovery_scan() {
  local cleanup_result

  if [[ "${STRICT_PAIR_RECOVERY_SCAN_FD_OPEN}" == "1" ]]; then
    exec 7>&-
    STRICT_PAIR_RECOVERY_SCAN_FD_OPEN=0
  fi
  if [[ "${STRICT_PAIR_RECOVERY_SCAN_OWNED}" != "1" ]]; then
    return
  fi
  cleanup_result="$(
    "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
      "${STRICT_PAIR_RECOVERY_SCAN_PATH}" \
      "${recovery_scan_output_sha256}" "${EUID}" <<'PY'
import hashlib
import os
import pathlib
import re
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_sha256 = sys.argv[2]
expected_uid = int(sys.argv[3])
if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
    raise SystemExit("scheduler recovery scan cleanup lacks an authenticated digest")
metadata = os.lstat(path)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or metadata.st_uid != expected_uid
    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
):
    raise SystemExit("scheduler recovery scan changed before secure cleanup")
digest = hashlib.sha256()
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("scheduler recovery scan inode changed during cleanup")
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
finally:
    os.close(fd)
if digest.hexdigest() != expected_sha256:
    raise SystemExit("scheduler recovery scan digest changed before cleanup")
parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("scheduler recovery scan path changed before unlink")
    os.unlink(path.name, dir_fd=parent_fd)
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
print("removed")
PY
  )" || return
  [[ "${cleanup_result}" == "removed" ]] || return 2
  STRICT_PAIR_RECOVERY_SCAN_OWNED=0
  STRICT_PAIR_RECOVERY_SCAN_PATH=""
  recovery_scan_securely_unlinked="true"
}

strict_pair_scheduler_client() {
  "${STRICT_PAIR_TOOL_ENV}" -i \
    LC_ALL=C \
    "SLURM_CONF=${STRICT_PAIR_SLURM_CONF}" \
    "$@"
}

strict_pair_run_raw_candidate_query() {
  local phase="$1"
  local expected_job_ids_csv="$2"
  local off_candidate_job_ids_csv="$3"
  local on_candidate_job_ids_csv="$4"
  local unattributed_candidate_job_ids_csv="$5"
  local label="$6"
  local query_path
  local query_status
  local query_result
  local cleanup_status

  case "${phase}" in
    pre|post|identity|cancel) ;;
    *) return 2 ;;
  esac
  [[ "${label}" =~ ^[a-z0-9-]+$ ]] || return 2
  query_path="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/candidate-query-${label}.scan"
  [[ ! -e "${query_path}" && ! -L "${query_path}" ]] || return 2
  set -o noclobber
  if ! exec 7> "${query_path}"; then
    set +o noclobber
    return 2
  fi
  set +o noclobber
  if strict_pair_scheduler_client \
      "${STRICT_PAIR_TOOL_SCONTROL}" show job --oneliner \
      "${expected_job_ids_csv}" >&7 2>/dev/null; then
    query_status=0
  else
    query_status=$?
  fi
  query_result="$(
    "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
      "${query_path}" "7" "${query_status}" "${phase}" \
      "${expected_job_ids_csv}" \
      "${off_candidate_job_ids_csv:--}" \
      "${on_candidate_job_ids_csv:--}" \
      "${unattributed_candidate_job_ids_csv:--}" \
      "${PAIR_ID}" "${STRICT_PAIR_ENVIRONMENT}" \
      "${STRICT_PAIR_SUBMISSION_NONCE}" \
      "${STRICT_PAIR_MANIFEST_SHA256}" "${EUID}" \
      "${STRICT_PAIR_TOOL_SCONTROL}" \
      "${STRICT_PAIR_OFF_SNAPSHOT}" \
      "${STRICT_PAIR_ON_SNAPSHOT}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

(
    path_raw,
    writer_fd_raw,
    status_raw,
    phase,
    expected_csv,
    off_candidates_csv,
    on_candidates_csv,
    unattributed_candidates_csv,
    pair_id,
    environment,
    nonce,
    pair_manifest_sha256,
    expected_user_id,
    scontrol_path,
    off_snapshot_dir,
    on_snapshot_dir,
) = sys.argv[1:]
expected_work_dirs = {"off": off_snapshot_dir, "on": on_snapshot_dir}


def parse_ids(raw):
    if raw == "-":
        return []
    values = raw.split(",")
    if (
        not values
        or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values)
        or len(values) != len(set(values))
    ):
        raise SystemExit("raw candidate query received malformed job IDs")
    return values


path = pathlib.Path(path_raw)
writer_fd = int(writer_fd_raw)
status = int(status_raw)
if status < 0 or status > 255:
    raise SystemExit("raw candidate query status is out of range")
expected_ids = parse_ids(expected_csv)
off_candidates = parse_ids(off_candidates_csv)
on_candidates = parse_ids(on_candidates_csv)
unattributed_candidates = parse_ids(unattributed_candidates_csv)
flattened_candidates = off_candidates + on_candidates + unattributed_candidates
if (
    not expected_ids
    or len(flattened_candidates) != len(set(flattened_candidates))
    or set(expected_ids) != set(flattened_candidates)
):
    raise SystemExit("raw candidate query received inconsistent candidate sets")

metadata = os.lstat(path)
writer_metadata = os.fstat(writer_fd)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or stat.S_IMODE(metadata.st_mode) != 0o600
    or metadata.st_uid != int(expected_user_id)
    or (metadata.st_dev, metadata.st_ino)
    != (writer_metadata.st_dev, writer_metadata.st_ino)
):
    raise SystemExit("raw candidate query is not one private regular file")
os.fsync(writer_fd)
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("raw candidate query inode changed while opening")
    digest = hashlib.sha256()
    byte_count = 0
    line_count = 0
    normalized_line_count = 0
    unterminated_final_line = False
    records = {"off": [], "on": []}
    required_keys = {
        "Comment",
        "JobId",
        "JobName",
        "JobState",
        "Reason",
        "UserId",
        "WorkDir",
    }
    with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
        for raw_line in stream:
            digest.update(raw_line)
            byte_count += len(raw_line)
            line_count += 1
            if not raw_line.endswith(b"\n"):
                unterminated_final_line = True
                continue
            try:
                line = raw_line.decode("ascii")
            except UnicodeDecodeError:
                continue
            fields = {}
            duplicate_keys = set()
            for token in line.split():
                key, separator, value = token.partition("=")
                if not separator:
                    continue
                if key in fields:
                    duplicate_keys.add(key)
                fields[key] = value
            job_id = fields.get("JobId")
            if (
                job_id not in expected_ids
                or required_keys & duplicate_keys
                or not required_keys <= set(fields)
            ):
                continue
            user_match = re.fullmatch(r"[^()\s]+\(([0-9]+)\)", fields["UserId"])
            if user_match is None or user_match.group(1) != expected_user_id:
                continue
            matching_arms = []
            for arm, arm_candidates in (
                ("off", off_candidates),
                ("on", on_candidates),
            ):
                if (
                    job_id in arm_candidates or job_id in unattributed_candidates
                ) and fields["JobName"] == f"{arm}-{environment}-{pair_id}" and fields[
                    "Comment"
                ] == (
                    f"nemo-rl-strict-pair-v1:{arm}:{nonce}:"
                    f"{pair_manifest_sha256}"
                ) and fields["WorkDir"] == expected_work_dirs[arm]:
                    matching_arms.append(arm)
            if len(matching_arms) != 1:
                continue
            state = fields["JobState"]
            reason = fields["Reason"]
            held = state == "PENDING" and reason == "JobHeldUser"
            if phase == "pre" and not held:
                continue
            if phase == "post" and (
                reason == "JobHeldUser"
                or state not in {"PENDING", "CONFIGURING", "RUNNING"}
            ):
                continue
            if phase == "cancel" and not (
                state == "CANCELLED" and reason != "JobHeldUser"
            ):
                continue
            if phase not in {"pre", "post", "identity", "cancel"}:
                raise SystemExit("raw candidate query received an invalid phase")
            records[matching_arms[0]].append(
                {
                    "comment": fields["Comment"],
                    "held": held,
                    "job_id": job_id,
                    "job_name": fields["JobName"],
                    "job_state": state,
                    "reason": reason,
                    "user_id": user_match.group(1),
                    "work_dir": fields["WorkDir"],
                }
            )
            normalized_line_count += 1
    os.fchmod(writer_fd, 0o400)
    os.fsync(writer_fd)
finally:
    os.close(fd)

directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("raw candidate query path changed before unlink")
    os.unlink(path.name, dir_fd=directory_fd)
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)

authenticated_ids = []
unresolved_ids = []
authenticated_by_arm = {"off": [], "on": []}
for job_id in expected_ids:
    occurrences = [
        (arm, record)
        for arm in ("off", "on")
        for record in records[arm]
        if record["job_id"] == job_id
    ]
    normalization_status = 0 if normalized_line_count == line_count else 1
    if (
        status != 0
        or normalization_status != 0
        or unterminated_final_line
        or len(occurrences) != 1
    ):
        unresolved_ids.append(job_id)
        continue
    arm, _record = occurrences[0]
    authenticated_ids.append(job_id)
    authenticated_by_arm[arm].append(job_id)

document = {
    "argv": [scontrol_path, "show", "job", "--oneliner", expected_csv],
    "authenticated_job_ids": authenticated_ids,
    "byte_count": byte_count,
    "candidate_job_ids": {
        "off": off_candidates,
        "on": on_candidates,
        "unattributed": unattributed_candidates,
    },
    "complete": (
        status == 0
        and normalization_status == 0
        and not unterminated_final_line
    ),
    "line_count": line_count,
    "normalization_status": normalization_status,
    "output_sha256_raw": digest.hexdigest(),
    "phase": phase,
    "records": records,
    "securely_unlinked": True,
    "status": status,
    "unterminated_final_line": unterminated_final_line,
    "unresolved_job_ids": unresolved_ids,
}
print(json.dumps(document, sort_keys=True, separators=(",", ":")))
print(
    "\t".join(
        (
            ",".join(authenticated_ids) or "-",
            ",".join(unresolved_ids) or "-",
            ",".join(authenticated_by_arm["off"]) or "-",
            ",".join(authenticated_by_arm["on"]) or "-",
        )
    )
)
PY
  )" || {
    "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
      "${query_path}" "7" "${EUID}" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
writer_fd = int(sys.argv[2])
expected_uid = int(sys.argv[3])
try:
    metadata = os.lstat(path)
except FileNotFoundError:
    raise SystemExit(0)
writer_metadata = os.fstat(writer_fd)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_nlink != 1
    or stat.S_IMODE(metadata.st_mode) not in {0o400, 0o600}
    or metadata.st_uid != expected_uid
    or (metadata.st_dev, metadata.st_ino)
    != (writer_metadata.st_dev, writer_metadata.st_ino)
):
    raise SystemExit("failed raw candidate query cannot be safely removed")
os.fsync(writer_fd)
directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    current = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("failed raw candidate query path changed before unlink")
    os.unlink(path.name, dir_fd=directory_fd)
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
    cleanup_status=$?
    exec 7>&-
    (( cleanup_status == 0 )) || return 2
    return 2
  }
  exec 7>&-
  printf '%s\n' "${query_result}"
}

strict_pair_json_array_append() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B -c '
import json
import sys

items = json.loads(sys.argv[1])
item = json.loads(sys.argv[2])
if not isinstance(items, list) or not isinstance(item, dict):
    raise SystemExit("candidate query evidence has invalid JSON shape")
items.append(item)
print(json.dumps(items, sort_keys=True, separators=(",", ":")))
' "$1" "$2"
}

STRICT_PAIR_LAST_QUERY_AUTHENTICATED="-"
STRICT_PAIR_LAST_QUERY_UNRESOLVED="-"
STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED="-"
STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED="-"
STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=0
strict_pair_normalize_candidate_arm_mapping() {
  local expected_job_ids_csv="$1"
  local off_candidate_job_ids_csv="$2"
  local on_candidate_job_ids_csv="$3"
  local unattributed_candidate_job_ids_csv="$4"
  local bucket_csv job_id seen=""
  local normalized_off="" normalized_on="" normalized_unattributed=""
  local in_off in_on in_unattributed membership_count

  [[ "${expected_job_ids_csv}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || return 2
  for bucket_csv in \
      "${off_candidate_job_ids_csv}" \
      "${on_candidate_job_ids_csv}" \
      "${unattributed_candidate_job_ids_csv}"; do
    [[ "${bucket_csv}" == "-" || \
       "${bucket_csv}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || return 2
    seen=""
    for job_id in ${bucket_csv//,/ }; do
      [[ "${job_id}" == "-" ]] && continue
      [[ ",${expected_job_ids_csv}," == *",${job_id},"* && \
         ",${seen}," != *",${job_id},"* ]] || return 2
      seen="${seen:+${seen},}${job_id}"
    done
  done
  seen=""
  for job_id in ${expected_job_ids_csv//,/ }; do
    [[ ",${seen}," != *",${job_id},"* ]] || return 2
    seen="${seen:+${seen},}${job_id}"
    in_off=0
    in_on=0
    in_unattributed=0
    [[ ",${off_candidate_job_ids_csv}," == *",${job_id},"* ]] && in_off=1
    [[ ",${on_candidate_job_ids_csv}," == *",${job_id},"* ]] && in_on=1
    [[ ",${unattributed_candidate_job_ids_csv}," == *",${job_id},"* ]] && \
      in_unattributed=1
    membership_count=$((in_off + in_on + in_unattributed))
    (( membership_count > 0 )) || return 2
    if (( membership_count == 1 && in_off == 1 )); then
      normalized_off="${normalized_off:+${normalized_off},}${job_id}"
    elif (( membership_count == 1 && in_on == 1 )); then
      normalized_on="${normalized_on:+${normalized_on},}${job_id}"
    else
      # A candidate arm is not authenticated identity.  Ambiguous membership
      # is represented once and exact scheduler labels choose the sole arm.
      normalized_unattributed="${normalized_unattributed:+${normalized_unattributed},}${job_id}"
    fi
  done
  printf '%s\t%s\t%s\n' \
    "${normalized_off:--}" "${normalized_on:--}" \
    "${normalized_unattributed:--}"
}

strict_pair_capture_raw_candidate_query() {
  local phase="$1"
  local expected_job_ids_csv="$2"
  local off_candidate_job_ids_csv="$3"
  local on_candidate_job_ids_csv="$4"
  local unattributed_candidate_job_ids_csv="$5"
  local label="$6"
  local collection_name="$7"
  local result
  local attempt
  local summary
  local current_collection
  local updated_collection
  local normalized_candidate_mapping

  normalized_candidate_mapping="$(
    strict_pair_normalize_candidate_arm_mapping \
      "${expected_job_ids_csv}" \
      "${off_candidate_job_ids_csv:--}" \
      "${on_candidate_job_ids_csv:--}" \
      "${unattributed_candidate_job_ids_csv:--}"
  )" || {
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  }
  IFS=$'\t' read -r \
    off_candidate_job_ids_csv \
    on_candidate_job_ids_csv \
    unattributed_candidate_job_ids_csv <<< "${normalized_candidate_mapping}"

  result="$(
    strict_pair_run_raw_candidate_query \
      "${phase}" "${expected_job_ids_csv}" \
      "${off_candidate_job_ids_csv:--}" \
      "${on_candidate_job_ids_csv:--}" \
      "${unattributed_candidate_job_ids_csv:--}" "${label}"
  )" || {
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  }
  if [[ "${result}" != *$'\n'* ]]; then
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  fi
  attempt="${result%%$'\n'*}"
  summary="${result#*$'\n'}"
  IFS=$'\t' read -r \
    STRICT_PAIR_LAST_QUERY_AUTHENTICATED \
    STRICT_PAIR_LAST_QUERY_UNRESOLVED \
    STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED \
    STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED <<< "${summary}"
  current_collection="${!collection_name}"
  updated_collection="$(
    strict_pair_json_array_append "${current_collection}" "${attempt}"
  )" || {
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  }
  printf -v "${collection_name}" '%s' "${updated_collection}"
}

strict_pair_capture_raw_lifecycle_query() {
  local phase="$1"
  local expected_job_ids_csv="$2"
  local off_candidate_job_ids_csv="$3"
  local on_candidate_job_ids_csv="$4"
  local label="$5"
  local target_name="$6"
  local result
  local attempt
  local summary

  [[ "${phase}" == "pre" || "${phase}" == "post" ]] || return 2
  result="$(
    strict_pair_run_raw_candidate_query \
      "${phase}" "${expected_job_ids_csv}" \
      "${off_candidate_job_ids_csv}" \
      "${on_candidate_job_ids_csv}" - "${label}"
  )" || {
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  }
  # Command substitution removes only the final LF.  The remaining LF is the
  # exact boundary between the canonical query document and its tab summary.
  if [[ "${result}" != *$'\n'* ]]; then
    STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED=1
    return 2
  fi
  attempt="${result%%$'\n'*}"
  summary="${result#*$'\n'}"
  IFS=$'\t' read -r \
    STRICT_PAIR_LAST_QUERY_AUTHENTICATED \
    STRICT_PAIR_LAST_QUERY_UNRESOLVED \
    STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED \
    STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED <<< "${summary}"
  printf -v "${target_name}" '%s' "${attempt}"
}

strict_pair_publish_submission_receipt() {
  local outcome="$1"
  local stage="$2"
  local rollback_confirmed="$3"
  local operation="${4:-publish}"

  # This function runs in a command-substitution child.  Once it starts the
  # durable publication transaction, parent INT/TERM are deferred until the
  # child has either returned the committed digest or failed before commit.
  trap '' INT TERM
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" \
    "${operation}" "${outcome}" "${stage}" "${rollback_confirmed}" \
    "${PAIR_ID}" \
    "${STRICT_PAIR_SUBMISSION_NONCE}" \
    "${STRICT_PAIR_MANIFEST_PATH}" "${STRICT_PAIR_MANIFEST_SHA256}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" \
    "${off_status}" "${on_status}" \
    "${off_job_id:--}" "${on_job_id:--}" \
    "${off_job_id_source}" "${on_job_id_source}" \
    "${off_held_output_sha256:--}" "${on_held_output_sha256:--}" \
    "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" \
    "${off_accepted_id_record_sha256:--}" \
    "${off_record_job_id:--}" \
    "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" \
    "${on_accepted_id_record_sha256:--}" \
    "${on_record_job_id:--}" \
    "${off_sbatch_status}" "${on_sbatch_status}" \
    "${off_relay_status}" "${on_relay_status}" \
    "${off_writer_drained}" "${on_writer_drained}" \
    "${off_started_unix_ns}" "${on_started_unix_ns}" \
    "${off_drained_unix_ns}" "${on_drained_unix_ns}" \
    "${pre_release_query}" \
    "${release_status}" "${release_output_sha256}" \
    "${post_release_query}" \
    "${cancel_status}" "${cancel_output_sha256}" \
    "${cancel_job_ids_csv}" \
    "${post_cancel_queries}" \
    "${pre_cancel_queries}" \
    "${off_cleanup_candidate_job_ids_csv:--}" \
    "${on_cleanup_candidate_job_ids_csv:--}" \
    "${recovery_scan_status}" \
    "${recovery_scan_normalization_status}" \
    "${recovery_scan_output_sha256}" \
    "${recovery_scan_byte_count}" \
    "${recovery_scan_line_count}" \
    "${recovery_scan_parsed_record_count}" \
    "${recovery_scan_identity_match_counts}" \
    "${recovery_scan_records}" \
    "${recovery_scan_unterminated_final_line}" \
    "${recovery_scan_unterminated_candidate_job_ids}" \
    "${recovery_scan_securely_unlinked}" \
    "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
    "${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}" \
    "${STRICT_PAIR_TOOL_SBATCH}" "${STRICT_PAIR_TOOL_SBATCH_SHA256}" \
    "${STRICT_PAIR_TOOL_SCANCEL}" "${STRICT_PAIR_TOOL_SCANCEL_SHA256}" \
    "${STRICT_PAIR_TOOL_SCONTROL}" "${STRICT_PAIR_TOOL_SCONTROL_SHA256}" \
    "${STRICT_PAIR_TOOL_ENV}" "${STRICT_PAIR_TOOL_ENV_SHA256}" \
    "${STRICT_PAIR_SLURM_CONF}" "${STRICT_PAIR_SLURM_CONF_SHA256}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import tempfile

(
    receipt_raw,
    operation,
    outcome,
    stage,
    rollback_confirmed_raw,
    pair_id,
    submission_nonce,
    pair_manifest_path,
    pair_manifest_sha256,
    submission_contract_path,
    submission_contract_sha256,
    off_wrapper_status_raw,
    on_wrapper_status_raw,
    off_job_id_raw,
    on_job_id_raw,
    off_job_id_source,
    on_job_id_source,
    off_held_sha256_raw,
    on_held_sha256_raw,
    off_accepted_id_record_path,
    off_accepted_id_record_sha256,
    off_record_job_id_raw,
    on_accepted_id_record_path,
    on_accepted_id_record_sha256,
    on_record_job_id_raw,
    off_sbatch_status_raw,
    on_sbatch_status_raw,
    off_relay_status_raw,
    on_relay_status_raw,
    off_writer_drained_raw,
    on_writer_drained_raw,
    off_started_unix_ns_raw,
    on_started_unix_ns_raw,
    off_drained_unix_ns_raw,
    on_drained_unix_ns_raw,
    pre_release_query_raw,
    release_status_raw,
    release_sha256_raw,
    post_release_query_raw,
    cancel_status_raw,
    cancel_sha256_raw,
    cancel_job_ids_csv,
    post_cancel_queries_raw,
    pre_cancel_queries_raw,
    off_cleanup_candidate_job_ids_csv,
    on_cleanup_candidate_job_ids_csv,
    recovery_scan_status_raw,
    recovery_scan_normalization_status_raw,
    recovery_scan_sha256_raw,
    recovery_scan_byte_count_raw,
    recovery_scan_line_count_raw,
    recovery_scan_parsed_record_count_raw,
    recovery_scan_identity_match_counts_raw,
    recovery_scan_records_raw,
    recovery_scan_unterminated_final_line_raw,
    recovery_scan_unterminated_candidate_job_ids_raw,
    recovery_scan_securely_unlinked_raw,
    runtime_tool_manifest_path,
    runtime_tool_manifest_sha256,
    sbatch_path,
    sbatch_sha256,
    scancel_path,
    scancel_sha256,
    scontrol_path,
    scontrol_sha256,
    env_path,
    env_sha256,
    slurm_conf_path,
    slurm_conf_sha256,
) = sys.argv[1:]

if operation not in {"adopt", "publish"}:
    raise SystemExit("unsupported submission-receipt operation")

DIGEST = re.compile(r"[0-9a-f]{64}")
failure_stages = {
    "arm_submit",
    "job_id_validation",
    "pre_release_validation",
    "release",
    "post_release_validation",
    "receipt_publication",
    "unexpected_exit",
}


def digest_or_none(value: str):
    if value == "-":
        return None
    if DIGEST.fullmatch(value) is None:
        raise SystemExit("receipt evidence has malformed SHA-256")
    return value


def job_id_or_none(value: str):
    if value == "-":
        return None
    if re.fullmatch(r"[0-9]+", value) is None or int(value) == 0:
        raise SystemExit("receipt evidence has malformed job ID")
    return value


def status_or_none(value: str):
    if value == "-":
        return None
    if re.fullmatch(r"[0-9]+", value) is None or not 0 <= int(value) <= 255:
        raise SystemExit("receipt evidence has malformed command status")
    return int(value)


def count_or_none(value: str):
    if value == "-":
        return None
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise SystemExit("receipt evidence has malformed nonnegative count")
    return int(value)


def positive_count_or_none(value: str):
    result = count_or_none(value)
    if result == 0:
        raise SystemExit("receipt evidence expected a positive count")
    return result


def job_ids_csv_or_empty(value: str):
    if value == "-":
        return []
    result = value.split(",")
    if any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in result):
        raise SystemExit("receipt evidence has malformed job-ID list")
    if len(result) != len(set(result)):
        raise SystemExit("receipt evidence has duplicate job IDs in one arm")
    return result


def records_or_none(value: str):
    if value == "null":
        return None
    document = json.loads(value)
    if not isinstance(document, dict):
        raise SystemExit("normalized scheduler records must be an object or null")
    return document


def list_or_none(value: str):
    if value == "null":
        return None
    document = json.loads(value)
    if not isinstance(document, list):
        raise SystemExit("normalized scheduler list evidence must be an array or null")
    return document


def raw_query_or_none(value: str):
    if value == "null":
        return None
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit("raw scheduler-query evidence is not JSON") from error
    if not isinstance(document, dict):
        raise SystemExit("raw scheduler-query evidence must be an object or null")
    return document


def validate_raw_candidate_queries(raw, phase, expected_candidate_ids):
    try:
        attempts = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit("raw candidate-query evidence is not JSON") from error
    if not isinstance(attempts, list):
        raise SystemExit("raw candidate-query evidence must be a list")
    expected_candidate_set = set(expected_candidate_ids)
    queried_ids = set()
    last_resolution = {}
    authenticated_by_arm = {"off": [], "on": []}
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != {
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
        }:
            raise SystemExit("raw candidate-query attempt has unexpected keys")
        argv = attempt["argv"]
        if attempt["phase"] != phase:
            raise SystemExit("raw candidate-query phase is inconsistent")
        if (
            not isinstance(argv, list)
            or len(argv) != 5
            or argv[:4] != [scontrol_path, "show", "job", "--oneliner"]
            or not isinstance(argv[4], str)
        ):
            raise SystemExit("raw candidate-query argv is invalid")
        attempt_ids = argv[4].split(",")
        if (
            not attempt_ids
            or any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in attempt_ids)
            or len(attempt_ids) != len(set(attempt_ids))
            or not set(attempt_ids) <= expected_candidate_set
        ):
            raise SystemExit("raw candidate-query selector is invalid")
        candidate_job_ids = attempt["candidate_job_ids"]
        if (
            not isinstance(candidate_job_ids, dict)
            or set(candidate_job_ids) != {"off", "on", "unattributed"}
            or any(
                not isinstance(candidate_job_ids[key], list)
                for key in ("off", "on", "unattributed")
            )
        ):
            raise SystemExit("raw candidate-query arm candidates are invalid")
        flattened_candidates = [
            value
            for key in ("off", "on", "unattributed")
            for value in candidate_job_ids[key]
        ]
        if (
            any(
                not isinstance(value, str)
                or re.fullmatch(r"[1-9][0-9]*", value) is None
                for value in flattened_candidates
            )
            or len(flattened_candidates) != len(set(flattened_candidates))
            or set(flattened_candidates) != set(attempt_ids)
        ):
            raise SystemExit("raw candidate-query arm mapping is inconsistent")
        if (
            type(attempt["status"]) is not int
            or attempt["status"] < 0
            or attempt["status"] > 255
            or type(attempt["normalization_status"]) is not int
            or attempt["normalization_status"] < 0
            or attempt["normalization_status"] > 255
            or type(attempt["byte_count"]) is not int
            or attempt["byte_count"] < 0
            or type(attempt["line_count"]) is not int
            or attempt["line_count"] < 0
            or type(attempt["complete"]) is not bool
            or type(attempt["unterminated_final_line"]) is not bool
            or attempt["securely_unlinked"] is not True
            or not isinstance(attempt["output_sha256_raw"], str)
            or DIGEST.fullmatch(attempt["output_sha256_raw"]) is None
            or attempt["complete"]
            != (
                attempt["status"] == 0
                and attempt["normalization_status"] == 0
                and not attempt["unterminated_final_line"]
            )
        ):
            raise SystemExit("raw candidate-query scalar evidence is invalid")
        records = attempt["records"]
        if (
            not isinstance(records, dict)
            or set(records) != {"off", "on"}
            or any(not isinstance(records[arm], list) for arm in ("off", "on"))
        ):
            raise SystemExit("raw candidate-query records are invalid")
        if attempt["complete"] and attempt["line_count"] != sum(
            len(records[arm]) for arm in ("off", "on")
        ):
            raise SystemExit("complete raw query line count differs from its records")
        occurrences = {job_id: [] for job_id in attempt_ids}
        for arm in ("off", "on"):
            expected_comment = (
                f"nemo-rl-strict-pair-v1:{arm}:{submission_nonce}:"
                f"{pair_manifest_sha256}"
            )
            for record in records[arm]:
                if not isinstance(record, dict) or set(record) != {
                    "comment",
                    "held",
                    "job_id",
                    "job_name",
                    "job_state",
                    "reason",
                    "user_id",
                    "work_dir",
                }:
                    raise SystemExit("raw candidate-query record has unexpected keys")
                job_id = record["job_id"]
                if (
                    job_id not in occurrences
                    or record["comment"] != expected_comment
                    or record["job_name"] != f"{arm}-{environment}-{pair_id}"
                    or record["user_id"] != str(os.geteuid())
                    or record["work_dir"]
                    != execution_environment["arms"][arm]["scheduler"][
                        "batch_working_directory"
                    ]
                    or job_id not in candidate_job_ids[arm]
                    and job_id not in candidate_job_ids["unattributed"]
                    or type(record["held"]) is not bool
                    or any(
                        not isinstance(record[key], str)
                        for key in ("job_id", "job_state", "reason", "work_dir")
                    )
                ):
                    raise SystemExit("raw candidate-query record identity is invalid")
                derived_held = (
                    record["job_state"] == "PENDING"
                    and record["reason"] == "JobHeldUser"
                )
                if record["held"] != derived_held:
                    raise SystemExit("raw candidate-query held derivation is invalid")
                if phase == "cancel" and (
                    record["job_state"] != "CANCELLED"
                    or record["reason"] == "JobHeldUser"
                ):
                    raise SystemExit("raw post-cancel record is not canceled")
                if phase == "pre" and not derived_held:
                    raise SystemExit("raw pre-release record is not held")
                if phase == "post" and (
                    record["reason"] == "JobHeldUser"
                    or record["job_state"]
                    not in {"PENDING", "CONFIGURING", "RUNNING"}
                ):
                    raise SystemExit("raw post-release record is invalid")
                occurrences[job_id].append((arm, record))
        derived_authenticated = []
        derived_unresolved = []
        for job_id in attempt_ids:
            if attempt["complete"] and len(occurrences[job_id]) == 1:
                derived_authenticated.append(job_id)
            else:
                derived_unresolved.append(job_id)
        if (
            attempt["authenticated_job_ids"] != derived_authenticated
            or attempt["unresolved_job_ids"] != derived_unresolved
        ):
            raise SystemExit("raw candidate-query resolution is inconsistent")
        queried_ids.update(attempt_ids)
        for job_id in attempt_ids:
            last_resolution[job_id] = (
                occurrences[job_id][0] if job_id in derived_authenticated else None
            )
    if queried_ids != expected_candidate_set:
        raise SystemExit("raw candidate queries do not cover every candidate")
    authenticated_ids = []
    for job_id in expected_candidate_ids:
        resolution = last_resolution.get(job_id)
        if resolution is None:
            continue
        arm, _record = resolution
        authenticated_ids.append(job_id)
        if job_id not in authenticated_by_arm[arm]:
            authenticated_by_arm[arm].append(job_id)
    return attempts, authenticated_ids, authenticated_by_arm


if outcome not in {"failed-closed", "released", "rollback-unconfirmed"}:
    raise SystemExit("invalid strict pair submission outcome")
if (outcome == "released" and stage != "complete") or (
    outcome != "released" and stage not in failure_stages
):
    raise SystemExit("invalid outcome/stage combination")
if rollback_confirmed_raw not in {"null", "true", "false"}:
    raise SystemExit("rollback_confirmed must be null or an exact lowercase boolean")
rollback_confirmed = {
    "false": False,
    "null": None,
    "true": True,
}[rollback_confirmed_raw]
expected_rollback_confirmed = {
    "failed-closed": True,
    "released": None,
    "rollback-unconfirmed": False,
}[outcome]
if rollback_confirmed is not expected_rollback_confirmed:
    raise SystemExit("receipt outcome and rollback confirmation disagree")
if DIGEST.fullmatch(submission_nonce) is None:
    raise SystemExit("submission nonce must be 64 lowercase hex")
for value in (
    pair_manifest_sha256,
    submission_contract_sha256,
    runtime_tool_manifest_sha256,
    sbatch_sha256,
    scancel_sha256,
    scontrol_sha256,
    env_sha256,
    slurm_conf_sha256,
    off_accepted_id_record_sha256,
    on_accepted_id_record_sha256,
):
    if DIGEST.fullmatch(value) is None:
        raise SystemExit("receipt sealed input has malformed SHA-256")


def verify_file(path_raw: str, expected_sha256: str, label: str) -> None:
    path = pathlib.Path(path_raw)
    metadata = os.lstat(path)
    if (
        not path.is_absolute()
        or os.path.realpath(path) != str(path)
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise SystemExit(f"{label} must remain one canonical regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit(f"{label} SHA-256 changed before receipt publication")


for path_raw, expected_sha256, label in (
    (pair_manifest_path, pair_manifest_sha256, "Pair manifest"),
    (
        submission_contract_path,
        submission_contract_sha256,
        "submission contract",
    ),
    (
        runtime_tool_manifest_path,
        runtime_tool_manifest_sha256,
        "runtime-tool manifest",
    ),
    (sbatch_path, sbatch_sha256, "sbatch"),
    (scancel_path, scancel_sha256, "scancel"),
    (scontrol_path, scontrol_sha256, "scontrol"),
    (env_path, env_sha256, "env"),
    (slurm_conf_path, slurm_conf_sha256, "SLURM_CONF"),
    (
        off_accepted_id_record_path,
        off_accepted_id_record_sha256,
        "OFF accepted-ID record",
    ),
    (
        on_accepted_id_record_path,
        on_accepted_id_record_sha256,
        "ON accepted-ID record",
    ),
):
    verify_file(path_raw, expected_sha256, label)


def reject_duplicate_members(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise SystemExit(f"duplicate JSON member in sealed input: {key}")
        document[key] = value
    return document


try:
    pair_manifest_document = json.loads(
        pathlib.Path(pair_manifest_path).read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_members,
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit("Pair manifest is not strict ASCII JSON") from error
if pair_manifest_document.get("schema") != "nemo-rl-strict-single-env-pair-v2":
    raise SystemExit("Pair manifest schema differs from receipt producer")
selection = pair_manifest_document.get("selection")
if not isinstance(selection, dict) or set(selection) != {
    "config",
    "environment",
    "fixture",
    "gym_resources",
}:
    raise SystemExit("Pair manifest selection has unexpected keys")
environment = selection["environment"]
if environment not in {"reasoning_gym", "citation", "freeform"}:
    raise SystemExit("Pair manifest selection environment is invalid")
expected_selection_paths = {
    "reasoning_gym": {
        "config": (
            "examples/nemo_gym/nemotron-3.5-nano/"
            "single_env_reasoning_gym_sc.yaml"
        ),
        "gym_resources": {
            "config": "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
            "requirements": "resources_servers/reasoning_gym/requirements.txt",
            "verifier_source": "resources_servers/reasoning_gym/app.py",
        },
    },
    "citation": {
        "config": (
            "examples/nemo_gym/nemotron-3.5-nano/single_env_citation_sc.yaml"
        ),
        "gym_resources": {
            "config": (
                "resources_servers/format_verification/configs/"
                "citation_format.yaml"
            ),
            "requirements": (
                "resources_servers/format_verification/requirements.txt"
            ),
            "verifier_source": "resources_servers/format_verification/app.py",
        },
    },
    "freeform": {
        "config": (
            "examples/nemo_gym/nemotron-3.5-nano/single_env_freeform_sc.yaml"
        ),
        "gym_resources": {
            "config": (
                "resources_servers/format_verification/configs/"
                "freeform_formatting.yaml"
            ),
            "requirements": (
                "resources_servers/format_verification/requirements.txt"
            ),
            "verifier_source": "resources_servers/format_verification/app.py",
        },
    },
}[environment]
config = selection["config"]
fixture = selection["fixture"]
gym_resources = selection["gym_resources"]
if (
    not isinstance(config, dict)
    or set(config) != {"path", "sha256"}
    or config["path"] != expected_selection_paths["config"]
    or not isinstance(config["sha256"], str)
    or DIGEST.fullmatch(config["sha256"]) is None
):
    raise SystemExit("Pair manifest selection config is invalid")
if (
    not isinstance(fixture, dict)
    or set(fixture) != {"path", "rows", "sha256"}
    or not isinstance(fixture["path"], str)
    or type(fixture["rows"]) is not int
    or fixture["rows"] != 5
    or not isinstance(fixture["sha256"], str)
    or DIGEST.fullmatch(fixture["sha256"]) is None
):
    raise SystemExit("Pair manifest selection fixture is invalid")
if (
    not isinstance(gym_resources, dict)
    or set(gym_resources) != {"config", "requirements", "verifier_source"}
):
    raise SystemExit("Pair manifest selected Gym resources are invalid")
for resource_name, expected_path in expected_selection_paths[
    "gym_resources"
].items():
    resource = gym_resources.get(resource_name)
    if (
        not isinstance(resource, dict)
        or set(resource) != {"path", "sha256"}
        or resource["path"] != expected_path
        or not isinstance(resource["sha256"], str)
        or DIGEST.fullmatch(resource["sha256"]) is None
    ):
        raise SystemExit(f"Pair manifest selected Gym {resource_name} is invalid")
artifacts = pair_manifest_document.get("artifacts")
source = pair_manifest_document.get("source")
acceptance = pair_manifest_document.get("acceptance")
model_transport = pair_manifest_document.get("model_transport")
if (
    not isinstance(artifacts, dict)
    or not isinstance(source, dict)
    or not isinstance(acceptance, dict)
    or not isinstance(model_transport, dict)
):
    raise SystemExit("Pair manifest selection provenance parents are malformed")
gym_source = source.get("gym")
if not isinstance(gym_source, dict):
    raise SystemExit("Pair manifest selected Gym provenance parent is malformed")
if artifacts.get("fixture") != fixture:
    raise SystemExit("Pair manifest selection fixture provenance is inconsistent")
if source.get("config_sha256") != config["sha256"]:
    raise SystemExit("Pair manifest selection config provenance is inconsistent")
source_root = pathlib.Path(source.get("root", ""))
gym_root = pathlib.Path(gym_source.get("path", ""))
verify_file(fixture["path"], fixture["sha256"], "selected fixture")
verify_file(
    str(source_root / config["path"]),
    config["sha256"],
    "selected environment config",
)
for resource_name, resource in gym_resources.items():
    verify_file(
        str(gym_root / resource["path"]),
        resource["sha256"],
        f"selected Gym {resource_name}",
    )

deployment = pair_manifest_document.get("deployment")
if not isinstance(deployment, dict) or not isinstance(deployment.get("root"), str):
    raise SystemExit("Pair deployment provenance is malformed")
receipt_source = {}
for component, directory_name in (
    ("bridge", "Megatron-Bridge"),
    ("mcore", "Megatron-LM"),
):
    record = source.get(component)
    expected_root = f'{deployment["root"]}/runnable/{directory_name}'
    if (
        not isinstance(record, dict)
        or set(record) != {"head", "root", "tree"}
        or record.get("root") != expected_root
        or not isinstance(record.get("head"), str)
        or re.fullmatch(r"[0-9a-f]{40}", record["head"]) is None
        or not isinstance(record.get("tree"), str)
        or re.fullmatch(r"[0-9a-f]{40}", record["tree"]) is None
    ):
        raise SystemExit(f"Pair {component} source provenance is malformed")
    receipt_source[component] = record


def validate_execution_environment(value):
    if not isinstance(value, dict) or set(value) != {
        "arm_launcher",
        "arms",
        "fixed",
        "schema",
    }:
        raise SystemExit("Pair execution environment has unexpected keys")
    arm_launcher = value["arm_launcher"]
    fixed = value["fixed"]
    arms = value["arms"]
    paths = pair_manifest_document.get("paths")
    snapshots = source.get("snapshots")
    if (
        value["schema"] != "nemo-rl-strict-execution-environment-v1"
        or not isinstance(arm_launcher, dict)
        or set(arm_launcher)
        != {"ambient_merge", "argv_prefix", "forbidden_caller_names"}
        or arm_launcher
        != {
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
        }
        or not isinstance(fixed, dict)
        or set(fixed)
        != {
            "cpus_per_worker",
            "nemo_skills_sandbox_port",
            "ray_log_sync_frequency",
            "sandbox_command",
            "train_path",
            "val_path",
        }
        or not isinstance(arms, dict)
        or set(arms) != {"off", "on"}
        or not isinstance(paths, dict)
        or not isinstance(snapshots, dict)
    ):
        raise SystemExit("Pair execution environment is malformed")
    expected_fixed = {
        "cpus_per_worker": "144",
        "nemo_skills_sandbox_port": "6000",
        "ray_log_sync_frequency": "60",
        "sandbox_command": "/start-with-nginx.sh",
        "train_path": fixture["path"],
        "val_path": fixture["path"],
    }
    if fixed != expected_fixed:
        raise SystemExit("Pair fixed execution environment differs from policy")
    for arm in ("off", "on"):
        record = arms[arm]
        snapshot = snapshots.get(arm)
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "base_log_dir",
                "cache_read",
                "hf_datasets_cache",
                "hf_home",
                "hf_hub_cache",
                "persistent_cache",
                "results_dir",
                "scheduler",
                "setup_command",
            }
            or not isinstance(snapshot, dict)
        ):
            raise SystemExit(f"Pair {arm} execution environment is malformed")
        snapshot_path = snapshot.get("path")
        arm_results = f'{paths.get("results_root")}/{arm}'
        arm_persistent_cache = f'{paths.get("cache_root")}/{arm}'
        arm_hf_home = f'{paths.get("hf_home")}/{arm}'
        expected_record = {
            "base_log_dir": f"{arm_results}/ray_logs",
            "cache_read": {
                "entry_count": 0,
                "mode": "0700",
                "path": f"{arm_persistent_cache}/cache_read",
                "policy": "empty-at-publication-and-job-entry-no-read",
            },
            "hf_datasets_cache": f"{arm_hf_home}/hub",
            "hf_home": arm_hf_home,
            "hf_hub_cache": f"{arm_hf_home}/hub",
            "persistent_cache": arm_persistent_cache,
            "results_dir": arm_results,
            "scheduler": {
                "batch_working_directory": snapshot_path,
                "sbatch_chdir_argument": f"--chdir={snapshot_path}",
                "sbatch_client_cwd": snapshot_path,
                "slurm_submit_dir": snapshot_path,
            },
            "setup_command": record.get("setup_command"),
        }
        setup_command = record.get("setup_command")
        if (
            record != expected_record
            or not isinstance(setup_command, dict)
            or set(setup_command) != {"byte_count", "sha256"}
            or type(setup_command.get("byte_count")) is not int
            or setup_command["byte_count"] <= 0
            or not isinstance(setup_command.get("sha256"), str)
            or DIGEST.fullmatch(setup_command["sha256"]) is None
        ):
            raise SystemExit(f"Pair {arm} execution environment differs from policy")
    if (
        arms["off"]["persistent_cache"] == arms["on"]["persistent_cache"]
        or arms["off"]["hf_home"] == arms["on"]["hf_home"]
    ):
        raise SystemExit("Pair arm write roots are not disjoint")
    return value


execution_environment = validate_execution_environment(
    pair_manifest_document.get("execution_environment")
)


def validate_wandb(value):
    derivation = (
        "sha256-ascii:nemo-rl-strict-wandb-v1:"
        "{environment}:{pair_id}:{arm}"
    )
    expected = {
        "arms": {
            arm: {
                "name": f"{arm}-{environment}-{pair_id}",
                "name_template": f"{arm}-{{environment}}-{{pair_id}}",
                "run_id": hashlib.sha256(
                    (
                        f"nemo-rl-strict-wandb-v1:{environment}:"
                        f"{pair_id}:{arm}"
                    ).encode("ascii")
                ).hexdigest(),
            }
            for arm in ("off", "on")
        },
        "entity": "nvidia",
        "group": {
            "template": "{environment}-{pair_id}",
            "value": f"{environment}-{pair_id}",
        },
        "project": "nano35-rlvr-convergence",
        "resume": "never",
        "run_id_derivation": derivation,
    }
    if value != expected:
        raise SystemExit("Pair W&B identity differs from policy")
    return value


wandb = validate_wandb(pair_manifest_document.get("wandb"))

off_job_id = job_id_or_none(off_job_id_raw)
on_job_id = job_id_or_none(on_job_id_raw)
if off_writer_drained_raw not in {"false", "true"} or on_writer_drained_raw not in {
    "false",
    "true",
}:
    raise SystemExit("submission writer-drain evidence must be a boolean")
held_submissions = {
    "off": {
        "accepted_id_record": {
            "path": off_accepted_id_record_path,
            "parsed_job_id": job_id_or_none(off_record_job_id_raw),
            "sha256": off_accepted_id_record_sha256,
        },
        "candidate_job_id": off_job_id,
        "candidate_job_id_source": off_job_id_source,
        "candidate_job_id_sha256_ascii_no_newline": digest_or_none(
            off_held_sha256_raw
        ),
        "submission_rpc": {
            "drained_unix_ns": positive_count_or_none(off_drained_unix_ns_raw),
            "relay_status": status_or_none(off_relay_status_raw),
            "sbatch_status": status_or_none(off_sbatch_status_raw),
            "started_unix_ns": positive_count_or_none(off_started_unix_ns_raw),
            "writer_drained": off_writer_drained_raw == "true",
        },
        "wrapper_status": status_or_none(off_wrapper_status_raw),
    },
    "on": {
        "accepted_id_record": {
            "path": on_accepted_id_record_path,
            "parsed_job_id": job_id_or_none(on_record_job_id_raw),
            "sha256": on_accepted_id_record_sha256,
        },
        "candidate_job_id": on_job_id,
        "candidate_job_id_source": on_job_id_source,
        "candidate_job_id_sha256_ascii_no_newline": digest_or_none(
            on_held_sha256_raw
        ),
        "submission_rpc": {
            "drained_unix_ns": positive_count_or_none(on_drained_unix_ns_raw),
            "relay_status": status_or_none(on_relay_status_raw),
            "sbatch_status": status_or_none(on_sbatch_status_raw),
            "started_unix_ns": positive_count_or_none(on_started_unix_ns_raw),
            "writer_drained": on_writer_drained_raw == "true",
        },
        "wrapper_status": status_or_none(on_wrapper_status_raw),
    },
}
for record in held_submissions.values():
    if record["submission_rpc"]["drained_unix_ns"] is not None and (
        record["submission_rpc"]["started_unix_ns"] is None
        or record["submission_rpc"]["drained_unix_ns"]
        < record["submission_rpc"]["started_unix_ns"]
    ):
        raise SystemExit("submission RPC drain timestamp precedes its start")
    if record["submission_rpc"]["writer_drained"] != (
        record["submission_rpc"]["relay_status"] == 0
    ):
        raise SystemExit("submission writer-drain evidence is inconsistent")
    if record["wrapper_status"] is None:
        raise SystemExit("held-submission wrapper status cannot be null")
    if record["candidate_job_id"] is None:
        if record["candidate_job_id_source"] != "none":
            raise SystemExit("missing held job ID has a nonempty authority source")
        if record["candidate_job_id_sha256_ascii_no_newline"] is not None:
            raise SystemExit("missing held job ID has a canonical job-ID SHA-256")
    else:
        if record["candidate_job_id_source"] not in {
            "accepted-id-record",
            "scheduler-recovery",
        }:
            raise SystemExit("held job ID has an invalid authority source")
        expected = hashlib.sha256(
            record["candidate_job_id"].encode("ascii")
        ).hexdigest()
        if record["candidate_job_id_sha256_ascii_no_newline"] != expected:
            raise SystemExit("held job-ID SHA-256 does not bind the decimal job ID")
        if record["candidate_job_id_source"] == "accepted-id-record" and (
            record["accepted_id_record"]["parsed_job_id"]
            != record["candidate_job_id"]
        ):
            raise SystemExit("record-backed job ID differs from parsed durable bytes")
        if record["candidate_job_id_source"] == "scheduler-recovery" and (
            record["accepted_id_record"]["parsed_job_id"]
            == record["candidate_job_id"]
        ):
            raise SystemExit("scheduler recovery redundantly claims record-backed ID")
    parsed_record_job_id = record["accepted_id_record"]["parsed_job_id"]
    if parsed_record_job_id is not None:
        expected_record = hashlib.sha256(
            (parsed_record_job_id + "\n").encode("ascii")
        ).hexdigest()
        if record["accepted_id_record"]["sha256"] != expected_record:
            raise SystemExit(
                "accepted-ID record SHA-256 does not bind parsed decimal plus LF"
            )
job_ids = [value for value in (off_job_id, on_job_id) if value is not None]
if len(job_ids) != len(set(job_ids)) and stage != "job_id_validation":
    raise SystemExit("duplicate job IDs are only valid as rejected evidence")
pair_job_ids_csv = ",".join(job_ids)
recovery_scan_unterminated_candidate_job_ids = list_or_none(
    recovery_scan_unterminated_candidate_job_ids_raw
)
rollback_candidates = {
    "off": job_ids_csv_or_empty(off_cleanup_candidate_job_ids_csv),
    "on": job_ids_csv_or_empty(on_cleanup_candidate_job_ids_csv),
    "unattributed": (
        []
        if recovery_scan_unterminated_candidate_job_ids is None
        else recovery_scan_unterminated_candidate_job_ids
    ),
}
rollback_candidate_ids = []
for arm in ("off", "on", "unattributed"):
    for job_id in rollback_candidates[arm]:
        if job_id not in rollback_candidate_ids:
            rollback_candidate_ids.append(job_id)
rollback_candidate_ids_csv = ",".join(rollback_candidate_ids)
pre_query = raw_query_or_none(pre_release_query_raw)
post_query = raw_query_or_none(post_release_query_raw)
pre_release_by_arm = {"off": [], "on": []}
post_release_by_arm = {"off": [], "on": []}
for query, phase in ((pre_query, "pre"), (post_query, "post")):
    if query is None:
        continue
    attempts, _authenticated, authenticated_by_arm = validate_raw_candidate_queries(
        json.dumps([query], sort_keys=True, separators=(",", ":")),
        phase,
        job_ids,
    )
    if len(attempts) != 1:
        raise SystemExit("lifecycle scheduler query must contain one attempt")
    if phase == "pre":
        pre_release_by_arm = authenticated_by_arm
    else:
        post_release_by_arm = authenticated_by_arm
recovery_scan_status = status_or_none(recovery_scan_status_raw)
recovery_scan_normalization_status = status_or_none(
    recovery_scan_normalization_status_raw
)
recovery_scan_sha256 = digest_or_none(recovery_scan_sha256_raw)
recovery_scan_byte_count = count_or_none(recovery_scan_byte_count_raw)
recovery_scan_line_count = count_or_none(recovery_scan_line_count_raw)
recovery_scan_parsed_record_count = count_or_none(
    recovery_scan_parsed_record_count_raw
)
recovery_scan_identity_match_counts = records_or_none(
    recovery_scan_identity_match_counts_raw
)
recovery_scan_records = records_or_none(recovery_scan_records_raw)
recovery_authenticated_job_ids = []
if recovery_scan_unterminated_final_line_raw == "-":
    recovery_scan_unterminated_final_line = None
elif recovery_scan_unterminated_final_line_raw in {"false", "true"}:
    recovery_scan_unterminated_final_line = (
        recovery_scan_unterminated_final_line_raw == "true"
    )
else:
    raise SystemExit("recovery scan final-line evidence is malformed")
if recovery_scan_status is None:
    if recovery_scan_securely_unlinked_raw != "-":
        raise SystemExit("absent scheduler recovery scan has cleanup evidence")
    if any(
        value is not None
        for value in (
            recovery_scan_sha256,
            recovery_scan_normalization_status,
            recovery_scan_byte_count,
            recovery_scan_line_count,
            recovery_scan_parsed_record_count,
            recovery_scan_identity_match_counts,
            recovery_scan_records,
            recovery_scan_unterminated_final_line,
            recovery_scan_unterminated_candidate_job_ids,
        )
    ):
        raise SystemExit("absent scheduler recovery scan has evidence")
    recovery_scan = None
else:
    if (
        recovery_scan_sha256 is None
        or recovery_scan_normalization_status is None
    ):
        raise SystemExit("attempted scheduler recovery scan lacks base evidence")
    if recovery_scan_normalization_status == 0:
        if (
            recovery_scan_byte_count is None
            or recovery_scan_line_count is None
            or recovery_scan_parsed_record_count is None
            or not isinstance(recovery_scan_unterminated_final_line, bool)
            or not isinstance(recovery_scan_identity_match_counts, dict)
            or set(recovery_scan_identity_match_counts) != {"off", "on"}
            or any(
                type(value) is not int or value < 0
                for value in recovery_scan_identity_match_counts.values()
            )
            or not isinstance(recovery_scan_records, dict)
            or set(recovery_scan_records) != {"off", "on"}
            or not isinstance(recovery_scan_unterminated_candidate_job_ids, list)
            or any(
                not isinstance(job_id, str)
                or re.fullmatch(r"[1-9][0-9]*", job_id) is None
                for job_id in recovery_scan_unterminated_candidate_job_ids
            )
            or len(recovery_scan_unterminated_candidate_job_ids)
            != len(set(recovery_scan_unterminated_candidate_job_ids))
            or recovery_scan_unterminated_candidate_job_ids
            != sorted(recovery_scan_unterminated_candidate_job_ids, key=int)
            or (
                bool(recovery_scan_unterminated_candidate_job_ids)
                and not recovery_scan_unterminated_final_line
            )
        ):
            raise SystemExit("normalized scheduler recovery scan lacks exact evidence")
        if any(
            not isinstance(recovery_scan_records[arm], list)
            for arm in ("off", "on")
        ):
            raise SystemExit("recovery scan identity matches must be per-arm lists")
        recovery_record_job_ids = {"off": [], "on": []}
        for arm in ("off", "on"):
            expected_comment = (
                f"nemo-rl-strict-pair-v1:{arm}:{submission_nonce}:"
                f"{pair_manifest_sha256}"
            )
            if recovery_scan_identity_match_counts[arm] != len(
                recovery_scan_records[arm]
            ):
                raise SystemExit("recovery scan match counts contradict its records")
            for record in recovery_scan_records[arm]:
                if not isinstance(record, dict) or set(record) != {
                    "comment",
                    "held",
                    "job_id",
                    "job_name",
                    "job_state",
                    "reason",
                    "user_id",
                    "work_dir",
                }:
                    raise SystemExit("recovery scan record has unexpected keys")
                job_id = record["job_id"]
                if (
                    not isinstance(job_id, str)
                    or re.fullmatch(r"[1-9][0-9]*", job_id) is None
                    or record["comment"] != expected_comment
                    or record["job_name"] != f"{arm}-{environment}-{pair_id}"
                    or record["user_id"] != str(os.geteuid())
                    or record["work_dir"]
                    != execution_environment["arms"][arm]["scheduler"][
                        "batch_working_directory"
                    ]
                    or type(record["held"]) is not bool
                    or any(
                        not isinstance(record[key], str)
                        for key in ("job_state", "reason", "work_dir")
                    )
                ):
                    raise SystemExit("recovery scan record identity is invalid")
                derived_held = (
                    record["job_state"] == "PENDING"
                    and record["reason"] == "JobHeldUser"
                )
                if record["held"] != derived_held:
                    raise SystemExit("recovery scan held derivation is invalid")
                if not derived_held:
                    raise SystemExit("recovery scan identity record was not held")
                if job_id not in recovery_record_job_ids[arm]:
                    recovery_record_job_ids[arm].append(job_id)
            recovery_record_job_ids[arm].sort(key=int)
        for arm in ("off", "on"):
            preferred = held_submissions[arm]["accepted_id_record"]["parsed_job_id"]
            if preferred in recovery_record_job_ids[arm]:
                recovery_record_job_ids[arm].remove(preferred)
                recovery_record_job_ids[arm].insert(0, preferred)
            for job_id in recovery_record_job_ids[arm]:
                if job_id not in recovery_authenticated_job_ids:
                    recovery_authenticated_job_ids.append(job_id)
        if not set(recovery_authenticated_job_ids) <= set(rollback_candidate_ids):
            raise SystemExit("recovery scan identity is absent from rollback candidates")
    elif any(
        value is not None
        for value in (
            recovery_scan_byte_count,
            recovery_scan_line_count,
            recovery_scan_parsed_record_count,
            recovery_scan_identity_match_counts,
            recovery_scan_records,
            recovery_scan_unterminated_final_line,
            recovery_scan_unterminated_candidate_job_ids,
        )
    ):
        raise SystemExit("failed recovery normalization has parsed evidence")
    recovery_scan = {
        "argv": [scontrol_path, "show", "job", "--oneliner"],
        "byte_count": recovery_scan_byte_count,
        "identity_match_counts": recovery_scan_identity_match_counts,
        "line_count": recovery_scan_line_count,
        "normalization_status": recovery_scan_normalization_status,
        "output_sha256_raw": recovery_scan_sha256,
        "parsed_record_count": recovery_scan_parsed_record_count,
        "records": recovery_scan_records,
        "securely_unlinked": recovery_scan_securely_unlinked_raw == "true",
        "status": recovery_scan_status,
        "unterminated_final_line": recovery_scan_unterminated_final_line,
        "unterminated_candidate_job_ids": (
            recovery_scan_unterminated_candidate_job_ids
        ),
    }
    if recovery_scan_normalization_status != 0:
        recovery_authenticated_job_ids = []
    if recovery_scan_securely_unlinked_raw != "true":
        raise SystemExit("scheduler recovery scan was not securely removed")

all_recovery_candidate_ids = []
for arm in ("off", "on"):
    for job_id in rollback_candidates[arm]:
        if job_id not in all_recovery_candidate_ids:
            all_recovery_candidate_ids.append(job_id)
if recovery_scan_unterminated_candidate_job_ids is not None:
    for job_id in recovery_scan_unterminated_candidate_job_ids:
        if job_id not in all_recovery_candidate_ids:
            all_recovery_candidate_ids.append(job_id)
if outcome == "released":
    pre_cancel_queries, pre_cancel_authenticated_job_ids, pre_cancel_by_arm = (
        validate_raw_candidate_queries(pre_cancel_queries_raw, "identity", [])
    )
else:
    pre_cancel_queries, pre_cancel_authenticated_job_ids, pre_cancel_by_arm = (
        validate_raw_candidate_queries(
            pre_cancel_queries_raw, "identity", all_recovery_candidate_ids
        )
    )
for arm in ("off", "on"):
    preferred = held_submissions[arm]["accepted_id_record"]["parsed_job_id"]
    unique = sorted(set(pre_cancel_by_arm[arm]), key=int)
    if preferred in unique:
        unique.remove(preferred)
        unique.insert(0, preferred)
    pre_cancel_by_arm[arm] = unique
pre_cancel_authenticated_job_ids = []
for arm in ("off", "on"):
    for job_id in pre_cancel_by_arm[arm]:
        if job_id not in pre_cancel_authenticated_job_ids:
            pre_cancel_authenticated_job_ids.append(job_id)

release_status = status_or_none(release_status_raw)
release_sha256 = digest_or_none(release_sha256_raw)
if release_status is None:
    if release_sha256 is not None:
        raise SystemExit("absent release has output SHA-256")
    release = None
else:
    if release_sha256 is None:
        raise SystemExit("attempted release lacks output SHA-256")
    release = {
        "argv": [scontrol_path, "release", pair_job_ids_csv],
        "output_sha256_ascii_no_newline": release_sha256,
        "status": release_status,
    }

cancel_status = status_or_none(cancel_status_raw)
cancellations = []
if cancel_status is not None:
    cancel_job_ids = cancel_job_ids_csv.split(",")
    if not cancel_job_ids or any(
        re.fullmatch(r"[0-9]+", value) is None or int(value) == 0
        for value in cancel_job_ids
    ):
        raise SystemExit("cancellation has malformed job IDs")
    if len(cancel_job_ids) != len(set(cancel_job_ids)):
        raise SystemExit("cancellation has duplicate job IDs")
    cancel_sha256 = digest_or_none(cancel_sha256_raw)
    if cancel_sha256 is None:
        raise SystemExit("attempted cancellation lacks output SHA-256")
    if cancel_job_ids != pre_cancel_authenticated_job_ids:
        raise SystemExit("cancellation differs from the authenticated pre-cancel set")
    post_cancel_queries, post_cancel_authenticated_job_ids, post_cancel_by_arm = (
        validate_raw_candidate_queries(
            post_cancel_queries_raw, "cancel", cancel_job_ids
        )
    )
    post_cancel_authenticated_set = set(post_cancel_authenticated_job_ids)
    post_cancel_authenticated_job_ids = [
        job_id for job_id in cancel_job_ids if job_id in post_cancel_authenticated_set
    ]
    cancellations.append(
        {
            "argv": [scancel_path, cancel_job_ids_csv],
            "job_ids": cancel_job_ids,
            "output_sha256_ascii_no_newline": cancel_sha256,
            "status": cancel_status,
        }
    )
elif cancel_sha256_raw != "-" or cancel_job_ids_csv != "-":
    raise SystemExit("absent cancellation has evidence")
else:
    cancel_job_ids = []
    post_cancel_queries, post_cancel_authenticated_job_ids, post_cancel_by_arm = (
        validate_raw_candidate_queries(post_cancel_queries_raw, "cancel", [])
    )
    if pre_cancel_authenticated_job_ids:
        raise SystemExit("authenticated pre-cancel jobs were not canceled")

authenticated_job_ids_by_arm = {"off": [], "on": []}
if outcome == "released":
    for arm in ("off", "on"):
        authenticated_job_ids_by_arm[arm] = [
            job_id
            for job_id in pre_release_by_arm[arm]
            if job_id in post_release_by_arm[arm]
        ]
else:
    authenticated_job_ids_by_arm = pre_cancel_by_arm


def authenticated_job_label(arm, job_id):
    return {
        "comment": (
            f"nemo-rl-strict-pair-v1:{arm}:{submission_nonce}:"
            f"{pair_manifest_sha256}"
        ),
        "job_id": job_id,
        "job_name": f"{arm}-{environment}-{pair_id}",
        "user_id": str(os.geteuid()),
    }


authenticated_jobs = {
    arm: [
        authenticated_job_label(arm, job_id)
        for job_id in authenticated_job_ids_by_arm[arm]
    ]
    for arm in ("off", "on")
}

recovery_scan_complete = (
    recovery_scan is not None
    and recovery_scan["status"] == 0
    and recovery_scan["normalization_status"] == 0
    and not recovery_scan["unterminated_final_line"]
    and recovery_scan["parsed_record_count"] == recovery_scan["line_count"]
)
no_job_evidence_confirmed = (
    not recovery_authenticated_job_ids
    and not rollback_candidate_ids
    and recovery_scan_complete
    and all(
        record["accepted_id_record"]["parsed_job_id"] is None
        and record["accepted_id_record"]["sha256"]
        == hashlib.sha256(b"").hexdigest()
        for record in held_submissions.values()
    )
)
cancellation_evidence_confirmed = (
    bool(recovery_authenticated_job_ids)
    and recovery_scan_complete
    and set(pre_cancel_authenticated_job_ids) == set(all_recovery_candidate_ids)
    and len(pre_cancel_authenticated_job_ids) == len(all_recovery_candidate_ids)
    and len(cancellations) == 1
    and cancellations[0]["job_ids"] == recovery_authenticated_job_ids
    and cancellations[0]["status"] == 0
    and post_cancel_authenticated_job_ids == recovery_authenticated_job_ids
)
rollback_evidence_confirmed = (
    no_job_evidence_confirmed or cancellation_evidence_confirmed
)
if outcome == "failed-closed" and not rollback_evidence_confirmed:
    raise SystemExit("failed-closed outcome lacks confirmed rollback evidence")
if outcome == "rollback-unconfirmed" and rollback_evidence_confirmed:
    raise SystemExit("rollback-unconfirmed outcome contradicts scheduler evidence")

document = {
    "acceptance": acceptance,
    "authenticated_jobs": authenticated_jobs,
    "cancellations": cancellations,
    "execution_environment": execution_environment,
    "held_submissions": held_submissions,
    "model_transport": model_transport,
    "outcome": outcome,
    "pair": {
        "id": pair_id,
        "manifest": {
            "path": pair_manifest_path,
            "sha256": pair_manifest_sha256,
        },
    },
    "post_cancel_queries": post_cancel_queries,
    "post_release_query": post_query,
    "pre_cancel_queries": pre_cancel_queries,
    "pre_release_query": pre_query,
    "recovery_query": recovery_scan,
    "release": release,
    "receipt": {
        "path": str(pathlib.Path(receipt_raw)),
        "schema": "nemo-rl-strict-pair-submission-receipt-v2",
    },
    "rollback_candidates": rollback_candidates,
    "rollback_confirmed": rollback_confirmed,
    "runtime_tools": {
        "manifest": {
            "path": runtime_tool_manifest_path,
            "sha256": runtime_tool_manifest_sha256,
        },
        "schema": "nemo-rl-strict-runtime-tools-v2",
    },
    "scheduler_tools": {
        "client_environment": {
            "ambient_merge": False,
            "env": {"path": env_path, "sha256": env_sha256},
            "variables": {
                "LC_ALL": "C",
                "SLURM_CONF": {
                    "path": slurm_conf_path,
                    "sha256": slurm_conf_sha256,
                },
            },
        },
        "sbatch": {"path": sbatch_path, "sha256": sbatch_sha256},
        "scancel": {"path": scancel_path, "sha256": scancel_sha256},
        "scontrol": {"path": scontrol_path, "sha256": scontrol_sha256},
    },
    "schema": "nemo-rl-strict-pair-submission-receipt-v2",
    "selection": selection,
    "source": receipt_source,
    "stage": stage,
    "submission_contract": {
        "path": submission_contract_path,
        "sha256": submission_contract_sha256,
    },
    "submission_nonce": submission_nonce,
    "wandb": wandb,
}
try:
    submission_contract_document = json.loads(
        pathlib.Path(submission_contract_path).read_bytes()
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit("submission contract is not valid JSON") from error
if submission_contract_document.get("schema") != (
    "nemo-rl-strict-pair-submission-contract-v2"
):
    raise SystemExit("submission contract schema differs from receipt producer")
if submission_contract_document.get("pair") != {"environment": environment}:
    raise SystemExit("submission contract environment differs from Pair selection")
receipt_contract = submission_contract_document.get("receipt")
if not isinstance(receipt_contract, dict) or receipt_contract.get("schema") != (
    "nemo-rl-strict-pair-submission-receipt-v2"
):
    raise SystemExit("submission contract receipt schema differs from producer")
if receipt_contract.get("required_root_keys") != sorted(document):
    raise SystemExit("submission contract receipt root keys differ from producer")
contract_examples = receipt_contract.get("examples")
if not isinstance(contract_examples, dict) or set(contract_examples) != {
    "failed_closed",
    "released",
    "rollback_unconfirmed",
}:
    raise SystemExit("submission contract lacks exact normative receipt examples")
if any(not isinstance(example, dict) for example in contract_examples.values()) or any(
    set(example) != set(document) for example in contract_examples.values()
):
    raise SystemExit("submission contract example root shape differs from producer")
nullable_object_fields = [
    "post_release_query",
    "pre_release_query",
    "recovery_query",
    "release",
]
if receipt_contract.get("nullable_object_fields") != nullable_object_fields:
    raise SystemExit("submission contract nullable object fields differ from producer")
for field in nullable_object_fields:
    values = [example[field] for example in contract_examples.values()]
    object_examples = [value for value in values if isinstance(value, dict)]
    if (
        any(value is not None and not isinstance(value, dict) for value in values)
        or not object_examples
        or any(set(value) != set(object_examples[0]) for value in object_examples)
    ):
        raise SystemExit(
            f"submission contract nullable object shape differs at receipt.{field}"
        )


def validate_nested_key_shape(actual, example, path):
    field = path.removeprefix("receipt.")
    if field in nullable_object_fields:
        if actual is None:
            return
        if not isinstance(actual, dict):
            raise SystemExit(f"submission contract expected object or null at {path}")
        example = next(
            value
            for value in (candidate[field] for candidate in contract_examples.values())
            if isinstance(value, dict)
        )
    if isinstance(actual, dict):
        if not isinstance(example, dict) or set(actual) != set(example):
            raise SystemExit(f"submission contract nested shape differs at {path}")
        for key in actual:
            validate_nested_key_shape(actual[key], example[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(example, list):
            raise SystemExit(f"submission contract array shape differs at {path}")
        if actual and example:
            for index, item in enumerate(actual):
                validate_nested_key_shape(item, example[0], f"{path}[{index}]")


example_name = {
    "failed-closed": "failed_closed",
    "released": "released",
    "rollback-unconfirmed": "rollback_unconfirmed",
}[outcome]
validate_nested_key_shape(document, contract_examples[example_name], "receipt")
if outcome == "released":
    if (
        off_job_id is None
        or on_job_id is None
        or len(job_ids) != 2
        or any(record["wrapper_status"] != 0 for record in held_submissions.values())
        or any(
            len(authenticated_jobs[arm]) != 1
            or authenticated_jobs[arm][0]["job_id"]
            != held_submissions[arm]["candidate_job_id"]
            for arm in ("off", "on")
        )
        or any(
            record["submission_rpc"]["sbatch_status"] != 0
            or record["submission_rpc"]["relay_status"] != 0
            or not record["submission_rpc"]["writer_drained"]
            for record in held_submissions.values()
        )
        or not recovery_scan_complete
        or recovery_scan["identity_match_counts"] != {"off": 1, "on": 1}
        or any(
            len(recovery_scan["records"][arm]) != 1
            for arm in ("off", "on")
        )
        or recovery_authenticated_job_ids != job_ids
        or rollback_candidate_ids != job_ids
        or any(
            record["candidate_job_id_sha256_ascii_no_newline"] is None
            for record in held_submissions.values()
        )
        or pre_query is None
        or pre_query["status"] != 0
        or not pre_query["complete"]
        or pre_query["unresolved_job_ids"]
        or release is None
        or release["status"] != 0
        or post_query is None
        or post_query["status"] != 0
        or not post_query["complete"]
        or post_query["unresolved_job_ids"]
        or cancellations
    ):
        raise SystemExit("released receipt lacks complete successful lifecycle evidence")

receipt = pathlib.Path(receipt_raw)
if receipt.name != "PAIR_SUBMISSION_RECEIPT.json":
    raise SystemExit("submission receipt must use the canonical filename")
payload = (
    json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    + b"\n"
)
digest = hashlib.sha256(payload).hexdigest()
if operation == "adopt":
    metadata = os.lstat(receipt)
    if (
        not receipt.is_absolute()
        or os.path.realpath(receipt) != str(receipt)
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or receipt.read_bytes() != payload
    ):
        raise SystemExit("committed receipt differs from exact publisher bytes")
    if metadata.st_nlink != 1:
        for entry in receipt.parent.iterdir():
            if not entry.name.startswith(".PAIR_SUBMISSION_RECEIPT.json.candidate."):
                continue
            entry_metadata = os.lstat(entry)
            if (
                entry_metadata.st_dev != metadata.st_dev
                or entry_metadata.st_ino != metadata.st_ino
            ):
                continue
            if (
                not stat.S_ISREG(entry_metadata.st_mode)
                or stat.S_IMODE(entry_metadata.st_mode) != 0o400
                or entry.read_bytes() != payload
            ):
                raise SystemExit("committed receipt candidate link is invalid")
            os.unlink(entry)
        directory_fd = os.open(
            receipt.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        reconciled_metadata = os.lstat(receipt)
        if (
            reconciled_metadata.st_dev != metadata.st_dev
            or reconciled_metadata.st_ino != metadata.st_ino
            or reconciled_metadata.st_nlink != 1
            or receipt.read_bytes() != payload
        ):
            raise SystemExit("committed receipt candidate links were not reconciled")
    print(digest)
    raise SystemExit(0)
if receipt.is_symlink() or receipt.exists():
    raise SystemExit("submission receipt target already exists")
parent = receipt.parent
metadata = os.lstat(parent)
if os.path.realpath(parent) != str(parent) or not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("submission receipt parent must be one canonical directory")
if stat.S_IMODE(metadata.st_mode) != 0o700:
    raise SystemExit("submission receipt parent must have mode 700")
candidate_fd, candidate_raw = tempfile.mkstemp(
    prefix=".PAIR_SUBMISSION_RECEIPT.json.candidate.", dir=parent
)
candidate = pathlib.Path(candidate_raw)
candidate_metadata = os.fstat(candidate_fd)
link_committed = False
try:
    os.fchmod(candidate_fd, 0o400)
    payload_view = memoryview(payload)
    while payload_view:
        written = os.write(candidate_fd, payload_view)
        if written <= 0:
            raise SystemExit("submission receipt candidate write made no progress")
        payload_view = payload_view[written:]
    os.fsync(candidate_fd)
    os.close(candidate_fd)
    candidate_fd = -1
    os.link(candidate, receipt)
    link_committed = True
finally:
    if candidate_fd >= 0:
        os.close(candidate_fd)
    candidate_cleanup_error = None
    for _attempt in range(2):
        try:
            named_candidate_metadata = os.lstat(candidate)
        except FileNotFoundError:
            candidate_cleanup_error = None
            break
        if (
            named_candidate_metadata.st_dev != candidate_metadata.st_dev
            or named_candidate_metadata.st_ino != candidate_metadata.st_ino
            or not stat.S_ISREG(named_candidate_metadata.st_mode)
        ):
            raise SystemExit("submission receipt candidate identity changed")
        try:
            os.unlink(candidate)
        except OSError as error:
            candidate_cleanup_error = error
        else:
            candidate_cleanup_error = None
            break
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if candidate_cleanup_error is not None:
        raise candidate_cleanup_error
if not link_committed:
    raise SystemExit("submission receipt link was not committed")
receipt_metadata = os.lstat(receipt)
if (
    receipt_metadata.st_dev != candidate_metadata.st_dev
    or receipt_metadata.st_ino != candidate_metadata.st_ino
    or receipt_metadata.st_nlink != 1
    or stat.S_IMODE(receipt_metadata.st_mode) != 0o400
    or hashlib.sha256(receipt.read_bytes()).hexdigest() != digest
):
    raise SystemExit("published submission receipt changed after atomic publication")
print(digest)
PY
}

strict_pair_adopt_committed_submission_receipt() {
  trap '' INT TERM
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" \
    "${PAIR_ID}" "${STRICT_PAIR_SUBMISSION_NONCE}" \
    "${STRICT_PAIR_MANIFEST_PATH}" "${STRICT_PAIR_MANIFEST_SHA256}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

(
    receipt_raw,
    pair_id,
    nonce,
    pair_manifest_path,
    pair_manifest_sha256,
    contract_path,
    contract_sha256,
) = sys.argv[1:]
receipt = pathlib.Path(receipt_raw)
metadata = os.lstat(receipt)
if (
    not receipt.is_absolute()
    or os.path.realpath(receipt) != str(receipt)
    or receipt.name != "PAIR_SUBMISSION_RECEIPT.json"
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISREG(metadata.st_mode)
    or stat.S_IMODE(metadata.st_mode) != 0o400
):
    raise SystemExit("committed submission receipt is not one sealed canonical file")
payload = receipt.read_bytes()
try:
    document = json.loads(payload)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit("committed submission receipt is not canonical JSON") from error
canonical = (
    json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    + b"\n"
)
if payload != canonical:
    raise SystemExit("committed submission receipt bytes are not canonical")
if document.get("schema") != "nemo-rl-strict-pair-submission-receipt-v2":
    raise SystemExit("committed submission receipt schema mismatch")
if document.get("submission_nonce") != nonce or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
    raise SystemExit("committed submission receipt nonce mismatch")
if document.get("pair") != {
    "id": pair_id,
    "manifest": {"path": pair_manifest_path, "sha256": pair_manifest_sha256},
}:
    raise SystemExit("committed submission receipt Pair binding mismatch")
if document.get("submission_contract") != {
    "path": contract_path,
    "sha256": contract_sha256,
}:
    raise SystemExit("committed submission receipt contract binding mismatch")
if document.get("receipt") != {
    "path": receipt_raw,
    "schema": "nemo-rl-strict-pair-submission-receipt-v2",
}:
    raise SystemExit("committed submission receipt path self-binding mismatch")
for path_raw, expected_digest in (
    (pair_manifest_path, pair_manifest_sha256),
    (contract_path, contract_sha256),
):
    path = pathlib.Path(path_raw)
    path_metadata = os.lstat(path)
    if (
        not path.is_absolute()
        or os.path.realpath(path) != str(path)
        or stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISREG(path_metadata.st_mode)
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest
    ):
        raise SystemExit("committed submission receipt input anchor changed")


def reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON member in committed input: {key}")
        result[key] = value
    return result


try:
    pair_manifest_document = json.loads(
        pathlib.Path(pair_manifest_path).read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_members,
    )
    contract_document = json.loads(
        pathlib.Path(contract_path).read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_members,
    )
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit("committed submission receipt input is not strict JSON") from error
if pair_manifest_document.get("schema") != "nemo-rl-strict-single-env-pair-v2":
    raise SystemExit("committed Pair manifest schema mismatch")
if contract_document.get("schema") != "nemo-rl-strict-pair-submission-contract-v2":
    raise SystemExit("committed submission contract schema mismatch")
selection = pair_manifest_document.get("selection")
if not isinstance(selection, dict) or set(selection) != {
    "config",
    "environment",
    "fixture",
    "gym_resources",
}:
    raise SystemExit("committed Pair selection has unexpected keys")
environment = selection["environment"]
selection_paths = {
    "reasoning_gym": {
        "config": (
            "examples/nemo_gym/nemotron-3.5-nano/"
            "single_env_reasoning_gym_sc.yaml"
        ),
        "gym_resources": {
            "config": "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
            "requirements": "resources_servers/reasoning_gym/requirements.txt",
            "verifier_source": "resources_servers/reasoning_gym/app.py",
        },
    },
    "citation": {
        "config": "examples/nemo_gym/nemotron-3.5-nano/single_env_citation_sc.yaml",
        "gym_resources": {
            "config": (
                "resources_servers/format_verification/configs/"
                "citation_format.yaml"
            ),
            "requirements": "resources_servers/format_verification/requirements.txt",
            "verifier_source": "resources_servers/format_verification/app.py",
        },
    },
    "freeform": {
        "config": "examples/nemo_gym/nemotron-3.5-nano/single_env_freeform_sc.yaml",
        "gym_resources": {
            "config": (
                "resources_servers/format_verification/configs/"
                "freeform_formatting.yaml"
            ),
            "requirements": "resources_servers/format_verification/requirements.txt",
            "verifier_source": "resources_servers/format_verification/app.py",
        },
    },
}
if environment not in selection_paths:
    raise SystemExit("committed Pair selection environment is invalid")
config = selection["config"]
fixture = selection["fixture"]
gym_resources = selection["gym_resources"]
digest = re.compile(r"[0-9a-f]{64}")
if (
    not isinstance(config, dict)
    or set(config) != {"path", "sha256"}
    or config.get("path") != selection_paths[environment]["config"]
    or not isinstance(config.get("sha256"), str)
    or digest.fullmatch(config["sha256"]) is None
    or not isinstance(fixture, dict)
    or set(fixture) != {"path", "rows", "sha256"}
    or not isinstance(fixture.get("path"), str)
    or not pathlib.Path(fixture["path"]).is_absolute()
    or type(fixture.get("rows")) is not int
    or fixture["rows"] != 5
    or not isinstance(fixture.get("sha256"), str)
    or digest.fullmatch(fixture["sha256"]) is None
    or not isinstance(gym_resources, dict)
    or set(gym_resources) != {"config", "requirements", "verifier_source"}
):
    raise SystemExit("committed Pair selection provenance is malformed")
for name, expected_path in selection_paths[environment]["gym_resources"].items():
    resource = gym_resources.get(name)
    if (
        not isinstance(resource, dict)
        or set(resource) != {"path", "sha256"}
        or resource.get("path") != expected_path
        or not isinstance(resource.get("sha256"), str)
        or digest.fullmatch(resource["sha256"]) is None
    ):
        raise SystemExit(f"committed selected Gym {name} is malformed")
artifacts = pair_manifest_document.get("artifacts")
source = pair_manifest_document.get("source")
if (
    not isinstance(artifacts, dict)
    or not isinstance(source, dict)
    or artifacts.get("fixture") != fixture
    or source.get("config_sha256") != config["sha256"]
):
    raise SystemExit("committed Pair selection provenance is inconsistent")
if document.get("selection") != selection:
    raise SystemExit("committed submission receipt selection binding mismatch")
if document.get("acceptance") != pair_manifest_document.get("acceptance"):
    raise SystemExit("committed submission receipt acceptance-policy binding mismatch")
if document.get("execution_environment") != pair_manifest_document.get(
    "execution_environment"
):
    raise SystemExit("committed submission receipt execution binding mismatch")
if document.get("model_transport") != pair_manifest_document.get("model_transport"):
    raise SystemExit("committed submission receipt model-transport binding mismatch")
if document.get("wandb") != pair_manifest_document.get("wandb"):
    raise SystemExit("committed submission receipt W&B binding mismatch")
if document.get("source") != {
    "bridge": source.get("bridge"),
    "mcore": source.get("mcore"),
}:
    raise SystemExit("committed submission receipt component-source binding mismatch")
if contract_document.get("pair") != {"environment": selection["environment"]}:
    raise SystemExit("committed submission contract environment binding mismatch")
receipt_contract = contract_document.get("receipt")
nullable_object_fields = [
    "post_release_query",
    "pre_release_query",
    "recovery_query",
    "release",
]
if (
    not isinstance(receipt_contract, dict)
    or receipt_contract.get("schema")
    != "nemo-rl-strict-pair-submission-receipt-v2"
    or receipt_contract.get("required_root_keys") != sorted(document)
    or receipt_contract.get("nullable_object_fields") != nullable_object_fields
):
    raise SystemExit("committed submission contract receipt shape mismatch")
outcome = document.get("outcome")
stage = document.get("stage")
rollback_confirmed = document.get("rollback_confirmed")
expected_rollback = {
    "failed-closed": True,
    "released": None,
    "rollback-unconfirmed": False,
}
if outcome not in expected_rollback or rollback_confirmed is not expected_rollback[outcome]:
    raise SystemExit("committed submission receipt outcome mismatch")


def validate_nested_key_shape(actual, example, path):
    field = path.removeprefix("receipt.")
    if field in nullable_object_fields:
        if actual is None:
            return
        if not isinstance(actual, dict):
            raise SystemExit(f"committed receipt expected object or null at {path}")
        example = next(
            value
            for value in (candidate[field] for candidate in examples.values())
            if isinstance(value, dict)
        )
    if isinstance(actual, dict):
        if not isinstance(example, dict) or set(actual) != set(example):
            raise SystemExit(f"committed receipt nested shape differs at {path}")
        for key in actual:
            validate_nested_key_shape(actual[key], example[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(example, list):
            raise SystemExit(f"committed receipt array shape differs at {path}")
        if actual and example:
            for index, item in enumerate(actual):
                validate_nested_key_shape(item, example[0], f"{path}[{index}]")


examples = receipt_contract.get("examples")
example_name = {
    "failed-closed": "failed_closed",
    "released": "released",
    "rollback-unconfirmed": "rollback_unconfirmed",
}[outcome]
if (
    not isinstance(examples, dict)
    or set(examples) != {"failed_closed", "released", "rollback_unconfirmed"}
    or any(not isinstance(example, dict) for example in examples.values())
    or any(set(example) != set(document) for example in examples.values())
):
    raise SystemExit("committed submission contract examples are malformed")
for field in nullable_object_fields:
    values = [example[field] for example in examples.values()]
    object_examples = [value for value in values if isinstance(value, dict)]
    if (
        any(value is not None and not isinstance(value, dict) for value in values)
        or not object_examples
        or any(set(value) != set(object_examples[0]) for value in object_examples)
    ):
        raise SystemExit(
            f"committed receipt nullable object shape differs at receipt.{field}"
        )
validate_nested_key_shape(document, examples[example_name], "receipt")
if outcome == "released" and stage != "complete":
    raise SystemExit("committed released receipt stage mismatch")
held = document.get("held_submissions")
if not isinstance(held, dict) or set(held) != {"off", "on"}:
    raise SystemExit("committed submission receipt lacks both arms")
authenticated_jobs = document.get("authenticated_jobs")
if not isinstance(authenticated_jobs, dict) or set(authenticated_jobs) != {
    "off",
    "on",
}:
    raise SystemExit("committed submission receipt lacks authenticated-job labels")
candidate_job_ids = []
identity_authenticated = []
authenticated_label_ids = {"off": [], "on": []}
for arm in ("off", "on"):
    record = held[arm]
    if not isinstance(record, dict):
        raise SystemExit("committed held-submission record is malformed")
    candidate_job_id = record.get("candidate_job_id")
    if candidate_job_id is not None and re.fullmatch(
        r"[1-9][0-9]*", candidate_job_id
    ) is None:
        raise SystemExit("committed submission receipt has malformed candidate job ID")
    labels = authenticated_jobs[arm]
    if not isinstance(labels, list):
        raise SystemExit("committed authenticated-job labels must be an array")
    expected_comment = (
        f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair_manifest_sha256}"
    )
    for label in labels:
        if not isinstance(label, dict) or set(label) != {
            "comment",
            "job_id",
            "job_name",
            "user_id",
        }:
            raise SystemExit("committed authenticated-job label is malformed")
        if (
            any(not isinstance(label[key], str) for key in label)
            or re.fullmatch(r"[1-9][0-9]*", label["job_id"]) is None
            or label["comment"] != expected_comment
            or label["job_name"] != f"{arm}-{environment}-{pair_id}"
            or label["user_id"] != str(os.geteuid())
        ):
            raise SystemExit("committed authenticated-job identity is malformed")
        authenticated_label_ids[arm].append(label["job_id"])
    if len(authenticated_label_ids[arm]) != len(set(authenticated_label_ids[arm])):
        raise SystemExit("committed authenticated-job labels contain duplicates")
    authenticated = (
        candidate_job_id is not None
        and sum(label["job_id"] == candidate_job_id for label in labels) == 1
    )
    candidate_job_ids.append(candidate_job_id or "none")
    identity_authenticated.append("true" if authenticated else "false")
if set(authenticated_label_ids["off"]) & set(authenticated_label_ids["on"]):
    raise SystemExit("committed authenticated job is attributed to both arms")
if outcome == "released" and any(
    authenticated_label_ids[arm] != [held[arm].get("candidate_job_id")]
    for arm in ("off", "on")
):
    raise SystemExit("committed released receipt lacks exact arm authentication")
recovery = document.get("recovery_query")
if recovery is not None and (
    not isinstance(recovery, dict) or recovery.get("securely_unlinked") is not True
):
    raise SystemExit("committed submission receipt lacks recovery cleanup proof")
print(
    "\t".join(
        (
            outcome,
            stage,
            candidate_job_ids[0],
            identity_authenticated[0],
            candidate_job_ids[1],
            identity_authenticated[1],
            hashlib.sha256(payload).hexdigest(),
        )
    )
)
PY
}

strict_pair_adopt_and_emit_committed_receipt() {
  local adopted_receipt
  local adopted_outcome adopted_stage adopted_off_candidate_job_id \
    adopted_off_identity_authenticated adopted_on_candidate_job_id \
    adopted_on_identity_authenticated adopted_receipt_sha256
  local adopted_rollback_confirmed
  local exact_receipt_sha256

  [[ -f "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" && \
      ! -L "${STRICT_PAIR_SUBMISSION_RECEIPT_PATH}" ]] || return 1
  adopted_receipt="$(
    strict_pair_adopt_committed_submission_receipt 2>/dev/null
  )" || return 1
  IFS=$'\t' read -r adopted_outcome adopted_stage \
    adopted_off_candidate_job_id adopted_off_identity_authenticated \
    adopted_on_candidate_job_id adopted_on_identity_authenticated \
    adopted_receipt_sha256 \
    <<< "${adopted_receipt}"
  case "${adopted_outcome}" in
    released) adopted_rollback_confirmed=null ;;
    failed-closed) adopted_rollback_confirmed=true ;;
    rollback-unconfirmed) adopted_rollback_confirmed=false ;;
    *) return 1 ;;
  esac
  exact_receipt_sha256="$(
    strict_pair_publish_submission_receipt \
      "${adopted_outcome}" "${adopted_stage}" \
      "${adopted_rollback_confirmed}" adopt
  )" || return 1
  [[ "${exact_receipt_sha256}" == "${adopted_receipt_sha256}" ]] || return 1
  [[ "${adopted_off_candidate_job_id}" == "none" ]] || \
    off_job_id="${adopted_off_candidate_job_id}"
  [[ "${adopted_on_candidate_job_id}" == "none" ]] || \
    on_job_id="${adopted_on_candidate_job_id}"
  off_identity_authenticated="${adopted_off_identity_authenticated}"
  on_identity_authenticated="${adopted_on_identity_authenticated}"
  strict_pair_emit_result \
    "${adopted_outcome}" "${adopted_stage}" \
    "${adopted_receipt_sha256}"
}

strict_pair_emit_result() {
  local outcome="$1"
  local stage="$2"
  local receipt_sha256="$3"
  local off_result_identity on_result_identity

  [[ "${recovery_scan_status}" == "-" || \
      "${recovery_scan_securely_unlinked}" == "true" ]] || return 2
  # The sealed receipt is the scheduler-state commit point.  Disarm
  # rollback before the infallible, single-line OOB handoff so EXIT can never
  # mutate jobs contrary to a committed receipt.
  STRICT_PAIR_TERMINAL_RECEIPT_PUBLISHED=1
  if [[ "${off_identity_authenticated}" == "true" ]]; then
    off_result_identity="off_job_id=${off_job_id:-none} off_identity_authenticated=true"
  else
    off_result_identity="off_candidate_job_id=${off_job_id:-none} off_identity_authenticated=false"
  fi
  if [[ "${on_identity_authenticated}" == "true" ]]; then
    on_result_identity="on_job_id=${on_job_id:-none} on_identity_authenticated=true"
  else
    on_result_identity="on_candidate_job_id=${on_job_id:-none} on_identity_authenticated=false"
  fi
  (
    trap '' PIPE
    printf '%s\n' "STRICT_PAIR_RESULT schema=nemo-rl-strict-pair-submission-receipt-v2 outcome=${outcome} stage=${stage} pair_id=${PAIR_ID} environment=${STRICT_PAIR_ENVIRONMENT} nonce=${STRICT_PAIR_SUBMISSION_NONCE} ${off_result_identity} ${on_result_identity} receipt_path=${STRICT_PAIR_SUBMISSION_RECEIPT_PATH} receipt_sha256=${receipt_sha256}"
  ) 2>/dev/null || true
}

strict_pair_fail_submission() {
  local stage="$1"
  local message="$2"
  local job_id
  local receipt_sha256
  local failure_outcome="failed-closed"
  local rollback_confirmed="true"
  local seen=""
  local candidate_job_ids_csv=""
  local all_candidate_job_ids_csv=""
  local off_cancel_candidate_job_ids_csv=""
  local on_cancel_candidate_job_ids_csv=""
  local off_authenticated_job_ids_csv=""
  local on_authenticated_job_ids_csv=""
  local off_post_cancel_authenticated_job_ids_csv=""
  local on_post_cancel_authenticated_job_ids_csv=""
  local query_off_job_ids_csv="-"
  local query_on_job_ids_csv="-"
  local query_unattributed_job_ids_csv="-"
  local preferred_job_id=""
  local query_index=0

  for job_id in \
      ${off_cleanup_candidate_job_ids_csv//,/ } \
      ${on_cleanup_candidate_job_ids_csv//,/ } \
      ${unterminated_candidate_job_ids_csv//,/ }; do
    if [[ ! "${job_id}" =~ ^[0-9]+$ || ",${seen}," == *",${job_id},"* ]]; then
      continue
    fi
    seen="${seen:+${seen},}${job_id}"
    all_candidate_job_ids_csv="${all_candidate_job_ids_csv:+${all_candidate_job_ids_csv},}${job_id}"
  done

  pre_cancel_queries="[]"
  pre_cancel_authenticated_job_ids_csv="-"
  pre_cancel_unresolved_job_ids_csv="-"
  if [[ -n "${recovery_scan_matched_job_ids_csv}" ]]; then
    if ! strict_pair_capture_raw_candidate_query \
        identity "${recovery_scan_matched_job_ids_csv}" \
        "${off_recovery_authenticated_job_ids_csv:--}" \
        "${on_recovery_authenticated_job_ids_csv:--}" - \
        "pre-${query_index}" pre_cancel_queries; then
      failure_outcome="rollback-unconfirmed"
      rollback_confirmed="false"
    else
      off_authenticated_job_ids_csv="$(
        strict_pair_canonical_job_id_union - \
          "${off_authenticated_job_ids_csv:--}" \
          "${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED}"
      )" || off_authenticated_job_ids_csv=""
      on_authenticated_job_ids_csv="$(
        strict_pair_canonical_job_id_union - \
          "${on_authenticated_job_ids_csv:--}" \
          "${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED}"
      )" || on_authenticated_job_ids_csv=""
    fi
    ((query_index += 1))
  fi

  # Independently retry every unresolved conservative candidate.  A stale
  # durable ID can therefore never suppress authentication/cancellation of a
  # distinct scheduler-backed ID, and an unterminated scan contributes only
  # its numeric JobId until this complete targeted query succeeds.
  for job_id in ${all_candidate_job_ids_csv//,/ }; do
    if [[ ",${off_authenticated_job_ids_csv},${on_authenticated_job_ids_csv}," == \
          *",${job_id},"* ]]; then
      continue
    fi
    query_off_job_ids_csv="-"
    query_on_job_ids_csv="-"
    query_unattributed_job_ids_csv="-"
    if [[ ",${unterminated_candidate_job_ids_csv}," == *",${job_id},"* ]]; then
      query_unattributed_job_ids_csv="${job_id}"
    else
      [[ ",${off_cleanup_candidate_job_ids_csv}," == *",${job_id},"* ]] && \
        query_off_job_ids_csv="${job_id}"
      [[ ",${on_cleanup_candidate_job_ids_csv}," == *",${job_id},"* ]] && \
        query_on_job_ids_csv="${job_id}"
    fi
    if ! strict_pair_capture_raw_candidate_query \
        identity "${job_id}" \
        "${query_off_job_ids_csv}" "${query_on_job_ids_csv}" \
        "${query_unattributed_job_ids_csv}" \
        "pre-${query_index}" pre_cancel_queries; then
      failure_outcome="rollback-unconfirmed"
      rollback_confirmed="false"
      ((query_index += 1))
      continue
    fi
    off_authenticated_job_ids_csv="$(
      strict_pair_canonical_job_id_union - \
        "${off_authenticated_job_ids_csv:--}" \
        "${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED}"
    )" || off_authenticated_job_ids_csv=""
    on_authenticated_job_ids_csv="$(
      strict_pair_canonical_job_id_union - \
        "${on_authenticated_job_ids_csv:--}" \
        "${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED}"
    )" || on_authenticated_job_ids_csv=""
    ((query_index += 1))
  done

  off_identity_authenticated=false
  on_identity_authenticated=false
  if [[ -n "${off_job_id}" && ",${off_authenticated_job_ids_csv}," == \
        *",${off_job_id},"* ]]; then
    off_identity_authenticated=true
  fi
  if [[ -n "${on_job_id}" && ",${on_authenticated_job_ids_csv}," == \
        *",${on_job_id},"* ]]; then
    on_identity_authenticated=true
  fi

  preferred_job_id=""
  if [[ -n "${off_record_job_id}" && \
        ",${off_authenticated_job_ids_csv}," == *",${off_record_job_id},"* ]]; then
    preferred_job_id="${off_record_job_id}"
  fi
  off_cancel_candidate_job_ids_csv="$(
    strict_pair_canonical_job_id_union \
      "${preferred_job_id}" "${off_authenticated_job_ids_csv:--}"
  )" || off_cancel_candidate_job_ids_csv=""
  preferred_job_id=""
  if [[ -n "${on_record_job_id}" && \
        ",${on_authenticated_job_ids_csv}," == *",${on_record_job_id},"* ]]; then
    preferred_job_id="${on_record_job_id}"
  fi
  on_cancel_candidate_job_ids_csv="$(
    strict_pair_canonical_job_id_union \
      "${preferred_job_id}" "${on_authenticated_job_ids_csv:--}"
  )" || on_cancel_candidate_job_ids_csv=""
  seen=""
  for job_id in \
      ${off_cancel_candidate_job_ids_csv//,/ } \
      ${on_cancel_candidate_job_ids_csv//,/ }; do
    if [[ ",${seen}," == *",${job_id},"* ]]; then
      continue
    fi
    seen="${seen:+${seen},}${job_id}"
    candidate_job_ids_csv="${candidate_job_ids_csv:+${candidate_job_ids_csv},}${job_id}"
  done
  pre_cancel_authenticated_job_ids_csv="${candidate_job_ids_csv:--}"
  pre_cancel_job_ids_csv="${all_candidate_job_ids_csv:--}"
  pre_cancel_unresolved_job_ids_csv=""
  for job_id in ${all_candidate_job_ids_csv//,/ }; do
    if [[ ",${candidate_job_ids_csv}," != *",${job_id},"* ]]; then
      pre_cancel_unresolved_job_ids_csv="${pre_cancel_unresolved_job_ids_csv:+${pre_cancel_unresolved_job_ids_csv},}${job_id}"
    fi
  done
  pre_cancel_unresolved_job_ids_csv="${pre_cancel_unresolved_job_ids_csv:--}"
  if [[ "${pre_cancel_unresolved_job_ids_csv}" != "-" ]]; then
    failure_outcome="rollback-unconfirmed"
    rollback_confirmed="false"
  fi

  cancel_job_ids_csv="-"
  if [[ -n "${candidate_job_ids_csv}" ]]; then
    cancel_job_ids_csv="${candidate_job_ids_csv}"
    if cancel_output="$(
      strict_pair_scheduler_client \
        "${STRICT_PAIR_TOOL_SCANCEL}" "${cancel_job_ids_csv}" 2>/dev/null
    )"; then
      cancel_status=0
    else
      cancel_status=$?
    fi
    cancel_output_sha256="$(strict_pair_sha256_text "${cancel_output}")"
    post_cancel_queries="[]"
    query_index=0
    if strict_pair_capture_raw_candidate_query \
        cancel "${cancel_job_ids_csv}" \
        "${off_cancel_candidate_job_ids_csv:--}" \
        "${on_cancel_candidate_job_ids_csv:--}" - \
        "post-${query_index}" post_cancel_queries; then
      off_post_cancel_authenticated_job_ids_csv="${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED/-/}"
      on_post_cancel_authenticated_job_ids_csv="${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED/-/}"
    fi
    ((query_index += 1))
    for job_id in ${cancel_job_ids_csv//,/ }; do
      if [[ ",${off_post_cancel_authenticated_job_ids_csv},${on_post_cancel_authenticated_job_ids_csv}," == \
            *",${job_id},"* ]]; then
        continue
      fi
      query_off_job_ids_csv="-"
      query_on_job_ids_csv="-"
      [[ ",${off_cancel_candidate_job_ids_csv}," == *",${job_id},"* ]] && \
        query_off_job_ids_csv="${job_id}"
      [[ ",${on_cancel_candidate_job_ids_csv}," == *",${job_id},"* ]] && \
        query_on_job_ids_csv="${job_id}"
      if strict_pair_capture_raw_candidate_query \
          cancel "${job_id}" \
          "${query_off_job_ids_csv}" "${query_on_job_ids_csv}" - \
          "post-${query_index}" post_cancel_queries; then
        off_post_cancel_authenticated_job_ids_csv="$(
          strict_pair_canonical_job_id_union - \
            "${off_post_cancel_authenticated_job_ids_csv:--}" \
            "${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED}"
        )" || off_post_cancel_authenticated_job_ids_csv=""
        on_post_cancel_authenticated_job_ids_csv="$(
          strict_pair_canonical_job_id_union - \
            "${on_post_cancel_authenticated_job_ids_csv:--}" \
            "${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED}"
        )" || on_post_cancel_authenticated_job_ids_csv=""
      fi
      ((query_index += 1))
    done
    seen=""
    cancel_query_authenticated_job_ids_csv=""
    for job_id in \
        ${off_cancel_candidate_job_ids_csv//,/ } \
        ${on_cancel_candidate_job_ids_csv//,/ }; do
      if [[ ",${off_post_cancel_authenticated_job_ids_csv},${on_post_cancel_authenticated_job_ids_csv}," != \
            *",${job_id},"* || ",${seen}," == *",${job_id},"* ]]; then
        continue
      fi
      seen="${seen:+${seen},}${job_id}"
      cancel_query_authenticated_job_ids_csv="${cancel_query_authenticated_job_ids_csv:+${cancel_query_authenticated_job_ids_csv},}${job_id}"
    done
    cancel_query_authenticated_job_ids_csv="${cancel_query_authenticated_job_ids_csv:--}"
    cancel_query_unresolved_job_ids_csv=""
    for job_id in ${cancel_job_ids_csv//,/ }; do
      if [[ ",${cancel_query_authenticated_job_ids_csv}," != *",${job_id},"* ]]; then
        cancel_query_unresolved_job_ids_csv="${cancel_query_unresolved_job_ids_csv:+${cancel_query_unresolved_job_ids_csv},}${job_id}"
      fi
    done
    cancel_query_unresolved_job_ids_csv="${cancel_query_unresolved_job_ids_csv:--}"
    cancel_query_status=0
    [[ "${cancel_query_unresolved_job_ids_csv}" == "-" ]] || cancel_query_status=1
  else
    if [[ "${recovery_scan_status}" != "0" || \
          "${recovery_scan_normalization_status}" != "0" || \
          "${recovery_scan_unterminated_final_line}" != "false" || \
          "${recovery_scan_identity_match_counts}" != '{"off":0,"on":0}' || \
          -n "${off_record_job_id}" || -n "${on_record_job_id}" || \
          "${off_accepted_id_record_sha256}" != \
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" || \
          "${on_accepted_id_record_sha256}" != \
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ]]; then
      failure_outcome="rollback-unconfirmed"
      rollback_confirmed="false"
    fi
  fi
  if [[ "${cancel_job_ids_csv}" != "-" ]] && \
     { (( cancel_status != 0 || cancel_query_status != 0 )) || \
       [[ "${cancel_query_authenticated_job_ids_csv}" != \
            "${cancel_job_ids_csv}" || \
          "${cancel_query_unresolved_job_ids_csv}" != "-" ]]; }; then
    failure_outcome="rollback-unconfirmed"
    rollback_confirmed="false"
  fi
  if [[ "${recovery_scan_status}" != "0" || \
        "${recovery_scan_normalization_status}" != "0" || \
        "${recovery_scan_unterminated_final_line}" != "false" || \
        "${recovery_scan_parsed_record_count}" != \
          "${recovery_scan_line_count}" ]]; then
    failure_outcome="rollback-unconfirmed"
    rollback_confirmed="false"
  fi
  if [[ -n "${recovery_scan_matched_job_ids_csv}" && \
        "${cancel_job_ids_csv}" != "${recovery_scan_matched_job_ids_csv}" ]]; then
    failure_outcome="rollback-unconfirmed"
    rollback_confirmed="false"
  fi
  if [[ "${STRICT_PAIR_CANDIDATE_QUERY_EVIDENCE_FAILED}" != "0" ]]; then
    strict_pair_error "strict pair raw candidate-query evidence failed before terminal publication."
    return
  fi
  if ! strict_pair_cleanup_recovery_scan || \
     [[ "${recovery_scan_status}" != "-" && \
        "${recovery_scan_securely_unlinked}" != "true" ]]; then
    strict_pair_error "strict pair scheduler recovery evidence could not be securely removed."
    return
  fi
  if ! receipt_sha256="$(
    strict_pair_publish_submission_receipt \
      "${failure_outcome}" "${stage}" "${rollback_confirmed}"
  )"; then
    if strict_pair_adopt_and_emit_committed_receipt; then
      strict_pair_error "${message}"
      return
    fi
    strict_pair_error "strict pair submission failed and its sealed receipt could not be published."
    return
  fi
  strict_pair_emit_result "${failure_outcome}" "${stage}" "${receipt_sha256}"
  strict_pair_error "${message}"
}

strict_pair_fail_if_deferred_signal() {
  local boundary="$1"

  if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" != "0" ]]; then
    strict_pair_fail_submission unexpected_exit \
      "strict pair launcher received a signal ${boundary}."
  fi
}

STRICT_PAIR_TERMINAL_RECEIPT_PUBLISHED=0
STRICT_PAIR_SUBMISSION_GUARD_ACTIVE=0
STRICT_PAIR_LIFECYCLE_CLEANUP_ACTIVE=0
STRICT_PAIR_DEFERRED_SIGNAL_STATUS=0
off_identity_authenticated=false
on_identity_authenticated=false
off_child_quiesced=0
on_child_quiesced=0
strict_pair_defer_parent_signal() {
  local signal_status="$1"

  if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" == "0" ]]; then
    STRICT_PAIR_DEFERRED_SIGNAL_STATUS="${signal_status}"
  fi
}

strict_pair_wait_for_arm_quiescence() {
  local arm="$1"
  local child_pid
  local child_status=0

  if [[ "${arm}" == "off" ]]; then
    child_pid="${off_pid:-}"
  else
    child_pid="${on_pid:-}"
  fi
  if [[ ! "${child_pid}" =~ ^[0-9]+$ ]]; then
    return
  fi
  while true; do
    if wait "${child_pid}"; then
      child_status=0
    else
      child_status=$?
    fi
    if ! kill -0 "${child_pid}" 2>/dev/null; then
      break
    fi
  done
  if [[ "${arm}" == "off" ]]; then
    off_status="${child_status}"
    off_child_quiesced=1
  else
    on_status="${child_status}"
    on_child_quiesced=1
  fi
}

strict_pair_lifecycle_cleanup() {
  local original_status=$?

  trap - EXIT
  trap 'strict_pair_defer_parent_signal 130' INT
  trap 'strict_pair_defer_parent_signal 143' TERM
  set +e
  if [[ "${STRICT_PAIR_SUBMISSION_GUARD_ACTIVE}" == "1" && \
        "${STRICT_PAIR_TERMINAL_RECEIPT_PUBLISHED}" != "1" && \
        "${STRICT_PAIR_LIFECYCLE_CLEANUP_ACTIVE}" != "1" ]]; then
    STRICT_PAIR_LIFECYCLE_CLEANUP_ACTIVE=1
    if strict_pair_adopt_and_emit_committed_receipt; then
      trap - INT TERM
      cleanup_pair_temporary_files
      exit "${original_status}"
    fi
    [[ "${off_child_quiesced}" == "1" ]] || \
      strict_pair_wait_for_arm_quiescence off
    [[ "${on_child_quiesced}" == "1" ]] || \
      strict_pair_wait_for_arm_quiescence on
    strict_pair_capture_accepted_id_records
    strict_pair_recover_missing_accepted_ids
    strict_pair_fail_submission unexpected_exit \
      "strict pair launcher exited before publishing its terminal submission receipt."
    [[ "${original_status}" != "0" ]] || original_status=2
  fi
  trap - INT TERM
  cleanup_pair_temporary_files
  exit "${original_status}"
}
trap strict_pair_lifecycle_cleanup EXIT
trap 'strict_pair_defer_parent_signal 130' INT
trap 'strict_pair_defer_parent_signal 143' TERM

off_pid=""
on_pid=""
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "submit" ]]; then
  prepare_strict_pair_accepted_id_records
fi
: > "${off_stdout}"
: > "${off_stderr}"
: > "${on_stdout}"
: > "${on_stderr}"
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "submit" ]]; then
  STRICT_PAIR_SUBMISSION_GUARD_ACTIVE=1
fi
if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" == "0" ]]; then
  run_arm_phase off 0 "${off_stdout}" "${off_stderr}" &
  off_pid=$!
fi
if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" == "0" ]]; then
  run_arm_phase on 0 "${on_stdout}" "${on_stderr}" &
  on_pid=$!
fi

strict_pair_wait_for_arm_quiescence off
strict_pair_wait_for_arm_quiescence on
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "submit" ]]; then
  strict_pair_capture_accepted_id_records
  strict_pair_recover_missing_accepted_ids
else
  close_strict_pair_accepted_id_fds
fi
"${STRICT_PAIR_TOOL_CAT}" -- "${off_stdout}" "${on_stdout}"
"${STRICT_PAIR_TOOL_CAT}" -- "${off_stderr}" "${on_stderr}" >&2

if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "dry-run" ]]; then
  if (( off_status != 0 || on_status != 0 )); then
    strict_pair_error "strict pair dry-run failed: off_status=${off_status} on_status=${on_status}"
  fi
  echo "STRICT_PAIR_DRY_RUN_VALIDATED pair_id=${PAIR_ID} off_status=0 on_status=0 submitted=0"
else
  off_held_record="$(extract_strict_held_record off "${off_stdout}")" || \
    off_held_record=""
  on_held_record="$(extract_strict_held_record on "${on_stdout}")" || \
    on_held_record=""
  if [[ -n "${off_held_record}" ]]; then
    read -r off_marker_job_id off_marker_job_id_sha256 \
      off_marker_record_sha256 off_sbatch_status off_relay_status \
      off_writer_drained off_started_unix_ns off_drained_unix_ns \
      <<< "${off_held_record}"
  fi
  if [[ -n "${on_held_record}" ]]; then
    read -r on_marker_job_id on_marker_job_id_sha256 \
      on_marker_record_sha256 on_sbatch_status on_relay_status \
      on_writer_drained on_started_unix_ns on_drained_unix_ns \
      <<< "${on_held_record}"
  fi
  strict_pair_fail_if_deferred_signal "while submissions were in flight"
  if (( off_status != 0 || on_status != 0 )); then
    strict_pair_fail_submission arm_submit \
      "strict held submission failed: off_status=${off_status} on_status=${on_status}"
  fi
  if [[ -z "${off_job_id}" || -z "${on_job_id}" ]]; then
    strict_pair_fail_submission job_id_validation \
      "strict submit did not return exactly one numeric held job ID per arm."
  fi
  if [[ "${off_marker_job_id:-}" != "${off_job_id}" || \
        "${off_marker_job_id_sha256:-}" != "${off_held_output_sha256}" || \
        "${off_marker_record_sha256:-}" != "${off_accepted_id_record_sha256}" || \
        "${on_marker_job_id:-}" != "${on_job_id}" || \
        "${on_marker_job_id_sha256:-}" != "${on_held_output_sha256}" || \
        "${on_marker_record_sha256:-}" != "${on_accepted_id_record_sha256}" ]]; then
    strict_pair_fail_submission job_id_validation \
      "strict held markers did not exactly bind the durable accepted-ID records."
  fi
  if [[ "${off_job_id}" == "${on_job_id}" ]]; then
    strict_pair_fail_submission job_id_validation \
      "strict OFF and ON held submissions returned the same job ID."
  fi
  if [[ "${recovery_scan_status}" != "0" || \
        "${recovery_scan_normalization_status}" != "0" || \
        "${recovery_scan_unterminated_final_line}" != "false" || \
        "${recovery_scan_parsed_record_count}" != \
          "${recovery_scan_line_count}" || \
        "${recovery_scan_identity_match_counts}" != '{"off":1,"on":1}' ]]; then
    strict_pair_fail_submission pre_release_validation \
      "strict scheduler recovery scan was incomplete or ambiguous before release."
  fi
  EXPECTED_PAIR_MANIFEST_SHA256="${STRICT_PAIR_MANIFEST_SHA256}"
  if ! strict_pair_verify_manifest || \
     ! strict_pair_verify_submission_contract || \
     ! strict_pair_verify_slurm_export_before_sbatch \
        "${STRICT_PAIR_OFF_SLURM_EXPORT}" \
        "${STRICT_PAIR_OFF_SLURM_EXPORT_SHA256}" || \
     ! strict_pair_verify_slurm_export_before_sbatch \
        "${STRICT_PAIR_ON_SLURM_EXPORT}" \
        "${STRICT_PAIR_ON_SLURM_EXPORT_SHA256}"; then
    strict_pair_fail_submission pre_release_validation \
      "strict pair anchors changed after held submission and before release."
  fi
  strict_pair_fail_if_deferred_signal "before the pre-release scheduler query"
  scheduler_job_ids="${off_job_id},${on_job_id}"
  if ! strict_pair_capture_raw_lifecycle_query \
      pre "${scheduler_job_ids}" "${off_job_id}" "${on_job_id}" \
      pre-release pre_release_query || \
     [[ "${STRICT_PAIR_LAST_QUERY_AUTHENTICATED}" != \
          "${scheduler_job_ids}" || \
        "${STRICT_PAIR_LAST_QUERY_UNRESOLVED}" != "-" || \
        "${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED}" != "${off_job_id}" || \
        "${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED}" != "${on_job_id}" ]]; then
    strict_pair_fail_submission pre_release_validation \
      "strict pair jobs were not both authenticated as held before release."
  fi
  strict_pair_fail_if_deferred_signal "during the pre-release scheduler query"
  if release_output="$(
    strict_pair_scheduler_client \
      "${STRICT_PAIR_TOOL_SCONTROL}" release "${scheduler_job_ids}" 2>/dev/null
  )"; then
    release_status=0
  else
    release_status=$?
  fi
  release_output_sha256="$(strict_pair_sha256_text "${release_output}")"
  if (( release_status != 0 )); then
    strict_pair_fail_submission release \
      "single-RPC strict pair release failed."
  fi
  strict_pair_fail_if_deferred_signal "during the single-RPC pair release"
  if ! strict_pair_capture_raw_lifecycle_query \
      post "${scheduler_job_ids}" "${off_job_id}" "${on_job_id}" \
      post-release post_release_query || \
     [[ "${STRICT_PAIR_LAST_QUERY_AUTHENTICATED}" != \
          "${scheduler_job_ids}" || \
        "${STRICT_PAIR_LAST_QUERY_UNRESOLVED}" != "-" || \
        "${STRICT_PAIR_LAST_QUERY_OFF_AUTHENTICATED}" != "${off_job_id}" || \
        "${STRICT_PAIR_LAST_QUERY_ON_AUTHENTICATED}" != "${on_job_id}" ]]; then
    strict_pair_fail_submission post_release_validation \
      "strict pair scheduler state was asymmetric or invalid after release."
  fi
  strict_pair_fail_if_deferred_signal "during the post-release scheduler query"
  off_identity_authenticated=true
  on_identity_authenticated=true
  if ! strict_pair_cleanup_recovery_scan || \
     [[ "${recovery_scan_securely_unlinked}" != "true" ]]; then
    strict_pair_fail_submission receipt_publication \
      "strict pair recovery evidence could not be removed before terminal publication."
  fi
  if ! submission_receipt_sha256="$(
    strict_pair_publish_submission_receipt released complete null
  )"; then
    if ! strict_pair_adopt_and_emit_committed_receipt; then
      strict_pair_fail_submission receipt_publication \
        "strict pair released jobs but could not publish the terminal receipt."
    fi
  else
    strict_pair_emit_result released complete "${submission_receipt_sha256}"
  fi
fi
