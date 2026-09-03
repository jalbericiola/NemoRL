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

# Direct use requires a caller that has already excluded dynamic-loader
# injection. The authoritative parent invokes this arm through an authenticated
# privileged Bash with an explicit clean environment.
if [[ "$-" != *p* ]]; then
  echo "ERROR: nano35_single_env_pair.sh must be executed directly through its privileged Bash shebang." >&2
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

_strict_pair_expected_source="${DEPLOYMENT_ROOT}/runnable/NemoRL/examples/nemo_gym/nemotron-3.5-nano/nano35_single_env_pair.sh"
if [[ "${BASH_SOURCE[0]}" != "${_strict_pair_expected_source}" || \
      -L "${BASH_SOURCE[0]}" || ! -f "${BASH_SOURCE[0]}" || \
      ! -x "${BASH_SOURCE[0]}" || -w "${BASH_SOURCE[0]}" ]]; then
  echo "ERROR: nano35_single_env_pair.sh must be the sealed deployment entrypoint at ${_strict_pair_expected_source}." >&2
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
        echo "ERROR: duplicate arm-wrapper runnable-manifest entry." >&2
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
  echo "ERROR: bootstrap sha256sum failed for arm wrapper." >&2
  exit 2
fi
_strict_pair_actual_sha256="${_strict_pair_hash_output%% *}"
if [[ -z "${_strict_pair_source_manifest_sha256}" || \
      "${_strict_pair_actual_sha256}" != "${_strict_pair_source_manifest_sha256}" ]]; then
  echo "ERROR: arm wrapper is absent or drifted in the authenticated runnable manifest." >&2
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

LAUNCHER_REL="examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"

# shellcheck source=strict_pair_contract.sh
source "${_strict_pair_contract}"
strict_pair_select_environment
CONFIG_PATH="${STRICT_PAIR_CONFIG_RELATIVE}"
strict_pair_load_runtime_tools
SCRIPT_DIR="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${SCRIPT_DIR}")"
PROJECT_ROOT="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${PROJECT_ROOT}")"
unset _strict_pair_actual_sha256 _strict_pair_contract \
  _strict_pair_contract_manifest_sha256 _strict_pair_environment_name \
  _strict_pair_expected_source _strict_pair_hash_output \
  _strict_pair_guard_path _strict_pair_manifest_line \
  _strict_pair_manifest_path _strict_pair_manifest_sha256 \
  _strict_pair_nemo_manifest _strict_pair_source_manifest_sha256

usage() {
  echo "Usage: STRICT_PAIR_ENVIRONMENT=<reasoning_gym|citation|freeform> EXPECTED_PAIR_MANIFEST_SHA256=<sha256> ${BASH_SOURCE[0]} <off|on>" >&2
  echo "       Direct launches are forbidden; use launch_pair.sh." >&2
}

if (( $# < 1 )); then
  echo "ERROR: shared-prefix arm is required." >&2
  usage
  exit 2
fi

ARM="$1"
shift
case "${ARM}" in
  off)
    SHARED_PREFIX_MODE="observe"
    ;;
  on)
    SHARED_PREFIX_MODE="train"
    ;;
  *)
    echo "ERROR: shared-prefix arm must be exactly 'off' or 'on'; got '${ARM}'." >&2
    usage
    exit 2
    ;;
esac

# Normalize Hydra's add/force-add/delete prefixes before matching. Every
# scientific config namespace is closed: accepting a new leaf by default would
# let the two arms drift as schemas evolve.
hydra_key_is_protected() {
  local key="$1"
  case "${key}" in
    checkpointing|checkpointing.*|cluster|cluster.*|data|data.*|\
    data_plane|data_plane.*|env|env.*|grpo|grpo.*|logger|logger.*|\
    loss_fn|loss_fn.*|on_policy_distillation|on_policy_distillation.*|\
    policy|policy.*|reward_penalties|reward_penalties.*|\
    async_rl|async_rl.*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

for override in "$@"; do
  if [[ "${override}" != *=* && "${override}" != \~* ]]; then
    echo "ERROR: unexpected extra arm or malformed Hydra override: '${override}'." >&2
    usage
    exit 2
  fi

  override_key="${override%%=*}"
  override_key="${override_key#++}"
  override_key="${override_key#+}"
  override_key="${override_key#\~}"
  if [[ ! "${override_key}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: malformed or unsupported Hydra override key: '${override_key}'." >&2
    exit 2
  fi
  if hydra_key_is_protected "${override_key}"; then
    echo "ERROR: strict single-env pair forbids overriding protected Hydra key: ${override_key}" >&2
    exit 2
  fi
  echo "ERROR: strict single-env pair does not accept unsupported Hydra key: ${override_key}" >&2
  exit 2
done
unset override override_key

if [[ -z "${STRICT_PAIR_LAUNCH_MODE+x}" ]]; then
  strict_pair_error "Direct arm launch is forbidden; use launch_pair.sh."
fi

: "${PAIR_ID:?PAIR_ID is required to correlate the OFF/ON arms}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required; strict pair runs must log to W&B}"
: "${TRAIN_PATH:?TRAIN_PATH is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${HF_HOME:?HF_HOME is required; the strict pair derives disjoint arm-local Hugging Face cache roots from this parent}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
: "${STRICT_PAIR_JOB_WRAPPER:?STRICT_PAIR_JOB_WRAPPER is required}"
: "${STRICT_PAIR_SLURM_CONF:?STRICT_PAIR_SLURM_CONF is required}"
: "${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256:?EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256 is required}"
: "${STRICT_PAIR_SUBMISSION_NONCE:?STRICT_PAIR_SUBMISSION_NONCE is required}"
: "${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256:?STRICT_PAIR_SUBMISSION_CONTRACT_SHA256 is required}"
: "${EXPECTED_NEMO_HEAD:?EXPECTED_NEMO_HEAD is required}"
: "${EXPECTED_GYM_GITLINK_COMMIT:?EXPECTED_GYM_GITLINK_COMMIT is required}"
: "${EXPECTED_BRIDGE_HEAD:?EXPECTED_BRIDGE_HEAD is required}"
: "${EXPECTED_BRIDGE_TREE:?EXPECTED_BRIDGE_TREE is required}"
: "${EXPECTED_MCORE_HEAD:?EXPECTED_MCORE_HEAD is required}"
: "${EXPECTED_MCORE_TREE:?EXPECTED_MCORE_TREE is required}"

STRICT_PAIR_EXPORT_PREPARE_ONLY="${STRICT_PAIR_EXPORT_PREPARE_ONLY:-0}"
case "${STRICT_PAIR_EXPORT_PREPARE_ONLY}" in
  0)
    : "${EXPECTED_PAIR_MANIFEST_SHA256:?Direct arm launch is forbidden; use launch_pair.sh}"
    strict_pair_require_digest EXPECTED_PAIR_MANIFEST_SHA256
    strict_pair_require_digest EXPECTED_STRICT_PAIR_OFF_SLURM_EXPORT_SHA256
    strict_pair_require_digest EXPECTED_STRICT_PAIR_ON_SLURM_EXPORT_SHA256
    STRICT_PAIR_OFF_SLURM_EXPORT_SHA256="${EXPECTED_STRICT_PAIR_OFF_SLURM_EXPORT_SHA256}"
    STRICT_PAIR_ON_SLURM_EXPORT_SHA256="${EXPECTED_STRICT_PAIR_ON_SLURM_EXPORT_SHA256}"
    ;;
  1)
    if [[ -n "${EXPECTED_PAIR_MANIFEST_SHA256+x}" || \
          -n "${EXPECTED_STRICT_PAIR_OFF_SLURM_EXPORT_SHA256+x}" || \
          -n "${EXPECTED_STRICT_PAIR_ON_SLURM_EXPORT_SHA256+x}" ]]; then
      strict_pair_error "export preparation must occur before all Pair-manifest anchors exist."
    fi
    ;;
  *)
    strict_pair_error "STRICT_PAIR_EXPORT_PREPARE_ONLY must be parent-owned 0 or 1."
    ;;
esac

if [[ ! "${PAIR_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || (( ${#PAIR_ID} > 64 )); then
  strict_pair_error "PAIR_ID must be a 1-64 character filesystem-safe identifier."
fi
case "${STRICT_PAIR_LAUNCH_MODE}" in
  dry-run|submit) ;;
  *)
    strict_pair_error "STRICT_PAIR_LAUNCH_MODE must be dry-run or submit."
    ;;
esac

if [[ "${WANDB_API_KEY}" == *[[:space:]]* || ${#WANDB_API_KEY} -lt 20 ]]; then
  strict_pair_error "WANDB_API_KEY must be one non-whitespace secret of at least 20 characters."
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
done < <(compgen -A variable TRITON_AUTOTUNE_BLOCK || true)

if [[ -n "${WANDB_DISABLED+x}" ]]; then
  strict_pair_error "WANDB_DISABLED must be unset; strict pair evidence requires online W&B."
fi
if [[ -n "${WANDB_MODE+x}" && "${WANDB_MODE}" != "online" ]]; then
  strict_pair_error "WANDB_MODE must be unset or 'online'; got '${WANDB_MODE}'."
fi
if [[ -n "${WANDB_RUN_ID+x}" ]]; then
  strict_pair_error "WANDB_RUN_ID must be unset; each arm derives a fresh isolated W&B identity."
fi
if [[ -n "${USE_SNAPSHOT+x}" && "${USE_SNAPSHOT}" != "1" ]]; then
  strict_pair_error "USE_SNAPSHOT must be unset or 1; live-tree/container fallback is forbidden."
fi
if [[ -n "${CODE_SNAPSHOT_DIRNAME+x}" ]]; then
  strict_pair_error "CODE_SNAPSHOT_DIRNAME must be unset; the strict pair owns its snapshot parent."
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

strict_pair_load_arm_contract "${PROJECT_ROOT}" "${SCRIPT_DIR}"
strict_pair_verify_submission_contract
if [[ "${STRICT_PAIR_EXPORT_PREPARE_ONLY}" == "0" ]]; then
  strict_pair_verify_manifest
fi

ARM_EXP_NAME="${ARM}-${STRICT_PAIR_ENVIRONMENT}-${PAIR_ID}"
ARM_WANDB_RUN_GROUP="${STRICT_PAIR_ENVIRONMENT}-${PAIR_ID}"
ARM_WANDB_RUN_ID="$(
  strict_pair_sha256_text \
    "nemo-rl-strict-wandb-v1:${STRICT_PAIR_ENVIRONMENT}:${PAIR_ID}:${ARM}"
)"
ARM_RESULTS_DIR="${RESULTS_DIR%/}/${ARM}"
ARM_PERSISTENT_CACHE="${PERSISTENT_CACHE%/}/${ARM}"
ARM_CACHE_READ_DIR="${ARM_PERSISTENT_CACHE}/cache_read"
ARM_HF_HOME="${HF_HOME%/}/${ARM}"
"${STRICT_PAIR_TOOL_MKDIR}" -p -- \
  "${ARM_RESULTS_DIR}" "${ARM_PERSISTENT_CACHE}" \
  "${ARM_CACHE_READ_DIR}" "${ARM_HF_HOME}"
"${STRICT_PAIR_TOOL_CHMOD}" 700 \
  "${ARM_RESULTS_DIR}" "${ARM_PERSISTENT_CACHE}" \
  "${ARM_CACHE_READ_DIR}" "${ARM_HF_HOME}"
strict_pair_require_canonical_dir "${ARM_RESULTS_DIR}" "arm RESULTS_DIR"
strict_pair_require_mode "${ARM_RESULTS_DIR}" "700" "arm RESULTS_DIR"
strict_pair_require_canonical_dir \
  "${ARM_PERSISTENT_CACHE}" "arm PERSISTENT_CACHE"
strict_pair_require_mode \
  "${ARM_PERSISTENT_CACHE}" "700" "arm PERSISTENT_CACHE"
strict_pair_require_empty_private_dir \
  "${ARM_CACHE_READ_DIR}" "arm cache_read input"
strict_pair_require_empty_private_dir "${ARM_HF_HOME}" "arm HF_HOME"
if [[ "${ARM}" == "off" ]]; then
  ARM_SNAPSHOT_DIR="${STRICT_PAIR_OFF_SNAPSHOT}"
  STRICT_PAIR_SNAPSHOT_MANIFEST_SHA256="${STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256}"
  STRICT_PAIR_SLURM_EXPORT_FILE="${STRICT_PAIR_OFF_SLURM_EXPORT}"
  STRICT_PAIR_EXPECTED_SLURM_EXPORT_SHA256="${STRICT_PAIR_OFF_SLURM_EXPORT_SHA256:-}"
else
  ARM_SNAPSHOT_DIR="${STRICT_PAIR_ON_SNAPSHOT}"
  STRICT_PAIR_SNAPSHOT_MANIFEST_SHA256="${STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256}"
  STRICT_PAIR_SLURM_EXPORT_FILE="${STRICT_PAIR_ON_SLURM_EXPORT}"
  STRICT_PAIR_EXPECTED_SLURM_EXPORT_SHA256="${STRICT_PAIR_ON_SLURM_EXPORT_SHA256:-}"
fi
strict_pair_verify_snapshot "${ARM}" "${ARM_SNAPSHOT_DIR}"
LAUNCHER="${ARM_SNAPSHOT_DIR}/${LAUNCHER_REL}"
EXPECTED_DETERMINISM_ATTESTATION="SHARED_PREFIX_DETERMINISM_ATTESTED mode=${SHARED_PREFIX_MODE} env_controls=5 triton_autotune=absent model_overrides=3 torch_deterministic=true mcore_backward=true total_controls=9"
EXPECTED_DETERMINISM_ATTESTATION_SHA256="$(
  strict_pair_sha256_text "${EXPECTED_DETERMINISM_ATTESTATION}"
)"

# Recompute mutable large-artifact anchors at the final source-launcher
# boundary. The batch job still needs the external deployment package to check
# these immediately before the driver and after successful completion.
MODEL_PRE_SHA256="$(strict_pair_model_tree_sha256_v1 "${MODEL_PATH}")"
CONTAINER_PRE_SHA256="$(strict_pair_sha256_file "${CONTAINER}")"
SANDBOX_CONTAINER_PRE_SHA256="$(strict_pair_sha256_file "${SANDBOX_CONTAINER}")"
FIXTURE_PRE_SHA256="$(strict_pair_sha256_file "${TRAIN_PATH}")"
if [[ "${FIXTURE_PRE_SHA256}" != "${STRICT_PAIR_FIXTURE_SHA256}" || \
      "${MODEL_PRE_SHA256}" != "${STRICT_PAIR_MODEL_TREE_SHA256}" || \
      "${CONTAINER_PRE_SHA256}" != "${STRICT_PAIR_CONTAINER_SHA256}" || \
      "${SANDBOX_CONTAINER_PRE_SHA256}" != "${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" ]]; then
  strict_pair_error "fixture/model/container changed after parent authentication."
fi

echo "STRICT_PAIR_DEPLOYMENT_READY=${EXPECTED_DEPLOYMENT_READY} ready_file_sha256=${EXPECTED_DEPLOYMENT_READY_FILE_SHA256} nemo_manifest_sha256=${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256} bridge_manifest_sha256=${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256} mcore_manifest_sha256=${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256} root=${DEPLOYMENT_ROOT}"
if [[ "${STRICT_PAIR_EXPORT_PREPARE_ONLY}" == "0" ]]; then
  echo "STRICT_PAIR_MANIFEST_SHA256=${EXPECTED_PAIR_MANIFEST_SHA256} path=${STRICT_PAIR_MANIFEST_PATH}"
fi
echo "STRICT_PAIR_FIXTURE_PRE_SHA256=${FIXTURE_PRE_SHA256} rows=${STRICT_PAIR_FIXTURE_ROWS} path=${TRAIN_PATH}"
echo "STRICT_PAIR_MODEL_PRE_SHA256=${MODEL_PRE_SHA256} path=${MODEL_PATH}"
echo "STRICT_PAIR_CONTAINER_PRE_SHA256=${CONTAINER_PRE_SHA256} path=${CONTAINER}"
echo "STRICT_PAIR_RUN pair_id=${PAIR_ID} environment=${STRICT_PAIR_ENVIRONMENT} config=${CONFIG_PATH} arm=${ARM} exp_name=${ARM_EXP_NAME} results=${ARM_RESULTS_DIR} cache=${ARM_PERSISTENT_CACHE}"
echo "STRICT_PAIR_RUNTIME_ATTESTATION_REQUIRED=1 arm=${ARM} mode=${SHARED_PREFIX_MODE} count=4 sha256_ascii_no_newline=${EXPECTED_DETERMINISM_ATTESTATION_SHA256}"
echo "STRICT_PAIR_JOB_INTERVAL_GATE_REQUIRED=1 implementation=${STRICT_PAIR_JOB_WRAPPER} sha256=${STRICT_PAIR_JOB_WRAPPER_SHA256} pre=driver-exec post=driver-success artifacts=pair-manifest,fixture,model,container,sandbox-container,snapshot"

# The public launcher keeps its existing swe/rlvr modes. This wrapper selects
# the judge-free allocation path while supplying the sealed SingleController
# recipe and exact colocated four-rank shape.
DRY_RUN_ENV=()
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "dry-run" && \
      "${STRICT_PAIR_EXPORT_PREPARE_ONLY}" == "0" ]]; then
  DRY_RUN_ENV=(DRY_RUN=1)
fi
PAIR_MANIFEST_ENV=()
ACCEPTED_ID_ENV=()
if [[ "${STRICT_PAIR_EXPORT_PREPARE_ONLY}" == "0" ]]; then
  PAIR_MANIFEST_ENV=(
    "STRICT_PAIR_MANIFEST_PATH=${STRICT_PAIR_MANIFEST_PATH}"
    "EXPECTED_PAIR_MANIFEST_SHA256=${EXPECTED_PAIR_MANIFEST_SHA256}"
  )
  if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "submit" ]]; then
    : "${STRICT_PAIR_ACCEPTED_ID_FD:?parent-owned accepted-ID descriptor is required}"
    : "${STRICT_PAIR_ACCEPTED_ID_RECORD:?parent-owned accepted-ID path is required}"
    if [[ ! "${STRICT_PAIR_ACCEPTED_ID_FD}" =~ ^[0-9]+$ ]]; then
      strict_pair_error "parent-owned accepted-ID descriptor must be decimal."
    fi
    if [[ "${ARM}" == "off" ]]; then
      EXPECTED_ACCEPTED_ID_RECORD="${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}"
    else
      EXPECTED_ACCEPTED_ID_RECORD="${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}"
    fi
    if [[ "${STRICT_PAIR_ACCEPTED_ID_RECORD}" != \
          "${EXPECTED_ACCEPTED_ID_RECORD}" ]]; then
      strict_pair_error "parent-owned accepted-ID path differs from the Pair contract."
    fi
    ACCEPTED_ID_ENV=(
      "STRICT_PAIR_ACCEPTED_ID_FD=${STRICT_PAIR_ACCEPTED_ID_FD}"
      "STRICT_PAIR_ACCEPTED_ID_RECORD=${STRICT_PAIR_ACCEPTED_ID_RECORD}"
    )
  fi
fi
exec "${STRICT_PAIR_TOOL_ENV}" -i \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  LC_ALL=C \
  ${DRY_RUN_ENV[@]+"${DRY_RUN_ENV[@]}"} \
  PAIR_ID="${PAIR_ID}" \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  TRAIN_ENTRYPOINT="./examples/run_grpo_single_controller.py" \
  EXP_NAME="${ARM_EXP_NAME}" \
  WANDB_NAME="${ARM_EXP_NAME}" \
  WANDB_RESUME=never \
  WANDB_RUN_ID="${ARM_WANDB_RUN_ID}" \
  RESULTS_DIR="${ARM_RESULTS_DIR}" \
  PERSISTENT_CACHE="${ARM_PERSISTENT_CACHE}" \
  HF_HOME="${ARM_HF_HOME}" \
  HF_HUB_CACHE="${ARM_HF_HOME}/hub" \
  HF_DATASETS_CACHE="${ARM_HF_HOME}/hub" \
  TRAIN_PATH="${TRAIN_PATH}" \
  VAL_PATH="${TRAIN_PATH}" \
  MODEL_PATH="${MODEL_PATH}" \
  CONTAINER="${CONTAINER}" \
  SANDBOX_CONTAINER="${SANDBOX_CONTAINER}" \
  DEPLOYMENT_ROOT="${DEPLOYMENT_ROOT}" \
  STRICT_PAIR_JOB_WRAPPER="${STRICT_PAIR_JOB_WRAPPER}" \
  EXPECTED_DEPLOYMENT_READY="${EXPECTED_DEPLOYMENT_READY}" \
  EXPECTED_DEPLOYMENT_READY_FILE_SHA256="${EXPECTED_DEPLOYMENT_READY_FILE_SHA256}" \
  EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256="${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" \
  EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256="${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256}" \
  EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256="${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256}" \
  EXPECTED_NEMO_HEAD="${EXPECTED_NEMO_HEAD}" \
  EXPECTED_GYM_GITLINK_COMMIT="${EXPECTED_GYM_GITLINK_COMMIT}" \
  EXPECTED_GYM_TREE="${EXPECTED_GYM_TREE}" \
  EXPECTED_BRIDGE_HEAD="${EXPECTED_BRIDGE_HEAD}" \
  EXPECTED_BRIDGE_TREE="${EXPECTED_BRIDGE_TREE}" \
  EXPECTED_MCORE_HEAD="${EXPECTED_MCORE_HEAD}" \
  EXPECTED_MCORE_TREE="${EXPECTED_MCORE_TREE}" \
  BASE_LOG_DIR="${ARM_RESULTS_DIR}/ray_logs" \
  CPUS_PER_WORKER=144 \
  SANDBOX_COMMAND=/start-with-nginx.sh \
  NEMO_SKILLS_SANDBOX_PORT=6000 \
  RAY_LOG_SYNC_FREQUENCY=60 \
  NUM_TRAIN_NODES=1 \
  NUM_GEN_NODES=0 \
  NUM_GYM_NODES=0 \
  NUM_EXTERNAL_SERVICE_NODES=0 \
  SEGMENT_SIZE=1 \
  GPUS_PER_NODE=4 \
  COLOCATED_GENERATION=1 \
  DEDICATED_RAY_HEAD=0 \
  ENABLE_MTP_INFERENCE=0 \
  USE_CUSTOM_VLLM=0 \
  INTERACTIVE=0 \
  USE_SNAPSHOT=1 \
  STRICT_PREBUILT_SNAPSHOT_DIR="${ARM_SNAPSHOT_DIR}" \
  EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256="${STRICT_PAIR_SNAPSHOT_MANIFEST_SHA256}" \
  RAY_SUB="${STRICT_PAIR_JOB_WRAPPER}" \
  BATCH_SCRIPT="${STRICT_PAIR_JOB_WRAPPER}" \
  WALLTIME=04:00:00 \
  SLURM_QOS=normal \
  WANDB_ENTITY=nvidia \
  WANDB_PROJ=nano35-rlvr-convergence \
  WANDB_RUN_GROUP="${ARM_WANDB_RUN_GROUP}" \
  SLURM_PARTITION=batch \
  SLURM_ACCOUNT=nemotron_sw_post \
  STRICT_PAIR_SLURM_CONF="${STRICT_PAIR_SLURM_CONF}" \
  EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256="${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256}" \
  STRICT_PAIR_SUBMISSION_NONCE="${STRICT_PAIR_SUBMISSION_NONCE}" \
  STRICT_PAIR_SUBMISSION_CONTRACT_SHA256="${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" \
  EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256="${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" \
  STRICT_PAIR_ENVIRONMENT="${STRICT_PAIR_ENVIRONMENT}" \
  ${PAIR_MANIFEST_ENV[@]+"${PAIR_MANIFEST_ENV[@]}"} \
  ${ACCEPTED_ID_ENV[@]+"${ACCEPTED_ID_ENV[@]}"} \
  STRICT_PAIR_ARM="${ARM}" \
  STRICT_PAIR_SHARED_PREFIX_MODE="${SHARED_PREFIX_MODE}" \
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SCHEMA=nemo-rl-shared-prefix-determinism-attestation-v1 \
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION="${EXPECTED_DETERMINISM_ATTESTATION}" \
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SHA256="${EXPECTED_DETERMINISM_ATTESTATION_SHA256}" \
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_COUNT=4 \
  STRICT_PAIR_LAUNCH_MODE="${STRICT_PAIR_LAUNCH_MODE}" \
  EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256="${STRICT_PAIR_JOB_WRAPPER_SHA256}" \
  EXPECTED_STRICT_PAIR_FIXTURE_SHA256="${STRICT_PAIR_FIXTURE_SHA256}" \
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
  STRICT_PAIR_PREPARE_SLURM_EXPORT="${STRICT_PAIR_EXPORT_PREPARE_ONLY}" \
  STRICT_PAIR_SLURM_EXPORT_FILE="${STRICT_PAIR_SLURM_EXPORT_FILE}" \
  EXPECTED_STRICT_PAIR_SLURM_EXPORT_SHA256="${STRICT_PAIR_EXPECTED_SLURM_EXPORT_SHA256}" \
  "${STRICT_PAIR_TOOL_BASH}" -p "${LAUNCHER}" swe \
  "policy.shared_prefix_training.mode=${SHARED_PREFIX_MODE}" \
  "policy.shared_prefix_training.require_deterministic_execution=true" \
  "policy.generation.vllm_cfg.gpu_memory_utilization=0.1" \
  "policy.generation.vllm_cfg.strict_model_transport=capture" \
  'policy.megatron_cfg.env_vars.RESULTS_DIR=${RESULTS_DIR}' \
  'policy.megatron_cfg.env_vars.NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR=${NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR}'
