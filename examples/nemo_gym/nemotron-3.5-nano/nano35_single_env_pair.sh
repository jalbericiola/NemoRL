#!/bin/bash
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(realpath -- "${SCRIPT_DIR}/../../..")"
LAUNCHER_REL="examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"
CONFIG_PATH="examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml"
EXPECTED_FIXTURE_SHA256="da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"

# shellcheck source=strict_pair_contract.sh
source "${SCRIPT_DIR}/strict_pair_contract.sh"

usage() {
  echo "Usage: EXPECTED_PAIR_MANIFEST_SHA256=<sha256> bash ${BASH_SOURCE[0]} <off|on>" >&2
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

: "${PAIR_ID:?PAIR_ID is required to correlate the OFF/ON arms}"
: "${EXPECTED_PAIR_MANIFEST_SHA256:?Direct arm launch is forbidden; use launch_pair.sh}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required; strict pair runs must log to W&B}"
: "${TRAIN_PATH:?TRAIN_PATH is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
: "${STRICT_PAIR_JOB_WRAPPER:?STRICT_PAIR_JOB_WRAPPER is required}"
: "${EXPECTED_NEMO_HEAD:?EXPECTED_NEMO_HEAD is required}"
: "${EXPECTED_GYM_GITLINK_COMMIT:?EXPECTED_GYM_GITLINK_COMMIT is required}"
: "${STRICT_PAIR_LAUNCH_MODE:?STRICT_PAIR_LAUNCH_MODE is parent-owned; use launch_pair.sh}"

if [[ ! "${PAIR_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || (( ${#PAIR_ID} > 64 )); then
  strict_pair_error "PAIR_ID must be a 1-64 character filesystem-safe identifier."
fi
strict_pair_require_digest EXPECTED_PAIR_MANIFEST_SHA256
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
  BATCH_SCRIPT RAY_SUB MOUNTS EXTRA_MOUNTS USE_CUSTOM_VLLM INTERACTIVE \
  CHECKPOINTING_SAVE_BY WALLTIME SLURM_QOS SLURM_ACCOUNT DRY_RUN \
  WANDB_ENTITY WANDB_PROJ WANDB_RUN_GROUP; do
  if [[ -n "${!forbidden_name+x}" ]]; then
    strict_pair_error "${forbidden_name} must be unset; the strict pair owns this launcher input."
  fi
done
unset forbidden_name

strict_pair_load_arm_contract "${PROJECT_ROOT}" "${SCRIPT_DIR}"
strict_pair_verify_manifest

ARM_EXP_NAME="${ARM}-${PAIR_ID}"
ARM_RESULTS_DIR="${RESULTS_DIR%/}/${ARM}"
ARM_PERSISTENT_CACHE="${PERSISTENT_CACHE%/}/${ARM}"
if [[ "${ARM}" == "off" ]]; then
  ARM_SNAPSHOT_DIR="${STRICT_PAIR_OFF_SNAPSHOT}"
  STRICT_PAIR_SNAPSHOT_MANIFEST_SHA256="${STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256}"
else
  ARM_SNAPSHOT_DIR="${STRICT_PAIR_ON_SNAPSHOT}"
  STRICT_PAIR_SNAPSHOT_MANIFEST_SHA256="${STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256}"
fi
strict_pair_verify_snapshot "${ARM}" "${ARM_SNAPSHOT_DIR}"
LAUNCHER="${ARM_SNAPSHOT_DIR}/${LAUNCHER_REL}"
EXPECTED_DETERMINISM_ATTESTATION="SHARED_PREFIX_DETERMINISM_ATTESTED mode=${SHARED_PREFIX_MODE} env_controls=4 triton_autotune=absent model_overrides=3 torch_deterministic=true total_controls=8"
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
  strict_pair_error "fixture/model/container changed after PAIR_MANIFEST verification."
fi

echo "STRICT_PAIR_DEPLOYMENT_READY=${EXPECTED_DEPLOYMENT_READY} ready_file_sha256=${EXPECTED_DEPLOYMENT_READY_FILE_SHA256} nemo_manifest_sha256=${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256} bridge_manifest_sha256=${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256} mcore_manifest_sha256=${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256} root=${DEPLOYMENT_ROOT}"
echo "STRICT_PAIR_MANIFEST_SHA256=${EXPECTED_PAIR_MANIFEST_SHA256} path=${STRICT_PAIR_MANIFEST_PATH}"
echo "STRICT_PAIR_FIXTURE_PRE_SHA256=${FIXTURE_PRE_SHA256} rows=${STRICT_PAIR_FIXTURE_ROWS} path=${TRAIN_PATH}"
echo "STRICT_PAIR_MODEL_PRE_SHA256=${MODEL_PRE_SHA256} path=${MODEL_PATH}"
echo "STRICT_PAIR_CONTAINER_PRE_SHA256=${CONTAINER_PRE_SHA256} path=${CONTAINER}"
echo "STRICT_PAIR_RUN pair_id=${PAIR_ID} arm=${ARM} exp_name=${ARM_EXP_NAME} results=${ARM_RESULTS_DIR} cache=${ARM_PERSISTENT_CACHE}"
echo "STRICT_PAIR_RUNTIME_ATTESTATION_REQUIRED=1 arm=${ARM} mode=${SHARED_PREFIX_MODE} count=4 sha256_ascii_no_newline=${EXPECTED_DETERMINISM_ATTESTATION_SHA256}"
echo "STRICT_PAIR_JOB_INTERVAL_GATE_REQUIRED=1 implementation=${STRICT_PAIR_JOB_WRAPPER} sha256=${STRICT_PAIR_JOB_WRAPPER_SHA256} pre=driver-exec post=driver-success artifacts=pair-manifest,fixture,model,container,sandbox-container,snapshot"

# The public launcher keeps its existing swe/rlvr modes. This wrapper selects
# the judge-free allocation path while supplying the sealed SingleController
# recipe and exact colocated four-rank shape.
DRY_RUN_ENV=()
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "dry-run" ]]; then
  DRY_RUN_ENV=(DRY_RUN=1)
fi
exec env \
  "${DRY_RUN_ENV[@]}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  TRAIN_ENTRYPOINT="./examples/run_grpo_single_controller.py" \
  EXP_NAME="${ARM_EXP_NAME}" \
  RESULTS_DIR="${ARM_RESULTS_DIR}" \
  PERSISTENT_CACHE="${ARM_PERSISTENT_CACHE}" \
  TRAIN_PATH="${TRAIN_PATH}" \
  VAL_PATH="${TRAIN_PATH}" \
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
  WANDB_RUN_GROUP="${PAIR_ID}" \
  SLURM_PARTITION=batch \
  SLURM_ACCOUNT=nemotron_sw_post \
  STRICT_PAIR_MANIFEST_PATH="${STRICT_PAIR_MANIFEST_PATH}" \
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
  bash "${LAUNCHER}" swe \
  "policy.shared_prefix_training.mode=${SHARED_PREFIX_MODE}" \
  "policy.shared_prefix_training.require_deterministic_execution=true" \
  'policy.megatron_cfg.env_vars.RESULTS_DIR=${RESULTS_DIR}' \
  'policy.megatron_cfg.env_vars.NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR=${NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR}'
