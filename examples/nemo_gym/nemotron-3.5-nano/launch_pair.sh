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
ARM_WRAPPER="${SCRIPT_DIR}/nano35_single_env_pair.sh"
EXPECTED_FIXTURE_SHA256="da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"

# shellcheck source=strict_pair_contract.sh
source "${SCRIPT_DIR}/strict_pair_contract.sh"

if (( $# != 1 )) || [[ "$1" != "--dry-run" && "$1" != "--submit" ]]; then
  echo "Usage: PAIR_ID=<id> bash ${BASH_SOURCE[0]} <--dry-run|--submit>" >&2
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
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
: "${STRICT_PAIR_JOB_WRAPPER:?STRICT_PAIR_JOB_WRAPPER is required}"
: "${EXPECTED_NEMO_HEAD:?EXPECTED_NEMO_HEAD is required}"
: "${EXPECTED_GYM_GITLINK_COMMIT:?EXPECTED_GYM_GITLINK_COMMIT is required}"

if [[ ! "${PAIR_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || (( ${#PAIR_ID} > 64 )); then
  strict_pair_error "PAIR_ID must be a 1-64 character filesystem-safe identifier."
fi
if [[ -n "${EXPECTED_PAIR_MANIFEST_SHA256+x}" ]]; then
  strict_pair_error "EXPECTED_PAIR_MANIFEST_SHA256 is parent-owned output and must be unset."
fi
if [[ -n "${CODE_SNAPSHOT_DIRNAME+x}" ]]; then
  strict_pair_error "CODE_SNAPSHOT_DIRNAME must be unset; the strict pair owns its fresh snapshot parent."
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

strict_pair_prepare_contract "${PROJECT_ROOT}" "${SCRIPT_DIR}"

strict_pair_build_snapshot off
strict_pair_build_snapshot on
strict_pair_publish_manifest
echo "STRICT_PAIR_MANIFEST_SHA256=${STRICT_PAIR_MANIFEST_SHA256} path=${STRICT_PAIR_MANIFEST_PATH}"
echo "STRICT_PAIR_PLAN pair_id=${PAIR_ID} arms=off:observe,on:train mode=${STRICT_PAIR_LAUNCH_MODE} submissions=parallel partition=batch"

off_stdout="$(mktemp "${RESULTS_DIR}/.strict-pair-off.stdout.XXXXXX")"
off_stderr="$(mktemp "${RESULTS_DIR}/.strict-pair-off.stderr.XXXXXX")"
on_stdout="$(mktemp "${RESULTS_DIR}/.strict-pair-on.stdout.XXXXXX")"
on_stderr="$(mktemp "${RESULTS_DIR}/.strict-pair-on.stderr.XXXXXX")"
cleanup_arm_output() {
  rm -f -- "${off_stdout}" "${off_stderr}" "${on_stdout}" "${on_stderr}"
}
trap cleanup_arm_output EXIT

EXPECTED_PAIR_MANIFEST_SHA256="${STRICT_PAIR_MANIFEST_SHA256}" \
  STRICT_PAIR_LAUNCH_MODE="${STRICT_PAIR_LAUNCH_MODE}" \
  EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256="${STRICT_PAIR_MODEL_TREE_SHA256}" \
  EXPECTED_STRICT_PAIR_CONTAINER_SHA256="${STRICT_PAIR_CONTAINER_SHA256}" \
  EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256="${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" \
  bash "${ARM_WRAPPER}" off >"${off_stdout}" 2>"${off_stderr}" &
off_pid=$!
EXPECTED_PAIR_MANIFEST_SHA256="${STRICT_PAIR_MANIFEST_SHA256}" \
  STRICT_PAIR_LAUNCH_MODE="${STRICT_PAIR_LAUNCH_MODE}" \
  EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256="${STRICT_PAIR_MODEL_TREE_SHA256}" \
  EXPECTED_STRICT_PAIR_CONTAINER_SHA256="${STRICT_PAIR_CONTAINER_SHA256}" \
  EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256="${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" \
  bash "${ARM_WRAPPER}" on >"${on_stdout}" 2>"${on_stderr}" &
on_pid=$!

off_status=0
on_status=0
wait "${off_pid}" || off_status=$?
wait "${on_pid}" || on_status=$?
cat -- "${off_stdout}" "${on_stdout}"
cat -- "${off_stderr}" "${on_stderr}" >&2
if (( off_status != 0 || on_status != 0 )); then
  strict_pair_error "strict pair launch failed: off_status=${off_status} on_status=${on_status}"
fi
if [[ "${STRICT_PAIR_LAUNCH_MODE}" == "dry-run" ]]; then
  echo "STRICT_PAIR_DRY_RUN_VALIDATED pair_id=${PAIR_ID} off_status=0 on_status=0 submitted=0"
else
  echo "STRICT_PAIR_LAUNCHED pair_id=${PAIR_ID} off_status=0 on_status=0 submitted=2"
fi
