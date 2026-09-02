#!/bin/bash
set -euo pipefail

# =============================================================================
# nano35_launch.sh
#
# Public launcher for Nemotron 3.5 Nano post-training on a SLURM cluster.
#
# The SWE and RLVR workload semantics live in sibling YAML files. This launcher
# handles Slurm submission, code snapshotting, persistent caches, container
# mounts, and deployment-specific overrides.
#
# Usage:
#
#   EXP_NAME=nano35-swe \
#   MODEL_PATH=/path/to/nano35-checkpoint \
#   TRAIN_PATH=/path/to/train.jsonl \
#   VAL_PATH=/path/to/val.jsonl \
#   CONTAINER=/path/to/nemo-rl-container.sqsh \
#   SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
#   PERSISTENT_CACHE=/path/to/persistent/cache \
#   SLURM_PARTITION=batch \
#   SLURM_ACCOUNT=your_account \
#   SIF_DIR=/path/to/swe-sif-root \
#   bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe
#
#   EXP_NAME=nano35-rlvr \
#   MODEL_PATH=/path/to/nano35-checkpoint \
#   TRAIN_PATH=/path/to/train.jsonl \
#   VAL_PATH=/path/to/val.jsonl \
#   CONTAINER=/path/to/nemo-rl-container.sqsh \
#   SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
#   PERSISTENT_CACHE=/path/to/persistent/cache \
#   SLURM_PARTITION=batch \
#   SLURM_ACCOUNT=your_account \
#   GENRM_MODEL=/path/to/genrm-checkpoint \
#   GENRM_REASONING_PARSER=/path/to/ultra_v3_reasoning_parser.py \
#   NL2BASH_JUDGE_MODEL=/path/to/general-judge-checkpoint \
#   SAFETY_JUDGE_MODEL=/path/to/safety-checkpoint \
#   bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh rlvr
#
# Optional knobs:
#   WALLTIME=4:00:00                       Slurm --time
#   SLURM_QOS=                             Slurm --qos; defaults to short when
#                                          WALLTIME is under two hours
#   SLURM_RESERVATION=                     Slurm --reservation
#   SLURM_DEPENDENCY=                      Extra Slurm dependency, merged with
#                                          singleton (e.g. afterany:<jobid>)
#   EXCLUDE_NODES=                         Slurm --exclude
#   NUM_TRAIN_NODES=                        Training (Megatron) nodes
#   NUM_GEN_NODES=                          Policy-generation nodes
#   NUM_GYM_NODES=                          In-cluster NeMo Gym judge nodes
#   COLOCATED_GENERATION=0                  Set to 1 only for the exact
#                                          single-node shape: train=1, gen=0,
#                                          Gym=0, segment=1, four GPUs/node.
#                                          Requires a SingleController-compatible
#                                          entrypoint and four-rank model topology.
#   NUM_EXTERNAL_SERVICE_NODES=0            Nodes reserved outside training Ray
#   GENRM_SEGMENT_SIZE=                      Segment size for the external
#                                          service hetgroup
#   BATCH_SCRIPT=ray.sub                    Slurm entrypoint; external services
#                                          may wrap ray.sub
#   ENABLE_MTP_INFERENCE=0                 1 to enable MTP speculative decoding
#   NUM_SPECULATIVE_TOKENS=5               MTP speculative tokens
#   MAX_NUM_BATCHED_TOKENS=8480            vLLM max batched tokens (MTP)
#   NRL_MAX_STEPS=                         Override grpo.max_num_steps
#   EXTRA_MOUNTS=                          Comma-separated host:container pairs
#   USE_SNAPSHOT=1                         Snapshot source tree at submission
#   USE_CUSTOM_VLLM=0                      1 to source a custom vLLM checkout
#   DRY_RUN=0                              1 to print TRAIN_CMD and exit
#   INTERACTIVE=0                          1 to bring up Ray and idle for attach
#                                          (no training driver) for debugging
#   INTERACTIVE_WAIT=1                     0 to submit and return immediately
#   INTERACTIVE_WALLTIME=                  override WALLTIME for the interactive alloc
#   HF_HOME=                               HuggingFace cache root (recommended)
#   HF_TOKEN=                              HuggingFace API token
#   WANDB_API_KEY=                         Weights & Biases API key
#   WANDB_PROJ=nemotron-3.5-nano           W&B project
#   WANDB_ENTITY=                          W&B entity
#   SLURM_COMMENT=                         Job-reaper exemption JSON
#
# Hydra overrides are forwarded verbatim as positional arguments:
#   bash .../nano35_launch.sh swe policy.megatron_cfg.optimizer.lr=1e-6
#
# The reference profiles target four-GPU GB200 nodes. With external service
# nodes, Slurm uses two heterogeneous components so the services remain outside
# the training Ray cluster. Each component must be divisible by its own segment
# size.
# =============================================================================

# =============================================================================
# Recipe selection
# =============================================================================
_NANO35_SCRIPT_SOURCE="${BASH_SOURCE[0]}"
case "${_NANO35_SCRIPT_SOURCE}" in
  /*) _NANO35_SCRIPT_ABSOLUTE="${_NANO35_SCRIPT_SOURCE}" ;;
  */*) _NANO35_SCRIPT_ABSOLUTE="${PWD}/${_NANO35_SCRIPT_SOURCE}" ;;
  *) _NANO35_SCRIPT_ABSOLUTE="${PWD}/${_NANO35_SCRIPT_SOURCE}" ;;
esac
_NANO35_SCRIPT_DIR_RAW="${_NANO35_SCRIPT_ABSOLUTE%/*}"
STRICT_PAIR_HOST_RUNTIME=0
if [[ -n "${STRICT_PREBUILT_SNAPSHOT_DIR:-}" ]]; then
  if [[ "$-" != *p* ]]; then
    echo "ERROR: strict prebuilt mode requires privileged Bash (-p)." >&2
    exit 2
  fi
  while IFS= read -r _nano35_environment_name; do
    case "${_nano35_environment_name}" in
      BASH_ENV|ENV|BASHOPTS|SHELLOPTS|BASH_COMPAT|CDPATH|GLOBIGNORE|BASH_XTRACEFD|\
      PYTHON*|GIT_*|LD_*|DYLD_*|BASH_FUNC_*%%)
        echo "ERROR: hostile startup environment variable must be unset: ${_nano35_environment_name}" >&2
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

  : "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256:?strict prebuilt mode requires bootstrap SHA-256}"
  : "${EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256:?strict prebuilt mode requires snapshot-manifest SHA-256}"
  if [[ -f /usr/bin/sha256sum && ! -L /usr/bin/sha256sum && \
        -x /usr/bin/sha256sum && ! -w /usr/bin/sha256sum ]]; then
    _NANO35_BOOTSTRAP_SHA256SUM=/usr/bin/sha256sum
  elif [[ -f /sbin/sha256sum && ! -L /sbin/sha256sum && \
          -x /sbin/sha256sum && ! -w /sbin/sha256sum ]]; then
    _NANO35_BOOTSTRAP_SHA256SUM=/sbin/sha256sum
  else
    echo "ERROR: no supported fixed bootstrap sha256sum is available." >&2
    exit 2
  fi
  if ! _nano35_hash_output="$(
    "${_NANO35_BOOTSTRAP_SHA256SUM}" -- "${_NANO35_BOOTSTRAP_SHA256SUM}"
  )"; then
    echo "ERROR: bootstrap sha256sum failed while authenticating itself." >&2
    exit 2
  fi
  _nano35_actual_sha256="${_nano35_hash_output%% *}"
  if [[ "${_nano35_actual_sha256}" != \
        "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" ]]; then
    echo "ERROR: bootstrap sha256sum SHA-256 mismatch." >&2
    exit 2
  fi

  _nano35_expected_launcher="${STRICT_PREBUILT_SNAPSHOT_DIR}/examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"
  _nano35_contract="${STRICT_PREBUILT_SNAPSHOT_DIR}/examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh"
  _nano35_snapshot_manifest="${STRICT_PREBUILT_SNAPSHOT_DIR}/strict-pair-snapshot-manifest.sha256"
  if [[ "${BASH_SOURCE[0]}" != "${_nano35_expected_launcher}" ]]; then
    echo "ERROR: strict prebuilt launcher must be the exact authenticated snapshot path." >&2
    exit 2
  fi
  for _nano35_guard_path in "${BASH_SOURCE[0]}" "${_nano35_contract}" \
                            "${_nano35_snapshot_manifest}"; do
    if [[ -L "${_nano35_guard_path}" || ! -f "${_nano35_guard_path}" || \
          -w "${_nano35_guard_path}" ]]; then
      echo "ERROR: strict prebuilt startup input must be regular, non-symlink, and sealed: ${_nano35_guard_path}" >&2
      exit 2
    fi
  done
  if [[ ! -x "${BASH_SOURCE[0]}" || ! -x "${_nano35_contract}" ]]; then
    echo "ERROR: strict prebuilt launcher and contract must be executable." >&2
    exit 2
  fi
  if ! _nano35_hash_output="$(
    "${_NANO35_BOOTSTRAP_SHA256SUM}" -- "${_nano35_snapshot_manifest}"
  )"; then
    echo "ERROR: bootstrap sha256sum failed for strict snapshot manifest." >&2
    exit 2
  fi
  _nano35_actual_sha256="${_nano35_hash_output%% *}"
  if [[ "${_nano35_actual_sha256}" != \
        "${EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256}" ]]; then
    echo "ERROR: strict prebuilt snapshot manifest SHA-256 mismatch before source." >&2
    exit 2
  fi
  _nano35_launcher_manifest_sha256=""
  _nano35_contract_manifest_sha256=""
  while IFS= read -r _nano35_manifest_line; do
    _nano35_manifest_sha256="${_nano35_manifest_line%%  *}"
    _nano35_manifest_path="${_nano35_manifest_line#*  }"
    case "${_nano35_manifest_path}" in
      examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh)
        [[ -z "${_nano35_launcher_manifest_sha256}" ]] || {
          echo "ERROR: duplicate nano35 launcher snapshot-manifest entry." >&2
          exit 2
        }
        _nano35_launcher_manifest_sha256="${_nano35_manifest_sha256}"
        ;;
      examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh)
        [[ -z "${_nano35_contract_manifest_sha256}" ]] || {
          echo "ERROR: duplicate strict contract snapshot-manifest entry." >&2
          exit 2
        }
        _nano35_contract_manifest_sha256="${_nano35_manifest_sha256}"
        ;;
    esac
  done < "${_nano35_snapshot_manifest}"
  for _nano35_guard_record in \
    "${BASH_SOURCE[0]}:${_nano35_launcher_manifest_sha256}:launcher" \
    "${_nano35_contract}:${_nano35_contract_manifest_sha256}:contract"; do
    _nano35_guard_path="${_nano35_guard_record%%:*}"
    _nano35_guard_rest="${_nano35_guard_record#*:}"
    _nano35_expected_sha256="${_nano35_guard_rest%%:*}"
    _nano35_guard_label="${_nano35_guard_rest#*:}"
    if ! _nano35_hash_output="$(
      "${_NANO35_BOOTSTRAP_SHA256SUM}" -- "${_nano35_guard_path}"
    )"; then
      echo "ERROR: bootstrap sha256sum failed for strict ${_nano35_guard_label}." >&2
      exit 2
    fi
    _nano35_actual_sha256="${_nano35_hash_output%% *}"
    if [[ -z "${_nano35_expected_sha256}" || \
          "${_nano35_actual_sha256}" != "${_nano35_expected_sha256}" ]]; then
      echo "ERROR: strict ${_nano35_guard_label} is absent or drifted before source." >&2
      exit 2
    fi
  done

  # The parent arm anchored the complete snapshot. Re-authenticate the exact
  # launcher and sibling contract before sourcing any snapshot-owned shell.
  # shellcheck source=strict_pair_contract.sh
  source "${_nano35_contract}"
  strict_pair_load_runtime_tools
  STRICT_PAIR_HOST_RUNTIME=1
  STRICT_PAIR_PREPARE_SLURM_EXPORT="${STRICT_PAIR_PREPARE_SLURM_EXPORT:-0}"
  case "${STRICT_PAIR_PREPARE_SLURM_EXPORT}" in
    0|1) ;;
    *)
      echo "ERROR: STRICT_PAIR_PREPARE_SLURM_EXPORT must be parent-owned 0 or 1." >&2
      exit 2
      ;;
  esac
  NANO35_REALPATH="${STRICT_PAIR_TOOL_REALPATH}"
  NANO35_SHA256SUM="${STRICT_PAIR_TOOL_SHA256SUM}"
  NANO35_PYTHON="${STRICT_PAIR_TOOL_PYTHON}"
  NANO35_STAT="${STRICT_PAIR_TOOL_STAT}"
  NANO35_FIND="${STRICT_PAIR_TOOL_FIND}"
  NANO35_SBATCH="${STRICT_PAIR_TOOL_SBATCH}"
  NANO35_ENV="${STRICT_PAIR_TOOL_ENV}"
  NANO35_DATE="${STRICT_PAIR_TOOL_DATE}"
  NANO35_GREP="${STRICT_PAIR_TOOL_GREP}"
  NANO35_LN="${STRICT_PAIR_TOOL_LN}"
  NANO35_MKDIR="${STRICT_PAIR_TOOL_MKDIR}"
  NANO35_MKTEMP="${STRICT_PAIR_TOOL_MKTEMP}"
  NANO35_CHMOD="${STRICT_PAIR_TOOL_CHMOD}"
  NANO35_RM="${STRICT_PAIR_TOOL_RM}"
  unset _NANO35_BOOTSTRAP_SHA256SUM _nano35_actual_sha256 \
    _nano35_contract _nano35_contract_manifest_sha256 \
    _nano35_environment_name _nano35_expected_launcher \
    _nano35_expected_sha256 _nano35_guard_label _nano35_guard_path \
    _nano35_guard_record _nano35_guard_rest _nano35_hash_output \
    _nano35_launcher_manifest_sha256 _nano35_manifest_line \
    _nano35_manifest_path _nano35_manifest_sha256 \
    _nano35_snapshot_manifest
else
  STRICT_PAIR_PREPARE_SLURM_EXPORT=0
  NANO35_REALPATH=realpath
  NANO35_SHA256SUM=sha256sum
  NANO35_PYTHON=python3
  NANO35_STAT=stat
  NANO35_FIND=find
  NANO35_SBATCH=sbatch
  NANO35_ENV=env
  NANO35_DATE=date
  NANO35_GREP=grep
  NANO35_LN=ln
  NANO35_MKDIR=mkdir
  NANO35_MKTEMP=mktemp
  NANO35_CHMOD=chmod
  NANO35_RM=rm
fi
SCRIPT_DIR="$(cd -- "${_NANO35_SCRIPT_DIR_RAW}" && pwd -P)"
PROJECT_ROOT="$("${NANO35_REALPATH}" "${SCRIPT_DIR}/../../..")"
unset _NANO35_SCRIPT_ABSOLUTE _NANO35_SCRIPT_DIR_RAW _NANO35_SCRIPT_SOURCE

if [[ $# -lt 1 ]]; then
  echo "Usage: bash ${BASH_SOURCE[0]} <swe|rlvr> [Hydra overrides ...]" >&2
  exit 2
fi

RECIPE="$1"
shift

case "${RECIPE}" in
  swe)
    CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/swe.yaml}"
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-16}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
    NUM_GYM_NODES="${NUM_GYM_NODES:-0}"
    NUM_EXTERNAL_SERVICE_NODES=0
    SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
    GENRM_BASE_URL=""
    GENRM_MODEL=""
    GENRM_API_MODEL_NAME=""
    NL2BASH_JUDGE_MODEL=""
    SAFETY_JUDGE_MODEL=""
    ;;
  rlvr)
    CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr.yaml}"
    NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
    NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
    NUM_GYM_NODES="${NUM_GYM_NODES:-6}"
    NUM_EXTERNAL_SERVICE_NODES="${NUM_EXTERNAL_SERVICE_NODES:-16}"
    SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

    : "${GENRM_MODEL:?GENRM_MODEL is required for the RLVR recipe}"
    : "${GENRM_REASONING_PARSER:?GENRM_REASONING_PARSER is required for the RLVR recipe}"
    : "${NL2BASH_JUDGE_MODEL:?NL2BASH_JUDGE_MODEL is required for the RLVR recipe}"
    : "${SAFETY_JUDGE_MODEL:?SAFETY_JUDGE_MODEL is required for the RLVR recipe}"

    GENRM_BASE_URL="__GENRM_BASE_URL__"
    GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
    GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"
    GENRM_SERVED_MODEL_NAME="${GENRM_SERVED_MODEL_NAME:-model}"
    GENRM_API_MODEL_NAME="${GENRM_API_MODEL_NAME:-${GENRM_SERVED_MODEL_NAME}}"
    NUM_GENRM_NODES="${NUM_GENRM_NODES:-${NUM_EXTERNAL_SERVICE_NODES}}"
    GENRM_VLLM_PORT="${GENRM_VLLM_PORT:-8000}"
    GENRM_LB_PORT="${GENRM_LB_PORT:-9213}"
    GENRM_STARTUP_TIMEOUT="${GENRM_STARTUP_TIMEOUT:-3600}"
    GENRM_CONTAINER="${GENRM_CONTAINER:-${CONTAINER:-}}"
    GENRM_VLLM_PYTHON="${GENRM_VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
    GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-ultra_v3}"
    GENRM_TOOL_CALL_PARSER="${GENRM_TOOL_CALL_PARSER:-qwen3_coder}"
    GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-1}"
    GENRM_COMPILATION_CONFIG="${GENRM_COMPILATION_CONFIG:-{\"pass_config\":{\"fuse_allreduce_rms\":false}}}"
    GENRM_MODEL_LOADER_EXTRA_CONFIG="${GENRM_MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":96}}"
    GENRM_TOOLS_DIR_HOST="${GENRM_TOOLS_DIR_HOST:-${PROJECT_ROOT}/tools/external_genrm}"
    RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
    BATCH_SCRIPT="${BATCH_SCRIPT:-${PROJECT_ROOT}/tools/external_genrm/run_in_allocation.sh}"
    export \
      GENRM_COMPILATION_CONFIG \
      GENRM_CONTAINER \
      GENRM_ENABLE_EXPERT_PARALLEL \
      GENRM_LB_PORT \
      GENRM_MODEL \
      GENRM_MODEL_LOADER_EXTRA_CONFIG \
      GENRM_REASONING_PARSER \
      GENRM_REASONING_PARSER_NAME \
      GENRM_REPLICAS \
      GENRM_SERVED_MODEL_NAME \
      GENRM_STARTUP_TIMEOUT \
      GENRM_TENSOR_PARALLEL_SIZE \
      GENRM_TOOL_CALL_PARSER \
      GENRM_TOOLS_DIR_HOST \
      GENRM_VLLM_PORT \
      GENRM_VLLM_PYTHON \
      NUM_GENRM_NODES
    ;;
  *)
    echo "ERROR: unknown recipe '${RECIPE}'; expected swe or rlvr." >&2
    exit 2
    ;;
esac

# =============================================================================
# Required environment
# =============================================================================
: "${EXP_NAME:?EXP_NAME is required (used for job name, W&B run, checkpoint/log dirs)}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"

# Driver script, relative to the repo root. SingleController recipes need
# ./examples/run_grpo_single_controller.py; defaulting to the async-GRPO driver
# keeps an unset value reproducing the historical run.
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/nemo_gym/run_grpo_nemo_gym.py}"
: "${MODEL_PATH:?MODEL_PATH is required (initial policy checkpoint, HF repo id or local path)}"
: "${TRAIN_PATH:?TRAIN_PATH is required (training data jsonl path)}"
: "${VAL_PATH:?VAL_PATH is required (validation data jsonl path)}"
: "${CONTAINER:?CONTAINER is required (NGC image URI or .sqsh path)}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required (nemo-skills sandbox image)}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required (shared directory for vLLM/Triton/Inductor caches)}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  STRICT_PAIR_EXPECTED_WANDB_NAME="${STRICT_PAIR_ARM}-${STRICT_PAIR_ENVIRONMENT}-${PAIR_ID}"
  STRICT_PAIR_EXPECTED_WANDB_GROUP="${STRICT_PAIR_ENVIRONMENT}-${PAIR_ID}"
  STRICT_PAIR_EXPECTED_WANDB_RUN_ID="$(
    strict_pair_sha256_text \
      "nemo-rl-strict-wandb-v1:${STRICT_PAIR_ENVIRONMENT}:${PAIR_ID}:${STRICT_PAIR_ARM}"
  )"
  if [[ "${BASE_LOG_DIR:-}" != "${RESULTS_DIR}/ray_logs" || \
        "${CPUS_PER_WORKER:-}" != "144" || \
        "${HF_HUB_CACHE:-}" != "${HF_HOME}/hub" || \
        "${HF_DATASETS_CACHE:-}" != "${HF_HOME}/hub" || \
        "${SANDBOX_COMMAND:-}" != "/start-with-nginx.sh" || \
        "${NEMO_SKILLS_SANDBOX_PORT:-}" != "6000" || \
        "${RAY_LOG_SYNC_FREQUENCY:-}" != "60" || \
        "${VAL_PATH}" != "${TRAIN_PATH}" ]]; then
    echo "ERROR: strict execution controls differ from the closed pair contract." >&2
    exit 2
  fi
  if [[ "${EXP_NAME}" != "${STRICT_PAIR_EXPECTED_WANDB_NAME}" || \
        "${WANDB_NAME:-}" != "${STRICT_PAIR_EXPECTED_WANDB_NAME}" || \
        "${WANDB_ENTITY:-}" != "nvidia" || \
        "${WANDB_PROJ:-}" != "nano35-rlvr-convergence" || \
        "${WANDB_RESUME:-}" != "never" || \
        "${WANDB_RUN_GROUP:-}" != "${STRICT_PAIR_EXPECTED_WANDB_GROUP}" || \
        "${WANDB_RUN_ID:-}" != "${STRICT_PAIR_EXPECTED_WANDB_RUN_ID}" ]]; then
    echo "ERROR: strict W&B identity differs from the closed pair contract." >&2
    exit 2
  fi
  if [[ -n "${SETUP_COMMAND+x}" || -n "${SLURM_SUBMIT_DIR+x}" ]]; then
    echo "ERROR: SETUP_COMMAND and SLURM_SUBMIT_DIR must be unset before strict rendering." >&2
    exit 2
  fi
fi
cd "${PROJECT_ROOT}"

# SingleController is outside the directories historically overlaid into the
# container. Bind its exact committed bytes explicitly when selected; silently
# falling back to a container-builtin driver can change orchestration semantics
# while the printed command still looks correct.
SINGLE_CONTROLLER_ENTRYPOINT_REL="examples/run_grpo_single_controller.py"
SINGLE_CONTROLLER_ENTRYPOINT_SHA256=""
SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST=""

sha256_file() {
  local path="$1"
  local digest
  local output

  if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
    if ! output="$("${NANO35_SHA256SUM}" -- "${path}")"; then
      echo "ERROR: authenticated sha256sum failed for ${path}." >&2
      return 1
    fi
    digest="${output%% *}"
  elif command -v sha256sum >/dev/null 2>&1; then
    read -r digest ignored < <(sha256sum -- "${path}")
  elif command -v shasum >/dev/null 2>&1; then
    read -r digest ignored < <(shasum -a 256 -- "${path}")
  else
    echo "ERROR: sha256sum or shasum is required to authenticate the SingleController entrypoint." >&2
    return 1
  fi
  echo "${digest}"
}

if [[ "${TRAIN_ENTRYPOINT}" == "./${SINGLE_CONTROLLER_ENTRYPOINT_REL}" ]]; then
  SINGLE_CONTROLLER_ENTRYPOINT_SOURCE="${PROJECT_ROOT}/${SINGLE_CONTROLLER_ENTRYPOINT_REL}"
  if [[ -L "${SINGLE_CONTROLLER_ENTRYPOINT_SOURCE}" ]]; then
    echo "ERROR: SingleController entrypoint must not be a symbolic link: ${SINGLE_CONTROLLER_ENTRYPOINT_SOURCE}" >&2
    exit 1
  fi
  if [[ ! -f "${SINGLE_CONTROLLER_ENTRYPOINT_SOURCE}" ]]; then
    echo "ERROR: SingleController entrypoint is missing or not a regular file: ${SINGLE_CONTROLLER_ENTRYPOINT_SOURCE}" >&2
    exit 1
  fi
  if [[ -n "${STRICT_PREBUILT_SNAPSHOT_DIR:-}" ]]; then
    if [[ "${PROJECT_ROOT}" != "${STRICT_PREBUILT_SNAPSHOT_DIR}" ]]; then
      echo "ERROR: strict prebuilt mode must execute nano35_launch.sh from the authenticated snapshot." >&2
      exit 1
    fi
  elif ! git -C "${PROJECT_ROOT}" diff --quiet HEAD -- "${SINGLE_CONTROLLER_ENTRYPOINT_REL}"; then
    echo "ERROR: SingleController entrypoint differs from committed HEAD; refusing a mutable container bind." >&2
    exit 1
  fi
  SINGLE_CONTROLLER_ENTRYPOINT_SHA256="$(sha256_file "${SINGLE_CONTROLLER_ENTRYPOINT_SOURCE}")"
fi
# Judge models are recipe-specific. RLVR needs GenRM, NL2Bash, and safety
# judges; SWE uses code-execution rewards and needs none of them. Set these per
# recipe; unset variables skip the corresponding override.
NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-}"
SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-}"
GENRM_BASE_URL="${GENRM_BASE_URL:-}"
GENRM_MODEL="${GENRM_MODEL:-}"
GENRM_API_MODEL_NAME="${GENRM_API_MODEL_NAME:-}"
GENRM_OVERRIDE=""
if [[ -n "${GENRM_BASE_URL}" ]]; then
  GENRM_OVERRIDE="++env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=${GENRM_BASE_URL}"
  if [[ -n "${GENRM_API_MODEL_NAME}" ]]; then
    GENRM_OVERRIDE="${GENRM_OVERRIDE} ++env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_API_MODEL_NAME}"
  fi
elif [[ -n "${GENRM_MODEL}" ]]; then
  GENRM_OVERRIDE="env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_MODEL}"
fi

# SIF_DIR: for the SWE recipe — directory containing Apptainer .sif
# images for SWE-Bench / SWE-Gym / R2E-Gym instances. The yaml's
# container_formatter uses `${sif_dir}/...` paths. Unset for non-SWE recipes.
SIF_DIR="${SIF_DIR:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

# The SWE recipe interpolates `${sif_dir}/...` paths at runtime. The
# exemplar config carries only a placeholder, so hard-require SIF_DIR whenever
# the selected config actually uses it (mirrors the teacher-path guard).
if "${NANO35_GREP}" -q '${sif_dir}' "${CONFIG_PATH}"; then
  : "${SIF_DIR:?SIF_DIR is required for the SWE recipe (directory of apptainer .sif images)}"
fi

# =============================================================================
# Job identity — fixed name for singleton.
# Slurm --dependency=singleton serialises queued submissions with the same name
# so a resubmission after preemption resumes from the latest checkpoint instead
# of running in parallel.
# =============================================================================
JOB_NAME="${EXP_NAME}"

# =============================================================================
# Output directories
# =============================================================================
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"

# Per-submission dirs for logs and Slurm output (timestamped for history).
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  RUN_DIR="${RESULTS_DIR}/runs/${PAIR_ID:?strict prebuilt mode requires PAIR_ID}"
else
  RUN_DIR="${RESULTS_DIR}/runs/$("${NANO35_DATE}" +%Y%m%d-%H%M)"
fi
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
if [[ "${STRICT_PAIR_PREPARE_SLURM_EXPORT}" != "1" ]]; then
  "${NANO35_MKDIR}" -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
  "${NANO35_LN}" -sfn "$("${NANO35_REALPATH}" "${RUN_DIR}")" "${RESULTS_DIR}/runs/latest"
fi

# ray.sub reads BASE_LOG_DIR and creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs/ for
# ray infrastructure logs (ray-head.log, ray-driver.log, ray-worker-*.log,
# topology probes, attach scripts, etc.).
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/ray_logs}"

# =============================================================================
# SLURM configuration
# =============================================================================
WALLTIME="${WALLTIME:-4:00:00}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
SLURM_COMMENT="${SLURM_COMMENT:-}"
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" && \
      "${STRICT_PAIR_PREPARE_SLURM_EXPORT}" != "1" ]]; then
  : "${STRICT_PAIR_SUBMISSION_NONCE:?strict pair submission nonce is required}"
  : "${EXPECTED_PAIR_MANIFEST_SHA256:?strict pair manifest SHA-256 is required}"
  SLURM_COMMENT="nemo-rl-strict-pair-v1:${STRICT_PAIR_ARM}:${STRICT_PAIR_SUBMISSION_NONCE}:${EXPECTED_PAIR_MANIFEST_SHA256}"
fi
SLURM_COMMENT_ARGS=()
if [[ -n "${SLURM_COMMENT}" ]]; then
  SLURM_COMMENT_ARGS=(--comment="${SLURM_COMMENT}")
fi

slurm_walltime_seconds() {
  local value="$1"
  local days=0
  local -a fields

  if [[ "${value}" == *-* ]]; then
    days="${value%%-*}"
    value="${value#*-}"
  fi
  [[ "${days}" =~ ^[0-9]+$ ]] || return 1

  IFS=: read -r -a fields <<< "${value}"
  for field in "${fields[@]}"; do
    [[ "${field}" =~ ^[0-9]+$ ]] || return 1
  done

  case "${#fields[@]}" in
    1)
      if (( days > 0 )); then
        echo $((10#${days} * 86400 + 10#${fields[0]} * 3600))
      else
        echo $((10#${fields[0]} * 60))
      fi
      ;;
    2)
      if (( days > 0 )); then
        echo $((10#${days} * 86400 + 10#${fields[0]} * 3600 + 10#${fields[1]} * 60))
      else
        echo $((10#${fields[0]} * 60 + 10#${fields[1]}))
      fi
      ;;
    3)
      echo $((10#${days} * 86400 + 10#${fields[0]} * 3600 + 10#${fields[1]} * 60 + 10#${fields[2]}))
      ;;
    *) return 1 ;;
  esac
}

if [[ -z "${SLURM_QOS}" ]]; then
  if WALLTIME_SECONDS="$(slurm_walltime_seconds "${WALLTIME}")"; then
    if (( WALLTIME_SECONDS < 2 * 60 * 60 )); then
      SLURM_QOS=short
    fi
  else
    echo "[WARN] Could not parse WALLTIME=${WALLTIME}; leaving SLURM_QOS unset." >&2
  fi
fi
# INTERACTIVE=1 brings up the Ray cluster and idles for attachment (no training
# driver), so you can run/debug the recipe by hand. INTERACTIVE_WAIT=1 (default)
# blocks until Ray is ready; INTERACTIVE_WALLTIME overrides WALLTIME for the alloc.
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"
# If set (format DD:HH:MM:SS), training stops early to reserve time for a final
# checkpoint save before walltime. Unset to use the YAML's default and let
# slurm walltime end the job naturally — fine when each step checkpoints.
CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-}"

# =============================================================================
# Container & mounts
# =============================================================================
export CONTAINER
MOUNTS="${MOUNTS:-}"

# GB200 NVL72 defaults to 4 GPUs/node. Allow H100 smoke configs to request
# their native 8-GPU node shape through the launch environment.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# =============================================================================
# HuggingFace configuration
# =============================================================================
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"
else
  echo "[WARN] HF_HOME is not set — HuggingFace will use the default cache (~/.cache/huggingface) per-node." >&2
fi

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-nemotron-3.5-nano}"
WANDB_NAME="${EXP_NAME}"
WANDB_ENABLED=False
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED=True
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
  fi
  if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
    export WANDB_NAME WANDB_RESUME WANDB_RUN_GROUP WANDB_RUN_ID
  fi
else
  echo "[WARN] WANDB_API_KEY is not set — W&B logging will be disabled." >&2
fi

# =============================================================================
# Training overrides
# =============================================================================
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# =============================================================================
# MTP speculative decoding (optional)
# =============================================================================
ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8480}"
MTP_EXTRA_ARGS=""
if [[ "${ENABLE_MTP_INFERENCE}" == "1" ]]; then
  MTP_EXTRA_ARGS="\
++policy.generation.vllm_cfg.enable_prefix_caching=true \
++policy.generation.vllm_kwargs.enable_chunked_prefill=true \
++policy.generation.vllm_kwargs.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
++policy.generation.vllm_kwargs.mamba_cache_mode=align \
~policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.method=mtp"
  echo "MTP speculative decoding ENABLED (num_speculative_tokens=${NUM_SPECULATIVE_TOKENS})"
fi

# =============================================================================
# Job shape. Recipe-specific defaults are selected above and can be overridden
# through NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES.
# =============================================================================
NUM_EXTERNAL_SERVICE_NODES="${NUM_EXTERNAL_SERVICE_NODES:-0}"
COLOCATED_GENERATION="${COLOCATED_GENERATION:-0}"
SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
DEDICATED_RAY_HEAD="${DEDICATED_RAY_HEAD:-0}"

if (( NUM_TRAIN_NODES <= 0 )); then
  echo "ERROR: NUM_TRAIN_NODES must be > 0 (got ${NUM_TRAIN_NODES})" >&2; exit 1
fi
if (( NUM_GYM_NODES < 0 )); then
  echo "ERROR: NUM_GYM_NODES must be >= 0 (got ${NUM_GYM_NODES})" >&2; exit 1
fi
if (( NUM_EXTERNAL_SERVICE_NODES < 0 )); then
  echo "ERROR: NUM_EXTERNAL_SERVICE_NODES must be >= 0 (got ${NUM_EXTERNAL_SERVICE_NODES})" >&2; exit 1
fi

case "${COLOCATED_GENERATION}" in
  0)
    if (( NUM_GEN_NODES <= 0 )); then
      echo "ERROR: NUM_GEN_NODES must be > 0 (got ${NUM_GEN_NODES})" >&2
      echo "  NUM_GEN_NODES=0 is allowed only with the exact COLOCATED_GENERATION=1 single-node shape." >&2
      exit 1
    fi
    NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES))
    GENERATION_RESOURCE_NODES="${NUM_GEN_NODES}"
    COLOCATED_GENERATION_HYDRA_ARGS=""
    ;;
  1)
    if (( NUM_TRAIN_NODES != 1 || NUM_GEN_NODES != 0 || NUM_GYM_NODES != 0 || \
          NUM_EXTERNAL_SERVICE_NODES != 0 || SEGMENT_SIZE != 1 || GPUS_PER_NODE != 4 )); then
      echo "ERROR: COLOCATED_GENERATION=1 requires the exact one-node/four-GPU shape:" >&2
      echo "  NUM_TRAIN_NODES=1 NUM_GEN_NODES=0 NUM_GYM_NODES=0 NUM_EXTERNAL_SERVICE_NODES=0 SEGMENT_SIZE=1 GPUS_PER_NODE=4" >&2
      echo "  got train=${NUM_TRAIN_NODES} gen=${NUM_GEN_NODES} gym=${NUM_GYM_NODES} external=${NUM_EXTERNAL_SERVICE_NODES} segment=${SEGMENT_SIZE} gpus_per_node=${GPUS_PER_NODE}" >&2
      exit 1
    fi
    if [[ "${DEDICATED_RAY_HEAD}" != "0" ]]; then
      echo "ERROR: COLOCATED_GENERATION=1 requires DEDICATED_RAY_HEAD=0 (got ${DEDICATED_RAY_HEAD})" >&2
      exit 1
    fi

    # The policy and vLLM worker groups time-share the same four GPUs.  NeMo-RL
    # selects its CUDA-IPC weight synchronizer when colocation is enabled and
    # refit_transport is null.
    NUM_ACTOR_NODES=1
    GENERATION_RESOURCE_NODES=1
    export DEDICATED_RAY_HEAD=0
    COLOCATED_GENERATION_HYDRA_ARGS="\
cluster.gpus_per_node=4 \
policy.generation.backend=vllm \
policy.generation.colocated.enabled=true \
policy.generation.colocated.resources.gpus_per_node=4 \
++policy.generation.refit_transport=null"

    # These launcher-owned values are the safety boundary for the one-node
    # mode.  Refuse command-line Hydra attempts to replace a protected object
    # or field instead of relying on argument ordering.
    for override in "$@"; do
      override_key="${override%%=*}"
      override_key="${override_key#++}"
      override_key="${override_key#+}"
      override_key="${override_key#\~}"
      case "${override_key}" in
        cluster|cluster.num_nodes|cluster.gpus_per_node|cluster.segment_size|\
        policy.generation|policy.generation.backend|\
        policy.generation.colocated|\
        policy.generation.colocated.enabled|\
        policy.generation.colocated.resources|\
        policy.generation.colocated.resources.num_nodes|\
        policy.generation.colocated.resources.gpus_per_node|\
        policy.generation.refit_transport|env.nemo_gym.num_gpu_nodes)
          echo "ERROR: COLOCATED_GENERATION=1 forbids overriding launcher-owned Hydra key: ${override_key}" >&2
          exit 1
          ;;
      esac
    done
    unset override override_key
    ;;
  *)
    echo "ERROR: COLOCATED_GENERATION must be exactly 0 or 1 (got ${COLOCATED_GENERATION})" >&2
    exit 1
    ;;
esac

NUM_RAY_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))
NUM_TOTAL_NODES=$((NUM_RAY_NODES + NUM_EXTERNAL_SERVICE_NODES))

# GB200 NVL72 topology: validate the training and external-service components
# separately because Slurm schedules them as distinct heterogeneous groups.
GENRM_SEGMENT_SIZE="${GENRM_SEGMENT_SIZE:-${SEGMENT_SIZE}}"
if (( NUM_RAY_NODES < SEGMENT_SIZE )); then
  echo "ERROR: NUM_RAY_NODES=${NUM_RAY_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi
if (( NUM_RAY_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NeMo RL nodes=${NUM_RAY_NODES} is not divisible by SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} = ${NUM_RAY_NODES}" >&2
  exit 1
fi
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  if (( GENRM_SEGMENT_SIZE <= 0 )); then
    echo "ERROR: GENRM_SEGMENT_SIZE must be > 0." >&2
    exit 1
  fi
  if (( NUM_EXTERNAL_SERVICE_NODES % GENRM_SEGMENT_SIZE != 0 )); then
    echo "ERROR: External service nodes=${NUM_EXTERNAL_SERVICE_NODES} is not divisible by GENRM_SEGMENT_SIZE=${GENRM_SEGMENT_SIZE}." >&2
    exit 1
  fi
fi

# =============================================================================
# NeMo Skills sandbox (for math_formal_lean, ns_tools, etc.)
# =============================================================================
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# =============================================================================
# Ray log sync
# =============================================================================
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

CODE_ROOT="/opt/nemo-rl"
USE_CUSTOM_VLLM="${USE_CUSTOM_VLLM:-0}"
case "${USE_CUSTOM_VLLM}" in
  1)
    VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "
    ;;
  0)
    VLLM_ENV_SOURCE=""
    ;;
  *)
    echo "ERROR: USE_CUSTOM_VLLM must be 0 or 1, got: ${USE_CUSTOM_VLLM}" >&2
    exit 1
    ;;
esac

# =============================================================================
# Persistent cache directories
# =============================================================================
# Lustre holds the warm persistent cache. At job start, SETUP_COMMAND clears
# stale /tmp caches then seeds node-local /tmp from Lustre. JIT writes go to
# /tmp to avoid Lustre metadata contention from parallel compilation.
_vllm_cache_precision="bf16"
CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
LUSTRE_VLLM_CACHE="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_CUBIN_CACHE="/tmp/nemo_rl_flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-0}"

export LUSTRE_VLLM_CACHE
export LUSTRE_INDUCTOR_CACHE
export LUSTRE_TRITON_CACHE
export CACHE_READ_DIR
export CACHE_WRITE_DIR
export NRL_VLLM_LOCAL_CACHE_DIR
export INDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export CACHE_SYNC_FREQUENCY

if [[ "${STRICT_PAIR_PREPARE_SLURM_EXPORT}" != "1" ]]; then
  "${NANO35_MKDIR}" -p "${LUSTRE_FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
    "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" \
    "${CACHE_READ_DIR}" "${CACHE_WRITE_DIR}"
fi

# Read path  : cache_read/*.tar.zst   — compute nodes extract tarballs (hundreds of concurrent reads)
# Write path : cache_write/*/        — sidecar rsyncs individual files (one sequential writer)
# Splitting reads (tarball) from writes (directory) avoids Lustre MDT invalidation storms
# and lets rsync accumulate the union of all roles' kernels across jobs.
if [[ "${STRICT_PAIR_HOST_RUNTIME}" != "1" ]]; then
for _name in inductor_cache triton_cache; do
  _write_dir="${CACHE_WRITE_DIR}/${_name}"
  _old_dir="${PERSISTENT_CACHE}/${_name}"

  # One-time migration: move legacy dir → cache_write/ (instant rename, same FS)
  if ([ ! -d "$_write_dir" ] || [ -z "$(ls -A "$_write_dir" 2>/dev/null)" ]) \
     && [ -d "$_old_dir" ] && [ -n "$(ls -A "$_old_dir" 2>/dev/null)" ]; then
    [ -d "$_write_dir" ] && rmdir "$_write_dir" 2>/dev/null
    mv "$_old_dir" "$_write_dir" 2>/dev/null \
      && echo "[CACHE] Moved legacy ${_name}/ → cache_write/${_name}/" \
      || echo "[CACHE] Failed to move legacy ${_name}/"
  fi
done

# vLLM: migrate the most recent legacy seed dir → cache_write/ (one-time, instant rename)
_vllm_write="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
_vllm_read_tar="${CACHE_READ_DIR}/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"

if [ ! -d "$_vllm_write" ] || [ -z "$(ls -A "$_vllm_write" 2>/dev/null)" ]; then
  _best="$(ls -1dt \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"* \
    2>/dev/null \
    | while IFS= read -r d; do
        [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && echo "$d" && break
      done
  )" || true
  if [ -n "$_best" ]; then
    [ -d "$_vllm_write" ] && rmdir "$_vllm_write" 2>/dev/null || true
    mv "$_best" "$_vllm_write" 2>/dev/null \
      && echo "[CACHE] Moved $(basename "$_best") → cache_write/vllm_compile_cache_${_vllm_cache_precision}/" \
      || echo "[CACHE] Failed to move vLLM cache"
  fi
fi

# Purge redundant legacy vLLM cache directories.
# The old sidecar wrote every vLLM seed as a separate directory on Lustre
# (e.g. vllm_compile_cache_bf16_2058, _3072, ...). With cache_write/ + tarball,
# only cache_write/vllm_compile_cache_{precision}/ matters. All seed copies are
# content-addressed duplicates — safe to remove after migration.
_purge_count=0
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"*; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}"/vllm_compile_cache_[0-9]*/; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_warm"; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
if (( _purge_count > 0 )); then
  echo "[CACHE] Purged ${_purge_count} redundant legacy vLLM cache directories from ${PERSISTENT_CACHE}/"
fi
fi

# =============================================================================
# Code snapshot
# =============================================================================
# Snapshot the git-tracked source tree so the code is frozen at submission time.
# This guarantees we know exactly which code was used for a given experiment.
# Set USE_SNAPSHOT=0 to skip (runs from container built-in or live checkout).
# Interactive mode defaults to the live checkout for fast iteration; batch snapshots.
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

STRICT_PREBUILT_SNAPSHOT_DIR="${STRICT_PREBUILT_SNAPSHOT_DIR:-}"
STRICT_PREBUILT_SNAPSHOT_MANIFEST_NAME="strict-pair-snapshot-manifest.sha256"
STRICT_PREBUILT_SNAPSHOT="0"
OVERLAY_MOUNT_OPTIONS=""
if [[ -n "${STRICT_PREBUILT_SNAPSHOT_DIR}" ]]; then
  if [[ "${USE_SNAPSHOT}" != "1" ]]; then
    echo "ERROR: STRICT_PREBUILT_SNAPSHOT_DIR requires USE_SNAPSHOT=1." >&2
    exit 1
  fi
  if [[ "${STRICT_PREBUILT_SNAPSHOT_DIR}" != /* || \
        -L "${STRICT_PREBUILT_SNAPSHOT_DIR}" || \
        ! -d "${STRICT_PREBUILT_SNAPSHOT_DIR}" || \
        "$("${NANO35_REALPATH}" -- "${STRICT_PREBUILT_SNAPSHOT_DIR}")" != "${STRICT_PREBUILT_SNAPSHOT_DIR}" ]]; then
    echo "ERROR: strict prebuilt snapshot must be one canonical, non-symlink directory: ${STRICT_PREBUILT_SNAPSHOT_DIR}" >&2
    exit 1
  fi
  if [[ ! "${EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256 must be an explicit lowercase SHA-256." >&2
    exit 1
  fi
  STRICT_PREBUILT_SNAPSHOT_MANIFEST="${STRICT_PREBUILT_SNAPSHOT_DIR}/${STRICT_PREBUILT_SNAPSHOT_MANIFEST_NAME}"
  if [[ -L "${STRICT_PREBUILT_SNAPSHOT_MANIFEST}" || ! -f "${STRICT_PREBUILT_SNAPSHOT_MANIFEST}" ]]; then
    echo "ERROR: strict prebuilt snapshot manifest must be a regular, non-symlink file." >&2
    exit 1
  fi
  if [[ "$(sha256_file "${STRICT_PREBUILT_SNAPSHOT_MANIFEST}")" != \
        "${EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256}" ]]; then
    echo "ERROR: strict prebuilt snapshot manifest SHA-256 mismatch." >&2
    exit 1
  fi
  if ! (cd -- "${STRICT_PREBUILT_SNAPSHOT_DIR}" && \
        "${NANO35_SHA256SUM}" --check --strict --quiet -- "${STRICT_PREBUILT_SNAPSHOT_MANIFEST_NAME}"); then
    echo "ERROR: strict prebuilt snapshot content verification failed." >&2
    exit 1
  fi
  STRICT_PREBUILT_SNAPSHOT_SYMLINKS="${STRICT_PREBUILT_SNAPSHOT_DIR}/strict-pair-snapshot-symlinks.json"
  STRICT_PREBUILT_SNAPSHOT_MODES="${STRICT_PREBUILT_SNAPSHOT_DIR}/strict-pair-snapshot-modes.json"
  if [[ -L "${STRICT_PREBUILT_SNAPSHOT_SYMLINKS}" || \
        ! -f "${STRICT_PREBUILT_SNAPSHOT_SYMLINKS}" || \
        -L "${STRICT_PREBUILT_SNAPSHOT_MODES}" || \
        ! -f "${STRICT_PREBUILT_SNAPSHOT_MODES}" ]]; then
    echo "ERROR: strict prebuilt snapshot inventory manifests must be regular, non-symlink files." >&2
    exit 1
  fi
  if ! "${NANO35_PYTHON}" -I -B - \
    "${STRICT_PREBUILT_SNAPSHOT_DIR}" \
    "${STRICT_PREBUILT_SNAPSHOT_MANIFEST}" \
    "${STRICT_PREBUILT_SNAPSHOT_SYMLINKS}" \
    "${STRICT_PREBUILT_SNAPSHOT_MODES}" <<'PY'
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
sha_manifest = pathlib.Path(sys.argv[2])
symlink_manifest = pathlib.Path(sys.argv[3])
mode_manifest = pathlib.Path(sys.argv[4])
regular = set()
for line in sha_manifest.read_text(encoding="ascii").splitlines():
    _, separator, relative = line.partition("  ")
    if not separator or relative in regular:
        raise SystemExit("malformed or duplicate snapshot SHA manifest entry")
    regular.add(relative)
symlink_document = json.loads(symlink_manifest.read_text(encoding="ascii"))
if symlink_document.get("schema") != "nemo-rl-strict-snapshot-symlinks-v1":
    raise SystemExit("snapshot symlink manifest schema mismatch")
symlinks = symlink_document.get("symlinks")
if not isinstance(symlinks, dict) or any(
    not isinstance(key, str) or not isinstance(value, str)
    for key, value in symlinks.items()
):
    raise SystemExit("malformed snapshot symlink manifest")
mode_document = json.loads(mode_manifest.read_text(encoding="ascii"))
if mode_document.get("schema") != "nemo-rl-strict-snapshot-modes-v1":
    raise SystemExit("snapshot mode manifest schema mismatch")
regular_file_executable = mode_document.get("regular_file_executable")
if not isinstance(regular_file_executable, dict) or any(
    not isinstance(key, str) or not isinstance(value, bool)
    for key, value in regular_file_executable.items()
):
    raise SystemExit("malformed snapshot mode manifest")
actual_regular = set()
actual_symlinks = {}
actual_executable = {}


def raise_walk_error(error: OSError) -> None:
    raise error


for directory, directory_names, file_names in os.walk(
    root, onerror=raise_walk_error, followlinks=False
):
    directory_names.sort()
    file_names.sort()
    for name in list(directory_names):
        path = pathlib.Path(directory) / name
        if path.is_symlink():
            relative = path.relative_to(root).as_posix()
            actual_symlinks[relative] = os.readlink(path)
            directory_names.remove(name)
    for name in file_names:
        path = pathlib.Path(directory) / name
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            actual_symlinks[relative] = os.readlink(path)
        elif stat.S_ISREG(metadata.st_mode):
            if path != sha_manifest:
                actual_regular.add(relative)
                actual_executable[relative] = bool(metadata.st_mode & 0o111)
        else:
            raise SystemExit(f"snapshot contains special path: {relative}")
if actual_regular != regular:
    raise SystemExit("snapshot regular-file inventory differs from SHA manifest")
if actual_symlinks != symlinks:
    raise SystemExit("snapshot symlink inventory differs from symlink manifest")
if actual_executable != regular_file_executable:
    raise SystemExit("snapshot executable-mode inventory differs from mode manifest")
PY
  then
    echo "ERROR: strict prebuilt snapshot inventory verification failed." >&2
    exit 1
  fi
  writable_snapshot_path=""
  strict_snapshot_find_inventory="$(
    "${NANO35_MKTEMP}" "${RESULTS_DIR}/.strict-snapshot-find.XXXXXX"
  )"
  if ! "${NANO35_FIND}" "${STRICT_PREBUILT_SNAPSHOT_DIR}" \
      \( -type d -o -type f \) -print0 > "${strict_snapshot_find_inventory}"; then
    "${NANO35_RM}" -f -- "${strict_snapshot_find_inventory}"
    echo "ERROR: strict prebuilt snapshot inventory enumeration failed." >&2
    exit 1
  fi
  "${NANO35_CHMOD}" 400 "${strict_snapshot_find_inventory}"
  strict_snapshot_find_sha256="$(sha256_file "${strict_snapshot_find_inventory}")"
  while IFS= read -r -d '' snapshot_path; do
    if snapshot_mode="$("${NANO35_STAT}" -c '%a' -- "${snapshot_path}" 2>/dev/null)"; then
      :
    else
      snapshot_mode="$("${NANO35_STAT}" -f '%Lp' -- "${snapshot_path}")"
    fi
    if (( (8#${snapshot_mode} & 8#222) != 0 )); then
      writable_snapshot_path="${snapshot_path}"
      break
    fi
  done < "${strict_snapshot_find_inventory}"
  if [[ "$(sha256_file "${strict_snapshot_find_inventory}")" != \
        "${strict_snapshot_find_sha256}" ]]; then
    "${NANO35_RM}" -f -- "${strict_snapshot_find_inventory}"
    echo "ERROR: strict prebuilt snapshot inventory changed while consumed." >&2
    exit 1
  fi
  "${NANO35_RM}" -f -- "${strict_snapshot_find_inventory}"
  if [[ -n "${writable_snapshot_path}" ]]; then
    echo "ERROR: strict prebuilt snapshot contains a writable path: ${writable_snapshot_path}" >&2
    exit 1
  fi
  SNAPSHOT_DIR="${STRICT_PREBUILT_SNAPSHOT_DIR}"
  echo "Code snapshot: ${SNAPSHOT_DIR} (strict prebuilt, authenticated, read-only)"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
  STRICT_PREBUILT_SNAPSHOT="1"
  OVERLAY_MOUNT_OPTIONS=":ro"
elif [[ "${USE_SNAPSHOT}" == "1" ]]; then
  if [[ ! -f "${PROJECT_ROOT}/tools/code_snapshot.sh" ]]; then
    echo "ERROR: tools/code_snapshot.sh not found at ${PROJECT_ROOT}/tools/code_snapshot.sh" >&2
    echo "  Set USE_SNAPSHOT=0 to run from the live checkout instead." >&2
    exit 1
  fi
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}")

  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" ]] && [[ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    "${NANO35_MKDIR}" -p "${SNAPSHOT_DIR}/3rdparty"
    "${NANO35_LN}" -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

if [[ -n "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" && "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT="${SNAPSHOT_DIR}/${SINGLE_CONTROLLER_ENTRYPOINT_REL}"
  if [[ -L "${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT}" || ! -f "${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT}" ]]; then
    echo "ERROR: code snapshot is missing a regular, non-symlink SingleController entrypoint: ${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT}" >&2
    exit 1
  fi
  SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT_SHA256="$(sha256_file "${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT}")"
  if [[ "${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" != "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" ]]; then
    echo "ERROR: code-snapshot SingleController entrypoint SHA-256 mismatch: expected ${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}, got ${SNAPSHOT_SINGLE_CONTROLLER_ENTRYPOINT_SHA256}." >&2
    exit 1
  fi
  SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST="${SNAPSHOT_DIR}/nano35-entrypoint-manifest.sha256"
  if [[ -L "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}" || ( -e "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}" && ! -f "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}" ) ]]; then
    echo "ERROR: snapshot entrypoint manifest must be a regular, non-symlink file: ${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}" >&2
    exit 1
  fi
  if [[ "${STRICT_PREBUILT_SNAPSHOT}" == "1" ]]; then
    if [[ ! -f "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}" || \
          "$(< "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}")" != \
          "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}  ${SINGLE_CONTROLLER_ENTRYPOINT_REL}" ]]; then
      echo "ERROR: strict prebuilt snapshot entrypoint manifest is missing or differs from authenticated source." >&2
      exit 1
    fi
  else
    printf '%s  %s\n' \
      "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" \
      "${SINGLE_CONTROLLER_ENTRYPOINT_REL}" \
      > "${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}"
  fi
  echo "SingleController entrypoint manifest: ${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST}"
fi

# =============================================================================
# Container mounts
# =============================================================================
# By default, nemo_rl and the selected recipe directory from the code snapshot
# are overlaid into the container. Everything else uses the container's built-in
# code at /opt/nemo-rl.
#
# To overlay additional components (e.g. a local Megatron-LM checkout), pass
# EXTRA_MOUNTS as a comma-separated list of host:container pairs:
#
#   EXTRA_MOUNTS="/path/to/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" bash nano35_launch.sh swe
#
# Container paths for reference:
#   /opt/nemo-rl/nemo_rl                                              — Python package
#   /opt/nemo-rl/examples/configs                                     — YAML configs
#   /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM           — Megatron-LM
#   /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   — Megatron-Bridge
#   /opt/nemo-rl/3rdparty/Gym-workspace/Gym                           — NeMo-Gym
#   /opt/nemo-rl/3rdparty/vllm                                        — vLLM
# =============================================================================
_append_mount() {
  if [[ -z "${MOUNTS}" ]]; then
    MOUNTS="$1"
  else
    MOUNTS="${MOUNTS},$1"
  fi
}

if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  _append_mount "${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl${OVERLAY_MOUNT_OPTIONS}"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  _append_mount "${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs${OVERLAY_MOUNT_OPTIONS}"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/nemo_gym/nemotron-3.5-nano" ]]; then
  _append_mount "${OVERLAY_SOURCE}/examples/nemo_gym/nemotron-3.5-nano:/opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano${OVERLAY_MOUNT_OPTIONS}"
  echo "  Mount: Nano 3.5 recipes → /opt/nemo-rl/examples/nemo_gym/nemotron-3.5-nano"
fi
if [[ -n "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" ]]; then
  _append_mount "${OVERLAY_SOURCE}/${SINGLE_CONTROLLER_ENTRYPOINT_REL}:/opt/nemo-rl/${SINGLE_CONTROLLER_ENTRYPOINT_REL}:ro"
  echo "  Mount: SingleController entrypoint → /opt/nemo-rl/${SINGLE_CONTROLLER_ENTRYPOINT_REL} (read-only, sha256=${SINGLE_CONTROLLER_ENTRYPOINT_SHA256})"
fi
if [[ "${STRICT_PREBUILT_SNAPSHOT}" == "1" && \
      ( -L "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" || \
        ! -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ) ]]; then
  echo "ERROR: strict prebuilt snapshot is missing its authenticated Reasoning Gym gitlink contents." >&2
  exit 1
fi
if [[ -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ]]; then
  _append_mount "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym${OVERLAY_MOUNT_OPTIONS}"
  echo "  Mount: Gym → /opt/nemo-rl/3rdparty/Gym-workspace/Gym"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  _append_mount "${SNAPSHOT_DIR}:${SNAPSHOT_DIR}${OVERLAY_MOUNT_OPTIONS}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  _append_mount "${EXTRA_MOUNTS}"
  echo "  Extra mounts: ${EXTRA_MOUNTS}"
fi

export MOUNTS

# =============================================================================
# Resolve ray.sub
# =============================================================================
RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2
  exit 1
fi
BATCH_SCRIPT="${BATCH_SCRIPT:-${RAY_SUB}}"
if [[ ! -f "${BATCH_SCRIPT}" ]]; then
  echo "ERROR: batch script not found at ${BATCH_SCRIPT}" >&2
  exit 1
fi
if [[ "${STRICT_PREBUILT_SNAPSHOT}" == "1" && \
      ( "${RAY_SUB}" != "${BATCH_SCRIPT}" || \
        "${BATCH_SCRIPT}" != "${STRICT_PAIR_JOB_WRAPPER:?strict prebuilt mode requires STRICT_PAIR_JOB_WRAPPER}" ) ]]; then
  echo "ERROR: strict prebuilt mode requires RAY_SUB and BATCH_SCRIPT to be the authenticated job wrapper." >&2
  exit 1
fi
export RAY_SUB

# =============================================================================
# Per-node cache setup (SETUP_COMMAND)
# =============================================================================
# Triton, Inductor, and FlashInfer cubins compile/download to node-local /tmp to
# avoid Lustre race conditions and file lock contention during concurrent JIT
# compilation. Ordinary recipes may seed /tmp from a warm Lustre cache. Strict
# pairs deliberately prohibit seed reads and start from cleared local caches.
#
# IMPORTANT: Stale /tmp caches from previous jobs can cause hangs (e.g. the
# Triton bundler skipping non-empty temp dirs). Every mode clears /tmp first;
# only non-strict mode may then seed from Lustre.
# =============================================================================
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
read -r -d '' SETUP_COMMAND <<'SETUPEOF' || true
echo "[CACHE SETUP] Clearing stale node-local caches; strict pairs forbid cache seeds."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_* \
  /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache
mkdir -p /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache
echo "[CACHE SETUP] Done."
SETUPEOF
else
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
command -v zstd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq zstd; } 2>/dev/null || true
echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
WARM_SEED="${NRL_VLLM_CACHE_SEED_DIR}"
LOCAL_IND="${INDUCTOR_CACHE_DIR}"
LOCAL_TRI="${TRITON_CACHE_DIR}"
CACHE_READ="${CACHE_READ_DIR}"

# vLLM caches are per-instance (VLLM_CACHE_ROOT_{seed}). Clear ALL from prior jobs.
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "\$LOCAL_IND" "\$LOCAL_TRI"
mkdir -p "\$LOCAL_IND" "\$LOCAL_TRI"

_seed_cache() {
  local tarball="\$1" local_dir="\$2" name="\$3"
  if [ -f "\$tarball" ]; then
    tar --zstd -xf "\$tarball" -C "\$local_dir" \
      && echo "[CACHE SEED] \$name: seeded from tarball (\$(du -sh "\$local_dir" 2>/dev/null | cut -f1))" \
      || echo "[CACHE SEED] \$name: tarball extract failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

# Seed vLLM compile cache from cache_read/ tarball (one per precision).
rm -rf "\$WARM_SEED"
_vllm_tar="\$CACHE_READ/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"
if [ -f "\$_vllm_tar" ]; then
  mkdir -p "\$WARM_SEED"
  tar --zstd -xf "\$_vllm_tar" -C "\$WARM_SEED" \
    && echo "[CACHE SEED] vLLM (${_vllm_cache_precision}): seeded from tarball (\$(du -sh "\$WARM_SEED" 2>/dev/null | cut -f1))" \
    || echo "[CACHE SEED] vLLM: tarball extract failed (non-fatal)"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
fi

_seed_cache "\$CACHE_READ/inductor_cache.tar.zst" "\$LOCAL_IND" "Inductor"
_seed_cache "\$CACHE_READ/triton_cache.tar.zst" "\$LOCAL_TRI" "Triton"

echo "[CACHE SEED] Done."
SETUPEOF
fi
export SETUP_COMMAND

# =============================================================================
# Build the training command
# =============================================================================
# Stage-specific hyperparameters (batch sizes, advantage clip, MoE parallelism,
# learning rate, etc.) live in CONFIG_PATH. The launcher only passes the
# per-run overrides: cluster shape, paths, judge endpoints, logging.
# =============================================================================
UV_RUNNER=uv
UV_CACHE_JOB_ID_EXPR="\${SLURM_JOB_ID:-default}"
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  UV_RUNNER="${STRICT_PAIR_UV_SHIM}"
  # The external job wrapper authenticates the live scheduler ID, exports it
  # as STRICT_PAIR_BOUND_JOB_ID, and deliberately scrubs SLURM_JOB_ID before
  # the driver.  Keep the expansion deferred until that sealed job boundary;
  # a fallback would alias every strict job to one shared /tmp cache.
  UV_CACHE_JOB_ID_EXPR="\${STRICT_PAIR_BOUND_JOB_ID}"
fi
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
WANDB_INIT_TIMEOUT=300 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=/tmp/nemo-gym-uv-cache-${UV_CACHE_JOB_ID_EXPR} \
UV_LOCK_TIMEOUT=1800 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME:-} \
HF_TOKEN=\${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
${UV_RUNNER} run ${TRAIN_ENTRYPOINT} \
--config ${CONFIG_PATH} \
policy.model_name=${MODEL_PATH} \
cluster.num_nodes=${NUM_ACTOR_NODES} \
cluster.segment_size=${SEGMENT_SIZE} \
policy.generation.colocated.resources.num_nodes=${GENERATION_RESOURCE_NODES} \
${COLOCATED_GENERATION_HYDRA_ARGS} \
env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
${CHECKPOINTING_SAVE_BY:+checkpointing.checkpoint_must_save_by=${CHECKPOINTING_SAVE_BY}} \
data.train.data_path=${TRAIN_PATH} \
data.validation.data_path=${VAL_PATH} \
${GENRM_OVERRIDE:+${GENRM_OVERRIDE}} \
${NL2BASH_JUDGE_MODEL:+env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NL2BASH_JUDGE_MODEL}} \
${SAFETY_JUDGE_MODEL:+env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${SAFETY_JUDGE_MODEL}} \
${SIF_DIR:+sif_dir=${SIF_DIR}} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${MTP_EXTRA_ARGS} \
${*}"

export COMMAND="${TRAIN_CMD}"

# The authoritative parent renders both complete Slurm payloads before it
# publishes PAIR_MANIFEST.json. A later launch re-renders and byte-compares the
# payload, so COMMAND/mount/cache drift cannot hide behind a matching name list.
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  : "${STRICT_PAIR_SLURM_EXPORT_FILE:?strict prebuilt mode requires its arm export-file path}"
  HF_TOKEN="${HF_TOKEN:-}"
  if [[ "${STRICT_PAIR_PREPARE_SLURM_EXPORT}" == "1" ]]; then
    if [[ -n "${EXPECTED_STRICT_PAIR_SLURM_EXPORT_SHA256:-}" ]]; then
      echo "ERROR: pre-Pair export rendering cannot accept an expected payload SHA-256." >&2
      exit 2
    fi
    strict_pair_publish_or_verify_slurm_export \
      "${STRICT_PAIR_SLURM_EXPORT_FILE}"
    echo "STRICT_PAIR_SLURM_EXPORT_PREPARED arm=${STRICT_PAIR_ARM} schema=nemo-rl-strict-slurm-export-file-v2 sha256=${STRICT_PAIR_ACTIVE_SLURM_EXPORT_SHA256} names=${#STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES[@]}"
    exit 0
  fi
  if [[ ! "${EXPECTED_STRICT_PAIR_SLURM_EXPORT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: strict launch requires its parent-bound Slurm export SHA-256." >&2
    exit 2
  fi
  strict_pair_publish_or_verify_slurm_export \
    "${STRICT_PAIR_SLURM_EXPORT_FILE}" \
    "${EXPECTED_STRICT_PAIR_SLURM_EXPORT_SHA256}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  Nemotron 3.5 Nano — ${EXP_NAME} (${NUM_TOTAL_NODES}-node)"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Config:      ${CONFIG_PATH}"
echo "  Nodes:       ${NUM_TOTAL_NODES} total"
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
echo "    Hetgroup 0: ${NUM_RAY_NODES} NeMo RL nodes  (segment=${SEGMENT_SIZE})"
fi
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
if [[ "${COLOCATED_GENERATION}" == "1" ]]; then
echo "    vLLM gen:  colocated on the training node (1 node, ${GPUS_PER_NODE} shared GPUs, CUDA-IPC refit)"
echo "    Ray head:  shared on the GPU node (DEDICATED_RAY_HEAD=0)"
else
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
fi
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
echo "    Hetgroup 1: ${NUM_EXTERNAL_SERVICE_NODES} external GenRM nodes  (segment=${GENRM_SEGMENT_SIZE})"
echo "      GenRM:    ${GENRM_REPLICAS} independent TP=${GENRM_TENSOR_PARALLEL_SIZE}, DP=1 servers; LB port=${GENRM_LB_PORT}"
fi
echo "  Walltime:    ${WALLTIME}"
echo "  Batch script: ${BATCH_SCRIPT}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}  (stable — auto-resumes across jobs)"
echo "  Run dir:     ${RUN_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Slurm logs:  ${SLURM_LOG_DIR}"
echo "  W&B:         ${WANDB_PROJ} / ${WANDB_NAME} (enabled=${WANDB_ENABLED})"
echo ""
echo "  Model:       ${MODEL_PATH}"
echo "  Train data:  ${TRAIN_PATH}"
echo "  Val data:    ${VAL_PATH}"
echo "  Container:   ${CONTAINER}"
echo "  Custom vLLM: ${USE_CUSTOM_VLLM}"
echo "  Sandbox:     ${SANDBOX_CONTAINER}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
echo "  Snapshot:    ${SNAPSHOT_DIR}"
fi
echo ""
echo "  Monitor:  squeue -u \$USER -n ${JOB_NAME}"
echo "  Logs:     tail -f ${SLURM_LOG_DIR}/*.out"
echo "  Latest:   ls -la ${RESULTS_DIR}/runs/latest"
echo ""
echo "================================================================"
echo ""

# =============================================================================
# Record code provenance in the run directory
# =============================================================================
{
  echo "timestamp: $("${NANO35_DATE}" -Iseconds)"
  if [[ "${STRICT_PREBUILT_SNAPSHOT}" == "1" ]]; then
    echo "branch: authenticated-parent-snapshot"
    echo "commit: recorded-in-pair-manifest"
    echo "dirty: false (exact snapshot inventory authenticated)"
    echo "strict_pair_manifest: ${STRICT_PAIR_MANIFEST_PATH:?strict prebuilt mode requires STRICT_PAIR_MANIFEST_PATH}"
    echo "strict_prebuilt_snapshot_manifest_sha256: ${EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256}"
    echo "strict_pair_job_wrapper: ${BATCH_SCRIPT}"
    echo "strict_pair_job_wrapper_sha256: ${EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256:?strict prebuilt mode requires job-wrapper SHA-256}"
  else
    echo "branch: $(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty: $(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | head -20)"
  fi
  echo "snapshot: ${USE_SNAPSHOT}"
  if [[ "${USE_SNAPSHOT}" == "1" ]]; then
    echo "snapshot_dir: ${SNAPSHOT_DIR}"
  fi
  if [[ -n "${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}" ]]; then
    echo "single_controller_entrypoint_sha256: ${SINGLE_CONTROLLER_ENTRYPOINT_SHA256}"
    echo "single_controller_entrypoint_manifest: ${SINGLE_CONTROLLER_ENTRYPOINT_MANIFEST:-live-tree-no-manifest}"
  fi
  echo "container: ${CONTAINER}"
  echo "config: ${CONFIG_PATH}"
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

# =============================================================================
# Dry-run mode: print everything, don't submit
# =============================================================================
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — printing TRAIN_CMD and exiting without submission."
  echo ""
  echo "--- TRAIN_CMD ---"
  echo "${TRAIN_CMD}"
  echo "--- end ---"
  exit 0
fi

# =============================================================================
# Interactive mode: bring up Ray and idle for attachment (no training driver)
# =============================================================================
# With COMMAND empty, ray.sub starts the Ray cluster, writes <jobid>-attach.sh,
# then idles. We save the driver command to <jobid>-run-cmd.sh so you can attach
# and run it by hand, edit it, and re-run without requeueing.
if [[ "${INTERACTIVE}" == "1" ]]; then
  if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
    echo "ERROR: INTERACTIVE=1 is not supported with external service nodes." >&2
    echo "  Use DRY_RUN=1 to inspect the command or submit the batch job normally." >&2
    exit 1
  fi
  unset COMMAND 2>/dev/null || true   # empty COMMAND -> ray.sub idle/interactive mode
  WALLTIME="${INTERACTIVE_WALLTIME:-${WALLTIME}}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE — ${NUM_TOTAL_NODES}-node allocation (walltime ${WALLTIME})"
  echo "  Ray will start and idle until you attach."
  echo "================================================================"

  SBATCH_OUTPUT=$("${NANO35_SBATCH}" \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    ${SLURM_COMMENT_ARGS[@]+"${SLURM_COMMENT_ARGS[@]}"} \
    "${RAY_SUB}")
  echo "${SBATCH_OUTPUT}"
  JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
  [[ -z "${JOB_ID}" ]] && { echo "ERROR: could not parse job ID from sbatch output." >&2; exit 1; }

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"
  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  echo ""
  echo "  Driver command saved to:  ${CMD_FILE}"
  echo "  When Ray is up:"
  echo "    bash ${ATTACH_SCRIPT}                          # shell on the head node (Ray already up)"
  echo "    source ${CMD_FILE}                             # run the recipe inside that shell"
  echo "    # or non-interactively: COMMAND=\"\$(cat ${CMD_FILE})\" bash ${ATTACH_SCRIPT}"
  echo "  Edit ${CMD_FILE} and re-source to iterate without requeueing.  Cancel: scancel ${JOB_ID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Waiting for Ray (Ctrl+C to stop waiting; the job keeps running)..."
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      [[ -z "${state}" ]] && { echo "  Job ${JOB_ID} left the queue. Check: sacct -j ${JOB_ID}"; exit 1; }
      [[ "${state}" != "${prev_state}" ]] && { echo "  [$(date +%H:%M:%S)] state: ${state}"; prev_state="${state}"; }
      sleep 15
    done
    echo ""
    echo "  Ray is ready — attach: bash ${ATTACH_SCRIPT}"
  fi
  exit 0
fi

# =============================================================================
# Submit
# =============================================================================
# Always serialise same-name submissions via singleton; optionally chain after
# another job with SLURM_DEPENDENCY (e.g. "afterany:3044848" or "afterok:JOBID").
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEPENDENCY="singleton"
[[ -n "${SLURM_DEPENDENCY}" ]] && DEPENDENCY="singleton,${SLURM_DEPENDENCY}"

SBATCH_SCRIPT_ARGS=("${BATCH_SCRIPT}")
SBATCH_PARSABLE_ARGS=()
SBATCH_HOLD_ARGS=()
SBATCH_CHDIR_ARGS=()
SBATCH_CLIENT=("${NANO35_SBATCH}")
if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  strict_pair_verify_slurm_export_before_sbatch \
    "${STRICT_PAIR_SLURM_EXPORT_FILE}" \
    "${EXPECTED_STRICT_PAIR_SLURM_EXPORT_SHA256}"
  SBATCH_SCRIPT_ARGS=(
    "--export-file=${STRICT_PAIR_SLURM_EXPORT_FILE}"
    "${BATCH_SCRIPT}"
    --pair-manifest "${STRICT_PAIR_MANIFEST_PATH}"
    --pair-manifest-sha256 "${EXPECTED_PAIR_MANIFEST_SHA256}"
    --arm "${STRICT_PAIR_ARM}"
  )
  SBATCH_PARSABLE_ARGS=(--parsable)
  SBATCH_HOLD_ARGS=(--hold)
  SBATCH_CHDIR_ARGS=("--chdir=${PROJECT_ROOT}")
  SBATCH_CLIENT=(
    "${NANO35_ENV}" -i
    LC_ALL=C
    "SLURM_CONF=${STRICT_PAIR_SLURM_CONF}"
    "${NANO35_SBATCH}"
  )
fi

nano35_invoke_sbatch() {
  if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
    exec "${SBATCH_CLIENT[@]}" \
    ${SBATCH_PARSABLE_ARGS[@]+"${SBATCH_PARSABLE_ARGS[@]}"} \
    ${SBATCH_HOLD_ARGS[@]+"${SBATCH_HOLD_ARGS[@]}"} \
    ${SBATCH_CHDIR_ARGS[@]+"${SBATCH_CHDIR_ARGS[@]}"} \
    --nodes="${NUM_RAY_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --dependency="${DEPENDENCY}" \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    ${SLURM_COMMENT_ARGS[@]+"${SLURM_COMMENT_ARGS[@]}"} \
    : \
    --nodes="${NUM_EXTERNAL_SERVICE_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}-genrm" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --segment="${GENRM_SEGMENT_SIZE}" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    "${SBATCH_SCRIPT_ARGS[@]}"
  else
    exec "${SBATCH_CLIENT[@]}" \
    ${SBATCH_PARSABLE_ARGS[@]+"${SBATCH_PARSABLE_ARGS[@]}"} \
    ${SBATCH_HOLD_ARGS[@]+"${SBATCH_HOLD_ARGS[@]}"} \
    ${SBATCH_CHDIR_ARGS[@]+"${SBATCH_CHDIR_ARGS[@]}"} \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --dependency="${DEPENDENCY}" \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    ${SLURM_COMMENT_ARGS[@]+"${SLURM_COMMENT_ARGS[@]}"} \
    "${SBATCH_SCRIPT_ARGS[@]}"
  fi
}

if [[ "${STRICT_PAIR_HOST_RUNTIME}" == "1" ]]; then
  : "${STRICT_PAIR_ACCEPTED_ID_FD:?strict accepted-ID descriptor is required}"
  : "${STRICT_PAIR_ACCEPTED_ID_RECORD:?strict accepted-ID record path is required}"
  if [[ "${STRICT_PAIR_ARM}" == "off" ]]; then
    STRICT_PAIR_EXPECTED_ACCEPTED_ID_FD=8
  else
    STRICT_PAIR_EXPECTED_ACCEPTED_ID_FD=9
  fi
  STRICT_PAIR_EXPECTED_ACCEPTED_ID_RECORD="${STRICT_PAIR_MANIFEST_PATH%/*}/strict_pair_submission_state/${PAIR_ID}/${STRICT_PAIR_ARM}.job-id"
  if [[ "${STRICT_PAIR_ACCEPTED_ID_FD}" != \
        "${STRICT_PAIR_EXPECTED_ACCEPTED_ID_FD}" || \
        "${STRICT_PAIR_ACCEPTED_ID_RECORD}" != \
        "${STRICT_PAIR_EXPECTED_ACCEPTED_ID_RECORD}" ]]; then
    echo "ERROR: strict accepted-ID descriptor/path differs from the parent contract." >&2
    exit 2
  fi
  SBATCH_STATUS=0
  STRICT_PAIR_DEFERRED_SIGNAL_STATUS=0
  strict_pair_relay_sbatch_output() {
    "${NANO35_PYTHON}" -I -B -c '
import os
import signal
import sys

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
destination_fd = int(sys.argv[1])
while True:
    chunk = os.read(0, 1024 * 1024)
    if not chunk:
        break
    view = memoryview(chunk)
    while view:
        written = os.write(destination_fd, view)
        if written <= 0:
            raise SystemExit("accepted-ID relay made no write progress")
        view = view[written:]
os.fsync(destination_fd)
' "${STRICT_PAIR_ACCEPTED_ID_FD}"
  }
  strict_pair_defer_scheduler_signal() {
    local signal_status="$1"

    if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" == "0" ]]; then
      STRICT_PAIR_DEFERRED_SIGNAL_STATUS="${signal_status}"
    fi
  }
  trap 'strict_pair_defer_scheduler_signal 130' INT
  trap 'strict_pair_defer_scheduler_signal 143' TERM
  STRICT_PAIR_SBATCH_STARTED_UNIX_NS="$(
    "${NANO35_PYTHON}" -I -B -c 'import time; print(time.time_ns())'
  )"
  set +e
  if [[ "${STRICT_PAIR_ARM}" == "off" ]]; then
    (trap '' INT TERM; nano35_invoke_sbatch 8>&-) | \
      (trap '' INT TERM; strict_pair_relay_sbatch_output)
    STRICT_PAIR_PIPELINE_STATUS=("${PIPESTATUS[@]}")
  else
    (trap '' INT TERM; nano35_invoke_sbatch 9>&-) | \
      (trap '' INT TERM; strict_pair_relay_sbatch_output)
    STRICT_PAIR_PIPELINE_STATUS=("${PIPESTATUS[@]}")
  fi
  set -e
  SBATCH_STATUS="${STRICT_PAIR_PIPELINE_STATUS[0]:-125}"
  STRICT_PAIR_RELAY_STATUS="${STRICT_PAIR_PIPELINE_STATUS[1]:-125}"
  STRICT_PAIR_WRITER_DRAINED_UNIX_NS="$(
    "${NANO35_PYTHON}" -I -B -c 'import time; print(time.time_ns())'
  )"
  trap - INT TERM
  if (( STRICT_PAIR_RELAY_STATUS != 0 )); then
    echo "ERROR: strict sbatch accepted-ID relay failed." >&2
    exit "${STRICT_PAIR_RELAY_STATUS}"
  fi
  if ! JOB_ID="$(
    "${NANO35_PYTHON}" -I -B - \
      "${STRICT_PAIR_ACCEPTED_ID_RECORD}" \
      "${STRICT_PAIR_ACCEPTED_ID_FD}" <<'PY'
import os
import pathlib
import re
import stat
import sys

path = pathlib.Path(sys.argv[1])
fd = int(sys.argv[2])
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
    raise SystemExit("strict accepted-ID record inode changed before sealing")
os.fsync(fd)
read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
read_fd = os.open(path, read_flags)
try:
    read_stat = os.fstat(read_fd)
    if (read_stat.st_dev, read_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
        raise SystemExit("strict accepted-ID record path no longer names its open inode")
    chunks = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
finally:
    os.close(read_fd)
os.fchmod(fd, 0o400)
os.fsync(fd)
directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
payload = b"".join(chunks)
if re.fullmatch(rb"[1-9][0-9]*\n", payload) is None:
    raise SystemExit("strict sbatch --parsable output must be one numeric job ID")
print(payload[:-1].decode("ascii"))
PY
  )"; then
    echo "ERROR: strict sbatch accepted-ID record is invalid or could not be sealed." >&2
    exit 2
  fi
  if [[ "${STRICT_PAIR_ARM}" == "off" ]]; then
    exec 8>&-
  else
    exec 9>&-
  fi
  SBATCH_OUTPUT_SHA256="$(strict_pair_sha256_text "${JOB_ID}")"
  ACCEPTED_ID_RECORD_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_ACCEPTED_ID_RECORD}")"
  echo "STRICT_PAIR_HELD arm=${STRICT_PAIR_ARM} job_id=${JOB_ID} job_id_sha256_ascii_no_newline=${SBATCH_OUTPUT_SHA256} accepted_id_record_sha256=${ACCEPTED_ID_RECORD_SHA256} sbatch_status=${SBATCH_STATUS} relay_status=${STRICT_PAIR_RELAY_STATUS} writer_drained=true started_unix_ns=${STRICT_PAIR_SBATCH_STARTED_UNIX_NS} drained_unix_ns=${STRICT_PAIR_WRITER_DRAINED_UNIX_NS}"
  if [[ "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}" != "0" ]]; then
    exit "${STRICT_PAIR_DEFERRED_SIGNAL_STATUS}"
  fi
  if (( SBATCH_STATUS != 0 )); then
    echo "ERROR: strict sbatch failed after writing an accepted-ID record." >&2
    exit "${SBATCH_STATUS}"
  fi
else
  SBATCH_OUTPUT="$(nano35_invoke_sbatch)"
  echo "${SBATCH_OUTPUT}"
  if ! JOB_ID=$("${NANO35_GREP}" -oE '[0-9]+$' <<< "${SBATCH_OUTPUT}"); then
    JOB_ID=""
  fi
fi

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
