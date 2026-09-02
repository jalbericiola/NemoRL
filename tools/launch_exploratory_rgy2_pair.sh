#!/bin/bash
# Submit a bounded one-node-per-arm Reasoning-Gym OFF/ON GPU smoke.
#
# This launcher is deliberately NON-ACCEPTANCE evidence.  It exists to expose
# integration, generation, reward, MTP, and backward failures before the
# strict receipt/evaluator package is released.
set -euo pipefail

readonly HOST_PATH="/cm/local/apps/python3/bin:/cm/local/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
readonly SLURM_CONF_PATH="/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf"
readonly BASE_DEPLOYMENT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/deployments/validated_shared_prefix_20260830q"
readonly BASE_READY="441548f85b9779788d458a0d4deeefcca789ed8b46810a1f6a929492cd27d4cf"
readonly SOURCE_DEPLOYMENT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/deployments/staged_shared_prefix_deterministic_backward_6b37b731_ea61731f_133d5970"
readonly SOURCE_READY="c404bd37ea661a84078f903d78e5426d994324a0bae57ce88b1b6b0495d4533f"
readonly SOURCE_READY_FILE_SHA256="0a0ae2bff5706576453c68b9a34d915386ceb8883c779c618d10018a790286a9"
readonly SOURCE_DEPLOYMENT_MANIFEST_SHA256="c404bd37ea661a84078f903d78e5426d994324a0bae57ce88b1b6b0495d4533f"
readonly NEMO_RUNNABLE_MANIFEST_SHA256="a54cbcc612add5c37b3756eeb847c2d4ae2dc2a876b8b43348559d5fda49c563"
readonly BRIDGE_RUNNABLE_MANIFEST_SHA256="951cd5ca4b97a5b23bd9b1040336cf57f7acc364f82f063876d2d5b5ad8368c8"
readonly MCORE_RUNNABLE_MANIFEST_SHA256="c4197483d43cfc3fc48283d26a5525e13673200f88ec8705128fc9d031932ea7"

readonly BASE_RUNNABLE="${BASE_DEPLOYMENT}/runnable"
readonly SOURCE_RUNNABLE="${SOURCE_DEPLOYMENT}/runnable"
readonly NEMO_ROOT="${SOURCE_RUNNABLE}/NemoRL"
readonly BRIDGE_ROOT="${SOURCE_RUNNABLE}/Megatron-Bridge"
readonly MCORE_ROOT="${SOURCE_RUNNABLE}/Megatron-LM"
readonly GYM_ROOT="${BASE_RUNNABLE}/NemoRL/3rdparty/Gym-workspace/Gym"
readonly AUTOMODEL_ROOT="${BASE_RUNNABLE}/NemoRL/3rdparty/Automodel-workspace/Automodel"
readonly CONFIG_PATH="examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml"
readonly POLICY_GYM_CONFIG_REL="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
readonly REASONING_GYM_CONFIG_REL="resources_servers/reasoning_gym/configs/reasoning_gym.yaml"
readonly TRAIN_ENTRYPOINT="./examples/run_grpo_single_controller.py"
readonly NANO_LAUNCHER="${NEMO_ROOT}/examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"
readonly RAY_SUB_PATH="${NEMO_ROOT}/ray.sub"
readonly FIXTURE_PATH="${BASE_RUNNABLE}/single_env_ab/data/reasoning_gym_example.jsonl"
readonly MODEL_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/venkats/nemo-evaluator-rundirs/nano_v35_sft/conversions/upsampled-iter6000/hf"
readonly CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/sauramishra/containers/rl-gym.63635108.sqsh"
readonly SANDBOX_CONTAINER="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/mopd_nano_fast/images/nemo-skills-sandbox-no-sync.sqsh"
readonly STATE_ROOT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/exploratory_rgy2_133d_NON_ACCEPTANCE"

readonly EXPECTED_NEMO_HEAD="133d59702416737ff0ea80e62ea82e879555c6dd"
readonly EXPECTED_NEMO_TREE="77673342e6b810d42f2959f24e5740cd79334448"
readonly EXPECTED_LAUNCHER_SHA256="85625d987601a6fae3d61513cc01577e6ad33e4211b5cade92a59d4316804f3d"
readonly EXPECTED_CONFIG_SHA256="a15ce1dfdff8a6e3450e98c94201e8768fe5e0c39a2c210ad8e8d471880dc579"
readonly EXPECTED_ENTRYPOINT_SHA256="45f99fbcde57b6265648c3197e767cdcebf20e67d28c08f43128a8d79570dedb"
readonly EXPECTED_RAY_SUB_SHA256="f26b378656450c9fc6f498c4b0083981693572f2a9b1e7a17050c4f729384e82"
readonly EXPECTED_FIXTURE_SHA256="da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"
readonly ACCOUNT="nemotron_sw_post"
readonly PARTITION="batch"
readonly QOS="normal"
readonly WALLTIME="04:00:00"

usage() {
  printf 'Usage: %s <--check|--submit>\n' "${0##*/}" >&2
}

if (( $# != 1 )); then
  usage
  exit 2
fi
readonly launch_mode="$1"
case "${launch_mode}" in
  --check|--submit) ;;
  *) usage; exit 2 ;;
esac

sha256_file() {
  /usr/bin/sha256sum -- "$1" | /usr/bin/awk '{print $1}'
}

assert_exact_reasoning_gym_services() {
  local recipe_path="$1"
  local config_paths
  local embedded_services
  local resolved_services
  local forbidden_service

  /usr/bin/awk '
    /^env:$/ { in_env = 1; next }
    in_env && /^[^[:space:]#]/ { exit }
    in_env && /^  _override_: true[[:space:]]*$/ { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${recipe_path}" || {
    printf 'ERROR: Reasoning-Gym recipe must replace the inherited env map\n' >&2
    return 2
  }

  for forbidden_service in policy_model_reasoning_off safety_judge_model \
    nl2bash_judge_model genrm_model; do
    if /usr/bin/awk -v key="${forbidden_service}" \
      '$0 == "    " key ":" { found = 1 } END { exit(found ? 0 : 1) }' \
      "${recipe_path}"; then
      printf 'ERROR: Reasoning-Gym recipe embeds forbidden service: %s\n' \
        "${forbidden_service}" >&2
      return 2
    fi
  done

  config_paths="$(/usr/bin/awk '
    /^    config_paths:$/ { in_paths = 1; next }
    in_paths && /^      - / {
      path = $0
      sub(/^      - /, "", path)
      print path
      next
    }
    in_paths { exit }
  ' "${recipe_path}")"
  [[ "${config_paths}" == "$(printf '%s\n%s' \
    "${POLICY_GYM_CONFIG_REL}" "${REASONING_GYM_CONFIG_REL}")" ]] || {
    printf 'ERROR: Reasoning-Gym recipe has unexpected config_paths\n' >&2
    return 2
  }

  embedded_services="$(/usr/bin/awk '
    /^  nemo_gym:$/ { in_nemo_gym = 1; next }
    in_nemo_gym && /^[^[:space:]#]/ { exit }
    in_nemo_gym && /^    [A-Za-z_][A-Za-z0-9_]*:/ {
      current = $0
      sub(/^    /, "", current)
      sub(/:.*/, "", current)
      next
    }
    in_nemo_gym && /^      (responses_api_models|responses_api_agents|resources_servers):/ {
      print current
    }
  ' "${recipe_path}")"
  resolved_services="$({
    printf '%s\n' "${embedded_services}"
    while IFS= read -r config_path; do
      /usr/bin/awk '
        /^[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*$/ {
          service = $0
          sub(/:.*/, "", service)
          print service
        }
      ' "${GYM_ROOT}/${config_path}"
    done <<< "${config_paths}"
  } | /usr/bin/sort -u)"
  [[ "${resolved_services}" == "$(printf '%s\n%s\n%s' \
    policy_model reasoning_gym reasoning_gym_simple_agent)" ]] || {
    printf 'ERROR: unexpected Reasoning-Gym service set:\n%s\n' \
      "${resolved_services}" >&2
    return 2
  }

  printf 'EXPLORATORY_RGY2_GYM_SERVICES_GREEN services=policy_model,reasoning_gym,reasoning_gym_simple_agent\n'
}

for required_dir in "${BASE_DEPLOYMENT}" "${SOURCE_DEPLOYMENT}" \
  "${NEMO_ROOT}" "${BRIDGE_ROOT}" "${MCORE_ROOT}" "${GYM_ROOT}" \
  "${AUTOMODEL_ROOT}" "${MODEL_PATH}"; do
  [[ -d "${required_dir}" && ! -L "${required_dir}" ]] || {
    printf 'ERROR: required directory is missing or aliased: %s\n' "${required_dir}" >&2
    exit 2
  }
done
for required_file in "${BASE_DEPLOYMENT}/READY" "${SOURCE_DEPLOYMENT}/READY" \
  "${SOURCE_DEPLOYMENT}/DEPLOYMENT.sha256" \
  "${SOURCE_DEPLOYMENT}/NemoRL.runnable.sha256" \
  "${SOURCE_DEPLOYMENT}/Megatron-Bridge.runnable.sha256" \
  "${SOURCE_DEPLOYMENT}/Megatron-LM.runnable.sha256" \
  "${NANO_LAUNCHER}" "${NEMO_ROOT}/${CONFIG_PATH}" \
  "${NEMO_ROOT}/examples/run_grpo_single_controller.py" "${RAY_SUB_PATH}" \
  "${GYM_ROOT}/${POLICY_GYM_CONFIG_REL}" \
  "${GYM_ROOT}/${REASONING_GYM_CONFIG_REL}" \
  "${FIXTURE_PATH}" "${CONTAINER}" "${SANDBOX_CONTAINER}"; do
  [[ -f "${required_file}" && ! -L "${required_file}" ]] || {
    printf 'ERROR: required file is missing or aliased: %s\n' "${required_file}" >&2
    exit 2
  }
done
unset required_dir required_file

[[ "$(< "${BASE_DEPLOYMENT}/READY")" == "${BASE_READY}" ]]
[[ "$(< "${SOURCE_DEPLOYMENT}/READY")" == "${SOURCE_READY}" ]]
[[ "$(sha256_file "${SOURCE_DEPLOYMENT}/READY")" == "${SOURCE_READY_FILE_SHA256}" ]]
[[ "$(sha256_file "${SOURCE_DEPLOYMENT}/DEPLOYMENT.sha256")" == "${SOURCE_DEPLOYMENT_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${SOURCE_DEPLOYMENT}/NemoRL.runnable.sha256")" == "${NEMO_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${SOURCE_DEPLOYMENT}/Megatron-Bridge.runnable.sha256")" == "${BRIDGE_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${SOURCE_DEPLOYMENT}/Megatron-LM.runnable.sha256")" == "${MCORE_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(git -C "${NEMO_ROOT}" rev-parse HEAD^{commit})" == "${EXPECTED_NEMO_HEAD}" ]]
[[ "$(git -C "${NEMO_ROOT}" rev-parse HEAD^{tree})" == "${EXPECTED_NEMO_TREE}" ]]
[[ "$(sha256_file "${NANO_LAUNCHER}")" == "${EXPECTED_LAUNCHER_SHA256}" ]]
[[ "$(sha256_file "${NEMO_ROOT}/${CONFIG_PATH}")" == "${EXPECTED_CONFIG_SHA256}" ]]
[[ "$(sha256_file "${NEMO_ROOT}/examples/run_grpo_single_controller.py")" == "${EXPECTED_ENTRYPOINT_SHA256}" ]]
[[ "$(sha256_file "${RAY_SUB_PATH}")" == "${EXPECTED_RAY_SUB_SHA256}" ]]
[[ "$(sha256_file "${FIXTURE_PATH}")" == "${EXPECTED_FIXTURE_SHA256}" ]]
[[ "$(wc -l < "${FIXTURE_PATH}" | tr -d '[:space:]')" == 5 ]]
assert_exact_reasoning_gym_services "${NEMO_ROOT}/${CONFIG_PATH}"

printf 'EXPLORATORY_RGY2_133D_PRECHECK_GREEN acceptance=false source_ready=%s nemo=%s fixture=%s nodes_per_arm=1 gpus_per_arm=4 steps=2\n' \
  "${SOURCE_READY}" "${EXPECTED_NEMO_HEAD}" "${EXPECTED_FIXTURE_SHA256}"
if [[ "${launch_mode}" == "--check" ]]; then
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != *[[:space:]]* && ${#WANDB_API_KEY} -ge 20 ]] || {
  printf 'ERROR: a valid inherited WANDB_API_KEY is required\n' >&2
  exit 2
}
umask 077
readonly pair_nonce="$(/cm/local/apps/python3/bin/python3 -I -B -c 'import secrets; print(secrets.token_hex(8))')"
readonly pair_id="exploratory-rgy2-133d-NON-ACCEPTANCE-$(date -u +%Y%m%dT%H%M%SZ)-${pair_nonce}"
readonly pair_root="${STATE_ROOT}/${pair_id}"
mkdir -p -- "${pair_root}/off" "${pair_root}/on"

{
  printf 'status=EXPLORATORY_NON_ACCEPTANCE\nscientific_acceptance=false\n'
  printf 'purpose=surface_runtime_generation_reward_mtp_backward_breakage\n'
  printf 'pair_id=%s\nsource_deployment=%s\nsource_ready=%s\n' \
    "${pair_id}" "${SOURCE_DEPLOYMENT}" "${SOURCE_READY}"
  printf 'nemo_head=%s\nnemo_tree=%s\nlauncher_sha256=%s\nconfig_sha256=%s\n' \
    "${EXPECTED_NEMO_HEAD}" "${EXPECTED_NEMO_TREE}" \
    "${EXPECTED_LAUNCHER_SHA256}" "${EXPECTED_CONFIG_SHA256}"
  printf 'fixture=%s\nfixture_sha256=%s\n' "${FIXTURE_PATH}" "${EXPECTED_FIXTURE_SHA256}"
  printf 'arms=off:observe,on:train\nresources=per_arm_nodes:1,per_arm_gpus:4,colocated:true\n'
  printf 'schedule=max_steps:2,max_epochs:20,prompts_per_step:1,generations_per_prompt:4\n'
  printf 'topology=TP2/CP2/PP1/EP4/ETP1/SP/MTP5/full-recompute\n'
  printf 'wandb=online,entity:nvidia,project:nano35-rlvr-convergence,group:%s\n' "${pair_id}"
  printf 'scheduler=account:%s,partition:%s,qos:%s,time:%s\n' \
    "${ACCOUNT}" "${PARTITION}" "${QOS}" "${WALLTIME}"
} > "${pair_root}/NON_ACCEPTANCE_PROVENANCE.txt"
chmod 0444 "${pair_root}/NON_ACCEPTANCE_PROVENANCE.txt"

launch_arm() {
  local arm="$1"
  local shared_prefix_mode="$2"
  local arm_root="${pair_root}/${arm}"
  local cache_root="${arm_root}/cache"
  local receipt_root="${arm_root}/runtime_receipts"
  local extra_mounts
  mkdir -p -- "${cache_root}" "${receipt_root}"
  extra_mounts="${BASE_DEPLOYMENT}:${BASE_DEPLOYMENT}:ro,${SOURCE_DEPLOYMENT}:${SOURCE_DEPLOYMENT}:ro,${AUTOMODEL_ROOT}:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel:ro,${BRIDGE_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:ro,${MCORE_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:ro,${GYM_ROOT}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:ro,${MODEL_PATH}:${MODEL_PATH}:ro"

  export PATH="${HOST_PATH}" SLURM_CONF="${SLURM_CONF_PATH}"
  export EXP_NAME="x-rgy2-133d-${pair_nonce}-${arm}"
  export CONFIG_PATH TRAIN_ENTRYPOINT MODEL_PATH CONTAINER SANDBOX_CONTAINER
  export TRAIN_PATH="${FIXTURE_PATH}" VAL_PATH="${FIXTURE_PATH}"
  export RESULTS_DIR="${arm_root}" CHECKPOINT_DIR="${arm_root}/checkpoints"
  export PERSISTENT_CACHE="${cache_root}" BASE_LOG_DIR="${arm_root}/ray_logs"
  export NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR="${receipt_root}"
  export NEMO_GYM_VLLM_TRANSPORT_LOG="${arm_root}/model-io-transport.jsonl"
  export NEMORL_SHARED_PREFIX_RUNTIME_TRACE=1
  export CUBLAS_WORKSPACE_CONFIG=":4096:8" MAMBA_DETERMINISTIC=1
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NCCL_ALGO=Ring
  export NRL_SP_DETERMINISTIC_BACKWARD=1 PYTHONHASHSEED=42
  unset TRITON_CACHE_AUTOTUNING
  export SLURM_PARTITION="${PARTITION}" SLURM_ACCOUNT="${ACCOUNT}"
  export SLURM_QOS="${QOS}" WALLTIME
  export NUM_TRAIN_NODES=1 NUM_GEN_NODES=0 NUM_GYM_NODES=0
  export NUM_EXTERNAL_SERVICE_NODES=0 SEGMENT_SIZE=1 GPUS_PER_NODE=4
  export CPUS_PER_WORKER=144 COLOCATED_GENERATION=1 DEDICATED_RAY_HEAD=0
  export ENABLE_MTP_INFERENCE=0 NUM_SPECULATIVE_TOKENS=0 NRL_MAX_STEPS=2
  export NRL_USE_FASTOKENS=1 USE_SNAPSHOT=0 USE_CUSTOM_VLLM=0
  export INTERACTIVE=0 INTERACTIVE_WAIT=0 DRY_RUN=0
  export RAY_SUB="${RAY_SUB_PATH}" BATCH_SCRIPT="${RAY_SUB_PATH}"
  export CACHE_SYNC_FREQUENCY=0 CACHE_SEED_ENABLED=0
  export HF_HOME="${cache_root}/hf_home" HF_MODULES_CACHE="${cache_root}/hf_modules_cache"
  export HF_HUB_CACHE="${cache_root}/hf_home/hub" HF_DATASETS_CACHE="${cache_root}/hf_home/datasets"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_TOKEN=
  # The runtime image already contains the frozen NeMo-RL environment.  Do not
  # let `uv run` resync editable packages or write metadata into authenticated
  # read-only source deployments.
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
  export VIRTUAL_ENV=/opt/nemo_rl_venv UV_NO_SYNC=1
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export WANDB_API_KEY WANDB_PROJ=nano35-rlvr-convergence
  export WANDB_PROJECT=nano35-rlvr-convergence WANDB_ENTITY=nvidia
  export WANDB_MODE=online WANDB_RESUME=never WANDB_RUN_GROUP="${pair_id}"
  export NL2BASH_JUDGE_MODEL= SAFETY_JUDGE_MODEL= GENRM_BASE_URL=
  export GENRM_MODEL= GENRM_API_MODEL_NAME= SIF_DIR=
  export SANDBOX_COMMAND=/start-with-nginx.sh NEMO_SKILLS_SANDBOX_PORT=6000
  export SANDBOX_BASE_PORT=6001 SANDBOX_EXTRA_MOUNTS= SANDBOX_ENV_VARS=
  export UV_CACHE_DIR_OVERRIDE= MOUNTS="${arm_root}:${arm_root}"
  export EXTRA_MOUNTS="${extra_mounts}"
  export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=diff.ignoreSubmodules GIT_CONFIG_VALUE_0=all
  export SLURM_COMMENT='{"purpose":"EXPLORATORY_NON_ACCEPTANCE_RGY2_133D","owner":"jalbericiola"}'

  # nano35_launch.sh concatenates overrides into a shell command. Preserve
  # the YAML quotes through that second shell parse so Hydra receives a
  # string (the Megatron env_vars schema rejects an integer).
  exec /bin/bash "${NANO_LAUNCHER}" swe \
    "policy.shared_prefix_training.mode=${shared_prefix_mode}" \
    policy.shared_prefix_training.require_deterministic_execution=true \
    "policy.megatron_cfg.env_vars.RESULTS_DIR=${arm_root}" \
    "policy.megatron_cfg.env_vars.NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR=${receipt_root}" \
    '++policy.megatron_cfg.env_vars.NEMORL_SHARED_PREFIX_RUNTIME_TRACE=\"1\"' \
    ++policy.generation.vllm_cfg.enforce_eager=true \
    ++policy.generation.vllm_cfg.enable_prefix_caching=false \
    '~policy.generation.vllm_kwargs.compilation_config' \
    logger.wandb_enabled=true logger.tensorboard_enabled=false \
    ++logger.wandb.entity=nvidia logger.wandb.project=nano35-rlvr-convergence \
    "++logger.wandb.group=${pair_id}" "logger.wandb.name=${pair_id}-${arm}"
}

launch_arm off observe >"${pair_root}/off/launcher.out" 2>"${pair_root}/off/launcher.err" &
readonly off_pid=$!
launch_arm on train >"${pair_root}/on/launcher.out" 2>"${pair_root}/on/launcher.err" &
readonly on_pid=$!
off_status=0
on_status=0
wait "${off_pid}" || off_status=$?
wait "${on_pid}" || on_status=$?
cat -- "${pair_root}/off/launcher.out" "${pair_root}/on/launcher.out"
cat -- "${pair_root}/off/launcher.err" "${pair_root}/on/launcher.err" >&2
if (( off_status != 0 || on_status != 0 )); then
  printf 'ERROR: exploratory launch failed off_status=%s on_status=%s pair_root=%s\n' \
    "${off_status}" "${on_status}" "${pair_root}" >&2
  exit 1
fi
readonly off_job_id="$(sed -n 's/^Submitted batch job \([1-9][0-9]*\)$/\1/p' "${pair_root}/off/launcher.out" | tail -1)"
readonly on_job_id="$(sed -n 's/^Submitted batch job \([1-9][0-9]*\)$/\1/p' "${pair_root}/on/launcher.out" | tail -1)"
[[ "${off_job_id}" =~ ^[1-9][0-9]*$ && "${on_job_id}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'ERROR: could not parse both candidate job IDs; pair_root=%s\n' "${pair_root}" >&2
  exit 1
}
printf 'EXPLORATORY_RGY2_133D_PAIR_SUBMITTED acceptance=false pair_id=%s off_candidate_job=%s on_candidate_job=%s pair_root=%s wandb_group=%s\n' \
  "${pair_id}" "${off_job_id}" "${on_job_id}" "${pair_root}" "${pair_id}"
