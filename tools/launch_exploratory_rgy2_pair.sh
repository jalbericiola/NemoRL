#!/bin/bash
# Submit a bounded one-node-per-arm Reasoning-Gym OFF/ON GPU smoke from one
# complete immutable NeMo-RL 45783907 source overlay, with the unquantized MoE
# backend left at vLLM's production auto selection. For this model/image that
# must resolve to FlashInfer TRTLLM, exercising the routed-expert reload fix,
# shared-prefix training, and first-step optimizer headroom together.
#
# This launcher is deliberately NON-ACCEPTANCE evidence.  It exists to expose
# integration, generation, reward, MTP, and backward failures before the
# strict receipt/evaluator package is released.
set -euo pipefail

readonly HOST_PATH="/cm/local/apps/python3/bin:/cm/local/apps/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
readonly SLURM_CONF_PATH="/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf"
readonly BASE_DEPLOYMENT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/deployments/validated_shared_prefix_20260830q"
readonly BASE_READY="441548f85b9779788d458a0d4deeefcca789ed8b46810a1f6a929492cd27d4cf"
readonly NEMO_DEPLOYMENT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/non_acceptance_overlays/NON_ACCEPTANCE_full_nemorl_45783907_ebad1b2b_4ab5167b"
readonly NEMO_READY="8f3b4ff5f34fddf29afadc0cfe3a8ef00371ec789d5495f2d3c98805c9a6360d"
readonly NEMO_READY_FILE_SHA256="5b53af0c0066ce9526430b6b3ab2f706418141468f83577767b07850774a324f"
readonly NEMO_DEPLOYMENT_MANIFEST_SHA256="8f3b4ff5f34fddf29afadc0cfe3a8ef00371ec789d5495f2d3c98805c9a6360d"
readonly NEMO_RUNNABLE_MANIFEST_SHA256="f76235b62c04324c5a2f5930c7b175d78a41b6eb2107b0ac59f7619a54159bbd"
readonly COMPONENT_DEPLOYMENT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/non_acceptance_overlays/NON_ACCEPTANCE_component_bridge166164da_mcore8b8870ad"
readonly COMPONENT_READY="46a203f315040e3f6e2127f86a9437c3cdb2916a84f41eeb3e38b8de56acfedd"
readonly COMPONENT_READY_FILE_SHA256="fc54e1592a8b7548a7c7efbf692fe630b259f3bb8346a20f5086df50ee567827"
readonly COMPONENT_DEPLOYMENT_MANIFEST_SHA256="46a203f315040e3f6e2127f86a9437c3cdb2916a84f41eeb3e38b8de56acfedd"
readonly BRIDGE_RUNNABLE_MANIFEST_SHA256="9229a620edee832ea653aa3187c717de3990e6e44ea8317c8aa759c8062c5ea8"
readonly MCORE_RUNNABLE_MANIFEST_SHA256="a840c79dc124b248e56572c256e21b99fa4b4e23941e92000d22be04c29f5720"

readonly BASE_RUNNABLE="${BASE_DEPLOYMENT}/runnable"
readonly NEMO_RUNNABLE="${NEMO_DEPLOYMENT}/runnable"
readonly COMPONENT_RUNNABLE="${COMPONENT_DEPLOYMENT}/runnable"
readonly NEMO_ROOT="${NEMO_RUNNABLE}/NemoRL"
readonly BRIDGE_ROOT="${COMPONENT_RUNNABLE}/Megatron-Bridge"
readonly MCORE_ROOT="${COMPONENT_RUNNABLE}/Megatron-LM"
readonly FUSED_KERNEL_PATH="${MCORE_ROOT}/megatron/core/models/hybrid/shared_prefix_fused.py"
readonly VLLM_BACKEND_PATH="${NEMO_ROOT}/nemo_rl/models/generation/vllm/vllm_backend.py"
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
readonly STATE_ROOT="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/jalbericiola/rlvr41_spfx_validation/exploratory_rgy2_flashinfer_45783907_qfix_mem10_NON_ACCEPTANCE"

readonly EXPECTED_NEMO_HEAD="45783907244256fce0ed8c8f7512c3fd6ed51acd"
readonly EXPECTED_NEMO_TREE="5e446d1ba1e5b544f09b6459526a8d77263d4ab6"
readonly EXPECTED_NEMO_BRIDGE_GITLINK="ebad1b2bf676710bc810a6a4e5daf9d899bd4e79"
readonly EXPECTED_BRIDGE_HEAD="166164da6de0ee9bbe2906175b9e7c41411d084b"
readonly EXPECTED_BRIDGE_TREE="e6dc1beb26dac683877b19d9422f2a2fa96037d1"
readonly EXPECTED_MCORE_HEAD="8b8870ad4a54f24a7177f8c70e0403933d3f6dce"
readonly EXPECTED_MCORE_TREE="4452be0376d785ce7e2e0303e813c3173b8f0a3c"
readonly EXPECTED_FUSED_KERNEL_SHA256="73c55784840efdcfbc1b6e305bdc8a553f08b503e554719c0fb5b2947e3aa0b8"
readonly EXPECTED_VLLM_BACKEND_BLOB="09efa7276c020517faca2396d1b8032fef6f0be3"
readonly EXPECTED_VLLM_BACKEND_SHA256="df4420b5aec847a19dd049798109b271461ff85e9b126709aa5134bcd7902502"
readonly EXPECTED_NANO_LAUNCHER_SHA256="85625d987601a6fae3d61513cc01577e6ad33e4211b5cade92a59d4316804f3d"
readonly EXPECTED_CONFIG_SHA256="b2d618c7be9458b4f141913323351aa03a540564a161fbf453a20c0c7a82fc6f"
readonly EXPECTED_ENTRYPOINT_SHA256="45f99fbcde57b6265648c3197e767cdcebf20e67d28c08f43128a8d79570dedb"
readonly EXPECTED_RAY_SUB_SHA256="f26b378656450c9fc6f498c4b0083981693572f2a9b1e7a17050c4f729384e82"
readonly EXPECTED_FIXTURE_SHA256="da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"
readonly ACCOUNT="nemotron_sw_post"
readonly PARTITION="batch"
readonly QOS="short"
readonly WALLTIME="02:00:00"
# The 0.2 calibration completed one full optimizer update, then retained
# 53.60 GiB in the awake vLLM process while the pipelined second rollout
# overlapped MTP cross-entropy.  The trainer then used 129.68 GiB and had only
# 335.56 MiB free for a 448-MiB allocation.  Four 2k-token eager rollouts need
# far less KV capacity, so halve the vLLM reservation for the two-step canary.
readonly VLLM_GPU_MEMORY_UTILIZATION="0.1"

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
export PATH="${HOST_PATH}"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0

sha256_file() {
  /usr/bin/sha256sum -- "$1" | /usr/bin/awk '{print $1}'
}

readonly self_path="$(/usr/bin/readlink -f -- "$0")"
readonly self_dir="$(/usr/bin/dirname -- "${self_path}")"
readonly helper_path="${self_dir}/submit_exploratory_rgy2_from_netrc.sh"
[[ "$0" == /* && "${self_path}" == "$0" && -f "${self_path}" && ! -L "${self_path}" ]]
[[ -f "${helper_path}" && ! -L "${helper_path}" ]]
readonly self_sha256="$(sha256_file "${self_path}")"
readonly helper_sha256="$(sha256_file "${helper_path}")"
readonly expected_bundle_name="EXPLORATORY_RGY2_TOPOLOGY_RETRY_NON_ACCEPTANCE_${self_sha256}_${helper_sha256}"
[[ "$(/usr/bin/basename -- "${self_dir}")" == "${expected_bundle_name}" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${self_dir}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${self_path}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${helper_path}")" == "555" ]]

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

for required_dir in "${BASE_DEPLOYMENT}" "${NEMO_DEPLOYMENT}" \
  "${COMPONENT_DEPLOYMENT}" \
  "${NEMO_ROOT}" "${BRIDGE_ROOT}" "${MCORE_ROOT}" "${GYM_ROOT}" \
  "${AUTOMODEL_ROOT}" "${MODEL_PATH}"; do
  [[ -d "${required_dir}" && ! -L "${required_dir}" ]] || {
    printf 'ERROR: required directory is missing or aliased: %s\n' "${required_dir}" >&2
    exit 2
  }
done
for required_file in "${BASE_DEPLOYMENT}/READY" \
  "${NEMO_DEPLOYMENT}/READY" "${NEMO_DEPLOYMENT}/NON_ACCEPTANCE" \
  "${NEMO_DEPLOYMENT}/DEPLOYMENT.sha256" \
  "${NEMO_DEPLOYMENT}/NemoRL.runnable.sha256" \
  "${COMPONENT_DEPLOYMENT}/READY" "${COMPONENT_DEPLOYMENT}/NON_ACCEPTANCE" \
  "${COMPONENT_DEPLOYMENT}/DEPLOYMENT.sha256" \
  "${COMPONENT_DEPLOYMENT}/Megatron-Bridge.runnable.sha256" \
  "${COMPONENT_DEPLOYMENT}/Megatron-LM.runnable.sha256" \
  "${NANO_LAUNCHER}" "${NEMO_ROOT}/${CONFIG_PATH}" \
  "${NEMO_ROOT}/examples/run_grpo_single_controller.py" "${RAY_SUB_PATH}" \
  "${FUSED_KERNEL_PATH}" "${VLLM_BACKEND_PATH}" \
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
[[ "$(< "${NEMO_DEPLOYMENT}/READY")" == "${NEMO_READY}" ]]
[[ "$(< "${NEMO_DEPLOYMENT}/NON_ACCEPTANCE")" == "scientific_acceptance=false" ]]
[[ "$(sha256_file "${NEMO_DEPLOYMENT}/READY")" == "${NEMO_READY_FILE_SHA256}" ]]
[[ "$(sha256_file "${NEMO_DEPLOYMENT}/DEPLOYMENT.sha256")" == "${NEMO_DEPLOYMENT_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${NEMO_DEPLOYMENT}/NemoRL.runnable.sha256")" == "${NEMO_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(< "${COMPONENT_DEPLOYMENT}/READY")" == "${COMPONENT_READY}" ]]
[[ "$(< "${COMPONENT_DEPLOYMENT}/NON_ACCEPTANCE")" == "scientific_acceptance=false" ]]
[[ "$(sha256_file "${COMPONENT_DEPLOYMENT}/READY")" == "${COMPONENT_READY_FILE_SHA256}" ]]
[[ "$(sha256_file "${COMPONENT_DEPLOYMENT}/DEPLOYMENT.sha256")" == "${COMPONENT_DEPLOYMENT_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${COMPONENT_DEPLOYMENT}/Megatron-Bridge.runnable.sha256")" == "${BRIDGE_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(sha256_file "${COMPONENT_DEPLOYMENT}/Megatron-LM.runnable.sha256")" == "${MCORE_RUNNABLE_MANIFEST_SHA256}" ]]
[[ "$(git -C "${NEMO_ROOT}" rev-parse HEAD^{commit})" == "${EXPECTED_NEMO_HEAD}" ]]
[[ "$(git -C "${NEMO_ROOT}" rev-parse HEAD^{tree})" == "${EXPECTED_NEMO_TREE}" ]]
[[ "$(git -C "${NEMO_ROOT}" rev-parse "HEAD:nemo_rl/models/generation/vllm/vllm_backend.py")" == "${EXPECTED_VLLM_BACKEND_BLOB}" ]]
[[ "$(sha256_file "${VLLM_BACKEND_PATH}")" == "${EXPECTED_VLLM_BACKEND_SHA256}" ]]
[[ "$(git -C "${BRIDGE_ROOT}" rev-parse HEAD^{commit})" == "${EXPECTED_BRIDGE_HEAD}" ]]
[[ "$(git -C "${BRIDGE_ROOT}" rev-parse HEAD^{tree})" == "${EXPECTED_BRIDGE_TREE}" ]]
[[ "$(git -C "${MCORE_ROOT}" rev-parse HEAD^{commit})" == "${EXPECTED_MCORE_HEAD}" ]]
[[ "$(git -C "${MCORE_ROOT}" rev-parse HEAD^{tree})" == "${EXPECTED_MCORE_TREE}" ]]
[[ "$(git -C "${NEMO_ROOT}" ls-tree HEAD 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge | /usr/bin/awk '{print $3}')" == "${EXPECTED_NEMO_BRIDGE_GITLINK}" ]]
[[ "$(git -C "${BRIDGE_ROOT}" ls-tree HEAD 3rdparty/Megatron-LM | /usr/bin/awk '{print $3}')" == "${EXPECTED_MCORE_HEAD}" ]]
for repository_root in "${NEMO_ROOT}" "${BRIDGE_ROOT}" "${MCORE_ROOT}"; do
  ! git -C "${repository_root}" symbolic-ref -q HEAD >/dev/null
done
unset repository_root
[[ "$(sha256_file "${NANO_LAUNCHER}")" == "${EXPECTED_NANO_LAUNCHER_SHA256}" ]]
[[ "$(sha256_file "${NEMO_ROOT}/${CONFIG_PATH}")" == "${EXPECTED_CONFIG_SHA256}" ]]
[[ "$(sha256_file "${NEMO_ROOT}/examples/run_grpo_single_controller.py")" == "${EXPECTED_ENTRYPOINT_SHA256}" ]]
[[ "$(sha256_file "${RAY_SUB_PATH}")" == "${EXPECTED_RAY_SUB_SHA256}" ]]
[[ "$(sha256_file "${FUSED_KERNEL_PATH}")" == "${EXPECTED_FUSED_KERNEL_SHA256}" ]]
[[ "$(sha256_file "${FIXTURE_PATH}")" == "${EXPECTED_FIXTURE_SHA256}" ]]
[[ "$(wc -l < "${FIXTURE_PATH}" | tr -d '[:space:]')" == 5 ]]
assert_exact_reasoning_gym_services "${NEMO_ROOT}/${CONFIG_PATH}"

# This is an explicitly non-acceptance smoke.  The overlay stager already read
# and hashed every tracked regular file before atomically publishing a 0555
# tree.  Re-running the acceptance package and all 7,564 per-file checks here
# delayed GPU allocation by minutes without adding independent GPU evidence.
# Bind the publication through READY/manifest digests, exact Git identities and
# the runtime-critical file hashes above, and leave the full re-verification to
# the separately running acceptance-deployment lane.
[[ "$(/usr/bin/stat -c '%a' -- "${NEMO_DEPLOYMENT}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${COMPONENT_DEPLOYMENT}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${NEMO_ROOT}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${BRIDGE_ROOT}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${MCORE_ROOT}")" == "555" ]]
[[ "$(/usr/bin/stat -c '%a' -- "${NANO_LAUNCHER}")" == "555" ]]
for immutable_nemo_file in "${NEMO_ROOT}/${CONFIG_PATH}" \
  "${NEMO_ROOT}/examples/run_grpo_single_controller.py" "${RAY_SUB_PATH}" \
  "${VLLM_BACKEND_PATH}"; do
  [[ "$(/usr/bin/stat -c '%a' -- "${immutable_nemo_file}")" == "444" ]]
done
unset immutable_nemo_file

printf 'EXPLORATORY_RGY2_FLASHINFER_PRECHECK_GREEN acceptance=false inventory_rehash=false moe_backend=auto_expected_flashinfer_trtllm nemo_ready=%s component_ready=%s nemo=%s backend=%s bridge=%s mcore=%s fixture=%s nodes_per_arm=1 gpus_per_arm=4 steps=2 vllm_gpu_memory_utilization=%s\n' \
  "${NEMO_READY}" "${COMPONENT_READY}" "${EXPECTED_NEMO_HEAD}" \
  "${EXPECTED_VLLM_BACKEND_SHA256}" \
  "${EXPECTED_BRIDGE_HEAD}" "${EXPECTED_MCORE_HEAD}" \
  "${EXPECTED_FIXTURE_SHA256}" \
  "${VLLM_GPU_MEMORY_UTILIZATION}"
if [[ "${launch_mode}" == "--check" ]]; then
  exit 0
fi

[[ -n "${WANDB_API_KEY:-}" && "${WANDB_API_KEY}" != *[[:space:]]* && ${#WANDB_API_KEY} -ge 20 ]] || {
  printf 'ERROR: a valid inherited WANDB_API_KEY is required\n' >&2
  exit 2
}
umask 077
readonly pair_nonce="$(/cm/local/apps/python3/bin/python3 -I -B -c 'import secrets; print(secrets.token_hex(8))')"
readonly pair_id="exploratory-rgy2-flashinfer-45783907-qfix-mem10-NON-ACCEPTANCE-$(date -u +%Y%m%dT%H%M%SZ)-${pair_nonce}"
readonly pair_root="${STATE_ROOT}/${pair_id}"
mkdir -p -- "${pair_root}/off" "${pair_root}/on"

{
  printf 'status=EXPLORATORY_NON_ACCEPTANCE\nscientific_acceptance=false\n'
  printf 'purpose=verify_production_flashinfer_refit_shared_prefix_topology_and_optimizer_headroom\n'
  printf 'moe_backend=auto_expected_flashinfer_trtllm\nproduction_flashinfer_refit=exercised\n'
  printf 'pair_id=%s\nnemo_deployment=%s\nnemo_ready=%s\ncomponent_deployment=%s\ncomponent_ready=%s\n' \
    "${pair_id}" "${NEMO_DEPLOYMENT}" "${NEMO_READY}" \
    "${COMPONENT_DEPLOYMENT}" "${COMPONENT_READY}"
  printf 'nemo_deployment_manifest_sha256=%s\nnemo_runnable_manifest_sha256=%s\ncomponent_deployment_manifest_sha256=%s\nbridge_runnable_manifest_sha256=%s\nmcore_runnable_manifest_sha256=%s\n' \
    "${NEMO_DEPLOYMENT_MANIFEST_SHA256}" "${NEMO_RUNNABLE_MANIFEST_SHA256}" \
    "${COMPONENT_DEPLOYMENT_MANIFEST_SHA256}" \
    "${BRIDGE_RUNNABLE_MANIFEST_SHA256}" "${MCORE_RUNNABLE_MANIFEST_SHA256}"
  printf 'nemo_head=%s\nnemo_tree=%s\nbridge_head=%s\nbridge_tree=%s\nmcore_head=%s\nmcore_tree=%s\npair_launcher_path=%s\npair_launcher_sha256=%s\npair_helper_sha256=%s\nnano_launcher_sha256=%s\nconfig_sha256=%s\n' \
    "${EXPECTED_NEMO_HEAD}" "${EXPECTED_NEMO_TREE}" \
    "${EXPECTED_BRIDGE_HEAD}" "${EXPECTED_BRIDGE_TREE}" \
    "${EXPECTED_MCORE_HEAD}" "${EXPECTED_MCORE_TREE}" \
    "${self_path}" "${self_sha256}" "${helper_sha256}" \
    "${EXPECTED_NANO_LAUNCHER_SHA256}" "${EXPECTED_CONFIG_SHA256}"
  printf 'vllm_backend_blob=%s\nvllm_backend_sha256=%s\n' \
    "${EXPECTED_VLLM_BACKEND_BLOB}" "${EXPECTED_VLLM_BACKEND_SHA256}"
  printf 'nemo_live_mounts=automatic_no_ro_flag,published_dirs_and_executables_0555,published_files_0444\n'
  printf 'nemo_deployment_self_mount=%s:%s:ro\n' \
    "${NEMO_DEPLOYMENT}" "${NEMO_DEPLOYMENT}"
  printf 'fixture=%s\nfixture_sha256=%s\n' "${FIXTURE_PATH}" "${EXPECTED_FIXTURE_SHA256}"
  printf 'arms=off:observe,on:train\nresources=per_arm_nodes:1,per_arm_gpus:4,colocated:true,vllm_gpu_memory_utilization:%s\n' \
    "${VLLM_GPU_MEMORY_UTILIZATION}"
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
  # USE_SNAPSHOT=0 makes the immutable 45783907 nano launcher mount its own
  # nemo_rl package, configs, Nano recipe, and SingleController entrypoint.
  # Those automatic mounts omit an OCI `ro` flag, so this NON_ACCEPTANCE run
  # relies on the fail-closed publication modes checked above (dirs and tracked
  # executables 0555, other tracked files 0444); Python bytecode and uv writes
  # are disabled.
  # The identical-path read-only deployment bind makes nano35_launch.sh's host
  # working directory visible to Pyxis.  It does not target or overlap /opt.
  # Do not add a second NeMo parent/child bind under /opt/nemo-rl here.
  extra_mounts="${NEMO_DEPLOYMENT}:${NEMO_DEPLOYMENT}:ro,${BASE_DEPLOYMENT}:${BASE_DEPLOYMENT}:ro,${AUTOMODEL_ROOT}:/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel:ro,${BRIDGE_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:ro,${MCORE_ROOT}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:ro,${GYM_ROOT}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:ro,${MODEL_PATH}:${MODEL_PATH}:ro"
  [[ ",${extra_mounts}," == *",${NEMO_DEPLOYMENT}:${NEMO_DEPLOYMENT}:ro,"* ]]
  [[ "${extra_mounts}" != *':/opt/nemo-rl:ro'* ]]
  [[ "${extra_mounts}" != *':/opt/nemo-rl/nemo_rl'* ]]
  [[ "${extra_mounts}" != *'/vllm_backend.py:'* ]]

  export PATH="${HOST_PATH}" SLURM_CONF="${SLURM_CONF_PATH}"
  export EXP_NAME="x-rgy2-mtp-${pair_nonce}-${arm}"
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
  export SLURM_COMMENT='{"purpose":"EXPLORATORY_NON_ACCEPTANCE_RGY2_FLASHINFER_457_QFIX","owner":"jalbericiola"}'

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
    "policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}" \
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
printf 'EXPLORATORY_RGY2_FLASHINFER_PAIR_SUBMITTED acceptance=false moe_backend=auto_expected_flashinfer_trtllm pair_id=%s off_candidate_job=%s on_candidate_job=%s pair_root=%s wandb_group=%s\n' \
  "${pair_id}" "${off_job_id}" "${on_job_id}" "${pair_root}" "${pair_id}"
