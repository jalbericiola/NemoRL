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

# Shared fail-closed helpers for launch_pair.sh and nano35_single_env_pair.sh.
# This file only defines functions; callers retain set -euo pipefail ownership.

strict_pair_error() {
  echo "ERROR: $*" >&2
  return 2
}

strict_pair_sha256_file() {
  local path="$1"
  local digest
  local ignored

  if command -v sha256sum >/dev/null 2>&1; then
    read -r digest ignored < <(sha256sum -- "${path}")
  elif command -v shasum >/dev/null 2>&1; then
    read -r digest ignored < <(shasum -a 256 -- "${path}")
  else
    strict_pair_error "sha256sum or shasum is required to authenticate inputs."
    return
  fi
  echo "${digest}"
}

strict_pair_sha256_text() {
  python3 -I -B - "$1" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("ascii")).hexdigest())
PY
}

strict_pair_file_mode() {
  local path="$1"
  local mode

  if mode="$(stat -c '%a' -- "${path}" 2>/dev/null)"; then
    :
  else
    mode="$(stat -f '%Lp' -- "${path}")"
  fi
  echo "${mode}"
}

strict_pair_require_mode() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual

  actual="$(strict_pair_file_mode "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    strict_pair_error "${label} must have mode ${expected}; got ${actual}: ${path}"
  fi
}

strict_pair_require_readonly_executable() {
  local path="$1"
  local label="$2"
  local mode
  local mode_value

  mode="$(strict_pair_file_mode "${path}")"
  mode_value=$((8#${mode}))
  if (( (mode_value & 8#222) != 0 )) || [[ ! -x "${path}" ]]; then
    strict_pair_error "${label} must be executable and have no write bits; got mode ${mode}: ${path}"
  fi
}

strict_pair_require_digest() {
  local name="$1"
  local value="${!name:-}"

  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    strict_pair_error "${name} must be an explicit lowercase SHA-256."
  fi
}

strict_pair_require_commit() {
  local name="$1"
  local value="${!name:-}"

  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    strict_pair_error "${name} must be an explicit lowercase 40-hex Git commit."
  fi
}

strict_pair_load_source_identity() {
  local gym_relative="3rdparty/Gym-workspace/Gym"
  local gym_index_record
  local gym_mode
  local gym_stage
  local gym_path

  strict_pair_require_commit EXPECTED_NEMO_HEAD
  strict_pair_require_commit EXPECTED_GYM_GITLINK_COMMIT
  if [[ "${EXPECTED_NEMO_HEAD}" == "d7b49a459f08670b6534a56deb99e432f576028a" ]]; then
    strict_pair_error "legacy NeMo-RL deployment d7b49a459f08670b6534a56deb99e432f576028a is forbidden for the strict live pair."
  fi

  STRICT_PAIR_NEMO_HEAD="$(git -C "${STRICT_PAIR_PROJECT_ROOT}" rev-parse HEAD)"
  STRICT_PAIR_NEMO_TREE="$(git -C "${STRICT_PAIR_PROJECT_ROOT}" rev-parse 'HEAD^{tree}')"
  if [[ "${STRICT_PAIR_NEMO_HEAD}" != "${EXPECTED_NEMO_HEAD}" ]]; then
    strict_pair_error "deployed NeMo-RL HEAD differs from EXPECTED_NEMO_HEAD."
  fi

  gym_index_record="$(
    git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --stage -- "${gym_relative}"
  )"
  read -r gym_mode STRICT_PAIR_GYM_GITLINK_COMMIT gym_stage gym_path <<< "${gym_index_record}"
  if [[ "${gym_mode}" != "160000" || "${gym_stage}" != "0" || \
        "${gym_path}" != "${gym_relative}" ]]; then
    strict_pair_error "Reasoning Gym must be one exact stage-0 gitlink at ${gym_relative}."
  fi
  if [[ "${STRICT_PAIR_GYM_GITLINK_COMMIT}" != "${EXPECTED_GYM_GITLINK_COMMIT}" ]]; then
    strict_pair_error "Reasoning Gym gitlink differs from EXPECTED_GYM_GITLINK_COMMIT."
  fi
  STRICT_PAIR_GYM_ROOT="${STRICT_PAIR_PROJECT_ROOT}/${gym_relative}"
  strict_pair_require_canonical_dir "${STRICT_PAIR_GYM_ROOT}" "deployed Reasoning Gym gitlink"
  if [[ "$(git -C "${STRICT_PAIR_GYM_ROOT}" rev-parse HEAD)" != \
        "${STRICT_PAIR_GYM_GITLINK_COMMIT}" ]]; then
    strict_pair_error "deployed Reasoning Gym HEAD differs from its authenticated gitlink."
  fi
  STRICT_PAIR_GYM_TREE="$(git -C "${STRICT_PAIR_GYM_ROOT}" rev-parse 'HEAD^{tree}')"
}

strict_pair_require_canonical_file() {
  local path="$1"
  local label="$2"
  local canonical

  if [[ "${path}" != /* ]]; then
    strict_pair_error "${label} must be absolute and canonical: ${path}"
    return
  fi
  if [[ -L "${path}" || ! -f "${path}" ]]; then
    strict_pair_error "${label} must be a regular, non-symlink file: ${path}"
    return
  fi
  canonical="$(realpath -- "${path}")"
  if [[ "${canonical}" != "${path}" ]]; then
    strict_pair_error "${label} must already be canonical; resolved ${path} to ${canonical}."
  fi
}

strict_pair_require_canonical_dir() {
  local path="$1"
  local label="$2"
  local canonical

  if [[ "${path}" != /* ]]; then
    strict_pair_error "${label} must be absolute and canonical: ${path}"
    return
  fi
  if [[ -L "${path}" || ! -d "${path}" ]]; then
    strict_pair_error "${label} must be a real, non-symlink directory: ${path}"
    return
  fi
  canonical="$(realpath -- "${path}")"
  if [[ "${canonical}" != "${path}" ]]; then
    strict_pair_error "${label} must already be canonical; resolved ${path} to ${canonical}."
  fi
}

strict_pair_model_tree_sha256_v1() {
  python3 -I -B - "$1" <<'PY'
import hashlib
import os
import stat
import sys

root = os.path.abspath(sys.argv[1])
if os.path.realpath(root) != root:
    raise SystemExit("model root is noncanonical")
tree = hashlib.sha256()
for directory, directory_names, file_names in os.walk(
    root, topdown=True, followlinks=False
):
    directory_names.sort()
    file_names.sort()
    for name in directory_names:
        mode = os.lstat(os.path.join(directory, name)).st_mode
        if not stat.S_ISDIR(mode):
            raise SystemExit("model tree contains a symlink or special directory")
    for name in file_names:
        path = os.path.join(directory, name)
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise SystemExit("model tree contains a symlink or special file")
        relative = os.path.relpath(path, root).replace(os.sep, "/").encode(
            "utf-8", "surrogateescape"
        )
        content = hashlib.sha256()
        with open(path, "rb", buffering=0) as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                content.update(chunk)
        tree.update(b"F\0" + relative + b"\0" + content.digest())
print(tree.hexdigest())
PY
}

strict_pair_verify_deployment() {
  local ready_path
  local ready_sha256
  local ready_content
  local manifest_name
  local expected_name
  local manifest_path
  local manifest_sha256
  local job_wrapper_relative

  strict_pair_require_canonical_dir "${DEPLOYMENT_ROOT}" "DEPLOYMENT_ROOT"
  strict_pair_require_mode "${DEPLOYMENT_ROOT}" "500" "DEPLOYMENT_ROOT"
  for expected_name in \
    EXPECTED_DEPLOYMENT_READY \
    EXPECTED_DEPLOYMENT_READY_FILE_SHA256 \
    EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256; do
    strict_pair_require_digest "${expected_name}"
  done

  ready_path="${DEPLOYMENT_ROOT}/READY"
  strict_pair_require_canonical_file "${ready_path}" "deployment READY"
  strict_pair_require_mode "${ready_path}" "444" "deployment READY"
  ready_sha256="$(strict_pair_sha256_file "${ready_path}")"
  if [[ "${ready_sha256}" != "${EXPECTED_DEPLOYMENT_READY_FILE_SHA256}" ]]; then
    strict_pair_error "deployment READY file SHA-256 mismatch."
  fi
  ready_content="$(< "${ready_path}")"
  if [[ "${ready_content}" != "${EXPECTED_DEPLOYMENT_READY}" ]]; then
    strict_pair_error "deployment READY content differs from EXPECTED_DEPLOYMENT_READY."
  fi

  while read -r manifest_name expected_name; do
    manifest_path="${DEPLOYMENT_ROOT}/${manifest_name}"
    strict_pair_require_canonical_file "${manifest_path}" "deployment runnable manifest"
    strict_pair_require_mode "${manifest_path}" "400" "deployment runnable manifest"
    manifest_sha256="$(strict_pair_sha256_file "${manifest_path}")"
    if [[ "${manifest_sha256}" != "${!expected_name}" ]]; then
      strict_pair_error "${manifest_name} SHA-256 mismatch."
    fi
    if ! (cd -- "${DEPLOYMENT_ROOT}" && sha256sum --check --strict --quiet -- "${manifest_name}"); then
      strict_pair_error "${manifest_name} content verification failed."
    fi
  done <<'MANIFESTS'
NemoRL.runnable.sha256 EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256
Megatron-Bridge.runnable.sha256 EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256
Megatron-LM.runnable.sha256 EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256
MANIFESTS

  strict_pair_require_digest EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256
  strict_pair_require_canonical_file "${STRICT_PAIR_JOB_WRAPPER}" "STRICT_PAIR_JOB_WRAPPER"
  case "${STRICT_PAIR_JOB_WRAPPER}" in
    "${DEPLOYMENT_ROOT}"/*) ;;
    *)
      strict_pair_error "STRICT_PAIR_JOB_WRAPPER must resolve under DEPLOYMENT_ROOT."
      ;;
  esac
  strict_pair_require_readonly_executable "${STRICT_PAIR_JOB_WRAPPER}" "STRICT_PAIR_JOB_WRAPPER"
  STRICT_PAIR_JOB_WRAPPER_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_JOB_WRAPPER}")"
  if [[ "${STRICT_PAIR_JOB_WRAPPER_SHA256}" != "${EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256}" ]]; then
    strict_pair_error "STRICT_PAIR_JOB_WRAPPER SHA-256 mismatch."
  fi
  job_wrapper_relative="${STRICT_PAIR_JOB_WRAPPER#${DEPLOYMENT_ROOT}/}"
  if ! grep -Fqxh \
    "${STRICT_PAIR_JOB_WRAPPER_SHA256}  ${STRICT_PAIR_JOB_WRAPPER}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" \
    "${DEPLOYMENT_ROOT}/Megatron-Bridge.runnable.sha256" \
    "${DEPLOYMENT_ROOT}/Megatron-LM.runnable.sha256" >/dev/null; then
    strict_pair_error "STRICT_PAIR_JOB_WRAPPER is not authenticated by a runnable manifest: ${job_wrapper_relative}"
  fi
}

strict_pair_prepare_contract() {
  local project_root="$1"
  local recipe_dir="$2"
  local source_path
  local source_rel

  STRICT_PAIR_PROJECT_ROOT="${project_root}"
  STRICT_PAIR_RECIPE_DIR="${recipe_dir}"
  strict_pair_require_canonical_dir "${STRICT_PAIR_PROJECT_ROOT}" "NeMo-RL source root"
  strict_pair_require_canonical_dir "${RESULTS_DIR}" "RESULTS_DIR"
  strict_pair_require_canonical_dir "${PERSISTENT_CACHE}" "PERSISTENT_CACHE"
  strict_pair_require_canonical_dir "${MODEL_PATH}" "MODEL_PATH"
  strict_pair_require_canonical_file "${CONTAINER}" "CONTAINER"
  strict_pair_require_canonical_file "${SANDBOX_CONTAINER}" "SANDBOX_CONTAINER"
  strict_pair_require_canonical_file "${TRAIN_PATH}" "TRAIN_PATH"

  STRICT_PAIR_FIXTURE_SHA256="$(strict_pair_sha256_file "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_SHA256}" != "${EXPECTED_FIXTURE_SHA256}" ]]; then
    strict_pair_error "TRAIN_PATH SHA-256 mismatch: expected ${EXPECTED_FIXTURE_SHA256}, got ${STRICT_PAIR_FIXTURE_SHA256}."
  fi
  STRICT_PAIR_FIXTURE_ROWS="$(awk 'END { print NR }' "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_ROWS}" != "5" ]]; then
    strict_pair_error "authenticated TRAIN_PATH must contain exactly 5 rows; got ${STRICT_PAIR_FIXTURE_ROWS}."
  fi

  strict_pair_verify_deployment

  STRICT_PAIR_DEPLOYED_NEMO_ROOT="${DEPLOYMENT_ROOT}/runnable/NemoRL"
  strict_pair_require_canonical_dir "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "deployed NeMo-RL root"
  strict_pair_require_mode "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "500" "deployed NeMo-RL root"
  if [[ "${STRICT_PAIR_PROJECT_ROOT}" != "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" ]]; then
    strict_pair_error "launcher source must be exactly DEPLOYMENT_ROOT/runnable/NemoRL."
  fi
  if ! git -C "${STRICT_PAIR_PROJECT_ROOT}" diff --cached --quiet HEAD --; then
    strict_pair_error "deployed NeMo-RL index differs from HEAD."
  fi
  if [[ -n "$(git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --others --exclude-standard)" ]]; then
    strict_pair_error "deployed NeMo-RL source contains untracked files."
  fi
  if [[ -n "$(git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --others --ignored --exclude-standard)" ]]; then
    strict_pair_error "deployed NeMo-RL source contains ignored files."
  fi
  strict_pair_load_source_identity
  python3 -I -B - \
    "${STRICT_PAIR_PROJECT_ROOT}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" <<'PY'
import hashlib
import os
import pathlib
import stat
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
manifest = {}
for line in manifest_path.read_text(encoding="ascii").splitlines():
    digest, separator, raw_path = line.partition("  ")
    if not separator or raw_path in manifest:
        raise SystemExit("malformed or duplicate NemoRL runnable manifest entry")
    manifest[raw_path] = digest
def verify_repo(repo: pathlib.Path) -> None:
    cached = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet", "HEAD", "--"],
        check=False,
    )
    if cached.returncode != 0:
        raise SystemExit(f"deployed repository index differs from HEAD: {repo}")
    untracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
        text=False,
    )
    if untracked:
        raise SystemExit(f"deployed repository contains untracked files: {repo}")
    ignored = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ],
        text=False,
    )
    if ignored:
        raise SystemExit(f"deployed repository contains ignored files: {repo}")
    entries = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--stage", "-z"], text=False
    ).split(b"\0")
    for raw_entry in entries:
        if not raw_entry:
            continue
        metadata_raw, separator, relative_raw = raw_entry.partition(b"\t")
        if not separator:
            raise SystemExit("malformed git index record")
        mode_raw, object_id_raw, stage_raw = metadata_raw.split(b" ")
        if stage_raw != b"0":
            raise SystemExit("unmerged git index entry in deployed source")
        relative = os.fsdecode(relative_raw)
        path = repo / relative
        metadata = os.lstat(path)
        if mode_raw == b"160000":
            if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
                raise SystemExit(f"gitlink is not one real directory: {path}")
            actual_head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual_head != object_id_raw.decode("ascii"):
                raise SystemExit(f"gitlink HEAD differs from authenticated index: {path}")
            verify_repo(path)
        elif mode_raw == b"120000":
            if not stat.S_ISLNK(metadata.st_mode):
                raise SystemExit(f"tracked symlink changed type: {path}")
            actual_object = subprocess.check_output(
                ["git", "-C", str(repo), "hash-object", "--stdin"],
                input=os.fsencode(os.readlink(path)),
                text=False,
            ).strip()
            if actual_object != object_id_raw:
                raise SystemExit(f"tracked symlink differs from authenticated index: {path}")
        elif mode_raw in {b"100644", b"100755"}:
            if not stat.S_ISREG(metadata.st_mode):
                raise SystemExit(f"tracked source is not regular: {path}")
            if metadata.st_mode & 0o222:
                raise SystemExit(f"tracked source has write bits: {path}")
            executable = bool(metadata.st_mode & 0o111)
            if executable != (mode_raw == b"100755"):
                raise SystemExit(f"tracked source executable mode differs from index: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if manifest.get(str(path)) != digest:
                raise SystemExit(
                    f"tracked source is absent or drifted in runnable manifest: {path}"
                )
        else:
            raise SystemExit(f"unsupported git index mode {mode_raw!r}: {path}")


verify_repo(root)
PY

  STRICT_PAIR_MODEL_TREE_SHA256="$(strict_pair_model_tree_sha256_v1 "${MODEL_PATH}")"
  STRICT_PAIR_CONTAINER_SHA256="$(strict_pair_sha256_file "${CONTAINER}")"
  STRICT_PAIR_SANDBOX_CONTAINER_SHA256="$(strict_pair_sha256_file "${SANDBOX_CONTAINER}")"
  for source_rel in \
    examples/run_grpo_single_controller.py \
    examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh \
    examples/nemo_gym/nemotron-3.5-nano/nano35_single_env_pair.sh \
    examples/nemo_gym/nemotron-3.5-nano/launch_pair.sh \
    examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh \
    examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml; do
    source_path="${STRICT_PAIR_PROJECT_ROOT}/${source_rel}"
    if ! git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --error-unmatch -- "${source_rel}" >/dev/null 2>&1; then
      strict_pair_error "strict-pair source must be present in the git index: ${source_rel}"
    fi
    strict_pair_require_canonical_file "${source_path}" "strict-pair source"
    if ! git -C "${STRICT_PAIR_PROJECT_ROOT}" diff --quiet HEAD -- "${source_rel}"; then
      strict_pair_error "strict-pair source differs from authenticated HEAD: ${source_rel}"
    fi
  done

  STRICT_PAIR_CONFIG_SHA256="$(strict_pair_sha256_file "${recipe_dir}/single_env_reasoning_gym_sc.yaml")"
  STRICT_PAIR_ENTRYPOINT_SHA256="$(strict_pair_sha256_file "${project_root}/examples/run_grpo_single_controller.py")"
  STRICT_PAIR_LAUNCHER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_launch.sh")"
  STRICT_PAIR_ARM_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_single_env_pair.sh")"
  STRICT_PAIR_PARENT_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/launch_pair.sh")"
  STRICT_PAIR_CONTRACT_SHA256="$(strict_pair_sha256_file "${recipe_dir}/strict_pair_contract.sh")"
  STRICT_PAIR_MANIFEST_PATH="${RESULTS_DIR}/PAIR_MANIFEST.json"
  STRICT_PAIR_SNAPSHOT_PARENT="${RESULTS_DIR}/code_snapshots_strict_pairs/${PAIR_ID}"
  STRICT_PAIR_OFF_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/off-${PAIR_ID}"
  STRICT_PAIR_ON_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/on-${PAIR_ID}"
}

# Load only the values needed to verify the parent-anchored pair contract in an
# arm process. The authoritative parent already paid for complete deployment,
# source, and artifact authentication before it built either snapshot. Repeating
# the three multi-repository runnable-manifest scans in both arms adds minutes of
# redundant I/O without strengthening the job-boundary PRE/POST gate.
strict_pair_load_arm_contract() {
  local project_root="$1"
  local recipe_dir="$2"
  local expected_name

  STRICT_PAIR_PROJECT_ROOT="${project_root}"
  STRICT_PAIR_RECIPE_DIR="${recipe_dir}"
  strict_pair_require_canonical_dir "${RESULTS_DIR}" "RESULTS_DIR"
  strict_pair_require_canonical_dir "${PERSISTENT_CACHE}" "PERSISTENT_CACHE"
  strict_pair_require_canonical_dir "${MODEL_PATH}" "MODEL_PATH"
  strict_pair_require_canonical_file "${CONTAINER}" "CONTAINER"
  strict_pair_require_canonical_file "${SANDBOX_CONTAINER}" "SANDBOX_CONTAINER"
  strict_pair_require_canonical_file "${TRAIN_PATH}" "TRAIN_PATH"
  strict_pair_require_canonical_dir "${DEPLOYMENT_ROOT}" "DEPLOYMENT_ROOT"
  strict_pair_require_mode "${DEPLOYMENT_ROOT}" "500" "DEPLOYMENT_ROOT"

  for expected_name in \
    EXPECTED_DEPLOYMENT_READY \
    EXPECTED_DEPLOYMENT_READY_FILE_SHA256 \
    EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256 \
    EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256 \
    EXPECTED_STRICT_PAIR_CONTAINER_SHA256 \
    EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256; do
    strict_pair_require_digest "${expected_name}"
  done

  STRICT_PAIR_DEPLOYED_NEMO_ROOT="${DEPLOYMENT_ROOT}/runnable/NemoRL"
  strict_pair_require_canonical_dir "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "deployed NeMo-RL root"
  strict_pair_require_mode "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "500" "deployed NeMo-RL root"
  if [[ "${STRICT_PAIR_PROJECT_ROOT}" != "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" ]]; then
    strict_pair_error "launcher source must be exactly DEPLOYMENT_ROOT/runnable/NemoRL."
  fi

  strict_pair_require_canonical_file "${STRICT_PAIR_JOB_WRAPPER}" "STRICT_PAIR_JOB_WRAPPER"
  case "${STRICT_PAIR_JOB_WRAPPER}" in
    "${DEPLOYMENT_ROOT}"/*) ;;
    *)
      strict_pair_error "STRICT_PAIR_JOB_WRAPPER must resolve under DEPLOYMENT_ROOT."
      ;;
  esac
  strict_pair_require_readonly_executable "${STRICT_PAIR_JOB_WRAPPER}" "STRICT_PAIR_JOB_WRAPPER"
  STRICT_PAIR_JOB_WRAPPER_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_JOB_WRAPPER}")"
  if [[ "${STRICT_PAIR_JOB_WRAPPER_SHA256}" != "${EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256}" ]]; then
    strict_pair_error "STRICT_PAIR_JOB_WRAPPER SHA-256 mismatch."
  fi

  STRICT_PAIR_FIXTURE_SHA256="$(strict_pair_sha256_file "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_SHA256}" != "${EXPECTED_FIXTURE_SHA256}" ]]; then
    strict_pair_error "TRAIN_PATH SHA-256 mismatch: expected ${EXPECTED_FIXTURE_SHA256}, got ${STRICT_PAIR_FIXTURE_SHA256}."
  fi
  STRICT_PAIR_FIXTURE_ROWS="$(awk 'END { print NR }' "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_ROWS}" != "5" ]]; then
    strict_pair_error "authenticated TRAIN_PATH must contain exactly 5 rows; got ${STRICT_PAIR_FIXTURE_ROWS}."
  fi

  strict_pair_load_source_identity
  STRICT_PAIR_CONFIG_SHA256="$(strict_pair_sha256_file "${recipe_dir}/single_env_reasoning_gym_sc.yaml")"
  STRICT_PAIR_ENTRYPOINT_SHA256="$(strict_pair_sha256_file "${project_root}/examples/run_grpo_single_controller.py")"
  STRICT_PAIR_LAUNCHER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_launch.sh")"
  STRICT_PAIR_ARM_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_single_env_pair.sh")"
  STRICT_PAIR_PARENT_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/launch_pair.sh")"
  STRICT_PAIR_CONTRACT_SHA256="$(strict_pair_sha256_file "${recipe_dir}/strict_pair_contract.sh")"
  STRICT_PAIR_MODEL_TREE_SHA256="${EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256}"
  STRICT_PAIR_CONTAINER_SHA256="${EXPECTED_STRICT_PAIR_CONTAINER_SHA256}"
  STRICT_PAIR_SANDBOX_CONTAINER_SHA256="${EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256}"
  STRICT_PAIR_MANIFEST_PATH="${RESULTS_DIR}/PAIR_MANIFEST.json"
  STRICT_PAIR_SNAPSHOT_PARENT="${RESULTS_DIR}/code_snapshots_strict_pairs/${PAIR_ID}"
  STRICT_PAIR_OFF_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/off-${PAIR_ID}"
  STRICT_PAIR_ON_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/on-${PAIR_ID}"
  strict_pair_require_canonical_file \
    "${STRICT_PAIR_OFF_SNAPSHOT}/strict-pair-snapshot-manifest.sha256" \
    "off strict snapshot manifest"
  strict_pair_require_canonical_file \
    "${STRICT_PAIR_ON_SNAPSHOT}/strict-pair-snapshot-manifest.sha256" \
    "on strict snapshot manifest"
  STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256="$(
    strict_pair_sha256_file "${STRICT_PAIR_OFF_SNAPSHOT}/strict-pair-snapshot-manifest.sha256"
  )"
  STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256="$(
    strict_pair_sha256_file "${STRICT_PAIR_ON_SNAPSHOT}/strict-pair-snapshot-manifest.sha256"
  )"
}

strict_pair_verify_snapshot() {
  local arm="$1"
  local snapshot="$2"
  local manifest
  local manifest_sha256
  local entrypoint_manifest
  local symlink_manifest
  local mode_manifest
  local writable_path=""
  local path
  local mode

  strict_pair_require_canonical_dir "${snapshot}" "${arm} strict snapshot"
  manifest="${snapshot}/strict-pair-snapshot-manifest.sha256"
  strict_pair_require_canonical_file "${manifest}" "${arm} strict snapshot manifest"
  strict_pair_require_mode "${manifest}" "400" "${arm} strict snapshot manifest"
  manifest_sha256="$(strict_pair_sha256_file "${manifest}")"
  if ! (cd -- "${snapshot}" && \
        sha256sum --check --strict --quiet -- "${manifest##*/}"); then
    strict_pair_error "${arm} strict snapshot content verification failed."
  fi
  symlink_manifest="${snapshot}/strict-pair-snapshot-symlinks.json"
  strict_pair_require_canonical_file "${symlink_manifest}" "${arm} snapshot symlink manifest"
  mode_manifest="${snapshot}/strict-pair-snapshot-modes.json"
  strict_pair_require_canonical_file "${mode_manifest}" "${arm} snapshot mode manifest"
  python3 -I -B - \
    "${snapshot}" "${manifest}" "${symlink_manifest}" "${mode_manifest}" <<'PY'
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
for directory, directory_names, file_names in os.walk(root, followlinks=False):
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
  if [[ "$(strict_pair_sha256_file "${snapshot}/examples/run_grpo_single_controller.py")" != \
        "${STRICT_PAIR_ENTRYPOINT_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot entrypoint differs from authenticated source."
  fi
  if [[ "$(strict_pair_sha256_file "${snapshot}/examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml")" != \
        "${STRICT_PAIR_CONFIG_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot config differs from authenticated source."
  fi
  entrypoint_manifest="${snapshot}/nano35-entrypoint-manifest.sha256"
  strict_pair_require_canonical_file "${entrypoint_manifest}" "${arm} entrypoint manifest"
  strict_pair_require_mode "${entrypoint_manifest}" "400" "${arm} entrypoint manifest"
  if [[ "$(< "${entrypoint_manifest}")" != \
        "${STRICT_PAIR_ENTRYPOINT_SHA256}  examples/run_grpo_single_controller.py" ]]; then
    strict_pair_error "${arm} entrypoint manifest differs from authenticated source."
  fi
  while IFS= read -r -d '' path; do
    mode="$(strict_pair_file_mode "${path}")"
    if (( (8#${mode} & 8#222) != 0 )); then
      writable_path="${path}"
      break
    fi
  done < <(find "${snapshot}" \( -type d -o -type f \) -print0)
  if [[ -n "${writable_path}" ]]; then
    strict_pair_error "${arm} strict snapshot contains a writable path: ${writable_path}"
  fi
  if [[ "${arm}" == "off" ]]; then
    STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256="${manifest_sha256}"
  else
    STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256="${manifest_sha256}"
  fi
}

strict_pair_list_snapshot_paths() {
  local tracked_path

  while IFS= read -r -d '' tracked_path; do
    # `git ls-files --recurse-submodules` includes the outer mode-160000
    # gitlink records as well as the recursively tracked leaves. Copying a
    # gitlink directory recursively would admit untracked submodule bytes, so
    # emit only regular files and symlinks from the already verified source.
    if [[ ! -d "${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}" || \
          -L "${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}" ]]; then
      printf '%s\0' "${tracked_path}"
    fi
  done < <(git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --recurse-submodules -z)
}

strict_pair_build_snapshot() {
  local arm="$1"
  local snapshot
  local manifest
  local entrypoint_manifest
  local symlink_manifest
  local mode_manifest
  local tracked_path
  local source_path
  local snapshot_path
  local source_sha256
  local snapshot_sha256

  if [[ "${arm}" == "off" ]]; then
    snapshot="${STRICT_PAIR_OFF_SNAPSHOT}"
  elif [[ "${arm}" == "on" ]]; then
    snapshot="${STRICT_PAIR_ON_SNAPSHOT}"
  else
    strict_pair_error "unsupported snapshot arm: ${arm}"
  fi
  mkdir -p -- "${STRICT_PAIR_SNAPSHOT_PARENT}"
  strict_pair_require_canonical_dir "${STRICT_PAIR_SNAPSHOT_PARENT}" "strict snapshot parent"
  if ! mkdir -- "${snapshot}" 2>/dev/null; then
    strict_pair_error "strict arm snapshot already exists or is reserved; use a new PAIR_ID: ${snapshot}"
  fi
  rsync -a --from0 --files-from=<(strict_pair_list_snapshot_paths) \
    "${STRICT_PAIR_PROJECT_ROOT}/" "${snapshot}/"
  manifest="${snapshot}/strict-pair-snapshot-manifest.sha256"
  : > "${manifest}"
  while IFS= read -r -d '' tracked_path; do
    source_path="${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}"
    snapshot_path="${snapshot}/${tracked_path}"
    if [[ -L "${source_path}" ]]; then
      if [[ ! -L "${snapshot_path}" || \
            "$(readlink -- "${snapshot_path}")" != "$(readlink -- "${source_path}")" ]]; then
        strict_pair_error "snapshot symlink differs from authenticated source: ${tracked_path}"
      fi
    elif [[ -f "${source_path}" && ! -L "${snapshot_path}" && -f "${snapshot_path}" ]]; then
      source_sha256="$(strict_pair_sha256_file "${source_path}")"
      snapshot_sha256="$(strict_pair_sha256_file "${snapshot_path}")"
      if [[ "${source_sha256}" != "${snapshot_sha256}" ]]; then
        strict_pair_error "snapshot file differs from authenticated source: ${tracked_path}"
      fi
      printf '%s  %s\n' "${snapshot_sha256}" "${tracked_path}" >> "${manifest}"
    else
      strict_pair_error "snapshot is missing a tracked source entry: ${tracked_path}"
    fi
  done < <(strict_pair_list_snapshot_paths)
  symlink_manifest="${snapshot}/strict-pair-snapshot-symlinks.json"
  python3 -I -B - "${snapshot}" "${symlink_manifest}" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
symlinks = {}
for directory, directory_names, file_names in os.walk(root, followlinks=False):
    directory_names.sort()
    file_names.sort()
    for name in list(directory_names):
        path = pathlib.Path(directory) / name
        if path.is_symlink():
            symlinks[path.relative_to(root).as_posix()] = os.readlink(path)
            directory_names.remove(name)
    for name in file_names:
        path = pathlib.Path(directory) / name
        if path.is_symlink():
            symlinks[path.relative_to(root).as_posix()] = os.readlink(path)
document = {
    "schema": "nemo-rl-strict-snapshot-symlinks-v1",
    "symlinks": symlinks,
}
with output.open("x", encoding="ascii", newline="\n") as stream:
    json.dump(document, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
  printf '%s  %s\n' \
    "$(strict_pair_sha256_file "${symlink_manifest}")" \
    "${symlink_manifest#${snapshot}/}" \
    >> "${manifest}"
  entrypoint_manifest="${snapshot}/nano35-entrypoint-manifest.sha256"
  printf '%s  %s\n' \
    "${STRICT_PAIR_ENTRYPOINT_SHA256}" \
    "examples/run_grpo_single_controller.py" \
    > "${entrypoint_manifest}"
  printf '%s  %s\n' \
    "$(strict_pair_sha256_file "${entrypoint_manifest}")" \
    "${entrypoint_manifest#${snapshot}/}" \
    >> "${manifest}"
  mode_manifest="${snapshot}/strict-pair-snapshot-modes.json"
  python3 -I -B - "${snapshot}" "${manifest}" "${mode_manifest}" <<'PY'
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
sha_manifest = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
files = {}
for directory, directory_names, file_names in os.walk(root, followlinks=False):
    directory_names.sort()
    file_names.sort()
    directory_names[:] = [
        name
        for name in directory_names
        if not (pathlib.Path(directory) / name).is_symlink()
    ]
    for name in file_names:
        path = pathlib.Path(directory) / name
        metadata = os.lstat(path)
        if stat.S_ISREG(metadata.st_mode) and path != sha_manifest:
            files[path.relative_to(root).as_posix()] = bool(metadata.st_mode & 0o111)
files[output.relative_to(root).as_posix()] = False
document = {
    "regular_file_executable": files,
    "schema": "nemo-rl-strict-snapshot-modes-v1",
}
with output.open("x", encoding="ascii", newline="\n") as stream:
    json.dump(document, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
  printf '%s  %s\n' \
    "$(strict_pair_sha256_file "${mode_manifest}")" \
    "${mode_manifest#${snapshot}/}" \
    >> "${manifest}"
  chmod 400 "${manifest}"
  chmod 400 "${entrypoint_manifest}"
  chmod 400 "${symlink_manifest}"
  chmod 400 "${mode_manifest}"
  chmod -R a-w "${snapshot}"
  strict_pair_verify_snapshot "${arm}" "${snapshot}"
}

strict_pair_load_snapshots() {
  strict_pair_verify_snapshot off "${STRICT_PAIR_OFF_SNAPSHOT}"
  strict_pair_verify_snapshot on "${STRICT_PAIR_ON_SNAPSHOT}"
}

strict_pair_render_manifest() {
  local output="$1"

  python3 -I -B - \
    "${output}" "${PAIR_ID}" "${RESULTS_DIR}" "${PERSISTENT_CACHE}" \
    "${DEPLOYMENT_ROOT}" "${EXPECTED_DEPLOYMENT_READY}" \
    "${EXPECTED_DEPLOYMENT_READY_FILE_SHA256}" \
    "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" \
    "${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256}" \
    "${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256}" \
    "${MODEL_PATH}" "${STRICT_PAIR_MODEL_TREE_SHA256}" \
    "${CONTAINER}" "${STRICT_PAIR_CONTAINER_SHA256}" \
    "${SANDBOX_CONTAINER}" "${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" \
    "${TRAIN_PATH}" "${STRICT_PAIR_FIXTURE_SHA256}" \
    "${STRICT_PAIR_PROJECT_ROOT}" "${STRICT_PAIR_NEMO_HEAD}" "${STRICT_PAIR_NEMO_TREE}" \
    "${STRICT_PAIR_GYM_ROOT}" "${STRICT_PAIR_GYM_GITLINK_COMMIT}" "${STRICT_PAIR_GYM_TREE}" \
    "${STRICT_PAIR_CONFIG_SHA256}" "${STRICT_PAIR_ENTRYPOINT_SHA256}" \
    "${STRICT_PAIR_LAUNCHER_SHA256}" "${STRICT_PAIR_ARM_WRAPPER_SHA256}" \
    "${STRICT_PAIR_PARENT_WRAPPER_SHA256}" "${STRICT_PAIR_CONTRACT_SHA256}" \
    "${STRICT_PAIR_SNAPSHOT_PARENT}" "${STRICT_PAIR_LAUNCH_MODE}" \
    "${STRICT_PAIR_JOB_WRAPPER}" "${STRICT_PAIR_JOB_WRAPPER_SHA256}" \
    "${STRICT_PAIR_OFF_SNAPSHOT}" "${STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256}" \
    "${STRICT_PAIR_ON_SNAPSHOT}" "${STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output,
    pair_id,
    results_root,
    cache_root,
    deployment_root,
    deployment_ready,
    deployment_ready_file_sha256,
    nemo_manifest_sha256,
    bridge_manifest_sha256,
    mcore_manifest_sha256,
    model_path,
    model_tree_sha256,
    container_path,
    container_sha256,
    sandbox_container_path,
    sandbox_container_sha256,
    fixture_path,
    fixture_sha256,
    source_root,
    source_head,
    source_tree,
    gym_root,
    gym_gitlink_commit,
    gym_tree,
    config_sha256,
    entrypoint_sha256,
    launcher_sha256,
    arm_wrapper_sha256,
    parent_wrapper_sha256,
    contract_sha256,
    snapshot_parent,
    launch_mode,
    job_wrapper_path,
    job_wrapper_sha256,
    off_snapshot_path,
    off_snapshot_manifest_sha256,
    on_snapshot_path,
    on_snapshot_manifest_sha256,
) = sys.argv[1:]

def determinism_marker(mode: str) -> str:
    return (
        "SHARED_PREFIX_DETERMINISM_ATTESTED "
        f"mode={mode} env_controls=4 triton_autotune=absent "
        "model_overrides=3 torch_deterministic=true total_controls=8"
    )


off_marker = determinism_marker("observe")
on_marker = determinism_marker("train")
manifest = {
    "arms": {"off": "observe", "on": "train"},
    "artifacts": {
        "container": {"path": container_path, "sha256": container_sha256},
        "fixture": {
            "path": fixture_path,
            "rows": 5,
            "sha256": fixture_sha256,
        },
        "model": {"path": model_path, "tree_sha256_v1": model_tree_sha256},
        "sandbox_container": {
            "path": sandbox_container_path,
            "sha256": sandbox_container_sha256,
        },
    },
    "campaign": {
        "async_grpo": None,
        "checkpointing_enabled": False,
        "data_plane_enabled": True,
        "data_shuffle": False,
        "epochs": 20,
        "generations_per_prompt": 4,
        "generation": {
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
        },
        "generation_seed_base": 42,
        "launch_mode": launch_mode,
        "logging": {
            "tensorboard_enabled": False,
            "wandb_enabled": True,
            "wandb_entity": "nvidia",
            "wandb_project": "nano35-rlvr-convergence",
        },
        "nodes": 1,
        "padding_multiple": 128,
        "prompts_per_step": 1,
        "require_deterministic_execution": True,
        "reward_and_advantage": {
            "advantage_clip": {"high": 20.0, "low": -20.0},
            "advantage_normalization": True,
            "dynamic_sampling": False,
            "effort_shaping": {
                "low_penalty": 1.0,
                "low_string": "{reasoning effort: efficient}",
                "low_ub": 15000,
                "low_weight": 0.1,
            },
            "invalid_tool_call": {
                "legacy_penalize_flag": True,
                "token_advantage": -5.0,
            },
            "leave_one_out_baseline": True,
            "loss": {
                "force_on_policy_ratio": True,
                "kl_input_clamp_value": None,
                "kl_output_clamp_value": None,
                "ratio_clip_c": None,
                "ratio_clip_max": 0.28,
                "ratio_clip_min": 0.2,
                "reference_policy_kl_penalty": 0.0,
                "reference_policy_kl_type": "k3",
                "sequence_level_importance_ratios": False,
                "token_level": True,
                "truncated_importance_sampling_ratio": 5.0,
                "truncated_importance_sampling_ratio_min": 0.2,
                "truncated_importance_sampling_type": "tis",
                "use_kl_in_reward": False,
                "use_importance_sampling_correction": True,
                "use_on_policy_kl_approximation": True,
            },
            "malformed_thinking_token_advantage": None,
            "metrics": {
                "advantage_estimator_output_before_token_override_and_clip": [
                    "train/advantages/mean",
                    "train/advantages/min",
                    "train/advantages/max",
                ],
                "advantage_estimator_reward_input_after_processing": "train/reward",
                "effective_reward": "train/verifier_reward",
                "effective_reward_legacy_alias": "train/total_reward/mean",
                "final_effective_advantage_after_override_and_clip": None,
                "pre_penalty_reward": "train/pre_penalty_environment_reward",
                "raw_task_score": "train/raw_environment_reward",
                "verifier_native_raw_score_alias": (
                    "train/reasoning_gym_simple_agent/score/mean"
                ),
            },
            "overlong_filtering": False,
            "required_step_relations": {
                "effort_low_sample_rate": "train/effort_low_sample_rate == 0",
                "effort_reward_delta": "train/effort_reward_delta == 0",
                "raw_equals_pre_penalty": (
                    "train/raw_environment_reward == "
                    "train/pre_penalty_environment_reward"
                ),
                "reward_processing_delta": "train/reward_processing_delta == 0",
                "reward_equals_effective": (
                    "train/reward == train/verifier_reward == "
                    "train/total_reward/mean"
                ),
            },
            "reward_scaling": {
                "enabled": False,
                "source_max": 1.0,
                "source_min": 0.0,
                "target_max": 1.0,
                "target_min": 0.0,
            },
            "reward_shaping": {
                "enabled": False,
                "max_response_length": 768,
                "overlong_buffer_length": 128,
                "overlong_buffer_penalty": 1.0,
                "stop_properly_penalty_coef": None,
            },
            "sample_mask": {
                "env_flagged_samples": True,
                "seq_logprob_error_threshold": 2.0,
            },
            "zeroing_penalties": {
                "duplicated_reasoning": True,
                "empty_final_answer": True,
                "malformed_think_tag": True,
                "thinking_tags": ["<think>", "</think>"],
                "token_ids": {"think_close": 13, "think_open": 12, "unwanted": [2]},
                "unwanted_token": True,
            },
        },
        "rollout_seed_opt_in": True,
        "slurm": {
            "account": "nemotron_sw_post",
            "partition": "batch",
            "qos": "normal",
            "walltime": "04:00:00",
        },
        "steps": 100,
        "training_mtp": {
            "detach_heads": True,
            "layers": 5,
            "loss_scale": 0.3,
            "repeated_layer": True,
        },
        "training_topology": "TP2/CP2/PP1/EP4/ETP1/SP",
        "vllm_tp": 4,
    },
    "deployment": {
        "bridge_runnable_manifest_sha256": bridge_manifest_sha256,
        "mcore_runnable_manifest_sha256": mcore_manifest_sha256,
        "nemo_runnable_manifest_sha256": nemo_manifest_sha256,
        "ready": deployment_ready,
        "ready_file_sha256": deployment_ready_file_sha256,
        "root": deployment_root,
    },
    "pair_id": pair_id,
    "paths": {
        "cache_root": cache_root,
        "results_root": results_root,
        "snapshot_parent": snapshot_parent,
    },
    "schema": "nemo-rl-strict-single-env-pair-v1",
    "runtime_attestation": {
        "expected_count_per_fresh_process_group": 4,
        "lines": {
            "off": {
                "mode": "observe",
                "sha256_ascii_no_newline": hashlib.sha256(
                    off_marker.encode("ascii")
                ).hexdigest(),
                "text": off_marker,
            },
            "on": {
                "mode": "train",
                "sha256_ascii_no_newline": hashlib.sha256(
                    on_marker.encode("ascii")
                ).hexdigest(),
                "text": on_marker,
            },
        },
        "receipt_requires_line_count_and_hash": True,
        "schema": "nemo-rl-shared-prefix-determinism-attestation-v1",
    },
    "source": {
        "arm_wrapper_sha256": arm_wrapper_sha256,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "entrypoint_sha256": entrypoint_sha256,
        "gym": {
            "gitlink_commit": gym_gitlink_commit,
            "path": gym_root,
            "tree": gym_tree,
        },
        "head": source_head,
        "job_wrapper": {
            "path": job_wrapper_path,
            "sha256": job_wrapper_sha256,
        },
        "launcher_sha256": launcher_sha256,
        "parent_wrapper_sha256": parent_wrapper_sha256,
        "root": source_root,
        "snapshots": {
            "off": {
                "config_sha256": config_sha256,
                "entrypoint_sha256": entrypoint_sha256,
                "manifest_sha256": off_snapshot_manifest_sha256,
                "path": off_snapshot_path,
            },
            "on": {
                "config_sha256": config_sha256,
                "entrypoint_sha256": entrypoint_sha256,
                "manifest_sha256": on_snapshot_manifest_sha256,
                "path": on_snapshot_path,
            },
        },
        "tree": source_tree,
    },
}
path = pathlib.Path(output)
with path.open("x", encoding="ascii", newline="\n") as stream:
    json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY
}

strict_pair_publish_manifest() {
  local candidate
  local candidate_sha256

  candidate="$(mktemp "${RESULTS_DIR}/.PAIR_MANIFEST.json.candidate.XXXXXX")"
  # The renderer requires exclusive creation, so remove mktemp's placeholder.
  rm -- "${candidate}"
  strict_pair_render_manifest "${candidate}"
  chmod 400 "${candidate}"
  candidate_sha256="$(strict_pair_sha256_file "${candidate}")"

  if ln -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}" 2>/dev/null; then
    rm -- "${candidate}"
  else
    strict_pair_require_canonical_file "${STRICT_PAIR_MANIFEST_PATH}" "PAIR_MANIFEST.json"
    strict_pair_require_mode "${STRICT_PAIR_MANIFEST_PATH}" "400" "PAIR_MANIFEST.json"
    if ! cmp -s -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}"; then
      rm -- "${candidate}"
      strict_pair_error "PAIR_MANIFEST.json already exists with different contract bytes."
      return
    fi
    rm -- "${candidate}"
  fi
  STRICT_PAIR_MANIFEST_SHA256="${candidate_sha256}"
}

strict_pair_verify_manifest() {
  local candidate
  local actual_sha256

  strict_pair_require_digest EXPECTED_PAIR_MANIFEST_SHA256
  strict_pair_require_canonical_file "${STRICT_PAIR_MANIFEST_PATH}" "PAIR_MANIFEST.json"
  strict_pair_require_mode "${STRICT_PAIR_MANIFEST_PATH}" "400" "PAIR_MANIFEST.json"
  actual_sha256="$(strict_pair_sha256_file "${STRICT_PAIR_MANIFEST_PATH}")"
  if [[ "${actual_sha256}" != "${EXPECTED_PAIR_MANIFEST_SHA256}" ]]; then
    strict_pair_error "PAIR_MANIFEST.json SHA-256 differs from the parent-provided anchor."
  fi

  candidate="$(mktemp "${RESULTS_DIR}/.PAIR_MANIFEST.json.verify.XXXXXX")"
  rm -- "${candidate}"
  strict_pair_render_manifest "${candidate}"
  if ! cmp -s -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}"; then
    rm -- "${candidate}"
    strict_pair_error "PAIR_MANIFEST.json bytes differ from recomputed canonical inputs."
    return
  fi
  rm -- "${candidate}"
}
