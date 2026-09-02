# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano"
PAIR_LAUNCHER = RECIPE_DIR / "launch_pair.sh"
ARM_WRAPPER = RECIPE_DIR / "nano35_single_env_pair.sh"
CONFIG = RECIPE_DIR / "single_env_reasoning_gym_sc.yaml"
TEST_FIXTURE = Path(__file__).parent / "data/reasoning_gym_example.jsonl"
EXPECTED_FIXTURE_SHA256 = (
    "da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"
)
ENTRYPOINT = REPO_ROOT / "examples/run_grpo_single_controller.py"


@dataclass(frozen=True)
class PairRun:
    process: subprocess.CompletedProcess[str]
    pair_id: str
    manifest: bytes | None
    manifest_mode: int | None
    export_payloads: dict[str, bytes]
    export_modes: dict[str, int]
    export_files_present_after_process: bool
    sbatch_payloads: dict[str, bytes]
    sbatch_argv: dict[str, tuple[str, ...]]
    scheduler_argv: dict[str, tuple[str, ...]]
    submission_contract: bytes | None
    submission_contract_mode: int | None
    submission_receipt: bytes | None
    submission_receipt_mode: int | None
    submission_receipt_nlink: int | None
    scheduler_states: dict[str, str]
    submission_state_residue: tuple[str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_system_tool(*candidates: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK):
            return path.resolve()
    raise RuntimeError(f"no supported system tool found: {candidates}")


BOOTSTRAP_SHA256SUM = _canonical_system_tool("/usr/bin/sha256sum", "/sbin/sha256sum")
HOST_PYTHON = _canonical_system_tool("/usr/bin/python3")
HOST_TOOLS = {
    "awk": _canonical_system_tool("/usr/bin/awk"),
    "bash": _canonical_system_tool("/bin/bash", "/usr/bin/bash"),
    "cat": _canonical_system_tool("/bin/cat", "/usr/bin/cat"),
    "chmod": _canonical_system_tool("/bin/chmod", "/usr/bin/chmod"),
    "cmp": _canonical_system_tool("/usr/bin/cmp", "/bin/cmp"),
    "date": _canonical_system_tool("/bin/date", "/usr/bin/date"),
    "env": _canonical_system_tool("/usr/bin/env", "/bin/env"),
    "find": _canonical_system_tool("/usr/bin/find", "/bin/find"),
    "git": _canonical_system_tool("/usr/bin/git"),
    "grep": _canonical_system_tool("/usr/bin/grep", "/bin/grep"),
    "ln": _canonical_system_tool("/bin/ln", "/usr/bin/ln"),
    "mkdir": _canonical_system_tool("/bin/mkdir", "/usr/bin/mkdir"),
    "mktemp": _canonical_system_tool("/usr/bin/mktemp", "/bin/mktemp"),
    # Deployment fixtures replace this placeholder with a mode-500 shim. Linux
    # production accepts only the manifest-pinned /usr/bin/nvidia-smi path.
    "nvidia_smi": _canonical_system_tool("/usr/bin/true", "/bin/true"),
    "python": HOST_PYTHON,
    "readlink": _canonical_system_tool("/usr/bin/readlink", "/bin/readlink"),
    "realpath": _canonical_system_tool("/bin/realpath", "/usr/bin/realpath"),
    "rm": _canonical_system_tool("/bin/rm", "/usr/bin/rm"),
    "rsync": _canonical_system_tool("/usr/bin/rsync", "/bin/rsync"),
    # Hermetic dry runs do not invoke this placeholder. Deployment fixtures
    # replace the scheduler tools with sealed Darwin-only stateful shims.
    "sbatch": _canonical_system_tool("/usr/bin/true", "/bin/true"),
    "scancel": _canonical_system_tool("/usr/bin/true", "/bin/true"),
    "scontrol": _canonical_system_tool("/usr/bin/true", "/bin/true"),
    "sha256sum": BOOTSTRAP_SHA256SUM,
    "stat": _canonical_system_tool("/usr/bin/stat", "/bin/stat"),
    "wc": _canonical_system_tool("/usr/bin/wc", "/bin/wc"),
}
CONTAINER_PYTHON = (
    "/root/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
)
CONTAINER_PYTHON_SHA256 = (
    "92ed50fd9dde3654d421d165214a95361e1889210b3d2063001d6c2e75eef2ab"
)
CONTAINER_UV = "/root/.local/bin/uv"
CONTAINER_UV_SHA256 = "b9f74e398b6b15826a4b68b5a83d039036d47df64013e7faf1a9974ec199c144"
STRICT_SETUP_COMMAND = (
    b'echo "[CACHE SETUP] Clearing stale node-local caches; strict pairs forbid '
    b'cache seeds."\n'
    b"rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_* \\\n"
    b"  /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache\n"
    b"mkdir -p /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache\n"
    b'echo "[CACHE SETUP] Done."'
)
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
    "EXPECTED_MCORE_HEAD",
    "EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256",
    "EXPECTED_MCORE_TREE",
    "EXPECTED_NEMO_HEAD",
    "EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_COUNT",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SCHEMA",
    "EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SHA256",
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


def _seal_tracked_files(repository: Path) -> None:
    entries = subprocess.check_output(
        ["git", "-C", str(repository), "ls-files", "--stage", "-z"]
    ).split(b"\0")
    for raw_entry in entries:
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", maxsplit=1)
        mode = metadata.split(b" ", maxsplit=1)[0]
        path = repository / os.fsdecode(raw_path)
        if mode == b"100755":
            path.chmod(0o500)
        elif mode == b"100644":
            path.chmod(0o400)
        elif mode == b"160000":
            _seal_tracked_files(path)


def _rewrite_runnable_manifest_digest(manifest: Path, path: Path) -> None:
    suffix = f"  {path}"
    lines = manifest.read_text(encoding="ascii").splitlines()
    matches = [index for index, line in enumerate(lines) if line.endswith(suffix)]
    assert len(matches) == 1
    lines[matches[0]] = f"{_sha256(path)}{suffix}"
    manifest.chmod(0o600)
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    manifest.chmod(0o400)


def _write_hostile_fake_tools(root: Path) -> Path:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    for name in sorted({*HOST_TOOLS, "python3"}):
        fake_tool = fake_bin / name
        fake_tool.write_text(
            f"#!/bin/bash\necho 'HOSTILE_FAKE_TOOL_EXECUTED={name}' >&2\nexit 73\n",
            encoding="ascii",
        )
        fake_tool.chmod(0o755)
    return fake_bin


def _make_deployment(root: Path, kind: str) -> tuple[Path, Path, Path, dict[str, str]]:
    deployment = root / "deployment"
    nemo_root = deployment / "runnable/NemoRL"
    recipe = nemo_root / "examples/nemo_gym/nemotron-3.5-nano"
    recipe.mkdir(parents=True)
    for name in (
        "launch_pair.sh",
        "nano35_launch.sh",
        "nano35_single_env_pair.sh",
        "rlvr.yaml",
        "rlvr_sc.yaml",
        "single_env_citation_sc.yaml",
        "single_env_freeform_sc.yaml",
        "single_env_reasoning_gym_sc.yaml",
        "strict_pair_contract.sh",
    ):
        shutil.copy2(RECIPE_DIR / name, recipe / name)
    (nemo_root / "examples").mkdir(exist_ok=True)
    shutil.copy2(ENTRYPOINT, nemo_root / "examples/run_grpo_single_controller.py")
    if kind == "publisher_candidate_unlink_failure":
        launcher = recipe / "launch_pair.sh"
        launcher_source = launcher.read_text(encoding="utf-8")
        counter_anchor = (
            "    candidate_cleanup_error = None\n    for _attempt in range(2):\n"
        )
        unlink_anchor = "        try:\n            os.unlink(candidate)\n        except OSError as error:\n"
        assert launcher_source.count(counter_anchor) == 1
        assert launcher_source.count(unlink_anchor) == 1
        launcher_source = launcher_source.replace(
            counter_anchor,
            "    candidate_cleanup_error = None\n"
            "    injected_candidate_unlink_failures = 0\n"
            "    for _attempt in range(2):\n",
        ).replace(
            unlink_anchor,
            "        try:\n"
            "            injected_candidate_unlink_failures += 1\n"
            "            if injected_candidate_unlink_failures <= 2:\n"
            "                raise OSError('injected candidate unlink failure')\n"
            "            os.unlink(candidate)\n"
            "        except OSError as error:\n",
        )
        launcher.write_text(launcher_source, encoding="utf-8")
    (nemo_root / "tools").mkdir()
    shutil.copy2(
        REPO_ROOT / "tools/code_snapshot.sh", nemo_root / "tools/code_snapshot.sh"
    )
    shutil.copy2(REPO_ROOT / ".gitignore", nemo_root / ".gitignore")
    if kind == "valid_symlink":
        (nemo_root / "tracked-runtime-link").symlink_to(".gitignore")
    subprocess.run(["git", "init", "-q", str(nemo_root)], check=True)
    gym_source = root / "gym-source"
    subprocess.run(["git", "init", "-q", str(gym_source)], check=True)
    (gym_source / "gym_runtime.py").write_text(
        "GYM_RUNTIME_VERSION = 1\n", encoding="ascii"
    )
    for relative in (
        "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
        "resources_servers/reasoning_gym/requirements.txt",
        "resources_servers/reasoning_gym/app.py",
        "resources_servers/format_verification/configs/citation_format.yaml",
        "resources_servers/format_verification/configs/freeform_formatting.yaml",
        "resources_servers/format_verification/requirements.txt",
        "resources_servers/format_verification/app.py",
    ):
        asset = gym_source / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text(f"# test Gym asset: {relative}\n", encoding="ascii")
    subprocess.run(["git", "-C", str(gym_source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(gym_source),
            "-c",
            "user.name=NeMo RL test",
            "-c",
            "user.email=nemo-rl-test@nvidia.com",
            "commit",
            "-qm",
            "test: construct Gym submodule",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(nemo_root),
            "submodule",
            "add",
            "-q",
            str(gym_source),
            "3rdparty/Gym-workspace/Gym",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(nemo_root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(nemo_root),
            "-c",
            "user.name=NeMo RL test",
            "-c",
            "user.email=nemo-rl-test@nvidia.com",
            "commit",
            "-qm",
            "test: construct strict-pair deployment",
        ],
        check=True,
    )
    if kind == "wrong_gitlink":
        gym_checkout = nemo_root / "3rdparty/Gym-workspace/Gym"
        (gym_checkout / "gym_runtime.py").write_text(
            "GYM_RUNTIME_VERSION = 2\n", encoding="ascii"
        )
        subprocess.run(["git", "-C", str(gym_checkout), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(gym_checkout),
                "-c",
                "user.name=NeMo RL test",
                "-c",
                "user.email=nemo-rl-test@nvidia.com",
                "commit",
                "-qm",
                "test: drift Gym gitlink",
            ],
            check=True,
        )

    component_roots = {
        "Megatron-Bridge.runnable.sha256": (
            deployment / "runnable/Megatron-Bridge"
        ),
        "Megatron-LM.runnable.sha256": deployment / "runnable/Megatron-LM",
    }
    for component_root in component_roots.values():
        component_root.mkdir(parents=True)
        (component_root / "runtime.py").write_text(
            f'COMPONENT = "{component_root.name}"\n', encoding="ascii"
        )
        subprocess.run(["git", "init", "-q", str(component_root)], check=True)
        subprocess.run(["git", "-C", str(component_root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(component_root),
                "-c",
                "user.name=NeMo RL test",
                "-c",
                "user.email=nemo-rl-test@nvidia.com",
                "commit",
                "-qm",
                "test: construct deployed component",
            ],
            check=True,
        )
    job_wrapper = deployment / "strict_pair_job_wrapper.sh"
    job_wrapper.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="ascii")
    job_wrapper.chmod(0o500)
    uv_shim = deployment / "strict_pair_uv.sh"
    uv_shim.write_text(
        '#!/bin/bash -p\nexec /root/.local/bin/uv "$@"\n', encoding="ascii"
    )
    uv_shim.chmod(0o500)
    off_job_id = "41001"
    on_job_id = "41002"
    if kind == "sbatch_empty":
        off_job_id = on_job_id = ""
    elif kind == "recovery_malformed_identity_no_relay":
        off_job_id = on_job_id = ""
    elif kind == "sbatch_nonnumeric":
        off_job_id = on_job_id = "not-a-job-id"
    elif kind in {"sbatch_duplicate", "sbatch_duplicate_recovery_query_failure"}:
        off_job_id = on_job_id = "41001"
    elif kind in {"recovery_partial_visibility", "recovery_unterminated_candidate"}:
        off_job_id = "51001"
        on_job_id = "51002"
    if kind in {
        "recovery_malformed_identity_no_relay",
        "recovery_partial_visibility",
        "recovery_unterminated_candidate",
    }:
        off_scheduler_job_id = "41001"
        on_scheduler_job_id = "41002"
    else:
        off_scheduler_job_id = off_job_id
        on_scheduler_job_id = on_job_id
    scheduler_state = root / "scheduler-state"
    scheduler_state.mkdir(mode=0o700)
    sbatch_shim = deployment / "strict_pair_test_sbatch.sh"
    sbatch_source = """#!/bin/bash -p
set -euo pipefail
arm="${!#}"
case "${arm}" in
  off) job_id=__OFF_JOB_ID__; scheduler_job_id=__OFF_SCHEDULER_JOB_ID__ ;;
  on) job_id=__ON_JOB_ID__; scheduler_job_id=__ON_SCHEDULER_JOB_ID__ ;;
  *) echo "unexpected test arm: ${arm}" >&2; exit 64 ;;
esac
if [[ "__KIND__" == "sbatch_on_failure" && "${arm}" == "on" ]]; then
  exit 79
fi
export_file=""
comment=""
job_name=""
chdir=""
parsable_count=0
hold_count=0
for argument in "$@"; do
  case "${argument}" in
    --export-file=*) export_file="${argument#--export-file=}" ;;
    --comment=*) comment="${argument#--comment=}" ;;
    --job-name=*) job_name="${argument#--job-name=}" ;;
    --chdir=*) chdir="${argument#--chdir=}" ;;
    --parsable) parsable_count=$((parsable_count + 1)) ;;
    --hold) hold_count=$((hold_count + 1)) ;;
  esac
done
[[ -n "${export_file}" && -f "${export_file}" ]] || exit 65
[[ -n "${comment}" && -n "${job_name}" && -n "${chdir}" ]] || exit 65
[[ "${PWD}" == "${chdir}" ]] || exit 67
[[ "${parsable_count}" == 1 && "${hold_count}" == 1 ]] || exit 66
/bin/cp -- "${export_file}" "__STATE__/sbatch-${arm}.env"
printf '%s\\0' "$@" > "__STATE__/sbatch-${arm}-argv.nul"
if [[ "${scheduler_job_id}" =~ ^[0-9]+$ ]]; then
  printf 'HELD\\n' > "__STATE__/${scheduler_job_id}.state"
  printf '%s\\n' "${comment}" > "__STATE__/${scheduler_job_id}.comment"
  printf '%s\\n' "${job_name}" > "__STATE__/${scheduler_job_id}.job-name"
fi
if [[ "${scheduler_job_id}" =~ ^[0-9]+$ ]]; then
  printf '%s' "${chdir}" > "__STATE__/${scheduler_job_id}.work-dir"
fi
if [[ "__KIND__" == "recovery_lower_id" && "${arm}" == "off" ]]; then
  printf 'HELD\\n' > "__STATE__/31001.state"
  printf '%s\\n' "${comment}" > "__STATE__/31001.comment"
  printf '%s\\n' "${job_name}" > "__STATE__/31001.job-name"
fi
if [[ "__KIND__" == "recovery_lower_id" && "${arm}" == "off" ]]; then
  printf '%s' "${chdir}" > "__STATE__/31001.work-dir"
fi
printf '%s\n' "${job_id}"
"""
    sbatch_source = (
        sbatch_source.replace("__OFF_JOB_ID__", off_job_id)
        .replace("__ON_JOB_ID__", on_job_id)
        .replace("__OFF_SCHEDULER_JOB_ID__", off_scheduler_job_id)
        .replace("__ON_SCHEDULER_JOB_ID__", on_scheduler_job_id)
        .replace("__STATE__", str(scheduler_state))
        .replace("__KIND__", kind)
    )
    sbatch_shim.write_text(
        sbatch_source,
        encoding="ascii",
    )
    sbatch_shim.chmod(0o500)
    scontrol_shim = deployment / "strict_pair_test_scontrol.sh"
    scontrol_shim.write_text(
        """#!/bin/bash -p
set -euo pipefail
state_root="__STATE__"
kind="__KIND__"
if [[ "$1" == "release" ]]; then
  printf '%s\\0' "$@" > "${state_root}/scontrol-release-argv.nul"
  [[ "$#" == 2 ]] || exit 70
  if [[ "${kind}" == "scontrol_release_failure" ]]; then
    exit 71
  fi
  IFS=, read -r off_id on_id <<< "$2"
  [[ -n "${off_id}" && -n "${on_id}" ]] || exit 72
  printf 'RELEASED\\n' > "${state_root}/${off_id}.state"
  if [[ "${kind}" != "scontrol_asymmetric" ]]; then
    printf 'RELEASED\\n' > "${state_root}/${on_id}.state"
  fi
  exit 0
fi
if [[ "$1" != "show" || "$2" != "job" || "$3" != "--oneliner" || \
      ( "$#" != 3 && "$#" != 4 ) ]]; then
  exit 73
fi
query_count=0
if [[ -f "${state_root}/query-count" ]]; then
  query_count=$(/bin/cat "${state_root}/query-count")
fi
query_count=$((query_count + 1))
printf '%s\\n' "${query_count}" > "${state_root}/query-count"
printf '%s\\0' "$@" > "${state_root}/scontrol-query-${query_count}-argv.nul"
ids=()
if [[ "$#" == 3 ]]; then
  shopt -s nullglob
  for state_path in "${state_root}"/*.state; do
    state_id="${state_path##*/}"
    state_id="${state_id%.state}"
    ids+=("${state_id}")
  done
  shopt -u nullglob
else
  IFS=, read -r -a ids <<< "$4"
fi
if [[ "${kind}" == "recovery_partial_visibility" && "${query_count}" == 1 ]]; then
  ids=(41001)
fi
if [[ "${kind}" == "recovery_unterminated_candidate" && "${query_count}" == 1 ]]; then
  job_id=41001
  state=$(/bin/cat "${state_root}/${job_id}.state")
  comment=$(/bin/cat "${state_root}/${job_id}.comment")
  job_name=$(/bin/cat "${state_root}/${job_id}.job-name")
  work_dir=$(/bin/cat "${state_root}/${job_id}.work-dir")
  comment="${comment} WorkDir=${work_dir}"
  printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=JobHeldUser' "${job_id}" "${job_name}" "${comment}"
  exit 0
fi
if [[ "${kind}" == "sbatch_duplicate_recovery_query_failure" && "${query_count}" == 1 ]]; then
  exit 75
fi
for index in "${!ids[@]}"; do
  job_id="${ids[$index]}"
  state=$(/bin/cat "${state_root}/${job_id}.state")
  comment=$(/bin/cat "${state_root}/${job_id}.comment")
  job_name=$(/bin/cat "${state_root}/${job_id}.job-name")
  work_dir=$(/bin/cat "${state_root}/${job_id}.work-dir")
  if [[ "${kind}" == "scheduler_wrong_workdir" && "${job_name}" == on-* ]]; then
    work_dir="${work_dir}/wrong"
  fi
  comment="${comment} WorkDir=${work_dir}"
  if [[ "${kind}" == "recovery_malformed_identity_no_relay" && \
        "${query_count}" == 1 && "${index}" == 0 ]]; then
    printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING\n' "${job_id}" "${job_name}" "${comment}"
    continue
  fi
  if [[ "${index}" == "$((${#ids[@]} - 1))" && \
        ( ( "${kind}" == "pre_release_unterminated" && "${query_count}" == 2 ) || \
          ( "${kind}" == "post_release_unterminated" && "${query_count}" == 3 ) ) ]]; then
    case "${state}" in
      HELD) printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=JobHeldUser' "${job_id}" "${job_name}" "${comment}" ;;
      RELEASED) printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=Resources' "${job_id}" "${job_name}" "${comment}" ;;
      *) exit 74 ;;
    esac
    exit 0
  fi
  case "${state}" in
    HELD) printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=JobHeldUser\\n' "${job_id}" "${job_name}" "${comment}" ;;
    RELEASED)
      if [[ "${kind}" == "post_release_held_reason_running" && "${query_count}" == 3 ]]; then
        printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=RUNNING Reason=JobHeldUser\\n' "${job_id}" "${job_name}" "${comment}"
        continue
      fi
      if [[ "${index}" == 0 ]]; then
        printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=RUNNING Reason=None\\n' "${job_id}" "${job_name}" "${comment}"
      else
        printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=Resources\\n' "${job_id}" "${job_name}" "${comment}"
      fi
      ;;
    CANCELLED) printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=CANCELLED Reason=None\\n' "${job_id}" "${job_name}" "${comment}" ;;
    *) exit 74 ;;
  esac
  if [[ "${kind}" == "scontrol_query_failure" && "${query_count}" == 2 && "${index}" == 0 ]]; then
    exit 75
  fi
done
if [[ "${kind}" == "scontrol_duplicate_query" && "${query_count}" == 1 ]]; then
  first_id="${ids[0]}"
  first_comment=$(/bin/cat "${state_root}/${first_id}.comment")
  first_job_name=$(/bin/cat "${state_root}/${first_id}.job-name")
  first_work_dir=$(/bin/cat "${state_root}/${first_id}.work-dir")
  first_comment="${first_comment} WorkDir=${first_work_dir}"
  printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING Reason=JobHeldUser\\n' "${first_id}" "${first_job_name}" "${first_comment}"
fi
if [[ ( "${kind}" == "pre_release_malformed_duplicate" && "${query_count}" == 2 ) || \
      ( "${kind}" == "post_release_malformed_duplicate" && "${query_count}" == 3 ) ]]; then
  first_id="${ids[0]}"
  first_comment=$(/bin/cat "${state_root}/${first_id}.comment")
  first_job_name=$(/bin/cat "${state_root}/${first_id}.job-name")
  first_work_dir=$(/bin/cat "${state_root}/${first_id}.work-dir")
  first_comment="${first_comment} WorkDir=${first_work_dir}"
  printf 'JobId=%s JobName=%s UserId=test(__EUID__) Comment=%s JobState=PENDING\\n' "${first_id}" "${first_job_name}" "${first_comment}"
fi
""".replace("__STATE__", str(scheduler_state))
        .replace("__KIND__", kind)
        .replace("__EUID__", str(os.geteuid())),
        encoding="ascii",
    )
    scontrol_shim.chmod(0o500)
    scancel_shim = deployment / "strict_pair_test_scancel.sh"
    scancel_shim.write_text(
        """#!/bin/bash -p
set -euo pipefail
state_root="__STATE__"
kind="__KIND__"
printf '%s\\0' "$@" > "${state_root}/scancel-argv.nul"
[[ "$#" == 1 ]] || exit 76
if [[ "${kind}" == "scancel_failure" ]]; then
  exit 77
fi
if [[ "${kind}" != "scancel_no_effect" ]]; then
  IFS=, read -r -a ids <<< "$1"
  for job_id in "${ids[@]}"; do
    [[ -z "${job_id}" ]] || printf 'CANCELLED\\n' > "${state_root}/${job_id}.state"
  done
fi
""".replace("__STATE__", str(scheduler_state)).replace("__KIND__", kind),
        encoding="ascii",
    )
    scancel_shim.chmod(0o500)
    nvidia_smi_shim = deployment / "strict_pair_test_nvidia_smi.sh"
    nvidia_smi_shim.write_text(
        """#!/bin/bash -p
set -euo pipefail
[[ "$#" == 2 ]]
[[ "$1" == "--query-gpu=name,driver_version" ]]
[[ "$2" == "--format=csv,noheader,nounits" ]]
printf 'NVIDIA GB200, 575.57.08\\n%.0s' {1..4}
""",
        encoding="ascii",
    )
    nvidia_smi_shim.chmod(0o500)
    slurm_conf = deployment / "strict_pair_test_slurm.conf"
    slurm_conf.write_text("ClusterName=strict-pair-test\n", encoding="ascii")
    slurm_conf.chmod(0o400)
    runtime_tool_manifest = deployment / "strict_pair_runtime_tools.json"
    host_tool_records = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in HOST_TOOLS.items()
    }
    host_tool_records["sbatch"] = {
        "path": str(sbatch_shim),
        "sha256": _sha256(sbatch_shim),
    }
    host_tool_records["scancel"] = {
        "path": str(scancel_shim),
        "sha256": _sha256(scancel_shim),
    }
    host_tool_records["scontrol"] = {
        "path": str(scontrol_shim),
        "sha256": _sha256(scontrol_shim),
    }
    host_tool_records["nvidia_smi"] = {
        "path": str(nvidia_smi_shim),
        "sha256": _sha256(nvidia_smi_shim),
    }
    runtime_tool_document = {
        "schema": "nemo-rl-strict-runtime-tools-v2",
        "host": host_tool_records,
        "container": {
            "python": {
                "path": CONTAINER_PYTHON,
                "sha256": CONTAINER_PYTHON_SHA256,
            },
            "uv": {"path": CONTAINER_UV, "sha256": CONTAINER_UV_SHA256},
            "uv_shim": {"path": str(uv_shim), "sha256": _sha256(uv_shim)},
        },
    }
    runtime_tool_manifest.write_text(
        json.dumps(runtime_tool_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    runtime_tool_manifest.chmod(0o400)
    payloads = deployment / "payloads"
    payloads.mkdir(parents=True)
    ready_identity = "a" * 64
    ready = deployment / "READY"
    ready.write_text(f"{ready_identity}\n", encoding="ascii")
    ready.chmod(0o444)

    manifests: dict[str, Path] = {}
    expected_names = {
        "NemoRL.runnable.sha256": "EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256",
        "Megatron-Bridge.runnable.sha256": ("EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256"),
        "Megatron-LM.runnable.sha256": "EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256",
    }
    for index, manifest_name in enumerate(expected_names):
        payload = payloads / f"payload-{index}.txt"
        payload.write_text(f"deployment payload {index}\n", encoding="ascii")
        manifest = deployment / manifest_name
        if manifest_name == "NemoRL.runnable.sha256":
            tracked = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(nemo_root),
                    "ls-files",
                    "--recurse-submodules",
                    "-z",
                ]
            ).split(b"\0")
            entries = []
            for raw_relative in tracked:
                if not raw_relative:
                    continue
                tracked_path = nemo_root / os.fsdecode(raw_relative)
                if tracked_path.is_symlink() or tracked_path.is_dir():
                    continue
                entries.append(f"{_sha256(tracked_path)}  {tracked_path}")
            entries.append(f"{_sha256(job_wrapper)}  {job_wrapper}")
            entries.append(f"{_sha256(runtime_tool_manifest)}  {runtime_tool_manifest}")
            entries.append(f"{_sha256(uv_shim)}  {uv_shim}")
            entries.append(f"{_sha256(sbatch_shim)}  {sbatch_shim}")
            entries.append(f"{_sha256(scancel_shim)}  {scancel_shim}")
            entries.append(f"{_sha256(scontrol_shim)}  {scontrol_shim}")
            entries.append(f"{_sha256(slurm_conf)}  {slurm_conf}")
            manifest.write_text("\n".join(sorted(entries)) + "\n", encoding="ascii")
        else:
            component_root = component_roots[manifest_name]
            tracked = subprocess.check_output(
                ["git", "-C", str(component_root), "ls-files", "-z"]
            ).split(b"\0")
            entries = [
                f"{_sha256(component_root / os.fsdecode(relative))}  "
                f"{component_root / os.fsdecode(relative)}"
                for relative in tracked
                if relative
            ]
            manifest.write_text("\n".join(sorted(entries)) + "\n", encoding="ascii")
        manifest.chmod(0o400)
        manifests[manifest_name] = manifest

    env = {
        "DEPLOYMENT_ROOT": str(deployment),
        "EXPECTED_DEPLOYMENT_READY": ready_identity,
        "EXPECTED_DEPLOYMENT_READY_FILE_SHA256": _sha256(ready),
        "EXPECTED_NEMO_HEAD": subprocess.check_output(
            ["git", "-C", str(nemo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "EXPECTED_GYM_GITLINK_COMMIT": subprocess.check_output(
            ["git", "-C", str(gym_source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "EXPECTED_BRIDGE_HEAD": subprocess.check_output(
            ["git", "-C", str(component_roots["Megatron-Bridge.runnable.sha256"]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "EXPECTED_BRIDGE_TREE": subprocess.check_output(
            ["git", "-C", str(component_roots["Megatron-Bridge.runnable.sha256"]), "rev-parse", "HEAD^{tree}"],
            text=True,
        ).strip(),
        "EXPECTED_MCORE_HEAD": subprocess.check_output(
            ["git", "-C", str(component_roots["Megatron-LM.runnable.sha256"]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "EXPECTED_MCORE_TREE": subprocess.check_output(
            ["git", "-C", str(component_roots["Megatron-LM.runnable.sha256"]), "rev-parse", "HEAD^{tree}"],
            text=True,
        ).strip(),
    }
    env.update(
        {
            expected_name: _sha256(manifests[manifest_name])
            for manifest_name, expected_name in expected_names.items()
        }
    )
    env["STRICT_PAIR_JOB_WRAPPER"] = str(job_wrapper)
    env["EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256"] = _sha256(job_wrapper)
    env.update(
        {
            "STRICT_PAIR_RUNTIME_TOOL_MANIFEST": str(runtime_tool_manifest),
            "EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256": _sha256(
                runtime_tool_manifest
            ),
            "STRICT_PAIR_HOST_PYTHON": str(HOST_PYTHON),
            "EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256": _sha256(HOST_PYTHON),
            "EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256": _sha256(
                BOOTSTRAP_SHA256SUM
            ),
            "STRICT_PAIR_CONTAINER_PYTHON": CONTAINER_PYTHON,
            "EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256": (CONTAINER_PYTHON_SHA256),
            "STRICT_PAIR_CONTAINER_UV": CONTAINER_UV,
            "EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256": CONTAINER_UV_SHA256,
            "STRICT_PAIR_UV_SHIM": str(uv_shim),
            "EXPECTED_STRICT_PAIR_UV_SHIM_SHA256": _sha256(uv_shim),
            "STRICT_PAIR_SLURM_CONF": str(slurm_conf),
            "EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256": _sha256(slurm_conf),
        }
    )

    if kind in {
        "valid",
        "valid_submodule",
        "valid_symlink",
        "wrong_gitlink",
        "sbatch_empty",
        "sbatch_nonnumeric",
        "sbatch_duplicate",
        "sbatch_duplicate_recovery_query_failure",
        "sbatch_on_failure",
        "publisher_candidate_unlink_failure",
        "pre_release_unterminated",
        "pre_release_malformed_duplicate",
        "post_release_unterminated",
        "post_release_held_reason_running",
        "post_release_malformed_duplicate",
        "recovery_lower_id",
        "recovery_malformed_identity_no_relay",
        "recovery_partial_visibility",
        "recovery_unterminated_candidate",
        "scheduler_wrong_workdir",
        "scancel_failure",
        "scancel_no_effect",
        "scontrol_asymmetric",
        "scontrol_duplicate_query",
        "scontrol_query_failure",
        "scontrol_release_failure",
    }:
        pass
    elif kind == "root_relative":
        env["DEPLOYMENT_ROOT"] = deployment.name
    elif kind == "root_noncanonical":
        (root / "component").mkdir()
        env["DEPLOYMENT_ROOT"] = str(root / "component" / ".." / deployment.name)
    elif kind == "root_symlink":
        alias = root / "deployment-alias"
        alias.symlink_to(deployment, target_is_directory=True)
        env["DEPLOYMENT_ROOT"] = str(alias)
    elif kind == "root_mode":
        pass
    elif kind == "ready_missing":
        ready.unlink()
    elif kind == "ready_symlink":
        target = deployment / "READY.target"
        ready.replace(target)
        ready.symlink_to(target)
    elif kind == "ready_mode":
        ready.chmod(0o600)
    elif kind == "ready_hash":
        env["EXPECTED_DEPLOYMENT_READY_FILE_SHA256"] = "0" * 64
    elif kind == "ready_content":
        env["EXPECTED_DEPLOYMENT_READY"] = "b" * 64
    elif kind.startswith("nemo_manifest_"):
        manifest = manifests["NemoRL.runnable.sha256"]
        operation = kind.removeprefix("nemo_manifest_")
        if operation == "missing":
            manifest.unlink()
        elif operation == "symlink":
            target = deployment / "NemoRL.runnable.sha256.target"
            manifest.replace(target)
            manifest.symlink_to(target)
        elif operation == "mode":
            manifest.chmod(0o600)
        elif operation == "hash":
            env["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = "0" * 64
        elif operation == "payload_mutated":
            (nemo_root / "examples/run_grpo_single_controller.py").write_text(
                "# mutated deployment source\n", encoding="ascii"
            )
        elif operation == "malformed":
            manifest.chmod(0o600)
            manifest.write_text("not a sha256 manifest\n", encoding="ascii")
            manifest.chmod(0o400)
            env["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = _sha256(manifest)
        else:
            raise ValueError(f"unknown deployment kind: {kind}")
    elif kind == "job_wrapper_missing":
        job_wrapper.unlink()
    elif kind == "job_wrapper_symlink":
        target = deployment / "strict_pair_job_wrapper.target.sh"
        job_wrapper.replace(target)
        job_wrapper.symlink_to(target)
    elif kind == "job_wrapper_writable":
        job_wrapper.chmod(0o700)
    elif kind == "job_wrapper_hash":
        env["EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256"] = "0" * 64
    elif kind == "job_wrapper_mutated":
        job_wrapper.chmod(0o700)
        with job_wrapper.open("a", encoding="ascii") as stream:
            stream.write("# mutation\n")
        job_wrapper.chmod(0o500)
    elif kind == "job_wrapper_not_listed":
        manifest = manifests["NemoRL.runnable.sha256"]
        manifest.chmod(0o600)
        manifest.write_text(
            "".join(
                line
                for line in manifest.read_text(encoding="ascii").splitlines(True)
                if str(job_wrapper) not in line
            ),
            encoding="ascii",
        )
        manifest.chmod(0o400)
        env["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = _sha256(manifest)
    elif kind == "source_staged":
        subprocess.run(
            [
                "git",
                "-C",
                str(nemo_root),
                "update-index",
                "--chmod=+x",
                ".gitignore",
            ],
            check=True,
        )
    elif kind == "source_unstaged":
        dirty_path = nemo_root / ".gitignore"
        dirty_path.write_text(
            "__pycache__/\n# unstaged tracked drift\n", encoding="ascii"
        )
        _rewrite_runnable_manifest_digest(
            manifests["NemoRL.runnable.sha256"], dirty_path
        )
        env["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = _sha256(
            manifests["NemoRL.runnable.sha256"]
        )
    elif kind == "gym_unstaged":
        dirty_path = (
            nemo_root
            / "3rdparty/Gym-workspace/Gym/resources_servers/reasoning_gym/app.py"
        )
        dirty_path.write_text("# unstaged Gym drift\n", encoding="ascii")
        _rewrite_runnable_manifest_digest(
            manifests["NemoRL.runnable.sha256"], dirty_path
        )
        env["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = _sha256(
            manifests["NemoRL.runnable.sha256"]
        )
    elif kind == "bridge_unstaged":
        dirty_path = (
            component_roots["Megatron-Bridge.runnable.sha256"] / "runtime.py"
        )
        dirty_path.write_text("# unstaged Bridge drift\n", encoding="ascii")
        _rewrite_runnable_manifest_digest(
            manifests["Megatron-Bridge.runnable.sha256"], dirty_path
        )
        env["EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256"] = _sha256(
            manifests["Megatron-Bridge.runnable.sha256"]
        )
    elif kind == "mcore_unstaged":
        dirty_path = component_roots["Megatron-LM.runnable.sha256"] / "runtime.py"
        dirty_path.write_text("# unstaged MCore drift\n", encoding="ascii")
        _rewrite_runnable_manifest_digest(
            manifests["Megatron-LM.runnable.sha256"], dirty_path
        )
        env["EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256"] = _sha256(
            manifests["Megatron-LM.runnable.sha256"]
        )
    elif kind == "source_untracked":
        (nemo_root / "unlisted_runtime.py").write_text(
            "raise RuntimeError('must not run')\n", encoding="ascii"
        )
    elif kind == "source_ignored":
        ignored = nemo_root / "__pycache__/unlisted_runtime.pyc"
        ignored.parent.mkdir()
        ignored.write_bytes(b"ignored runtime bytes\n")
    elif kind == "source_mode_drift":
        pass
    else:
        raise ValueError(f"unknown deployment kind: {kind}")
    _seal_tracked_files(nemo_root)
    for component_root in component_roots.values():
        _seal_tracked_files(component_root)
        component_root.chmod(0o500)
    if kind == "source_mode_drift":
        (nemo_root / ".gitignore").chmod(0o500)
    nemo_root.chmod(0o500)
    deployment.chmod(0o700 if kind == "root_mode" else 0o500)
    return deployment, nemo_root, job_wrapper, env


def _drift_published_snapshot(snapshot: Path, kind: str) -> Path | None:
    entrypoint = snapshot / "examples/run_grpo_single_controller.py"
    if kind == "missing":
        entrypoint.parent.chmod(0o700)
        entrypoint.unlink()
        entrypoint.parent.chmod(0o500)
    elif kind == "mutated":
        entrypoint.chmod(0o600)
        entrypoint.write_bytes(entrypoint.read_bytes() + b"\n# mutation\n")
        entrypoint.chmod(0o400)
    elif kind == "symlink":
        entrypoint.parent.chmod(0o700)
        entrypoint.unlink()
        entrypoint.symlink_to(snapshot / ".gitignore")
        entrypoint.parent.chmod(0o500)
    elif kind == "missing_gym":
        gym_file = snapshot / "3rdparty/Gym-workspace/Gym/gym_runtime.py"
        gym_file.parent.chmod(0o700)
        gym_file.unlink()
        gym_file.parent.chmod(0o500)
    elif kind == "extra_readonly":
        snapshot.chmod(0o700)
        extra = snapshot / "unlisted_runtime.py"
        extra.write_text("raise RuntimeError('unlisted')\n", encoding="ascii")
        extra.chmod(0o400)
        snapshot.chmod(0o500)
    elif kind == "rewrite_manifest":
        manifest = snapshot / "strict-pair-snapshot-manifest.sha256"
        entrypoint.chmod(0o600)
        manifest.chmod(0o600)
        entrypoint.write_bytes(entrypoint.read_bytes() + b"\n# mutation\n")
        replacement = _sha256(entrypoint)
        lines = []
        for line in manifest.read_text(encoding="ascii").splitlines():
            digest, relative = line.split("  ", maxsplit=1)
            if relative == "examples/run_grpo_single_controller.py":
                digest = replacement
            lines.append(f"{digest}  {relative}")
        manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
        entrypoint.chmod(0o400)
        manifest.chmod(0o400)
    elif kind == "symlink_target":
        link = snapshot / "tracked-runtime-link"
        snapshot.chmod(0o700)
        link.unlink()
        link.symlink_to("examples/run_grpo_single_controller.py")
        snapshot.chmod(0o500)
    elif kind == "mode_drift":
        entrypoint.chmod(0o500)
    elif kind == "unreadable_directory":
        snapshot.chmod(0o700)
        unreadable = snapshot / "unreadable-runtime-directory"
        unreadable.mkdir(mode=0o700)
        unreadable.chmod(0)
        snapshot.chmod(0o500)
        return unreadable
    else:
        raise ValueError(f"unknown snapshot drift: {kind}")
    return None


def _drift_published_pair_manifest(manifest: Path, kind: str) -> None:
    if kind == "missing":
        manifest.unlink()
    elif kind == "symlink":
        target = manifest.with_name("PAIR_MANIFEST.target.json")
        manifest.replace(target)
        manifest.symlink_to(target)
    elif kind == "mode":
        manifest.chmod(0o600)
    elif kind == "mutation":
        manifest.chmod(0o600)
        manifest.write_bytes(manifest.read_bytes() + b"mutation\n")
        manifest.chmod(0o400)
    else:
        raise ValueError(f"unknown Pair-manifest drift: {kind}")


def _run_pair(
    *arguments: str,
    use_parent: bool = True,
    use_deployed_script: bool = True,
    include_wandb_key: bool = True,
    include_train_path: bool = True,
    fixture_kind: str = "valid",
    deployment_kind: str = "valid",
    preexisting_snapshot: bool = False,
    preexisting_pair_manifest: str | None = None,
    preexisting_submission_receipt: bytes | None = None,
    startup_attack: str | None = None,
    direct_invocation: bool = True,
    post_publication_snapshot_tamper: str | None = None,
    post_publication_pair_manifest_tamper: str | None = None,
    preseed_cache_read_arm: str | None = None,
    cross_arm_write_root_alias: str | None = None,
    **env_overrides: str,
) -> PairRun:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        pair_id = env_overrides.pop("PAIR_ID", f"strict-{root.name}")
        fake_bin = _write_hostile_fake_tools(root)
        fixture = root / "reasoning_gym_example.jsonl"
        shutil.copyfile(TEST_FIXTURE, fixture)
        train_path = str(fixture)
        if fixture_kind == "mutated":
            payload = bytearray(fixture.read_bytes())
            payload[0] = ord("[")
            fixture.write_bytes(payload)
        elif fixture_kind == "wrong_hash":
            fixture.write_text("{}\n" * 5, encoding="utf-8")
        elif fixture_kind == "symlink":
            target = root / "fixture-target.jsonl"
            fixture.replace(target)
            fixture.symlink_to(target)
        elif fixture_kind == "relative":
            train_path = fixture.name
        elif fixture_kind == "directory":
            train_path = str(root)
        elif fixture_kind == "fifo":
            fixture.unlink()
            os.mkfifo(fixture)
        elif fixture_kind == "noncanonical":
            (root / "subdirectory").mkdir()
            train_path = str(root / "subdirectory" / ".." / fixture.name)
        elif fixture_kind != "valid":
            raise ValueError(f"unknown fixture_kind: {fixture_kind}")

        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model":"test"}\n', encoding="ascii")
        container = root / "nemo-rl.sqsh"
        container.write_bytes(b"test training container\n")
        sandbox_container = root / "sandbox.sqsh"
        sandbox_container.write_bytes(b"test sandbox container\n")
        results = root / "results"
        cache = root / "cache"
        hf_home = root / "hf-cache"
        results.mkdir()
        results.chmod(0o700)
        cache.mkdir()
        hf_home.mkdir()
        if preseed_cache_read_arm is not None:
            assert preseed_cache_read_arm in {"off", "on"}
            cache_read = cache / preseed_cache_read_arm / "cache_read"
            cache_read.mkdir(parents=True)
            (cache_read / "unbound.tar.zst").write_bytes(b"hostile preseed\n")
        if cross_arm_write_root_alias is not None:
            assert cross_arm_write_root_alias in {"persistent_cache", "hf_home"}
            write_root = {
                "persistent_cache": cache,
                "hf_home": hf_home,
            }[cross_arm_write_root_alias]
            (write_root / "off").mkdir(exist_ok=True)
            (write_root / "on").symlink_to(write_root / "off", target_is_directory=True)
        deployment, nemo_root, _, deployment_env = _make_deployment(
            root, deployment_kind
        )
        scheduler_state = root / "scheduler-state"

        env = {
            "HOME": str(root),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PAIR_ID": pair_id,
            "STRICT_PAIR_ENVIRONMENT": "reasoning_gym",
            "EXP_NAME": "caller-must-not-select",
            "MODEL_PATH": str(model),
            "CONTAINER": str(container),
            "SANDBOX_CONTAINER": str(sandbox_container),
            "PERSISTENT_CACHE": str(cache),
            "RESULTS_DIR": str(results),
            "SLURM_PARTITION": "caller-must-not-select",
            "HF_HOME": str(hf_home),
            "CONFIG_PATH": "caller-must-not-select.yaml",
            "TRAIN_ENTRYPOINT": "./caller_must_not_select.py",
            "VAL_PATH": "caller-validation-path-must-not-survive",
            "NUM_TRAIN_NODES": "99",
            "NUM_GEN_NODES": "99",
            "NUM_GYM_NODES": "99",
            "NUM_EXTERNAL_SERVICE_NODES": "99",
            "SEGMENT_SIZE": "99",
            "GPUS_PER_NODE": "99",
            "COLOCATED_GENERATION": "0",
            "DEDICATED_RAY_HEAD": "1",
            "AMBIENT_EXPORT_SENTINEL": "must-not-cross-slurm-boundary",
            **deployment_env,
        }
        if include_train_path:
            env["TRAIN_PATH"] = train_path
        if include_wandb_key:
            env["WANDB_API_KEY"] = "0" * 32
        if startup_attack in {"BASH_ENV", "ENV"}:
            startup_file = root / f"hostile-{startup_attack.lower()}.sh"
            startup_file.write_text(
                f"echo HOSTILE_STARTUP_EXECUTED={startup_attack} >&2\nexit 73\n",
                encoding="ascii",
            )
            env[startup_attack] = str(startup_file)
        elif startup_attack is not None and startup_attack.startswith("function:"):
            function_name = startup_attack.removeprefix("function:")
            env[f"BASH_FUNC_{function_name}%%"] = (
                "() { echo HOSTILE_EXPORTED_FUNCTION_EXECUTED="
                f"{function_name} >&2; return 73; }}"
            )
        elif startup_attack is not None and startup_attack.startswith("environment:"):
            environment_name = startup_attack.removeprefix("environment:")
            env[environment_name] = "hostile-startup-value"
        elif startup_attack is not None:
            raise ValueError(f"unsupported startup attack: {startup_attack}")
        env.update({key: str(value) for key, value in env_overrides.items()})

        snapshot_parent = results / "code_snapshots_strict_pairs" / pair_id
        if preexisting_snapshot:
            snapshot = snapshot_parent / f"off-{pair_id}"
            snapshot.mkdir(parents=True)
        if preexisting_pair_manifest is not None:
            manifest_path = results / "PAIR_MANIFEST.json"
            manifest_path.write_text(preexisting_pair_manifest, encoding="ascii")
            manifest_path.chmod(0o400)
        if preexisting_submission_receipt is not None:
            receipt_path = results / "PAIR_SUBMISSION_RECEIPT.json"
            receipt_path.write_bytes(preexisting_submission_receipt)
            receipt_path.chmod(0o400)

        deployed_recipe = nemo_root / "examples/nemo_gym/nemotron-3.5-nano"
        if use_deployed_script:
            script = (
                deployed_recipe / "launch_pair.sh"
                if use_parent
                else deployed_recipe / "nano35_single_env_pair.sh"
            )
        else:
            script = PAIR_LAUNCHER if use_parent else ARM_WRAPPER
        effective_arguments = arguments
        if use_parent and not effective_arguments:
            effective_arguments = ("--dry-run",)
        command = [str(script), *effective_arguments]
        if not direct_invocation:
            command = ["/bin/bash", str(script), *effective_arguments]
        export_paths = {
            arm: results / "strict_pair_slurm_exports" / pair_id / f"{arm}.env"
            for arm in ("off", "on")
        }
        export_payloads: dict[str, bytes] = {}
        export_modes: dict[str, int] = {}
        unreadable_snapshot_path: Path | None = None
        try:
            child = subprocess.Popen(
                command,
                cwd=nemo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while child.poll() is None:
                for arm, export_path in export_paths.items():
                    if arm not in export_payloads and export_path.is_file():
                        export_payloads[arm] = export_path.read_bytes()
                        export_modes[arm] = stat.S_IMODE(export_path.stat().st_mode)
                time.sleep(0.002)
            stdout, stderr = child.communicate()
            process = subprocess.CompletedProcess(
                command, child.returncode, stdout, stderr
            )
            manifest_path = results / "PAIR_MANIFEST.json"
            manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
            manifest_mode = (
                stat.S_IMODE(manifest_path.stat().st_mode)
                if manifest_path.is_file()
                else None
            )
            if (
                process.returncode == 0
                and manifest is not None
                and (
                    post_publication_snapshot_tamper is not None
                    or post_publication_pair_manifest_tamper is not None
                )
            ):
                manifest_document = json.loads(manifest)
                if post_publication_snapshot_tamper is not None:
                    unreadable_snapshot_path = _drift_published_snapshot(
                        Path(manifest_document["source"]["snapshots"]["off"]["path"]),
                        post_publication_snapshot_tamper,
                    )
                if post_publication_pair_manifest_tamper is not None:
                    _drift_published_pair_manifest(
                        manifest_path, post_publication_pair_manifest_tamper
                    )
                verification_env = dict(env)
                verification_env.update(
                    {
                        "STRICT_PAIR_LAUNCH_MODE": "dry-run",
                        "EXPECTED_PAIR_MANIFEST_SHA256": hashlib.sha256(
                            manifest
                        ).hexdigest(),
                        "EXPECTED_STRICT_PAIR_OFF_SLURM_EXPORT_SHA256": (
                            manifest_document["slurm_export_boundary"]["arms"]["off"][
                                "sha256"
                            ]
                        ),
                        "EXPECTED_STRICT_PAIR_ON_SLURM_EXPORT_SHA256": (
                            manifest_document["slurm_export_boundary"]["arms"]["on"][
                                "sha256"
                            ]
                        ),
                        "EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256": (
                            manifest_document["artifacts"]["model"]["tree_sha256_v1"]
                        ),
                        "EXPECTED_STRICT_PAIR_CONTAINER_SHA256": (
                            manifest_document["artifacts"]["container"]["sha256"]
                        ),
                        "EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256": (
                            manifest_document["artifacts"]["sandbox_container"][
                                "sha256"
                            ]
                        ),
                        "STRICT_PAIR_SUBMISSION_CONTRACT_SHA256": (
                            manifest_document["scheduler_submission"]["contract"][
                                "sha256"
                            ]
                        ),
                        "STRICT_PAIR_SUBMISSION_NONCE": manifest_document[
                            "scheduler_submission"
                        ]["nonce"],
                    }
                )
                assert set(export_payloads) == {"off", "on"}
                for arm, export_path in export_paths.items():
                    export_path.write_bytes(export_payloads[arm])
                    export_path.chmod(0o400)
                try:
                    process = subprocess.run(
                        [str(deployed_recipe / "nano35_single_env_pair.sh"), "off"],
                        cwd=nemo_root,
                        env=verification_env,
                        capture_output=True,
                        text=True,
                    )
                finally:
                    for export_path in export_paths.values():
                        export_path.unlink(missing_ok=True)
            export_files_present_after_process = any(
                export_path.exists() or export_path.is_symlink()
                for export_path in export_paths.values()
            )
            sbatch_payloads = {}
            sbatch_argv = {}
            for arm in ("off", "on"):
                capture = scheduler_state / f"sbatch-{arm}.env"
                argv_capture = scheduler_state / f"sbatch-{arm}-argv.nul"
                if capture.is_file():
                    sbatch_payloads[arm] = capture.read_bytes()
                if argv_capture.is_file():
                    argv_records = argv_capture.read_bytes().split(b"\0")
                    assert argv_records[-1] == b""
                    sbatch_argv[arm] = tuple(
                        record.decode("ascii") for record in argv_records[:-1]
                    )
            scheduler_argv = {}
            for argv_capture in sorted(scheduler_state.glob("*-argv.nul")):
                name = argv_capture.name.removesuffix("-argv.nul")
                argv_records = argv_capture.read_bytes().split(b"\0")
                assert argv_records[-1] == b""
                scheduler_argv[name] = tuple(
                    record.decode("ascii") for record in argv_records[:-1]
                )
            submission_contract_path = results / "STRICT_PAIR_SUBMISSION_CONTRACT.json"
            submission_receipt_path = results / "PAIR_SUBMISSION_RECEIPT.json"
            submission_contract = (
                submission_contract_path.read_bytes()
                if submission_contract_path.is_file()
                else None
            )
            submission_contract_mode = (
                stat.S_IMODE(submission_contract_path.stat().st_mode)
                if submission_contract_path.is_file()
                else None
            )
            submission_receipt = (
                submission_receipt_path.read_bytes()
                if submission_receipt_path.is_file()
                else None
            )
            submission_receipt_mode = (
                stat.S_IMODE(submission_receipt_path.stat().st_mode)
                if submission_receipt_path.is_file()
                else None
            )
            submission_receipt_nlink = (
                submission_receipt_path.stat().st_nlink
                if submission_receipt_path.is_file()
                else None
            )
            scheduler_states = {
                path.stem: path.read_text(encoding="ascii").strip()
                for path in scheduler_state.glob("*.state")
            }
            residue_paths = [
                *results.glob(".PAIR_SUBMISSION_RECEIPT.json.candidate.*"),
                *results.glob("strict_pair_submission_state/*/*.scan"),
            ]
            submission_state_residue = tuple(
                sorted(
                    str(path.relative_to(results))
                    for path in residue_paths
                    if path.is_file()
                )
            )
        finally:
            if unreadable_snapshot_path is not None:
                unreadable_snapshot_path.chmod(0o500)
            shutil.rmtree(snapshot_parent, ignore_errors=True)
            deployment.chmod(0o700)
            nemo_root.chmod(0o700)
        return PairRun(
            process,
            pair_id,
            manifest,
            manifest_mode,
            export_payloads,
            export_modes,
            export_files_present_after_process,
            sbatch_payloads,
            sbatch_argv,
            scheduler_argv,
            submission_contract,
            submission_contract_mode,
            submission_receipt,
            submission_receipt_mode,
            submission_receipt_nlink,
            scheduler_states,
            submission_state_residue,
        )


def _training_commands(result: subprocess.CompletedProcess[str]) -> list[str]:
    return re.findall(
        r"--- TRAIN_CMD ---\n(.*?)\n--- end ---", result.stdout, flags=re.DOTALL
    )


def _command_for_mode(commands: list[str], mode: str) -> str:
    matches = [
        command
        for command in commands
        if f"policy.shared_prefix_training.mode={mode}" in command
    ]
    assert len(matches) == 1
    return matches[0]


def _parse_slurm_export_payload(payload: bytes) -> list[tuple[str, bytes]]:
    assert payload.endswith(b"\0")
    records = payload[:-1].split(b"\0")
    parsed = []
    for record in records:
        raw_name, separator, value = record.partition(b"=")
        assert separator == b"="
        parsed.append((raw_name.decode("ascii"), value))
    return parsed


def _compose_strict_config(mode: str, config_path: Path = CONFIG):
    # Keep heavyweight Ray/NeMo imports out of hermetic launcher-test collection.
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.single_controller_utils import MasterConfig
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = parse_hydra_overrides(
        load_config(config_path),
        [
            f"policy.shared_prefix_training.mode={mode}",
            "policy.shared_prefix_training.require_deterministic_execution=true",
            "policy.megatron_cfg.env_vars.RESULTS_DIR=/strict-pair/results",
            "policy.megatron_cfg.env_vars."
            "NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR="
            "/strict-pair/results/shared_prefix_determinism_receipts/12345-0",
        ],
    )
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return MasterConfig.model_validate(resolved)


def _run_contract_harness(
    root: Path, body: str, **environment: str
) -> subprocess.CompletedProcess[str]:
    script = root / "strict-pair-contract-harness.sh"
    if script.exists():
        script.chmod(0o600)
    script.write_text(
        """#!/bin/bash -p
set -euo pipefail
source "${CONTRACT_PATH}"
STRICT_PAIR_TOOL_PYTHON="${TOOL_PYTHON}"
STRICT_PAIR_TOOL_MKTEMP="${TOOL_MKTEMP}"
STRICT_PAIR_TOOL_RM="${TOOL_RM}"
STRICT_PAIR_TOOL_CHMOD="${TOOL_CHMOD}"
STRICT_PAIR_TOOL_SHA256SUM="${TOOL_SHA256SUM}"
STRICT_PAIR_TOOL_LN="${TOOL_LN}"
STRICT_PAIR_TOOL_CMP="${TOOL_CMP}"
STRICT_PAIR_TOOL_STAT="${TOOL_STAT}"
STRICT_PAIR_TOOL_REALPATH="${TOOL_REALPATH}"
STRICT_PAIR_TOOL_ENV="${TOOL_ENV}"
STRICT_PAIR_TOOL_GIT="${TOOL_GIT}"
STRICT_PAIR_TOOL_FIND="${TOOL_FIND}"
"""
        + body
        + "\n",
        encoding="ascii",
    )
    script.chmod(0o500)
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "CONTRACT_PATH": str(RECIPE_DIR / "strict_pair_contract.sh"),
        "TEST_ROOT": str(root),
        "TOOL_PYTHON": str(HOST_PYTHON),
        "TOOL_MKTEMP": str(HOST_TOOLS["mktemp"]),
        "TOOL_RM": str(HOST_TOOLS["rm"]),
        "TOOL_CHMOD": str(HOST_TOOLS["chmod"]),
        "TOOL_SHA256SUM": str(HOST_TOOLS["sha256sum"]),
        "TOOL_LN": str(HOST_TOOLS["ln"]),
        "TOOL_CMP": str(HOST_TOOLS["cmp"]),
        "TOOL_STAT": str(HOST_TOOLS["stat"]),
        "TOOL_REALPATH": str(HOST_TOOLS["realpath"]),
        "TOOL_ENV": str(HOST_TOOLS["env"]),
        "TOOL_GIT": str(HOST_TOOLS["git"]),
        "TOOL_FIND": str(HOST_TOOLS["find"]),
        **environment,
    }
    return subprocess.run([str(script)], env=env, capture_output=True, text=True)


_INITIALIZE_SLURM_EXPORT_VALUES = """
for name in "${STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES[@]}"; do
  printf -v "${name}" 'value:%s' "${name}"
done
HF_TOKEN="${SPECIAL_VALUE:-value:HF_TOKEN}"
"""


def _start_manifest_publisher(
    root: Path, *, config_digest: str
) -> subprocess.Popen[str]:
    script = root / "publish-pair-manifest.sh"
    runtime_manifest = root / "runtime-tools.json"
    if not runtime_manifest.exists():
        runtime_manifest.write_text("{}\n", encoding="ascii")
        runtime_manifest.chmod(0o400)
    for arm in ("off", "on"):
        (root / "cache" / arm / "cache_read").mkdir(parents=True, exist_ok=True)
        (root / "hf-home" / arm).mkdir(parents=True, exist_ok=True)
        (root / "cache" / arm).chmod(0o700)
        (root / "cache" / arm / "cache_read").chmod(0o700)
        (root / "hf-home" / arm).chmod(0o700)
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    export_digests = {}
    for arm in ("off", "on"):
        export_path = exports / f"{arm}.env"
        if not export_path.exists():
            snapshot = (
                root
                / "snapshots"
                / "concurrent-pair"
                / f"{arm}-concurrent-pair"
            )
            arm_hf_home = root / "hf-home" / arm
            values = {
                name: f"value:{name}".encode("ascii")
                for name in SLURM_EXPORT_ALLOWED_NAMES
            }
            values.update(
                {
                    "BASE_LOG_DIR": str(root / arm / "ray_logs").encode("ascii"),
                    "CPUS_PER_WORKER": b"144",
                    "EXP_NAME": f"{arm}-reasoning_gym-concurrent-pair".encode(
                        "ascii"
                    ),
                    "EXPECTED_BRIDGE_HEAD": b"8" * 40,
                    "EXPECTED_BRIDGE_TREE": b"9" * 40,
                    "EXPECTED_MCORE_HEAD": b"a" * 40,
                    "EXPECTED_MCORE_TREE": b"b" * 40,
                    "EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256": b"f" * 64,
                    "HF_DATASETS_CACHE": str(arm_hf_home / "hub").encode("ascii"),
                    "HF_HOME": str(arm_hf_home).encode("ascii"),
                    "HF_HUB_CACHE": str(arm_hf_home / "hub").encode("ascii"),
                    "NEMO_SKILLS_SANDBOX_PORT": b"6000",
                    "PAIR_ID": b"concurrent-pair",
                    "PERSISTENT_CACHE": str(root / "cache" / arm).encode("ascii"),
                    "RAY_LOG_SYNC_FREQUENCY": b"60",
                    "RESULTS_DIR": str(root / arm).encode("ascii"),
                    "SANDBOX_COMMAND": b"/start-with-nginx.sh",
                    "SETUP_COMMAND": STRICT_SETUP_COMMAND,
                    "STRICT_PAIR_ENVIRONMENT": b"reasoning_gym",
                    "STRICT_PREBUILT_SNAPSHOT_DIR": str(snapshot).encode("ascii"),
                    "TRAIN_PATH": b"/fixture.jsonl",
                    "VAL_PATH": b"/fixture.jsonl",
                    "WANDB_ENTITY": b"nvidia",
                    "WANDB_NAME": f"{arm}-reasoning_gym-concurrent-pair".encode(
                        "ascii"
                    ),
                    "WANDB_PROJ": b"nano35-rlvr-convergence",
                    "WANDB_RESUME": b"never",
                    "WANDB_RUN_GROUP": b"reasoning_gym-concurrent-pair",
                    "WANDB_RUN_ID": hashlib.sha256(
                        (
                            "nemo-rl-strict-wandb-v1:reasoning_gym:"
                            f"concurrent-pair:{arm}"
                        ).encode("ascii")
                    ).hexdigest().encode("ascii"),
                }
            )
            payload = b"".join(
                name.encode("ascii") + b"=" + values[name] + b"\0"
                for name in SLURM_EXPORT_ALLOWED_NAMES
            )
            export_path.write_bytes(payload)
            export_path.chmod(0o400)
        export_digests[arm] = _sha256(export_path)
    if not script.exists():
        script.write_text(
            """#!/bin/bash -p
set -euo pipefail
source "${STRICT_PAIR_CONTRACT_PATH}"
STRICT_PAIR_TOOL_PYTHON="${TOOL_PYTHON}"
STRICT_PAIR_TOOL_MKTEMP="${TOOL_MKTEMP}"
STRICT_PAIR_TOOL_RM="${TOOL_RM}"
STRICT_PAIR_TOOL_CHMOD="${TOOL_CHMOD}"
STRICT_PAIR_TOOL_SHA256SUM="${TOOL_SHA256SUM}"
STRICT_PAIR_TOOL_LN="${TOOL_LN}"
STRICT_PAIR_TOOL_CMP="${TOOL_CMP}"
STRICT_PAIR_TOOL_STAT="${TOOL_STAT}"
STRICT_PAIR_TOOL_REALPATH="${TOOL_REALPATH}"
PAIR_ID=concurrent-pair
STRICT_PAIR_ENVIRONMENT=reasoning_gym
STRICT_PAIR_CONFIG_RELATIVE=examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml
STRICT_PAIR_VERIFIER_METRIC=train/reasoning_gym_simple_agent/score/mean
PERSISTENT_CACHE="${RESULTS_DIR}/cache"
HF_HOME="${RESULTS_DIR}/hf-home"
DEPLOYMENT_ROOT=/deployment
EXPECTED_DEPLOYMENT_READY="$(printf 'a%.0s' {1..64})"
EXPECTED_DEPLOYMENT_READY_FILE_SHA256="$(printf 'b%.0s' {1..64})"
EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256="$(printf 'c%.0s' {1..64})"
EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256="$(printf 'd%.0s' {1..64})"
EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256="$(printf 'e%.0s' {1..64})"
MODEL_PATH=/model
STRICT_PAIR_MODEL_TREE_SHA256="$(printf 'f%.0s' {1..64})"
CONTAINER=/container.sqsh
STRICT_PAIR_CONTAINER_SHA256="$(printf '1%.0s' {1..64})"
SANDBOX_CONTAINER=/sandbox.sqsh
STRICT_PAIR_SANDBOX_CONTAINER_SHA256="$(printf '2%.0s' {1..64})"
TRAIN_PATH=/fixture.jsonl
STRICT_PAIR_FIXTURE_SHA256="$(printf '3%.0s' {1..64})"
STRICT_PAIR_FIXTURE_ROWS=5
STRICT_PAIR_PROJECT_ROOT=/deployment/runnable/NemoRL
STRICT_PAIR_NEMO_HEAD="$(printf '4%.0s' {1..40})"
STRICT_PAIR_NEMO_TREE="$(printf '5%.0s' {1..40})"
STRICT_PAIR_GYM_ROOT=/deployment/runnable/NemoRL/3rdparty/Gym-workspace/Gym
STRICT_PAIR_GYM_GITLINK_COMMIT="$(printf '6%.0s' {1..40})"
STRICT_PAIR_GYM_TREE="$(printf '7%.0s' {1..40})"
STRICT_PAIR_BRIDGE_ROOT=/deployment/runnable/Megatron-Bridge
STRICT_PAIR_BRIDGE_HEAD="$(printf '8%.0s' {1..40})"
STRICT_PAIR_BRIDGE_TREE="$(printf '9%.0s' {1..40})"
STRICT_PAIR_MCORE_ROOT=/deployment/runnable/Megatron-LM
STRICT_PAIR_MCORE_HEAD="$(printf 'a%.0s' {1..40})"
STRICT_PAIR_MCORE_TREE="$(printf 'b%.0s' {1..40})"
STRICT_PAIR_GYM_CONFIG_RELATIVE=resources_servers/reasoning_gym/configs/reasoning_gym.yaml
STRICT_PAIR_GYM_CONFIG_SHA256="$(printf 'c%.0s' {1..64})"
STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE=resources_servers/reasoning_gym/requirements.txt
STRICT_PAIR_GYM_REQUIREMENTS_SHA256="$(printf 'd%.0s' {1..64})"
STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE=resources_servers/reasoning_gym/app.py
STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256="$(printf 'e%.0s' {1..64})"
STRICT_PAIR_CONFIG_SHA256="${CONFIG_DIGEST}"
STRICT_PAIR_ENTRYPOINT_SHA256="$(printf '6%.0s' {1..64})"
STRICT_PAIR_LAUNCHER_SHA256="$(printf '7%.0s' {1..64})"
STRICT_PAIR_ARM_WRAPPER_SHA256="$(printf '8%.0s' {1..64})"
STRICT_PAIR_PARENT_WRAPPER_SHA256="$(printf '9%.0s' {1..64})"
STRICT_PAIR_CONTRACT_SHA256="$(printf '0%.0s' {1..64})"
STRICT_PAIR_SNAPSHOT_PARENT="${RESULTS_DIR}/snapshots/concurrent-pair"
STRICT_PAIR_LAUNCH_MODE=dry-run
STRICT_PAIR_JOB_WRAPPER=/deployment/job-wrapper.sh
STRICT_PAIR_JOB_WRAPPER_SHA256="$(printf 'a%.0s' {1..64})"
STRICT_PAIR_OFF_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/off-concurrent-pair"
STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256="$(printf 'b%.0s' {1..64})"
STRICT_PAIR_ON_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/on-concurrent-pair"
STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256="$(printf 'c%.0s' {1..64})"
STRICT_PAIR_OFF_SLURM_EXPORT="${RESULTS_DIR}/exports/off.env"
STRICT_PAIR_OFF_SLURM_EXPORT_SHA256="${OFF_EXPORT_SHA256}"
STRICT_PAIR_ON_SLURM_EXPORT="${RESULTS_DIR}/exports/on.env"
STRICT_PAIR_ON_SLURM_EXPORT_SHA256="${ON_EXPORT_SHA256}"
STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES_CSV="$(
  IFS=,
  printf '%s' "${STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES[*]}"
)"
STRICT_PAIR_SUBMISSION_NONCE="$(printf '1%.0s' {1..64})"
STRICT_PAIR_SUBMISSION_CONTRACT_PATH="${RESULTS_DIR}/STRICT_PAIR_SUBMISSION_CONTRACT.json"
STRICT_PAIR_SUBMISSION_CONTRACT_SHA256="$(printf 'f%.0s' {1..64})"
STRICT_PAIR_OFF_ACCEPTED_ID_RECORD="${RESULTS_DIR}/off.job-id"
STRICT_PAIR_ON_ACCEPTED_ID_RECORD="${RESULTS_DIR}/on.job-id"
STRICT_PAIR_RUNTIME_TOOL_MANIFEST="${RUNTIME_TOOL_MANIFEST}"
EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256="${RUNTIME_TOOL_MANIFEST_SHA256}"
STRICT_PAIR_BOOTSTRAP_SHA256SUM="${TOOL_SHA256SUM}"
EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256="${TOOL_SHA256SUM_SHA256}"
STRICT_PAIR_MANIFEST_PATH="${RESULTS_DIR}/PAIR_MANIFEST.json"
strict_pair_publish_manifest
echo "${STRICT_PAIR_MANIFEST_SHA256}"
""",
            encoding="ascii",
        )
        script.chmod(0o500)
    env = {
        "PATH": os.environ["PATH"],
        "RESULTS_DIR": str(root),
        "CONFIG_DIGEST": config_digest,
        "STRICT_PAIR_CONTRACT_PATH": str(RECIPE_DIR / "strict_pair_contract.sh"),
        "RUNTIME_TOOL_MANIFEST": str(runtime_manifest),
        "RUNTIME_TOOL_MANIFEST_SHA256": _sha256(runtime_manifest),
        "OFF_EXPORT_SHA256": export_digests["off"],
        "ON_EXPORT_SHA256": export_digests["on"],
        "TOOL_PYTHON": str(HOST_PYTHON),
        "TOOL_MKTEMP": str(HOST_TOOLS["mktemp"]),
        "TOOL_RM": str(HOST_TOOLS["rm"]),
        "TOOL_CHMOD": str(HOST_TOOLS["chmod"]),
        "TOOL_SHA256SUM": str(HOST_TOOLS["sha256sum"]),
        "TOOL_SHA256SUM_SHA256": _sha256(HOST_TOOLS["sha256sum"]),
        "TOOL_LN": str(HOST_TOOLS["ln"]),
        "TOOL_CMP": str(HOST_TOOLS["cmp"]),
        "TOOL_STAT": str(HOST_TOOLS["stat"]),
        "TOOL_REALPATH": str(HOST_TOOLS["realpath"]),
    }
    return subprocess.Popen(
        [str(script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_authoritative_pair_builds_exact_parallel_arm_contract() -> None:
    run = _run_pair()
    result = run.process

    assert result.returncode == 0, result.stderr
    assert "HOSTILE_FAKE_TOOL_EXECUTED" not in result.stderr
    assert run.manifest is not None
    assert run.manifest_mode == 0o400
    manifest = json.loads(run.manifest)
    uv_shim = manifest["runtime_tools"]["document"]["container"]["uv_shim"]["path"]
    commands = _training_commands(result)
    assert len(commands) == 2
    for arm, mode in (("off", "observe"), ("on", "train")):
        command = _command_for_mode(commands, mode)
        required_once = (
            f"{uv_shim} run ./examples/run_grpo_single_controller.py",
            "UV_CACHE_DIR=/tmp/nemo-gym-uv-cache-${STRICT_PAIR_BOUND_JOB_ID}",
            "--config examples/nemo_gym/nemotron-3.5-nano/"
            "single_env_reasoning_gym_sc.yaml",
            f"policy.shared_prefix_training.mode={mode}",
            "policy.shared_prefix_training.require_deterministic_execution=true",
            "policy.megatron_cfg.env_vars.RESULTS_DIR=${RESULTS_DIR}",
            "policy.megatron_cfg.env_vars."
            "NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR="
            "${NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR}",
            "cluster.num_nodes=1",
            "cluster.segment_size=1",
            "cluster.gpus_per_node=4",
            "policy.generation.backend=vllm",
            "policy.generation.colocated.enabled=true",
            "policy.generation.colocated.resources.num_nodes=1",
            "policy.generation.colocated.resources.gpus_per_node=4",
            "policy.generation.vllm_cfg.gpu_memory_utilization=0.6",
            "++policy.generation.refit_transport=null",
            "env.nemo_gym.num_gpu_nodes=0",
            "logger.wandb_enabled=True",
            f"logger.wandb.name={arm}-reasoning_gym-{run.pair_id}",
            "logger.wandb.project=nano35-rlvr-convergence",
        )
        for token in required_once:
            assert command.count(token) == 1, token
        assert "SLURM_JOB_ID:-default" not in command
        assert " uv run " not in f" {command} "
        assert command.count(f"{uv_shim} run ") == 1
        train_path = command.split("data.train.data_path=", maxsplit=1)[1].split()[0]
        validation_path = command.split("data.validation.data_path=", maxsplit=1)[
            1
        ].split()[0]
        assert train_path == validation_path

    assert "caller-must-not-select" not in result.stdout
    assert result.stdout.count("Code snapshot:") == 2
    assert result.stdout.count("SingleController entrypoint manifest:") == 2
    assert result.stdout.count("Mount: SingleController entrypoint") == 2
    assert result.stdout.count("(read-only, sha256=") == 2
    assert "live-tree-no-manifest" not in result.stdout
    assert "mode=dry-run submissions=parallel partition=batch" in result.stdout
    assert "STRICT_PAIR_DRY_RUN_VALIDATED" in result.stdout
    assert result.stdout.count("STRICT_PAIR_JOB_INTERVAL_GATE_REQUIRED=1") == 2
    assert result.stdout.count("STRICT_PAIR_RUNTIME_ATTESTATION_REQUIRED=1") == 2
    entrypoint_sha256 = _sha256(ENTRYPOINT)
    assert result.stdout.count(f"sha256={entrypoint_sha256}") == 2

    assert manifest["schema"] == "nemo-rl-strict-single-env-pair-v2"
    assert manifest["pair_id"] == run.pair_id
    assert manifest["selection"] == {
        "config": {
            "path": "examples/nemo_gym/nemotron-3.5-nano/"
            "single_env_reasoning_gym_sc.yaml",
            "sha256": manifest["source"]["config_sha256"],
        },
        "environment": "reasoning_gym",
        "fixture": manifest["artifacts"]["fixture"],
        "gym_resources": {
            "config": {
                "path": "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
                "sha256": manifest["selection"]["gym_resources"]["config"]["sha256"],
            },
            "requirements": {
                "path": "resources_servers/reasoning_gym/requirements.txt",
                "sha256": manifest["selection"]["gym_resources"]["requirements"][
                    "sha256"
                ],
            },
            "verifier_source": {
                "path": "resources_servers/reasoning_gym/app.py",
                "sha256": manifest["selection"]["gym_resources"]["verifier_source"][
                    "sha256"
                ],
            },
        },
    }
    assert manifest["determinism_receipt_dir"] == "shared_prefix_determinism_receipts"
    assert manifest["arms"] == {"off": "observe", "on": "train"}
    assert set(manifest["source"]["bridge"]) == {"head", "root", "tree"}
    assert set(manifest["source"]["mcore"]) == {"head", "root", "tree"}
    for component, directory in (
        ("bridge", "Megatron-Bridge"),
        ("mcore", "Megatron-LM"),
    ):
        source_record = manifest["source"][component]
        assert source_record["root"] == (
            f'{manifest["deployment"]["root"]}/runnable/{directory}'
        )
        assert re.fullmatch(r"[0-9a-f]{40}", source_record["head"])
        assert re.fullmatch(r"[0-9a-f]{40}", source_record["tree"])
    assert manifest["wandb"] == {
        "arms": {
            arm: {
                "name": f"{arm}-reasoning_gym-{run.pair_id}",
                "name_template": f"{arm}-{{environment}}-{{pair_id}}",
                "run_id": hashlib.sha256(
                    (
                        f"nemo-rl-strict-wandb-v1:reasoning_gym:"
                        f"{run.pair_id}:{arm}"
                    ).encode("ascii")
                ).hexdigest(),
            }
            for arm in ("off", "on")
        },
        "entity": "nvidia",
        "group": {
            "template": "{environment}-{pair_id}",
            "value": f"reasoning_gym-{run.pair_id}",
        },
        "project": "nano35-rlvr-convergence",
        "resume": "never",
        "run_id_derivation": (
            "sha256-ascii:nemo-rl-strict-wandb-v1:"
            "{environment}:{pair_id}:{arm}"
        ),
    }
    execution_environment = manifest["execution_environment"]
    assert set(execution_environment) == {"arm_launcher", "arms", "fixed", "schema"}
    assert execution_environment["schema"] == (
        "nemo-rl-strict-execution-environment-v1"
    )
    assert execution_environment["arm_launcher"] == {
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
    assert execution_environment["fixed"] == {
        "cpus_per_worker": "144",
        "nemo_skills_sandbox_port": "6000",
        "ray_log_sync_frequency": "60",
        "sandbox_command": "/start-with-nginx.sh",
        "train_path": manifest["artifacts"]["fixture"]["path"],
        "val_path": manifest["artifacts"]["fixture"]["path"],
    }
    for arm in ("off", "on"):
        arm_execution = execution_environment["arms"][arm]
        assert set(arm_execution) == {
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
        assert arm_execution["cache_read"] == {
            "entry_count": 0,
            "mode": "0700",
            "path": f'{arm_execution["persistent_cache"]}/cache_read',
            "policy": "empty-at-publication-and-job-entry-no-read",
        }
        snapshot_path = manifest["source"]["snapshots"][arm]["path"]
        assert arm_execution["scheduler"] == {
            "batch_working_directory": snapshot_path,
            "sbatch_chdir_argument": f"--chdir={snapshot_path}",
            "sbatch_client_cwd": snapshot_path,
            "slurm_submit_dir": snapshot_path,
        }
    assert execution_environment["arms"]["off"]["persistent_cache"] != (
        execution_environment["arms"]["on"]["persistent_cache"]
    )
    assert execution_environment["arms"]["off"]["hf_home"] != (
        execution_environment["arms"]["on"]["hf_home"]
    )
    assert run.submission_contract is not None
    assert run.submission_contract_mode == 0o400
    submission_contract = json.loads(run.submission_contract)
    assert submission_contract["schema"] == (
        "nemo-rl-strict-pair-submission-contract-v2"
    )
    assert submission_contract["pair"] == {"environment": "reasoning_gym"}
    receipt_contract = submission_contract["receipt"]
    assert receipt_contract["schema"] == ("nemo-rl-strict-pair-submission-receipt-v2")
    expected_receipt_keys = set(receipt_contract["required_root_keys"])
    assert expected_receipt_keys == set(receipt_contract["examples"]["released"])
    assert expected_receipt_keys == set(receipt_contract["examples"]["failed_closed"])
    assert expected_receipt_keys == set(
        receipt_contract["examples"]["rollback_unconfirmed"]
    )
    assert manifest["scheduler_submission"]["receipt"]["schema"] == (
        "nemo-rl-strict-pair-submission-receipt-v2"
    )
    slurm_boundary = manifest["slurm_export_boundary"]
    assert slurm_boundary["schema"] == "nemo-rl-strict-slurm-export-file-v2"
    assert slurm_boundary["format"] == "nul-separated-name-value"
    assert slurm_boundary["ambient_merge"] is False
    assert slurm_boundary["get_user_env"] is False
    assert slurm_boundary["allowed_names"] == list(SLURM_EXPORT_ALLOWED_NAMES)
    assert tuple(sorted(SLURM_EXPORT_ALLOWED_NAMES)) == SLURM_EXPORT_ALLOWED_NAMES
    assert len(SLURM_EXPORT_ALLOWED_NAMES) == 77
    assert slurm_boundary["job_argv"] == [
        "--pair-manifest",
        "{pair_manifest_path}",
        "--pair-manifest-sha256",
        "{pair_manifest_sha256}",
        "--arm",
        "{arm}",
    ]
    assert set(run.export_payloads) == {"off", "on"}
    assert set(run.export_modes) == {"off", "on"}
    for arm in ("off", "on"):
        export_record = slurm_boundary["arms"][arm]
        assert export_record["path"].endswith(f"/{arm}.env")
        assert re.fullmatch(r"[0-9a-f]{64}", export_record["sha256"])
        payload = run.export_payloads[arm]
        assert run.export_modes[arm] == 0o400
        assert hashlib.sha256(payload).hexdigest() == export_record["sha256"]
        parsed_payload = _parse_slurm_export_payload(payload)
        assert [name for name, _ in parsed_payload] == list(SLURM_EXPORT_ALLOWED_NAMES)
        values = dict(parsed_payload)
        assert values["COLOCATED_GENERATION"] == b"1"
        assert values["SEGMENT_SIZE"] == b"1"
        assert values["MODEL_PATH"] == manifest["artifacts"]["model"]["path"].encode(
            "ascii"
        )
        assert values["PAIR_ID"] == run.pair_id.encode("ascii")
        assert values["EXPECTED_BRIDGE_HEAD"] == manifest["source"]["bridge"][
            "head"
        ].encode("ascii")
        assert values["EXPECTED_BRIDGE_TREE"] == manifest["source"]["bridge"][
            "tree"
        ].encode("ascii")
        assert values["EXPECTED_MCORE_HEAD"] == manifest["source"]["mcore"][
            "head"
        ].encode("ascii")
        assert values["EXPECTED_MCORE_TREE"] == manifest["source"]["mcore"][
            "tree"
        ].encode("ascii")
        assert values["EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256"] == (
            manifest["scheduler_submission"]["contract"]["sha256"].encode("ascii")
        )
        assert values["WANDB_ENTITY"] == b"nvidia"
        assert values["WANDB_PROJ"] == b"nano35-rlvr-convergence"
        assert values["WANDB_RESUME"] == b"never"
        assert values["WANDB_RUN_GROUP"] == (
            f"reasoning_gym-{run.pair_id}".encode("ascii")
        )
        assert values["WANDB_NAME"] == manifest["wandb"]["arms"][arm][
            "name"
        ].encode("ascii")
        assert values["WANDB_RUN_ID"] == manifest["wandb"]["arms"][arm][
            "run_id"
        ].encode("ascii")
        assert values["HF_HOME"] == manifest["execution_environment"]["arms"][
            arm
        ]["hf_home"].encode("ascii")
        assert values["HF_HUB_CACHE"] == manifest["execution_environment"][
            "arms"
        ][arm]["hf_hub_cache"].encode("ascii")
        assert values["HF_DATASETS_CACHE"] == manifest["execution_environment"][
            "arms"
        ][arm]["hf_datasets_cache"].encode("ascii")
        setup_command = manifest["execution_environment"]["arms"][arm][
            "setup_command"
        ]
        assert values["SETUP_COMMAND"] == STRICT_SETUP_COMMAND
        assert not any(
            forbidden in values["SETUP_COMMAND"].lower()
            for forbidden in (b"cache_read", b"tar", b"zstd", b"persistent_cache")
        )
        assert setup_command == {
            "byte_count": len(values["SETUP_COMMAND"]),
            "sha256": hashlib.sha256(values["SETUP_COMMAND"]).hexdigest(),
        }
        assert (
            values["STRICT_PAIR_SHARED_PREFIX_MODE"]
            == {
                "off": b"observe",
                "on": b"train",
            }[arm]
        )
        for positional_only in (
            "EXPECTED_PAIR_MANIFEST_SHA256",
            "STRICT_PAIR_ARM",
            "STRICT_PAIR_MANIFEST_PATH",
        ):
            assert positional_only not in values
    assert (
        slurm_boundary["arms"]["off"]["sha256"]
        != slurm_boundary["arms"]["on"]["sha256"]
    )
    assert run.export_files_present_after_process is False
    assert b"0" * 32 not in run.manifest
    assert manifest["container_entry_boundary"] == {
        "bash_args": ["-p"],
        "bash_path": "/bin/bash",
        "env_path": "/usr/bin/env",
        "sha256sum": {
            "path": "/usr/bin/sha256sum",
            "sha256": (
                "f3d040161f5c29e4c7cd4e3d6bb513ce9a43b9d1bd06f456a6aab3d34d0f1e33"
            ),
        },
        "unset_environment": ["BASH_ENV", "ENV"],
    }
    runtime_tools = manifest["runtime_tools"]
    assert set(runtime_tools) == {
        "bootstrap_sha256sum",
        "document",
        "manifest",
    }
    assert runtime_tools["bootstrap_sha256sum"] == {
        "path": str(BOOTSTRAP_SHA256SUM),
        "sha256": _sha256(BOOTSTRAP_SHA256SUM),
    }
    assert (
        runtime_tools["manifest"]["sha256"]
        == hashlib.sha256(
            json.dumps(
                runtime_tools["document"], sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            + b"\n"
        ).hexdigest()
    )
    assert runtime_tools["document"]["schema"] == ("nemo-rl-strict-runtime-tools-v2")
    assert set(runtime_tools["document"]["host"]) == set(HOST_TOOLS)
    assert set(runtime_tools["document"]["container"]) == {
        "python",
        "uv",
        "uv_shim",
    }
    assert manifest["campaign"]["training_topology"] == "TP2/CP2/PP1/EP4/ETP1/SP"
    assert manifest["campaign"]["padding_multiple"] == 128
    assert manifest["campaign"]["epochs"] == 20
    assert manifest["campaign"]["steps"] == 100
    assert manifest["campaign"]["generations_per_prompt"] == 4
    assert manifest["campaign"]["hardware"] == {"gpu_model": "NVIDIA GB200"}
    assert manifest["campaign"]["require_deterministic_execution"] is True
    assert manifest["campaign"]["threat_model"] == {
        "cooperative_exclusive_campaign_root": True,
        "malicious_same_uid_active_mutation": "out_of_scope",
        "trusted_operator_and_code": True,
    }
    campaign_bytes = json.dumps(
        manifest["campaign"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    reward_and_advantage_bytes = json.dumps(
        manifest["campaign"]["reward_and_advantage"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert b"\n" not in campaign_bytes
    assert b"\n" not in reward_and_advantage_bytes
    assert (
        manifest["pair_campaign_sha256"] == hashlib.sha256(campaign_bytes).hexdigest()
    )
    assert manifest["pair_campaign_sha256"] == (
        "19a5003a0c8aee69b48cab0c16dde4fb1c03dda559fb0b7bf87ac89f550a5de1"
    )
    assert (
        manifest["pair_campaign_reward_and_advantage_sha256"]
        == hashlib.sha256(reward_and_advantage_bytes).hexdigest()
    )
    assert manifest["pair_campaign_reward_and_advantage_sha256"] == (
        "4148701a37c26bc9d1ce956f55783a6c4ad72dd38f3fc6c9bdfb5056aafd7f91"
    )
    reward_contract = manifest["campaign"]["reward_and_advantage"]
    assert reward_contract["advantage_normalization"] is True
    assert reward_contract["advantage_clip"] == {"high": 20.0, "low": -20.0}
    assert reward_contract["reward_scaling"] == {
        "enabled": False,
        "source_max": 1.0,
        "source_min": 0.0,
        "target_max": 1.0,
        "target_min": 0.0,
    }
    assert reward_contract["reward_shaping"] == {
        "enabled": False,
        "max_response_length": 768,
        "overlong_buffer_length": 128,
        "overlong_buffer_penalty": 1.0,
        "stop_properly_penalty_coef": None,
    }
    assert reward_contract["loss"] == {
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
        "use_importance_sampling_correction": True,
        "use_kl_in_reward": False,
        "use_on_policy_kl_approximation": True,
    }
    assert reward_contract["sample_mask"] == {
        "env_flagged_samples": True,
        "seq_logprob_error_threshold": 2.0,
    }
    assert reward_contract["metrics"] == {
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
    }
    assert reward_contract["zeroing_penalties"] == {
        "duplicated_reasoning": True,
        "empty_final_answer": True,
        "malformed_think_tag": True,
        "thinking_tags": ["<think>", "</think>"],
        "token_ids": {"think_close": 13, "think_open": 12, "unwanted": [2]},
        "unwanted_token": True,
    }
    assert reward_contract["required_step_relations"] == {
        "effort_low_sample_rate": "train/effort_low_sample_rate == 0",
        "effort_reward_delta": "train/effort_reward_delta == 0",
        "raw_equals_pre_penalty": (
            "train/raw_environment_reward == train/pre_penalty_environment_reward"
        ),
        "reward_processing_delta": "train/reward_processing_delta == 0",
        "reward_equals_effective": (
            "train/reward == train/verifier_reward == train/total_reward/mean"
        ),
    }
    assert manifest["campaign"]["generation"] == {
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "vllm_gpu_memory_utilization": 0.6,
    }

    assert manifest["campaign"]["slurm"] == {
        "account": "nemotron_sw_post",
        "partition": "batch",
        "qos": "normal",
        "walltime": "04:00:00",
    }
    assert manifest["campaign"]["logging"] == {
        "tensorboard_enabled": False,
        "wandb_enabled": True,
        "wandb_entity": "nvidia",
        "wandb_group_template": "{environment}-{pair_id}",
        "wandb_project": "nano35-rlvr-convergence",
        "wandb_run_name_templates": {
            "off": "off-{environment}-{pair_id}",
            "on": "on-{environment}-{pair_id}",
        },
    }
    runtime_attestation = manifest["runtime_attestation"]
    assert runtime_attestation["expected_count_per_fresh_process_group"] == 4
    assert runtime_attestation["receipt_requires_line_count_and_hash"] is True
    assert (
        runtime_attestation["schema"]
        == "nemo-rl-shared-prefix-determinism-attestation-v1"
    )
    for arm in ("off", "on"):
        snapshot = manifest["source"]["snapshots"][arm]
        assert snapshot["path"].endswith(f"/{arm}-{run.pair_id}")
        assert re.fullmatch(r"[0-9a-f]{64}", snapshot["manifest_sha256"])
        assert snapshot["entrypoint_sha256"] == entrypoint_sha256
        assert snapshot["config_sha256"] == _sha256(CONFIG)
        mode = {"off": "observe", "on": "train"}[arm]
        marker = (
            "SHARED_PREFIX_DETERMINISM_ATTESTED "
            f"mode={mode} env_controls=5 triton_autotune=absent "
            "model_overrides=3 torch_deterministic=true mcore_backward=true "
            "total_controls=9"
        )
        attestation = runtime_attestation["lines"][arm]
        assert attestation["mode"] == mode
        assert attestation["text"] == marker
        assert (
            attestation["sha256_ascii_no_newline"]
            == hashlib.sha256(marker.encode("ascii")).hexdigest()
        )
        assert (
            f"STRICT_PAIR_RUNTIME_ATTESTATION_REQUIRED=1 arm={arm} mode={mode} "
            "count=4 sha256_ascii_no_newline="
            f"{attestation['sha256_ascii_no_newline']}"
        ) in result.stdout
    gym = manifest["source"]["gym"]
    assert re.fullmatch(r"[0-9a-f]{40}", gym["gitlink_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", gym["tree"])
    manifest_sha256 = hashlib.sha256(run.manifest).hexdigest()
    assert result.stdout.count(f"STRICT_PAIR_MANIFEST_SHA256={manifest_sha256}") == 3


def test_slurm_export_payload_preserves_opaque_values_without_ambient_merge() -> None:
    token = "hf,comma=equals\nnewline"
    run = _run_pair(HF_TOKEN=token)
    assert run.process.returncode == 0, run.process.stderr
    assert run.manifest is not None
    manifest = json.loads(run.manifest)
    assert token not in run.manifest.decode("ascii")
    assert (
        "AMBIENT_EXPORT_SENTINEL"
        not in manifest["slurm_export_boundary"]["allowed_names"]
    )
    assert set(run.export_payloads) == {"off", "on"}
    for arm, payload in run.export_payloads.items():
        values = dict(_parse_slurm_export_payload(payload))
        assert values["HF_TOKEN"] == token.encode("ascii")
        assert b"AMBIENT_EXPORT_SENTINEL" not in payload
        assert (
            hashlib.sha256(payload).hexdigest()
            == manifest["slurm_export_boundary"]["arms"][arm]["sha256"]
        )
    assert run.export_files_present_after_process is False


@pytest.mark.parametrize("arm", ["off", "on"])
def test_pair_rejects_unbound_cache_read_preseed(arm: str) -> None:
    run = _run_pair(preseed_cache_read_arm=arm)
    assert run.process.returncode != 0
    assert "arm cache_read input must be empty" in run.process.stderr
    assert run.manifest is None
    assert run.export_files_present_after_process is False


@pytest.mark.parametrize("root_name", ["persistent_cache", "hf_home"])
def test_pair_rejects_cross_arm_write_root_alias(root_name: str) -> None:
    run = _run_pair(cross_arm_write_root_alias=root_name)
    assert run.process.returncode != 0
    assert "must be a real, non-symlink directory" in run.process.stderr
    assert run.manifest is None
    assert run.export_files_present_after_process is False


def test_submit_uses_export_file_and_positional_pair_identity() -> None:
    run = _run_pair("--submit")
    assert run.process.returncode == 0, run.process.stderr
    assert run.manifest is not None
    manifest = json.loads(run.manifest)
    manifest_sha256 = hashlib.sha256(run.manifest).hexdigest()
    wrapper = manifest["source"]["job_wrapper"]["path"]
    assert set(run.sbatch_payloads) == {"off", "on"}
    assert set(run.sbatch_argv) == {"off", "on"}
    for arm in ("off", "on"):
        export_path = manifest["slurm_export_boundary"]["arms"][arm]["path"]
        assert run.sbatch_payloads[arm] == run.export_payloads[arm]
        assert (
            hashlib.sha256(run.sbatch_payloads[arm]).hexdigest()
            == manifest["slurm_export_boundary"]["arms"][arm]["sha256"]
        )
        arm_results = Path(manifest["paths"]["results_root"]) / arm
        expected_sbatch_argv = (
            "--parsable",
            "--hold",
            f"--chdir={manifest['source']['snapshots'][arm]['path']}",
            "--nodes=1",
            "--account=nemotron_sw_post",
            f"--job-name={arm}-reasoning_gym-{run.pair_id}",
            "--partition=batch",
            "--time=04:00:00",
            "--gres=gpu:4",
            "--exclusive",
            "--mem=0",
            "--dependency=singleton",
            "--segment=1",
            f"--output={arm_results}/runs/{run.pair_id}/slurm/%j.out",
            f"--error={arm_results}/runs/{run.pair_id}/slurm/%j.err",
            "--qos=normal",
            (
                "--comment=nemo-rl-strict-pair-v1:"
                f"{arm}:{manifest['scheduler_submission']['nonce']}:"
                f"{manifest_sha256}"
            ),
            f"--export-file={export_path}",
            wrapper,
            "--pair-manifest",
            f"{manifest['paths']['results_root']}/PAIR_MANIFEST.json",
            "--pair-manifest-sha256",
            manifest_sha256,
            "--arm",
            arm,
        )
        assert run.sbatch_argv[arm] == expected_sbatch_argv
        job_id = {"off": "41001", "on": "41002"}[arm]
        output_sha256 = hashlib.sha256(job_id.encode("ascii")).hexdigest()
        record_sha256 = hashlib.sha256(f"{job_id}\n".encode("ascii")).hexdigest()
        assert (
            run.process.stdout.count(
                f"STRICT_PAIR_HELD arm={arm} job_id={job_id} "
                f"job_id_sha256_ascii_no_newline={output_sha256} "
                f"accepted_id_record_sha256={record_sha256}"
            )
            == 1
        )
    assert "--export=ALL" not in run.process.stdout
    assert "--export=NONE" not in run.process.stdout
    assert "--export=NIL" not in run.process.stdout
    assert "--get-user-env" not in run.process.stdout
    assert run.submission_receipt is not None
    assert run.submission_receipt_mode == 0o400
    assert run.submission_receipt_nlink == 1
    receipt = json.loads(run.submission_receipt)
    receipt_sha256 = hashlib.sha256(run.submission_receipt).hexdigest()
    assert run.submission_contract is not None
    submission_contract = json.loads(run.submission_contract)
    receipt_contract = submission_contract["receipt"]
    normative_released = receipt_contract["examples"]["released"]
    assert set(receipt) == set(receipt_contract["required_root_keys"])
    assert set(receipt) == set(normative_released)
    for arm in ("off", "on"):
        assert set(receipt["held_submissions"][arm]) == set(
            normative_released["held_submissions"][arm]
        )
        assert set(receipt["authenticated_jobs"][arm][0]) == set(
            normative_released["authenticated_jobs"][arm][0]
        )
    for query_name in ("pre_release_query", "post_release_query"):
        assert set(receipt[query_name]) == set(normative_released[query_name])
    assert receipt["outcome"] == "released"
    assert receipt["schema"] == "nemo-rl-strict-pair-submission-receipt-v2"
    assert receipt["selection"] == manifest["selection"]
    assert receipt["execution_environment"] == manifest["execution_environment"]
    assert receipt["wandb"] == manifest["wandb"]
    assert receipt["source"] == {
        "bridge": manifest["source"]["bridge"],
        "mcore": manifest["source"]["mcore"],
    }
    assert receipt["stage"] == "complete"
    assert receipt["rollback_confirmed"] is None
    assert receipt["held_submissions"]["off"]["candidate_job_id"] == "41001"
    assert receipt["held_submissions"]["on"]["candidate_job_id"] == "41002"
    assert receipt["authenticated_jobs"] == {
        arm: [
            {
                "comment": (
                    f"nemo-rl-strict-pair-v1:{arm}:"
                    f"{receipt['submission_nonce']}:{manifest_sha256}"
                ),
                "job_id": job_id,
                "job_name": f"{arm}-reasoning_gym-{run.pair_id}",
                "user_id": str(os.geteuid()),
            }
        ]
        for arm, job_id in (("off", "41001"), ("on", "41002"))
    }
    for arm, job_id in (("off", "41001"), ("on", "41002")):
        record = receipt["held_submissions"][arm]["accepted_id_record"]
        assert record == {
            "path": manifest["scheduler_submission"]["accepted_id_records"][arm][
                "path"
            ],
            "parsed_job_id": job_id,
            "sha256": hashlib.sha256(f"{job_id}\n".encode("ascii")).hexdigest(),
        }
    assert receipt["pre_release_query"]["records"]["off"][0]["held"] is True
    assert receipt["pre_release_query"]["records"]["on"][0]["held"] is True
    assert receipt["post_release_query"]["records"]["off"][0]["job_state"] == (
        "RUNNING"
    )
    assert receipt["post_release_query"]["records"]["on"][0]["job_state"] == ("PENDING")
    for phase, query_name in (
        ("pre", "pre_release_query"),
        ("post", "post_release_query"),
    ):
        query = receipt[query_name]
        assert query["phase"] == phase
        assert query["complete"] is True
        assert query["unterminated_final_line"] is False
        assert query["securely_unlinked"] is True
        assert query["authenticated_job_ids"] == ["41001", "41002"]
        assert query["unresolved_job_ids"] == []
        assert query["line_count"] == 2
        for arm in ("off", "on"):
            assert query["records"][arm][0]["work_dir"] == (
                manifest["source"]["snapshots"][arm]["path"]
            )
        assert query["byte_count"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", query["output_sha256_raw"])
    assert receipt["pre_cancel_queries"] == []
    assert receipt["post_cancel_queries"] == []
    assert receipt["cancellations"] == []
    assert run.scheduler_argv["scontrol-release"] == (
        "release",
        "41001,41002",
    )
    assert run.scheduler_argv["scontrol-query-1"] == (
        "show",
        "job",
        "--oneliner",
    )
    assert run.scheduler_argv["scontrol-query-2"] == (
        "show",
        "job",
        "--oneliner",
        "41001,41002",
    )
    assert run.scheduler_argv["scontrol-query-3"] == (
        "show",
        "job",
        "--oneliner",
        "41001,41002",
    )
    assert run.process.stdout.count("STRICT_PAIR_RESULT ") == 1
    assert (
        "STRICT_PAIR_RESULT "
        "schema=nemo-rl-strict-pair-submission-receipt-v2 "
        "outcome=released stage=complete "
        f"pair_id={run.pair_id} environment=reasoning_gym "
        f"nonce={receipt['submission_nonce']} "
        "off_job_id=41001 off_identity_authenticated=true "
        "on_job_id=41002 on_identity_authenticated=true "
        f"receipt_path={manifest['scheduler_submission']['receipt']['path']} "
        f"receipt_sha256={receipt_sha256}"
    ) in run.process.stdout
    assert "STRICT_PAIR_LAUNCHED" not in run.process.stdout
    assert run.export_files_present_after_process is False


@pytest.mark.parametrize(
    ("deployment_kind", "query_name", "phase", "stage"),
    [
        (
            "pre_release_unterminated",
            "pre_release_query",
            "pre",
            "pre_release_validation",
        ),
        (
            "post_release_unterminated",
            "post_release_query",
            "post",
            "post_release_validation",
        ),
    ],
)
def test_lifecycle_query_requires_complete_lf_framing_before_authentication(
    deployment_kind: str,
    query_name: str,
    phase: str,
    stage: str,
) -> None:
    run = _run_pair("--submit", deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["schema"] == "nemo-rl-strict-pair-submission-receipt-v2"
    assert receipt["outcome"] == "failed-closed"
    assert receipt["stage"] == stage
    assert receipt["rollback_confirmed"] is True
    query = receipt[query_name]
    assert query["phase"] == phase
    assert query["status"] == 0
    assert query["normalization_status"] == 0
    assert query["complete"] is False
    assert query["unterminated_final_line"] is True
    assert query["authenticated_job_ids"] == []
    assert query["unresolved_job_ids"] == ["41001", "41002"]
    assert query["securely_unlinked"] is True
    assert query["line_count"] == 2
    assert query["byte_count"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", query["output_sha256_raw"])
    assert {
        arm: [label["job_id"] for label in receipt["authenticated_jobs"][arm]]
        for arm in ("off", "on")
    } == {"off": ["41001"], "on": ["41002"]}
    assert receipt["cancellations"][0]["job_ids"] == ["41001", "41002"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "CANCELLED"}
    assert run.submission_state_residue == ()


def test_post_release_rejects_job_held_reason_in_running_state() -> None:
    run = _run_pair("--submit", deployment_kind="post_release_held_reason_running")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "failed-closed"
    assert receipt["stage"] == "post_release_validation"
    assert receipt["rollback_confirmed"] is True
    query = receipt["post_release_query"]
    assert query["status"] == 0
    assert query["complete"] is True
    assert query["unterminated_final_line"] is False
    assert query["authenticated_job_ids"] == []
    assert query["unresolved_job_ids"] == ["41001", "41002"]
    assert query["records"] == {"off": [], "on": []}
    assert {
        arm: [label["job_id"] for label in receipt["authenticated_jobs"][arm]]
        for arm in ("off", "on")
    } == {"off": ["41001"], "on": ["41002"]}
    assert receipt["cancellations"][0]["job_ids"] == ["41001", "41002"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "CANCELLED"}


def test_scheduler_workdir_mismatch_is_never_authenticated() -> None:
    run = _run_pair("--submit", deployment_kind="scheduler_wrong_workdir")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "rollback-unconfirmed"
    assert receipt["stage"] == "pre_release_validation"
    assert receipt["rollback_confirmed"] is False
    recovery = receipt["recovery_query"]
    assert recovery["line_count"] == 2
    assert recovery["parsed_record_count"] == 2
    assert recovery["identity_match_counts"] == {"off": 1, "on": 0}
    assert recovery["records"]["off"][0]["work_dir"] == (
        receipt["execution_environment"]["arms"]["off"]["scheduler"][
            "slurm_submit_dir"
        ]
    )
    assert recovery["records"]["on"] == []
    assert [
        record["job_id"] for record in receipt["authenticated_jobs"]["off"]
    ] == ["41001"]
    assert receipt["authenticated_jobs"]["on"] == []
    assert receipt["cancellations"][0]["job_ids"] == ["41001"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "HELD"}


@pytest.mark.parametrize(
    ("deployment_kind", "query_name", "phase", "stage"),
    [
        (
            "pre_release_malformed_duplicate",
            "pre_release_query",
            "pre",
            "pre_release_validation",
        ),
        (
            "post_release_malformed_duplicate",
            "post_release_query",
            "post",
            "post_release_validation",
        ),
    ],
)
def test_lifecycle_query_rejects_lf_terminated_malformed_duplicate(
    deployment_kind: str, query_name: str, phase: str, stage: str
) -> None:
    run = _run_pair("--submit", deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "failed-closed"
    assert receipt["stage"] == stage
    assert receipt["rollback_confirmed"] is True
    query = receipt[query_name]
    assert query["phase"] == phase
    assert query["status"] == 0
    assert query["normalization_status"] == 1
    assert query["complete"] is False
    assert query["unterminated_final_line"] is False
    assert query["line_count"] == 3
    assert query["authenticated_job_ids"] == []
    assert query["unresolved_job_ids"] == ["41001", "41002"]
    assert {
        arm: [label["job_id"] for label in receipt["authenticated_jobs"][arm]]
        for arm in ("off", "on")
    } == {"off": ["41001"], "on": ["41002"]}
    assert receipt["cancellations"][0]["job_ids"] == ["41001", "41002"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "CANCELLED"}
    assert run.submission_state_residue == ()


@pytest.mark.parametrize("deployment_kind", ["sbatch_empty", "sbatch_nonnumeric"])
def test_submit_rejects_missing_or_malformed_job_ids(
    deployment_kind: str,
) -> None:
    run = _run_pair("--submit", deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert "strict sbatch --parsable output must be one numeric job ID" in (
        run.process.stderr
    )
    assert "STRICT_PAIR_LAUNCHED" not in run.process.stdout
    assert run.export_files_present_after_process is False


def test_submit_rejects_duplicate_off_on_job_ids() -> None:
    run = _run_pair("--submit", deployment_kind="sbatch_duplicate")
    assert run.process.returncode != 0
    assert "strict OFF and ON held submissions returned the same job ID" in (
        run.process.stderr
    )
    assert run.submission_receipt is not None
    assert run.submission_receipt_mode == 0o400
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "failed-closed"
    assert receipt["stage"] == "job_id_validation"
    assert receipt["rollback_confirmed"] is True
    authenticated_job_ids = {
        arm: [label["job_id"] for label in receipt["authenticated_jobs"][arm]]
        for arm in ("off", "on")
    }
    assert sorted(authenticated_job_ids["off"] + authenticated_job_ids["on"]) == [
        "41001"
    ]
    authenticated_arm = next(
        arm for arm, job_ids in authenticated_job_ids.items() if job_ids
    )
    unauthenticated_arm = "on" if authenticated_arm == "off" else "off"
    assert receipt["cancellations"][0]["job_ids"] == ["41001"]
    assert receipt["pre_cancel_queries"][-1]["authenticated_job_ids"] == ["41001"]
    assert receipt["post_cancel_queries"][-1]["authenticated_job_ids"] == ["41001"]
    assert run.scheduler_states == {"41001": "CANCELLED"}
    assert run.submission_state_residue == ()
    assert (
        sum(
            line.startswith("STRICT_PAIR_RESULT ")
            for line in run.process.stdout.splitlines()
        )
        == 1
    )
    result_line = next(
        line
        for line in run.process.stdout.splitlines()
        if line.startswith("STRICT_PAIR_RESULT ")
    )
    assert (
        f"{unauthenticated_arm}_candidate_job_id=41001 "
        f"{unauthenticated_arm}_identity_authenticated=false"
    ) in result_line
    assert (
        f"{authenticated_arm}_job_id=41001 "
        f"{authenticated_arm}_identity_authenticated=true"
    ) in result_line
    assert f"{unauthenticated_arm}_job_id=41001" not in result_line
    assert "STRICT_PAIR_LAUNCHED" not in run.process.stdout
    assert run.export_files_present_after_process is False


def test_duplicate_job_id_with_failed_recovery_uses_unattributed_retry() -> None:
    run = _run_pair(
        "--submit", deployment_kind="sbatch_duplicate_recovery_query_failure"
    )
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "rollback-unconfirmed"
    assert receipt["stage"] == "job_id_validation"
    assert receipt["rollback_confirmed"] is False
    assert receipt["recovery_query"]["status"] == 75
    retry = receipt["pre_cancel_queries"][0]
    assert retry["candidate_job_ids"] == {
        "off": [],
        "on": [],
        "unattributed": ["41001"],
    }
    assert retry["authenticated_job_ids"] == ["41001"]
    assert receipt["cancellations"][0]["job_ids"] == ["41001"]
    assert run.scheduler_states == {"41001": "CANCELLED"}
    assert run.submission_state_residue == ()


def test_malformed_recovery_line_cannot_confirm_rollback() -> None:
    run = _run_pair(
        "--submit", deployment_kind="recovery_malformed_identity_no_relay"
    )
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "rollback-unconfirmed"
    assert receipt["stage"] == "job_id_validation"
    assert receipt["rollback_confirmed"] is False
    recovery = receipt["recovery_query"]
    assert recovery["line_count"] == 2
    assert recovery["parsed_record_count"] == 1
    assert receipt["rollback_candidates"] == {
        "off": ["41001"],
        "on": ["41002"],
        "unattributed": [],
    }
    assert {
        arm: [label["job_id"] for label in receipt["authenticated_jobs"][arm]]
        for arm in ("off", "on")
    } == {"off": ["41001"], "on": ["41002"]}
    assert receipt["cancellations"][0]["job_ids"] == ["41001", "41002"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "CANCELLED"}
    assert run.submission_state_residue == ()


def test_duplicate_recovery_occurrences_are_preserved_but_canceled_once() -> None:
    run = _run_pair("--submit", deployment_kind="scontrol_duplicate_query")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "failed-closed"
    assert receipt["rollback_confirmed"] is True
    recovery = receipt["recovery_query"]
    assert recovery["identity_match_counts"] == {"off": 2, "on": 1}
    assert [record["job_id"] for record in recovery["records"]["off"]] == [
        "41001",
        "41001",
    ]
    assert [record["job_id"] for record in recovery["records"]["on"]] == ["41002"]
    assert receipt["cancellations"][0]["job_ids"] == ["41001", "41002"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "CANCELLED"}
    assert run.submission_state_residue == ()


def test_recovery_cancellation_keeps_accepted_id_first_with_lower_extra_id() -> None:
    run = _run_pair("--submit", deployment_kind="recovery_lower_id")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "failed-closed"
    assert receipt["rollback_confirmed"] is True
    assert receipt["rollback_candidates"] == {
        "off": ["41001", "31001"],
        "on": ["41002"],
        "unattributed": [],
    }
    assert receipt["cancellations"][0]["job_ids"] == [
        "41001",
        "31001",
        "41002",
    ]
    assert run.scheduler_argv["scancel"] == ("41001,31001,41002",)
    assert run.scheduler_states == {
        "31001": "CANCELLED",
        "41001": "CANCELLED",
        "41002": "CANCELLED",
    }
    assert run.submission_state_residue == ()


def test_incomplete_visibility_keeps_unresolved_ids_rollback_unconfirmed() -> None:
    run = _run_pair("--submit", deployment_kind="recovery_partial_visibility")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "rollback-unconfirmed"
    assert receipt["rollback_confirmed"] is False
    assert receipt["rollback_candidates"] == {
        "off": ["51001", "41001"],
        "on": ["51002"],
        "unattributed": [],
    }
    assert [label["job_id"] for label in receipt["authenticated_jobs"]["off"]] == [
        "41001"
    ]
    assert receipt["authenticated_jobs"]["on"] == []
    assert receipt["held_submissions"]["off"]["candidate_job_id"] == "41001"
    assert receipt["held_submissions"]["on"]["candidate_job_id"] == "51002"
    assert receipt["cancellations"][0]["job_ids"] == ["41001"]
    assert {"51001", "51002"}.issubset(
        {
            job_id
            for query in receipt["pre_cancel_queries"]
            for job_id in query["unresolved_job_ids"]
        }
    )
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "HELD"}
    assert run.submission_state_residue == ()
    result_line = next(
        line
        for line in run.process.stdout.splitlines()
        if line.startswith("STRICT_PAIR_RESULT ")
    )
    assert "off_job_id=41001 off_identity_authenticated=true" in result_line
    assert "on_candidate_job_id=51002 on_identity_authenticated=false" in result_line
    assert "on_job_id=51002" not in result_line


def test_unterminated_recovery_line_requires_complete_targeted_authentication() -> None:
    run = _run_pair("--submit", deployment_kind="recovery_unterminated_candidate")
    assert run.process.returncode != 0
    assert run.submission_receipt is not None
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "rollback-unconfirmed"
    assert receipt["rollback_confirmed"] is False
    recovery = receipt["recovery_query"]
    assert recovery["unterminated_final_line"] is True
    assert recovery["unterminated_candidate_job_ids"] == ["41001"]
    assert recovery["identity_match_counts"] == {"off": 0, "on": 0}
    assert recovery["records"] == {"off": [], "on": []}
    assert receipt["rollback_candidates"]["unattributed"] == ["41001"]
    assert receipt["held_submissions"]["off"]["candidate_job_id"] == "51001"
    assert receipt["held_submissions"]["on"]["candidate_job_id"] == "51002"
    assert [label["job_id"] for label in receipt["authenticated_jobs"]["off"]] == [
        "41001"
    ]
    assert receipt["authenticated_jobs"]["on"] == []
    promoted = [
        query
        for query in receipt["pre_cancel_queries"]
        if query["candidate_job_ids"]["unattributed"] == ["41001"]
        and query["authenticated_job_ids"] == ["41001"]
    ]
    assert len(promoted) == 1
    assert promoted[0]["unterminated_final_line"] is False
    assert promoted[0]["complete"] is True
    assert receipt["cancellations"][0]["job_ids"] == ["41001"]
    assert run.scheduler_states == {"41001": "CANCELLED", "41002": "HELD"}
    assert run.submission_state_residue == ()


def test_publisher_candidate_unlink_fault_adopts_exact_committed_receipt() -> None:
    run = _run_pair("--submit", deployment_kind="publisher_candidate_unlink_failure")
    assert run.process.returncode == 0
    assert run.submission_receipt is not None
    assert run.submission_receipt_mode == 0o400
    assert run.submission_receipt_nlink == 1
    receipt = json.loads(run.submission_receipt)
    assert receipt["outcome"] == "released"
    assert receipt["rollback_confirmed"] is None
    assert run.scheduler_states == {"41001": "RELEASED", "41002": "RELEASED"}
    assert "scancel" not in run.scheduler_argv
    assert run.submission_state_residue == ()


def test_arbitrary_preexisting_submission_receipt_is_not_adopted() -> None:
    hostile_receipt = b'{"schema":"nemo-rl-strict-pair-submission-receipt-v1"}\n'
    run = _run_pair(
        "--submit",
        preexisting_submission_receipt=hostile_receipt,
    )
    assert run.process.returncode != 0
    assert run.submission_receipt == hostile_receipt
    assert run.submission_receipt_mode == 0o400
    assert run.scheduler_states == {}
    assert run.sbatch_argv == {}
    assert "PAIR_SUBMISSION_RECEIPT.json must be absent" in run.process.stderr
    assert "STRICT_PAIR_RESULT " not in run.process.stdout


def test_export_renderer_emits_exact_sorted_nul_records_and_requires_every_name(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.env"
    candidate.write_bytes(b"")
    special = "comma,value=with-equals\nand-newline"
    result = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + 'strict_pair_render_slurm_export_payload "${TEST_ROOT}/candidate.env"',
        SPECIAL_VALUE=special,
    )
    assert result.returncode == 0, result.stderr
    expected_values = {
        name: f"value:{name}".encode("ascii") for name in SLURM_EXPORT_ALLOWED_NAMES
    }
    expected_values["HF_TOKEN"] = special.encode("ascii")
    expected = b"".join(
        name.encode("ascii") + b"=" + expected_values[name] + b"\0"
        for name in SLURM_EXPORT_ALLOWED_NAMES
    )
    assert candidate.read_bytes() == expected
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o400

    candidate.chmod(0o600)
    missing = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + """
unset MODEL_PATH
strict_pair_render_slurm_export_payload "${TEST_ROOT}/candidate.env"
""",
    )
    assert missing.returncode != 0
    assert "strict Slurm export value is unset: MODEL_PATH" in missing.stderr


def test_export_rerender_rejects_value_drift_after_pair_anchor(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir(mode=0o700)
    target = exports / "off.env"
    prepared = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + 'strict_pair_publish_or_verify_slurm_export "${TEST_ROOT}/exports/off.env"',
        SPECIAL_VALUE="first-secret-value",
    )
    assert prepared.returncode == 0, prepared.stderr
    expected_sha256 = _sha256(target)

    rerendered = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + """
strict_pair_publish_or_verify_slurm_export \
  "${TEST_ROOT}/exports/off.env" "${EXPECTED_EXPORT_SHA256}"
""",
        SPECIAL_VALUE="different-secret-value",
        EXPECTED_EXPORT_SHA256=expected_sha256,
    )
    assert rerendered.returncode != 0
    assert (
        "strict Slurm export bytes differ from the parent-bound arm payload"
        in rerendered.stderr
    )
    assert list(exports.glob(".export.*")) == []


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("mode", "strict Slurm export file must have mode 400"),
        ("symlink", "strict Slurm export file must be a regular, non-symlink file"),
    ],
)
def test_export_verification_failure_removes_secret_candidate(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir(mode=0o700)
    target = exports / "off.env"
    prepared = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + 'strict_pair_publish_or_verify_slurm_export "${TEST_ROOT}/exports/off.env"',
    )
    assert prepared.returncode == 0, prepared.stderr
    expected_sha256 = _sha256(target)
    if tamper == "mode":
        target.chmod(0o600)
    else:
        symlink_target = exports / "off.target.env"
        target.rename(symlink_target)
        target.symlink_to(symlink_target)

    verified = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + """
strict_pair_publish_or_verify_slurm_export \
  "${TEST_ROOT}/exports/off.env" "${EXPECTED_EXPORT_SHA256}"
""",
        EXPECTED_EXPORT_SHA256=expected_sha256,
    )
    assert verified.returncode != 0
    assert message in verified.stderr
    assert list(exports.glob(".export.*")) == []


def test_export_mutation_is_rejected_immediately_before_sbatch(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir(mode=0o700)
    target = exports / "on.env"
    prepared = _run_contract_harness(
        tmp_path,
        _INITIALIZE_SLURM_EXPORT_VALUES
        + 'strict_pair_publish_or_verify_slurm_export "${TEST_ROOT}/exports/on.env"',
    )
    assert prepared.returncode == 0, prepared.stderr
    expected_sha256 = _sha256(target)
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b"mutation")
    target.chmod(0o400)

    verified = _run_contract_harness(
        tmp_path,
        """
strict_pair_verify_slurm_export_before_sbatch \
  "${TEST_ROOT}/exports/on.env" "${EXPECTED_EXPORT_SHA256}"
""",
        EXPECTED_EXPORT_SHA256=expected_sha256,
    )
    assert verified.returncode != 0
    assert "changed immediately before sbatch" in verified.stderr


def test_partial_git_inventory_failure_has_no_snapshot_side_effects(
    tmp_path: Path,
) -> None:
    result = _run_contract_harness(
        tmp_path,
        """
RESULTS_DIR="${TEST_ROOT}"
STRICT_PAIR_PROJECT_ROOT="${TEST_ROOT}"
strict_pair_git() {
  printf 'partial-runtime.py\\0'
  return 73
}
if strict_pair_write_snapshot_path_inventory \
    "${TEST_ROOT}/snapshot-paths.nul"; then
  echo "partial git inventory unexpectedly succeeded" >&2
  exit 91
fi
test ! -e "${TEST_ROOT}/snapshot-paths.nul"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "tracked snapshot path enumeration failed" in result.stderr
    assert list(tmp_path.glob(".strict-pair-git-ls-files.*")) == []
    assert not (tmp_path / "snapshot").exists()


def test_partial_find_inventory_failure_has_no_verification_side_effects(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    result = _run_contract_harness(
        tmp_path,
        """
partial_find() {
  printf '%s\\0' "${1}/partial-runtime.py"
  return 73
}
STRICT_PAIR_TOOL_FIND=partial_find
if strict_pair_write_snapshot_find_inventory \
    "${TEST_ROOT}/snapshot" "${TEST_ROOT}/find-inventory.nul" off; then
  echo "partial find inventory unexpectedly succeeded" >&2
  exit 92
fi
test ! -e "${TEST_ROOT}/find-inventory.nul"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "off strict snapshot inventory enumeration failed" in result.stderr
    assert list(snapshot.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="requires non-root permission checks")
def test_model_tree_walk_fails_closed_on_directory_enumeration_error(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    unreadable = model / "unreadable"
    unreadable.mkdir(parents=True)
    unreadable.chmod(0)
    try:
        result = _run_contract_harness(
            tmp_path,
            'strict_pair_model_tree_sha256_v1 "${TEST_ROOT}/model"',
        )
    finally:
        unreadable.chmod(0o700)
    assert result.returncode != 0
    assert "PermissionError" in result.stderr or "Permission denied" in result.stderr


@pytest.mark.parametrize("startup_name", ["BASH_ENV", "ENV"])
def test_pair_rejects_shell_startup_files_before_they_execute(
    startup_name: str,
) -> None:
    run = _run_pair(startup_attack=startup_name)
    assert run.process.returncode == 2
    assert f"hostile startup environment variable must be unset: {startup_name}" in (
        run.process.stderr
    )
    assert "HOSTILE_STARTUP_EXECUTED" not in run.process.stderr
    assert run.manifest is None


@pytest.mark.parametrize(
    "tool_name",
    sorted({*HOST_TOOLS, "python3"}),
)
def test_pair_rejects_exported_tool_functions_without_executing_them(
    tool_name: str,
) -> None:
    run = _run_pair(startup_attack=f"function:{tool_name}")
    assert run.process.returncode == 2
    assert "hostile startup environment variable must be unset: BASH_FUNC_" in (
        run.process.stderr
    )
    assert "HOSTILE_EXPORTED_FUNCTION_EXECUTED" not in run.process.stderr
    assert run.manifest is None


@pytest.mark.parametrize(
    "environment_name",
    [
        "BASH_COMPAT",
        "CDPATH",
        "GIT_CONFIG_GLOBAL",
        "GIT_EXTERNAL_DIFF",
        "GLOBIGNORE",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    ],
)
def test_pair_rejects_hostile_interpreter_and_tool_environment(
    environment_name: str,
) -> None:
    run = _run_pair(startup_attack=f"environment:{environment_name}")
    assert run.process.returncode == 2
    assert (
        f"hostile startup environment variable must be unset: {environment_name}"
        in run.process.stderr
    )
    assert run.manifest is None


def test_pair_rejects_nonprivileged_explicit_bash_invocation() -> None:
    run = _run_pair(direct_invocation=False)
    assert run.process.returncode == 2
    assert "must be executed directly through its privileged Bash shebang" in (
        run.process.stderr
    )
    assert run.manifest is None


@pytest.mark.parametrize(
    "environment", ["", "reasoning-gym", "CITATION", "citation,freeform"]
)
def test_pair_rejects_environment_outside_closed_selector(environment: str) -> None:
    run = _run_pair(STRICT_PAIR_ENVIRONMENT=environment)
    assert run.process.returncode == 2
    assert (
        "STRICT_PAIR_ENVIRONMENT must be exactly reasoning_gym, citation, or freeform"
        in run.process.stderr
    )
    assert run.manifest is None
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    ("environment", "config_name", "metric", "gym_config"),
    [
        (
            "reasoning_gym",
            "single_env_reasoning_gym_sc.yaml",
            "train/reasoning_gym_simple_agent/score/mean",
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
        ),
        (
            "citation",
            "single_env_citation_sc.yaml",
            "train/citation_format_simple_agent/reward/mean",
            "resources_servers/format_verification/configs/citation_format.yaml",
        ),
        (
            "freeform",
            "single_env_freeform_sc.yaml",
            "train/freeform_formatting_simple_agent/reward/mean",
            ("resources_servers/format_verification/configs/freeform_formatting.yaml"),
        ),
    ],
)
def test_closed_environment_selector_maps_exact_metric_and_assets(
    tmp_path: Path,
    environment: str,
    config_name: str,
    metric: str,
    gym_config: str,
) -> None:
    run = _run_contract_harness(
        tmp_path,
        """
strict_pair_select_environment
printf '%s\\t%s\\t%s\\t%s\\t%s\\n' \\
  "${STRICT_PAIR_CONFIG_RELATIVE}" \\
  "${STRICT_PAIR_VERIFIER_METRIC}" \\
  "${STRICT_PAIR_GYM_CONFIG_RELATIVE}" \\
  "${STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE}" \\
  "${STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE}"
""",
        STRICT_PAIR_ENVIRONMENT=environment,
    )
    assert run.returncode == 0, run.stderr
    config_path, actual_metric, actual_gym_config, verifier, requirements = (
        run.stdout.strip().split("\t")
    )
    assert config_path.endswith(f"/{config_name}")
    assert actual_metric == metric
    assert actual_gym_config == gym_config
    expected_root = (
        "resources_servers/reasoning_gym"
        if environment == "reasoning_gym"
        else "resources_servers/format_verification"
    )
    assert verifier == f"{expected_root}/app.py"
    assert requirements == f"{expected_root}/requirements.txt"


@pytest.mark.parametrize(
    ("config_name", "gym_config", "max_sequence_length"),
    [
        (
            "single_env_reasoning_gym_sc.yaml",
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
            2048,
        ),
        (
            "single_env_citation_sc.yaml",
            "resources_servers/format_verification/configs/citation_format.yaml",
            4096,
        ),
        (
            "single_env_freeform_sc.yaml",
            (
                "resources_servers/format_verification/configs/"
                "freeform_formatting.yaml"
            ),
            2048,
        ),
    ],
)
def test_closed_environment_overlays_compose_as_strict_train_recipes(
    config_name: str, gym_config: str, max_sequence_length: int
) -> None:
    config = _compose_strict_config("train", RECIPE_DIR / config_name)
    assert config.policy["max_total_sequence_length"] == max_sequence_length
    assert config.policy["generation"]["vllm_cfg"]["max_model_len"] == (
        max_sequence_length
    )
    assert config.env["nemo_gym"]["config_paths"][-1] == gym_config
    assert config.grpo.max_num_steps == 100
    assert config.grpo.max_num_epochs == 20
    assert config.policy["shared_prefix_training"]["mode"] == "train"
    assert config.policy["shared_prefix_training"][
        "require_deterministic_execution"
    ]


def test_off_and_on_commands_differ_only_by_arm_identity_paths_and_mode() -> None:
    run = _run_pair()
    assert run.process.returncode == 0, run.process.stderr
    commands = _training_commands(run.process)
    off = _command_for_mode(commands, "observe")
    on = _command_for_mode(commands, "train")
    normalized_on = (
        on.replace(
            "policy.shared_prefix_training.mode=train",
            "policy.shared_prefix_training.mode=observe",
        )
        .replace(
            f"logger.wandb.name=on-reasoning_gym-{run.pair_id}",
            f"logger.wandb.name=off-reasoning_gym-{run.pair_id}",
        )
        .replace("/results/on/", "/results/off/")
        .replace("/hf-cache/on", "/hf-cache/off")
    )
    assert off == normalized_on


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("OFF",),
        ("ON",),
        ("disabled",),
        ("off,on",),
        ("off", "on"),
        ("on", "off"),
        ("off", "not-a-hydra-override"),
    ],
)
def test_arm_wrapper_rejects_missing_invalid_or_ambiguous_arm(
    arguments: tuple[str, ...],
) -> None:
    run = _run_pair(*arguments, use_parent=False)
    assert run.process.returncode == 2
    assert "--- TRAIN_CMD ---" not in run.process.stdout


def test_direct_arm_launch_without_parent_anchor_is_rejected() -> None:
    run = _run_pair("off", use_parent=False)
    assert run.process.returncode != 0
    assert "Direct arm launch is forbidden; use launch_pair.sh" in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    "override",
    [
        "cluster.num_nodes=2",
        "+cluster.gpus_per_node=8",
        "++cluster.segment_size=2",
        "~cluster",
        "checkpointing.enabled=true",
        "grpo.max_num_epochs=1",
        "grpo.max_num_steps=1",
        "grpo.seed=7",
        "grpo.invalid_tool_call_advantage=null",
        "async_rl.min_groups_for_streaming_train=2",
        "~data_plane",
        "data.shuffle=true",
        "data.train.data_path=/tmp/bypass.jsonl",
        "policy.train_global_batch_size=8",
        "policy.make_sequence_length_divisible_by=8",
        "policy.quant_cfg=NVFP4_DEFAULT_CFG",
        "policy.shared_prefix_training.mode=disabled",
        "++policy.shared_prefix_training.require_deterministic_execution=false",
        "policy.megatron_cfg.tensor_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=1",
        "policy.megatron_cfg.expert_model_parallel_size=1",
        "policy.megatron_cfg.mtp_num_layers=0",
        "policy.megatron_cfg.recompute_granularity=selective",
        "policy.megatron_cfg.cuda_graph_impl=local",
        "policy.generation.nemo_gym_add_seed_per_rollout=false",
        "policy.generation.nemo_gym_per_rollout_seed_base=7",
        "policy.generation.max_new_tokens=767",
        "policy.generation.temperature=0.0",
        "policy.generation.vllm_cfg.tensor_parallel_size=2",
        "policy.generation.vllm_cfg.gpu_memory_utilization=0.85",
        "policy.generation.vllm_kwargs.max_num_seqs=8",
        "env.should_use_nemo_gym=false",
        "env.nemo_gym.config_paths=[]",
        "logger.wandb_enabled=false",
        "logger.tensorboard_enabled=true",
        "loss_fn.ratio_clip_min=0.1",
    ],
)
def test_arm_wrapper_rejects_every_scientific_override(override: str) -> None:
    run = _run_pair("off", override, use_parent=False)
    assert run.process.returncode == 2
    assert "forbids overriding protected Hydra key" in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    "override",
    ["=x", "+=x", "++=x", "~", " grpo.max_num_steps=1", "key@package=x"],
)
def test_arm_wrapper_rejects_malformed_override_keys(override: str) -> None:
    run = _run_pair("off", override, use_parent=False)
    assert run.process.returncode == 2
    assert "malformed" in run.process.stderr


def test_arm_wrapper_rejects_unknown_override_by_default() -> None:
    run = _run_pair("off", "diagnostics.enabled=true", use_parent=False)
    assert run.process.returncode == 2
    assert "does not accept unsupported Hydra key" in run.process.stderr


@pytest.mark.parametrize(
    ("env_overrides", "message"),
    [
        ({"USE_SNAPSHOT": "0"}, "live-tree/container fallback is forbidden"),
        ({"CODE_SNAPSHOT_DIRNAME": "caller"}, "CODE_SNAPSHOT_DIRNAME must be unset"),
        ({"ENABLE_MTP_INFERENCE": "1"}, "generation speculative decoding is off"),
        ({"NRL_MAX_STEPS": "1"}, "pins exactly 100 optimizer steps"),
        ({"TRITON_CACHE_AUTOTUNING": "1"}, "must be unset"),
        ({"TRITON_AUTOTUNE_BLOCK_X": "1"}, "TRITON_AUTOTUNE_BLOCK_X must be unset"),
        ({"TRITON_AUTOTUNE_BLOCKED": "1"}, "TRITON_AUTOTUNE_BLOCKED must be unset"),
        ({"WANDB_DISABLED": "false"}, "WANDB_DISABLED must be unset"),
        ({"WANDB_MODE": "offline"}, "WANDB_MODE must be unset or 'online'"),
        ({"WANDB_RUN_ID": "caller"}, "WANDB_RUN_ID must be unset"),
        ({"BASE_LOG_DIR": "/tmp/caller"}, "BASE_LOG_DIR must be unset"),
        ({"BATCH_SCRIPT": "/tmp/caller"}, "BATCH_SCRIPT must be unset"),
        ({"CPUS_PER_WORKER": "1"}, "CPUS_PER_WORKER must be unset"),
        ({"RAY_SUB": "/tmp/caller"}, "RAY_SUB must be unset"),
        ({"MOUNTS": "/tmp/a:/tmp/b"}, "MOUNTS must be unset"),
        ({"EXTRA_MOUNTS": "/tmp/a:/tmp/b"}, "EXTRA_MOUNTS must be unset"),
        ({"USE_CUSTOM_VLLM": "0"}, "USE_CUSTOM_VLLM must be unset"),
        ({"INTERACTIVE": "0"}, "INTERACTIVE must be unset"),
        (
            {"NEMO_SKILLS_SANDBOX_PORT": "9999"},
            "NEMO_SKILLS_SANDBOX_PORT must be unset",
        ),
        ({"RAY_LOG_SYNC_FREQUENCY": "1"}, "RAY_LOG_SYNC_FREQUENCY must be unset"),
        ({"SANDBOX_COMMAND": "/tmp/caller"}, "SANDBOX_COMMAND must be unset"),
        ({"SETUP_COMMAND": "caller"}, "SETUP_COMMAND must be unset"),
        ({"SLURM_SUBMIT_DIR": "/tmp/caller"}, "SLURM_SUBMIT_DIR must be unset"),
        ({"CHECKPOINTING_SAVE_BY": "01:00:00"}, "CHECKPOINTING_SAVE_BY must be unset"),
        ({"WALLTIME": "01:00:00"}, "WALLTIME must be unset"),
        ({"SLURM_QOS": "short"}, "SLURM_QOS must be unset"),
        ({"SLURM_ACCOUNT": "caller"}, "SLURM_ACCOUNT must be unset"),
        ({"DRY_RUN": "0"}, "DRY_RUN must be unset"),
        ({"WANDB_ENTITY": "caller"}, "WANDB_ENTITY must be unset"),
        ({"WANDB_NAME": "caller"}, "WANDB_NAME must be unset"),
        ({"WANDB_PROJ": "caller"}, "WANDB_PROJ must be unset"),
        ({"WANDB_RESUME": "allow"}, "WANDB_RESUME must be unset"),
        ({"WANDB_RUN_GROUP": "caller"}, "WANDB_RUN_GROUP must be unset"),
    ],
)
def test_pair_rejects_environment_bypasses(
    env_overrides: dict[str, str], message: str
) -> None:
    run = _run_pair(**env_overrides)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize("key", [None, "short", "contains whitespace secret"])
def test_pair_requires_valid_wandb_credentials(key: str | None) -> None:
    if key is None:
        run = _run_pair(include_wandb_key=False)
    else:
        run = _run_pair(WANDB_API_KEY=key)
    assert run.process.returncode != 0
    assert "WANDB_API_KEY" in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    ("fixture_kind", "message"),
    [
        ("mutated", "SHA-256 mismatch"),
        ("wrong_hash", "SHA-256 mismatch"),
        ("symlink", "regular, non-symlink file"),
        ("relative", "absolute and canonical"),
        ("directory", "regular, non-symlink file"),
        ("fifo", "regular, non-symlink file"),
        ("noncanonical", "must already be canonical"),
    ],
)
def test_pair_rejects_unsealed_fixture_path_or_bytes(
    fixture_kind: str, message: str
) -> None:
    run = _run_pair(fixture_kind=fixture_kind)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


def test_pair_requires_train_path() -> None:
    run = _run_pair(include_train_path=False)
    assert run.process.returncode != 0
    assert "TRAIN_PATH is required" in run.process.stderr


@pytest.mark.parametrize(
    ("env_overrides", "message"),
    [
        ({"EXPECTED_NEMO_HEAD": ""}, "EXPECTED_NEMO_HEAD is required"),
        (
            {"EXPECTED_NEMO_HEAD": "d7b49a459f08670b6534a56deb99e432f576028a"},
            "legacy NeMo-RL deployment d7b49a459f08670b6534a56deb99e432f576028a is forbidden",
        ),
        (
            {"EXPECTED_NEMO_HEAD": "0" * 40},
            "deployed NeMo-RL HEAD differs from EXPECTED_NEMO_HEAD",
        ),
        (
            {"EXPECTED_NEMO_HEAD": "A" * 40},
            "EXPECTED_NEMO_HEAD must be an explicit lowercase 40-hex Git commit",
        ),
        (
            {"EXPECTED_GYM_GITLINK_COMMIT": ""},
            "EXPECTED_GYM_GITLINK_COMMIT is required",
        ),
        (
            {"EXPECTED_GYM_GITLINK_COMMIT": "0" * 40},
            "Reasoning Gym gitlink differs from EXPECTED_GYM_GITLINK_COMMIT",
        ),
        ({"EXPECTED_BRIDGE_HEAD": ""}, "EXPECTED_BRIDGE_HEAD is required"),
        (
            {"EXPECTED_BRIDGE_HEAD": "0" * 40},
            "deployed Megatron-Bridge Git identity differs from its OOB anchors",
        ),
        ({"EXPECTED_MCORE_TREE": ""}, "EXPECTED_MCORE_TREE is required"),
        (
            {"EXPECTED_MCORE_TREE": "0" * 40},
            "deployed Megatron-LM Git identity differs from its OOB anchors",
        ),
    ],
)
def test_pair_requires_exact_nemo_and_reasoning_gym_commits(
    env_overrides: dict[str, str], message: str
) -> None:
    run = _run_pair(**env_overrides)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert run.manifest is None


def test_pair_rejects_launcher_source_outside_authenticated_deployment() -> None:
    run = _run_pair(use_deployed_script=False)
    assert run.process.returncode == 2
    assert (
        "launch_pair.sh must be the sealed deployment entrypoint" in run.process.stderr
    )
    assert run.manifest is None


@pytest.mark.parametrize(
    ("deployment_kind", "message"),
    [
        ("root_relative", "launch_pair.sh must be the sealed deployment entrypoint"),
        (
            "root_noncanonical",
            "launch_pair.sh must be the sealed deployment entrypoint",
        ),
        ("root_symlink", "launch_pair.sh must be the sealed deployment entrypoint"),
        ("root_mode", "DEPLOYMENT_ROOT must have mode 500"),
        ("ready_missing", "deployment READY must be a regular, non-symlink file"),
        ("ready_symlink", "deployment READY must be a regular, non-symlink file"),
        ("ready_mode", "deployment READY must have mode 444"),
        ("ready_hash", "deployment READY file SHA-256 mismatch"),
        ("ready_content", "deployment READY content differs"),
        (
            "nemo_manifest_missing",
            "pre-source NeMo-RL runnable manifest must be a regular non-symlink file",
        ),
        (
            "nemo_manifest_symlink",
            "pre-source NeMo-RL runnable manifest must be a regular non-symlink file",
        ),
        ("nemo_manifest_mode", "deployment runnable manifest must have mode 400"),
        (
            "nemo_manifest_hash",
            "pre-source NeMo-RL runnable manifest SHA-256 mismatch",
        ),
        ("nemo_manifest_payload_mutated", "content verification failed"),
        (
            "nemo_manifest_malformed",
            "launch_pair.sh is absent or drifted in the authenticated runnable manifest",
        ),
    ],
)
def test_pair_rejects_invalid_deployment_contract(
    deployment_kind: str, message: str
) -> None:
    run = _run_pair(deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    ("deployment_kind", "message"),
    [
        ("job_wrapper_missing", "content verification failed"),
        ("job_wrapper_symlink", "regular, non-symlink file"),
        ("job_wrapper_writable", "executable and have no write bits"),
        ("job_wrapper_hash", "STRICT_PAIR_JOB_WRAPPER SHA-256 mismatch"),
        ("job_wrapper_mutated", "content verification failed"),
        ("job_wrapper_not_listed", "is not authenticated by a runnable manifest"),
    ],
)
def test_pair_rejects_unauthenticated_job_wrapper(
    deployment_kind: str, message: str
) -> None:
    run = _run_pair(deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert run.manifest is None


@pytest.mark.parametrize(
    ("deployment_kind", "message"),
    [
        ("source_staged", "index differs from HEAD"),
        ("source_unstaged", "worktree differs from HEAD"),
        ("gym_unstaged", "worktree differs from HEAD"),
        ("bridge_unstaged", "worktree differs from HEAD"),
        ("mcore_unstaged", "worktree differs from HEAD"),
        ("source_untracked", "source contains untracked files"),
        ("source_ignored", "source contains ignored files"),
        ("source_mode_drift", "executable mode differs from index"),
    ],
)
def test_pair_rejects_dirty_or_mode_drifted_authenticated_source(
    deployment_kind: str, message: str
) -> None:
    run = _run_pair(deployment_kind=deployment_kind)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert run.manifest is None


def test_pair_accepts_exact_clean_authenticated_gitlink() -> None:
    run = _run_pair(deployment_kind="valid_submodule")
    assert run.process.returncode == 0, run.process.stderr
    assert "STRICT_PAIR_DRY_RUN_VALIDATED" in run.process.stdout


def test_pair_rejects_submodule_head_that_differs_from_gitlink() -> None:
    run = _run_pair(deployment_kind="wrong_gitlink")
    assert run.process.returncode != 0
    assert (
        "Reasoning Gym HEAD differs from its authenticated gitlink"
        in run.process.stderr
    )
    assert run.manifest is None


@pytest.mark.parametrize(
    ("snapshot_kind", "message"),
    [
        ("missing", "strict snapshot content verification failed"),
        ("mutated", "strict snapshot content verification failed"),
        ("symlink", "strict snapshot content verification failed"),
        ("missing_gym", "strict snapshot content verification failed"),
    ],
)
def test_pair_rejects_snapshot_entrypoint_fallback_or_mutation(
    snapshot_kind: str, message: str
) -> None:
    run = _run_pair(post_publication_snapshot_tamper=snapshot_kind)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


@pytest.mark.parametrize(
    ("deployment_kind", "tamper", "message"),
    [
        (
            "valid",
            "extra_readonly",
            "snapshot regular-file inventory differs from SHA manifest",
        ),
        (
            "valid",
            "rewrite_manifest",
            "PAIR_MANIFEST.json bytes differ from recomputed canonical inputs",
        ),
        (
            "valid_symlink",
            "symlink_target",
            "snapshot symlink inventory differs from symlink manifest",
        ),
        (
            "valid",
            "mode_drift",
            "snapshot executable-mode inventory differs from mode manifest",
        ),
    ],
)
def test_pair_detects_post_publication_snapshot_drift(
    deployment_kind: str, tamper: str, message: str
) -> None:
    run = _run_pair(
        deployment_kind=deployment_kind,
        post_publication_snapshot_tamper=tamper,
    )
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "STRICT_PAIR_DRY_RUN_VALIDATED" not in run.process.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="requires non-root permission checks")
def test_snapshot_walk_fails_closed_on_directory_enumeration_error() -> None:
    run = _run_pair(post_publication_snapshot_tamper="unreadable_directory")
    assert run.process.returncode != 0
    assert "PermissionError" in run.process.stderr or "Permission denied" in (
        run.process.stderr
    )
    assert "STRICT_PAIR_DRY_RUN_VALIDATED" not in run.process.stdout


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing", "PAIR_MANIFEST.json must be a regular, non-symlink file"),
        ("symlink", "PAIR_MANIFEST.json must be a regular, non-symlink file"),
        ("mode", "PAIR_MANIFEST.json must have mode 400"),
        (
            "mutation",
            "PAIR_MANIFEST.json SHA-256 differs from the parent-provided anchor",
        ),
    ],
)
def test_pair_detects_post_publication_pair_manifest_drift(
    tamper: str, message: str
) -> None:
    run = _run_pair(post_publication_pair_manifest_tamper=tamper)
    assert run.process.returncode != 0
    assert message in run.process.stderr
    assert "STRICT_PAIR_DRY_RUN_VALIDATED" not in run.process.stdout


def test_pair_rejects_preexisting_arm_snapshot() -> None:
    run = _run_pair(preexisting_snapshot=True)
    assert run.process.returncode == 2
    assert (
        "strict arm snapshot already exists or is reserved; use a new PAIR_ID"
        in run.process.stderr
    )
    assert run.manifest is None


def test_pair_rejects_preexisting_different_manifest() -> None:
    run = _run_pair(preexisting_pair_manifest="{}\n")
    assert run.process.returncode == 2
    assert "already exists with different contract bytes" in run.process.stderr
    assert "--- TRAIN_CMD ---" not in run.process.stdout


def test_atomic_pair_manifest_publishers_converge_on_identical_bytes(
    tmp_path: Path,
) -> None:
    digest = "d" * 64
    publishers = [
        _start_manifest_publisher(tmp_path, config_digest=digest) for _ in range(2)
    ]
    outcomes = [publisher.communicate(timeout=20) for publisher in publishers]
    assert [publisher.returncode for publisher in publishers] == [0, 0], outcomes
    rendered_digest = _sha256(tmp_path / "PAIR_MANIFEST.json")
    assert [stdout.strip() for stdout, _ in outcomes] == [
        rendered_digest,
        rendered_digest,
    ]
    assert stat.S_IMODE((tmp_path / "PAIR_MANIFEST.json").stat().st_mode) == 0o400


def test_atomic_pair_manifest_publishers_reject_concurrent_contract_collision(
    tmp_path: Path,
) -> None:
    publishers = [
        _start_manifest_publisher(tmp_path, config_digest=digest * 64)
        for digest in ("d", "e")
    ]
    outcomes = [publisher.communicate(timeout=20) for publisher in publishers]
    returncodes = sorted(publisher.returncode for publisher in publishers)
    assert returncodes == [0, 2], outcomes
    assert (
        sum(
            "PAIR_MANIFEST.json already exists with different contract bytes" in stderr
            for _, stderr in outcomes
        )
        == 1
    )


@pytest.mark.parametrize("mode", ["observe", "train"])
def test_strict_pair_config_composes_for_both_arms(mode: str) -> None:
    from omegaconf import OmegaConf
    from omegaconf.errors import MissingMandatoryValue

    from nemo_rl.algorithms.single_controller_utils.config import (
        validate_single_controller_config,
    )
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = load_config(CONFIG)
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    master_config = _compose_strict_config(mode)

    megatron = master_config.policy["megatron_cfg"]
    generation = master_config.policy["generation"]
    assert master_config.grpo.async_grpo is None
    assert master_config.data_plane["enabled"]
    assert master_config.grpo.num_prompts_per_step == 1
    assert master_config.grpo.num_generations_per_prompt == 4
    assert master_config.grpo.max_num_epochs == 20
    assert master_config.grpo.max_num_steps == 100
    assert master_config.grpo.seed == 42
    assert master_config.grpo.normalize_rewards is True
    assert master_config.grpo.use_leave_one_out_baseline is True
    assert master_config.grpo.adv_estimator.name == "grpo"
    assert master_config.grpo.adv_estimator.normalize_rewards is True
    assert master_config.grpo.adv_estimator.use_leave_one_out_baseline is True
    assert master_config.grpo.adv_estimator.reward_weights is None
    assert master_config.grpo.invalid_tool_call_advantage == -5.0
    assert master_config.grpo.malformed_thinking_advantage is None
    assert master_config.grpo.reward_scaling.model_dump() == {
        "enabled": False,
        "source_min": 0.0,
        "source_max": 1.0,
        "target_min": 0.0,
        "target_max": 1.0,
    }
    assert master_config.grpo.reward_shaping.model_dump() == {
        "enabled": False,
        "overlong_buffer_length": 128,
        "overlong_buffer_penalty": 1.0,
        "max_response_length": 768,
        "stop_properly_penalty_coef": None,
    }
    assert master_config.policy["train_global_batch_size"] == 4
    assert master_config.policy["train_micro_batch_size"] == 1
    assert master_config.policy["generation_batch_size"] == 4
    assert master_config.policy["logprob_batch_size"] == 1
    assert master_config.policy["max_total_sequence_length"] == 2048
    assert master_config.policy["logprob_chunk_size"] == 256
    assert master_config.policy["make_sequence_length_divisible_by"] == 128
    assert master_config.policy["quant_cfg"] is None
    assert megatron["tensor_model_parallel_size"] == 2
    assert megatron["context_parallel_size"] == 2
    assert megatron["pipeline_model_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 4
    assert megatron["expert_tensor_parallel_size"] == 1
    assert megatron["sequence_parallel"] is True
    assert megatron["activation_checkpointing"] is True
    assert megatron["mtp_num_layers"] == 5
    assert megatron["mtp_use_repeated_layer"] is True
    assert megatron["mtp_detach_heads"] is True
    assert megatron["mtp_loss_scaling_factor"] == 0.3
    assert megatron["recompute_granularity"] == "full"
    assert megatron["recompute_method"] == "uniform"
    assert megatron["recompute_num_layers"] == 1
    assert megatron["cuda_graph_impl"] == "none"
    assert megatron["env_vars"]["NRL_SP_DETERMINISTIC_BACKWARD"] == "1"
    assert megatron["env_vars"]["RESULTS_DIR"] == "/strict-pair/results"
    assert (
        megatron["env_vars"]["NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR"]
        == "/strict-pair/results/shared_prefix_determinism_receipts/12345-0"
    )
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 4
    assert generation["max_new_tokens"] == 768
    assert generation["temperature"] == 1.0
    assert generation["top_p"] == 1.0
    assert generation["top_k"] is None
    assert generation["val_temperature"] == 1.0
    assert generation["val_top_p"] == 1.0
    assert generation["val_top_k"] is None
    assert generation["nemo_gym_add_seed_per_rollout"] is True
    assert generation["nemo_gym_per_rollout_seed_base"] == 42
    assert master_config.policy["sequence_packing"]["train_mb_tokens"] == 4096
    assert master_config.policy["sequence_packing"]["logprob_mb_tokens"] == 4096
    assert not master_config.data["shuffle"]
    assert master_config.async_rl.min_groups_for_streaming_train == 1
    assert master_config.async_rl.max_inflight_prompts == 1
    assert master_config.async_rl.max_buffered_rollouts == 2
    assert not master_config.checkpointing["enabled"]
    assert master_config.logger["wandb_enabled"]
    assert not master_config.logger["tensorboard_enabled"]
    assert master_config.logger["wandb"]["entity"] == "nvidia"
    assert master_config.logger["wandb"]["project"] == "nano35-rlvr-convergence"
    nemo_gym = master_config.env["nemo_gym"]
    assert nemo_gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
    ]
    assert nemo_gym["skip_venv_if_present"] is True
    assert nemo_gym["port_range_low"] == 5000
    assert nemo_gym["port_range_high"] == 5999
    assert nemo_gym["gzip_responses_enabled"] is True
    assert nemo_gym["global_aiohttp_connector_limit"] == 4096
    assert nemo_gym["global_aiohttp_connector_limit_per_host"] == 4096
    assert not nemo_gym["run_response_cache_enabled"]
    service_config_kinds = {
        "responses_api_models",
        "responses_api_agents",
        "resources_servers",
    }
    embedded_service_names = {
        name
        for name, value in nemo_gym.items()
        if isinstance(value, dict) and service_config_kinds.intersection(value)
    }
    assert embedded_service_names == {"policy_model"}
    assert {
        "policy_model_reasoning_off",
        "safety_judge_model",
        "nl2bash_judge_model",
        "genrm_model",
    }.isdisjoint(nemo_gym)
    # The two authenticated Gym configs add the resource server and its agent;
    # the only embedded service is the policy-model overlay above.
    assert embedded_service_names | {
        "reasoning_gym",
        "reasoning_gym_simple_agent",
    } == {"policy_model", "reasoning_gym", "reasoning_gym_simple_agent"}
    assert master_config.env["should_mask_flagged_samples"] is True
    assert master_config.reward_penalties.model_dump() == {
        "penalize_duplicated_reasoning": True,
        "penalize_empty_final_answer": True,
        "penalize_unwanted_tokens": True,
        "penalize_malformed_think_tag": True,
        "token_ids": {"unwanted": [2], "think_open": 12, "think_close": 13},
    }
    shared_prefix = master_config.policy["shared_prefix_training"]
    assert shared_prefix.mode == mode
    assert shared_prefix.require_deterministic_execution

    validate_single_controller_config(master_config)


def test_strict_observe_recipe_passes_shared_prefix_policy_validation() -> None:
    from nemo_rl.models.policy import validate_shared_prefix_training_config

    master_config = _compose_strict_config("observe")
    validate_shared_prefix_training_config(master_config.policy)


def test_strict_train_recipe_passes_shared_prefix_policy_validation() -> None:
    from nemo_rl.models.policy import validate_shared_prefix_training_config

    master_config = _compose_strict_config("train")
    validate_shared_prefix_training_config(master_config.policy)
