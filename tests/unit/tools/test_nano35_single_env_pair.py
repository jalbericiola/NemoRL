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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_fake_rsync(root: Path) -> Path:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    rsync = fake_bin / "rsync"
    rsync.write_text(
        """#!/bin/bash
set -euo pipefail
destination="${!#}"
repo="${NANO35_TEST_REPO_ROOT:?}"
kind="${NANO35_TEST_SNAPSHOT_ENTRYPOINT_KIND:-valid}"
mkdir -p "${destination}/examples/nemo_gym/nemotron-3.5-nano"
cp -R "${repo}/examples/nemo_gym/nemotron-3.5-nano/." \
  "${destination}/examples/nemo_gym/nemotron-3.5-nano/"
mkdir -p "${destination}/examples"
mkdir -p "${destination}/tools"
cp "${repo}/.gitignore" "${destination}/.gitignore"
cp "${repo}/tools/code_snapshot.sh" "${destination}/tools/code_snapshot.sh"
if [[ -f "${repo}/.gitmodules" ]]; then
  cp "${repo}/.gitmodules" "${destination}/.gitmodules"
fi
if [[ -L "${repo}/tracked-runtime-link" ]]; then
  cp -P "${repo}/tracked-runtime-link" "${destination}/tracked-runtime-link"
fi
if [[ "${kind}" != "missing_gym" && \
      -f "${repo}/3rdparty/Gym-workspace/Gym/gym_runtime.py" ]]; then
  mkdir -p "${destination}/3rdparty/Gym-workspace/Gym"
  cp "${repo}/3rdparty/Gym-workspace/Gym/gym_runtime.py" \
    "${destination}/3rdparty/Gym-workspace/Gym/gym_runtime.py"
fi
case "${kind}" in
  valid|missing_gym)
    cp "${repo}/examples/run_grpo_single_controller.py" \
      "${destination}/examples/run_grpo_single_controller.py"
    ;;
  missing)
    ;;
  mutated)
    cp "${repo}/examples/run_grpo_single_controller.py" \
      "${destination}/examples/run_grpo_single_controller.py"
    chmod u+w "${destination}/examples/run_grpo_single_controller.py"
    printf '\n# snapshot mutation\n' >> \
      "${destination}/examples/run_grpo_single_controller.py"
    ;;
  symlink)
    ln -s "${repo}/examples/run_grpo_single_controller.py" \
      "${destination}/examples/run_grpo_single_controller.py"
    ;;
  *)
    echo "unknown snapshot kind: ${kind}" >&2
    exit 64
    ;;
esac
""",
        encoding="utf-8",
    )
    rsync.chmod(0o755)
    bash = fake_bin / "bash"
    bash.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == */nano35_single_env_pair.sh && \
      -n "${NANO35_TEST_PAIR_MANIFEST_TAMPER:-}" ]] && \
   mkdir "${RESULTS_DIR}/.strict-pair-test-manifest-tamper-once" 2>/dev/null; then
  manifest="${RESULTS_DIR}/PAIR_MANIFEST.json"
  case "${NANO35_TEST_PAIR_MANIFEST_TAMPER}" in
    missing)
      rm "${manifest}"
      ;;
    symlink)
      target="${RESULTS_DIR}/PAIR_MANIFEST.target.json"
      mv "${manifest}" "${target}"
      ln -s "${target}" "${manifest}"
      ;;
    mode)
      chmod 600 "${manifest}"
      ;;
    mutation)
      chmod 600 "${manifest}"
      printf 'mutation\n' >> "${manifest}"
      chmod 400 "${manifest}"
      ;;
    *)
      echo "unknown pair-manifest tamper: ${NANO35_TEST_PAIR_MANIFEST_TAMPER}" >&2
      exit 66
      ;;
  esac
fi
if [[ "${1:-}" == */nano35_single_env_pair.sh && \
      -n "${NANO35_TEST_ARM_TAMPER:-}" ]] && \
   mkdir "${RESULTS_DIR}/.strict-pair-test-snapshot-tamper-once" 2>/dev/null; then
  arm="${2:?arm argument is required}"
  snapshot="${RESULTS_DIR}/code_snapshots_strict_pairs/${PAIR_ID}/${arm}-${PAIR_ID}"
  case "${NANO35_TEST_ARM_TAMPER}" in
    extra_readonly)
      chmod u+w "${snapshot}"
      printf 'unlisted runtime bytes\n' > "${snapshot}/unlisted_runtime.py"
      chmod 400 "${snapshot}/unlisted_runtime.py"
      chmod a-w "${snapshot}"
      ;;
    rewrite_manifest)
      target="${snapshot}/examples/run_grpo_single_controller.py"
      manifest="${snapshot}/strict-pair-snapshot-manifest.sha256"
      chmod u+w "${target}" "${manifest}" "${snapshot}"
      printf '\n# post-publication mutation\n' >> "${target}"
      digest="$(sha256sum "${target}" | awk '{print $1}')"
      temporary="${snapshot}/.rewritten-manifest"
      while read -r old relative; do
        if [[ "${relative}" == "examples/run_grpo_single_controller.py" ]]; then
          printf '%s  %s\n' "${digest}" "${relative}"
        else
          printf '%s  %s\n' "${old}" "${relative}"
        fi
      done < "${manifest}" > "${temporary}"
      mv "${temporary}" "${manifest}"
      chmod 400 "${target}" "${manifest}"
      chmod a-w "${snapshot}"
      ;;
    symlink_target)
      link="${snapshot}/tracked-runtime-link"
      chmod u+w "${snapshot}"
      rm "${link}"
      ln -s examples/run_grpo_single_controller.py "${link}"
      chmod a-w "${snapshot}"
      ;;
    mode_drift)
      chmod u+x "${snapshot}/.gitignore"
      ;;
    *)
      echo "unknown arm tamper: ${NANO35_TEST_ARM_TAMPER}" >&2
      exit 65
      ;;
  esac
fi
exec /bin/bash "$@"
""",
        encoding="utf-8",
    )
    bash.chmod(0o755)
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
        "single_env_reasoning_gym_sc.yaml",
        "strict_pair_contract.sh",
    ):
        shutil.copy2(RECIPE_DIR / name, recipe / name)
    (nemo_root / "examples").mkdir(exist_ok=True)
    shutil.copy2(ENTRYPOINT, nemo_root / "examples/run_grpo_single_controller.py")
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

    job_wrapper = deployment / "strict_pair_job_wrapper.sh"
    job_wrapper.write_text("#!/bin/bash\nset -euo pipefail\n", encoding="ascii")
    job_wrapper.chmod(0o500)
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
            manifest.write_text("\n".join(sorted(entries)) + "\n", encoding="ascii")
        else:
            manifest.write_text(f"{_sha256(payload)}  {payload}\n", encoding="ascii")
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
    }
    env.update(
        {
            expected_name: _sha256(manifests[manifest_name])
            for manifest_name, expected_name in expected_names.items()
        }
    )
    env["STRICT_PAIR_JOB_WRAPPER"] = str(job_wrapper)
    env["EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256"] = _sha256(job_wrapper)

    if kind in {"valid", "valid_submodule", "valid_symlink", "wrong_gitlink"}:
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
                "--chmod=-x",
                "examples/nemo_gym/nemotron-3.5-nano/launch_pair.sh",
            ],
            check=True,
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
    if kind == "source_mode_drift":
        (nemo_root / ".gitignore").chmod(0o500)
    nemo_root.chmod(0o500)
    deployment.chmod(0o700 if kind == "root_mode" else 0o500)
    return deployment, nemo_root, job_wrapper, env


def _run_pair(
    *arguments: str,
    use_parent: bool = True,
    use_deployed_script: bool = True,
    include_wandb_key: bool = True,
    include_train_path: bool = True,
    fixture_kind: str = "valid",
    deployment_kind: str = "valid",
    snapshot_kind: str = "valid",
    preexisting_snapshot: bool = False,
    preexisting_pair_manifest: str | None = None,
    **env_overrides: str,
) -> PairRun:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir).resolve()
        pair_id = env_overrides.pop("PAIR_ID", f"strict-{root.name}")
        fake_bin = _write_fake_rsync(root)
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
        results.mkdir()
        cache.mkdir()
        deployment, nemo_root, _, deployment_env = _make_deployment(
            root, deployment_kind
        )

        env = {
            "HOME": str(root),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PAIR_ID": pair_id,
            "EXP_NAME": "caller-must-not-select",
            "MODEL_PATH": str(model),
            "CONTAINER": str(container),
            "SANDBOX_CONTAINER": str(sandbox_container),
            "PERSISTENT_CACHE": str(cache),
            "RESULTS_DIR": str(results),
            "SLURM_PARTITION": "caller-must-not-select",
            "HF_HOME": str(root / "hf-cache"),
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
            "NANO35_TEST_REPO_ROOT": str(nemo_root),
            "NANO35_TEST_SNAPSHOT_ENTRYPOINT_KIND": snapshot_kind,
            **deployment_env,
        }
        if include_train_path:
            env["TRAIN_PATH"] = train_path
        if include_wandb_key:
            env["WANDB_API_KEY"] = "0" * 32
        env.update({key: str(value) for key, value in env_overrides.items()})

        snapshot_parent = results / "code_snapshots_strict_pairs" / pair_id
        if preexisting_snapshot:
            snapshot = snapshot_parent / f"off-{pair_id}"
            snapshot.mkdir(parents=True)
        if preexisting_pair_manifest is not None:
            manifest_path = results / "PAIR_MANIFEST.json"
            manifest_path.write_text(preexisting_pair_manifest, encoding="ascii")
            manifest_path.chmod(0o400)

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
        command = ["bash", str(script), *effective_arguments]
        try:
            process = subprocess.run(
                command,
                cwd=nemo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            manifest_path = results / "PAIR_MANIFEST.json"
            manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
            manifest_mode = (
                stat.S_IMODE(manifest_path.stat().st_mode)
                if manifest_path.is_file()
                else None
            )
        finally:
            shutil.rmtree(snapshot_parent, ignore_errors=True)
            deployment.chmod(0o700)
            nemo_root.chmod(0o700)
        return PairRun(process, pair_id, manifest, manifest_mode)


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


def _compose_strict_config(mode: str):
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
        load_config(CONFIG),
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


def _start_manifest_publisher(
    root: Path, *, config_digest: str
) -> subprocess.Popen[str]:
    script = root / "publish-pair-manifest.sh"
    if not script.exists():
        script.write_text(
            """#!/bin/bash
set -euo pipefail
source "${STRICT_PAIR_CONTRACT_PATH}"
PAIR_ID=concurrent-pair
PERSISTENT_CACHE="${RESULTS_DIR}/cache"
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
STRICT_PAIR_PROJECT_ROOT=/deployment/runnable/NemoRL
STRICT_PAIR_NEMO_HEAD="$(printf '4%.0s' {1..40})"
STRICT_PAIR_NEMO_TREE="$(printf '5%.0s' {1..40})"
STRICT_PAIR_GYM_ROOT=/deployment/runnable/NemoRL/3rdparty/Gym-workspace/Gym
STRICT_PAIR_GYM_GITLINK_COMMIT="$(printf '6%.0s' {1..40})"
STRICT_PAIR_GYM_TREE="$(printf '7%.0s' {1..40})"
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
    }
    return subprocess.Popen(
        ["/bin/bash", str(script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_authoritative_pair_builds_exact_parallel_arm_contract() -> None:
    run = _run_pair()
    result = run.process

    assert result.returncode == 0, result.stderr
    commands = _training_commands(result)
    assert len(commands) == 2
    for arm, mode in (("off", "observe"), ("on", "train")):
        command = _command_for_mode(commands, mode)
        required_once = (
            "uv run ./examples/run_grpo_single_controller.py",
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
            "++policy.generation.refit_transport=null",
            "env.nemo_gym.num_gpu_nodes=0",
            "logger.wandb_enabled=True",
            f"logger.wandb.name={arm}-{run.pair_id}",
            "logger.wandb.project=nano35-rlvr-convergence",
        )
        for token in required_once:
            assert command.count(token) == 1, token
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

    assert run.manifest is not None
    assert run.manifest_mode == 0o400
    manifest = json.loads(run.manifest)
    assert manifest["schema"] == "nemo-rl-strict-single-env-pair-v1"
    assert manifest["pair_id"] == run.pair_id
    assert manifest["determinism_receipt_dir"] == "shared_prefix_determinism_receipts"
    assert manifest["arms"] == {"off": "observe", "on": "train"}
    assert manifest["campaign"]["training_topology"] == "TP2/CP2/PP1/EP4/ETP1/SP"
    assert manifest["campaign"]["padding_multiple"] == 128
    assert manifest["campaign"]["epochs"] == 20
    assert manifest["campaign"]["steps"] == 100
    assert manifest["campaign"]["generations_per_prompt"] == 4
    assert manifest["campaign"]["require_deterministic_execution"] is True
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
        "wandb_project": "nano35-rlvr-convergence",
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
            f"mode={mode} env_controls=4 triton_autotune=absent "
            "model_overrides=3 torch_deterministic=true total_controls=8"
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
            f"logger.wandb.name=on-{run.pair_id}",
            f"logger.wandb.name=off-{run.pair_id}",
        )
        .replace("/results/on/", "/results/off/")
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
        ({"BATCH_SCRIPT": "/tmp/caller"}, "BATCH_SCRIPT must be unset"),
        ({"RAY_SUB": "/tmp/caller"}, "RAY_SUB must be unset"),
        ({"MOUNTS": "/tmp/a:/tmp/b"}, "MOUNTS must be unset"),
        ({"EXTRA_MOUNTS": "/tmp/a:/tmp/b"}, "EXTRA_MOUNTS must be unset"),
        ({"USE_CUSTOM_VLLM": "0"}, "USE_CUSTOM_VLLM must be unset"),
        ({"INTERACTIVE": "0"}, "INTERACTIVE must be unset"),
        ({"CHECKPOINTING_SAVE_BY": "01:00:00"}, "CHECKPOINTING_SAVE_BY must be unset"),
        ({"WALLTIME": "01:00:00"}, "WALLTIME must be unset"),
        ({"SLURM_QOS": "short"}, "SLURM_QOS must be unset"),
        ({"SLURM_ACCOUNT": "caller"}, "SLURM_ACCOUNT must be unset"),
        ({"DRY_RUN": "0"}, "DRY_RUN must be unset"),
        ({"WANDB_ENTITY": "caller"}, "WANDB_ENTITY must be unset"),
        ({"WANDB_PROJ": "caller"}, "WANDB_PROJ must be unset"),
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
        "launcher source must be exactly DEPLOYMENT_ROOT/runnable/NemoRL"
        in run.process.stderr
    )
    assert run.manifest is None


@pytest.mark.parametrize(
    ("deployment_kind", "message"),
    [
        ("root_relative", "DEPLOYMENT_ROOT must be absolute and canonical"),
        ("root_noncanonical", "DEPLOYMENT_ROOT must already be canonical"),
        ("root_symlink", "DEPLOYMENT_ROOT must be a real, non-symlink directory"),
        ("root_mode", "DEPLOYMENT_ROOT must have mode 500"),
        ("ready_missing", "deployment READY must be a regular, non-symlink file"),
        ("ready_symlink", "deployment READY must be a regular, non-symlink file"),
        ("ready_mode", "deployment READY must have mode 444"),
        ("ready_hash", "deployment READY file SHA-256 mismatch"),
        ("ready_content", "deployment READY content differs"),
        ("nemo_manifest_missing", "deployment runnable manifest must be a regular"),
        ("nemo_manifest_symlink", "deployment runnable manifest must be a regular"),
        ("nemo_manifest_mode", "deployment runnable manifest must have mode 400"),
        ("nemo_manifest_hash", "NemoRL.runnable.sha256 SHA-256 mismatch"),
        ("nemo_manifest_payload_mutated", "content verification failed"),
        ("nemo_manifest_malformed", "content verification failed"),
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
        ("missing", "snapshot is missing a tracked source entry"),
        ("mutated", "snapshot file differs from authenticated source"),
        ("symlink", "snapshot is missing a tracked source entry"),
        ("missing_gym", "snapshot is missing a tracked source entry"),
    ],
)
def test_pair_rejects_snapshot_entrypoint_fallback_or_mutation(
    snapshot_kind: str, message: str
) -> None:
    run = _run_pair(snapshot_kind=snapshot_kind)
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
def test_pair_rejects_post_publication_snapshot_tampering(
    deployment_kind: str, tamper: str, message: str
) -> None:
    run = _run_pair(
        deployment_kind=deployment_kind,
        NANO35_TEST_ARM_TAMPER=tamper,
    )
    assert run.process.returncode != 0
    assert message in run.process.stderr
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
def test_pair_rejects_post_publication_pair_manifest_tampering(
    tamper: str, message: str
) -> None:
    run = _run_pair(NANO35_TEST_PAIR_MANIFEST_TAMPER=tamper)
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
    assert master_config.env["nemo_gym"]["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
    ]
    assert not master_config.env["nemo_gym"]["run_response_cache_enabled"]
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
