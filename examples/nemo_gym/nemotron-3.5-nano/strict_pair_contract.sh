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

# Shared fail-closed helpers for launch_pair.sh and nano35_single_env_pair.sh.
# This file only defines functions; callers retain set -euo pipefail ownership.

strict_pair_error() {
  echo "ERROR: $*" >&2
  return 2
}

# Resolve the only admitted single-environment contracts.  Every derived value
# is later copied into the sealed Pair manifest; callers must forward the
# selected environment explicitly across each clean environment boundary.
strict_pair_select_environment() {
  case "${STRICT_PAIR_ENVIRONMENT:-}" in
    reasoning_gym)
      STRICT_PAIR_CONFIG_RELATIVE="examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml"
      STRICT_PAIR_FIXTURE_RELATIVE="tests/unit/tools/data/reasoning_gym_example.jsonl"
      STRICT_PAIR_EXPECTED_FIXTURE_SHA256="da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"
      STRICT_PAIR_VERIFIER_METRIC="train/reasoning_gym_simple_agent/score/mean"
      STRICT_PAIR_GYM_CONFIG_RELATIVE="resources_servers/reasoning_gym/configs/reasoning_gym.yaml"
      STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE="resources_servers/reasoning_gym/requirements.txt"
      STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE="resources_servers/reasoning_gym/app.py"
      ;;
    citation)
      STRICT_PAIR_CONFIG_RELATIVE="examples/nemo_gym/nemotron-3.5-nano/single_env_citation_sc.yaml"
      STRICT_PAIR_FIXTURE_RELATIVE="tests/unit/tools/data/citation_example.jsonl"
      STRICT_PAIR_EXPECTED_FIXTURE_SHA256="d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc"
      STRICT_PAIR_VERIFIER_METRIC="train/citation_format_simple_agent/reward/mean"
      STRICT_PAIR_GYM_CONFIG_RELATIVE="resources_servers/format_verification/configs/citation_format.yaml"
      STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE="resources_servers/format_verification/requirements.txt"
      STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE="resources_servers/format_verification/app.py"
      ;;
    freeform)
      STRICT_PAIR_CONFIG_RELATIVE="examples/nemo_gym/nemotron-3.5-nano/single_env_freeform_sc.yaml"
      STRICT_PAIR_FIXTURE_RELATIVE="tests/unit/tools/data/freeform_example.jsonl"
      STRICT_PAIR_EXPECTED_FIXTURE_SHA256="8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399"
      STRICT_PAIR_VERIFIER_METRIC="train/freeform_formatting_simple_agent/reward/mean"
      STRICT_PAIR_GYM_CONFIG_RELATIVE="resources_servers/format_verification/configs/freeform_formatting.yaml"
      STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE="resources_servers/format_verification/requirements.txt"
      STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE="resources_servers/format_verification/app.py"
      ;;
    *)
      strict_pair_error \
        "STRICT_PAIR_ENVIRONMENT must be exactly reasoning_gym, citation, or freeform."
      return
      ;;
  esac
}

# Privileged Bash deliberately ignores BASH_ENV, ENV, inherited shell options,
# and exported functions. Rejecting the still-visible environment records makes
# the absence of those startup inputs part of the rendered launch contract too.
# Dynamic-loader safety is provided before entry by the trusted caller, and by
# the exact Slurm export-file boundary for submitted jobs; it cannot be created
# retroactively by this shell function.
strict_pair_reject_startup_environment() {
  local environment_line
  local environment_name

  if [[ "$-" != *p* ]]; then
    strict_pair_error "strict pair scripts must be executed directly through their privileged Bash shebang."
    return
  fi
  while IFS= read -r environment_name; do
    case "${environment_name}" in
      BASH_ENV|ENV|BASHOPTS|SHELLOPTS|BASH_COMPAT|CDPATH|GLOBIGNORE|BASH_XTRACEFD|\
      PYTHON*|GIT_*|LD_*|DYLD_*|BASH_FUNC_*%%)
        strict_pair_error "hostile startup environment variable must be unset: ${environment_name}"
        return
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
}

if ! strict_pair_reject_startup_environment; then
  return 2 2>/dev/null || exit 2
fi

strict_pair_require_lower_digest_value() {
  local name="$1"
  local value="${!name:-}"

  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    strict_pair_error "${name} must be an explicit lowercase SHA-256."
  fi
}

strict_pair_select_bootstrap_sha256sum() {
  local actual
  local output
  local provided_sha256="${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256:-}"

  case "${OSTYPE}" in
    linux*)
      STRICT_PAIR_BOOTSTRAP_SHA256SUM=/usr/bin/sha256sum
      EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256="ffb52ec22da029403b8e2a2ee4e07bc4f111f1ec3c6d8fac2f2b5788891dd825"
      ;;
    darwin*)
      # Hermetic macOS unit-test contract only. Production is Linux/HSG.
      STRICT_PAIR_BOOTSTRAP_SHA256SUM=/sbin/sha256sum
      EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256="881f3812ac7be70d99bf635e5322b63f565af66f502be3b464036fef8f927300"
      ;;
    *)
      strict_pair_error "strict pair bootstrap is supported only on Linux/HSG or the macOS unit-test host."
      return
      ;;
  esac
  if [[ -n "${provided_sha256}" && "${provided_sha256}" != \
        "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" ]]; then
    strict_pair_error "caller-provided bootstrap sha256sum digest differs from the code-fixed OS bootstrap."
    return
  fi
  if [[ -L "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" || \
        ! -f "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" || \
        ! -x "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" || \
        -w "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" ]]; then
    strict_pair_error "bootstrap sha256sum must be a read-only regular system executable."
    return
  fi
  if ! output="$(
    "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- \
      "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}"
  )"; then
    strict_pair_error "bootstrap sha256sum failed while authenticating itself."
    return
  fi
  actual="${output%% *}"
  if [[ "${actual}" != \
        "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" ]]; then
    strict_pair_error "bootstrap sha256sum SHA-256 mismatch."
    return
  fi
  export STRICT_PAIR_BOOTSTRAP_SHA256SUM
}

strict_pair_bootstrap_sha256_file() {
  local path="$1"
  local digest
  local output

  if ! output="$("${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" -- "${path}")"; then
    strict_pair_error "bootstrap sha256sum failed for ${path}."
    return
  fi
  digest="${output%% *}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    strict_pair_error "bootstrap sha256sum returned a malformed digest for ${path}."
    return
  fi
  printf '%s\n' "${digest}"
}

# Authenticate the host toolchain before any contract helper consumes it.
# The NeMo-RL runnable-manifest digest is an explicit launch argument.  Its
# authenticated bytes uniquely bind the runtime-tool manifest; fixed OS
# bootstrap binaries then verify and parse that deployment-bound document.
strict_pair_load_runtime_tools() {
  local actual
  local line
  local nemo_manifest
  local provided_container_python="${STRICT_PAIR_CONTAINER_PYTHON:-}"
  local provided_container_python_sha256="${EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256:-}"
  local provided_container_uv="${STRICT_PAIR_CONTAINER_UV:-}"
  local provided_container_uv_sha256="${EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256:-}"
  local provided_host_python="${STRICT_PAIR_HOST_PYTHON:-}"
  local provided_host_python_sha256="${EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256:-}"
  local provided_manifest="${STRICT_PAIR_RUNTIME_TOOL_MANIFEST:-}"
  local provided_manifest_sha256="${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256:-}"
  local provided_uv_shim="${STRICT_PAIR_UV_SHIM:-}"
  local provided_uv_shim_sha256="${EXPECTED_STRICT_PAIR_UV_SHIM_SHA256:-}"
  local records
  local runnable_record_count=0
  local runnable_record_path
  local runnable_record_sha256
  local scope
  local name
  local path
  local digest

  : "${DEPLOYMENT_ROOT:?DEPLOYMENT_ROOT is required}"
  : "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256:?EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256 is required}"
  strict_pair_require_lower_digest_value EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256

  strict_pair_select_bootstrap_sha256sum
  nemo_manifest="${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256"
  if [[ "${nemo_manifest}" != /* || -L "${nemo_manifest}" || \
        ! -f "${nemo_manifest}" ]]; then
    strict_pair_error "deployment NeMo-RL runnable manifest must be one absolute regular non-symlink file."
    return
  fi
  actual="$(strict_pair_bootstrap_sha256_file "${nemo_manifest}")"
  if [[ "${actual}" != "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" ]]; then
    strict_pair_error "NeMo-RL runnable manifest differs from the admitted positional anchor."
    return
  fi
  STRICT_PAIR_RUNTIME_TOOL_MANIFEST="${DEPLOYMENT_ROOT}/strict_pair_runtime_tools.json"
  while IFS= read -r line; do
    runnable_record_sha256="${line%%  *}"
    runnable_record_path="${line#*  }"
    if [[ "${runnable_record_path}" == "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" ]]; then
      if [[ "${line}" != "${runnable_record_sha256}  ${runnable_record_path}" || \
            ! "${runnable_record_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
        strict_pair_error "runtime-tool runnable-manifest record is malformed."
        return
      fi
      EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256="${runnable_record_sha256}"
      (( runnable_record_count += 1 ))
    fi
  done < "${nemo_manifest}"
  if (( runnable_record_count != 1 )); then
    strict_pair_error "authenticated NeMo-RL runnable manifest must contain exactly one runtime-tool manifest record."
    return
  fi
  if [[ -n "${provided_manifest}" && \
        "${provided_manifest}" != "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" ]]; then
    strict_pair_error "caller-provided runtime-tool manifest path differs from the deployment-bound path."
    return
  fi
  if [[ -n "${provided_manifest_sha256}" && \
        "${provided_manifest_sha256}" != \
        "${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}" ]]; then
    strict_pair_error "caller-provided runtime-tool manifest digest differs from the authenticated runnable manifest."
    return
  fi
  case "${OSTYPE}" in
    linux*)
      STRICT_PAIR_HOST_PYTHON=/cm/local/apps/python312/bin/python3.12
      EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256="36bee55d1d2c90ceda25e65038809d69be3d3e6e82c94e5d9ec3b2ec1ccc9faa"
      ;;
    darwin*)
      STRICT_PAIR_HOST_PYTHON=/usr/bin/python3
      EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256="179301dcb41ea78accc3fa0048a7e6f6710d891945a751a34addd622020c1818"
      ;;
  esac
  if [[ -n "${provided_host_python}" && \
        "${provided_host_python}" != "${STRICT_PAIR_HOST_PYTHON}" ]]; then
    strict_pair_error "caller-provided host Python path differs from the code-fixed OS path."
    return
  fi
  if [[ -n "${provided_host_python_sha256}" && \
        "${provided_host_python_sha256}" != \
        "${EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256}" ]]; then
    strict_pair_error "caller-provided host Python digest differs from the code-fixed OS digest."
    return
  fi
  for path in "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
              "${STRICT_PAIR_HOST_PYTHON}"; do
    if [[ "${path}" != /* || -L "${path}" || ! -f "${path}" || \
          ( "${path}" == "${STRICT_PAIR_HOST_PYTHON}" && \
            ( ! -x "${path}" || -w "${path}" ) ) ]]; then
      strict_pair_error "runtime bootstrap input is not an absolute regular non-symlink file: ${path}"
      return
    fi
  done
  actual="$(strict_pair_bootstrap_sha256_file \
    "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}")"
  if [[ "${actual}" != \
        "${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}" ]]; then
    strict_pair_error "runtime-tool manifest SHA-256 mismatch."
    return
  fi
  actual="$(strict_pair_bootstrap_sha256_file "${STRICT_PAIR_HOST_PYTHON}")"
  if [[ "${actual}" != "${EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256}" ]]; then
    strict_pair_error "host Python SHA-256 mismatch."
    return
  fi

  records="$(
    "${STRICT_PAIR_HOST_PYTHON}" -I -B - \
      "${DEPLOYMENT_ROOT}" \
      "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
      "${STRICT_PAIR_HOST_PYTHON}" \
      "${EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256}" \
      "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" \
      "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

(
    deployment_raw,
    manifest_raw,
    host_python,
    host_python_sha256,
    bootstrap_sha256sum,
    bootstrap_sha256sum_sha256,
) = sys.argv[1:]

DIGEST = re.compile(r"[0-9a-f]{64}")
SAFE_PATH = re.compile(r"/[A-Za-z0-9._/+:-]+")
HOST_KEYS = {
    "awk",
    "bash",
    "cat",
    "chmod",
    "cmp",
    "date",
    "env",
    "find",
    "git",
    "grep",
    "ln",
    "mkdir",
    "mktemp",
    "nvidia_smi",
    "python",
    "readlink",
    "realpath",
    "rm",
    "rsync",
    "sbatch",
    "scancel",
    "scontrol",
    "sha256sum",
    "stat",
    "wc",
}
CONTAINER_KEYS = {"python", "uv", "uv_shim"}
EXPECTED_CONTAINER_TOOLS = {
    "python": {
        "path": (
            "/root/.local/share/uv/python/"
            "cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
        ),
        "sha256": (
            "92ed50fd9dde3654d421d165214a95361"
            "e1889210b3d2063001d6c2e75eef2ab"
        ),
    },
    "uv": {
        "path": "/root/.local/bin/uv",
        "sha256": (
            "b9f74e398b6b15826a4b68b5a83d039"
            "036d47df64013e7faf1a9974ec199c144"
        ),
    },
}
EXPECTED_LINUX_SCHEDULER_TOOLS = {
    "sbatch": {
        "path": "/cm/local/apps/slurm/25.11/bin/sbatch",
        "sha256": "ac1f483625d1005b60e0e650fab381ad55e1a52a42dc0c9bbf625fffcac789fc",
    },
    "scancel": {
        "path": "/cm/local/apps/slurm/25.11/bin/scancel",
        "sha256": "fb2ca904d41c954b993890f91f14bdd8b1ba85566c7006dbb66fd600dd9d6b96",
    },
    "scontrol": {
        "path": "/cm/local/apps/slurm/25.11/bin/scontrol",
        "sha256": "24333d205add15ce6a285ea81b7e55af7b0f35d26451472e6c3a011eba3b3594",
    },
}


def fail(message: str) -> None:
    raise SystemExit(message)


def reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = {}
    for key, value in pairs:
        if key in document:
            fail(f"duplicate JSON member: {key}")
        document[key] = value
    return document


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or SAFE_PATH.fullmatch(value) is None:
        fail(f"{label} must be one absolute shell-safe path")
    return value


def validate_root_owned_ancestry(path: pathlib.Path, label: str) -> None:
    current = path.parent
    while True:
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            fail(f"{label} ancestry must contain only real directories")
        if metadata.st_uid != 0 or metadata.st_gid != 0:
            fail(f"{label} ancestry must be owned by uid0/gid0")
        if metadata.st_mode & 0o022:
            fail(f"{label} ancestry must not be group/world writable")
        if current == current.parent:
            break
        current = current.parent


deployment = pathlib.Path(safe_path(deployment_raw, "DEPLOYMENT_ROOT"))
manifest = pathlib.Path(safe_path(manifest_raw, "runtime-tool manifest"))
if os.path.realpath(deployment) != str(deployment) or deployment.is_symlink():
    fail("DEPLOYMENT_ROOT must be canonical and non-symlink")
if stat.S_IMODE(os.lstat(deployment).st_mode) != 0o500:
    fail("DEPLOYMENT_ROOT must have mode 500")
if os.path.realpath(manifest) != str(manifest) or manifest.is_symlink():
    fail("runtime-tool manifest must be canonical and non-symlink")
if manifest != deployment / "strict_pair_runtime_tools.json":
    fail("runtime-tool manifest must use the canonical deployment filename")
try:
    manifest.relative_to(deployment)
except ValueError:
    fail("runtime-tool manifest must be below DEPLOYMENT_ROOT")
if not stat.S_ISREG(os.lstat(manifest).st_mode):
    fail("runtime-tool manifest must be regular")
if stat.S_IMODE(os.lstat(manifest).st_mode) != 0o400:
    fail("runtime-tool manifest must have mode 400")

try:
    document = json.loads(
        manifest.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_members,
    )
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    fail(f"invalid runtime-tool manifest: {error}")
if not isinstance(document, dict) or set(document) != {"schema", "host", "container"}:
    fail("runtime-tool manifest must have exact schema/host/container keys")
if document["schema"] != "nemo-rl-strict-runtime-tools-v2":
    fail("runtime-tool manifest schema mismatch")
host = document["host"]
container = document["container"]
if not isinstance(host, dict) or set(host) != HOST_KEYS:
    fail("runtime-tool manifest host tool inventory mismatch")
if not isinstance(container, dict) or set(container) != CONTAINER_KEYS:
    fail("runtime-tool manifest container tool inventory mismatch")

for name in sorted(HOST_KEYS):
    record = host[name]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        fail(f"host tool {name} record must contain only path and sha256")
    path = pathlib.Path(safe_path(record["path"], f"host tool {name}"))
    expected = record["sha256"]
    if not isinstance(expected, str) or DIGEST.fullmatch(expected) is None:
        fail(f"host tool {name} has invalid SHA-256")
    metadata = os.lstat(path)
    if os.path.realpath(path) != str(path) or stat.S_ISLNK(metadata.st_mode):
        fail(f"host tool {name} must be canonical and non-symlink")
    if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
        fail(f"host tool {name} must be a regular executable")
    if name == "nvidia_smi":
        if sys.platform == "darwin":
            try:
                path.relative_to(deployment)
            except ValueError:
                fail("Darwin test nvidia-smi must be a deployment-local shim")
        elif path != pathlib.Path("/usr/bin/nvidia-smi"):
            fail("host nvidia-smi must use the fixed /usr/bin/nvidia-smi path")
    darwin_test_scheduler_tool = False
    if name in {"nvidia_smi", "sbatch", "scancel", "scontrol"} and sys.platform == "darwin":
        try:
            path.relative_to(deployment)
        except ValueError:
            pass
        else:
            darwin_test_scheduler_tool = stat.S_IMODE(metadata.st_mode) == 0o500
    if (
        metadata.st_uid != 0 or metadata.st_gid != 0
    ) and not darwin_test_scheduler_tool:
        fail(f"host tool {name} must be owned by uid0/gid0")
    if metadata.st_mode & 0o022:
        fail(f"host tool {name} must not be group/world writable")
    if not darwin_test_scheduler_tool:
        validate_root_owned_ancestry(path, f"host tool {name}")
    if sha256(path) != expected:
        fail(f"host tool {name} SHA-256 mismatch")
    if sys.platform != "darwin" and name in EXPECTED_LINUX_SCHEDULER_TOOLS:
        if record != EXPECTED_LINUX_SCHEDULER_TOOLS[name]:
            fail(f"host scheduler tool {name} differs from the pinned HSG 25.11 inventory")
    print("host", name, path, expected, sep="\t")

for name in sorted(CONTAINER_KEYS):
    record = container[name]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        fail(f"container tool {name} record must contain only path and sha256")
    path = safe_path(record["path"], f"container tool {name}")
    expected = record["sha256"]
    if not isinstance(expected, str) or DIGEST.fullmatch(expected) is None:
        fail(f"container tool {name} has invalid SHA-256")
    if name in EXPECTED_CONTAINER_TOOLS and record != EXPECTED_CONTAINER_TOOLS[name]:
        fail(f"container tool {name} differs from the pinned image inventory")
    print("container", name, path, expected, sep="\t")

expected_pairs = {
    ("host", "python"): (host_python, host_python_sha256),
    ("host", "sha256sum"): (
        bootstrap_sha256sum,
        bootstrap_sha256sum_sha256,
    ),
}
for (scope, name), expected in expected_pairs.items():
    record = document[scope][name]
    if (record["path"], record["sha256"]) != expected:
        fail(f"flattened {scope}.{name} anchor differs from runtime-tool manifest")

uv_shim = container["uv_shim"]["path"]
uv_shim_sha256 = container["uv_shim"]["sha256"]
shim = pathlib.Path(uv_shim)
if os.path.realpath(shim) != str(shim) or shim.is_symlink():
    fail("STRICT_PAIR_UV_SHIM must be canonical and non-symlink")
try:
    shim.relative_to(deployment)
except ValueError:
    fail("STRICT_PAIR_UV_SHIM must be below DEPLOYMENT_ROOT")
metadata = os.lstat(shim)
if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o500:
    fail("STRICT_PAIR_UV_SHIM must be a regular mode-500 file")
if sha256(shim) != uv_shim_sha256:
    fail("STRICT_PAIR_UV_SHIM SHA-256 mismatch")
PY
  )" || {
    strict_pair_error "runtime-tool manifest validation failed."
    return
  }

  while IFS=$'\t' read -r scope name path digest; do
    case "${scope}:${name}" in
      host:awk) STRICT_PAIR_TOOL_AWK="${path}" ;;
      host:bash) STRICT_PAIR_TOOL_BASH="${path}" ;;
      host:cat) STRICT_PAIR_TOOL_CAT="${path}" ;;
      host:chmod) STRICT_PAIR_TOOL_CHMOD="${path}" ;;
      host:cmp) STRICT_PAIR_TOOL_CMP="${path}" ;;
      host:date) STRICT_PAIR_TOOL_DATE="${path}" ;;
      host:env)
        STRICT_PAIR_TOOL_ENV="${path}"
        STRICT_PAIR_TOOL_ENV_SHA256="${digest}"
        ;;
      host:find) STRICT_PAIR_TOOL_FIND="${path}" ;;
      host:git) STRICT_PAIR_TOOL_GIT="${path}" ;;
      host:grep) STRICT_PAIR_TOOL_GREP="${path}" ;;
      host:ln) STRICT_PAIR_TOOL_LN="${path}" ;;
      host:mkdir) STRICT_PAIR_TOOL_MKDIR="${path}" ;;
      host:mktemp) STRICT_PAIR_TOOL_MKTEMP="${path}" ;;
      host:nvidia_smi) STRICT_PAIR_TOOL_NVIDIA_SMI="${path}" ;;
      host:python) STRICT_PAIR_TOOL_PYTHON="${path}" ;;
      host:readlink) STRICT_PAIR_TOOL_READLINK="${path}" ;;
      host:realpath) STRICT_PAIR_TOOL_REALPATH="${path}" ;;
      host:rm) STRICT_PAIR_TOOL_RM="${path}" ;;
      host:rsync) STRICT_PAIR_TOOL_RSYNC="${path}" ;;
      host:sbatch)
        STRICT_PAIR_TOOL_SBATCH="${path}"
        STRICT_PAIR_TOOL_SBATCH_SHA256="${digest}"
        ;;
      host:scancel)
        STRICT_PAIR_TOOL_SCANCEL="${path}"
        STRICT_PAIR_TOOL_SCANCEL_SHA256="${digest}"
        ;;
      host:scontrol)
        STRICT_PAIR_TOOL_SCONTROL="${path}"
        STRICT_PAIR_TOOL_SCONTROL_SHA256="${digest}"
        ;;
      host:sha256sum) STRICT_PAIR_TOOL_SHA256SUM="${path}" ;;
      host:stat) STRICT_PAIR_TOOL_STAT="${path}" ;;
      host:wc) STRICT_PAIR_TOOL_WC="${path}" ;;
      container:python)
        STRICT_PAIR_CONTAINER_PYTHON="${path}"
        EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256="${digest}"
        ;;
      container:uv)
        STRICT_PAIR_CONTAINER_UV="${path}"
        EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256="${digest}"
        ;;
      container:uv_shim)
        STRICT_PAIR_UV_SHIM="${path}"
        EXPECTED_STRICT_PAIR_UV_SHIM_SHA256="${digest}"
        ;;
      *)
        strict_pair_error "unexpected runtime tool after validation: ${scope}.${name}"
        return
        ;;
    esac
  done <<< "${records}"
  for name in AWK BASH CAT CHMOD CMP DATE ENV FIND GIT GREP LN MKDIR \
              MKTEMP NVIDIA_SMI PYTHON READLINK REALPATH RM RSYNC SBATCH SCANCEL \
              SCONTROL SHA256SUM STAT WC; do
    path="STRICT_PAIR_TOOL_${name}"
    if [[ -z "${!path:-}" ]]; then
      strict_pair_error "validated runtime host tool is missing: ${name}"
      return
    fi
  done
  if [[ -n "${provided_container_python}" && \
        "${provided_container_python}" != "${STRICT_PAIR_CONTAINER_PYTHON}" ]] || \
     [[ -n "${provided_container_python_sha256}" && \
        "${provided_container_python_sha256}" != \
        "${EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256}" ]] || \
     [[ -n "${provided_container_uv}" && \
        "${provided_container_uv}" != "${STRICT_PAIR_CONTAINER_UV}" ]] || \
     [[ -n "${provided_container_uv_sha256}" && \
        "${provided_container_uv_sha256}" != \
        "${EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256}" ]] || \
     [[ -n "${provided_uv_shim}" && \
        "${provided_uv_shim}" != "${STRICT_PAIR_UV_SHIM}" ]] || \
     [[ -n "${provided_uv_shim_sha256}" && \
        "${provided_uv_shim_sha256}" != \
        "${EXPECTED_STRICT_PAIR_UV_SHIM_SHA256}" ]]; then
    strict_pair_error "caller-provided container runtime-tool anchors differ from the authenticated runtime-tool manifest."
    return
  fi
  if [[ "$("${STRICT_PAIR_TOOL_REALPATH}" -- "${BASH}")" != \
        "${STRICT_PAIR_TOOL_BASH}" ]]; then
    strict_pair_error "active privileged Bash differs from the authenticated host Bash."
    return
  fi
  if [[ "${PATH}" != "/usr/bin:/bin:/usr/sbin:/sbin" ]]; then
    strict_pair_error "strict host PATH changed after startup sanitization."
    return
  fi
  STRICT_PAIR_RUNTIME_TOOLS_LOADED=1
}

strict_pair_git() {
  "${STRICT_PAIR_TOOL_ENV}" -i \
    HOME=/nonexistent \
    LC_ALL=C \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_OPTIONAL_LOCKS=0 \
    GIT_TERMINAL_PROMPT=0 \
    "${STRICT_PAIR_TOOL_GIT}" \
    -c core.fsmonitor=false \
    -c core.hooksPath=/dev/null \
    -c diff.external= \
    "$@"
}

# Exact user payload admitted across the Slurm submission boundary. Slurm's
# own scheduler/SPANK variables are validated separately by the job wrapper;
# ambient caller variables are never merged into this NUL-delimited file.
STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES=(
  BASE_LOG_DIR
  BATCH_SCRIPT
  COLOCATED_GENERATION
  COMMAND
  CONTAINER
  CPUS_PER_WORKER
  DEDICATED_RAY_HEAD
  DEPLOYMENT_ROOT
  EXPECTED_BRIDGE_HEAD
  EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256
  EXPECTED_BRIDGE_TREE
  EXPECTED_DEPLOYMENT_READY
  EXPECTED_DEPLOYMENT_READY_FILE_SHA256
  EXPECTED_GYM_GITLINK_COMMIT
  EXPECTED_GYM_TREE
  EXPECTED_MCORE_HEAD
  EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256
  EXPECTED_MCORE_TREE
  EXPECTED_NEMO_HEAD
  EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_COUNT
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SCHEMA
  EXPECTED_SHARED_PREFIX_DETERMINISM_ATTESTATION_SHA256
  EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256
  EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256
  EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256
  EXPECTED_STRICT_PAIR_CONTAINER_SHA256
  EXPECTED_STRICT_PAIR_CONTAINER_UV_SHA256
  EXPECTED_STRICT_PAIR_FIXTURE_SHA256
  EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256
  EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256
  EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256
  EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256
  EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256
  EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256
  EXPECTED_STRICT_PAIR_UV_SHIM_SHA256
  EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256
  EXP_NAME
  GPUS_PER_NODE
  HF_DATASETS_CACHE
  HF_HOME
  HF_HUB_CACHE
  HF_TOKEN
  MODEL_PATH
  MOUNTS
  NEMO_SKILLS_SANDBOX_PORT
  NUM_EXTERNAL_SERVICE_NODES
  NUM_GEN_NODES
  NUM_GYM_NODES
  NUM_TRAIN_NODES
  PAIR_ID
  PERSISTENT_CACHE
  RAY_LOG_SYNC_FREQUENCY
  RAY_SUB
  RESULTS_DIR
  SANDBOX_COMMAND
  SANDBOX_CONTAINER
  SEGMENT_SIZE
  SETUP_COMMAND
  STRICT_PAIR_CONTAINER_PYTHON
  STRICT_PAIR_CONTAINER_UV
  STRICT_PAIR_ENVIRONMENT
  STRICT_PAIR_HOST_PYTHON
  STRICT_PAIR_JOB_WRAPPER
  STRICT_PAIR_LAUNCH_MODE
  STRICT_PAIR_RUNTIME_TOOL_MANIFEST
  STRICT_PAIR_SHARED_PREFIX_MODE
  STRICT_PAIR_UV_SHIM
  STRICT_PREBUILT_SNAPSHOT_DIR
  TRAIN_PATH
  VAL_PATH
  WANDB_API_KEY
  WANDB_ENTITY
  WANDB_NAME
  WANDB_PROJ
  WANDB_RESUME
  WANDB_RUN_GROUP
  WANDB_RUN_ID
)
STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES_CSV="$({
  IFS=,
  printf '%s' "${STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES[*]}"
})"

strict_pair_render_slurm_export_payload() {
  local output="$1"
  local name
  local value

  if [[ -L "${output}" || ! -f "${output}" ]]; then
    strict_pair_error "Slurm export candidate must be an exclusive regular file."
    return
  fi
  : > "${output}"
  for name in "${STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES[@]}"; do
    if [[ -z "${!name+x}" ]]; then
      strict_pair_error "strict Slurm export value is unset: ${name}"
      return
    fi
    value="${!name}"
    if ! printf '%s=%s\0' "${name}" "${value}" >> "${output}"; then
      strict_pair_error "failed to render strict Slurm export value: ${name}"
      return
    fi
  done
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${output}"
}

strict_pair_publish_or_verify_slurm_export() {
  local target="$1"
  local expected_sha256="${2:-}"
  local parent="${target%/*}"
  local candidate
  local candidate_sha256
  local target_sha256

  strict_pair_require_canonical_dir "${parent}" "strict Slurm export parent"
  strict_pair_require_mode "${parent}" "700" "strict Slurm export parent"
  if ! candidate="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" "${parent}/.export.XXXXXX"
  )"; then
    strict_pair_error "failed to create an exclusive strict Slurm export candidate."
    return
  fi
  if ! strict_pair_render_slurm_export_payload "${candidate}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    return 2
  fi
  if ! candidate_sha256="$(strict_pair_sha256_file "${candidate}")"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    return 2
  fi

  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ -z "${expected_sha256}" ]]; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      strict_pair_error "strict Slurm export target already exists during prepare: ${target}"
      return
    fi
    if ! strict_pair_require_canonical_file \
        "${target}" "strict Slurm export file" || \
       ! strict_pair_require_mode \
        "${target}" "400" "strict Slurm export file"; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      return 2
    fi
    if ! target_sha256="$(strict_pair_sha256_file "${target}")"; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      return 2
    fi
    if [[ "${target_sha256}" != "${expected_sha256}" || \
          "${candidate_sha256}" != "${expected_sha256}" ]] || \
       ! "${STRICT_PAIR_TOOL_CMP}" -s -- "${candidate}" "${target}"; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      strict_pair_error "strict Slurm export bytes differ from the parent-bound arm payload."
      return
    fi
  else
    if [[ -n "${expected_sha256}" ]]; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      strict_pair_error "parent-bound strict Slurm export file is missing: ${target}"
      return
    fi
    if ! "${STRICT_PAIR_TOOL_LN}" -- "${candidate}" "${target}"; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
      strict_pair_error "failed to atomically publish strict Slurm export file."
      return
    fi
    target_sha256="${candidate_sha256}"
  fi
  if ! "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"; then
    strict_pair_error "failed to remove strict Slurm export candidate containing secrets."
    return
  fi
  STRICT_PAIR_ACTIVE_SLURM_EXPORT_SHA256="${target_sha256}"
}

strict_pair_verify_slurm_export_before_sbatch() {
  local target="$1"
  local expected_sha256="$2"
  local actual_sha256

  if [[ ! "${expected_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    strict_pair_error "parent-bound strict Slurm export SHA-256 is malformed."
    return
  fi
  strict_pair_require_canonical_file "${target}" "strict Slurm export file"
  strict_pair_require_mode "${target}" "400" "strict Slurm export file"
  actual_sha256="$(strict_pair_sha256_file "${target}")"
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    strict_pair_error "strict Slurm export file changed immediately before sbatch."
  fi
}

strict_pair_sha256_file() {
  local path="$1"
  local digest
  local output

  if ! output="$("${STRICT_PAIR_TOOL_SHA256SUM}" -- "${path}")"; then
    strict_pair_error "authenticated sha256sum failed for ${path}."
    return
  fi
  digest="${output%% *}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    strict_pair_error "authenticated sha256sum returned a malformed digest for ${path}."
    return
  fi
  printf '%s\n' "${digest}"
}

strict_pair_sha256_text() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "$1" <<'PY'
import hashlib
import sys

print(hashlib.sha256(sys.argv[1].encode("ascii")).hexdigest())
PY
}

strict_pair_acceptance_policy_sha256() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - <<'PY'
import hashlib
import json

policy = {
    "hash_domain": "sha256-domain-nul-canonical-ascii-json-no-lf-v1",
    "live_reward_noninferiority": {
        "burn_in_steps": 10,
        "evaluated_steps": {
            "count": 90,
            "first": 11,
            "last": 100,
            "require_complete_paired_steps": True,
        },
        "margin": 0.1,
        "metric": "train/raw_environment_reward",
        "paired_step_bootstrap": {
            "confidence_level": 0.95,
            "interval": "percentile",
            "lower_quantile": 0.025,
            "resamples": 10000,
            "sampling_unit": "paired-step",
            "seed": 20260828,
            "statistic": "mean-on-minus-off",
            "upper_quantile": 0.975,
        },
        "primary_gate": (
            "lower-confidence-bound-strictly-greater-than-negative-margin"
        ),
        "tail": {
            "gate": "on-mean-greater-than-or-equal-to-off-mean-minus-margin",
            "margin": 0.1,
            "statistic": "mean-on-minus-mean-off",
            "steps": {"count": 25, "first": 76, "last": 100},
        },
    },
    "optimizer_update_witness": {
        "false_value": "fail",
        "json_type": "boolean",
        "ledger_field": "update_successful",
        "missing_or_non_boolean": "unverifiable",
        "required_steps": {"count": 100, "first": 1, "last": 100},
        "required_value": True,
        "wandb_metric": "train/update_successful",
    },
    "schema": "nemo-rl-strict-live-learning-acceptance-policy-v1",
}
payload = json.dumps(
    policy,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
print(
    hashlib.sha256(
        b"nemo-rl-strict-v2\0live-learning-acceptance-policy\0" + payload
    ).hexdigest()
)
PY
}

strict_pair_file_mode() {
  local path="$1"
  local mode

  if mode="$("${STRICT_PAIR_TOOL_STAT}" -c '%a' -- "${path}" 2>/dev/null)"; then
    :
  else
    mode="$("${STRICT_PAIR_TOOL_STAT}" -f '%Lp' -- "${path}")"
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

strict_pair_require_object_id() {
  local name="$1"
  local value="${!name:-}"

  if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    strict_pair_error "${name} must be an explicit lowercase 40-hex Git object ID."
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
  strict_pair_require_object_id EXPECTED_GYM_TREE
  strict_pair_require_commit EXPECTED_BRIDGE_HEAD
  strict_pair_require_object_id EXPECTED_BRIDGE_TREE
  strict_pair_require_commit EXPECTED_MCORE_HEAD
  strict_pair_require_object_id EXPECTED_MCORE_TREE
  if [[ "${EXPECTED_NEMO_HEAD}" == "d7b49a459f08670b6534a56deb99e432f576028a" ]]; then
    strict_pair_error "legacy NeMo-RL deployment d7b49a459f08670b6534a56deb99e432f576028a is forbidden for the strict live pair."
  fi

  STRICT_PAIR_NEMO_HEAD="$(strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" rev-parse HEAD)"
  STRICT_PAIR_NEMO_TREE="$(strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" rev-parse 'HEAD^{tree}')"
  if [[ "${STRICT_PAIR_NEMO_HEAD}" != "${EXPECTED_NEMO_HEAD}" ]]; then
    strict_pair_error "deployed NeMo-RL HEAD differs from EXPECTED_NEMO_HEAD."
  fi

  gym_index_record="$(
    strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --stage -- "${gym_relative}"
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
  if [[ "$(strict_pair_git -C "${STRICT_PAIR_GYM_ROOT}" rev-parse HEAD)" != \
        "${STRICT_PAIR_GYM_GITLINK_COMMIT}" ]]; then
    strict_pair_error "deployed Reasoning Gym HEAD differs from its authenticated gitlink."
  fi
  STRICT_PAIR_GYM_TREE="$(strict_pair_git -C "${STRICT_PAIR_GYM_ROOT}" rev-parse 'HEAD^{tree}')"
  if [[ "${STRICT_PAIR_GYM_TREE}" != "${EXPECTED_GYM_TREE}" ]]; then
    strict_pair_error "deployed Reasoning Gym tree differs from EXPECTED_GYM_TREE."
  fi

  STRICT_PAIR_BRIDGE_ROOT="${DEPLOYMENT_ROOT}/runnable/Megatron-Bridge"
  STRICT_PAIR_MCORE_ROOT="${DEPLOYMENT_ROOT}/runnable/Megatron-LM"
  strict_pair_require_canonical_dir \
    "${STRICT_PAIR_BRIDGE_ROOT}" "deployed Megatron-Bridge root"
  strict_pair_require_mode \
    "${STRICT_PAIR_BRIDGE_ROOT}" "500" "deployed Megatron-Bridge root"
  strict_pair_require_canonical_dir \
    "${STRICT_PAIR_MCORE_ROOT}" "deployed Megatron-LM root"
  strict_pair_require_mode \
    "${STRICT_PAIR_MCORE_ROOT}" "500" "deployed Megatron-LM root"
  STRICT_PAIR_BRIDGE_HEAD="$(
    strict_pair_git -C "${STRICT_PAIR_BRIDGE_ROOT}" rev-parse HEAD
  )"
  STRICT_PAIR_BRIDGE_TREE="$(
    strict_pair_git -C "${STRICT_PAIR_BRIDGE_ROOT}" rev-parse 'HEAD^{tree}'
  )"
  STRICT_PAIR_MCORE_HEAD="$(
    strict_pair_git -C "${STRICT_PAIR_MCORE_ROOT}" rev-parse HEAD
  )"
  STRICT_PAIR_MCORE_TREE="$(
    strict_pair_git -C "${STRICT_PAIR_MCORE_ROOT}" rev-parse 'HEAD^{tree}'
  )"
  if [[ "${STRICT_PAIR_BRIDGE_HEAD}" != "${EXPECTED_BRIDGE_HEAD}" || \
        "${STRICT_PAIR_BRIDGE_TREE}" != "${EXPECTED_BRIDGE_TREE}" ]]; then
    strict_pair_error "deployed Megatron-Bridge Git identity differs from its OOB anchors."
  fi
  if [[ "${STRICT_PAIR_MCORE_HEAD}" != "${EXPECTED_MCORE_HEAD}" || \
        "${STRICT_PAIR_MCORE_TREE}" != "${EXPECTED_MCORE_TREE}" ]]; then
    strict_pair_error "deployed Megatron-LM Git identity differs from its OOB anchors."
  fi
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
  canonical="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${path}")"
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
  canonical="$("${STRICT_PAIR_TOOL_REALPATH}" -- "${path}")"
  if [[ "${canonical}" != "${path}" ]]; then
    strict_pair_error "${label} must already be canonical; resolved ${path} to ${canonical}."
  fi
}

strict_pair_require_empty_private_dir() {
  local path="$1"
  local label="$2"

  strict_pair_require_canonical_dir "${path}" "${label}"
  strict_pair_require_mode "${path}" "700" "${label}"
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "${path}" "${EUID}" "${label}" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_uid = int(sys.argv[2])
label = sys.argv[3]
metadata = os.lstat(path)
if (
    not path.is_absolute()
    or os.path.realpath(path) != str(path)
    or stat.S_ISLNK(metadata.st_mode)
    or not stat.S_ISDIR(metadata.st_mode)
    or stat.S_IMODE(metadata.st_mode) != 0o700
    or metadata.st_uid != expected_uid
):
    raise SystemExit(f"{label} is not one canonical private directory")
with os.scandir(path) as entries:
    if next(entries, None) is not None:
        raise SystemExit(f"{label} must be empty")
PY
}

strict_pair_model_tree_sha256_v1() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "$1" <<'PY'
import hashlib
import os
import stat
import sys

root = os.path.abspath(sys.argv[1])
if os.path.realpath(root) != root:
    raise SystemExit("model root is noncanonical")
tree = hashlib.sha256()


def raise_walk_error(error: OSError) -> None:
    raise error


for directory, directory_names, file_names in os.walk(
    root, topdown=True, onerror=raise_walk_error, followlinks=False
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
    if ! (cd -- "${DEPLOYMENT_ROOT}" && \
          "${STRICT_PAIR_TOOL_SHA256SUM}" --check --strict --quiet -- "${manifest_name}"); then
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
  if ! "${STRICT_PAIR_TOOL_GREP}" -Fqxh \
    "${STRICT_PAIR_JOB_WRAPPER_SHA256}  ${STRICT_PAIR_JOB_WRAPPER}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" \
    "${DEPLOYMENT_ROOT}/Megatron-Bridge.runnable.sha256" \
    "${DEPLOYMENT_ROOT}/Megatron-LM.runnable.sha256" >/dev/null; then
    strict_pair_error "STRICT_PAIR_JOB_WRAPPER is not authenticated by a runnable manifest: ${job_wrapper_relative}"
  fi
  if ! "${STRICT_PAIR_TOOL_GREP}" -Fqxh \
    "${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}  ${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" >/dev/null; then
    strict_pair_error "runtime-tool manifest is not authenticated by NemoRL.runnable.sha256."
  fi
  if ! "${STRICT_PAIR_TOOL_GREP}" -Fqxh \
    "${EXPECTED_STRICT_PAIR_UV_SHIM_SHA256}  ${STRICT_PAIR_UV_SHIM}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" >/dev/null; then
    strict_pair_error "STRICT_PAIR_UV_SHIM is not authenticated by NemoRL.runnable.sha256."
  fi
}

strict_pair_prepare_contract() {
  local project_root="$1"
  local recipe_dir="$2"
  local source_path
  local source_rel

  strict_pair_select_environment
  STRICT_PAIR_PROJECT_ROOT="${project_root}"
  STRICT_PAIR_RECIPE_DIR="${recipe_dir}"
  STRICT_PAIR_ACCEPTANCE_POLICY_SHA256="$(
    strict_pair_acceptance_policy_sha256
  )"
  strict_pair_require_digest STRICT_PAIR_ACCEPTANCE_POLICY_SHA256
  TRAIN_PATH="${STRICT_PAIR_PROJECT_ROOT}/${STRICT_PAIR_FIXTURE_RELATIVE}"
  strict_pair_require_canonical_dir "${STRICT_PAIR_PROJECT_ROOT}" "NeMo-RL source root"
  strict_pair_require_canonical_dir "${RESULTS_DIR}" "RESULTS_DIR"
  strict_pair_require_mode "${RESULTS_DIR}" "700" "RESULTS_DIR"
  strict_pair_require_canonical_dir "${PERSISTENT_CACHE}" "PERSISTENT_CACHE"
  strict_pair_require_canonical_dir "${HF_HOME}" "HF_HOME"
  strict_pair_require_canonical_dir "${MODEL_PATH}" "MODEL_PATH"
  strict_pair_require_canonical_file "${CONTAINER}" "CONTAINER"
  strict_pair_require_canonical_file "${SANDBOX_CONTAINER}" "SANDBOX_CONTAINER"
  strict_pair_require_canonical_file "${TRAIN_PATH}" "TRAIN_PATH"

  STRICT_PAIR_FIXTURE_SHA256="$(strict_pair_sha256_file "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_SHA256}" != "${STRICT_PAIR_EXPECTED_FIXTURE_SHA256}" ]]; then
    strict_pair_error "TRAIN_PATH SHA-256 mismatch for ${STRICT_PAIR_ENVIRONMENT}: expected ${STRICT_PAIR_EXPECTED_FIXTURE_SHA256}, got ${STRICT_PAIR_FIXTURE_SHA256}."
  fi
  STRICT_PAIR_FIXTURE_ROWS="$("${STRICT_PAIR_TOOL_AWK}" 'END { print NR }' "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_ROWS}" != "5" ]]; then
    strict_pair_error "authenticated TRAIN_PATH must contain exactly 5 rows; got ${STRICT_PAIR_FIXTURE_ROWS}."
  fi

  strict_pair_verify_deployment
  strict_pair_validate_slurm_conf

  STRICT_PAIR_DEPLOYED_NEMO_ROOT="${DEPLOYMENT_ROOT}/runnable/NemoRL"
  strict_pair_require_canonical_dir "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "deployed NeMo-RL root"
  strict_pair_require_mode "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "500" "deployed NeMo-RL root"
  if [[ "${STRICT_PAIR_PROJECT_ROOT}" != "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" ]]; then
    strict_pair_error "launcher source must be exactly DEPLOYMENT_ROOT/runnable/NemoRL."
  fi
  if ! strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" diff \
      --no-ext-diff --no-textconv --cached --quiet HEAD --; then
    strict_pair_error "deployed NeMo-RL index differs from HEAD."
  fi
  if [[ -n "$(strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --others --exclude-standard)" ]]; then
    strict_pair_error "deployed NeMo-RL source contains untracked files."
  fi
  if [[ -n "$(strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files --others --ignored --exclude-standard)" ]]; then
    strict_pair_error "deployed NeMo-RL source contains ignored files."
  fi
  strict_pair_load_source_identity
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${STRICT_PAIR_PROJECT_ROOT}" \
    "${DEPLOYMENT_ROOT}/NemoRL.runnable.sha256" \
    "${STRICT_PAIR_TOOL_GIT}" \
    "${STRICT_PAIR_BRIDGE_ROOT}" \
    "${DEPLOYMENT_ROOT}/Megatron-Bridge.runnable.sha256" \
    "${STRICT_PAIR_MCORE_ROOT}" \
    "${DEPLOYMENT_ROOT}/Megatron-LM.runnable.sha256" <<'PY'
import hashlib
import os
import pathlib
import stat
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
git = sys.argv[3]
bridge_root = pathlib.Path(sys.argv[4])
bridge_manifest_path = pathlib.Path(sys.argv[5])
mcore_root = pathlib.Path(sys.argv[6])
mcore_manifest_path = pathlib.Path(sys.argv[7])
git_env = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
}
git_prefix = [
    git,
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "diff.external=",
]
def load_manifest(path: pathlib.Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, raw_path = line.partition("  ")
        if not separator or raw_path in result:
            raise SystemExit(f"malformed or duplicate runnable manifest entry: {path}")
        result[raw_path] = digest
    return result


def verify_repo(repo: pathlib.Path, manifest: dict[str, str]) -> None:
    cached = subprocess.run(
        git_prefix
        + [
            "-C",
            str(repo),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
        ],
        check=False,
        env=git_env,
    )
    if cached.returncode != 0:
        raise SystemExit(f"deployed repository index differs from HEAD: {repo}")
    worktree = subprocess.run(
        git_prefix
        + [
            "-C",
            str(repo),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=none",
            "--quiet",
            "HEAD",
            "--",
        ],
        check=False,
        env=git_env,
    )
    if worktree.returncode != 0:
        raise SystemExit(f"deployed repository worktree differs from HEAD: {repo}")
    untracked = subprocess.check_output(
        git_prefix
        + ["-C", str(repo), "ls-files", "--others", "--exclude-standard", "-z"],
        env=git_env,
        text=False,
    )
    if untracked:
        raise SystemExit(f"deployed repository contains untracked files: {repo}")
    ignored = subprocess.check_output(
        [
            *git_prefix,
            "-C",
            str(repo),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ],
        env=git_env,
        text=False,
    )
    if ignored:
        raise SystemExit(f"deployed repository contains ignored files: {repo}")
    entries = subprocess.check_output(
        git_prefix + ["-C", str(repo), "ls-files", "--stage", "-z"],
        env=git_env,
        text=False,
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
                git_prefix + ["-C", str(path), "rev-parse", "HEAD"],
                env=git_env,
                text=True,
            ).strip()
            if actual_head != object_id_raw.decode("ascii"):
                raise SystemExit(f"gitlink HEAD differs from authenticated index: {path}")
            verify_repo(path, manifest)
        elif mode_raw == b"120000":
            if not stat.S_ISLNK(metadata.st_mode):
                raise SystemExit(f"tracked symlink changed type: {path}")
            actual_object = subprocess.check_output(
                git_prefix
                + ["-C", str(repo), "hash-object", "--stdin", "--no-filters"],
                env=git_env,
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


verify_repo(root, load_manifest(manifest_path))
verify_repo(bridge_root, load_manifest(bridge_manifest_path))
verify_repo(mcore_root, load_manifest(mcore_manifest_path))
PY

  STRICT_PAIR_MODEL_TREE_SHA256="$(strict_pair_model_tree_sha256_v1 "${MODEL_PATH}")"
  STRICT_PAIR_CONTAINER_SHA256="$(strict_pair_sha256_file "${CONTAINER}")"
  STRICT_PAIR_SANDBOX_CONTAINER_SHA256="$(strict_pair_sha256_file "${SANDBOX_CONTAINER}")"
  for source_rel in \
    examples/run_grpo_single_controller.py \
    nemo_rl/utils/strict_model_transport.py \
    nemo_rl/models/generation/vllm/vllm_worker_async.py \
    nemo_rl/experience/rollout_manager.py \
    examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh \
    examples/nemo_gym/nemotron-3.5-nano/nano35_single_env_pair.sh \
    examples/nemo_gym/nemotron-3.5-nano/launch_pair.sh \
    examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh \
    examples/nemo_gym/nemotron-3.5-nano/single_env_reasoning_gym_sc.yaml \
    examples/nemo_gym/nemotron-3.5-nano/single_env_citation_sc.yaml \
    examples/nemo_gym/nemotron-3.5-nano/single_env_freeform_sc.yaml \
    tests/unit/tools/data/reasoning_gym_example.jsonl \
    tests/unit/tools/data/citation_example.jsonl \
    tests/unit/tools/data/freeform_example.jsonl; do
    source_path="${STRICT_PAIR_PROJECT_ROOT}/${source_rel}"
    if ! strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" ls-files \
        --error-unmatch -- "${source_rel}" >/dev/null 2>&1; then
      strict_pair_error "strict-pair source must be present in the git index: ${source_rel}"
    fi
    strict_pair_require_canonical_file "${source_path}" "strict-pair source"
    if ! strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" diff \
        --no-ext-diff --no-textconv --quiet HEAD -- "${source_rel}"; then
      strict_pair_error "strict-pair source differs from authenticated HEAD: ${source_rel}"
    fi
  done

  STRICT_PAIR_CONFIG_PATH="${project_root}/${STRICT_PAIR_CONFIG_RELATIVE}"
  strict_pair_require_canonical_file "${STRICT_PAIR_CONFIG_PATH}" "selected environment config"
  STRICT_PAIR_CONFIG_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_CONFIG_PATH}")"
  STRICT_PAIR_GYM_CONFIG_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_CONFIG_RELATIVE}"
  STRICT_PAIR_GYM_REQUIREMENTS_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE}"
  STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE}"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_CONFIG_PATH}" "selected Gym resource config"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_REQUIREMENTS_PATH}" "selected Gym requirements"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH}" "selected Gym verifier source"
  STRICT_PAIR_GYM_CONFIG_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_CONFIG_PATH}")"
  STRICT_PAIR_GYM_REQUIREMENTS_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_REQUIREMENTS_PATH}")"
  STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH}")"
  STRICT_PAIR_ENTRYPOINT_SHA256="$(strict_pair_sha256_file "${project_root}/examples/run_grpo_single_controller.py")"
  STRICT_PAIR_MODEL_TRANSPORT_COLLECTOR_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/utils/strict_model_transport.py")"
  STRICT_PAIR_MODEL_TRANSPORT_VLLM_ROUTE_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/models/generation/vllm/vllm_worker_async.py")"
  STRICT_PAIR_MODEL_TRANSPORT_ROLLOUT_FINALIZER_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/experience/rollout_manager.py")"
  STRICT_PAIR_LAUNCHER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_launch.sh")"
  STRICT_PAIR_ARM_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/nano35_single_env_pair.sh")"
  STRICT_PAIR_PARENT_WRAPPER_SHA256="$(strict_pair_sha256_file "${recipe_dir}/launch_pair.sh")"
  STRICT_PAIR_CONTRACT_SHA256="$(strict_pair_sha256_file "${recipe_dir}/strict_pair_contract.sh")"
  STRICT_PAIR_MANIFEST_PATH="${RESULTS_DIR}/PAIR_MANIFEST.json"
  STRICT_PAIR_SNAPSHOT_PARENT="${RESULTS_DIR}/code_snapshots_strict_pairs/${PAIR_ID}"
  STRICT_PAIR_OFF_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/off-${PAIR_ID}"
  STRICT_PAIR_ON_SNAPSHOT="${STRICT_PAIR_SNAPSHOT_PARENT}/on-${PAIR_ID}"
  STRICT_PAIR_SLURM_EXPORT_PARENT="${RESULTS_DIR}/strict_pair_slurm_exports/${PAIR_ID}"
  STRICT_PAIR_OFF_SLURM_EXPORT="${STRICT_PAIR_SLURM_EXPORT_PARENT}/off.env"
  STRICT_PAIR_ON_SLURM_EXPORT="${STRICT_PAIR_SLURM_EXPORT_PARENT}/on.env"
  STRICT_PAIR_SUBMISSION_CONTRACT_PATH="${RESULTS_DIR}/STRICT_PAIR_SUBMISSION_CONTRACT.json"
  STRICT_PAIR_SUBMISSION_RECEIPT_PATH="${RESULTS_DIR}/PAIR_SUBMISSION_RECEIPT.json"
  STRICT_PAIR_SUBMISSION_STATE_PARENT="${RESULTS_DIR}/strict_pair_submission_state/${PAIR_ID}"
  STRICT_PAIR_OFF_ACCEPTED_ID_RECORD="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/off.job-id"
  STRICT_PAIR_ON_ACCEPTED_ID_RECORD="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/on.job-id"
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

  strict_pair_select_environment
  STRICT_PAIR_PROJECT_ROOT="${project_root}"
  STRICT_PAIR_RECIPE_DIR="${recipe_dir}"
  strict_pair_require_canonical_dir "${RESULTS_DIR}" "RESULTS_DIR"
  strict_pair_require_mode "${RESULTS_DIR}" "700" "RESULTS_DIR"
  strict_pair_require_canonical_dir "${PERSISTENT_CACHE}" "PERSISTENT_CACHE"
  strict_pair_require_canonical_dir "${HF_HOME}" "HF_HOME"
  strict_pair_require_canonical_dir "${MODEL_PATH}" "MODEL_PATH"
  strict_pair_require_canonical_file "${CONTAINER}" "CONTAINER"
  strict_pair_require_canonical_file "${SANDBOX_CONTAINER}" "SANDBOX_CONTAINER"
  strict_pair_require_canonical_file "${TRAIN_PATH}" "TRAIN_PATH"
  strict_pair_require_canonical_dir "${DEPLOYMENT_ROOT}" "DEPLOYMENT_ROOT"
  strict_pair_require_mode "${DEPLOYMENT_ROOT}" "500" "DEPLOYMENT_ROOT"
  strict_pair_validate_slurm_conf

  for expected_name in \
    EXPECTED_DEPLOYMENT_READY \
    EXPECTED_DEPLOYMENT_READY_FILE_SHA256 \
    EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256 \
    EXPECTED_STRICT_PAIR_JOB_WRAPPER_SHA256 \
    EXPECTED_STRICT_PAIR_MODEL_TREE_SHA256 \
    EXPECTED_STRICT_PAIR_CONTAINER_SHA256 \
    EXPECTED_STRICT_PAIR_SANDBOX_CONTAINER_SHA256 \
    EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256; do
    strict_pair_require_digest "${expected_name}"
  done
  STRICT_PAIR_ACCEPTANCE_POLICY_SHA256="$(
    strict_pair_acceptance_policy_sha256
  )"
  if [[ "${STRICT_PAIR_ACCEPTANCE_POLICY_SHA256}" != \
        "${EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256}" ]]; then
    strict_pair_error "live-learning acceptance policy SHA-256 differs from the parent anchor."
  fi

  STRICT_PAIR_DEPLOYED_NEMO_ROOT="${DEPLOYMENT_ROOT}/runnable/NemoRL"
  strict_pair_require_canonical_dir "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "deployed NeMo-RL root"
  strict_pair_require_mode "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" "500" "deployed NeMo-RL root"
  if [[ "${STRICT_PAIR_PROJECT_ROOT}" != "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}" ]]; then
    strict_pair_error "launcher source must be exactly DEPLOYMENT_ROOT/runnable/NemoRL."
  fi
  if [[ "${TRAIN_PATH}" != \
        "${STRICT_PAIR_DEPLOYED_NEMO_ROOT}/${STRICT_PAIR_FIXTURE_RELATIVE}" ]]; then
    strict_pair_error "TRAIN_PATH must be the selected tracked strict fixture."
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
  if [[ "${STRICT_PAIR_FIXTURE_SHA256}" != "${STRICT_PAIR_EXPECTED_FIXTURE_SHA256}" ]]; then
    strict_pair_error "TRAIN_PATH SHA-256 mismatch for ${STRICT_PAIR_ENVIRONMENT}: expected ${STRICT_PAIR_EXPECTED_FIXTURE_SHA256}, got ${STRICT_PAIR_FIXTURE_SHA256}."
  fi
  STRICT_PAIR_FIXTURE_ROWS="$("${STRICT_PAIR_TOOL_AWK}" 'END { print NR }' "${TRAIN_PATH}")"
  if [[ "${STRICT_PAIR_FIXTURE_ROWS}" != "5" ]]; then
    strict_pair_error "authenticated TRAIN_PATH must contain exactly 5 rows; got ${STRICT_PAIR_FIXTURE_ROWS}."
  fi

  strict_pair_load_source_identity
  STRICT_PAIR_CONFIG_PATH="${project_root}/${STRICT_PAIR_CONFIG_RELATIVE}"
  strict_pair_require_canonical_file "${STRICT_PAIR_CONFIG_PATH}" "selected environment config"
  STRICT_PAIR_CONFIG_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_CONFIG_PATH}")"
  STRICT_PAIR_GYM_CONFIG_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_CONFIG_RELATIVE}"
  STRICT_PAIR_GYM_REQUIREMENTS_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE}"
  STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH="${STRICT_PAIR_GYM_ROOT}/${STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE}"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_CONFIG_PATH}" "selected Gym resource config"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_REQUIREMENTS_PATH}" "selected Gym requirements"
  strict_pair_require_canonical_file "${STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH}" "selected Gym verifier source"
  STRICT_PAIR_GYM_CONFIG_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_CONFIG_PATH}")"
  STRICT_PAIR_GYM_REQUIREMENTS_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_REQUIREMENTS_PATH}")"
  STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256="$(strict_pair_sha256_file "${STRICT_PAIR_GYM_VERIFIER_SOURCE_PATH}")"
  STRICT_PAIR_ENTRYPOINT_SHA256="$(strict_pair_sha256_file "${project_root}/examples/run_grpo_single_controller.py")"
  STRICT_PAIR_MODEL_TRANSPORT_COLLECTOR_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/utils/strict_model_transport.py")"
  STRICT_PAIR_MODEL_TRANSPORT_VLLM_ROUTE_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/models/generation/vllm/vllm_worker_async.py")"
  STRICT_PAIR_MODEL_TRANSPORT_ROLLOUT_FINALIZER_SHA256="$(strict_pair_sha256_file "${project_root}/nemo_rl/experience/rollout_manager.py")"
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
  STRICT_PAIR_SLURM_EXPORT_PARENT="${RESULTS_DIR}/strict_pair_slurm_exports/${PAIR_ID}"
  STRICT_PAIR_OFF_SLURM_EXPORT="${STRICT_PAIR_SLURM_EXPORT_PARENT}/off.env"
  STRICT_PAIR_ON_SLURM_EXPORT="${STRICT_PAIR_SLURM_EXPORT_PARENT}/on.env"
  STRICT_PAIR_SUBMISSION_CONTRACT_PATH="${RESULTS_DIR}/STRICT_PAIR_SUBMISSION_CONTRACT.json"
  STRICT_PAIR_SUBMISSION_RECEIPT_PATH="${RESULTS_DIR}/PAIR_SUBMISSION_RECEIPT.json"
  STRICT_PAIR_SUBMISSION_STATE_PARENT="${RESULTS_DIR}/strict_pair_submission_state/${PAIR_ID}"
  STRICT_PAIR_OFF_ACCEPTED_ID_RECORD="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/off.job-id"
  STRICT_PAIR_ON_ACCEPTED_ID_RECORD="${STRICT_PAIR_SUBMISSION_STATE_PARENT}/on.job-id"
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
  local find_inventory
  local find_inventory_sha256

  strict_pair_require_canonical_dir "${snapshot}" "${arm} strict snapshot"
  manifest="${snapshot}/strict-pair-snapshot-manifest.sha256"
  strict_pair_require_canonical_file "${manifest}" "${arm} strict snapshot manifest"
  strict_pair_require_mode "${manifest}" "400" "${arm} strict snapshot manifest"
  manifest_sha256="$(strict_pair_sha256_file "${manifest}")"
  if ! (cd -- "${snapshot}" && \
        "${STRICT_PAIR_TOOL_SHA256SUM}" --check --strict --quiet -- "${manifest##*/}"); then
    strict_pair_error "${arm} strict snapshot content verification failed."
  fi
  symlink_manifest="${snapshot}/strict-pair-snapshot-symlinks.json"
  strict_pair_require_canonical_file "${symlink_manifest}" "${arm} snapshot symlink manifest"
  mode_manifest="${snapshot}/strict-pair-snapshot-modes.json"
  strict_pair_require_canonical_file "${mode_manifest}" "${arm} snapshot mode manifest"
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
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
  if [[ "$(strict_pair_sha256_file "${snapshot}/examples/run_grpo_single_controller.py")" != \
        "${STRICT_PAIR_ENTRYPOINT_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot entrypoint differs from authenticated source."
  fi
  if [[ "$(strict_pair_sha256_file "${snapshot}/${STRICT_PAIR_CONFIG_RELATIVE}")" != \
        "${STRICT_PAIR_CONFIG_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot config differs from authenticated source."
  fi
  if [[ "$(strict_pair_sha256_file "${snapshot}/3rdparty/Gym-workspace/Gym/${STRICT_PAIR_GYM_CONFIG_RELATIVE}")" != \
        "${STRICT_PAIR_GYM_CONFIG_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot selected Gym resource config differs from authenticated source."
  fi
  if [[ "$(strict_pair_sha256_file "${snapshot}/3rdparty/Gym-workspace/Gym/${STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE}")" != \
        "${STRICT_PAIR_GYM_REQUIREMENTS_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot selected Gym requirements differ from authenticated source."
  fi
  if [[ "$(strict_pair_sha256_file "${snapshot}/3rdparty/Gym-workspace/Gym/${STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE}")" != \
        "${STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256}" ]]; then
    strict_pair_error "${arm} strict snapshot selected Gym verifier source differs from authenticated source."
  fi
  entrypoint_manifest="${snapshot}/nano35-entrypoint-manifest.sha256"
  strict_pair_require_canonical_file "${entrypoint_manifest}" "${arm} entrypoint manifest"
  strict_pair_require_mode "${entrypoint_manifest}" "400" "${arm} entrypoint manifest"
  if [[ "$(< "${entrypoint_manifest}")" != \
        "${STRICT_PAIR_ENTRYPOINT_SHA256}  examples/run_grpo_single_controller.py" ]]; then
    strict_pair_error "${arm} entrypoint manifest differs from authenticated source."
  fi
  find_inventory="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" \
      "${RESULTS_DIR}/.strict-pair-snapshot-find.XXXXXX"
  )"
  if ! strict_pair_write_snapshot_find_inventory \
      "${snapshot}" "${find_inventory}" "${arm}"; then
    return 2
  fi
  find_inventory_sha256="$(strict_pair_sha256_file "${find_inventory}")"
  while IFS= read -r -d '' path; do
    mode="$(strict_pair_file_mode "${path}")"
    if [[ ! "${mode}" =~ ^[0-7]{3,4}$ ]]; then
      "${STRICT_PAIR_TOOL_RM}" -f -- "${find_inventory}"
      strict_pair_error "${arm} strict snapshot path has an invalid mode: ${path}"
      return
    fi
    if (( (8#${mode} & 8#222) != 0 )); then
      writable_path="${path}"
      break
    fi
  done < "${find_inventory}"
  if [[ "$(strict_pair_sha256_file "${find_inventory}")" != \
        "${find_inventory_sha256}" ]]; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${find_inventory}"
    strict_pair_error "${arm} strict snapshot inventory changed while it was consumed."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -f -- "${find_inventory}"
  if [[ -n "${writable_path}" ]]; then
    strict_pair_error "${arm} strict snapshot contains a writable path: ${writable_path}"
  fi
  if [[ "${arm}" == "off" ]]; then
    STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256="${manifest_sha256}"
  else
    STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256="${manifest_sha256}"
  fi
}

strict_pair_write_snapshot_find_inventory() {
  local snapshot="$1"
  local output="$2"
  local arm="$3"

  if ! "${STRICT_PAIR_TOOL_FIND}" "${snapshot}" \
      \( -type d -o -type f \) -print0 > "${output}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${output}"
    strict_pair_error "${arm} strict snapshot inventory enumeration failed."
    return
  fi
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${output}"
}

strict_pair_write_snapshot_path_inventory() {
  local output="$1"
  local tracked_path
  local raw_inventory
  local raw_inventory_sha256

  raw_inventory="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" \
      "${RESULTS_DIR}/.strict-pair-git-ls-files.XXXXXX"
  )"
  if ! strict_pair_git -C "${STRICT_PAIR_PROJECT_ROOT}" \
      ls-files --recurse-submodules -z > "${raw_inventory}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${raw_inventory}" "${output}"
    strict_pair_error "tracked snapshot path enumeration failed."
    return
  fi
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${raw_inventory}"
  raw_inventory_sha256="$(strict_pair_sha256_file "${raw_inventory}")"

  while IFS= read -r -d '' tracked_path; do
    # `git ls-files --recurse-submodules` includes the outer mode-160000
    # gitlink records as well as the recursively tracked leaves. Copying a
    # gitlink directory recursively would admit untracked submodule bytes, so
    # emit only regular files and symlinks from the already verified source.
    if [[ ! -d "${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}" || \
          -L "${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}" ]]; then
      if ! printf '%s\0' "${tracked_path}" >> "${output}"; then
        "${STRICT_PAIR_TOOL_RM}" -f -- "${raw_inventory}" "${output}"
        strict_pair_error "failed to materialize tracked snapshot path inventory."
        return
      fi
    fi
  done < "${raw_inventory}"
  if [[ "$(strict_pair_sha256_file "${raw_inventory}")" != \
        "${raw_inventory_sha256}" ]]; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${raw_inventory}" "${output}"
    strict_pair_error "raw tracked path inventory changed while it was consumed."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -f -- "${raw_inventory}"
  if [[ ! -s "${output}" ]]; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${output}"
    strict_pair_error "tracked snapshot path inventory is empty."
    return
  fi
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${output}"
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
  local path_inventory
  local path_inventory_sha256

  if [[ "${arm}" == "off" ]]; then
    snapshot="${STRICT_PAIR_OFF_SNAPSHOT}"
  elif [[ "${arm}" == "on" ]]; then
    snapshot="${STRICT_PAIR_ON_SNAPSHOT}"
  else
    strict_pair_error "unsupported snapshot arm: ${arm}"
  fi
  path_inventory="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" \
      "${RESULTS_DIR}/.strict-pair-snapshot-paths.XXXXXX"
  )"
  if ! strict_pair_write_snapshot_path_inventory "${path_inventory}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
    return 2
  fi
  path_inventory_sha256="$(strict_pair_sha256_file "${path_inventory}")"

  "${STRICT_PAIR_TOOL_MKDIR}" -p -- "${STRICT_PAIR_SNAPSHOT_PARENT}"
  strict_pair_require_canonical_dir "${STRICT_PAIR_SNAPSHOT_PARENT}" "strict snapshot parent"
  if ! "${STRICT_PAIR_TOOL_MKDIR}" -- "${snapshot}" 2>/dev/null; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
    strict_pair_error "strict arm snapshot already exists or is reserved; use a new PAIR_ID: ${snapshot}"
    return
  fi
  if ! "${STRICT_PAIR_TOOL_RSYNC}" -a --from0 \
      --files-from="${path_inventory}" \
      "${STRICT_PAIR_PROJECT_ROOT}/" "${snapshot}/"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
    strict_pair_error "authenticated snapshot copy failed for arm ${arm}."
    return
  fi
  if [[ "$(strict_pair_sha256_file "${path_inventory}")" != \
        "${path_inventory_sha256}" ]]; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
    strict_pair_error "tracked snapshot path inventory changed during copy."
    return
  fi
  manifest="${snapshot}/strict-pair-snapshot-manifest.sha256"
  : > "${manifest}"
  while IFS= read -r -d '' tracked_path; do
    source_path="${STRICT_PAIR_PROJECT_ROOT}/${tracked_path}"
    snapshot_path="${snapshot}/${tracked_path}"
    if [[ -L "${source_path}" ]]; then
      if [[ ! -L "${snapshot_path}" || \
            "$("${STRICT_PAIR_TOOL_READLINK}" -- "${snapshot_path}")" != \
            "$("${STRICT_PAIR_TOOL_READLINK}" -- "${source_path}")" ]]; then
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
  done < "${path_inventory}"
  if [[ "$(strict_pair_sha256_file "${path_inventory}")" != \
        "${path_inventory_sha256}" ]]; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
    strict_pair_error "tracked snapshot path inventory changed during verification."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -f -- "${path_inventory}"
  symlink_manifest="${snapshot}/strict-pair-snapshot-symlinks.json"
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "${snapshot}" "${symlink_manifest}" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
symlinks = {}


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
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - "${snapshot}" "${manifest}" "${mode_manifest}" <<'PY'
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
sha_manifest = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
files = {}


def raise_walk_error(error: OSError) -> None:
    raise error


for directory, directory_names, file_names in os.walk(
    root, onerror=raise_walk_error, followlinks=False
):
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
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${manifest}"
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${entrypoint_manifest}"
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${symlink_manifest}"
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${mode_manifest}"
  "${STRICT_PAIR_TOOL_CHMOD}" -R a-w "${snapshot}"
  strict_pair_verify_snapshot "${arm}" "${snapshot}"
}

strict_pair_load_snapshots() {
  strict_pair_verify_snapshot off "${STRICT_PAIR_OFF_SNAPSHOT}"
  strict_pair_verify_snapshot on "${STRICT_PAIR_ON_SNAPSHOT}"
}

strict_pair_generate_submission_nonce() {
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - <<'PY'
import secrets

print(secrets.token_hex(32))
PY
}

strict_pair_validate_slurm_conf() {
  strict_pair_require_digest EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256
  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${STRICT_PAIR_SLURM_CONF}" \
    "${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256}" \
    "${DEPLOYMENT_ROOT}" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected_sha256 = sys.argv[2]
deployment = pathlib.Path(sys.argv[3])
metadata = os.lstat(path)
if not path.is_absolute() or os.path.realpath(path) != str(path):
    raise SystemExit("STRICT_PAIR_SLURM_CONF must be one canonical absolute path")
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("STRICT_PAIR_SLURM_CONF must be regular and non-symlink")
darwin_test_config = False
if sys.platform == "darwin":
    try:
        path.relative_to(deployment)
    except ValueError:
        pass
    else:
        darwin_test_config = stat.S_IMODE(metadata.st_mode) == 0o400
if not darwin_test_config:
    if str(path) != "/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf":
        raise SystemExit("STRICT_PAIR_SLURM_CONF must use the pinned HSG path")
    if expected_sha256 != (
        "2f81094a7a631b921d33513e6a3d74b9"
        "6360510b5f9766f75b0cf45ebd95a410"
    ):
        raise SystemExit("STRICT_PAIR_SLURM_CONF must use the pinned HSG SHA-256")
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise SystemExit("STRICT_PAIR_SLURM_CONF must be owned by uid0/gid0")
    if stat.S_IMODE(metadata.st_mode) != 0o644:
        raise SystemExit("STRICT_PAIR_SLURM_CONF must have mode 644")
actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit("STRICT_PAIR_SLURM_CONF SHA-256 mismatch")
PY
  STRICT_PAIR_SLURM_CONF_SHA256="${EXPECTED_STRICT_PAIR_SLURM_CONF_SHA256}"
}

strict_pair_render_submission_contract() {
  local output="$1"

  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${output}" "${STRICT_PAIR_SLURM_CONF}" \
    "${STRICT_PAIR_SLURM_CONF_SHA256}" \
    "${STRICT_PAIR_ENVIRONMENT}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
slurm_conf_path = sys.argv[2]
slurm_conf_sha256 = sys.argv[3]
environment = sys.argv[4]
if environment not in {"reasoning_gym", "citation", "freeform"}:
    raise SystemExit("strict submission contract environment is invalid")
host_keys = [
    "awk",
    "bash",
    "cat",
    "chmod",
    "cmp",
    "date",
    "env",
    "find",
    "git",
    "grep",
    "ln",
    "mkdir",
    "mktemp",
    "nvidia_smi",
    "python",
    "readlink",
    "realpath",
    "rm",
    "rsync",
    "sbatch",
    "scancel",
    "scontrol",
    "sha256sum",
    "stat",
    "wc",
]
failure_stages = [
    "arm_submit",
    "job_id_validation",
    "pre_release_validation",
    "release",
    "post_release_validation",
    "receipt_publication",
    "unexpected_exit",
]
receipt_root_keys = [
    "acceptance",
    "authenticated_jobs",
    "cancellations",
    "execution_environment",
    "held_submissions",
    "model_transport",
    "outcome",
    "pair",
    "post_cancel_queries",
    "post_release_query",
    "pre_cancel_queries",
    "pre_release_query",
    "receipt",
    "recovery_query",
    "release",
    "rollback_candidates",
    "rollback_confirmed",
    "runtime_tools",
    "scheduler_tools",
    "schema",
    "selection",
    "source",
    "stage",
    "submission_contract",
    "submission_nonce",
    "wandb",
]
digest = "0" * 64
submission_contract = {
    "path": "/strict/STRICT_PAIR_SUBMISSION_CONTRACT.json",
    "sha256": digest,
}
receipt_pair = {
    "id": "example-pair",
    "manifest": {"path": "/strict/PAIR_MANIFEST.json", "sha256": digest},
}
selection = {
    "config": {"path": "examples/selected-environment.yaml", "sha256": digest},
    "environment": environment,
    "fixture": {"path": "/strict/fixture.jsonl", "rows": 5, "sha256": digest},
    "gym_resources": {
        "config": {"path": "resources_servers/selected/config.yaml", "sha256": digest},
        "requirements": {
            "path": "resources_servers/selected/requirements.txt",
            "sha256": digest,
        },
        "verifier_source": {
            "path": "resources_servers/selected/server.py",
            "sha256": digest,
        },
    },
}
execution_environment = {
    "arm_launcher": {
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
    },
    "arms": {
        arm: {
            "base_log_dir": f"/strict/results/{arm}/ray_logs",
            "cache_read": {
                "entry_count": 0,
                "mode": "0700",
                "path": f"/strict/cache/{arm}/cache_read",
                "policy": "empty-at-publication-and-job-entry-no-read",
            },
            "hf_datasets_cache": f"/strict/hf/{arm}/hub",
            "hf_home": f"/strict/hf/{arm}",
            "hf_hub_cache": f"/strict/hf/{arm}/hub",
            "persistent_cache": f"/strict/cache/{arm}",
            "results_dir": f"/strict/results/{arm}",
            "scheduler": {
                "batch_working_directory": f"/strict/snapshots/{arm}-example-pair",
                "sbatch_chdir_argument": (
                    f"--chdir=/strict/snapshots/{arm}-example-pair"
                ),
                "sbatch_client_cwd": f"/strict/snapshots/{arm}-example-pair",
                "slurm_submit_dir": f"/strict/snapshots/{arm}-example-pair",
            },
            "setup_command": {"byte_count": 1, "sha256": digest},
        }
        for arm in ("off", "on")
    },
    "fixed": {
        "cpus_per_worker": "144",
        "nemo_skills_sandbox_port": "6000",
        "ray_log_sync_frequency": "60",
        "sandbox_command": "/start-with-nginx.sh",
        "train_path": "/strict/fixture.jsonl",
        "val_path": "/strict/fixture.jsonl",
    },
    "schema": "nemo-rl-strict-execution-environment-v1",
}
receipt_source = {
    "bridge": {
        "head": "1" * 40,
        "root": "/strict/deployment/runnable/Megatron-Bridge",
        "tree": "2" * 40,
    },
    "mcore": {
        "head": "3" * 40,
        "root": "/strict/deployment/runnable/Megatron-LM",
        "tree": "4" * 40,
    },
}
wandb = {
    "arms": {
        arm: {
            "name": f"{arm}-{environment}-example-pair",
            "name_template": f"{arm}-{{environment}}-{{pair_id}}",
            "run_id": hashlib.sha256(
                (
                    "nemo-rl-strict-wandb-v1:"
                    f"{environment}:example-pair:{arm}"
                ).encode("ascii")
            ).hexdigest(),
        }
        for arm in ("off", "on")
    },
    "entity": "nvidia",
    "group": {
        "template": "{environment}-{pair_id}",
        "value": f"{environment}-example-pair",
    },
    "project": "nano35-rlvr-convergence",
    "resume": "never",
    "run_id_derivation": (
        "sha256-ascii:nemo-rl-strict-wandb-v1:"
        "{environment}:{pair_id}:{arm}"
    ),
}
runtime_tools = {
    "manifest": {
        "path": "/strict/strict_pair_runtime_tools.json",
        "sha256": digest,
    },
    "schema": "nemo-rl-strict-runtime-tools-v2",
}


def strict_domain_sha256(label: str, value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        b"nemo-rl-strict-v2\0" + label.encode("ascii") + b"\0" + payload
    ).hexdigest()


acceptance = {
    "hash_domain": "sha256-domain-nul-canonical-ascii-json-no-lf-v1",
    "live_reward_noninferiority": {
        "burn_in_steps": 10,
        "evaluated_steps": {
            "count": 90,
            "first": 11,
            "last": 100,
            "require_complete_paired_steps": True,
        },
        "margin": 0.1,
        "metric": "train/raw_environment_reward",
        "paired_step_bootstrap": {
            "confidence_level": 0.95,
            "interval": "percentile",
            "lower_quantile": 0.025,
            "resamples": 10000,
            "sampling_unit": "paired-step",
            "seed": 20260828,
            "statistic": "mean-on-minus-off",
            "upper_quantile": 0.975,
        },
        "primary_gate": (
            "lower-confidence-bound-strictly-greater-than-negative-margin"
        ),
        "tail": {
            "gate": "on-mean-greater-than-or-equal-to-off-mean-minus-margin",
            "margin": 0.1,
            "statistic": "mean-on-minus-mean-off",
            "steps": {"count": 25, "first": 76, "last": 100},
        },
    },
    "optimizer_update_witness": {
        "false_value": "fail",
        "json_type": "boolean",
        "ledger_field": "update_successful",
        "missing_or_non_boolean": "unverifiable",
        "required_steps": {"count": 100, "first": 1, "last": 100},
        "required_value": True,
        "wandb_metric": "train/update_successful",
    },
    "schema": "nemo-rl-strict-live-learning-acceptance-policy-v1",
}
acceptance["policy_sha256"] = strict_domain_sha256(
    "live-learning-acceptance-policy", acceptance
)


model_transport = {
    "activation": {
        "arm_environment": {"name": "STRICT_PAIR_ARM", "off": "off", "on": "on"},
        "config_key": "policy.generation.vllm_cfg.strict_model_transport",
        "environment_environment": "STRICT_PAIR_ENVIRONMENT",
        "main_mode": "capture",
        "pair_id_environment": "PAIR_ID",
        "replay_mode": "replay",
        "results_dir_environment": "RESULTS_DIR",
    },
    "arms": ["off", "on"],
    "artifacts": {
        "bundle": {
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport-bundle.json",
            "schema": "nemo-rl-strict-model-transport-bundle-v1",
        },
        "directory": {
            "inventory": [
                "model-transport.jsonl",
                "model-transport-bundle.json",
                "model-transport-manifest.json",
            ],
            "mode": "0700",
            "precondition": "absent-at-pre-runtime-creates-exclusively",
            "relative_path": "strict_model_transport",
        },
        "log": {
            "framing": "canonical-ascii-json-line-lf",
            "lines": 4,
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport.jsonl",
            "schema": "nemo-rl-strict-model-transport-call-v1",
        },
        "manifest": {
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport-manifest.json",
            "schema": "nemo-rl-strict-model-transport-manifest-v1",
        },
        "nlink": 1,
        "owner": "effective-uid",
        "publication": "o-excl-fsync-atomic-seal",
    },
    "capture_window": {
        "concurrency": "arrival-independent",
        "duplicate_or_retry": "reject",
        "fixture_row_index": 0,
        "logical_rollout_indices": [0, 1, 2, 3],
        "main_after_seal": (
            "reject-until-rollout-finalizer-attests-step1-complete-then-pass-through"
        ),
        "replay_after_seal": "reject-terminal",
        "sample_count": 4,
        "seal": "atomic-after-four-successes",
        "seed_base": 42,
        "seed_derivation": "sha256-canonical-ascii-json-int63-v1",
        "step": 1,
    },
    "enabled": True,
    "hash_domain": "sha256-domain-nul-canonical-ascii-json-no-lf-v1",
    "http": {
        "authorization": "excluded",
        "body_boundary": "http-body-bytes-only",
        "cookies": "excluded",
        "direct_python_generation_during_replay": "reject",
        "encoding": "utf-8",
        "endpoint_allowlist": [
            {"logical_count": 4, "method": "POST", "path": "/v1/chat/completions"}
        ],
        "headers": "excluded",
        "max_bundle_bytes": 201326592,
        "max_request_body_bytes": 16777216,
        "max_response_body_bytes": 16777216,
        "probe_allowlist": [],
        "query": "forbidden",
        "request_media_type": "application/json",
        "response_media_type": "application/json",
        "response_status_code": 200,
        "streaming": False,
        "tokenize_count": 0,
        "unlisted_during_replay": "reject",
        "unlisted_during_window": "reject",
    },
    "schema": "nemo-rl-strict-model-transport-policy-v1",
    "sources": {
        "collector": {
            "path": "nemo_rl/utils/strict_model_transport.py",
            "sha256": digest,
        },
        "rollout_finalizer": {
            "path": "nemo_rl/experience/rollout_manager.py",
            "sha256": digest,
        },
        "vllm_route": {
            "path": "nemo_rl/models/generation/vllm/vllm_worker_async.py",
            "sha256": digest,
        },
    },
}
model_transport["policy_sha256"] = strict_domain_sha256(
    "model-transport-policy", model_transport
)
scheduler_tools = {
    "client_environment": {
        "ambient_merge": False,
        "env": {"path": "/usr/bin/env", "sha256": digest},
        "variables": {
            "LC_ALL": "C",
            "SLURM_CONF": {
                "path": slurm_conf_path,
                "sha256": slurm_conf_sha256,
            },
        },
    },
    "sbatch": {"path": "/strict/sbatch", "sha256": digest},
    "scancel": {"path": "/strict/scancel", "sha256": digest},
    "scontrol": {"path": "/strict/scontrol", "sha256": digest},
}


def state(arm, job_id, job_state, reason):
    return {
        "comment": f"nemo-rl-strict-pair-v1:{arm}:{'1' * 64}:{digest}",
        "held": job_state == "PENDING" and reason == "JobHeldUser",
        "job_id": job_id,
        "job_name": f"{arm}-{environment}-example-pair",
        "job_state": job_state,
        "reason": reason,
        "user_id": "1000",
        "work_dir": f"/strict/snapshots/{arm}-example-pair",
    }


def recovery_state(arm, candidate_job_id, job_state, reason):
    return state(arm, candidate_job_id, job_state, reason)


def query(phase, job_ids_csv, records, *, on_job_id=None):
    job_ids = job_ids_csv.split(",")
    off_job_id = job_ids[0]
    if on_job_id is None and len(job_ids) == 2:
        on_job_id = job_ids[1]
    authenticated_job_ids = [
        job_id
        for job_id in job_ids
        if sum(
            record["job_id"] == job_id
            for arm_records in records.values()
            for record in arm_records
        )
        == 1
    ]
    return {
        "argv": [
            scheduler_tools["scontrol"]["path"],
            "show",
            "job",
            "--json",
            job_ids_csv,
        ],
        "authenticated_job_ids": authenticated_job_ids,
        "byte_count": 1,
        "candidate_job_ids": {
            "off": [off_job_id],
            "on": [] if on_job_id is None else [on_job_id],
            "unattributed": [],
        },
        "complete": True,
        "line_count": len(job_ids),
        "normalization_status": 0,
        "output_sha256_raw": digest,
        "phase": phase,
        "records": records,
        "securely_unlinked": True,
        "status": 0,
        "unterminated_final_line": False,
        "unresolved_job_ids": [],
    }


def authenticated_job(arm, job_id):
    return {
        "comment": f"nemo-rl-strict-pair-v1:{arm}:{'1' * 64}:{digest}",
        "job_id": job_id,
        "job_name": f"{arm}-{environment}-example-pair",
        "user_id": "1000",
    }


def held(arm, job_id, wrapper_status=0, source="accepted-id-record"):
    record_payload = b"" if job_id is None else (job_id + "\n").encode("ascii")
    return {
        "accepted_id_record": {
            "path": f"/strict/strict_pair_submission_state/example-pair/{arm}.job-id",
            "parsed_job_id": job_id,
            "sha256": hashlib.sha256(record_payload).hexdigest(),
        },
        "candidate_job_id": job_id,
        "candidate_job_id_source": "none" if job_id is None else source,
        "candidate_job_id_sha256_ascii_no_newline": (
            None
            if job_id is None
            else hashlib.sha256(job_id.encode("ascii")).hexdigest()
        ),
        "submission_rpc": {
            "drained_unix_ns": 2,
            "relay_status": 0,
            "sbatch_status": 0,
            "started_unix_ns": 1,
            "writer_drained": True,
        },
        "wrapper_status": wrapper_status,
    }


def recovery(records):
    return {
        "argv": [scheduler_tools["scontrol"]["path"], "show", "job", "--json"],
        "byte_count": 1,
        "identity_match_counts": {
            arm: len(records[arm]) for arm in ("off", "on")
        },
        "line_count": sum(len(records[arm]) for arm in ("off", "on")),
        "normalization_status": 0,
        "output_sha256_raw": digest,
        "parsed_record_count": sum(
            len(records[arm]) for arm in ("off", "on")
        ),
        "records": records,
        "securely_unlinked": True,
        "status": 0,
        "unterminated_candidate_job_ids": [],
        "unterminated_final_line": False,
    }


receipt_self = {
    "path": "/strict/PAIR_SUBMISSION_RECEIPT.json",
    "schema": "nemo-rl-strict-pair-submission-receipt-v2",
}
released_example = {
    "acceptance": acceptance,
    "authenticated_jobs": {
        "off": [authenticated_job("off", "41001")],
        "on": [authenticated_job("on", "41002")],
    },
    "cancellations": [],
    "execution_environment": execution_environment,
    "held_submissions": {
        "off": held("off", "41001"),
        "on": held("on", "41002"),
    },
    "model_transport": model_transport,
    "outcome": "released",
    "pair": receipt_pair,
    "post_cancel_queries": [],
    "post_release_query": query(
        "post",
        "41001,41002",
        {
            "off": [state("off", "41001", "RUNNING", "None")],
            "on": [state("on", "41002", "PENDING", "Resources")],
        },
    ),
    "pre_cancel_queries": [],
    "pre_release_query": query(
        "pre",
        "41001,41002",
        {
            "off": [state("off", "41001", "PENDING", "JobHeldUser")],
            "on": [state("on", "41002", "PENDING", "JobHeldUser")],
        },
    ),
    "receipt": receipt_self,
    "recovery_query": None,
    "release": {
        "argv": [
            scheduler_tools["scontrol"]["path"],
            "release",
            "41001,41002",
        ],
        "output_sha256_ascii_no_newline": digest,
        "status": 0,
    },
    "rollback_candidates": {
        "off": ["41001"],
        "on": ["41002"],
        "unattributed": [],
    },
    "rollback_confirmed": None,
    "runtime_tools": runtime_tools,
    "scheduler_tools": scheduler_tools,
    "schema": "nemo-rl-strict-pair-submission-receipt-v2",
    "selection": selection,
    "source": receipt_source,
    "stage": "complete",
    "submission_contract": submission_contract,
    "submission_nonce": "1" * 64,
    "wandb": wandb,
}
failed_closed_example = {
    "acceptance": acceptance,
    "authenticated_jobs": {
        "off": [authenticated_job("off", "41001")],
        "on": [],
    },
    "cancellations": [
        {
            "argv": [scheduler_tools["scancel"]["path"], "41001"],
            "job_ids": ["41001"],
            "output_sha256_ascii_no_newline": digest,
            "status": 0,
        }
    ],
    "execution_environment": execution_environment,
    "held_submissions": {
        "off": held("off", "41001"),
        "on": held("on", None, wrapper_status=79),
    },
    "model_transport": model_transport,
    "outcome": "failed-closed",
    "pair": receipt_pair,
    "post_cancel_queries": [
        query(
            "cancel",
            "41001",
            {"off": [state("off", "41001", "CANCELLED", "None")], "on": []},
        )
    ],
    "post_release_query": None,
    "pre_cancel_queries": [
        query(
            "identity",
            "41001",
            {
                "off": [state("off", "41001", "PENDING", "JobHeldUser")],
                "on": [],
            },
        )
    ],
    "pre_release_query": None,
    "receipt": receipt_self,
    "recovery_query": recovery(
        {
            "off": [
                recovery_state("off", "41001", "PENDING", "JobHeldUser")
            ],
            "on": [],
        }
    ),
    "release": None,
    "rollback_candidates": {
        "off": ["41001"],
        "on": [],
        "unattributed": [],
    },
    "rollback_confirmed": True,
    "runtime_tools": runtime_tools,
    "scheduler_tools": scheduler_tools,
    "schema": "nemo-rl-strict-pair-submission-receipt-v2",
    "selection": selection,
    "source": receipt_source,
    "stage": "arm_submit",
    "submission_contract": submission_contract,
    "submission_nonce": "1" * 64,
    "wandb": wandb,
}
rollback_unconfirmed_example = dict(failed_closed_example)
rollback_unconfirmed_example.update(
    {
        "outcome": "rollback-unconfirmed",
        "post_cancel_queries": [],
        "rollback_confirmed": False,
    }
)
document = {
    "cancellation": {
        "argv": ["{runtime_tools.host.scancel.path}", "{accepted_job_ids_csv}"],
        "failure_stages": failure_stages,
        "job_id_order": ["off", "on"],
        "single_rpc": True,
    },
    "receipt": {
        "allowed_outcomes": [
            "failed-closed",
            "released",
            "rollback-unconfirmed",
        ],
        "allowed_stages": [*failure_stages, "complete"],
        "canonicalization": {
            "allow_nan": False,
            "encoding": "ascii",
            "file_mode": "0400",
            "json_keys": "sorted",
            "separators": [",", ":"],
            "sha256_includes_trailing_lf": True,
            "trailing_lf_count": 1,
        },
        "examples_are_normative_exact_shapes": True,
        "nullable_object_fields": [
            "post_release_query",
            "pre_release_query",
            "recovery_query",
            "release",
        ],
        "examples": {
            "failed_closed": failed_closed_example,
            "released": released_example,
            "rollback_unconfirmed": rollback_unconfirmed_example,
        },
        "result_marker": {
            "field_order": [
                "schema",
                "outcome",
                "stage",
                "pair_id",
                "environment",
                "nonce",
                "off_identity",
                "on_identity",
                "receipt_path",
                "receipt_sha256",
            ],
            "grammar": (
                "STRICT_PAIR_RESULT schema=<schema> outcome=<outcome> "
                "stage=<stage> pair_id=<pair_id> environment=<closed-enum> "
                "nonce=<nonce> "
                "off_(job_id|candidate_job_id)=<canonical-positive-int63-or-none> "
                "off_identity_authenticated=<true|false> "
                "on_(job_id|candidate_job_id)=<canonical-positive-int63-or-none> "
                "on_identity_authenticated=<true|false> "
                "receipt_path=<absolute-path> receipt_sha256=<sha256>"
            ),
            "job_id_label_rule": (
                "*_job_id is reserved for membership in authenticated_jobs; "
                "all other numeric values use *_candidate_job_id"
            ),
            "prefix": "STRICT_PAIR_RESULT",
        },
        "required_root_keys": receipt_root_keys,
        "schema": "nemo-rl-strict-pair-submission-receipt-v2",
        "sha256_delivery": "launcher_return_record_only",
    },
    "pair": {"environment": environment},
    "runtime_tools": {
        "host_keys": host_keys,
        "schema": "nemo-rl-strict-runtime-tools-v2",
    },
    "scheduler": {
        "accepted_id_records": {
            "arms": ["off", "on"],
            "format": "ascii-decimal-lf",
            "initial_mode": "0600",
            "parent_open": "O_CREAT|O_EXCL|O_WRONLY",
            "sealed_mode": "0400",
            "sha256_includes_lf": True,
        },
        "client_environment": {
            "ambient_merge": False,
            "argv_prefix": ["-i"],
            "executable": "{runtime_tools.host.env.path}",
            "variables": {
                "LC_ALL": "C",
                "SLURM_CONF": {
                    "path": slurm_conf_path,
                    "sha256": slurm_conf_sha256,
                },
            },
        },
        "query_capture": {
            "authentication_requires": [
                "status-zero",
                "normalization-status-zero",
                "complete-lf-framing",
                "exactly-one-identity-record",
                "exact-arm-snapshot-workdir",
            ],
            "candidate_id_is_not_identity": True,
            "overlapping_candidate_bucket_policy": "normalize-to-unattributed",
            "exclusive_private_raw_file": True,
            "raw_fields": [
                "byte_count",
                "line_count",
                "output_sha256_raw",
                "unterminated_final_line",
            ],
            "secure_unlink_before_receipt": True,
            "malformed_identity_line_policy": (
                "numeric-id-candidate-never-authenticated"
            ),
            "unterminated_line_policy": "candidate-only-never-authenticated",
        },
        "post_release_query": {
            "allowed_job_states": ["CONFIGURING", "PENDING", "RUNNING"],
            "argv": [
                "show",
                "job",
                "--json",
                "{off_job_id},{on_job_id}",
            ],
            "forbidden_reason": "JobHeldUser",
            "require_each_job_id_once": True,
            "work_dir_template": "{source.snapshots.{arm}.path}",
        },
        "post_cancel_queries": {
            "argv": [
                "show",
                "job",
                "--json",
                "{accepted_job_ids_csv}",
            ],
            "job_state": "CANCELLED",
            "require_each_job_id_once": True,
            "work_dir_template": "{source.snapshots.{arm}.path}",
        },
        "pre_cancel_queries": {
            "argv": [
                "show",
                "job",
                "--json",
                "{accepted_job_ids_csv}",
            ],
            "identity_fields": [
                "JobId",
                "JobName",
                "Comment",
                "UserId",
                "WorkDir",
            ],
            "require_each_job_id_once": True,
            "work_dir_template": "{source.snapshots.{arm}.path}",
        },
        "pre_release_query": {
            "argv": [
                "show",
                "job",
                "--json",
                "{off_job_id},{on_job_id}",
            ],
            "job_state": "PENDING",
            "reason": "JobHeldUser",
            "require_each_job_id_once": True,
            "work_dir_template": "{source.snapshots.{arm}.path}",
        },
        "release": {
            "argv": ["release", "{off_job_id},{on_job_id}"],
            "single_rpc": True,
        },
        "recovery_query": {
            "activation": "only-when-two-distinct-valid-accepted-ids-unavailable",
            "admission_authority": False,
            "argv": ["show", "job", "--json"],
            "comment_template": (
                "nemo-rl-strict-pair-v1:{arm}:"
                "{submission_nonce}:{pair_manifest_sha256}"
            ),
            "duplicate_match_admission": "reject",
            "identity_only": True,
            "job_name_template": "{arm}-{environment}-{pair_id}",
            "observed_matches_and_counts_preserved": True,
            "require_every_json_document_and_job_parseable": True,
            "rollback_authority": "fresh-exact-id-query-authentication-only",
            "targeted_reauthentication_before_cancel": True,
            "unrelated_records_in_receipt": "count-and-raw-sha-only",
            "use": "rollback-only-discovery",
            "user_id": "parent-euid-decimal",
            "work_dir_template": "{source.snapshots.{arm}.path}",
        },
        "submit": {
            "arms": ["off", "on"],
            "argv_prefix": ["--parsable", "--hold"],
            "comment_template": (
                "nemo-rl-strict-pair-v1:{arm}:"
                "{submission_nonce}:{pair_manifest_sha256}"
            ),
            "distinct_numeric_job_ids": True,
            "job_id_maximum": 9223372036854775807,
            "job_id_minimum": 1,
        },
    },
    "schema": "nemo-rl-strict-pair-submission-contract-v2",
}
with output.open("x", encoding="ascii", newline="\n") as stream:
    json.dump(
        document,
        stream,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY
}

strict_pair_publish_submission_contract() {
  local candidate

  if [[ -e "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" || \
        -L "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" ]]; then
    strict_pair_error "STRICT_PAIR_SUBMISSION_CONTRACT.json must be absent for a fresh strict pair."
    return
  fi
  candidate="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" \
      "${RESULTS_DIR}/.STRICT_PAIR_SUBMISSION_CONTRACT.json.candidate.XXXXXX"
  )"
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  if ! strict_pair_render_submission_contract "${candidate}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    strict_pair_error "failed to render strict pair submission contract."
    return
  fi
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${candidate}"
  if ! STRICT_PAIR_SUBMISSION_CONTRACT_SHA256="$(
    strict_pair_sha256_file "${candidate}"
  )"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    strict_pair_error "failed to hash strict pair submission contract."
    return
  fi
  if ! "${STRICT_PAIR_TOOL_LN}" -- \
      "${candidate}" "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    strict_pair_error "failed to atomically publish strict pair submission contract."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  strict_pair_require_canonical_file \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" \
    "STRICT_PAIR_SUBMISSION_CONTRACT.json"
  strict_pair_require_mode \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" "400" \
    "STRICT_PAIR_SUBMISSION_CONTRACT.json"
}

strict_pair_verify_submission_contract() {
  local actual_sha256
  local candidate

  strict_pair_require_digest STRICT_PAIR_SUBMISSION_CONTRACT_SHA256
  strict_pair_require_canonical_file \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" \
    "STRICT_PAIR_SUBMISSION_CONTRACT.json"
  strict_pair_require_mode \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" "400" \
    "STRICT_PAIR_SUBMISSION_CONTRACT.json"
  actual_sha256="$(
    strict_pair_sha256_file "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}"
  )"
  if [[ "${actual_sha256}" != "${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" ]]; then
    strict_pair_error "strict pair submission contract SHA-256 mismatch."
    return
  fi
  candidate="$(
    "${STRICT_PAIR_TOOL_MKTEMP}" \
      "${RESULTS_DIR}/.STRICT_PAIR_SUBMISSION_CONTRACT.json.verify.XXXXXX"
  )"
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  if ! strict_pair_render_submission_contract "${candidate}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    strict_pair_error "failed to recompute strict pair submission contract."
    return
  fi
  if ! "${STRICT_PAIR_TOOL_CMP}" -s -- \
      "${candidate}" "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}"; then
    "${STRICT_PAIR_TOOL_RM}" -f -- "${candidate}"
    strict_pair_error "strict pair submission contract differs from canonical bytes."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
}

strict_pair_render_manifest() {
  local output="$1"

  "${STRICT_PAIR_TOOL_PYTHON}" -I -B - \
    "${output}" "${PAIR_ID}" "${STRICT_PAIR_ENVIRONMENT}" \
    "${STRICT_PAIR_CONFIG_RELATIVE}" "${STRICT_PAIR_VERIFIER_METRIC}" \
    "${RESULTS_DIR}" "${PERSISTENT_CACHE}" "${HF_HOME}" \
    "${DEPLOYMENT_ROOT}" "${EXPECTED_DEPLOYMENT_READY}" \
    "${EXPECTED_DEPLOYMENT_READY_FILE_SHA256}" \
    "${EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256}" \
    "${EXPECTED_BRIDGE_RUNNABLE_MANIFEST_SHA256}" \
    "${EXPECTED_MCORE_RUNNABLE_MANIFEST_SHA256}" \
    "${MODEL_PATH}" "${STRICT_PAIR_MODEL_TREE_SHA256}" \
    "${CONTAINER}" "${STRICT_PAIR_CONTAINER_SHA256}" \
    "${SANDBOX_CONTAINER}" "${STRICT_PAIR_SANDBOX_CONTAINER_SHA256}" \
    "${TRAIN_PATH}" "${STRICT_PAIR_FIXTURE_SHA256}" \
    "${STRICT_PAIR_FIXTURE_ROWS}" \
    "${STRICT_PAIR_PROJECT_ROOT}" "${STRICT_PAIR_NEMO_HEAD}" "${STRICT_PAIR_NEMO_TREE}" \
    "${STRICT_PAIR_GYM_ROOT}" "${STRICT_PAIR_GYM_GITLINK_COMMIT}" "${STRICT_PAIR_GYM_TREE}" \
    "${STRICT_PAIR_BRIDGE_ROOT}" "${STRICT_PAIR_BRIDGE_HEAD}" \
    "${STRICT_PAIR_BRIDGE_TREE}" \
    "${STRICT_PAIR_MCORE_ROOT}" "${STRICT_PAIR_MCORE_HEAD}" \
    "${STRICT_PAIR_MCORE_TREE}" \
    "${STRICT_PAIR_GYM_CONFIG_RELATIVE}" "${STRICT_PAIR_GYM_CONFIG_SHA256}" \
    "${STRICT_PAIR_GYM_REQUIREMENTS_RELATIVE}" \
    "${STRICT_PAIR_GYM_REQUIREMENTS_SHA256}" \
    "${STRICT_PAIR_GYM_VERIFIER_SOURCE_RELATIVE}" \
    "${STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256}" \
    "${STRICT_PAIR_CONFIG_SHA256}" "${STRICT_PAIR_ENTRYPOINT_SHA256}" \
    "${STRICT_PAIR_MODEL_TRANSPORT_COLLECTOR_SHA256}" \
    "${STRICT_PAIR_MODEL_TRANSPORT_VLLM_ROUTE_SHA256}" \
    "${STRICT_PAIR_MODEL_TRANSPORT_ROLLOUT_FINALIZER_SHA256}" \
    "${STRICT_PAIR_ACCEPTANCE_POLICY_SHA256}" \
    "${STRICT_PAIR_LAUNCHER_SHA256}" "${STRICT_PAIR_ARM_WRAPPER_SHA256}" \
    "${STRICT_PAIR_PARENT_WRAPPER_SHA256}" "${STRICT_PAIR_CONTRACT_SHA256}" \
    "${STRICT_PAIR_SNAPSHOT_PARENT}" "${STRICT_PAIR_LAUNCH_MODE}" \
    "${STRICT_PAIR_JOB_WRAPPER}" "${STRICT_PAIR_JOB_WRAPPER_SHA256}" \
    "${STRICT_PAIR_OFF_SNAPSHOT}" "${STRICT_PAIR_OFF_SNAPSHOT_MANIFEST_SHA256}" \
    "${STRICT_PAIR_ON_SNAPSHOT}" "${STRICT_PAIR_ON_SNAPSHOT_MANIFEST_SHA256}" \
    "${STRICT_PAIR_BOOTSTRAP_SHA256SUM}" \
    "${EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256}" \
    "${STRICT_PAIR_RUNTIME_TOOL_MANIFEST}" \
    "${EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256}" \
    "${STRICT_PAIR_OFF_SLURM_EXPORT}" \
    "${STRICT_PAIR_OFF_SLURM_EXPORT_SHA256}" \
    "${STRICT_PAIR_ON_SLURM_EXPORT}" \
    "${STRICT_PAIR_ON_SLURM_EXPORT_SHA256}" \
    "${STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES_CSV}" \
    "${STRICT_PAIR_SUBMISSION_NONCE}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_PATH}" \
    "${STRICT_PAIR_SUBMISSION_CONTRACT_SHA256}" \
    "${STRICT_PAIR_OFF_ACCEPTED_ID_RECORD}" \
    "${STRICT_PAIR_ON_ACCEPTED_ID_RECORD}" "${EUID}" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

(
    output,
    pair_id,
    environment,
    config_path,
    verifier_metric,
    results_root,
    cache_root,
    hf_home,
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
    fixture_rows_raw,
    source_root,
    source_head,
    source_tree,
    gym_root,
    gym_gitlink_commit,
    gym_tree,
    bridge_root,
    bridge_head,
    bridge_tree,
    mcore_root,
    mcore_head,
    mcore_tree,
    gym_config_path,
    gym_config_sha256,
    gym_requirements_path,
    gym_requirements_sha256,
    gym_verifier_source_path,
    gym_verifier_source_sha256,
    config_sha256,
    entrypoint_sha256,
    model_transport_collector_sha256,
    model_transport_vllm_route_sha256,
    model_transport_rollout_finalizer_sha256,
    acceptance_policy_sha256,
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
    bootstrap_sha256sum_path,
    bootstrap_sha256sum_sha256,
    runtime_tool_manifest_path,
    runtime_tool_manifest_sha256,
    off_slurm_export_path,
    off_slurm_export_sha256,
    on_slurm_export_path,
    on_slurm_export_sha256,
    slurm_export_allowed_names_csv,
    submission_nonce,
    submission_contract_path,
    submission_contract_sha256,
    off_accepted_id_record,
    on_accepted_id_record,
    submitter_euid_raw,
) = sys.argv[1:]

if environment not in {"reasoning_gym", "citation", "freeform"}:
    raise SystemExit("Pair manifest environment is invalid")
if fixture_rows_raw != "5":
    raise SystemExit("Pair manifest fixture row count must be exactly five")
fixture_rows = int(fixture_rows_raw)


def reject_duplicate_members(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise SystemExit(f"duplicate JSON member: {key}")
        document[key] = value
    return document


runtime_tool_manifest_bytes = pathlib.Path(runtime_tool_manifest_path).read_bytes()
if hashlib.sha256(runtime_tool_manifest_bytes).hexdigest() != runtime_tool_manifest_sha256:
    raise SystemExit("runtime-tool manifest changed before Pair manifest rendering")
runtime_tool_document = json.loads(
    runtime_tool_manifest_bytes.decode("ascii"),
    object_pairs_hook=reject_duplicate_members,
)
slurm_export_allowed_names = slurm_export_allowed_names_csv.split(",")
if (
    not slurm_export_allowed_names
    or slurm_export_allowed_names != sorted(slurm_export_allowed_names)
    or len(slurm_export_allowed_names) != len(set(slurm_export_allowed_names))
):
    raise SystemExit("strict Slurm export names must be nonempty, sorted, and unique")


def parse_slurm_export(path_raw, expected_sha256):
    payload = pathlib.Path(path_raw).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise SystemExit("Slurm export changed before Pair manifest rendering")
    if not payload.endswith(b"\0"):
        raise SystemExit("Slurm export is not NUL terminated")
    values = {}
    for raw_record in payload[:-1].split(b"\0"):
        raw_name, separator, value = raw_record.partition(b"=")
        if not separator:
            raise SystemExit("Slurm export contains a malformed record")
        try:
            name = raw_name.decode("ascii")
        except UnicodeDecodeError as error:
            raise SystemExit("Slurm export name is not ASCII") from error
        if name in values:
            raise SystemExit("Slurm export contains a duplicate name")
        values[name] = value
    if sorted(values) != slurm_export_allowed_names:
        raise SystemExit("Slurm export names differ from the closed allowlist")
    return values


def export_ascii(values, name):
    try:
        return values[name].decode("ascii")
    except UnicodeDecodeError as error:
        raise SystemExit(f"strict Slurm export value is not ASCII: {name}") from error


def strict_domain_sha256(label: str, value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        b"nemo-rl-strict-v2\0" + label.encode("ascii") + b"\0" + payload
    ).hexdigest()


acceptance = {
    "hash_domain": "sha256-domain-nul-canonical-ascii-json-no-lf-v1",
    "live_reward_noninferiority": {
        "burn_in_steps": 10,
        "evaluated_steps": {
            "count": 90,
            "first": 11,
            "last": 100,
            "require_complete_paired_steps": True,
        },
        "margin": 0.1,
        "metric": "train/raw_environment_reward",
        "paired_step_bootstrap": {
            "confidence_level": 0.95,
            "interval": "percentile",
            "lower_quantile": 0.025,
            "resamples": 10000,
            "sampling_unit": "paired-step",
            "seed": 20260828,
            "statistic": "mean-on-minus-off",
            "upper_quantile": 0.975,
        },
        "primary_gate": (
            "lower-confidence-bound-strictly-greater-than-negative-margin"
        ),
        "tail": {
            "gate": "on-mean-greater-than-or-equal-to-off-mean-minus-margin",
            "margin": 0.1,
            "statistic": "mean-on-minus-mean-off",
            "steps": {"count": 25, "first": 76, "last": 100},
        },
    },
    "optimizer_update_witness": {
        "false_value": "fail",
        "json_type": "boolean",
        "ledger_field": "update_successful",
        "missing_or_non_boolean": "unverifiable",
        "required_steps": {"count": 100, "first": 1, "last": 100},
        "required_value": True,
        "wandb_metric": "train/update_successful",
    },
    "schema": "nemo-rl-strict-live-learning-acceptance-policy-v1",
}
acceptance["policy_sha256"] = strict_domain_sha256(
    "live-learning-acceptance-policy", acceptance
)
if acceptance["policy_sha256"] != acceptance_policy_sha256:
    raise SystemExit("live-learning acceptance policy SHA-256 argument is invalid")


off_export = parse_slurm_export(off_slurm_export_path, off_slurm_export_sha256)
on_export = parse_slurm_export(on_slurm_export_path, on_slurm_export_sha256)
fixed_execution_environment = {
    "cpus_per_worker": "144",
    "nemo_skills_sandbox_port": "6000",
    "ray_log_sync_frequency": "60",
    "sandbox_command": "/start-with-nginx.sh",
    "train_path": fixture_path,
    "val_path": fixture_path,
}
execution_arms = {}
for arm, values, snapshot_path in (
    ("off", off_export, off_snapshot_path),
    ("on", on_export, on_snapshot_path),
):
    arm_results_dir = f"{results_root}/{arm}"
    arm_persistent_cache = f"{cache_root}/{arm}"
    arm_cache_read = f"{arm_persistent_cache}/cache_read"
    arm_hf_home = f"{hf_home}/{arm}"
    transport_directory = pathlib.Path(arm_results_dir) / "strict_model_transport"
    if os.path.lexists(transport_directory):
        raise SystemExit(
            f"strict {arm} model-transport directory must be absent at Pair PRE"
        )
    exact_values = {
        "BASE_LOG_DIR": f"{arm_results_dir}/ray_logs",
        "CPUS_PER_WORKER": fixed_execution_environment["cpus_per_worker"],
        "EXP_NAME": f"{arm}-{environment}-{pair_id}",
        "EXPECTED_BRIDGE_HEAD": bridge_head,
        "EXPECTED_BRIDGE_TREE": bridge_tree,
        "EXPECTED_MCORE_HEAD": mcore_head,
        "EXPECTED_MCORE_TREE": mcore_tree,
        "EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256": (
            acceptance["policy_sha256"]
        ),
        "EXPECTED_STRICT_PAIR_SUBMISSION_CONTRACT_SHA256": (
            submission_contract_sha256
        ),
        "HF_DATASETS_CACHE": f"{arm_hf_home}/hub",
        "HF_HOME": arm_hf_home,
        "HF_HUB_CACHE": f"{arm_hf_home}/hub",
        "NEMO_SKILLS_SANDBOX_PORT": fixed_execution_environment[
            "nemo_skills_sandbox_port"
        ],
        "PAIR_ID": pair_id,
        "PERSISTENT_CACHE": arm_persistent_cache,
        "RAY_LOG_SYNC_FREQUENCY": fixed_execution_environment[
            "ray_log_sync_frequency"
        ],
        "RESULTS_DIR": arm_results_dir,
        "SANDBOX_COMMAND": fixed_execution_environment["sandbox_command"],
        "STRICT_PAIR_ENVIRONMENT": environment,
        "STRICT_PREBUILT_SNAPSHOT_DIR": snapshot_path,
        "TRAIN_PATH": fixed_execution_environment["train_path"],
        "VAL_PATH": fixed_execution_environment["val_path"],
        "WANDB_ENTITY": "nvidia",
        "WANDB_NAME": f"{arm}-{environment}-{pair_id}",
        "WANDB_PROJ": "nano35-rlvr-convergence",
        "WANDB_RESUME": "never",
        "WANDB_RUN_GROUP": f"{environment}-{pair_id}",
        "WANDB_RUN_ID": hashlib.sha256(
            (
                f"nemo-rl-strict-wandb-v1:{environment}:"
                f"{pair_id}:{arm}"
            ).encode("ascii")
        ).hexdigest(),
    }
    for name, expected in exact_values.items():
        if export_ascii(values, name) != expected:
            raise SystemExit(f"strict execution control differs for {arm}: {name}")
    setup_command = values["SETUP_COMMAND"]
    if not setup_command:
        raise SystemExit(f"strict SETUP_COMMAND is empty for {arm}")
    for private_path, label in (
        (arm_cache_read, "cache_read"),
        (arm_hf_home, "HF_HOME"),
    ):
        path = pathlib.Path(private_path)
        metadata = os.lstat(path)
        if (
            not path.is_absolute()
            or os.path.realpath(path) != str(path)
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != int(submitter_euid_raw)
        ):
            raise SystemExit(f"strict {arm} {label} is not a private directory")
        with os.scandir(path) as entries:
            if next(entries, None) is not None:
                raise SystemExit(f"strict {arm} {label} must be empty")
    scheduler_directory = snapshot_path
    execution_arms[arm] = {
        "base_log_dir": exact_values["BASE_LOG_DIR"],
        "cache_read": {
            "entry_count": 0,
            "mode": "0700",
            "path": arm_cache_read,
            "policy": "empty-at-publication-and-job-entry-no-read",
        },
        "hf_datasets_cache": exact_values["HF_DATASETS_CACHE"],
        "hf_home": arm_hf_home,
        "hf_hub_cache": exact_values["HF_HUB_CACHE"],
        "persistent_cache": arm_persistent_cache,
        "results_dir": arm_results_dir,
        "scheduler": {
            "batch_working_directory": scheduler_directory,
            "sbatch_chdir_argument": f"--chdir={scheduler_directory}",
            "sbatch_client_cwd": scheduler_directory,
            "slurm_submit_dir": scheduler_directory,
        },
        "setup_command": {
            "byte_count": len(setup_command),
            "sha256": hashlib.sha256(setup_command).hexdigest(),
        },
    }
if (
    execution_arms["off"]["persistent_cache"]
    == execution_arms["on"]["persistent_cache"]
    or execution_arms["off"]["hf_home"] == execution_arms["on"]["hf_home"]
):
    raise SystemExit("strict arm write roots must be disjoint")
execution_environment = {
    "arm_launcher": {
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
    },
    "arms": execution_arms,
    "fixed": fixed_execution_environment,
    "schema": "nemo-rl-strict-execution-environment-v1",
}
wandb = {
    "arms": {
        arm: {
            "name": export_ascii(values, "WANDB_NAME"),
            "name_template": f"{arm}-{{environment}}-{{pair_id}}",
            "run_id": export_ascii(values, "WANDB_RUN_ID"),
        }
        for arm, values in (("off", off_export), ("on", on_export))
    },
    "entity": "nvidia",
    "group": {
        "template": "{environment}-{pair_id}",
        "value": f"{environment}-{pair_id}",
    },
    "project": "nano35-rlvr-convergence",
    "resume": "never",
    "run_id_derivation": (
        "sha256-ascii:nemo-rl-strict-wandb-v1:"
        "{environment}:{pair_id}:{arm}"
    ),
}
if re.fullmatch(r"[0-9a-f]{64}", submission_nonce) is None:
    raise SystemExit("strict scheduler submission nonce must be 64 lowercase hex")
if re.fullmatch(r"[0-9a-f]{64}", submission_contract_sha256) is None:
    raise SystemExit("strict submission contract SHA-256 must be 64 lowercase hex")
if re.fullmatch(r"0|[1-9][0-9]*", submitter_euid_raw) is None:
    raise SystemExit("strict scheduler submitter EUID must be ASCII decimal")
submitter_euid = int(submitter_euid_raw)

def determinism_marker(mode: str) -> str:
    return (
        "SHARED_PREFIX_DETERMINISM_ATTESTED "
        f"mode={mode} env_controls=5 triton_autotune=absent "
        "model_overrides=3 torch_deterministic=true mcore_backward=true "
        "total_controls=9"
    )


off_marker = determinism_marker("observe")
on_marker = determinism_marker("train")


model_transport = {
    "activation": {
        "arm_environment": {"name": "STRICT_PAIR_ARM", "off": "off", "on": "on"},
        "config_key": "policy.generation.vllm_cfg.strict_model_transport",
        "environment_environment": "STRICT_PAIR_ENVIRONMENT",
        "main_mode": "capture",
        "pair_id_environment": "PAIR_ID",
        "replay_mode": "replay",
        "results_dir_environment": "RESULTS_DIR",
    },
    "arms": ["off", "on"],
    "artifacts": {
        "bundle": {
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport-bundle.json",
            "schema": "nemo-rl-strict-model-transport-bundle-v1",
        },
        "directory": {
            "inventory": [
                "model-transport.jsonl",
                "model-transport-bundle.json",
                "model-transport-manifest.json",
            ],
            "mode": "0700",
            "precondition": "absent-at-pre-runtime-creates-exclusively",
            "relative_path": "strict_model_transport",
        },
        "log": {
            "framing": "canonical-ascii-json-line-lf",
            "lines": 4,
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport.jsonl",
            "schema": "nemo-rl-strict-model-transport-call-v1",
        },
        "manifest": {
            "framing": "canonical-ascii-json-no-lf",
            "mode": "0400",
            "relative_path": "strict_model_transport/model-transport-manifest.json",
            "schema": "nemo-rl-strict-model-transport-manifest-v1",
        },
        "nlink": 1,
        "owner": "effective-uid",
        "publication": "o-excl-fsync-atomic-seal",
    },
    "capture_window": {
        "concurrency": "arrival-independent",
        "duplicate_or_retry": "reject",
        "fixture_row_index": 0,
        "logical_rollout_indices": [0, 1, 2, 3],
        "main_after_seal": (
            "reject-until-rollout-finalizer-attests-step1-complete-then-pass-through"
        ),
        "replay_after_seal": "reject-terminal",
        "sample_count": 4,
        "seal": "atomic-after-four-successes",
        "seed_base": 42,
        "seed_derivation": "sha256-canonical-ascii-json-int63-v1",
        "step": 1,
    },
    "enabled": True,
    "hash_domain": "sha256-domain-nul-canonical-ascii-json-no-lf-v1",
    "http": {
        "authorization": "excluded",
        "body_boundary": "http-body-bytes-only",
        "cookies": "excluded",
        "direct_python_generation_during_replay": "reject",
        "encoding": "utf-8",
        "endpoint_allowlist": [
            {"logical_count": 4, "method": "POST", "path": "/v1/chat/completions"}
        ],
        "headers": "excluded",
        "max_bundle_bytes": 201326592,
        "max_request_body_bytes": 16777216,
        "max_response_body_bytes": 16777216,
        "probe_allowlist": [],
        "query": "forbidden",
        "request_media_type": "application/json",
        "response_media_type": "application/json",
        "response_status_code": 200,
        "streaming": False,
        "tokenize_count": 0,
        "unlisted_during_replay": "reject",
        "unlisted_during_window": "reject",
    },
    "schema": "nemo-rl-strict-model-transport-policy-v1",
    "sources": {
        "collector": {
            "path": "nemo_rl/utils/strict_model_transport.py",
            "sha256": model_transport_collector_sha256,
        },
        "rollout_finalizer": {
            "path": "nemo_rl/experience/rollout_manager.py",
            "sha256": model_transport_rollout_finalizer_sha256,
        },
        "vllm_route": {
            "path": "nemo_rl/models/generation/vllm/vllm_worker_async.py",
            "sha256": model_transport_vllm_route_sha256,
        },
    },
}
model_transport["policy_sha256"] = strict_domain_sha256(
    "model-transport-policy", model_transport
)

manifest = {
    "acceptance": acceptance,
    "arms": {"off": "observe", "on": "train"},
    "artifacts": {
        "container": {"path": container_path, "sha256": container_sha256},
        "fixture": {
            "path": fixture_path,
            "rows": fixture_rows,
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
            "vllm_gpu_memory_utilization": 0.1,
        },
        "generation_seed_base": 42,
        "hardware": {"gpu_model": "NVIDIA GB200"},
        "launch_mode": launch_mode,
        "logging": {
            "tensorboard_enabled": False,
            "wandb_enabled": True,
            "wandb_entity": "nvidia",
            "wandb_group_template": "{environment}-{pair_id}",
            "wandb_project": "nano35-rlvr-convergence",
            "wandb_run_name_templates": {
                "off": "off-{environment}-{pair_id}",
                "on": "on-{environment}-{pair_id}",
            },
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
                "verifier_native_raw_score_alias": verifier_metric,
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
        "threat_model": {
            "cooperative_exclusive_campaign_root": True,
            "malicious_same_uid_active_mutation": "out_of_scope",
            "trusted_operator_and_code": True,
        },
        "training_mtp": {
            "detach_heads": True,
            "layers": 5,
            "loss_scale": 0.3,
            "repeated_layer": True,
        },
        "training_topology": "TP2/CP2/PP1/EP4/ETP1/SP",
        "vllm_tp": 4,
    },
    "container_entry_boundary": {
        "bash_args": ["-p"],
        "bash_path": "/bin/bash",
        "env_path": "/usr/bin/env",
        "sha256sum": {
            "path": "/usr/bin/sha256sum",
            "sha256": (
                "f3d040161f5c29e4c7cd4e3d6bb513ce"
                "9a43b9d1bd06f456a6aab3d34d0f1e33"
            ),
        },
        "unset_environment": ["BASH_ENV", "ENV"],
    },
    "determinism_receipt_dir": "shared_prefix_determinism_receipts",
    "deployment": {
        "bridge_runnable_manifest_sha256": bridge_manifest_sha256,
        "mcore_runnable_manifest_sha256": mcore_manifest_sha256,
        "nemo_runnable_manifest_sha256": nemo_manifest_sha256,
        "ready": deployment_ready,
        "ready_file_sha256": deployment_ready_file_sha256,
        "root": deployment_root,
    },
    "execution_environment": execution_environment,
    "model_transport": model_transport,
    "pair_id": pair_id,
    "paths": {
        "cache_root": cache_root,
        "hf_home": hf_home,
        "results_root": results_root,
        "snapshot_parent": snapshot_parent,
    },
    "schema": "nemo-rl-strict-single-env-pair-v2",
    "selection": {
        "config": {"path": config_path, "sha256": config_sha256},
        "environment": environment,
        "fixture": {
            "path": fixture_path,
            "rows": fixture_rows,
            "sha256": fixture_sha256,
        },
        "gym_resources": {
            "config": {
                "path": gym_config_path,
                "sha256": gym_config_sha256,
            },
            "requirements": {
                "path": gym_requirements_path,
                "sha256": gym_requirements_sha256,
            },
            "verifier_source": {
                "path": gym_verifier_source_path,
                "sha256": gym_verifier_source_sha256,
            },
        },
    },
    "wandb": wandb,
    "scheduler_submission": {
        "schema": "nemo-rl-strict-scheduler-submission-v1",
        "accepted_id_records": {
            "off": {
                "accepted_format": "ascii-positive-decimal-lf",
                "capture_format": "opaque-sbatch-stdout",
                "initial_mode": "0600",
                "path": off_accepted_id_record,
                "sealed_mode": "0400",
            },
            "on": {
                "accepted_format": "ascii-positive-decimal-lf",
                "capture_format": "opaque-sbatch-stdout",
                "initial_mode": "0600",
                "path": on_accepted_id_record,
                "sealed_mode": "0400",
            },
        },
        "contract": {
            "path": submission_contract_path,
            "sha256": submission_contract_sha256,
        },
        "identity": {
            "comment_template": (
                "nemo-rl-strict-pair-v1:{arm}:"
                "{submission_nonce}:{pair_manifest_sha256}"
            ),
            "job_names": {
                "off": f"off-{environment}-{pair_id}",
                "on": f"on-{environment}-{pair_id}",
            },
            "submitter_euid": submitter_euid,
        },
        "nonce": submission_nonce,
        "receipt": {
            "path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
            "schema": "nemo-rl-strict-pair-submission-receipt-v2",
        },
    },
    "slurm_export_boundary": {
        "allowed_names": slurm_export_allowed_names,
        "ambient_merge": False,
        "arms": {
            "off": {
                "path": off_slurm_export_path,
                "sha256": off_slurm_export_sha256,
            },
            "on": {
                "path": on_slurm_export_path,
                "sha256": on_slurm_export_sha256,
            },
        },
        "format": "nul-separated-name-value",
        "get_user_env": False,
        "job_argv": [
            "--pair-manifest",
            "{pair_manifest_path}",
            "--pair-manifest-sha256",
            "{pair_manifest_sha256}",
            "--arm",
            "{arm}",
        ],
        "schema": "nemo-rl-strict-slurm-export-file-v3",
    },
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
    "runtime_tools": {
        "bootstrap_sha256sum": {
            "path": bootstrap_sha256sum_path,
            "sha256": bootstrap_sha256sum_sha256,
        },
        "document": runtime_tool_document,
        "manifest": {
            "path": runtime_tool_manifest_path,
            "sha256": runtime_tool_manifest_sha256,
        },
    },
    "source": {
        "arm_wrapper_sha256": arm_wrapper_sha256,
        "bridge": {
            "head": bridge_head,
            "root": bridge_root,
            "tree": bridge_tree,
        },
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
        "mcore": {
            "head": mcore_head,
            "root": mcore_root,
            "tree": mcore_tree,
        },
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
campaign_bytes = json.dumps(
    manifest["campaign"],
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
reward_and_advantage_bytes = json.dumps(
    manifest["campaign"]["reward_and_advantage"],
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
if b"\n" in campaign_bytes or b"\n" in reward_and_advantage_bytes:
    raise SystemExit("canonical campaign JSON must not contain LF")
manifest["pair_campaign_sha256"] = hashlib.sha256(campaign_bytes).hexdigest()
manifest["pair_campaign_reward_and_advantage_sha256"] = hashlib.sha256(
    reward_and_advantage_bytes
).hexdigest()
path = pathlib.Path(output)
with path.open("x", encoding="ascii", newline="\n") as stream:
    json.dump(
        manifest,
        stream,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream.write("\n")
PY
}

strict_pair_publish_manifest() {
  local candidate
  local candidate_sha256

  candidate="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.PAIR_MANIFEST.json.candidate.XXXXXX")"
  # The renderer requires exclusive creation, so remove mktemp's placeholder.
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  strict_pair_render_manifest "${candidate}"
  "${STRICT_PAIR_TOOL_CHMOD}" 400 "${candidate}"
  candidate_sha256="$(strict_pair_sha256_file "${candidate}")"

  if "${STRICT_PAIR_TOOL_LN}" -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}" 2>/dev/null; then
    "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  else
    strict_pair_require_canonical_file "${STRICT_PAIR_MANIFEST_PATH}" "PAIR_MANIFEST.json"
    strict_pair_require_mode "${STRICT_PAIR_MANIFEST_PATH}" "400" "PAIR_MANIFEST.json"
    if ! "${STRICT_PAIR_TOOL_CMP}" -s -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}"; then
      "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
      strict_pair_error "PAIR_MANIFEST.json already exists with different contract bytes."
      return
    fi
    "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
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

  candidate="$("${STRICT_PAIR_TOOL_MKTEMP}" "${RESULTS_DIR}/.PAIR_MANIFEST.json.verify.XXXXXX")"
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
  strict_pair_render_manifest "${candidate}"
  if ! "${STRICT_PAIR_TOOL_CMP}" -s -- "${candidate}" "${STRICT_PAIR_MANIFEST_PATH}"; then
    "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
    strict_pair_error "PAIR_MANIFEST.json bytes differ from recomputed canonical inputs."
    return
  fi
  "${STRICT_PAIR_TOOL_RM}" -- "${candidate}"
}
