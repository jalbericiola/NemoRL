#!/bin/bash -p
set -euo pipefail

readonly STRICT_REPLAY_LINUX_SHA256SUM_PATH="/usr/bin/sha256sum"
readonly STRICT_REPLAY_LINUX_SHA256SUM_SHA256="ffb52ec22da029403b8e2a2ee4e07bc4f111f1ec3c6d8fac2f2b5788891dd825"
readonly STRICT_REPLAY_LINUX_PYTHON_PATH="/cm/local/apps/python312/bin/python3.12"
readonly STRICT_REPLAY_LINUX_PYTHON_SHA256="36bee55d1d2c90ceda25e65038809d69be3d3e6e82c94e5d9ec3b2ec1ccc9faa"
readonly STRICT_REPLAY_LINUX_SRUN_PATH="/cm/local/apps/slurm/25.11/bin/srun"
readonly STRICT_REPLAY_LINUX_SRUN_SHA256="133e439427b10f3fb9b826e73af6adb995fdf3df0e4cbf2be51ce3f8ff8197fd"
readonly STRICT_REPLAY_DARWIN_SHA256SUM_PATH="/sbin/sha256sum"
readonly STRICT_REPLAY_DARWIN_SHA256SUM_SHA256="881f3812ac7be70d99bf635e5322b63f565af66f502be3b464036fef8f927300"
readonly STRICT_REPLAY_DARWIN_PYTHON_PATH="/usr/bin/python3"
readonly STRICT_REPLAY_DARWIN_PYTHON_SHA256="179301dcb41ea78accc3fa0048a7e6f6710d891945a751a34addd622020c1818"

strict_replay_die() {
  echo "ERROR: $*" >&2
  exit 2
}

strict_replay_require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || strict_replay_die "${name} is required"
}

strict_replay_reject_startup_controls() {
  PATH=/usr/bin:/bin:/usr/sbin:/sbin
  IFS=$' \t\n'
  LC_ALL=C
  export PATH IFS LC_ALL
  umask 077
  local name
  while IFS= read -r name; do
    case "${name}" in
      BASH_ENV|ENV|BASHOPTS|SHELLOPTS|BASH_COMPAT|CDPATH|GLOBIGNORE|BASH_XTRACEFD|GIT_*|LD_*|DYLD_*|PYTHON*|BASH_FUNC_*)
        strict_replay_die "forbidden replay startup environment variable: ${name}"
        ;;
      STRICT_REPLAY_*|STRICT_PAIR_BOUND_JOB_ID)
        strict_replay_die "forbidden ambient replay authority: ${name}"
        ;;
    esac
  done < <(builtin compgen -e)
}

strict_replay_bind_executed_script() {
  local invoked="${BASH_SOURCE[0]}"
  [[ "${invoked}" == /* && -f "${invoked}" && ! -L "${invoked}" ]] || \
    strict_replay_die "replay wrapper must be invoked by absolute non-symlink path"
  STRICT_REPLAY_EXECUTED_SCRIPT_PATH="${invoked}"
  export STRICT_REPLAY_EXECUTED_SCRIPT_PATH
}

strict_replay_bootstrap_runtime() {
  local candidate
  local digest
  if [[ "${OSTYPE}" == darwin* ]]; then
    candidate="${STRICT_REPLAY_DARWIN_SHA256SUM_PATH}"
    digest="${STRICT_REPLAY_DARWIN_SHA256SUM_SHA256}"
    STRICT_REPLAY_PYTHON="${STRICT_REPLAY_DARWIN_PYTHON_PATH}"
    STRICT_REPLAY_PYTHON_SHA256="${STRICT_REPLAY_DARWIN_PYTHON_SHA256}"
  else
    candidate="${STRICT_REPLAY_LINUX_SHA256SUM_PATH}"
    digest="${STRICT_REPLAY_LINUX_SHA256SUM_SHA256}"
    STRICT_REPLAY_PYTHON="${STRICT_REPLAY_LINUX_PYTHON_PATH}"
    STRICT_REPLAY_PYTHON_SHA256="${STRICT_REPLAY_LINUX_PYTHON_SHA256}"
  fi
  [[ -f "${candidate}" && ! -L "${candidate}" && -x "${candidate}" && ! -w "${candidate}" ]] || \
    strict_replay_die "bootstrap sha256sum is unavailable"
  local actual
  IFS=' ' read -r actual _ < <("${candidate}" -- "${candidate}")
  [[ "${actual}" == "${digest}" ]] || strict_replay_die "bootstrap sha256sum digest mismatch"
  [[ -f "${STRICT_REPLAY_PYTHON}" && ! -L "${STRICT_REPLAY_PYTHON}" && -x "${STRICT_REPLAY_PYTHON}" ]] || \
    strict_replay_die "bootstrap Python is unavailable"
  IFS=' ' read -r actual _ < <("${candidate}" -- "${STRICT_REPLAY_PYTHON}")
  [[ "${actual}" == "${STRICT_REPLAY_PYTHON_SHA256}" ]] || strict_replay_die "bootstrap Python digest mismatch"
  STRICT_REPLAY_SHA256SUM_PATH="${candidate}"
  STRICT_REPLAY_SHA256SUM_SHA256="${digest}"
  export STRICT_REPLAY_SHA256SUM_PATH STRICT_REPLAY_SHA256SUM_SHA256 STRICT_REPLAY_PYTHON STRICT_REPLAY_PYTHON_SHA256
}

strict_replay_parse_cli() {
  [[ "$#" -eq 20 && "$1" == "--pair-manifest" && "$3" == "--pair-manifest-sha256" && \
     "$5" == "--pair-submission-receipt" && "$7" == "--pair-submission-receipt-sha256" && \
     "$9" == "--off-exit-receipt" && "${11}" == "--off-exit-receipt-sha256" && \
     "${13}" == "--replay-manifest" && "${15}" == "--replay-manifest-sha256" && \
     "${17}" == "--environment" && "${19}" == "--profile-id" ]] || \
    strict_replay_die "expected --pair-manifest PATH --pair-manifest-sha256 SHA256 --pair-submission-receipt PATH --pair-submission-receipt-sha256 SHA256 --off-exit-receipt PATH --off-exit-receipt-sha256 SHA256 --replay-manifest PATH --replay-manifest-sha256 SHA256 --environment ENVIRONMENT --profile-id PROFILE_ID"
  [[ "$2" == /* && "$2" != *$'\n'* && "$2" != *$'\r'* ]] || strict_replay_die "pair manifest path must be one absolute line"
  [[ "$4" =~ ^[0-9a-f]{64}$ ]] || strict_replay_die "pair manifest SHA-256 is malformed"
  [[ "$6" == /* && "$6" != *$'\n'* && "$6" != *$'\r'* ]] || strict_replay_die "pair submission receipt path must be one absolute line"
  [[ "$8" =~ ^[0-9a-f]{64}$ ]] || strict_replay_die "pair submission receipt SHA-256 is malformed"
  [[ "${10}" == /* && "${10}" != *$'\n'* && "${10}" != *$'\r'* ]] || strict_replay_die "OFF EXIT receipt path must be one absolute line"
  [[ "${12}" =~ ^[0-9a-f]{64}$ ]] || strict_replay_die "OFF EXIT receipt SHA-256 is malformed"
  [[ "${14}" == /* && "${14}" != *$'\n'* && "${14}" != *$'\r'* ]] || strict_replay_die "manifest path must be one absolute line"
  [[ "${16}" =~ ^[0-9a-f]{64}$ ]] || strict_replay_die "manifest SHA-256 is malformed"
  case "${18}:${20}" in
    citation:citation-string-match-v1|freeform:freeform-regex-v1) ;;
    *) strict_replay_die "environment/profile pair is not admitted" ;;
  esac
  STRICT_REPLAY_PAIR_MANIFEST_PATH="$2"
  STRICT_REPLAY_PAIR_MANIFEST_SHA256="$4"
  STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_PATH="$6"
  STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_SHA256="$8"
  STRICT_REPLAY_OFF_EXIT_RECEIPT_PATH="${10}"
  STRICT_REPLAY_OFF_EXIT_RECEIPT_SHA256="${12}"
  STRICT_REPLAY_MANIFEST_PATH="${14}"
  STRICT_REPLAY_MANIFEST_SHA256="${16}"
  STRICT_REPLAY_ENVIRONMENT="${18}"
  STRICT_REPLAY_PROFILE_ID="${20}"
  export STRICT_REPLAY_PAIR_MANIFEST_PATH STRICT_REPLAY_PAIR_MANIFEST_SHA256
  export STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_PATH STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_SHA256
  export STRICT_REPLAY_OFF_EXIT_RECEIPT_PATH STRICT_REPLAY_OFF_EXIT_RECEIPT_SHA256
  export STRICT_REPLAY_MANIFEST_PATH STRICT_REPLAY_MANIFEST_SHA256
  export STRICT_REPLAY_ENVIRONMENT STRICT_REPLAY_PROFILE_ID
}

strict_replay_main() {
  strict_replay_reject_startup_controls
  strict_replay_bind_executed_script
  strict_replay_parse_cli "$@"
  strict_replay_require_env SLURM_JOB_ID
  strict_replay_bootstrap_runtime
  exec "${STRICT_REPLAY_PYTHON}" -I -S -B - \
    "${STRICT_REPLAY_EXECUTED_SCRIPT_PATH}" \
    "${STRICT_REPLAY_PAIR_MANIFEST_PATH}" \
    "${STRICT_REPLAY_PAIR_MANIFEST_SHA256}" \
    "${STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_PATH}" \
    "${STRICT_REPLAY_PAIR_SUBMISSION_RECEIPT_SHA256}" \
    "${STRICT_REPLAY_OFF_EXIT_RECEIPT_PATH}" \
    "${STRICT_REPLAY_OFF_EXIT_RECEIPT_SHA256}" \
    "${STRICT_REPLAY_MANIFEST_PATH}" \
    "${STRICT_REPLAY_MANIFEST_SHA256}" \
    "${STRICT_REPLAY_ENVIRONMENT}" \
    "${STRICT_REPLAY_PROFILE_ID}" \
    "${SLURM_JOB_ID}" \
    "${STRICT_REPLAY_SHA256SUM_PATH}" \
    "${STRICT_REPLAY_SHA256SUM_SHA256}" \
    "${STRICT_REPLAY_PYTHON}" \
    "${STRICT_REPLAY_PYTHON_SHA256}" <<'PY'
import hashlib
import importlib
import importlib.util
import json
import math
import os
import posixpath
import re
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Optional

if sys.flags.no_site != 1:
    raise SystemExit("replay wrapper failed: bootstrap Python must run with -S")

DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
MAX_INT63 = (1 << 63) - 1
SNAPSHOT_SHA_MANIFEST = "strict-pair-snapshot-manifest.sha256"
REPLAY_MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
PROFILE_IDS = {
    "citation": "citation-string-match-v1",
    "freeform": "freeform-regex-v1",
}
PAIR_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
PAIR_SUBMISSION_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
PAIR_JOB_RECEIPT_SCHEMA = "nemo-rl-strict-pair-job-receipt-v2"
PROGRAM_PATHS = {
    "entrypoint": "examples/nemo_gym/run_strict_captured_replay_v2.py",
    "evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence_v2.py",
    "gym_child_bootstrap": "nemo_rl/environments/_strict_gym_child_bootstrap_v2/sitecustomize.py",
    "gym_child_runtime": "nemo_rl/environments/strict_gym_child_runtime_v2.py",
    "job_wrapper": "strict_pair_replay_job_wrapper_v2.sh",
    "manifest_utility": "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
    "raw_transport_owner": "nemo_rl/utils/strict_model_transport_replay_v3.py",
    "result_sealer": "nemo_rl/utils/strict_captured_replay_seal_v2.py",
    "runtime": "nemo_rl/algorithms/strict_captured_replay_runtime_v2.py",
    "submission_launcher": "strict_pair_replay_launch_v2.sh",
    "legacy_evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence.py",
    "main_step_ledger": "nemo_rl/utils/strict_main_step_ledger.py",
    "model_transport_utility": "nemo_rl/utils/strict_model_transport.py",
    "profile_registry": "nemo_rl/utils/strict_captured_replay_profiles.py",
}
PROGRAM_NAMES = frozenset(PROGRAM_PATHS)
LINUX_SRUN_PATH = Path("/cm/local/apps/slurm/25.11/bin/srun")
LINUX_SRUN_SHA256 = "133e439427b10f3fb9b826e73af6adb995fdf3df0e4cbf2be51ce3f8ff8197fd"


def fail(message: str) -> None:
    raise SystemExit(f"replay wrapper failed: {message}")


def reject_constant(value: str) -> Any:
    fail(f"non-finite JSON constant is forbidden: {value}")


def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON member is forbidden: {key}")
        result[key] = value
    return result


def parse_integer(value: str) -> int:
    if value == "-0":
        fail("scheduler raw JSON contains negative zero")
    return int(value)


def parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed == 0.0 and value.startswith("-")):
        fail("scheduler raw JSON contains non-finite/negative-zero float")
    return parsed


def reject_negative_zero(value: Any, *, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0.0):
            fail(f"{label} contains invalid floating value")
    elif isinstance(value, dict):
        for key, member in value.items():
            reject_negative_zero(member, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            reject_negative_zero(member, label=f"{label}[{index}]")


def canonical_absolute_path(value: Any, label: str) -> Path:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or posixpath.normpath(value) != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        fail(f"{label} is not one canonical absolute path")
    path = Path(value)
    cursor = Path("/")
    for component in path.parts[1:]:
        cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            fail(f"cannot inspect {label} component {cursor}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} contains a symlink component")
    return path


def _read_stable(path: Path, *, label: str, exact_mode: Optional[int], max_bytes: int, owner: str, allow_empty: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
    except OSError as error:
        fail(f"cannot stat {label}: {error}")
    mode = stat.S_IMODE(before.st_mode)
    if not stat.S_ISREG(before.st_mode):
        fail(f"{label} is not one immutable regular file")
    if owner == "evidence":
        if before.st_nlink != 1:
            fail(f"{label} is not one immutable regular file")
        if before.st_uid != os.geteuid():
            fail(f"{label} owner differs from EUID-owned evidence contract")
    elif owner == "host_file":
        if before.st_uid not in {0, os.geteuid()}:
            fail(f"{label} owner differs from authenticated host-file policy")
        if before.st_uid == 0:
            if mode & 0o022:
                fail(f"{label} must not be group/other writable")
        elif mode & 0o222:
            fail(f"{label} must not be writable")
    elif owner == "tool":
        if before.st_uid not in {0, os.geteuid()}:
            fail(f"{label} owner differs from trusted system-tool policy")
        if not mode & 0o111:
            fail(f"{label} must be executable")
        if before.st_uid == 0:
            if mode & 0o022:
                fail(f"{label} must not be group/other writable")
        elif mode & 0o222:
            fail(f"{label} must not be writable")
    else:
        fail(f"unknown reader owner policy for {label}")
    if exact_mode is not None and mode != exact_mode:
        fail(f"{label} mode differs from exact contract")
    if exact_mode is None and owner == "evidence" and mode & 0o222:
        fail(f"{label} must be nonwritable")
    minimum = 0 if allow_empty else 1
    if not minimum <= before.st_size <= max_bytes:
        fail(f"{label} size differs from admitted bounds")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        fail(f"cannot open {label}: {error}")
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            fail(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                fail(f"{label} truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            fail(f"{label} grew while reading")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = path.lstat()
    except OSError as error:
        fail(f"cannot restat {label}: {error}")
    fingerprint = lambda metadata: (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if not (fingerprint(before) == fingerprint(opened) == fingerprint(after) == fingerprint(named)):
        fail(f"{label} changed during stable read")
    return b"".join(chunks)


def stable_evidence_bytes(path: Path, *, label: str, exact_mode: Optional[int] = 0o400, max_bytes: int = 64 * 1024 * 1024, allow_empty: bool = False) -> bytes:
    return _read_stable(path, label=label, exact_mode=exact_mode, max_bytes=max_bytes, owner="evidence", allow_empty=allow_empty)


def stable_tool_bytes(path: Path, *, label: str, expected_sha256: str) -> bytes:
    if DIGEST_RE.fullmatch(expected_sha256) is None:
        fail(f"{label} digest is malformed")
    raw = _read_stable(path, label=label, exact_mode=None, max_bytes=64 * 1024 * 1024, owner="tool")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        fail(f"{label} digest differs from authenticated authority")
    return raw


def stable_authenticated_host_file_bytes(path: Path, *, label: str, expected_sha256: str) -> bytes:
    if DIGEST_RE.fullmatch(expected_sha256) is None:
        fail(f"{label} digest is malformed")
    raw = _read_stable(path, label=label, exact_mode=None, max_bytes=64 * 1024 * 1024, owner="host_file")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        fail(f"{label} digest differs from authenticated authority")
    return raw


def stable_container_image_sha256(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_owner_uid: int,
    expected_owner_gid: int,
) -> str:
    """Stream-authenticate the explicitly trusted foreign-owned container image."""
    if DIGEST_RE.fullmatch(expected_sha256) is None:
        fail(f"{label} digest is malformed")
    if (
        type(expected_owner_uid) is not int
        or expected_owner_uid != 153493
        or type(expected_owner_gid) is not int
        or expected_owner_gid != 30
    ):
        fail(f"{label} named publisher identity is not admitted")
    effective_uid = os.geteuid()
    if effective_uid == 0 or effective_uid == expected_owner_uid:
        fail(f"{label} publisher must be foreign to the replay process")
    path = canonical_absolute_path(str(path), f"{label} path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
    except OSError as error:
        fail(f"cannot stat {label}: {error}")
    mode = stat.S_IMODE(before.st_mode)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != expected_owner_uid
        or before.st_gid != expected_owner_gid
    ):
        fail(f"{label} is not an authenticated regular host file")
    if before.st_nlink != 1:
        fail(f"{label} must have exactly one hard link")
    if mode & 0o022:
        fail(f"{label} is group/other writable")
    try:
        effectively_writable = os.access(path, os.W_OK, effective_ids=True)
    except (OSError, TypeError) as error:
        fail(f"cannot determine effective {label} write access: {error}")
    if effectively_writable is not False:
        fail(f"{label} is writable by the replay process")
    if not 1 <= before.st_size <= 1 << 50:
        fail(f"{label} size differs from admitted bounds")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        fail(f"cannot open {label}: {error}")
    fingerprint = lambda metadata: (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    try:
        opened = os.fstat(descriptor)
        if fingerprint(before) != fingerprint(opened):
            fail(f"{label} changed while opening")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 4 << 20))
            if not chunk:
                fail(f"{label} truncated while hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            fail(f"{label} grew while hashing")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = os.lstat(path)
    except OSError as error:
        fail(f"cannot restat {label}: {error}")
    if not (
        fingerprint(before)
        == fingerprint(opened)
        == fingerprint(after)
        == fingerprint(named)
    ):
        fail(f"{label} changed during stable hash")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        fail(f"{label} digest differs from authenticated authority")
    return actual_sha256


def canonical_document(path: Path, *, label: str, trailing_lf: bool, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    raw = stable_evidence_bytes(path, label=label, exact_mode=0o400)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha256:
        fail(f"{label} SHA-256 differs")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_int=parse_integer,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not canonical ASCII JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{label} root is not an object")
    reject_negative_zero(value, label=label)
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    if trailing_lf:
        encoded += b"\n"
    if raw != encoded:
        fail(f"{label} bytes are not canonical")
    return value, raw


def parse_snapshot_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        fail("snapshot SHA manifest framing differs")
    result: dict[str, str] = {}
    for index, line in enumerate(raw[:-1].split(b"\n")):
        digest_raw, separator, relative_raw = line.partition(b"  ")
        if not separator or not relative_raw:
            fail(f"snapshot SHA manifest line {index} is malformed")
        try:
            digest = digest_raw.decode("ascii")
            relative = relative_raw.decode("ascii")
        except UnicodeDecodeError as error:
            fail(f"snapshot SHA manifest line {index} is not ASCII: {error}")
        if DIGEST_RE.fullmatch(digest) is None:
            fail(f"snapshot SHA manifest line {index} digest differs")
        if (
            not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(character in relative for character in ("\x00", "\n", "\r"))
            or posixpath.normpath(relative) != relative
            or relative == "."
            or ".." in relative.split("/")
        ):
            fail(f"snapshot SHA manifest line {index} path is unsafe")
        if relative in result:
            fail(f"snapshot SHA manifest repeats {relative}")
        result[relative] = digest
    if not result or SNAPSHOT_SHA_MANIFEST in result:
        fail("snapshot SHA manifest inventory is empty or self-referential")
    return result


def required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    return value


def authenticated_program(
    snapshot_root: Path,
    snapshot_manifest: dict[str, str],
    *,
    name: str,
    require_executable: bool,
) -> tuple[Path, str, bytes]:
    relative = PROGRAM_PATHS[name]
    digest = snapshot_manifest.get(relative)
    if digest is None:
        fail(f"snapshot SHA manifest does not bind replay program {name}")
    path = snapshot_root / relative
    raw = stable_evidence_bytes(path, label=f"replay program {name}", exact_mode=None)
    if hashlib.sha256(raw).hexdigest() != digest:
        fail(f"replay program {name} bytes differ")
    if require_executable:
        try:
            mode = stat.S_IMODE(path.lstat().st_mode)
        except OSError as error:
            fail(f"cannot stat replay program {name}: {error}")
        if not mode & 0o111:
            fail(f"replay program {name} is not executable")
    return path, digest, raw


def authenticate_program_closure(
    snapshot_root: Path,
    snapshot_manifest: dict[str, str],
    program: dict[str, Any],
    *,
    executable_name: str,
) -> dict[str, tuple[Path, str, bytes]]:
    """Stable-read the exact manifest-v4 program before any repo import."""
    if set(program) != PROGRAM_NAMES:
        fail("replay program keyset differs")
    authenticated: dict[str, tuple[Path, str, bytes]] = {}
    for name, relative in PROGRAM_PATHS.items():
        reference = required_mapping(program.get(name), f"program {name}")
        snapshot_sha256 = snapshot_manifest.get(relative)
        if snapshot_sha256 is None or reference != {
            "path": relative,
            "sha256": snapshot_sha256,
        }:
            fail(f"replay program {name} differs from authenticated Pair ON snapshot")
        authenticated[name] = authenticated_program(
            snapshot_root,
            snapshot_manifest,
            name=name,
            require_executable=name == executable_name,
        )
    return authenticated


def load_pair_bootstrap_authority(
    *,
    pair_manifest_path: Path,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: Path,
    pair_submission_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    pair_manifest, _ = canonical_document(
        pair_manifest_path,
        label="Pair manifest",
        trailing_lf=True,
        expected_sha256=pair_manifest_sha256,
    )
    if pair_manifest.get("schema") != PAIR_SCHEMA:
        fail("Pair manifest schema differs")
    pair_paths = required_mapping(pair_manifest.get("paths"), "Pair paths")
    results_root = canonical_absolute_path(
        pair_paths.get("results_root"), "Pair results root"
    )
    if pair_submission_receipt_path != results_root / "PAIR_SUBMISSION_RECEIPT.json":
        fail("Pair submission receipt path differs from Pair results root")
    pair_submission, _ = canonical_document(
        pair_submission_receipt_path,
        label="Pair submission receipt",
        trailing_lf=True,
        expected_sha256=pair_submission_receipt_sha256,
    )
    if (
        pair_submission.get("schema") != PAIR_SUBMISSION_SCHEMA
        or pair_submission.get("outcome") != "released"
        or pair_submission.get("stage") != "complete"
    ):
        fail("Pair submission receipt is not one released terminal receipt")
    pair_binding = required_mapping(pair_submission.get("pair"), "Pair submission pair")
    if (
        pair_binding.get("id") != pair_manifest.get("pair_id")
        or pair_binding.get("manifest") != {
            "path": str(pair_manifest_path),
            "sha256": pair_manifest_sha256,
        }
    ):
        fail("Pair submission receipt pair binding differs")
    if required_mapping(
        pair_submission.get("receipt"), "Pair submission receipt self-reference"
    ) != {"path": str(pair_submission_receipt_path), "schema": PAIR_SUBMISSION_SCHEMA}:
        fail("Pair submission receipt self-reference differs")
    source = required_mapping(pair_manifest.get("source"), "Pair source")
    snapshots = required_mapping(source.get("snapshots"), "Pair source snapshots")
    on_snapshot = required_mapping(snapshots.get("on"), "Pair ON snapshot")
    if set(on_snapshot) != {"config_sha256", "entrypoint_sha256", "manifest_sha256", "path"}:
        fail("Pair ON snapshot reference differs")
    snapshot_root = canonical_absolute_path(on_snapshot["path"], "Pair ON snapshot root")
    snapshot_manifest_path = snapshot_root / SNAPSHOT_SHA_MANIFEST
    snapshot_manifest_raw = stable_evidence_bytes(
        snapshot_manifest_path, label="Pair ON snapshot SHA manifest"
    )
    if hashlib.sha256(snapshot_manifest_raw).hexdigest() != on_snapshot["manifest_sha256"]:
        fail("Pair ON snapshot SHA manifest digest differs")
    snapshot_manifest = parse_snapshot_manifest(snapshot_manifest_raw)
    return pair_manifest, pair_submission, on_snapshot, snapshot_root, snapshot_manifest


def require_pair_bootstrap_runtime_join(
    pair_manifest: dict[str, Any],
    *,
    sha_tool_path: Path,
    sha_tool_sha256: str,
    host_python_path: Path,
    host_python_sha256: str,
) -> None:
    runtime_tools = required_mapping(
        pair_manifest.get("runtime_tools"), "Pair runtime tools"
    )
    bootstrap_sha256sum = required_mapping(
        runtime_tools.get("bootstrap_sha256sum"),
        "Pair bootstrap sha256sum",
    )
    document = required_mapping(
        runtime_tools.get("document"), "Pair runtime tool document"
    )
    host = required_mapping(document.get("host"), "Pair host runtime tools")
    host_python = required_mapping(host.get("python"), "Pair host Python")
    if bootstrap_sha256sum != {
        "path": str(sha_tool_path),
        "sha256": sha_tool_sha256,
    }:
        fail("bootstrap sha256sum differs from authenticated Pair runtime tools")
    if host_python != {
        "path": str(host_python_path),
        "sha256": host_python_sha256,
    }:
        fail("bootstrap Python differs from authenticated Pair runtime tools")


def ensure_authenticated_package(name: str, root: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if isinstance(module, types.ModuleType):
        return module
    package_root = root / Path(*name.split("."))
    module = types.ModuleType(name)
    module.__file__ = str(package_root / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(package_root)]
    sys.modules[name] = module
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = ensure_authenticated_package(parent_name, root)
        setattr(parent, leaf, module)
    return module


def import_authenticated_module(name: str, path: Path, *, raw: bytes):
    parent_name, _, leaf = name.rpartition(".")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = parent_name
    if path.name == "__init__.py":
        module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    if parent_name:
        parent = sys.modules.get(parent_name)
        if isinstance(parent, types.ModuleType):
            setattr(parent, leaf, module)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def require_absent(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        fail(f"cannot inspect {label}: {error}")
    fail(f"{label} must be absent before replay compute execution")


def ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        fail(f"cannot create directory {path}: {error}")
    try:
        path.chmod(0o700)
    except OSError as error:
        fail(f"cannot chmod directory {path}: {error}")
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.geteuid():
        fail(f"directory {path} differs from private mode-0700 contract")


def write_exclusive(path: Path, payload: bytes, *, mode: int) -> str:
    ensure_private_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError:
        fail(f"exclusive output already exists: {path}")
    except OSError as error:
        fail(f"cannot create {path}: {error}")
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    actual = stable_evidence_bytes(path, label=str(path), exact_mode=mode, max_bytes=max(1, len(payload)), allow_empty=len(payload) == 0)
    if actual != payload:
        fail(f"published bytes differ for {path}")
    return hashlib.sha256(payload).hexdigest()


def parse_job_id(text: str, *, label: str) -> str:
    if JOB_ID_RE.fullmatch(text) is None or int(text) > MAX_INT63:
        fail(f"{label} is not a canonical positive int63")
    return text


def normalize_scheduler_query(raw: bytes, *, phase: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        fail("scheduler raw output framing differs")
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_int=parse_integer,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"scheduler raw output is not strict JSON: {error}")
    if not isinstance(parsed, dict) or set(parsed) != {"errors", "jobs", "last_backfill", "last_update", "meta", "warnings"}:
        fail("scheduler raw JSON root differs")
    if (
        type(parsed["last_backfill"]) is not dict
        or type(parsed["last_update"]) is not dict
        or type(parsed["meta"]) is not dict
    ):
        fail("scheduler raw JSON root scalar/object fields differ")
    if parsed["errors"] != [] or parsed["warnings"] != []:
        fail("scheduler raw JSON errors/warnings must be exact empty lists")
    jobs = parsed["jobs"]
    if type(jobs) is not list or len(jobs) != 1 or type(jobs[0]) is not dict:
        fail("scheduler raw JSON must contain exactly one job object")
    source = jobs[0]
    required = {"job_id", "name", "comment", "current_working_directory", "state_reason", "user_id", "job_state", "restart_cnt", "hold"}
    if required - set(source):
        fail("scheduler raw JSON job is missing required fields")
    job_id = source["job_id"]
    user_id = source["user_id"]
    restart_count = source["restart_cnt"]
    states = source["job_state"]
    if type(job_id) is not int or not 1 <= job_id <= MAX_INT63:
        fail("scheduler raw job_id differs")
    if type(user_id) is not int or not 0 <= user_id <= (1 << 31) - 1:
        fail("scheduler raw user_id differs")
    if type(restart_count) is not int or restart_count != 0:
        fail("scheduler raw restart count differs")
    if type(source["hold"]) is not bool:
        fail("scheduler raw hold flag differs")
    if type(states) is not list or len(states) != 1 or type(states[0]) is not str:
        fail("scheduler raw job_state differs")
    def ascii_value(value: Any, *, name: str, maximum: int) -> str:
        if type(value) is not str or not value:
            fail(f"scheduler raw {name} must be a nonempty ASCII string")
        try:
            payload = value.encode("ascii")
        except UnicodeEncodeError:
            fail(f"scheduler raw {name} must be ASCII")
        if len(payload) > maximum:
            fail(f"scheduler raw {name} exceeds its bound")
        if any(byte < 0x20 or byte == 0x7F for byte in payload):
            fail(f"scheduler raw {name} contains a control character")
        return value
    job_name = ascii_value(source["name"], name="name", maximum=255)
    comment = ascii_value(source["comment"], name="comment", maximum=4096)
    work_dir = ascii_value(
        source["current_working_directory"],
        name="current_working_directory",
        maximum=4096,
    )
    job_state = ascii_value(states[0], name="job_state", maximum=64)
    reason = ascii_value(source["state_reason"], name="state_reason", maximum=4096)
    record = {
        "job_id": str(job_id),
        "job_name": job_name,
        "comment": comment,
        "user_id": str(user_id),
        "work_dir": str(
            canonical_absolute_path(
                work_dir,
                "scheduler current_working_directory",
            )
        ),
        "job_state": job_state,
        "reason": reason,
        "held": source["hold"],
        "restart_count": restart_count,
    }
    if phase == "PRE_RELEASE":
        if record["job_state"] != "PENDING" or record["reason"] != "JobHeldUser" or record["held"] is not True:
            fail("PRE_RELEASE scheduler record differs")
    else:
        if record["held"] is not False or record["reason"] == "JobHeldUser":
            fail(f"{phase} scheduler hold/reason differs")
    return record


def verify_scheduler_client_environment(client_environment: dict[str, Any]) -> None:
    slurm_conf = client_environment["variables"]["SLURM_CONF"]
    stable_authenticated_host_file_bytes(
        canonical_absolute_path(slurm_conf["path"], "authenticated SLURM_CONF path"),
        label="authenticated SLURM_CONF",
        expected_sha256=slurm_conf["sha256"],
    )


def scheduler_env(client_environment: dict[str, Any]) -> dict[str, str]:
    slurm_conf = client_environment["variables"]["SLURM_CONF"]
    return {"LC_ALL": "C", "SLURM_CONF": slurm_conf["path"]}


def run_scheduler_command(
    argv: list[str],
    *,
    env: dict[str, str],
    client_environment: dict[str, Any],
    label: str,
) -> subprocess.CompletedProcess[bytes]:
    verify_scheduler_client_environment(client_environment)
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env=env,
        cwd=str(snapshot_root),
    )
    if completed.returncode != 0:
        fail(f"{label} exited {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}")
    if completed.stderr not in {b"", None}:
        fail(f"{label} wrote stderr")
    return completed


def build_driver_env(
    manifest: dict[str, Any],
    *,
    authenticated_source: Any,
    job_id: str,
    scheduler_device_environment: dict[str, Any],
) -> dict[str, str]:
    _, _, export_records = manifest_utility._load_replay_slurm_export(
        source=authenticated_source,
        attempt_id=manifest["attempt_id"],
    )
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    for name, value in export_records:
        env[name] = value.decode("ascii")
    env["STRICT_PAIR_BOUND_JOB_ID"] = job_id
    env["CUDA_VISIBLE_DEVICES"] = scheduler_device_environment["cuda_visible_devices"]
    for name in (
        "gpu_device_ordinal",
        "nvidia_visible_devices",
        "rocr_visible_devices",
        "ze_affinity_mask",
    ):
        value = scheduler_device_environment[name]
        if value is not None:
            env[name.upper()] = value
    return env


def build_driver_argv(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    pre_receipt_path: Path,
    pre_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> list[str]:
    if (
        manifest.get("environment") != expected_environment
        or required_mapping(
            manifest.get("scorer_profile"), "driver scorer_profile"
        ).get("profile_id")
        != expected_profile_id
    ):
        fail("driver environment/profile differs from authenticated manifest")
    entrypoint = snapshot_root / program["entrypoint"]["path"]
    return [
        str(entrypoint),
        "--replay-driver-phase",
        "--replay-manifest",
        str(manifest_path),
        "--replay-manifest-sha256",
        manifest_sha256,
        "--pre-receipt",
        str(pre_receipt_path),
        "--pre-receipt-sha256",
        pre_receipt_sha256,
        "--environment",
        expected_environment,
        "--profile-id",
        expected_profile_id,
    ]


def container_mounts(*, manifest: dict[str, Any], pair_manifest: dict[str, Any]) -> str:
    pair_paths = pair_manifest["paths"]
    mounts = [
        f"{snapshot_root}:{snapshot_root}:ro",
        f"{snapshot_root}:/opt/nemo-rl:ro",
        f"{pair_paths['results_root']}:{pair_paths['results_root']}",
        f"{pair_paths['cache_root']}:{pair_paths['cache_root']}",
        f"{pair_paths['hf_home']}:{pair_paths['hf_home']}",
    ]
    unique = list(dict.fromkeys(mounts))
    return ",".join(unique)


def run_driver(*, manifest: dict[str, Any], pair_manifest: dict[str, Any], pre_receipt_path: Path, pre_receipt_sha256: str, job: dict[str, Any], scheduler_device_environment: dict[str, Any], client_environment: dict[str, Any], submission_env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    driver_env = build_driver_env(
        manifest,
        authenticated_source=authenticated_source,
        job_id=live_job_id,
        scheduler_device_environment=scheduler_device_environment,
    )
    entrypoint_argv = build_driver_argv(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        pre_receipt_path=pre_receipt_path,
        pre_receipt_sha256=pre_receipt_sha256,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if sys.platform == "darwin":
        argv = [str(host_python_path), "-I", "-B", *entrypoint_argv]
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            cwd=str(snapshot_root),
            env=driver_env,
        )
    stable_tool_bytes(LINUX_SRUN_PATH, label="srun", expected_sha256=LINUX_SRUN_SHA256)
    container_python = manifest["runtime_tools"]["document"]["container"]["python"]["path"]
    container_reference = manifest["replay_contract"]["gym_scorer"]["container"]
    container_image = canonical_absolute_path(
        container_reference["path"], "authenticated replay container image"
    )
    argv = [
        str(LINUX_SRUN_PATH),
        "--jobid",
        live_job_id,
        "-A",
        job["account"],
        "-p",
        job["partition"],
        "--nodes=1",
        "--ntasks=1",
        "--no-container-mount-home",
        f"--container-image={container_image}",
        f"--container-mounts={container_mounts(manifest=manifest, pair_manifest=pair_manifest)}",
        f"--container-workdir={snapshot_root}",
        "/usr/bin/env",
        "-i",
        *(f"{name}={value}" for name, value in sorted(driver_env.items())),
        container_python,
        "-I",
        "-B",
        *entrypoint_argv,
    ]
    stable_container_image_sha256(
        container_image,
        label="authenticated replay container image before srun",
        expected_sha256=container_reference["sha256"],
        expected_owner_uid=container_reference["owner_uid"],
        expected_owner_gid=container_reference["owner_gid"],
    )
    verify_scheduler_client_environment(client_environment)
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1800,
        cwd=str(snapshot_root),
        env=submission_env,
    )
    stable_container_image_sha256(
        container_image,
        label="authenticated replay container image after srun",
        expected_sha256=container_reference["sha256"],
        expected_owner_uid=container_reference["owner_uid"],
        expected_owner_gid=container_reference["owner_gid"],
    )
    return completed


def observed_hardware() -> dict[str, Any]:
    tool_ref = manifest["runtime_tools"]["document"]["host"]["nvidia_smi"]
    tool_path = canonical_absolute_path(
        tool_ref["path"], "authenticated nvidia-smi path"
    )
    tool_sha = tool_ref["sha256"]
    stable_tool_bytes(
        tool_path,
        label="authenticated nvidia-smi",
        expected_sha256=tool_sha,
    )
    result = subprocess.run(
        [str(tool_path), "--query-gpu=name,driver_version", "--format=csv,noheader"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    if result.returncode != 0 or result.stderr not in {b"", None}:
        fail("nvidia-smi query failed")
    try:
        raw_output = result.stdout.decode("ascii")
    except UnicodeDecodeError:
        fail("nvidia-smi output is not ASCII")
    if (
        "\r" in raw_output
        or not raw_output.endswith("\n")
        or raw_output.endswith("\n\n")
    ):
        fail("nvidia-smi output framing differs")
    lines = raw_output[:-1].split("\n")
    if len(lines) != 4 or any(not line or line != line.strip() for line in lines):
        fail("nvidia-smi must report exactly 4 GPU rows")
    ordered_rows = []
    for index, line in enumerate(lines):
        fields = [token.strip() for token in line.split(",")]
        if len(fields) != 2:
            fail("nvidia-smi GPU row shape differs")
        model, driver_version = fields
        if line != f"{model}, {driver_version}":
            fail("nvidia-smi GPU row formatting differs")
        if model != "NVIDIA GB200" or driver_version != "580.126.20":
            fail("compute hardware differs from required NVIDIA GB200 / 580.126.20")
        ordered_rows.append(
            {
                "index": index,
                "raw": line,
                "gpu_model": model,
                "driver_version": driver_version,
            }
        )
    return {
        "schema": "nemo-rl-strict-hardware-observation-v2",
        "gpu_model": "NVIDIA GB200",
        "driver_version": "580.126.20",
        "gpu_row_count": len(ordered_rows),
        "ordered_rows": ordered_rows,
        "raw_output_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "ordered_rows_sha256": evidence_utility.domain_sha256(
            "captured-replay-nvidia-smi-ordered-rows-v1", ordered_rows
        ),
        "nvidia_smi": {"path": str(tool_path), "sha256": tool_sha},
    }


def observed_scheduler_device_environment() -> dict[str, Any]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None or not cuda_visible.strip():
        fail("CUDA_VISIBLE_DEVICES is required for replay device attestation")
    gpu_ordinal = os.environ.get("GPU_DEVICE_ORDINAL")
    return {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": cuda_visible,
        "gpu_device_ordinal": gpu_ordinal,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
    }


def output_reference(name: str) -> dict[str, str]:
    declaration = manifest["artifacts"]["outputs"][name]
    path = canonical_absolute_path(declaration["path"], f"{name} output path")
    raw = stable_evidence_bytes(path, label=f"{name} output", exact_mode=0o400, max_bytes=256 * 1024 * 1024)
    return {"path": str(path), "schema": declaration["schema"], "sha256": hashlib.sha256(raw).hexdigest()}


if len(sys.argv) != 17:
    fail("bootstrap argument count differs from the V2 contract")
executed_script = canonical_absolute_path(sys.argv[1], "executed wrapper path")
pair_manifest_path = canonical_absolute_path(sys.argv[2], "Pair manifest path")
pair_manifest_sha256 = sys.argv[3]
pair_submission_receipt_path = canonical_absolute_path(
    sys.argv[4], "Pair submission receipt path"
)
pair_submission_receipt_sha256 = sys.argv[5]
trusted_off_exit_receipt_path = canonical_absolute_path(
    sys.argv[6], "trusted OFF EXIT receipt path"
)
trusted_off_exit_receipt_sha256 = sys.argv[7]
manifest_path = canonical_absolute_path(sys.argv[8], "replay execution manifest path")
manifest_sha256 = sys.argv[9]
expected_environment = sys.argv[10]
expected_profile_id = sys.argv[11]
if PROFILE_IDS.get(expected_environment) != expected_profile_id:
    fail("explicit environment/profile pair is not admitted")
live_job_id = parse_job_id(sys.argv[12], label="SLURM_JOB_ID")
sha_tool_path = canonical_absolute_path(sys.argv[13], "bootstrap sha256sum path")
sha_tool_sha256 = sys.argv[14]
host_python_path = canonical_absolute_path(sys.argv[15], "bootstrap Python path")
host_python_sha256 = sys.argv[16]
if DIGEST_RE.fullmatch(pair_manifest_sha256) is None:
    fail("Pair manifest SHA-256 argument is malformed")
if DIGEST_RE.fullmatch(pair_submission_receipt_sha256) is None:
    fail("Pair submission receipt SHA-256 argument is malformed")
if DIGEST_RE.fullmatch(trusted_off_exit_receipt_sha256) is None:
    fail("trusted OFF EXIT receipt SHA-256 argument is malformed")
if DIGEST_RE.fullmatch(manifest_sha256) is None:
    fail("manifest SHA-256 argument is malformed")
stable_tool_bytes(sha_tool_path, label="bootstrap sha256sum", expected_sha256=sha_tool_sha256)
stable_tool_bytes(host_python_path, label="bootstrap Python", expected_sha256=host_python_sha256)
(
    pair_manifest,
    pair_submission_receipt,
    pair_on_snapshot,
    snapshot_root,
    snapshot_manifest,
) = load_pair_bootstrap_authority(
    pair_manifest_path=pair_manifest_path,
    pair_manifest_sha256=pair_manifest_sha256,
    pair_submission_receipt_path=pair_submission_receipt_path,
    pair_submission_receipt_sha256=pair_submission_receipt_sha256,
)
require_pair_bootstrap_runtime_join(
    pair_manifest,
    sha_tool_path=sha_tool_path,
    sha_tool_sha256=sha_tool_sha256,
    host_python_path=host_python_path,
    host_python_sha256=host_python_sha256,
)
manifest, _ = canonical_document(
    manifest_path,
    label="replay execution manifest",
    trailing_lf=False,
    expected_sha256=manifest_sha256,
)
if manifest.get("schema") != REPLAY_MANIFEST_SCHEMA:
    fail("replay execution manifest is not manifest-v4")
if manifest["pair"]["manifest"] != {
    "path": str(pair_manifest_path),
    "schema": PAIR_SCHEMA,
    "sha256": pair_manifest_sha256,
}:
    fail("replay manifest Pair manifest binding differs from CLI authority")
if manifest["pair"]["submission_receipt"] != {
    "path": str(pair_submission_receipt_path),
    "schema": PAIR_SUBMISSION_SCHEMA,
    "sha256": pair_submission_receipt_sha256,
}:
    fail("replay manifest Pair submission binding differs from CLI authority")
if manifest["source_capture"]["job_receipts"]["exit"] != {
    "path": str(trusted_off_exit_receipt_path),
    "schema": PAIR_JOB_RECEIPT_SCHEMA,
    "sha256": trusted_off_exit_receipt_sha256,
}:
    fail("replay manifest trusted OFF EXIT binding differs from CLI authority")
if manifest["replay_contract"]["source_snapshot"] != {"arm": "on", "ref": pair_on_snapshot}:
    fail("replay manifest source snapshot differs from authenticated Pair ON")
selection = required_mapping(pair_manifest.get("selection"), "Pair selection")
if selection.get("environment") != expected_environment:
    fail("explicit environment differs from Pair authority")
scorer_profile = required_mapping(manifest.get("scorer_profile"), "scorer_profile")
if (
    manifest.get("environment") != expected_environment
    or scorer_profile.get("environment") != expected_environment
    or scorer_profile.get("profile_id") != expected_profile_id
):
    fail("replay manifest scorer profile identity differs from Pair authority")
program = required_mapping(
    required_mapping(manifest.get("replay_contract"), "replay_contract").get("program"),
    "replay program",
)
authenticated_programs = authenticate_program_closure(
    snapshot_root,
    snapshot_manifest,
    program,
    executable_name="job_wrapper",
)
wrapper_path, wrapper_sha256, wrapper_raw = authenticated_programs["job_wrapper"]
if executed_script != wrapper_path:
    fail("executed wrapper path differs from authenticated Pair ON wrapper")
if hashlib.sha256(stable_evidence_bytes(executed_script, label="executed wrapper bytes", exact_mode=None)).hexdigest() != wrapper_sha256:
    fail("executed wrapper bytes differ from authenticated Pair ON wrapper")
runner_path, _, runner_raw = authenticated_programs["entrypoint"]
manifest_utility_path, _, manifest_utility_raw = authenticated_programs[
    "manifest_utility"
]
evidence_utility_path, _, evidence_utility_raw = authenticated_programs[
    "evidence_utility"
]
result_sealer_path, _, result_sealer_raw = authenticated_programs["result_sealer"]
legacy_evidence_path, _, legacy_evidence_raw = authenticated_programs[
    "legacy_evidence_utility"
]
profile_registry_path, _, profile_registry_raw = authenticated_programs[
    "profile_registry"
]
runner = import_authenticated_module(
    "_strict_replay_runner_bootstrap", runner_path, raw=runner_raw
)
runner._authenticate_snapshot_program(
    manifest=manifest,
    manifest_sha256=manifest_sha256,
    pair_manifest=pair_manifest,
    execution_source_root=snapshot_root,
)
runner._activate_authenticated_source_roots(snapshot_root)
ensure_authenticated_package("nemo_rl", snapshot_root)
ensure_authenticated_package("nemo_rl.utils", snapshot_root)
import_authenticated_module(
    "nemo_rl.utils.strict_captured_replay_evidence",
    legacy_evidence_path,
    raw=legacy_evidence_raw,
)
import_authenticated_module(
    "nemo_rl.utils.strict_captured_replay_profiles",
    profile_registry_path,
    raw=profile_registry_raw,
)
evidence_utility = import_authenticated_module(
    "nemo_rl.utils.strict_captured_replay_evidence_v2",
    evidence_utility_path,
    raw=evidence_utility_raw,
)
manifest_utility = import_authenticated_module(
    "nemo_rl.utils.strict_captured_replay_manifest_v2",
    manifest_utility_path,
    raw=manifest_utility_raw,
)
result_sealer = import_authenticated_module(
    "nemo_rl.utils.strict_captured_replay_seal_v2",
    result_sealer_path,
    raw=result_sealer_raw,
)
runner._verify_imported_program_modules(execution_source_root=snapshot_root, program=program)
authenticated_source = manifest_utility.load_authenticated_off_source_capture(
    pair_manifest=pair_manifest,
    pair_manifest_path=str(pair_manifest_path),
    pair_manifest_sha256=pair_manifest_sha256,
    pair_submission_receipt_path=str(pair_submission_receipt_path),
    pair_submission_receipt_sha256=pair_submission_receipt_sha256,
    trusted_off_exit_receipt_path=str(trusted_off_exit_receipt_path),
    trusted_off_exit_receipt_sha256=trusted_off_exit_receipt_sha256,
)
manifest, loaded_manifest_sha256 = manifest_utility.load_replay_execution_manifest_v2(
    path=manifest_path,
    expected_sha256=manifest_sha256,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
if loaded_manifest_sha256 != manifest_sha256:
    fail("reloaded manifest digest differs")
pair_submission = authenticated_source.pair_submission_receipt
scheduler_tools = pair_submission["scheduler_tools"]
client_environment = scheduler_tools["client_environment"]
host_tools = {name: scheduler_tools[name] for name in ("sbatch", "scancel", "scontrol")}
for name in host_tools:
    stable_tool_bytes(canonical_absolute_path(host_tools[name]["path"], f"scheduler tool {name} path"), label=f"scheduler tool {name}", expected_sha256=host_tools[name]["sha256"])
submission_receipt_path = canonical_absolute_path(manifest["scheduler_submission"]["receipt"]["path"], "submission receipt path")
submission_document, submission_raw = canonical_document(
    submission_receipt_path,
    label="replay submission receipt",
    trailing_lf=True,
    expected_sha256=hashlib.sha256(stable_evidence_bytes(submission_receipt_path, label="submission receipt raw")).hexdigest(),
)
submission_document, submission_sha256 = evidence_utility.load_captured_replay_submission_receipt_v2(
    path=submission_receipt_path,
    expected_sha256=hashlib.sha256(submission_raw).hexdigest(),
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
candidate_job_id = parse_job_id(submission_document["candidate_job_id"], label="candidate_job_id")
if candidate_job_id != live_job_id:
    fail("candidate job ID differs from live SLURM_JOB_ID")
results_root = canonical_absolute_path(pair_manifest["paths"]["results_root"], "Pair results root")
job_root = results_root / "captured_replay" / "replay_job_state" / manifest["pair_id"] / manifest["attempt_id"] / f"{live_job_id}-0"
queries_dir = job_root / "queries"
receipts_dir = job_root / "receipts"
pre_raw_path = queries_dir / "PRE.scontrol.raw"
pre_query_path = queries_dir / "PRE.scontrol-query.json"
post_raw_path = queries_dir / "POST.scontrol.raw"
post_query_path = queries_dir / "POST.scontrol-query.json"
pre_receipt_path = receipts_dir / "PRE.json"
exit_receipt_path = receipts_dir / "EXIT.json"
evidence_index_path = canonical_absolute_path(manifest["artifacts"]["outputs"]["evidence_index"]["path"], "replay evidence index path")
output_root = canonical_absolute_path(manifest["artifacts"]["outputs"]["directory"]["path"], "replay output root")
ensure_private_directory(queries_dir)
ensure_private_directory(receipts_dir)
for path, label in (
    (pre_raw_path, "PRE scheduler raw"),
    (pre_query_path, "PRE scheduler query"),
    (post_raw_path, "POST scheduler raw"),
    (post_query_path, "POST scheduler query"),
    (pre_receipt_path, "PRE receipt"),
    (exit_receipt_path, "EXIT receipt"),
    (evidence_index_path, "evidence index"),
    (
        canonical_absolute_path(
            manifest["artifacts"]["outputs"]["result_inventory"]["path"],
            "result inventory path",
        ),
        "result inventory",
    ),
    (output_root, "runtime output root"),
):
    require_absent(path, label=label)
submission_env = scheduler_env(client_environment)
pre_completed = run_scheduler_command(
    [host_tools["scontrol"]["path"], "show", "job", "--json", live_job_id],
    env=submission_env,
    client_environment=client_environment,
    label="PRE scontrol show job --json",
)
pre_raw_sha = write_exclusive(pre_raw_path, pre_completed.stdout, mode=0o400)
pre_record = normalize_scheduler_query(pre_completed.stdout, phase="PRE")
pre_query_document = evidence_utility.build_captured_replay_scheduler_query_v2(
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    phase="PRE",
    raw_output_path=str(pre_raw_path),
    raw_output_sha256=pre_raw_sha,
    record=pre_record,
)
_, pre_query_sha = evidence_utility.publish_captured_replay_scheduler_query_v2(
    output=pre_query_path,
    document=pre_query_document,
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
job = {
    "account": pair_manifest["campaign"]["slurm"]["account"],
    "name": manifest["scheduler_submission"]["identity"]["job_name"],
    "num_nodes": pair_manifest["campaign"]["nodes"],
    "partition": pair_manifest["campaign"]["slurm"]["partition"],
    "qos": pair_manifest["campaign"]["slurm"]["qos"],
    "gpus_per_node": 4,
    "restart_count": 0,
}
pre_document = evidence_utility.build_captured_replay_pre_receipt_v2(
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    submission_receipt=submission_document,
    authenticated_job_id=live_job_id,
    job=job,
    pre_scheduler_query={
        "path": str(pre_query_path),
        "schema": evidence_utility.REPLAY_SCHEDULER_QUERY_SCHEMA,
        "sha256": pre_query_sha,
    },
)
_, pre_receipt_sha = evidence_utility.publish_captured_replay_pre_receipt_v2(
    output=pre_receipt_path,
    document=pre_document,
    replay_execution_manifest=manifest,
    submission_receipt=submission_document,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
driver_scheduler_device_environment = evidence_utility._validate_scheduler_device_environment(
    observed_scheduler_device_environment()
)
driver_hardware = evidence_utility._validate_hardware(
    observed_hardware(),
    replay_execution_manifest=manifest,
)
driver_result = run_driver(
    manifest=manifest,
    pair_manifest=pair_manifest,
    pre_receipt_path=pre_receipt_path,
    pre_receipt_sha256=pre_receipt_sha,
    job=job,
    scheduler_device_environment=driver_scheduler_device_environment,
    client_environment=client_environment,
    submission_env=submission_env,
)
if driver_result.returncode != 0:
    fail(
        "driver failed with exit code "
        f"{driver_result.returncode}: {driver_result.stderr.decode('utf-8', 'replace')}"
    )
post_completed = run_scheduler_command(
    [host_tools["scontrol"]["path"], "show", "job", "--json", live_job_id],
    env=submission_env,
    client_environment=client_environment,
    label="POST scontrol show job --json",
)
post_raw_sha = write_exclusive(post_raw_path, post_completed.stdout, mode=0o400)
post_record = normalize_scheduler_query(post_completed.stdout, phase="POST")
post_query_document = evidence_utility.build_captured_replay_scheduler_query_v2(
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    phase="POST",
    raw_output_path=str(post_raw_path),
    raw_output_sha256=post_raw_sha,
    record=post_record,
)
_, post_query_sha = evidence_utility.publish_captured_replay_scheduler_query_v2(
    output=post_query_path,
    document=post_query_document,
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
transport_consumption_ref = output_reference("transport_consumption")
transport_document, _ = evidence_utility.load_evidence_document(
    path=transport_consumption_ref["path"],
    expected_sha256=transport_consumption_ref["sha256"],
    trailing_lf=False,
)
if not isinstance(transport_document, dict):
    fail("transport consumption root is not an object")
transport_replay = transport_document.get("replay")
if not isinstance(transport_replay, dict):
    fail("transport consumption replay is missing")
replay_process = transport_replay.get("process")
if not isinstance(replay_process, dict):
    fail("transport consumption replay.process is missing")
if not isinstance(transport_replay.get("scheduler_device_environment"), dict):
    fail("transport consumption replay.scheduler_device_environment is missing")
driver_reported_device_environment = (
    evidence_utility.validate_scheduler_device_environment(
        transport_replay.get("scheduler_device_environment")
    )
)
if driver_reported_device_environment != driver_scheduler_device_environment:
    fail("driver scheduler device environment differs from wrapper observation")
outputs = {
    "scorer_call_index": output_reference("scorer_call_index"),
    "transport_consumption": transport_consumption_ref,
    "transcript_bundle": output_reference("transcript_bundle"),
    "replay_ledger": output_reference("replay_ledger"),
}
exit_document = evidence_utility.build_captured_replay_exit_receipt_v2(
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    submission_receipt=submission_document,
    pre_receipt=pre_document,
    driver_exit_code=0,
    post_scheduler_query={
        "path": str(post_query_path),
        "schema": evidence_utility.REPLAY_SCHEDULER_QUERY_SCHEMA,
        "sha256": post_query_sha,
    },
    hardware=driver_hardware,
    scheduler_device_environment=driver_scheduler_device_environment,
    driver_scheduler_device_environment=driver_reported_device_environment,
    driver_process=replay_process,
    outputs=outputs,
)
_, exit_receipt_sha = evidence_utility.publish_captured_replay_exit_receipt_v2(
    output=exit_receipt_path,
    document=exit_document,
    replay_execution_manifest=manifest,
    submission_receipt=submission_document,
    pre_receipt=pre_document,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
evidence_index_document = evidence_utility.build_captured_replay_evidence_index_v2(
    replay_execution_manifest=manifest,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    submission_receipt=submission_document,
    pre_receipt=pre_document,
    exit_receipt=exit_document,
)
_, evidence_index_sha = evidence_utility.publish_captured_replay_evidence_index_v2(
    output=evidence_index_path,
    document=evidence_index_document,
    replay_execution_manifest=manifest,
    submission_receipt=submission_document,
    pre_receipt=pre_document,
    exit_receipt=exit_document,
    authenticated_source=authenticated_source,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
result_inventory_declaration = manifest["artifacts"]["outputs"]["result_inventory"]
expected_result_inventory_path = output_root / result_sealer.RESULT_INVENTORY_V2_FILENAME
expected_inventory_identity = {
    "path": str(expected_result_inventory_path),
    "schema": result_sealer.RESULT_INVENTORY_V2_SCHEMA,
    "framing": "canonical-ascii-json-no-lf",
    "mode": "0400",
    "self_excluded": True,
    "terminal_directory_mode": "0555",
    "environment": expected_environment,
    "profile_id": expected_profile_id,
}
if any(
    result_inventory_declaration.get(name) != value
    or type(result_inventory_declaration.get(name)) is not type(value)
    for name, value in expected_inventory_identity.items()
):
    fail("replay result inventory declaration differs from authenticated sealer")
anchored_sha256 = {
    "evidence-index.json": evidence_index_sha,
    "model-transport-replay-consumption.json": outputs[
        "transport_consumption"
    ]["sha256"],
    "replay-ledger.json": outputs["replay_ledger"]["sha256"],
    "strict_gym_child_runtime/format-verification-call-index.json": outputs[
        "scorer_call_index"
    ]["sha256"],
    "transcript-bundle.json": outputs["transcript_bundle"]["sha256"],
}
if [record["path"] for record in result_inventory_declaration["anchors"]] != [
    record["path"]
    for record in result_inventory_declaration["files"]
    if record["path"] in anchored_sha256
]:
    fail("replay result inventory anchor ordering differs")
if {record["path"] for record in result_inventory_declaration["anchors"]} != set(
    anchored_sha256
):
    fail("replay result inventory anchor set differs")
result_inventory_path, result_inventory_sha = result_sealer.publish_sealed_result_v2(
    result_root=str(output_root),
    anchored_sha256=anchored_sha256,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
)
if result_inventory_path != str(expected_result_inventory_path):
    fail("sealed result inventory path differs from manifest")
print(
    json.dumps(
        {
            "attempt_id": manifest["attempt_id"],
            "authenticated_job_id": live_job_id,
            "pair_id": manifest["pair_id"],
            "pre_receipt_sha256": pre_receipt_sha,
            "exit_receipt_sha256": exit_receipt_sha,
            "evidence_index_sha256": evidence_index_sha,
            "result_inventory_path": result_inventory_path,
            "result_inventory_sha256": result_inventory_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

strict_replay_main "$@"
