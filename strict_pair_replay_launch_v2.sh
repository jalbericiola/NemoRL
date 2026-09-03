#!/bin/bash -p
set -euo pipefail

readonly STRICT_REPLAY_LINUX_SHA256SUM_PATH="/usr/bin/sha256sum"
readonly STRICT_REPLAY_LINUX_SHA256SUM_SHA256="ffb52ec22da029403b8e2a2ee4e07bc4f111f1ec3c6d8fac2f2b5788891dd825"
readonly STRICT_REPLAY_LINUX_PYTHON_PATH="/cm/local/apps/python312/bin/python3.12"
readonly STRICT_REPLAY_LINUX_PYTHON_SHA256="36bee55d1d2c90ceda25e65038809d69be3d3e6e82c94e5d9ec3b2ec1ccc9faa"
readonly STRICT_REPLAY_DARWIN_SHA256SUM_PATH="/sbin/sha256sum"
readonly STRICT_REPLAY_DARWIN_SHA256SUM_SHA256="881f3812ac7be70d99bf635e5322b63f565af66f502be3b464036fef8f927300"
readonly STRICT_REPLAY_DARWIN_PYTHON_PATH="/usr/bin/python3"
readonly STRICT_REPLAY_DARWIN_PYTHON_SHA256="179301dcb41ea78accc3fa0048a7e6f6710d891945a751a34addd622020c1818"

strict_replay_die() {
  echo "ERROR: $*" >&2
  exit 2
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
    strict_replay_die "replay launcher must be invoked by absolute non-symlink path"
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
    raise SystemExit("replay launcher failed: bootstrap Python must run with -S")

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
CLEANUP_REPORT_SCHEMA = "nemo-rl-strict-captured-replay-launcher-cleanup-report-v1"
CLEANUP_ARTIFACT_SUFFIXES = (
    "CLEANUP_CANCEL.stderr",
    "CLEANUP_CANCEL.stdout",
    "CLEANUP_POST.scontrol.raw",
    "CLEANUP_POST.scontrol.stderr",
    "CLEANUP_PRE.scontrol.raw",
    "CLEANUP_PRE.scontrol.stderr",
    "UNKNOWN_CANDIDATE.sbatch.stderr",
    "UNKNOWN_CANDIDATE.sbatch.stdout",
    "cleanup-report.json",
)
CLEANUP_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


def fail(message: str) -> None:
    raise SystemExit(f"replay launcher failed: {message}")


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


def canonical_document(path: Path, *, label: str, trailing_lf: bool, expected_sha256: str) -> dict[str, Any]:
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
    return value


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
    pair_manifest = canonical_document(
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
    pair_submission = canonical_document(
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
    fail(f"{label} must be absent before replay submission")


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


def create_exclusive_private_directory(path: Path) -> None:
    """Create one fresh per-attempt state directory; never reuse prior state."""
    ensure_private_directory(path.parent)
    require_absent(path, label="per-attempt submission-state directory")
    try:
        path.mkdir(mode=0o700)
    except OSError as error:
        fail(f"cannot exclusively create per-attempt state directory {path}: {error}")
    metadata = path.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        fail("created per-attempt submission-state directory differs")


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


def cleanup_artifact_path(
    *, submission_parent: Path, attempt_id: str, suffix: str
) -> Path:
    if type(attempt_id) is not str or attempt_id not in {"replay-1", "replay-2"}:
        fail("cleanup artifact attempt differs from the closed replay attempts")
    if type(suffix) is not str or suffix not in CLEANUP_ARTIFACT_SUFFIXES:
        fail("cleanup artifact suffix is not admitted")
    return submission_parent / f"{attempt_id}.{suffix}"


def require_cleanup_artifacts_absent(
    *, submission_parent: Path, attempt_id: str
) -> None:
    for suffix in CLEANUP_ARTIFACT_SUFFIXES:
        require_absent(
            cleanup_artifact_path(
                submission_parent=submission_parent,
                attempt_id=attempt_id,
                suffix=suffix,
            ),
            label=f"launcher cleanup artifact {suffix}",
        )


def parse_job_id(text: bytes) -> str:
    if not text.endswith(b"\n") or text.count(b"\n") != 1:
        fail("scheduler stdout must be exactly one LF-terminated line")
    try:
        value = text[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        fail(f"candidate job ID is not ASCII: {error}")
    if JOB_ID_RE.fullmatch(value) is None or int(value) > MAX_INT63:
        fail("candidate job ID is not a canonical positive int63")
    return value


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
    elif phase == "CLEANUP":
        if record["job_state"] in CLEANUP_TERMINAL_STATES:
            fail("CLEANUP scheduler record must be nonterminal")
    elif phase == "ROLLBACK":
        if (
            record["job_state"] != "CANCELLED"
            or record["held"] is not False
            or record["reason"] == "JobHeldUser"
        ):
            fail("ROLLBACK scheduler record differs")
    else:
        if record["held"] is not False or record["reason"] == "JobHeldUser":
            fail(f"{phase} scheduler hold/reason differs")
    return record


def build_submission_argv(
    *,
    manifest: dict[str, Any],
    pair_manifest_path: Path,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: Path,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: Path,
    trusted_off_exit_receipt_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
    snapshot_root: Path,
    comment: str,
) -> list[str]:
    pair = pair_manifest
    slurm = pair["campaign"]["slurm"]
    slurm_root = canonical_absolute_path(
        manifest["execution_environment"]["attempt"]["operational"]["slurm"],
        "operational Slurm root",
    )
    wrapper_relative = manifest["replay_contract"]["program"]["job_wrapper"]["path"]
    return [
        "--parsable",
        "--hold",
        f"--chdir={snapshot_root}",
        f"--nodes={pair['campaign']['nodes']}",
        f"--account={slurm['account']}",
        f"--job-name={manifest['scheduler_submission']['identity']['job_name']}",
        f"--partition={slurm['partition']}",
        "--time=04:00:00",
        "--gres=gpu:4",
        "--exclusive",
        "--mem=0",
        "--dependency=singleton",
        "--segment=1",
        f"--output={slurm_root}/slurm-%j.out",
        f"--error={slurm_root}/slurm-%j.err",
        f"--qos={slurm['qos']}",
        f"--comment={comment}",
        f"--export-file={manifest['slurm_export_boundary']['path']}",
        str(snapshot_root / wrapper_relative),
        "--pair-manifest",
        str(pair_manifest_path),
        "--pair-manifest-sha256",
        pair_manifest_sha256,
        "--pair-submission-receipt",
        str(pair_submission_receipt_path),
        "--pair-submission-receipt-sha256",
        pair_submission_receipt_sha256,
        "--off-exit-receipt",
        str(trusted_off_exit_receipt_path),
        "--off-exit-receipt-sha256",
        trusted_off_exit_receipt_sha256,
        "--replay-manifest",
        str(manifest_path),
        "--replay-manifest-sha256",
        manifest_sha256,
        "--environment",
        expected_environment,
        "--profile-id",
        expected_profile_id,
    ]


def verify_scheduler_client_environment(client_environment: dict[str, Any]) -> None:
    variables = client_environment["variables"]
    slurm_conf = variables["SLURM_CONF"]
    stable_authenticated_host_file_bytes(
        canonical_absolute_path(slurm_conf["path"], "authenticated SLURM_CONF path"),
        label="authenticated SLURM_CONF",
        expected_sha256=slurm_conf["sha256"],
    )


def scheduler_env(client_environment: dict[str, Any]) -> dict[str, str]:
    variables = client_environment["variables"]
    slurm_conf = variables["SLURM_CONF"]
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


def cleanup_authenticated_candidate(
    *,
    submission_parent: Path,
    attempt_id: str,
    pair_id: str,
    host_tools: dict[str, dict[str, str]],
    client_environment: dict[str, Any],
    submission_env: dict[str, str],
    candidate_job_id: str,
    expected_job_name: str,
    expected_comment: str,
    expected_user_id: str,
    expected_work_dir: Path,
) -> dict[str, Any]:
    def persist_process_result(
        prefix: str, completed: subprocess.CompletedProcess[bytes]
    ) -> dict[str, Any]:
        stdout_path = cleanup_artifact_path(
            submission_parent=submission_parent,
            attempt_id=attempt_id,
            suffix=f"{prefix}.stdout",
        )
        stderr_path = cleanup_artifact_path(
            submission_parent=submission_parent,
            attempt_id=attempt_id,
            suffix=f"{prefix}.stderr",
        )
        stdout_sha = write_exclusive(stdout_path, completed.stdout or b"", mode=0o400)
        stderr_sha = write_exclusive(stderr_path, completed.stderr or b"", mode=0o400)
        return {
            "argv": list(completed.args),
            "status": completed.returncode,
            "stdout": {"path": str(stdout_path), "sha256": stdout_sha},
            "stderr": {"path": str(stderr_path), "sha256": stderr_sha},
        }

    def persist_scontrol_query(
        prefix: str, completed: subprocess.CompletedProcess[bytes], phase: str
    ) -> dict[str, Any]:
        raw_path = cleanup_artifact_path(
            submission_parent=submission_parent,
            attempt_id=attempt_id,
            suffix=f"{prefix}.scontrol.raw",
        )
        stderr_path = cleanup_artifact_path(
            submission_parent=submission_parent,
            attempt_id=attempt_id,
            suffix=f"{prefix}.scontrol.stderr",
        )
        raw_sha = write_exclusive(raw_path, completed.stdout or b"", mode=0o400)
        stderr_sha = write_exclusive(
            stderr_path, completed.stderr or b"", mode=0o400
        )
        entry: dict[str, Any] = {
            "argv": list(completed.args),
            "status": completed.returncode,
            "raw_output": {"path": str(raw_path), "sha256": raw_sha},
            "stderr": {"path": str(stderr_path), "sha256": stderr_sha},
            "normalized_record": None,
            "normalization_error": None,
        }
        if completed.returncode == 0 and completed.stderr in {b"", None}:
            try:
                entry["normalized_record"] = normalize_scheduler_query(
                    completed.stdout or b"", phase=phase
                )
            except BaseException as error:
                entry["normalization_error"] = f"{type(error).__name__}: {error}"
        return entry

    expected_identity = {
        "job_id": candidate_job_id,
        "job_name": expected_job_name,
        "comment": expected_comment,
        "user_id": expected_user_id,
        "work_dir": str(expected_work_dir),
        "restart_count": 0,
    }
    report: dict[str, Any] = {
        "schema": CLEANUP_REPORT_SCHEMA,
        "attempt_id": attempt_id,
        "pair_id": pair_id,
        "candidate_job_id": candidate_job_id,
        "confirmed": False,
        "status": "cleanup-not-attempted",
        "pre_cancel_query": None,
        "cancellation": None,
        "post_cancel_query": None,
        "manual_recovery_hint": (
            f"Inspect candidate job {candidate_job_id} with scontrol --json and "
            f"cancel manually with {host_tools['scancel']['path']} {candidate_job_id} if it is the exact replay job."
        ),
    }
    try:
        verify_scheduler_client_environment(client_environment)
        pre_completed = subprocess.run(
            [host_tools["scontrol"]["path"], "show", "job", "--json", candidate_job_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=submission_env,
            cwd=str(snapshot_root),
        )
        report["pre_cancel_query"] = persist_scontrol_query(
            "CLEANUP_PRE", pre_completed, "CLEANUP"
        )
        pre_record = report["pre_cancel_query"]["normalized_record"]
        if (
            pre_completed.returncode != 0
            or pre_completed.stderr not in {b"", None}
            or pre_record is None
        ):
            report["status"] = "cleanup-pre-query-unconfirmed"
        elif any(pre_record[name] != expected_identity[name] for name in expected_identity):
            report["status"] = "cleanup-identity-mismatch"
        else:
            verify_scheduler_client_environment(client_environment)
            cancel_completed = subprocess.run(
                [host_tools["scancel"]["path"], candidate_job_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
                env=submission_env,
                cwd=str(snapshot_root),
            )
            report["cancellation"] = persist_process_result(
                "CLEANUP_CANCEL", cancel_completed
            )
            if cancel_completed.returncode != 0 or cancel_completed.stderr not in {b"", None}:
                report["status"] = "cleanup-cancel-unconfirmed"
            else:
                verify_scheduler_client_environment(client_environment)
                post_completed = subprocess.run(
                    [host_tools["scontrol"]["path"], "show", "job", "--json", candidate_job_id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=30,
                    env=submission_env,
                    cwd=str(snapshot_root),
                )
                report["post_cancel_query"] = persist_scontrol_query(
                    "CLEANUP_POST", post_completed, "ROLLBACK"
                )
                post_record = report["post_cancel_query"]["normalized_record"]
                expected_post_record = None
                if post_record is not None:
                    expected_post_record = {
                        **expected_identity,
                        "job_state": "CANCELLED",
                        "reason": post_record["reason"],
                        "held": False,
                    }
                if (
                    post_completed.returncode == 0
                    and post_completed.stderr in {b"", None}
                    and post_record is not None
                    and post_record == expected_post_record
                ):
                    report["confirmed"] = True
                    report["status"] = "cleanup-confirmed"
                else:
                    report["status"] = "cleanup-post-query-unconfirmed"
    except BaseException as error:
        report["status"] = "cleanup-exception"
        report["exception"] = f"{type(error).__name__}: {error}"
    payload = json.dumps(
        report, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    report_path = cleanup_artifact_path(
        submission_parent=submission_parent,
        attempt_id=attempt_id,
        suffix="cleanup-report.json",
    )
    report_sha256 = write_exclusive(report_path, payload, mode=0o400)
    return {
        "confirmed": report["confirmed"],
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "status": report["status"],
    }


def persist_unknown_candidate_report(
    *,
    submission_parent: Path,
    attempt_id: str,
    pair_id: str,
    sbatch_argv: list[str],
    completed: Optional[subprocess.CompletedProcess[bytes]],
    error: BaseException,
    expected_job_name: str,
    expected_comment: str,
    expected_user_id: str,
    expected_work_dir: Path,
) -> dict[str, str]:
    partial_stdout = getattr(error, "stdout", None)
    partial_stderr = getattr(error, "stderr", None)
    stdout = completed.stdout if completed is not None else partial_stdout
    stderr = completed.stderr if completed is not None else partial_stderr
    if not isinstance(stdout, bytes):
        stdout = b""
    if not isinstance(stderr, bytes):
        stderr = b""
    stdout_path = cleanup_artifact_path(
        submission_parent=submission_parent,
        attempt_id=attempt_id,
        suffix="UNKNOWN_CANDIDATE.sbatch.stdout",
    )
    stderr_path = cleanup_artifact_path(
        submission_parent=submission_parent,
        attempt_id=attempt_id,
        suffix="UNKNOWN_CANDIDATE.sbatch.stderr",
    )
    stdout_sha256 = write_exclusive(stdout_path, stdout, mode=0o400)
    stderr_sha256 = write_exclusive(stderr_path, stderr, mode=0o400)
    report = {
        "schema": CLEANUP_REPORT_SCHEMA,
        "attempt_id": attempt_id,
        "pair_id": pair_id,
        "candidate_job_id": None,
        "confirmed": False,
        "status": "cleanup-candidate-id-unknown",
        "sbatch": {
            "argv": list(sbatch_argv),
            "status": completed.returncode if completed is not None else None,
            "stdout": {"path": str(stdout_path), "sha256": stdout_sha256},
            "stderr": {"path": str(stderr_path), "sha256": stderr_sha256},
        },
        "expected_identity": {
            "job_name": expected_job_name,
            "comment": expected_comment,
            "user_id": expected_user_id,
            "work_dir": str(expected_work_dir),
        },
        "exception": f"{type(error).__name__}: {error}",
        "manual_recovery_hint": (
            "The sbatch result did not authenticate a candidate job ID. Use an "
            "authenticated scheduler client to enumerate jobs and match every exact "
            "expected_identity field; cancel only the matching held replay job. Never "
            "treat the captured stdout as job-ID authority."
        ),
    }
    payload = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    report_path = cleanup_artifact_path(
        submission_parent=submission_parent,
        attempt_id=attempt_id,
        suffix="cleanup-report.json",
    )
    report_sha256 = write_exclusive(report_path, payload, mode=0o400)
    return {
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "status": report["status"],
    }


if len(sys.argv) != 16:
    fail("bootstrap argument count differs from the V2 contract")
executed_script = canonical_absolute_path(sys.argv[1], "executed launcher path")
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
sha_tool_path = canonical_absolute_path(sys.argv[12], "bootstrap sha256sum path")
sha_tool_sha256 = sys.argv[13]
host_python_path = canonical_absolute_path(sys.argv[14], "bootstrap Python path")
host_python_sha256 = sys.argv[15]
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
manifest = canonical_document(
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
    executable_name="submission_launcher",
)
launcher_path, launcher_sha256, launcher_raw = authenticated_programs[
    "submission_launcher"
]
if executed_script != launcher_path:
    fail("executed launcher path differs from authenticated Pair ON launcher")
if hashlib.sha256(stable_evidence_bytes(executed_script, label="executed launcher bytes", exact_mode=None)).hexdigest() != launcher_sha256:
    fail("executed launcher bytes differ from authenticated Pair ON launcher")
runner_path, runner_sha256, runner_raw = authenticated_programs["entrypoint"]
manifest_utility_path, _, manifest_utility_raw = authenticated_programs[
    "manifest_utility"
]
evidence_utility_path, _, evidence_utility_raw = authenticated_programs[
    "evidence_utility"
]
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
accepted_path = canonical_absolute_path(manifest["scheduler_submission"]["accepted_id_record"]["path"], "accepted id path")
submission_receipt_path = canonical_absolute_path(manifest["scheduler_submission"]["receipt"]["path"], "submission receipt path")
submission_parent = accepted_path.parent
pre_release_raw_path = submission_parent / "PRE_RELEASE.scontrol.raw"
pre_release_query_path = submission_parent / "PRE_RELEASE.scontrol-query.json"
output_root = canonical_absolute_path(manifest["artifacts"]["outputs"]["directory"]["path"], "replay output root")
evidence_index_path = canonical_absolute_path(
    manifest["artifacts"]["outputs"]["evidence_index"]["path"],
    "replay evidence index path",
)
operational_root = canonical_absolute_path(
    manifest["execution_environment"]["attempt"]["operational"]["root"],
    "replay operational root",
)
slurm_root = canonical_absolute_path(
    manifest["execution_environment"]["attempt"]["operational"]["slurm"],
    "replay operational Slurm root",
)
launcher_operational_root = operational_root / "launcher"
create_exclusive_private_directory(submission_parent)
ensure_private_directory(operational_root)
ensure_private_directory(slurm_root)
ensure_private_directory(launcher_operational_root)
require_cleanup_artifacts_absent(
    submission_parent=launcher_operational_root,
    attempt_id=manifest["attempt_id"],
)
for path, label in (
    (accepted_path, "accepted ID record"),
    (submission_receipt_path, "submission receipt"),
    (pre_release_raw_path, "PRE_RELEASE scheduler raw"),
    (pre_release_query_path, "PRE_RELEASE scheduler query"),
    (evidence_index_path, "replay evidence index"),
    (output_root, "runtime output root"),
):
    require_absent(path, label=label)
comment = manifest["scheduler_submission"]["identity"]["comment_template"].format(
    attempt_id=manifest["attempt_id"],
    submission_nonce=manifest["scheduler_submission"]["nonce"],
    replay_manifest_sha256=manifest_sha256,
)
sbatch_argv = build_submission_argv(
    manifest=manifest,
    pair_manifest_path=pair_manifest_path,
    pair_manifest_sha256=pair_manifest_sha256,
    pair_submission_receipt_path=pair_submission_receipt_path,
    pair_submission_receipt_sha256=pair_submission_receipt_sha256,
    trusted_off_exit_receipt_path=trusted_off_exit_receipt_path,
    trusted_off_exit_receipt_sha256=trusted_off_exit_receipt_sha256,
    manifest_path=manifest_path,
    manifest_sha256=manifest_sha256,
    expected_environment=expected_environment,
    expected_profile_id=expected_profile_id,
    snapshot_root=snapshot_root,
    comment=comment,
)
submission_env = scheduler_env(client_environment)
candidate_job_id: Optional[str] = None
completed: Optional[subprocess.CompletedProcess[bytes]] = None
try:
    completed = run_scheduler_command(
        [host_tools["sbatch"]["path"], *sbatch_argv],
        env=submission_env,
        client_environment=client_environment,
        label="sbatch",
    )
    candidate_job_id = parse_job_id(completed.stdout)
    accepted_payload = f"{candidate_job_id}\n".encode("ascii")
    accepted_sha = write_exclusive(accepted_path, accepted_payload, mode=0o400)
    accepted_record = {
        "path": str(accepted_path),
        "sha256": accepted_sha,
        "parsed_candidate_job_id": candidate_job_id,
        "format": "ascii-positive-decimal-lf",
        "mode": "0400",
    }
    query_completed = run_scheduler_command(
        [host_tools["scontrol"]["path"], "show", "job", "--json", candidate_job_id],
        env=submission_env,
        client_environment=client_environment,
        label="scontrol show job --json",
    )
    raw_sha = write_exclusive(pre_release_raw_path, query_completed.stdout, mode=0o400)
    record = normalize_scheduler_query(query_completed.stdout, phase="PRE_RELEASE")
    query_document = evidence_utility.build_captured_replay_scheduler_query_v2(
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
        phase="PRE_RELEASE",
        raw_output_path=str(pre_release_raw_path),
        raw_output_sha256=raw_sha,
        record=record,
    )
    _, query_sha = evidence_utility.publish_captured_replay_scheduler_query_v2(
        output=pre_release_query_path,
        document=query_document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    submission_document = evidence_utility.build_captured_replay_submission_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
        replay_execution_manifest_path=str(manifest_path),
        replay_execution_manifest_sha256=manifest_sha256,
        scheduler_client_environment=client_environment,
        scheduler_tools=host_tools,
        sbatch_argv=sbatch_argv,
        parsable_stdout=completed.stdout.decode("ascii"),
        accepted_id_record=accepted_record,
        pre_release_scheduler_query={
            "path": str(pre_release_query_path),
            "schema": evidence_utility.REPLAY_SCHEDULER_QUERY_SCHEMA,
            "sha256": query_sha,
        },
    )
    _, submission_sha = evidence_utility.publish_captured_replay_submission_receipt_v2(
        output=submission_receipt_path,
        document=submission_document,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    run_scheduler_command(
        [host_tools["scontrol"]["path"], "release", candidate_job_id],
        env=submission_env,
        client_environment=client_environment,
        label="scontrol release",
    )
except BaseException as error:
    if candidate_job_id is not None:
        cleanup_result = cleanup_authenticated_candidate(
            submission_parent=launcher_operational_root,
            attempt_id=manifest["attempt_id"],
            pair_id=manifest["pair_id"],
            host_tools=host_tools,
            client_environment=client_environment,
            submission_env=submission_env,
            candidate_job_id=candidate_job_id,
            expected_job_name=manifest["scheduler_submission"]["identity"]["job_name"],
            expected_comment=comment,
            expected_user_id=str(os.geteuid()),
            expected_work_dir=snapshot_root,
        )
        if not cleanup_result["confirmed"]:
            raise SystemExit(
                "replay launcher failed: cleanup "
                f"{cleanup_result['status']} for candidate_job_id={candidate_job_id}; "
                f"see {cleanup_result['report_path']} sha256={cleanup_result['report_sha256']}; "
                f"original failure: {type(error).__name__}: {error}"
            ) from error
    else:
        unknown_result = persist_unknown_candidate_report(
            submission_parent=launcher_operational_root,
            attempt_id=manifest["attempt_id"],
            pair_id=manifest["pair_id"],
            sbatch_argv=[host_tools["sbatch"]["path"], *sbatch_argv],
            completed=completed,
            error=error,
            expected_job_name=manifest["scheduler_submission"]["identity"]["job_name"],
            expected_comment=comment,
            expected_user_id=str(os.geteuid()),
            expected_work_dir=snapshot_root,
        )
        raise SystemExit(
            "replay launcher failed before authenticating a candidate job ID; "
            f"see {unknown_result['report_path']} sha256={unknown_result['report_sha256']}; "
            "manual scheduler remediation is required; "
            f"original failure: {type(error).__name__}: {error}"
        ) from error
    raise
print(
    json.dumps(
        {
            "attempt_id": manifest["attempt_id"],
            "candidate_job_id": candidate_job_id,
            "pair_id": manifest["pair_id"],
            "submission_receipt_sha256": submission_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
}

strict_replay_main "$@"
