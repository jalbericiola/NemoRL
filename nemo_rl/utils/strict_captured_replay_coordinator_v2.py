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

"""Login-side coordinator for profile-bound strict captured replay V2.

The submit phase stops after the authenticated launcher has released the held
Slurm job.  The job wrapper owns terminal result publication.  After that job
finishes, the consume phase authenticates the wrapper's external FINAL receipt
and returns a detached snapshot.  The FINAL receipt, not a caller-supplied
result path or inventory digest, selects the sealed result.

The launcher's release RPC and its stdout receipt are not one atomic scheduler
transaction.  The launcher cancels every authenticated candidate on any error
it observes, and the coordinator's deadline exceeds all bounded launcher RPCs
plus cleanup.  An external termination in the narrow interval after a
successful release and before receipt delivery remains an explicit scheduler
commit/ack recovery boundary: operators must reconcile the manifest-bound job
identity rather than infer that coordinator failure means no job was released.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import posixpath
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from nemo_rl.utils.strict_captured_replay_evidence import load_evidence_document
from nemo_rl.utils.strict_captured_replay_evidence_v2 import (
    AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA,
    REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    REPLAY_POST_INDEX_V2_SCHEMA,
    REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
    REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    load_authenticated_captured_replay_result_v2,
    load_captured_replay_submission_receipt_v2,
    snapshot_authenticated_captured_replay_result_v2,
)
from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
    PAIR_MANIFEST_SCHEMA,
    PAIR_SLURM_EXPORT_SCHEMA,
    REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    REPLAY_JOB_ARGV_TEMPLATE_V2,
    REPLAY_SUBMISSION_CONTRACT_SCHEMA,
    SLURM_EXPORT_ALLOWED_NAMES,
    AuthenticatedOffSourceCapture,
    AuthenticatedReplayStaticInputs,
    build_replay_execution_manifest_v2,
    build_replay_submission_contract,
    load_authenticated_off_source_capture,
    load_authenticated_replay_static_inputs,
    load_replay_execution_manifest_v2,
    publish_replay_execution_manifest_v2,
    publish_replay_submission_contract,
)
from nemo_rl.utils.strict_captured_replay_profiles import (
    StrictCapturedReplayProfile,
    get_strict_captured_replay_profile,
)

PAIR79_RECORD_COUNT = 79
PAIR79_EXPORT_DIRECTORY = "slurm_exports"
REPLAY_MANIFEST_DIRECTORY = "manifests"
REPLAY_SUBMISSION_STATE_DIRECTORY = "replay_submission_state"
COORDINATOR_SUBMIT_RESULT_SCHEMA = "nemo-rl-strict-captured-replay-coordinator-submit-result-v1"
COORDINATOR_CONSUME_RESULT_SCHEMA = "nemo-rl-strict-captured-replay-coordinator-consume-result-v2"
_ATTEMPTS = frozenset({"replay-1", "replay-2"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_READ_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_REPLAY_BASH_PATH = "/bin/bash"
_REPLAY_COORDINATOR_CWD = "/"
_LAUNCH_TIMEOUT_SECONDS = 300.0
_LAUNCH_STDOUT_LIMIT = 64 * 1024
_LAUNCH_STDERR_LIMIT = 16 * 1024
_LAUNCH_ENVIRONMENT: dict[str, str] = {}
_LAUNCH_RECEIPT_KEYS = frozenset(
    {
        "attempt_id",
        "candidate_job_id",
        "pair_id",
        "submission_receipt_sha256",
    }
)
_AUTHENTICATED_RESULT_SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "pair_id",
        "environment",
        "profile_id",
        "attempt_id",
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_root",
        "result_inventory",
        "evidence_index",
        "outputs",
        "samples",
    }
)
_AUTHENTICATED_RESULT_OUTPUT_KEYS = frozenset(
    {
        "scorer_call_index",
        "transport_consumption",
        "transcript_bundle",
        "replay_ledger",
    }
)
_AUTHENTICATED_RESULT_SAMPLE_KEYS = frozenset(
    {
        "sample_index",
        "fixture_row_index",
        "rollout_index",
        "generation_seed",
        "model_transport_entry_sha256",
        "model_transport_request_body_sha256",
        "model_transport_response_body_sha256",
        "model_response_sha256",
        "match_details",
        "raw_environment_reward",
    }
)
_RESULT_INVENTORY_V2_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
_REPLAY_TRANSPORT_CONSUMPTION_V3_SCHEMA = "nemo-rl-strict-model-transport-replay-consumption-v3"
_REPLAY_TRANSCRIPT_BUNDLE_V4_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
_REPLAY_LEDGER_V5_SCHEMA = "nemo-rl-strict-captured-replay-step1-ledger-v5"


class StrictCapturedReplayCoordinatorError(ValueError):
    """Raised when coordinator input or output violates the closed contract."""


@dataclass(frozen=True, slots=True)
class Pair79ReplayExport:
    """Exact canonical Pair79 NUL export ready for exclusive publication."""

    path: str
    sha256: str
    records: tuple[tuple[str, bytes], ...]
    raw: bytes


@dataclass(frozen=True, slots=True)
class PreparedReplaySubmission:
    """Authenticated immutable replay inputs and exact launcher invocation."""

    pair_id: str
    attempt_id: str
    environment: str
    profile_id: str
    pair_manifest_path: str
    pair_manifest_sha256: str
    pair_submission_receipt_path: str
    pair_submission_receipt_sha256: str
    trusted_off_exit_receipt_path: str
    trusted_off_exit_receipt_sha256: str
    slurm_export_path: str
    slurm_export_sha256: str
    submission_contract_path: str
    submission_contract_sha256: str
    replay_manifest_path: str
    replay_manifest_sha256: str
    launcher_path: str
    launcher_sha256: str
    launcher_cwd: str
    launcher_argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubmittedReplay:
    """Prepared replay plus the authenticated launcher's release receipt."""

    prepared: PreparedReplaySubmission
    launcher_receipt: dict[str, str]
    submission_receipt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConsumedReplayResult:
    """Detached authenticated snapshot of one FINAL-selected result."""

    snapshot: dict[str, Any]


def _fail(message: str) -> None:
    raise StrictCapturedReplayCoordinatorError(message)


def _exact_string(value: Any, *, name: str) -> str:
    if type(value) is not str:
        _fail(f"{name} must be an exact string")
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        _fail(f"{name} must be one nonempty line without NUL")
    return value


def _digest(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if _DIGEST.fullmatch(result) is None:
        _fail(f"{name} must be one lowercase SHA-256 digest")
    return result


def _git_object_id(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if _GIT_OBJECT_ID.fullmatch(result) is None:
        _fail(f"{name} must be one lowercase 40-hex Git object ID")
    return result


def _attempt(value: Any) -> str:
    result = _exact_string(value, name="attempt_id")
    if result not in _ATTEMPTS:
        _fail("attempt_id must be exactly replay-1 or replay-2")
    return result


def _canonical_absolute(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if (
        not result.startswith("/")
        or result.startswith("//")
        or result.endswith("/")
        or posixpath.normpath(result) != result
    ):
        _fail(f"{name} must be one canonical absolute path")
    return result


def _safe_id(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if len(result) > 64 or _SAFE_ID.fullmatch(result) is None:
        _fail(f"{name} is not one bounded safe identifier")
    return result


def _exact_dict(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{name} must be an exact dictionary")
    if any(type(key) is not str for key in value):
        _fail(f"{name} keys must be exact strings")
    return value


def _exact_profile(*, expected_environment: Any, expected_profile_id: Any) -> StrictCapturedReplayProfile:
    environment = _exact_string(expected_environment, name="expected_environment")
    profile_id = _exact_string(expected_profile_id, name="expected_profile_id")
    return get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )


def _validate_pair_profile_before_publication(
    authenticated_source: AuthenticatedOffSourceCapture,
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[dict[str, Any], StrictCapturedReplayProfile]:
    if type(authenticated_source) is not AuthenticatedOffSourceCapture:
        _fail("authenticated_source must be an exact AuthenticatedOffSourceCapture")
    profile = _exact_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    pair = _exact_dict(authenticated_source.pair_manifest, name="authenticated Pair")
    if pair.get("schema") != PAIR_MANIFEST_SCHEMA or type(pair.get("schema")) is not str:
        _fail("authenticated Pair schema differs")
    selection = _exact_dict(pair.get("selection"), name="authenticated Pair selection")
    pair_environment = selection.get("environment")
    if type(pair_environment) is not str or pair_environment != profile.environment:
        _fail("authenticated Pair environment differs from explicit scorer profile")

    source = _exact_dict(pair.get("source"), name="authenticated Pair source")
    source_root = _canonical_absolute(source.get("root"), name="authenticated Pair source root")
    artifacts = _exact_dict(pair.get("artifacts"), name="authenticated Pair artifacts")
    fixture = _exact_dict(artifacts.get("fixture"), name="authenticated Pair fixture")
    if set(fixture) != {"path", "rows", "sha256"}:
        _fail("authenticated Pair fixture keyset differs")
    relative_fixture = PurePosixPath(profile.fixture_path)
    if (
        relative_fixture.is_absolute()
        or not relative_fixture.parts
        or any(part in {"", ".", ".."} for part in relative_fixture.parts)
    ):
        _fail("closed scorer profile fixture path is not canonical relative")
    expected_fixture_path = str(PurePosixPath(source_root) / relative_fixture)
    if (
        type(fixture.get("path")) is not str
        or fixture["path"] != expected_fixture_path
        or type(fixture.get("sha256")) is not str
        or fixture["sha256"] != profile.fixture_sha256
        or type(fixture.get("rows")) is not int
        or fixture["rows"] != profile.fixture_rows
    ):
        _fail("authenticated Pair fixture differs from explicit scorer profile")

    boundary = _exact_dict(
        pair.get("slurm_export_boundary"),
        name="authenticated Pair Slurm export boundary",
    )
    allowed_names = boundary.get("allowed_names")
    if (
        boundary.get("schema") != PAIR_SLURM_EXPORT_SCHEMA
        or type(boundary.get("schema")) is not str
        or type(allowed_names) is not list
        or any(type(name) is not str for name in allowed_names)
        or tuple(allowed_names) != SLURM_EXPORT_ALLOWED_NAMES
    ):
        _fail("authenticated Pair Slurm export boundary differs from exact Pair79")
    _validate_pair79_roster()
    return pair, profile


def _validate_pair79_roster() -> None:
    names = SLURM_EXPORT_ALLOWED_NAMES
    if (
        type(names) is not tuple
        or len(names) != PAIR79_RECORD_COUNT
        or len(set(names)) != PAIR79_RECORD_COUNT
        or any(
            type(name) is not str
            or not name
            or name.encode("ascii", "strict").decode("ascii") != name
            or "=" in name
            or "\x00" in name
            for name in names
        )
    ):
        _fail("V2 Pair79 roster is not the exact closed 79-name tuple")


def _pair79_nonempty_values(pair: Mapping[str, Any], *, attempt_id: str) -> dict[str, bytes]:
    source = _exact_dict(pair.get("source"), name="authenticated Pair source")
    gym = _exact_dict(source.get("gym"), name="authenticated Pair Gym source")
    snapshots = _exact_dict(source.get("snapshots"), name="authenticated Pair snapshots")
    on_snapshot = _exact_dict(snapshots.get("on"), name="authenticated Pair ON snapshot")
    paths = _exact_dict(pair.get("paths"), name="authenticated Pair paths")
    selection = _exact_dict(pair.get("selection"), name="authenticated Pair selection")
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    results_root = _canonical_absolute(paths.get("results_root"), name="Pair results root")
    values = {
        "EXPECTED_GYM_GITLINK_COMMIT": _git_object_id(gym.get("gitlink_commit"), name="Pair Gym gitlink commit"),
        "EXPECTED_GYM_TREE": _git_object_id(gym.get("tree"), name="Pair Gym tree"),
        "PAIR_ID": pair_id,
        "RESULTS_DIR": f"{results_root}/captured_replay/{attempt_id}",
        "STRICT_PAIR_ENVIRONMENT": _exact_string(selection.get("environment"), name="Pair environment"),
        "STRICT_PREBUILT_SNAPSHOT_DIR": _canonical_absolute(on_snapshot.get("path"), name="Pair ON snapshot path"),
    }
    return {name: value.encode("ascii", "strict") for name, value in values.items()}


def _build_pair79_replay_export(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    expected_environment: str,
    expected_profile_id: str,
) -> Pair79ReplayExport:
    """Build the one admitted 79-record NUL export with no ambient values."""
    attempt = _attempt(attempt_id)
    pair, _ = _validate_pair_profile_before_publication(
        authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    values = {name: b"" for name in SLURM_EXPORT_ALLOWED_NAMES}
    nonempty = _pair79_nonempty_values(pair, attempt_id=attempt)
    if set(nonempty) != {
        "EXPECTED_GYM_GITLINK_COMMIT",
        "EXPECTED_GYM_TREE",
        "PAIR_ID",
        "RESULTS_DIR",
        "STRICT_PAIR_ENVIRONMENT",
        "STRICT_PREBUILT_SNAPSHOT_DIR",
    }:
        raise AssertionError("unreachable Pair79 nonempty value set differs")
    values.update(nonempty)
    records = tuple((name, values[name]) for name in SLURM_EXPORT_ALLOWED_NAMES)
    raw = b"\0".join(name.encode("ascii") + b"=" + value for name, value in records) + b"\0"
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    path = f"{results_root}/captured_replay/{PAIR79_EXPORT_DIRECTORY}/{pair_id}/{attempt}.env"
    return Pair79ReplayExport(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        records=records,
        raw=raw,
    )


def _open_absolute_directory_no_symlinks(path: str) -> int:
    canonical = _canonical_absolute(path, name="directory")
    descriptor = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in Path(canonical).parts[1:]:
            child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_private_directory(metadata: os.stat_result, *, name: str) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.geteuid():
        _fail(f"{name} must be an EUID-owned mode-0700 directory")


def _ensure_private_directory_chain(root: str, components: Sequence[str]) -> int:
    if type(components) is not tuple or any(type(item) is not str for item in components):
        _fail("private directory components must be an exact tuple of strings")
    descriptor = _open_absolute_directory_no_symlinks(root)
    try:
        _require_private_directory(os.fstat(descriptor), name="Pair results root")
        for component in components:
            if component in {"", ".", ".."} or "/" in component or "\x00" in component:
                _fail("private directory component is not canonical")
            created = False
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
                created = True
                os.fsync(descriptor)
            except FileExistsError:
                pass
            child = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            if created:
                os.fchmod(child, 0o700)
            _require_private_directory(os.fstat(child), name=f"private directory {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Pair79 export write made no progress")
        offset += written


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_exact_size(descriptor: int, *, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            _fail("stable file truncated while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        _fail("stable file grew while reading")
    return b"".join(chunks)


def _publish_pair79_replay_export(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: str,
    expected_environment: str,
    expected_profile_id: str,
    document: Pair79ReplayExport,
) -> tuple[Path, str]:
    """Exclusively publish one atomic EUID-owned mode-0400 Pair79 export."""
    if type(document) is not Pair79ReplayExport:
        _fail("Pair79 export document must have the exact coordinator type")
    expected = _build_pair79_replay_export(
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if document != expected:
        _fail("Pair79 export document differs from freshly derived authority")
    path = Path(_canonical_absolute(document.path, name="Pair79 export path"))
    pair = authenticated_source.pair_manifest
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    attempt = _attempt(attempt_id)
    parent_fd = _ensure_private_directory_chain(
        results_root,
        ("captured_replay", PAIR79_EXPORT_DIRECTORY, pair_id),
    )
    candidate_fd: int | None = None
    candidate_created = False
    candidate_name = f".{attempt}.env.candidate"
    try:
        candidate_fd = os.open(
            candidate_name,
            _FILE_CREATE_FLAGS,
            0o400,
            dir_fd=parent_fd,
        )
        candidate_created = True
        _write_all(candidate_fd, document.raw)
        os.fchmod(candidate_fd, 0o400)
        os.fsync(candidate_fd)
        metadata = os.fstat(candidate_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(document.raw)
        ):
            _fail("Pair79 export candidate inode differs")
        os.close(candidate_fd)
        candidate_fd = None
        try:
            os.link(
                candidate_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Pair79 export already exists: {path}") from error
        os.unlink(candidate_name, dir_fd=parent_fd)
        candidate_created = False
        os.fsync(parent_fd)
        final_fd = os.open(path.name, _FILE_READ_FLAGS, dir_fd=parent_fd)
        try:
            final_metadata = os.fstat(final_fd)
            actual = _read_exact_size(final_fd, size=len(document.raw))
        finally:
            os.close(final_fd)
        if (
            not stat.S_ISREG(final_metadata.st_mode)
            or stat.S_IMODE(final_metadata.st_mode) != 0o400
            or final_metadata.st_uid != os.geteuid()
            or final_metadata.st_nlink != 1
            or actual != document.raw
            or hashlib.sha256(actual).hexdigest() != document.sha256
        ):
            _fail("published Pair79 export failed exact verification")
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)
        if candidate_created:
            try:
                os.unlink(candidate_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    return path, document.sha256


def load_authenticated_replay_source(
    *,
    pair_manifest_path: str,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: str,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: str,
    trusted_off_exit_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> AuthenticatedOffSourceCapture:
    """Load the Pair and derive the completed OFF capability from public APIs."""
    pair_path = _canonical_absolute(pair_manifest_path, name="Pair manifest path")
    pair_sha = _digest(pair_manifest_sha256, name="Pair manifest SHA-256")
    receipt_path = _canonical_absolute(
        pair_submission_receipt_path,
        name="Pair submission receipt path",
    )
    receipt_sha = _digest(
        pair_submission_receipt_sha256,
        name="Pair submission receipt SHA-256",
    )
    exit_path = _canonical_absolute(
        trusted_off_exit_receipt_path,
        name="trusted OFF EXIT receipt path",
    )
    exit_sha = _digest(
        trusted_off_exit_receipt_sha256,
        name="trusted OFF EXIT receipt SHA-256",
    )
    _exact_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    pair, loaded_pair_sha = load_evidence_document(
        path=pair_path,
        expected_sha256=pair_sha,
        trailing_lf=True,
    )
    if type(pair) is not dict or loaded_pair_sha != pair_sha:
        _fail("public Pair loader returned noncanonical authority")
    source = load_authenticated_off_source_capture(
        pair_manifest=pair,
        pair_manifest_path=pair_path,
        pair_manifest_sha256=pair_sha,
        pair_submission_receipt_path=receipt_path,
        pair_submission_receipt_sha256=receipt_sha,
        trusted_off_exit_receipt_path=exit_path,
        trusted_off_exit_receipt_sha256=exit_sha,
    )
    _validate_pair_profile_before_publication(
        source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    return source


def _contract_path(pair: Mapping[str, Any], *, attempt_id: str) -> str:
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    return (
        f"{results_root}/captured_replay/{REPLAY_SUBMISSION_STATE_DIRECTORY}/"
        f"{pair_id}/{_attempt(attempt_id)}.submission-contract.json"
    )


def _manifest_path(pair: Mapping[str, Any], *, attempt_id: str) -> str:
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    return f"{results_root}/captured_replay/{REPLAY_MANIFEST_DIRECTORY}/" f"{pair_id}/{_attempt(attempt_id)}.json"


def _ensure_publication_parents(pair: Mapping[str, Any]) -> None:
    pair_id = _safe_id(pair.get("pair_id"), name="Pair ID")
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    for components in (
        ("captured_replay", REPLAY_SUBMISSION_STATE_DIRECTORY, pair_id),
        ("captured_replay", REPLAY_MANIFEST_DIRECTORY, pair_id),
    ):
        descriptor = _ensure_private_directory_chain(results_root, components)
        os.close(descriptor)


def _build_replay_launcher_argv(
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    replay_manifest_path: str,
    replay_manifest_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[str, ...]:
    """Build the exact 20-token V2 launcher tail from authenticated anchors."""
    pair, profile = _validate_pair_profile_before_publication(
        authenticated_source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    replacements = {
        "pair_manifest_path": f"{results_root}/PAIR_MANIFEST.json",
        "pair_manifest_sha256": _digest(
            authenticated_source.pair_manifest_sha256,
            name="Pair manifest SHA-256",
        ),
        "pair_submission_receipt_path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
        "pair_submission_receipt_sha256": _digest(
            authenticated_source.pair_submission_receipt_sha256,
            name="Pair submission receipt SHA-256",
        ),
        "trusted_off_exit_receipt_path": _canonical_absolute(
            authenticated_source.trusted_off_exit_receipt_path,
            name="trusted OFF EXIT receipt path",
        ),
        "trusted_off_exit_receipt_sha256": _digest(
            authenticated_source.trusted_off_exit_receipt_sha256,
            name="trusted OFF EXIT receipt SHA-256",
        ),
        "replay_manifest_path": _canonical_absolute(
            replay_manifest_path,
            name="replay manifest path",
        ),
        "replay_manifest_sha256": _digest(
            replay_manifest_sha256,
            name="replay manifest SHA-256",
        ),
        "environment": profile.environment,
        "profile_id": profile.profile_id,
    }
    if (
        type(REPLAY_JOB_ARGV_TEMPLATE_V2) is not tuple
        or len(REPLAY_JOB_ARGV_TEMPLATE_V2) != 20
        or any(type(token) is not str for token in REPLAY_JOB_ARGV_TEMPLATE_V2)
    ):
        _fail("V2 launcher argv template is not the exact 20-token tuple")
    argv = tuple(token.format_map(replacements) for token in REPLAY_JOB_ARGV_TEMPLATE_V2)
    if (
        len(argv) != 20
        or argv[::2]
        != (
            "--pair-manifest",
            "--pair-manifest-sha256",
            "--pair-submission-receipt",
            "--pair-submission-receipt-sha256",
            "--off-exit-receipt",
            "--off-exit-receipt-sha256",
            "--replay-manifest",
            "--replay-manifest-sha256",
            "--environment",
            "--profile-id",
        )
        or any(type(token) is not str or not token or "\x00" in token for token in argv)
    ):
        _fail("constructed V2 launcher argv differs from exact20")
    return argv


def prepare_replay_submission(
    *,
    pair_manifest_path: str,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: str,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: str,
    trusted_off_exit_receipt_sha256: str,
    attempt_id: str,
    submission_nonce: str,
    expected_environment: str,
    expected_profile_id: str,
) -> PreparedReplaySubmission:
    """Publish Pair79, contract, and manifest-v4 through public V2 boundaries."""
    attempt = _attempt(attempt_id)
    nonce = _digest(submission_nonce, name="submission nonce")
    source = load_authenticated_replay_source(
        pair_manifest_path=pair_manifest_path,
        pair_manifest_sha256=pair_manifest_sha256,
        pair_submission_receipt_path=pair_submission_receipt_path,
        pair_submission_receipt_sha256=pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=trusted_off_exit_receipt_sha256,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    pair, profile = _validate_pair_profile_before_publication(
        source,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    _ensure_publication_parents(pair)

    export_document = _build_pair79_replay_export(
        authenticated_source=source,
        attempt_id=attempt,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    export_path, export_sha256 = _publish_pair79_replay_export(
        authenticated_source=source,
        attempt_id=attempt,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
        document=export_document,
    )
    if str(export_path) != export_document.path or export_sha256 != export_document.sha256:
        raise AssertionError("unreachable Pair79 publisher identity mismatch")

    contract = build_replay_submission_contract(
        authenticated_source=source,
        attempt_id=attempt,
        submission_nonce=nonce,
    )
    if (
        type(contract) is not dict
        or contract.get("schema") != REPLAY_SUBMISSION_CONTRACT_SCHEMA
        or type(contract.get("schema")) is not str
    ):
        _fail("public replay contract builder returned the wrong schema")
    contract_path, contract_sha256 = publish_replay_submission_contract(
        authenticated_source=source,
        attempt_id=attempt,
        document=contract,
    )
    expected_contract_path = _contract_path(pair, attempt_id=attempt)
    if str(contract_path) != expected_contract_path:
        _fail("public replay contract publisher returned the wrong path")
    contract_sha256 = _digest(
        contract_sha256,
        name="published replay submission contract SHA-256",
    )

    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=attempt,
    )
    if type(static_inputs) is not AuthenticatedReplayStaticInputs:
        _fail("public static-input loader returned the wrong capability type")
    if (
        static_inputs.slurm_export_path != export_document.path
        or static_inputs.slurm_export_sha256 != export_document.sha256
        or static_inputs.submission_contract_path != expected_contract_path
        or static_inputs.submission_contract_sha256 != contract_sha256
    ):
        _fail("public static-input loader returned different immutable inputs")

    manifest = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id=attempt,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    if (
        type(manifest) is not dict
        or manifest.get("schema") != REPLAY_EXECUTION_MANIFEST_V2_SCHEMA
        or type(manifest.get("schema")) is not str
    ):
        _fail("public replay manifest builder returned the wrong schema")
    manifest_path = _manifest_path(pair, attempt_id=attempt)
    published_manifest_path, manifest_sha256 = publish_replay_execution_manifest_v2(
        output=manifest_path,
        document=manifest,
        authenticated_source=source,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    if str(published_manifest_path) != manifest_path:
        _fail("public replay manifest publisher returned the wrong path")
    manifest_sha256 = _digest(
        manifest_sha256,
        name="published replay manifest SHA-256",
    )
    loaded_manifest, loaded_manifest_sha256 = load_replay_execution_manifest_v2(
        path=manifest_path,
        expected_sha256=manifest_sha256,
        authenticated_source=source,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    if type(loaded_manifest) is not dict or loaded_manifest != manifest or loaded_manifest_sha256 != manifest_sha256:
        _fail("public replay manifest reload returned different immutable authority")

    launcher_reference = _exact_dict(
        static_inputs.submission_contract.get("submission_launcher"),
        name="authenticated submission launcher",
    )
    if set(launcher_reference) != {"path", "sha256"}:
        _fail("authenticated submission launcher reference keyset differs")
    launcher_path = _canonical_absolute(
        launcher_reference.get("path"),
        name="authenticated submission launcher path",
    )
    launcher_sha256 = _digest(
        launcher_reference.get("sha256"),
        name="authenticated submission launcher SHA-256",
    )
    source_snapshot = _exact_dict(
        static_inputs.source_snapshot,
        name="authenticated ON snapshot",
    )
    launcher_cwd = _canonical_absolute(
        source_snapshot.get("path"),
        name="authenticated ON snapshot path",
    )
    replay_program = _exact_dict(
        static_inputs.replay_program,
        name="authenticated replay program",
    )
    program_launcher = _exact_dict(
        replay_program.get("submission_launcher"),
        name="authenticated replay program submission launcher",
    )
    if set(program_launcher) != {"path", "sha256"}:
        _fail("authenticated replay program launcher keyset differs")
    program_launcher_path = _exact_string(
        program_launcher.get("path"),
        name="authenticated replay program launcher member",
    )
    program_launcher_sha256 = _digest(
        program_launcher.get("sha256"),
        name="authenticated replay program launcher SHA-256",
    )
    if (
        program_launcher_path != "strict_pair_replay_launch_v2.sh"
        or launcher_path != f"{launcher_cwd}/{program_launcher_path}"
        or launcher_sha256 != program_launcher_sha256
    ):
        _fail("submission launcher is not the authenticated V2 ON-snapshot member")
    launcher_argv = _build_replay_launcher_argv(
        authenticated_source=source,
        replay_manifest_path=manifest_path,
        replay_manifest_sha256=manifest_sha256,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    return PreparedReplaySubmission(
        pair_id=_safe_id(pair.get("pair_id"), name="Pair ID"),
        attempt_id=attempt,
        environment=profile.environment,
        profile_id=profile.profile_id,
        pair_manifest_path=f"{results_root}/PAIR_MANIFEST.json",
        pair_manifest_sha256=_digest(
            source.pair_manifest_sha256,
            name="Pair manifest SHA-256",
        ),
        pair_submission_receipt_path=f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
        pair_submission_receipt_sha256=_digest(
            source.pair_submission_receipt_sha256,
            name="Pair submission receipt SHA-256",
        ),
        trusted_off_exit_receipt_path=_canonical_absolute(
            source.trusted_off_exit_receipt_path,
            name="trusted OFF EXIT receipt path",
        ),
        trusted_off_exit_receipt_sha256=_digest(
            source.trusted_off_exit_receipt_sha256,
            name="trusted OFF EXIT receipt SHA-256",
        ),
        slurm_export_path=export_document.path,
        slurm_export_sha256=export_document.sha256,
        submission_contract_path=expected_contract_path,
        submission_contract_sha256=contract_sha256,
        replay_manifest_path=manifest_path,
        replay_manifest_sha256=manifest_sha256,
        launcher_path=launcher_path,
        launcher_sha256=launcher_sha256,
        launcher_cwd=launcher_cwd,
        launcher_argv=launcher_argv,
    )


def _validate_prepared_replay_submission(value: Any) -> PreparedReplaySubmission:
    if type(value) is not PreparedReplaySubmission:
        _fail("prepared replay must have the exact coordinator type")
    pair_id = _safe_id(value.pair_id, name="prepared Pair ID")
    attempt = _attempt(value.attempt_id)
    profile = _exact_profile(
        expected_environment=value.environment,
        expected_profile_id=value.profile_id,
    )
    pair_manifest_path = _canonical_absolute(
        value.pair_manifest_path,
        name="prepared Pair manifest path",
    )
    pair_suffix = "/PAIR_MANIFEST.json"
    if not pair_manifest_path.endswith(pair_suffix):
        _fail("prepared Pair manifest path differs from canonical results root")
    results_root = pair_manifest_path[: -len(pair_suffix)]
    _canonical_absolute(results_root, name="prepared Pair results root")
    expected_paths = {
        "pair_submission_receipt_path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
        "slurm_export_path": (f"{results_root}/captured_replay/{PAIR79_EXPORT_DIRECTORY}/" f"{pair_id}/{attempt}.env"),
        "submission_contract_path": (
            f"{results_root}/captured_replay/{REPLAY_SUBMISSION_STATE_DIRECTORY}/"
            f"{pair_id}/{attempt}.submission-contract.json"
        ),
        "replay_manifest_path": (
            f"{results_root}/captured_replay/{REPLAY_MANIFEST_DIRECTORY}/" f"{pair_id}/{attempt}.json"
        ),
    }
    for field, expected in expected_paths.items():
        actual = _canonical_absolute(
            getattr(value, field),
            name=f"prepared {field}",
        )
        if actual != expected:
            _fail(f"prepared {field} differs from authenticated identity")
    exit_path = _canonical_absolute(
        value.trusted_off_exit_receipt_path,
        name="prepared trusted OFF EXIT receipt path",
    )
    digest_fields = (
        "pair_manifest_sha256",
        "pair_submission_receipt_sha256",
        "trusted_off_exit_receipt_sha256",
        "slurm_export_sha256",
        "submission_contract_sha256",
        "replay_manifest_sha256",
        "launcher_sha256",
    )
    for field in digest_fields:
        _digest(getattr(value, field), name=f"prepared {field}")
    launcher_cwd = _canonical_absolute(
        value.launcher_cwd,
        name="prepared authenticated ON snapshot path",
    )
    launcher_path = _canonical_absolute(
        value.launcher_path,
        name="prepared submission launcher path",
    )
    if launcher_path != f"{launcher_cwd}/strict_pair_replay_launch_v2.sh":
        _fail("prepared submission launcher is not the authenticated V2 member")
    expected_argv = (
        "--pair-manifest",
        pair_manifest_path,
        "--pair-manifest-sha256",
        value.pair_manifest_sha256,
        "--pair-submission-receipt",
        value.pair_submission_receipt_path,
        "--pair-submission-receipt-sha256",
        value.pair_submission_receipt_sha256,
        "--off-exit-receipt",
        exit_path,
        "--off-exit-receipt-sha256",
        value.trusted_off_exit_receipt_sha256,
        "--replay-manifest",
        value.replay_manifest_path,
        "--replay-manifest-sha256",
        value.replay_manifest_sha256,
        "--environment",
        profile.environment,
        "--profile-id",
        profile.profile_id,
    )
    if (
        type(value.launcher_argv) is not tuple
        or len(value.launcher_argv) != 20
        or any(type(token) is not str for token in value.launcher_argv)
        or value.launcher_argv != expected_argv
    ):
        _fail("prepared submission launcher argv differs from exact20 authority")
    return value


def _validate_prepared_source_binding(
    prepared: PreparedReplaySubmission,
    source: AuthenticatedOffSourceCapture,
) -> tuple[dict[str, Any], StrictCapturedReplayProfile]:
    pair, profile = _validate_pair_profile_before_publication(
        source,
        expected_environment=prepared.environment,
        expected_profile_id=prepared.profile_id,
    )
    results_root = _canonical_absolute(
        _exact_dict(pair.get("paths"), name="authenticated Pair paths").get("results_root"),
        name="Pair results root",
    )
    expected = {
        "pair_id": _safe_id(pair.get("pair_id"), name="authenticated Pair ID"),
        "pair_manifest_path": f"{results_root}/PAIR_MANIFEST.json",
        "pair_manifest_sha256": _digest(
            source.pair_manifest_sha256,
            name="authenticated Pair manifest SHA-256",
        ),
        "pair_submission_receipt_path": (f"{results_root}/PAIR_SUBMISSION_RECEIPT.json"),
        "pair_submission_receipt_sha256": _digest(
            source.pair_submission_receipt_sha256,
            name="authenticated Pair submission receipt SHA-256",
        ),
        "trusted_off_exit_receipt_path": _canonical_absolute(
            source.trusted_off_exit_receipt_path,
            name="authenticated trusted OFF EXIT receipt path",
        ),
        "trusted_off_exit_receipt_sha256": _digest(
            source.trusted_off_exit_receipt_sha256,
            name="authenticated trusted OFF EXIT receipt SHA-256",
        ),
    }
    if any(getattr(prepared, field) != value for field, value in expected.items()):
        _fail("prepared source anchors differ from authenticated OFF authority")
    return pair, profile


def _validate_source_arguments_match_prepared(
    prepared: PreparedReplaySubmission,
    *,
    pair_manifest_path: Any,
    pair_manifest_sha256: Any,
    pair_submission_receipt_path: Any,
    pair_submission_receipt_sha256: Any,
    trusted_off_exit_receipt_path: Any,
    trusted_off_exit_receipt_sha256: Any,
    expected_environment: Any,
    expected_profile_id: Any,
) -> None:
    profile = _exact_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    provided = {
        "pair_manifest_path": _canonical_absolute(
            pair_manifest_path,
            name="Pair manifest path",
        ),
        "pair_manifest_sha256": _digest(
            pair_manifest_sha256,
            name="Pair manifest SHA-256",
        ),
        "pair_submission_receipt_path": _canonical_absolute(
            pair_submission_receipt_path,
            name="Pair submission receipt path",
        ),
        "pair_submission_receipt_sha256": _digest(
            pair_submission_receipt_sha256,
            name="Pair submission receipt SHA-256",
        ),
        "trusted_off_exit_receipt_path": _canonical_absolute(
            trusted_off_exit_receipt_path,
            name="trusted OFF EXIT receipt path",
        ),
        "trusted_off_exit_receipt_sha256": _digest(
            trusted_off_exit_receipt_sha256,
            name="trusted OFF EXIT receipt SHA-256",
        ),
        "environment": profile.environment,
        "profile_id": profile.profile_id,
    }
    if any(getattr(prepared, field) != value for field, value in provided.items()):
        _fail("source arguments differ from prepared authenticated authority")


def _reauthenticate_prepared_submission(
    prepared: PreparedReplaySubmission,
) -> tuple[AuthenticatedOffSourceCapture, dict[str, Any]]:
    """Rebuild authority from immutable public inputs before any subprocess."""
    prepared = _validate_prepared_replay_submission(prepared)
    source = load_authenticated_replay_source(
        pair_manifest_path=prepared.pair_manifest_path,
        pair_manifest_sha256=prepared.pair_manifest_sha256,
        pair_submission_receipt_path=prepared.pair_submission_receipt_path,
        pair_submission_receipt_sha256=prepared.pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=prepared.trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=prepared.trusted_off_exit_receipt_sha256,
        expected_environment=prepared.environment,
        expected_profile_id=prepared.profile_id,
    )
    pair, profile = _validate_prepared_source_binding(prepared, source)
    static_inputs = load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=prepared.attempt_id,
    )
    if type(static_inputs) is not AuthenticatedReplayStaticInputs:
        _fail("public static-input loader returned the wrong capability type")
    if (
        type(static_inputs.attempt_id) is not str
        or static_inputs.attempt_id != prepared.attempt_id
        or static_inputs.slurm_export_path != prepared.slurm_export_path
        or static_inputs.slurm_export_sha256 != prepared.slurm_export_sha256
        or static_inputs.submission_contract_path != prepared.submission_contract_path
        or static_inputs.submission_contract_sha256 != prepared.submission_contract_sha256
    ):
        _fail("prepared immutable inputs differ after public reauthentication")
    snapshot = _exact_dict(
        static_inputs.source_snapshot,
        name="reauthenticated ON snapshot",
    )
    snapshot_root = _canonical_absolute(
        snapshot.get("path"),
        name="reauthenticated ON snapshot path",
    )
    program = _exact_dict(
        static_inputs.replay_program,
        name="reauthenticated replay program",
    )
    program_launcher = _exact_dict(
        program.get("submission_launcher"),
        name="reauthenticated replay program launcher",
    )
    contract = _exact_dict(
        static_inputs.submission_contract,
        name="reauthenticated replay submission contract",
    )
    contract_launcher = _exact_dict(
        contract.get("submission_launcher"),
        name="reauthenticated contract launcher",
    )
    if (
        snapshot_root != prepared.launcher_cwd
        or program_launcher
        != {
            "path": "strict_pair_replay_launch_v2.sh",
            "sha256": prepared.launcher_sha256,
        }
        or contract_launcher
        != {
            "path": prepared.launcher_path,
            "sha256": prepared.launcher_sha256,
        }
        or prepared.launcher_path != f"{snapshot_root}/{program_launcher.get('path', '')}"
    ):
        _fail("prepared launcher differs from public static reauthentication")
    manifest, manifest_sha256 = load_replay_execution_manifest_v2(
        path=prepared.replay_manifest_path,
        expected_sha256=prepared.replay_manifest_sha256,
        authenticated_source=source,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    if type(manifest) is not dict or manifest_sha256 != prepared.replay_manifest_sha256:
        _fail("public replay manifest loader returned different submission authority")
    manifest_attempt = _attempt(manifest.get("attempt_id"))
    scorer_profile = _exact_dict(
        manifest.get("scorer_profile"),
        name="replay manifest scorer profile",
    )
    if (
        _safe_id(manifest.get("pair_id"), name="replay manifest Pair ID") != prepared.pair_id
        or manifest_attempt != prepared.attempt_id
        or type(manifest.get("environment")) is not str
        or manifest["environment"] != profile.environment
        or type(scorer_profile.get("profile_id")) is not str
        or scorer_profile["profile_id"] != profile.profile_id
        or prepared.replay_manifest_path != _manifest_path(pair, attempt_id=manifest_attempt)
    ):
        _fail("prepared replay manifest identity/path differs from authenticated authority")
    return source, manifest


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mode,
        metadata.st_nlink,
    )


def _require_submission_launcher_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        _fail("submission launcher must be one EUID-owned single-link mode-0555 file")


def _require_submission_launcher_parent(metadata: os.stat_result) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        _fail("submission launcher parent must be an EUID-owned directory")


def _validate_open_submission_launcher(
    *,
    descriptor: int,
    parent_descriptor: int,
    canonical: Path,
    expected_sha256: str,
    expected_file_fingerprint: tuple[int, ...],
    expected_parent_fingerprint: tuple[int, ...],
) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    opened = os.fstat(descriptor)
    _require_submission_launcher_metadata(opened)
    raw = _read_exact_size(descriptor, size=opened.st_size)
    after = os.fstat(descriptor)
    named = os.stat(canonical.name, dir_fd=parent_descriptor, follow_symlinks=False)
    parent_after = os.fstat(parent_descriptor)
    fresh_parent = _open_absolute_directory_no_symlinks(str(canonical.parent))
    try:
        fresh_parent_metadata = os.fstat(fresh_parent)
        fresh_named = os.stat(
            canonical.name,
            dir_fd=fresh_parent,
            follow_symlinks=False,
        )
    finally:
        os.close(fresh_parent)
    _require_submission_launcher_metadata(named)
    _require_submission_launcher_metadata(fresh_named)
    _require_submission_launcher_parent(parent_after)
    _require_submission_launcher_parent(fresh_parent_metadata)
    opened_fingerprint = _metadata_fingerprint(opened)
    if (
        expected_file_fingerprint != opened_fingerprint
        or opened_fingerprint != _metadata_fingerprint(after)
        or opened_fingerprint != _metadata_fingerprint(named)
        or opened_fingerprint != _metadata_fingerprint(fresh_named)
        or expected_parent_fingerprint != _metadata_fingerprint(parent_after)
        or expected_parent_fingerprint != _metadata_fingerprint(fresh_parent_metadata)
        or len(raw) != opened.st_size
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        _fail("submission launcher differs from authenticated stable bytes")
    return raw


def _open_authenticated_submission_launcher(
    path: str,
    *,
    expected_sha256: str,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], bytes]:
    """Open and retain the exact launcher inode that the coordinator executes."""
    canonical = Path(_canonical_absolute(path, name="submission launcher path"))
    expected = _digest(expected_sha256, name="submission launcher SHA-256")
    parent_descriptor = _open_absolute_directory_no_symlinks(str(canonical.parent))
    descriptor: int | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        _require_submission_launcher_parent(parent_metadata)
        metadata = os.stat(
            canonical.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _require_submission_launcher_metadata(metadata)
        descriptor = os.open(
            canonical.name,
            _FILE_READ_FLAGS,
            dir_fd=parent_descriptor,
        )
        file_fingerprint = _metadata_fingerprint(metadata)
        parent_fingerprint = _metadata_fingerprint(parent_metadata)
        raw = _validate_open_submission_launcher(
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            canonical=canonical,
            expected_sha256=expected,
            expected_file_fingerprint=file_fingerprint,
            expected_parent_fingerprint=parent_fingerprint,
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return (
            descriptor,
            parent_descriptor,
            file_fingerprint,
            parent_fingerprint,
            raw,
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise


def _submission_launcher_fd_path(descriptor: int) -> str:
    if type(descriptor) is not int or descriptor < 3:
        _fail("authenticated submission launcher descriptor is invalid")
    if os.path.isdir("/proc/self/fd"):
        return f"/proc/self/fd/{descriptor}"
    if os.path.isdir("/dev/fd"):
        return f"/dev/fd/{descriptor}"
    _fail("platform has no authenticated descriptor execution namespace")


def _validate_replay_bash_authority(source: AuthenticatedOffSourceCapture) -> None:
    pair = _exact_dict(source.pair_manifest, name="authenticated Pair manifest")
    boundary = _exact_dict(
        pair.get("container_entry_boundary"),
        name="authenticated container entry boundary",
    )
    if boundary.get("bash_path") != _REPLAY_BASH_PATH or boundary.get("bash_args") != ["-p"]:
        _fail("authenticated Pair replay Bash boundary differs")
    metadata = os.stat(_REPLAY_BASH_PATH, follow_symlinks=True)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not stat.S_IMODE(metadata.st_mode) & 0o111
    ):
        _fail("authenticated replay Bash is not one root-owned protected executable")


def _sealed_submission_launcher_copy(raw: bytes) -> int:
    """Copy authenticated bytes to an anonymous immutable execution inode."""
    if type(raw) is not bytes or not 1 <= len(raw) <= 64 * 1024 * 1024:
        _fail("authenticated submission launcher bytes violate size/type bounds")
    if hasattr(os, "memfd_create") and hasattr(os, "MFD_ALLOW_SEALING"):
        descriptor = os.memfd_create(
            "nemo-rl-strict-replay-launcher",
            flags=os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
        seal = True
    else:
        descriptor, candidate = tempfile.mkstemp(prefix="nemo-rl-strict-replay-")
        os.unlink(candidate)
        seal = False
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("anonymous submission launcher write made no progress")
            offset += written
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        if seal:
            required_seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
            if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals != required_seals:
                _fail("anonymous submission launcher seals differ")
        os.lseek(descriptor, 0, os.SEEK_SET)
        metadata = os.fstat(descriptor)
        retained = _read_exact_size(descriptor, size=len(raw))
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 0
            or retained != raw
        ):
            _fail("anonymous submission launcher differs from authenticated bytes")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reject_json_constant(value: str) -> None:
    _fail(f"launcher stdout contains a non-finite JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"launcher stdout contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_launcher_receipt(raw: bytes, *, prepared: PreparedReplaySubmission) -> dict[str, str]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or len(raw) > 64 * 1024:
        _fail("launcher stdout must be one bounded canonical JSON line")
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictCapturedReplayCoordinatorError("launcher stdout is not strict UTF-8 JSON") from error
    if type(document) is not dict or set(document) != set(_LAUNCH_RECEIPT_KEYS):
        _fail("launcher stdout receipt keyset differs")
    if any(type(key) is not str or type(value) is not str for key, value in document.items()):
        _fail("launcher stdout receipt fields must be exact strings")
    if (
        document["attempt_id"] != prepared.attempt_id
        or document["pair_id"] != prepared.pair_id
        or _JOB_ID.fullmatch(document["candidate_job_id"]) is None
        or _DIGEST.fullmatch(document["submission_receipt_sha256"]) is None
    ):
        _fail("launcher stdout receipt identity differs")
    expected = (json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if raw != expected:
        _fail("launcher stdout receipt is not canonical ASCII JSON")
    return document


def _run_bounded_launcher(
    argv: list[str],
    *,
    pass_fds: tuple[int, ...],
) -> subprocess.CompletedProcess[bytes]:
    """Run the launcher with hard output bounds and a finite deadline."""
    process = subprocess.Popen(
        argv,
        cwd=_REPLAY_COORDINATOR_CWD,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(_LAUNCH_ENVIRONMENT),
        close_fds=True,
        pass_fds=pass_fds,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        process.wait()
        _fail("submission launcher pipes were not created")
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    streams = {
        process.stdout.fileno(): (process.stdout, _LAUNCH_STDOUT_LIMIT, stdout_buffer),
        process.stderr.fileno(): (process.stderr, _LAUNCH_STDERR_LIMIT, stderr_buffer),
    }
    selector = selectors.DefaultSelector()
    for descriptor, (stream, _limit, _buffer) in streams.items():
        os.set_blocking(descriptor, False)
        selector.register(stream, selectors.EVENT_READ, descriptor)
    deadline = time.monotonic() + _LAUNCH_TIMEOUT_SECONDS
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("submission launcher exceeded its execution deadline")
            events = selector.select(remaining)
            if not events:
                _fail("submission launcher exceeded its execution deadline")
            for key, _mask in events:
                descriptor = key.data
                stream, limit, buffer = streams[descriptor]
                chunk = os.read(descriptor, min(64 * 1024, limit + 1 - len(buffer)))
                if not chunk:
                    selector.unregister(stream)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limit:
                    _fail("submission launcher output exceeded its strict bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail("submission launcher exceeded its execution deadline")
        returncode = process.wait(timeout=remaining)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=bytes(stdout_buffer),
        stderr=bytes(stderr_buffer),
    )


def invoke_replay_launcher(prepared: PreparedReplaySubmission) -> dict[str, str]:
    """Invoke the authenticated launcher with an exact clean environment.

    This call does not claim atomicity between Slurm release and receipt
    delivery; see the module-level scheduler commit/ack boundary.
    """
    prepared = _validate_prepared_replay_submission(prepared)
    source, _manifest = _reauthenticate_prepared_submission(prepared)
    _validate_replay_bash_authority(source)
    descriptor, parent_descriptor, file_fingerprint, parent_fingerprint, raw = _open_authenticated_submission_launcher(
        prepared.launcher_path,
        expected_sha256=prepared.launcher_sha256,
    )
    try:
        sealed_descriptor = _sealed_submission_launcher_copy(raw)
    except BaseException:
        os.close(descriptor)
        os.close(parent_descriptor)
        raise
    try:
        completed = _run_bounded_launcher(
            [
                _REPLAY_BASH_PATH,
                "-p",
                _submission_launcher_fd_path(sealed_descriptor),
                *prepared.launcher_argv,
            ],
            pass_fds=(sealed_descriptor,),
        )
        _validate_open_submission_launcher(
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            canonical=Path(prepared.launcher_path),
            expected_sha256=prepared.launcher_sha256,
            expected_file_fingerprint=file_fingerprint,
            expected_parent_fingerprint=parent_fingerprint,
        )
    finally:
        os.close(sealed_descriptor)
        os.close(descriptor)
        os.close(parent_descriptor)
    if completed.returncode != 0:
        detail = completed.stderr[:16_384].decode("utf-8", "replace")
        _fail(f"submission launcher exited {completed.returncode}: {detail}")
    if completed.stderr not in {b"", None}:
        _fail("successful submission launcher wrote stderr")
    return _parse_launcher_receipt(completed.stdout, prepared=prepared)


def load_released_submission_receipt(
    *,
    prepared: PreparedReplaySubmission,
    launcher_receipt: Mapping[str, Any],
    pair_manifest_path: str,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: str,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: str,
    trusted_off_exit_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    """Reload the launcher's published receipt through the public V2 loader."""
    prepared = _validate_prepared_replay_submission(prepared)
    if type(launcher_receipt) is not dict:
        _fail("launcher receipt must be an exact dictionary")
    _validate_source_arguments_match_prepared(
        prepared,
        pair_manifest_path=pair_manifest_path,
        pair_manifest_sha256=pair_manifest_sha256,
        pair_submission_receipt_path=pair_submission_receipt_path,
        pair_submission_receipt_sha256=pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=trusted_off_exit_receipt_sha256,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    receipt = _parse_launcher_receipt(
        (
            json.dumps(
                launcher_receipt,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii"),
        prepared=prepared,
    )
    source, manifest = _reauthenticate_prepared_submission(prepared)
    scheduler_submission = _exact_dict(
        manifest.get("scheduler_submission"),
        name="replay manifest scheduler submission",
    )
    receipt_reference = _exact_dict(
        scheduler_submission.get("receipt"),
        name="replay submission receipt reference",
    )
    receipt_path = _canonical_absolute(
        receipt_reference.get("path"),
        name="replay submission receipt path",
    )
    document, loaded_receipt_sha256 = load_captured_replay_submission_receipt_v2(
        path=receipt_path,
        expected_sha256=receipt["submission_receipt_sha256"],
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=prepared.environment,
        expected_profile_id=prepared.profile_id,
    )
    if type(document) is not dict or loaded_receipt_sha256 != receipt["submission_receipt_sha256"]:
        _fail("public replay submission-receipt loader returned different authority")
    if (
        type(document.get("candidate_job_id")) is not str
        or document["candidate_job_id"] != receipt["candidate_job_id"]
        or type(document.get("pair_id")) is not str
        or document["pair_id"] != prepared.pair_id
        or type(document.get("attempt_id")) is not str
        or document["attempt_id"] != prepared.attempt_id
    ):
        _fail("authenticated submission receipt identity differs from launcher stdout")
    return document


def submit_replay(**kwargs: str) -> SubmittedReplay:
    """Prepare immutable replay inputs and release exactly one held job."""
    prepared = prepare_replay_submission(**kwargs)
    receipt = invoke_replay_launcher(prepared)
    submission_receipt = load_released_submission_receipt(
        prepared=prepared,
        launcher_receipt=receipt,
        **{
            name: kwargs[name]
            for name in (
                "pair_manifest_path",
                "pair_manifest_sha256",
                "pair_submission_receipt_path",
                "pair_submission_receipt_sha256",
                "trusted_off_exit_receipt_path",
                "trusted_off_exit_receipt_sha256",
                "expected_environment",
                "expected_profile_id",
            )
        },
    )
    return SubmittedReplay(
        prepared=prepared,
        launcher_receipt=receipt,
        submission_receipt=submission_receipt,
    )


def _bounded_positive_int(value: Any, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(f"{name} must be one exact bounded positive integer")
    return value


def _bounded_nonnegative_int(value: Any, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(f"{name} must be one exact bounded nonnegative integer")
    return value


def _snapshot_artifact_reference(
    value: Any,
    *,
    name: str,
    expected_path: str,
    expected_schema: str,
) -> dict[str, Any]:
    reference = _exact_dict(value, name=name)
    if set(reference) != {"path", "schema", "sha256"}:
        _fail(f"{name} keyset differs")
    actual_path = _canonical_absolute(reference.get("path"), name=f"{name} path")
    schema = _exact_string(reference.get("schema"), name=f"{name} schema")
    digest = _digest(reference.get("sha256"), name=f"{name} SHA-256")
    if digest == "0" * 64:
        _fail(f"{name} SHA-256 must be nonzero")
    if actual_path != expected_path or schema != expected_schema:
        _fail(f"{name} path/schema differs")
    return reference


def _validate_authenticated_result_snapshot(
    value: Any,
    *,
    replay_manifest_path: str,
    replay_manifest_sha256: str,
    submission_receipt_sha256: str,
    candidate_job_id: str,
    result_final_receipt_path: str,
    result_final_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
    authenticated_source: AuthenticatedOffSourceCapture,
) -> dict[str, Any]:
    if type(authenticated_source) is not AuthenticatedOffSourceCapture:
        _fail("authenticated source must have the exact OFF-source capability type")
    snapshot = _exact_dict(value, name="authenticated result snapshot")
    if set(snapshot) != set(_AUTHENTICATED_RESULT_SNAPSHOT_KEYS):
        _fail("authenticated result snapshot keyset differs")
    if type(snapshot.get("schema")) is not str or snapshot["schema"] != AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA:
        _fail("authenticated result snapshot schema differs")

    profile = _exact_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    pair = _exact_dict(
        authenticated_source.pair_manifest,
        name="authenticated Pair manifest",
    )
    pair_id = _safe_id(pair.get("pair_id"), name="authenticated Pair ID")
    paths = _exact_dict(pair.get("paths"), name="authenticated Pair paths")
    results_root = _canonical_absolute(
        paths.get("results_root"),
        name="authenticated Pair results root",
    )
    if type(snapshot.get("pair_id")) is not str or snapshot["pair_id"] != pair_id:
        _fail("authenticated result snapshot Pair ID differs")
    attempt = _attempt(snapshot.get("attempt_id"))
    expected_manifest_path = _manifest_path(pair, attempt_id=attempt)
    expected_result_root = f"{results_root}/captured_replay/{attempt}"
    submission_root = f"{results_root}/captured_replay/{REPLAY_SUBMISSION_STATE_DIRECTORY}/" f"{pair_id}/{attempt}"

    expected_candidate_job_id = _exact_string(
        candidate_job_id,
        name="candidate job ID",
    )
    if _JOB_ID.fullmatch(expected_candidate_job_id) is None or int(expected_candidate_job_id) > (1 << 63) - 1:
        _fail("candidate job ID must be one canonical positive decimal string")
    receipt_root = (
        f"{results_root}/captured_replay/replay_job_state/{pair_id}/{attempt}/"
        f"{expected_candidate_job_id}-0/receipts"
    )
    expected = {
        "environment": profile.environment,
        "profile_id": profile.profile_id,
        "candidate_job_id": expected_candidate_job_id,
        "authenticated_job_id": expected_candidate_job_id,
    }
    if any(
        type(snapshot.get(name)) is not str or snapshot[name] != expected_value
        for name, expected_value in expected.items()
    ):
        _fail("authenticated result snapshot identity differs")
    expected_run_id = hashlib.sha256(
        (f"nemo-rl-strict-replay-v2:{profile.environment}:" f"{pair_id}:{attempt}").encode("ascii")
    ).hexdigest()
    if type(snapshot.get("run_id")) is not str or snapshot["run_id"] != expected_run_id:
        _fail("authenticated result snapshot run ID differs")

    driver_process = _exact_dict(
        snapshot.get("driver_process"),
        name="authenticated result driver process",
    )
    if set(driver_process) != {"boot_id_sha256", "pid", "start_time_ticks"}:
        _fail("authenticated result driver process keyset differs")
    driver_boot_id = _digest(
        driver_process.get("boot_id_sha256"),
        name="authenticated result driver boot ID SHA-256",
    )
    if driver_boot_id == "0" * 64:
        _fail("authenticated result driver boot ID SHA-256 must be nonzero")
    driver_pid = _bounded_positive_int(
        driver_process.get("pid"),
        name="authenticated result driver PID",
        maximum=(1 << 31) - 1,
    )
    driver_start_ticks = _bounded_positive_int(
        driver_process.get("start_time_ticks"),
        name="authenticated result driver start ticks",
        maximum=(1 << 63) - 1,
    )
    scorer_process = _exact_dict(
        snapshot.get("scorer_process_identity"),
        name="authenticated result scorer process identity",
    )
    if set(scorer_process) != {"boot_id", "hostname", "pid", "start_ticks"}:
        _fail("authenticated result scorer process identity keyset differs")
    boot_id = _exact_string(
        scorer_process.get("boot_id"),
        name="authenticated result scorer boot ID",
    )
    if _BOOT_ID.fullmatch(boot_id) is None:
        _fail("authenticated result scorer boot ID differs")
    hostname = _exact_string(
        scorer_process.get("hostname"),
        name="authenticated result scorer hostname",
    )
    if len(hostname.encode("ascii", "strict")) > 255:
        _fail("authenticated result scorer hostname is too long")
    scorer_pid = _bounded_positive_int(
        scorer_process.get("pid"),
        name="authenticated result scorer PID",
        maximum=(1 << 31) - 1,
    )
    scorer_start_ticks = _bounded_positive_int(
        scorer_process.get("start_ticks"),
        name="authenticated result scorer start ticks",
        maximum=(1 << 63) - 1,
    )
    expected_driver_boot_id = hashlib.sha256((boot_id + "\n").encode("ascii")).hexdigest()
    if driver_boot_id != expected_driver_boot_id:
        _fail("authenticated result driver/scorer boot identity differs")
    if driver_pid == scorer_pid or (
        driver_pid,
        driver_start_ticks,
    ) == (
        scorer_pid,
        scorer_start_ticks,
    ):
        _fail("authenticated result driver/scorer process identity aliases")

    manifest_reference = _snapshot_artifact_reference(
        snapshot.get("manifest"),
        name="authenticated result manifest",
        expected_path=expected_manifest_path,
        expected_schema=REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    )
    submission_reference = _snapshot_artifact_reference(
        snapshot.get("submission_receipt"),
        name="authenticated result submission receipt",
        expected_path=f"{submission_root}/submission-receipt.json",
        expected_schema=REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    )
    _snapshot_artifact_reference(
        snapshot.get("pre_receipt"),
        name="authenticated result PRE receipt",
        expected_path=f"{receipt_root}/PRE.json",
        expected_schema=REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    )
    _snapshot_artifact_reference(
        snapshot.get("exit_receipt"),
        name="authenticated result EXIT receipt",
        expected_path=f"{receipt_root}/EXIT.json",
        expected_schema=REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    )
    final_reference = _snapshot_artifact_reference(
        snapshot.get("result_final_receipt"),
        name="authenticated result FINAL receipt",
        expected_path=f"{receipt_root}/FINAL.json",
        expected_schema=REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
    )
    _snapshot_artifact_reference(
        snapshot.get("result_inventory"),
        name="authenticated result inventory",
        expected_path=f"{expected_result_root}/result-inventory-v2.json",
        expected_schema=_RESULT_INVENTORY_V2_SCHEMA,
    )
    _snapshot_artifact_reference(
        snapshot.get("evidence_index"),
        name="authenticated result evidence index",
        expected_path=f"{expected_result_root}/evidence-index.json",
        expected_schema=REPLAY_POST_INDEX_V2_SCHEMA,
    )
    expected_references = (
        (
            manifest_reference,
            _canonical_absolute(
                replay_manifest_path,
                name="replay manifest path",
            ),
            _digest(
                replay_manifest_sha256,
                name="replay manifest SHA-256",
            ),
        ),
        (
            final_reference,
            _canonical_absolute(
                result_final_receipt_path,
                name="result FINAL receipt path",
            ),
            _digest(
                result_final_receipt_sha256,
                name="result FINAL receipt SHA-256",
            ),
        ),
    )
    if any(
        reference["path"] != expected_path or reference["sha256"] != expected_sha256
        for reference, expected_path, expected_sha256 in expected_references
    ):
        _fail("authenticated result snapshot OOB reference differs")
    if submission_reference["sha256"] != _digest(
        submission_receipt_sha256,
        name="submission receipt SHA-256",
    ):
        _fail("authenticated result snapshot submission receipt differs")

    result_root = _canonical_absolute(
        snapshot.get("result_root"),
        name="authenticated result root",
    )
    if result_root != expected_result_root:
        _fail("authenticated result snapshot result root differs")
    outputs = _exact_dict(
        snapshot.get("outputs"),
        name="authenticated result outputs",
    )
    if set(outputs) != set(_AUTHENTICATED_RESULT_OUTPUT_KEYS):
        _fail("authenticated result outputs keyset differs")
    output_authority = {
        "scorer_call_index": (
            f"{result_root}/{profile.scorer_terminal_index_path}",
            profile.call_index_schema,
        ),
        "transport_consumption": (
            f"{result_root}/model-transport-replay-consumption.json",
            _REPLAY_TRANSPORT_CONSUMPTION_V3_SCHEMA,
        ),
        "transcript_bundle": (
            f"{result_root}/transcript-bundle.json",
            _REPLAY_TRANSCRIPT_BUNDLE_V4_SCHEMA,
        ),
        "replay_ledger": (
            f"{result_root}/replay-ledger.json",
            _REPLAY_LEDGER_V5_SCHEMA,
        ),
    }
    for name in sorted(_AUTHENTICATED_RESULT_OUTPUT_KEYS):
        output_path, output_schema = output_authority[name]
        _snapshot_artifact_reference(
            outputs[name],
            name=f"authenticated result output {name}",
            expected_path=output_path,
            expected_schema=output_schema,
        )
    samples = snapshot.get("samples")
    if type(samples) is not list or len(samples) != 4:
        _fail("authenticated result snapshot must contain exact K=4 samples")
    for index, sample_value in enumerate(samples):
        sample = _exact_dict(
            sample_value,
            name=f"authenticated result sample {index}",
        )
        if set(sample) != set(_AUTHENTICATED_RESULT_SAMPLE_KEYS):
            _fail(f"authenticated result sample {index} keyset differs")
        for name, expected_value in (
            ("sample_index", index),
            ("fixture_row_index", 0),
            ("rollout_index", index),
        ):
            if type(sample.get(name)) is not int or sample[name] != expected_value:
                _fail(f"authenticated result sample {index} {name} differs")
        _bounded_nonnegative_int(
            sample.get("generation_seed"),
            name=f"authenticated result sample {index} generation seed",
            maximum=(1 << 63) - 1,
        )
        for name in (
            "model_transport_entry_sha256",
            "model_transport_request_body_sha256",
            "model_transport_response_body_sha256",
            "model_response_sha256",
        ):
            sample_digest = _digest(
                sample.get(name),
                name=f"authenticated result sample {index} {name}",
            )
            if sample_digest == "0" * 64:
                _fail(f"authenticated result sample {index} {name} must be nonzero")
        details = _exact_dict(
            sample.get("match_details"),
            name=f"authenticated result sample {index} match details",
        )
        if profile.environment == "citation":
            if set(details) != {"expected", "missing", "spurious", "passed"}:
                _fail(f"authenticated citation sample {index} details differ")
            for name in ("expected", "missing", "spurious"):
                strings = details.get(name)
                if type(strings) is not list or any(type(item) is not str for item in strings):
                    _fail(f"authenticated citation sample {index} {name} differs")
            passed = details.get("passed")
            if type(passed) is not bool or passed is not (not details["missing"] and not details["spurious"]):
                _fail(f"authenticated citation sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif profile.environment == "freeform":
            if set(details) != {"matching_lines", "min_matches", "passed"}:
                _fail(f"authenticated freeform sample {index} details differ")
            matching_lines = _bounded_nonnegative_int(
                details.get("matching_lines"),
                name=f"authenticated freeform sample {index} matching lines",
                maximum=(1 << 31) - 1,
            )
            minimum = _bounded_nonnegative_int(
                details.get("min_matches"),
                name=f"authenticated freeform sample {index} minimum matches",
                maximum=(1 << 31) - 1,
            )
            passed = details.get("passed")
            if type(passed) is not bool or passed is not (matching_lines >= minimum):
                _fail(f"authenticated freeform sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif profile.environment == "reasoning_gym":
            if set(details) != {"task_name", "score", "extracted_answer"}:
                _fail(f"authenticated reasoning-gym sample {index} details differ")
            if details.get("task_name") != "knights_knaves" or type(details.get("task_name")) is not str:
                _fail(f"authenticated reasoning-gym sample {index} task differs")
            score = details.get("score")
            if (
                type(score) is not float
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
                or (score == 0.0 and math.copysign(1.0, score) < 0.0)
            ):
                _fail(f"authenticated reasoning-gym sample {index} score differs")
            if type(details.get("extracted_answer")) is not str:
                _fail(f"authenticated reasoning-gym sample {index} extracted answer differs")
            expected_reward = score
        else:  # pragma: no cover - the closed profile registry rejects this.
            raise AssertionError("unreachable authenticated scorer profile")
        reward = sample.get("raw_environment_reward")
        if (
            type(reward) is not float
            or not math.isfinite(reward)
            or (reward == 0.0 and math.copysign(1.0, reward) < 0.0)
            or not 0.0 <= reward <= 1.0
            or reward != expected_reward
        ):
            _fail(f"authenticated result sample {index} reward differs")
    return snapshot


def consume_replay_result(
    *,
    pair_manifest_path: str,
    pair_manifest_sha256: str,
    pair_submission_receipt_path: str,
    pair_submission_receipt_sha256: str,
    trusted_off_exit_receipt_path: str,
    trusted_off_exit_receipt_sha256: str,
    replay_manifest_path: str,
    replay_manifest_sha256: str,
    submission_receipt_sha256: str,
    candidate_job_id: str,
    result_final_receipt_path: str,
    result_final_receipt_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> ConsumedReplayResult:
    """Authenticate one wrapper-published FINAL and detach its result snapshot."""
    source = load_authenticated_replay_source(
        pair_manifest_path=pair_manifest_path,
        pair_manifest_sha256=pair_manifest_sha256,
        pair_submission_receipt_path=pair_submission_receipt_path,
        pair_submission_receipt_sha256=pair_submission_receipt_sha256,
        trusted_off_exit_receipt_path=trusted_off_exit_receipt_path,
        trusted_off_exit_receipt_sha256=trusted_off_exit_receipt_sha256,
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    profile = _exact_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    manifest_path = _canonical_absolute(
        replay_manifest_path,
        name="replay manifest path",
    )
    manifest_sha = _digest(
        replay_manifest_sha256,
        name="replay manifest SHA-256",
    )
    submission_sha = _digest(
        submission_receipt_sha256,
        name="submission receipt SHA-256",
    )
    candidate = _exact_string(candidate_job_id, name="candidate job ID")
    if _JOB_ID.fullmatch(candidate) is None:
        _fail("candidate job ID must be one canonical positive decimal string")
    final_path = _canonical_absolute(
        result_final_receipt_path,
        name="result FINAL receipt path",
    )
    final_sha = _digest(
        result_final_receipt_sha256,
        name="result FINAL receipt SHA-256",
    )
    authenticated = load_authenticated_captured_replay_result_v2(
        authenticated_source=source,
        replay_execution_manifest_path=manifest_path,
        replay_execution_manifest_sha256=manifest_sha,
        submission_receipt_sha256=submission_sha,
        candidate_job_id=candidate,
        result_final_receipt_path=final_path,
        result_final_receipt_sha256=final_sha,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    snapshot = snapshot_authenticated_captured_replay_result_v2(authenticated)
    validated = _validate_authenticated_result_snapshot(
        snapshot,
        replay_manifest_path=manifest_path,
        replay_manifest_sha256=manifest_sha,
        submission_receipt_sha256=submission_sha,
        candidate_job_id=candidate,
        result_final_receipt_path=final_path,
        result_final_receipt_sha256=final_sha,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
        authenticated_source=source,
    )
    return ConsumedReplayResult(snapshot=copy.deepcopy(validated))


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--pair-manifest-sha256", required=True)
    parser.add_argument("--pair-submission-receipt", required=True)
    parser.add_argument("--pair-submission-receipt-sha256", required=True)
    parser.add_argument("--off-exit-receipt", required=True)
    parser.add_argument("--off-exit-receipt-sha256", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--profile-id", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="publish static inputs and release one replay job")
    _add_source_arguments(submit)
    submit.add_argument("--attempt-id", required=True)
    submit.add_argument("--submission-nonce", required=True)
    consume = commands.add_parser("consume", help="verify and consume one completed sealed result")
    _add_source_arguments(consume)
    consume.add_argument("--replay-manifest", required=True)
    consume.add_argument("--replay-manifest-sha256", required=True)
    consume.add_argument("--submission-receipt-sha256", required=True)
    consume.add_argument("--candidate-job-id", required=True)
    consume.add_argument("--result-final-receipt", required=True)
    consume.add_argument("--result-final-receipt-sha256", required=True)
    return parser


def _source_kwargs(arguments: argparse.Namespace) -> dict[str, str]:
    return {
        "pair_manifest_path": arguments.pair_manifest,
        "pair_manifest_sha256": arguments.pair_manifest_sha256,
        "pair_submission_receipt_path": arguments.pair_submission_receipt,
        "pair_submission_receipt_sha256": arguments.pair_submission_receipt_sha256,
        "trusted_off_exit_receipt_path": arguments.off_exit_receipt,
        "trusted_off_exit_receipt_sha256": arguments.off_exit_receipt_sha256,
        "expected_environment": arguments.environment,
        "expected_profile_id": arguments.profile_id,
    }


def _submit_json(result: SubmittedReplay) -> dict[str, Any]:
    prepared = result.prepared
    return {
        "schema": COORDINATOR_SUBMIT_RESULT_SCHEMA,
        "status": "released",
        "pair_id": prepared.pair_id,
        "attempt_id": prepared.attempt_id,
        "environment": prepared.environment,
        "profile_id": prepared.profile_id,
        "slurm_export": {
            "path": prepared.slurm_export_path,
            "sha256": prepared.slurm_export_sha256,
        },
        "submission_contract": {
            "path": prepared.submission_contract_path,
            "sha256": prepared.submission_contract_sha256,
        },
        "replay_manifest": {
            "path": prepared.replay_manifest_path,
            "sha256": prepared.replay_manifest_sha256,
        },
        "launcher_receipt": dict(result.launcher_receipt),
        "authenticated_submission": {
            "schema": result.submission_receipt.get("schema"),
            "status": result.submission_receipt.get("status"),
            "sha256": result.launcher_receipt["submission_receipt_sha256"],
        },
    }


def _consume_json(result: ConsumedReplayResult) -> dict[str, Any]:
    if type(result) is not ConsumedReplayResult:
        _fail("consumed replay result must have the exact coordinator type")
    snapshot = _exact_dict(
        result.snapshot,
        name="consumed authenticated result snapshot",
    )
    if (
        set(snapshot) != set(_AUTHENTICATED_RESULT_SNAPSHOT_KEYS)
        or type(snapshot.get("schema")) is not str
        or snapshot["schema"] != AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA
    ):
        _fail("consumed authenticated result snapshot differs")
    return {
        "schema": COORDINATOR_CONSUME_RESULT_SCHEMA,
        "status": "authenticated-and-snapshotted",
        "pair_id": snapshot["pair_id"],
        "attempt_id": snapshot["attempt_id"],
        "environment": snapshot["environment"],
        "profile_id": snapshot["profile_id"],
        "candidate_job_id": snapshot["candidate_job_id"],
        "authenticated_job_id": snapshot["authenticated_job_id"],
        "run_id": snapshot["run_id"],
        "driver_process": copy.deepcopy(snapshot["driver_process"]),
        "scorer_process_identity": copy.deepcopy(snapshot["scorer_process_identity"]),
        "replay_manifest": copy.deepcopy(snapshot["manifest"]),
        "submission_receipt": copy.deepcopy(snapshot["submission_receipt"]),
        "pre_receipt": copy.deepcopy(snapshot["pre_receipt"]),
        "exit_receipt": copy.deepcopy(snapshot["exit_receipt"]),
        "result_final_receipt": copy.deepcopy(snapshot["result_final_receipt"]),
        "result": {
            "root": snapshot["result_root"],
            "inventory": copy.deepcopy(snapshot["result_inventory"]),
            "evidence_index": copy.deepcopy(snapshot["evidence_index"]),
            "outputs": copy.deepcopy(snapshot["outputs"]),
            "samples": copy.deepcopy(snapshot["samples"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit submit or post-job consume phase."""
    arguments = _parser().parse_args(argv)
    os.umask(0o077)
    try:
        if arguments.command == "submit":
            result = submit_replay(
                **_source_kwargs(arguments),
                attempt_id=arguments.attempt_id,
                submission_nonce=arguments.submission_nonce,
            )
            output = _submit_json(result)
        elif arguments.command == "consume":
            result = consume_replay_result(
                **_source_kwargs(arguments),
                replay_manifest_path=arguments.replay_manifest,
                replay_manifest_sha256=arguments.replay_manifest_sha256,
                submission_receipt_sha256=arguments.submission_receipt_sha256,
                candidate_job_id=arguments.candidate_job_id,
                result_final_receipt_path=arguments.result_final_receipt,
                result_final_receipt_sha256=(arguments.result_final_receipt_sha256),
            )
            output = _consume_json(result)
        else:  # pragma: no cover - argparse closes the command set.
            raise AssertionError("unreachable coordinator command")
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"strict captured-replay coordinator failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "COORDINATOR_CONSUME_RESULT_SCHEMA",
    "COORDINATOR_SUBMIT_RESULT_SCHEMA",
    "ConsumedReplayResult",
    "PAIR79_RECORD_COUNT",
    "Pair79ReplayExport",
    "PreparedReplaySubmission",
    "StrictCapturedReplayCoordinatorError",
    "SubmittedReplay",
    "consume_replay_result",
    "invoke_replay_launcher",
    "load_authenticated_replay_source",
    "load_released_submission_receipt",
    "main",
    "prepare_replay_submission",
    "submit_replay",
]


if __name__ == "__main__":
    raise SystemExit(main())
