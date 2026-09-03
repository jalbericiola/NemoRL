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

"""Seal and independently verify a terminal strict captured-replay result.

The result directory is an evidence boundary, not a work directory.  Slurm
stdout/stderr, Ray logs, Hugging Face state, and runtime caches must live outside
this tree.  Publication is deliberately last: the job wrapper calls this module
only after the replay driver has returned (and therefore reaped its scorer), the
terminal output documents have been validated, and the evidence index has been
published.

``result-inventory-v1.json`` is self-excluded.  Its caller-carried SHA-256 is the
root of authority for offline verification; its entries are the exact static
allowlist of every other directory and file.  The publisher and verifier both
walk by directory descriptors with ``O_NOFOLLOW`` and reject extras, symlinks,
special files, writable evidence, wrong ownership, and multiply-linked files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import stat
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.utils.strict_captured_replay_profiles import (
        StrictCapturedReplayProfile,
    )

RESULT_INVENTORY_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v1"
RESULT_INVENTORY_FILENAME = "result-inventory-v1.json"
RESULT_INVENTORY_HASH_DOMAIN = "sha256-canonical-ascii-json-no-lf-v1"
RESULT_INVENTORY_V2_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
RESULT_INVENTORY_V2_FILENAME = "result-inventory-v2.json"
RESULT_INVENTORY_V2_HASH_DOMAIN = RESULT_INVENTORY_HASH_DOMAIN

# Only reasoning_gym currently has an admitted captured-replay execution path.
# Keep this closed inventory adjacent to the sealer: learning a filename from the
# directory being sealed would turn attacker-created files into admitted evidence.
RESULT_DIRECTORY_ALLOWLIST = (".", "strict_gym_child_runtime")
RESULT_FILE_ALLOWLIST = (
    "evidence-index.json",
    "model-transport-replay-consumption.json",
    "replay-ledger.json",
    "strict_gym_child_runtime/index.json",
    "strict_gym_child_runtime/reasoning-score-call-00000001.json",
    "strict_gym_child_runtime/reasoning-score-call-00000002.json",
    "strict_gym_child_runtime/reasoning-score-call-00000003.json",
    "strict_gym_child_runtime/reasoning-score-call-00000004.json",
    "strict_gym_child_runtime/reasoning-score-call-index.json",
    "strict_gym_child_runtime/reasoning-score-closed.json",
    "strict_gym_child_runtime/resource.json",
    "strict_gym_child_runtime/spec.json",
    "transcript-bundle.json",
)
RESULT_FILE_SCHEMA_ALLOWLIST = (
    "nemo-rl-strict-captured-replay-evidence-index-v3",
    "nemo-rl-strict-model-transport-replay-consumption-v2",
    "nemo-rl-strict-captured-replay-step1-ledger-v5",
    "nemo-rl-strict-gym-child-index-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-index-v1",
    "nemo-rl-strict-reasoning-score-closed-v1",
    "nemo-rl-strict-gym-child-receipt-v1",
    "nemo-rl-strict-gym-child-spec-v1",
    "nemo-rl-strict-step1-transcript-bundle-v4",
)
RESULT_ANCHOR_ALLOWLIST = frozenset(
    {
        "evidence-index.json",
        "model-transport-replay-consumption.json",
        "replay-ledger.json",
        "strict_gym_child_runtime/reasoning-score-call-index.json",
        "transcript-bundle.json",
    }
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_INVENTORY_BYTES = 128 * 1024
_MAX_RESULT_BYTES = 512 * 1024 * 1024
_VERIFIED_RESULT_MINT_TOKEN = object()
_VERIFIED_RESULT_V2_MINT_TOKEN = object()


class StrictCapturedReplaySealError(ValueError):
    """The terminal result does not satisfy the sealed-evidence contract."""


def _fail(message: str) -> None:
    raise StrictCapturedReplaySealError(message)


class VerifiedSealedResultV1:
    """Opaque, immutable authority minted only after nofollow verification.

    Consumers must pass this object back to :func:`consume_verified_sealed_result`.
    It deliberately exposes no paths, mappings, callbacks, or direct filesystem
    authority.  The consumer returns only exact built-in ``str``/``bytes`` tuples.
    """

    __slots__ = (
        "__files",
        "__inventory_raw",
        "__inventory_sha256",
        "__mint_token",
        "__result_root",
    )

    def __init__(
        self,
        *,
        _mint_token: object,
        result_root: str,
        inventory_sha256: str,
        inventory_raw: bytes,
        files: tuple[tuple[str, bytes], ...],
    ) -> None:
        if _mint_token is not _VERIFIED_RESULT_MINT_TOKEN:
            _fail("verified sealed-result authority may only be minted by verifier")
        object.__setattr__(self, "_VerifiedSealedResultV1__mint_token", _mint_token)
        object.__setattr__(self, "_VerifiedSealedResultV1__result_root", result_root)
        object.__setattr__(
            self,
            "_VerifiedSealedResultV1__inventory_sha256",
            inventory_sha256,
        )
        object.__setattr__(
            self, "_VerifiedSealedResultV1__inventory_raw", inventory_raw
        )
        object.__setattr__(self, "_VerifiedSealedResultV1__files", files)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedSealedResultV1 is immutable")

    def __reduce__(self) -> object:
        raise TypeError("VerifiedSealedResultV1 cannot cross a process boundary")


class VerifiedSealedResultV2:
    """Opaque authority for one explicitly selected profiled result."""

    __slots__ = (
        "__environment",
        "__files",
        "__inventory_raw",
        "__inventory_sha256",
        "__mint_token",
        "__profile_id",
        "__result_root",
    )

    def __init__(
        self,
        *,
        _mint_token: object,
        result_root: str,
        inventory_sha256: str,
        environment: str,
        profile_id: str,
        inventory_raw: bytes,
        files: tuple[tuple[str, bytes], ...],
    ) -> None:
        if _mint_token is not _VERIFIED_RESULT_V2_MINT_TOKEN:
            _fail(
                "verified profiled sealed-result authority may only be minted by verifier"
            )
        object.__setattr__(self, "_VerifiedSealedResultV2__mint_token", _mint_token)
        object.__setattr__(self, "_VerifiedSealedResultV2__result_root", result_root)
        object.__setattr__(
            self,
            "_VerifiedSealedResultV2__inventory_sha256",
            inventory_sha256,
        )
        object.__setattr__(self, "_VerifiedSealedResultV2__environment", environment)
        object.__setattr__(self, "_VerifiedSealedResultV2__profile_id", profile_id)
        object.__setattr__(
            self, "_VerifiedSealedResultV2__inventory_raw", inventory_raw
        )
        object.__setattr__(self, "_VerifiedSealedResultV2__files", files)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedSealedResultV2 is immutable")

    def __reduce__(self) -> object:
        raise TypeError("VerifiedSealedResultV2 cannot cross a process boundary")


def _canonical_absolute(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or posixpath.normpath(value) != value
        or any(not 0x20 <= ord(character) <= 0x7E for character in value)
    ):
        _fail(f"{name} must be one canonical absolute printable-ASCII path")
    return value


def _digest(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or _DIGEST_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        _fail(f"{name} must be one nonzero lowercase SHA-256")
    return value


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
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


def _open_absolute_directory(path: str) -> int:
    canonical = _canonical_absolute(path, name="result root")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in canonical.split("/")[1:]:
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_directory(
    descriptor: int, *, name: str, exact_mode: int
) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != exact_mode
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink < 2
    ):
        _fail(f"{name} must be an EUID-owned mode-{exact_mode:04o} real directory")
    return metadata


def _entry_names(descriptor: int) -> frozenset[str]:
    try:
        with os.scandir(descriptor) as entries:
            names = [entry.name for entry in entries]
    except OSError as error:
        _fail(f"cannot enumerate result directory: {error}")
    if any(
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        for name in names
    ):
        _fail("result directory contains an unsafe entry name")
    if len(names) != len(set(names)):
        _fail("result directory enumeration contains duplicate names")
    return frozenset(names)


def _open_child_directory(parent_fd: int, name: str, *, mode: int) -> int:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
    except OSError as error:
        _fail(f"cannot open result directory {name}: {error}")
    if _fingerprint(before) != _fingerprint(opened):
        os.close(descriptor)
        _fail(f"result directory {name} changed while opening")
    _require_directory(descriptor, name=name, exact_mode=mode)
    return descriptor


def _read_regular_at(
    parent_fd: int,
    name: str,
    *,
    relative: str,
    exact_mode: int,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[bytes, os.stat_result]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot stat result file {relative}: {error}")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != exact_mode
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or not (0 if allow_empty else 1) <= before.st_size <= maximum
    ):
        _fail(
            f"result file {relative} differs from the owned single-link "
            f"mode-{exact_mode:04o} regular-file contract"
        )
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot nofollow-open result file {relative}: {error}")
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(before):
            _fail(f"result file {relative} changed while opening")
        remaining = opened.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"result file {relative} truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"result file {relative} grew while reading")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot restat result file {relative}: {error}")
    if not (
        _fingerprint(before)
        == _fingerprint(opened)
        == _fingerprint(after)
        == _fingerprint(named)
    ):
        _fail(f"result file {relative} changed during stable read")
    return b"".join(chunks), named


def _read_relative_file(
    root_fd: int,
    relative: str,
    *,
    exact_mode: int = 0o400,
    maximum: int = _MAX_FILE_BYTES,
    allow_empty: bool = False,
) -> tuple[bytes, os.stat_result]:
    components = relative.split("/")
    if (
        relative.startswith("/")
        or posixpath.normpath(relative) != relative
        or any(component in {"", ".", ".."} for component in components)
    ):
        _fail(f"unsafe result-relative path: {relative!r}")
    if len(components) == 1:
        return _read_regular_at(
            root_fd,
            components[0],
            relative=relative,
            exact_mode=exact_mode,
            maximum=maximum,
            allow_empty=allow_empty,
        )
    if len(components) != 2 or components[0] != "strict_gym_child_runtime":
        _fail(f"result path is outside the closed directory allowlist: {relative}")
    child_fd = _open_child_directory(root_fd, components[0], mode=0o700)
    try:
        return _read_regular_at(
            child_fd,
            components[1],
            relative=relative,
            exact_mode=exact_mode,
            maximum=maximum,
            allow_empty=allow_empty,
        )
    finally:
        os.close(child_fd)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise StrictCapturedReplaySealError(
            "result inventory is not finite canonical ASCII JSON"
        ) from error


def _reject_constant(value: str) -> Any:
    _fail(f"result inventory contains non-finite JSON constant: {value}")


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"result inventory contains duplicate member {key!r}")
        result[key] = value
    return result


def _reject_negative_zero(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value) or (
            value == 0.0 and math.copysign(1.0, value) < 0.0
        ):
            _fail("result inventory contains a non-finite or negative-zero number")
    elif type(value) is dict:
        for member in value.values():
            _reject_negative_zero(member)
    elif type(value) is list:
        for member in value:
            _reject_negative_zero(member)


def _load_inventory(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StrictCapturedReplaySealError(
            "result inventory is not strict ASCII JSON"
        ) from error
    if type(document) is not dict:
        _fail("result inventory root must be an exact object")
    _reject_negative_zero(document)
    if _canonical_json(document) != raw:
        _fail("result inventory bytes are not canonical ASCII JSON without LF")
    return document


def _validate_result_document(
    raw: bytes, *, relative: str, expected_schema: str
) -> None:
    document = _load_inventory(raw)
    if document.get("schema") != expected_schema:
        _fail(f"sealed result schema differs for {relative}")


def _validate_reaped_scorer_bytes(
    raw: bytes,
    *,
    profile: StrictCapturedReplayProfile | None = None,
) -> None:
    """Validate quiescence from the same owned bytes admitted to inventory."""
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StrictCapturedReplaySealError(
            "reasoning score-call index is not strict ASCII JSON"
        ) from error
    if type(document) is not dict or _canonical_json(document) != raw:
        _fail("reasoning score-call index is not canonical")
    if profile is not None and (
        document.get("environment") != profile.environment
        or document.get("profile_id") != profile.profile_id
    ):
        _fail("scorer call-index environment/profile identity differs")
    quiescence = document.get("quiescence")
    if (
        type(quiescence) is not dict
        or quiescence.get("original_process_reaped") is not True
        or type(quiescence.get("wrapper_returncode")) is not int
    ):
        _fail("scorer quiescence does not prove the original process was reaped")


def _validate_live_inventory(root_fd: int, *, directory_mode: int) -> None:
    expected_root = {
        RESULT_INVENTORY_FILENAME,
        "evidence-index.json",
        "model-transport-replay-consumption.json",
        "replay-ledger.json",
        "strict_gym_child_runtime",
        "transcript-bundle.json",
    }
    if _entry_names(root_fd) != frozenset(expected_root):
        _fail("result root has an extra, missing, or renamed entry")
    child_fd = _open_child_directory(
        root_fd, "strict_gym_child_runtime", mode=directory_mode
    )
    try:
        expected_child = {
            relative.split("/", 1)[1]
            for relative in RESULT_FILE_ALLOWLIST
            if relative.startswith("strict_gym_child_runtime/")
        }
        if _entry_names(child_fd) != frozenset(expected_child):
            _fail("strict Gym result directory has an extra, missing, or renamed entry")
    finally:
        os.close(child_fd)


def _validate_inventory_shape(
    document: Mapping[str, Any], *, result_root: str
) -> dict[str, dict[str, Any]]:
    if type(document) is not dict or set(document) != {
        "schema",
        "hash_domain",
        "root",
        "self_excluded",
        "directories",
        "files",
        "totals",
        "publication",
    }:
        _fail("result inventory root keyset differs")
    if (
        document["schema"] != RESULT_INVENTORY_SCHEMA
        or document["hash_domain"] != RESULT_INVENTORY_HASH_DOMAIN
        or document["root"] != result_root
        or document["self_excluded"]
        != {
            "path": RESULT_INVENTORY_FILENAME,
            "policy": "excluded-from-files-and-totals",
        }
    ):
        _fail("result inventory identity/self-exclusion differs")
    directories = document["directories"]
    expected_directories = [
        {"mode": "0555", "path": relative} for relative in RESULT_DIRECTORY_ALLOWLIST
    ]
    if directories != expected_directories:
        _fail("result inventory directory allowlist differs")
    files = document["files"]
    if type(files) is not list or len(files) != len(RESULT_FILE_ALLOWLIST):
        _fail("result inventory file count differs")
    indexed: dict[str, dict[str, Any]] = {}
    for record in files:
        if type(record) is not dict or set(record) != {
            "path",
            "mode",
            "size",
            "sha256",
        }:
            _fail("result inventory file record shape differs")
        relative = record["path"]
        if (
            type(relative) is not str
            or relative not in RESULT_FILE_ALLOWLIST
            or relative == RESULT_INVENTORY_FILENAME
            or relative in indexed
            or record["mode"] != "0400"
            or type(record["size"]) is not int
            or not 1 <= record["size"] <= _MAX_FILE_BYTES
        ):
            _fail("result inventory file record differs from exact policy")
        _digest(record["sha256"], name=f"inventory SHA-256 for {relative}")
        indexed[relative] = dict(record)
    if list(indexed) != list(RESULT_FILE_ALLOWLIST):
        _fail("result inventory files are not in exact canonical allowlist order")
    total_bytes = sum(record["size"] for record in indexed.values())
    totals = document["totals"]
    if (
        type(totals) is not dict
        or set(totals) != {"directory_count", "file_count", "file_bytes"}
        or any(type(totals[name]) is not int for name in totals)
        or not 1 <= totals["file_bytes"] <= _MAX_RESULT_BYTES
    ):
        _fail("result inventory aggregate totals differ")
    if totals != {
        "directory_count": len(RESULT_DIRECTORY_ALLOWLIST),
        "file_count": len(RESULT_FILE_ALLOWLIST),
        "file_bytes": total_bytes,
    }:
        _fail("result inventory totals differ")
    publication = document["publication"]
    if (
        type(publication) is not dict
        or set(publication)
        != {"directories_mode", "files_mode", "nofollow_reverified", "order"}
        or publication.get("directories_mode") != "0555"
        or publication.get("files_mode") != "0400"
        or publication.get("nofollow_reverified") is not True
        or publication.get("order")
        != "inventory-then-bottom-up-directory-seal-then-rehash"
    ):
        _fail("result inventory publication contract differs")
    return indexed


def _write_inventory(root_fd: int, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(RESULT_INVENTORY_FILENAME, flags, 0o400, dir_fd=root_fd)
    except OSError as error:
        _fail(f"cannot exclusively publish result inventory: {error}")
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("short write while publishing result inventory")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_fd)


def publish_sealed_result(
    *, result_root: str, anchored_sha256: Mapping[str, str]
) -> tuple[str, str]:
    """Publish the last artifact, seal bottom-up, and nofollow-reverify it all.

    ``anchored_sha256`` must carry the five terminal references already validated
    by the wrapper.  It prevents a merely well-shaped but substituted terminal
    document from becoming admitted during inventory publication.
    """
    canonical_root = _canonical_absolute(result_root, name="result root")
    if type(anchored_sha256) is not dict or set(anchored_sha256) != set(
        RESULT_ANCHOR_ALLOWLIST
    ):
        _fail("terminal result anchor keyset differs")
    anchors = {
        name: _digest(value, name=f"terminal anchor {name}")
        for name, value in anchored_sha256.items()
    }
    root_fd = _open_absolute_directory(canonical_root)
    try:
        _require_directory(root_fd, name="result root", exact_mode=0o700)
        if RESULT_INVENTORY_FILENAME in _entry_names(root_fd):
            _fail("result inventory must be absent before terminal publication")

        # Validate the exact pre-inventory tree without learning its allowlist from
        # disk.  Add the self-excluded filename only for the post-publication walk.
        expected_pre_root = {
            name
            for name in (
                "evidence-index.json",
                "model-transport-replay-consumption.json",
                "replay-ledger.json",
                "strict_gym_child_runtime",
                "transcript-bundle.json",
            )
        }
        if _entry_names(root_fd) != frozenset(expected_pre_root):
            _fail("result root has an extra, missing, or renamed pre-seal entry")
        child_fd = _open_child_directory(
            root_fd, "strict_gym_child_runtime", mode=0o700
        )
        try:
            expected_child = {
                relative.split("/", 1)[1]
                for relative in RESULT_FILE_ALLOWLIST
                if relative.startswith("strict_gym_child_runtime/")
            }
            if _entry_names(child_fd) != frozenset(expected_child):
                _fail("strict Gym pre-seal inventory differs from exact allowlist")
        finally:
            os.close(child_fd)

        file_records: list[dict[str, Any]] = []
        total_file_bytes = 0
        for relative, expected_schema in zip(
            RESULT_FILE_ALLOWLIST, RESULT_FILE_SCHEMA_ALLOWLIST, strict=True
        ):
            raw, metadata = _read_relative_file(root_fd, relative)
            _validate_result_document(
                raw, relative=relative, expected_schema=expected_schema
            )
            if relative == "strict_gym_child_runtime/reasoning-score-call-index.json":
                _validate_reaped_scorer_bytes(raw)
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            total_file_bytes += metadata.st_size
            if total_file_bytes > _MAX_RESULT_BYTES:
                _fail("result files exceed aggregate byte limit")
            if relative in anchors and anchors[relative] != actual_sha256:
                _fail(f"terminal result anchor differs for {relative}")
            file_records.append(
                {
                    "path": relative,
                    "mode": "0400",
                    "size": metadata.st_size,
                    "sha256": actual_sha256,
                }
            )
        inventory = {
            "schema": RESULT_INVENTORY_SCHEMA,
            "hash_domain": RESULT_INVENTORY_HASH_DOMAIN,
            "root": canonical_root,
            "self_excluded": {
                "path": RESULT_INVENTORY_FILENAME,
                "policy": "excluded-from-files-and-totals",
            },
            "directories": [
                {"mode": "0555", "path": relative}
                for relative in RESULT_DIRECTORY_ALLOWLIST
            ],
            "files": file_records,
            "totals": {
                "directory_count": len(RESULT_DIRECTORY_ALLOWLIST),
                "file_count": len(RESULT_FILE_ALLOWLIST),
                "file_bytes": total_file_bytes,
            },
            "publication": {
                "directories_mode": "0555",
                "files_mode": "0400",
                "nofollow_reverified": True,
                "order": "inventory-then-bottom-up-directory-seal-then-rehash",
            },
        }
        payload = _canonical_json(inventory)
        _write_inventory(root_fd, payload)
        inventory_sha256 = hashlib.sha256(payload).hexdigest()

        child_fd = _open_child_directory(
            root_fd, "strict_gym_child_runtime", mode=0o700
        )
        try:
            os.fchmod(child_fd, 0o555)
            os.fsync(child_fd)
        finally:
            os.close(child_fd)
        os.fchmod(root_fd, 0o555)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)

    verified = verify_sealed_result(
        result_root=canonical_root,
        expected_inventory_sha256=inventory_sha256,
    )
    consume_verified_sealed_result(
        verified,
        expected_result_root=canonical_root,
        expected_inventory_sha256=inventory_sha256,
    )
    return f"{canonical_root}/{RESULT_INVENTORY_FILENAME}", inventory_sha256


def verify_sealed_result(
    *, result_root: str, expected_inventory_sha256: str
) -> VerifiedSealedResultV1:
    """Verify a sealed result and mint an in-process bytes-only authority."""
    canonical_root = _canonical_absolute(result_root, name="result root")
    expected_sha256 = _digest(
        expected_inventory_sha256, name="expected result inventory SHA-256"
    )
    root_fd = _open_absolute_directory(canonical_root)
    try:
        _require_directory(root_fd, name="sealed result root", exact_mode=0o555)
        _validate_live_inventory(root_fd, directory_mode=0o555)
        inventory_raw, _ = _read_regular_at(
            root_fd,
            RESULT_INVENTORY_FILENAME,
            relative=RESULT_INVENTORY_FILENAME,
            exact_mode=0o400,
            maximum=_MAX_INVENTORY_BYTES,
        )
        actual_inventory_sha256 = hashlib.sha256(inventory_raw).hexdigest()
        if actual_inventory_sha256 != expected_sha256:
            _fail("result inventory differs from caller-carried SHA-256")
        inventory = _load_inventory(inventory_raw)
        indexed = _validate_inventory_shape(inventory, result_root=canonical_root)
        verified_files: list[tuple[str, bytes]] = []
        verified_file_bytes = 0
        for relative, expected_schema in zip(
            RESULT_FILE_ALLOWLIST, RESULT_FILE_SCHEMA_ALLOWLIST, strict=True
        ):
            raw, metadata = (
                _read_relative_file(
                    root_fd,
                    relative,
                    # The directory reader normally expects in-progress 0700.  Open
                    # nested files directly below using a sealed child descriptor.
                )
                if "/" not in relative
                else _read_nested_sealed(root_fd, relative)
            )
            record = indexed[relative]
            if (
                metadata.st_size != record["size"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]
            ):
                _fail(f"sealed result file differs from inventory: {relative}")
            _validate_result_document(
                raw, relative=relative, expected_schema=expected_schema
            )
            if relative == "strict_gym_child_runtime/reasoning-score-call-index.json":
                _validate_reaped_scorer_bytes(raw)
            verified_file_bytes += len(raw)
            if verified_file_bytes > _MAX_RESULT_BYTES:
                _fail("sealed result exceeds aggregate byte limit")
            verified_files.append((relative, raw))
        if verified_file_bytes != inventory["totals"]["file_bytes"]:
            _fail("sealed result aggregate byte count differs")
        _validate_live_inventory(root_fd, directory_mode=0o555)
        _require_directory(root_fd, name="sealed result root", exact_mode=0o555)
    finally:
        os.close(root_fd)
    return VerifiedSealedResultV1(
        _mint_token=_VERIFIED_RESULT_MINT_TOKEN,
        result_root=canonical_root,
        inventory_sha256=actual_inventory_sha256,
        inventory_raw=inventory_raw,
        files=tuple(verified_files),
    )


def consume_verified_sealed_result(
    value: Any,
    *,
    expected_result_root: str,
    expected_inventory_sha256: str,
) -> tuple[tuple[str, bytes], ...]:
    """Return inert verified bytes from an exact verifier-minted authority.

    This is the only supported semantic-validator handoff.  In particular, a
    caller-supplied mapping or tuple that merely resembles the result is rejected.
    The capability is intentionally process-local and cannot be pickled.
    """
    canonical_root = _canonical_absolute(
        expected_result_root, name="expected result root"
    )
    expected_sha256 = _digest(
        expected_inventory_sha256, name="expected result inventory SHA-256"
    )
    if type(value) is not VerifiedSealedResultV1:
        _fail("sealed-result consumer requires exact verifier-minted authority")
    token = object.__getattribute__(value, "_VerifiedSealedResultV1__mint_token")
    result_root = object.__getattribute__(value, "_VerifiedSealedResultV1__result_root")
    inventory_sha256 = object.__getattribute__(
        value, "_VerifiedSealedResultV1__inventory_sha256"
    )
    inventory_raw = object.__getattribute__(
        value, "_VerifiedSealedResultV1__inventory_raw"
    )
    files = object.__getattribute__(value, "_VerifiedSealedResultV1__files")
    if (
        token is not _VERIFIED_RESULT_MINT_TOKEN
        or type(result_root) is not str
        or result_root != canonical_root
        or type(inventory_sha256) is not str
        or inventory_sha256 != expected_sha256
        or type(inventory_raw) is not bytes
        or hashlib.sha256(inventory_raw).hexdigest() != expected_sha256
        or type(files) is not tuple
        or len(files) != len(RESULT_FILE_ALLOWLIST)
    ):
        _fail("verified sealed-result authority identity differs")
    inventory = _load_inventory(inventory_raw)
    indexed = _validate_inventory_shape(inventory, result_root=canonical_root)
    consumed: list[tuple[str, bytes]] = []
    consumed_bytes = 0
    for item, relative, expected_schema in zip(
        files,
        RESULT_FILE_ALLOWLIST,
        RESULT_FILE_SCHEMA_ALLOWLIST,
        strict=True,
    ):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != relative
            or type(item[1]) is not bytes
            or not 1 <= len(item[1]) <= _MAX_FILE_BYTES
            or len(item[1]) != indexed[relative]["size"]
            or hashlib.sha256(item[1]).hexdigest() != indexed[relative]["sha256"]
        ):
            _fail(f"verified sealed-result payload differs for {relative}")
        _validate_result_document(
            item[1], relative=relative, expected_schema=expected_schema
        )
        consumed_bytes += len(item[1])
        if consumed_bytes > _MAX_RESULT_BYTES:
            _fail("verified sealed-result payload exceeds aggregate byte limit")
        consumed.append((relative, item[1]))
    if consumed_bytes != inventory["totals"]["file_bytes"]:
        _fail("verified sealed-result aggregate byte count differs")
    return tuple(consumed)


def _read_nested_sealed(root_fd: int, relative: str) -> tuple[bytes, os.stat_result]:
    directory, filename = relative.split("/", 1)
    if directory != "strict_gym_child_runtime" or "/" in filename:
        _fail(f"sealed result path is outside the exact allowlist: {relative}")
    child_fd = _open_child_directory(root_fd, directory, mode=0o555)
    try:
        return _read_regular_at(
            child_fd,
            filename,
            relative=relative,
            exact_mode=0o400,
            maximum=_MAX_FILE_BYTES,
        )
    finally:
        os.close(child_fd)


def _strict_captured_replay_profile(
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> StrictCapturedReplayProfile:
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )

    try:
        return get_strict_captured_replay_profile(
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
    except ValueError as error:
        raise StrictCapturedReplaySealError(str(error)) from error


def _profile_root_entries(
    profile: StrictCapturedReplayProfile,
    *,
    include_inventory: bool,
) -> frozenset[str]:
    entries = {relative.split("/", 1)[0] for relative in profile.result_files}
    if include_inventory:
        entries.add(RESULT_INVENTORY_V2_FILENAME)
    return frozenset(entries)


def _profile_child_entries(profile: StrictCapturedReplayProfile) -> frozenset[str]:
    return frozenset(
        relative.split("/", 1)[1]
        for relative in profile.result_files
        if relative.startswith("strict_gym_child_runtime/")
    )


def _validate_live_inventory_v2(
    root_fd: int,
    *,
    profile: StrictCapturedReplayProfile,
    directory_mode: int,
) -> None:
    if _entry_names(root_fd) != _profile_root_entries(profile, include_inventory=True):
        _fail("profiled result root has an extra, missing, or renamed entry")
    child_fd = _open_child_directory(
        root_fd,
        "strict_gym_child_runtime",
        mode=directory_mode,
    )
    try:
        if _entry_names(child_fd) != _profile_child_entries(profile):
            _fail(
                "profiled strict Gym result directory has an extra, missing, or renamed entry"
            )
    finally:
        os.close(child_fd)


def _validate_inventory_shape_v2(
    document: Mapping[str, Any],
    *,
    result_root: str,
    profile: StrictCapturedReplayProfile,
) -> dict[str, dict[str, Any]]:
    if type(document) is not dict or set(document) != {
        "schema",
        "hash_domain",
        "root",
        "environment",
        "profile_id",
        "self_excluded",
        "directories",
        "files",
        "totals",
        "publication",
    }:
        _fail("profiled result inventory root keyset differs")
    if (
        document["schema"] != RESULT_INVENTORY_V2_SCHEMA
        or document["hash_domain"] != RESULT_INVENTORY_V2_HASH_DOMAIN
        or document["root"] != result_root
        or document["environment"] != profile.environment
        or document["profile_id"] != profile.profile_id
        or document["self_excluded"]
        != {
            "path": RESULT_INVENTORY_V2_FILENAME,
            "policy": "excluded-from-files-and-totals",
        }
    ):
        _fail("profiled result inventory identity/self-exclusion differs")
    directories = document["directories"]
    expected_directories = [
        {"mode": "0555", "path": relative} for relative in profile.result_directories
    ]
    if directories != expected_directories:
        _fail("profiled result inventory directory allowlist differs")
    files = document["files"]
    if type(files) is not list or len(files) != len(profile.result_files):
        _fail("profiled result inventory file count differs")
    indexed: dict[str, dict[str, Any]] = {}
    for record in files:
        if type(record) is not dict or set(record) != {
            "path",
            "mode",
            "size",
            "sha256",
        }:
            _fail("profiled result inventory file record shape differs")
        relative = record["path"]
        if (
            type(relative) is not str
            or relative not in profile.result_files
            or relative == RESULT_INVENTORY_V2_FILENAME
            or relative in indexed
            or record["mode"] != "0400"
            or type(record["size"]) is not int
            or not 1 <= record["size"] <= _MAX_FILE_BYTES
        ):
            _fail("profiled result inventory file record differs from exact policy")
        _digest(
            record["sha256"],
            name=f"profiled inventory SHA-256 for {relative}",
        )
        indexed[relative] = dict(record)
    if list(indexed) != list(profile.result_files):
        _fail("profiled result inventory files are not in exact profile order")
    total_bytes = sum(record["size"] for record in indexed.values())
    totals = document["totals"]
    if (
        type(totals) is not dict
        or set(totals) != {"directory_count", "file_count", "file_bytes"}
        or any(type(totals[name]) is not int for name in totals)
        or not 1 <= totals["file_bytes"] <= _MAX_RESULT_BYTES
    ):
        _fail("profiled result inventory aggregate totals differ")
    if totals != {
        "directory_count": len(profile.result_directories),
        "file_count": len(profile.result_files),
        "file_bytes": total_bytes,
    }:
        _fail("profiled result inventory totals differ")
    publication = document["publication"]
    if (
        type(publication) is not dict
        or set(publication)
        != {"directories_mode", "files_mode", "nofollow_reverified", "order"}
        or publication.get("directories_mode") != "0555"
        or publication.get("files_mode") != "0400"
        or publication.get("nofollow_reverified") is not True
        or publication.get("order")
        != "inventory-then-bottom-up-directory-seal-then-rehash"
    ):
        _fail("profiled result inventory publication contract differs")
    return indexed


def _write_inventory_v2(root_fd: int, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            RESULT_INVENTORY_V2_FILENAME,
            flags,
            0o400,
            dir_fd=root_fd,
        )
    except OSError as error:
        _fail(f"cannot exclusively publish profiled result inventory: {error}")
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("short write while publishing profiled result inventory")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_fd)


def publish_sealed_result_v2(
    *,
    result_root: str,
    anchored_sha256: Mapping[str, str],
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[str, str]:
    """Publish and verify one caller-selected result-inventory-v2 profile."""
    profile = _strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    canonical_root = _canonical_absolute(result_root, name="result root")
    if type(anchored_sha256) is not dict or set(anchored_sha256) != set(
        profile.result_anchor_paths
    ):
        _fail("profiled terminal result anchor keyset differs")
    anchors = {
        name: _digest(value, name=f"profiled terminal anchor {name}")
        for name, value in anchored_sha256.items()
    }
    root_fd = _open_absolute_directory(canonical_root)
    try:
        _require_directory(root_fd, name="profiled result root", exact_mode=0o700)
        if RESULT_INVENTORY_V2_FILENAME in _entry_names(root_fd):
            _fail(
                "profiled result inventory must be absent before terminal publication"
            )
        if _entry_names(root_fd) != _profile_root_entries(
            profile,
            include_inventory=False,
        ):
            _fail(
                "profiled result root has an extra, missing, or renamed pre-seal entry"
            )
        child_fd = _open_child_directory(
            root_fd,
            "strict_gym_child_runtime",
            mode=0o700,
        )
        try:
            if _entry_names(child_fd) != _profile_child_entries(profile):
                _fail(
                    "profiled strict Gym pre-seal inventory differs from exact allowlist"
                )
        finally:
            os.close(child_fd)

        file_records: list[dict[str, Any]] = []
        total_file_bytes = 0
        for relative, expected_schema in zip(
            profile.result_files,
            profile.result_file_schemas,
            strict=True,
        ):
            raw, metadata = _read_relative_file(root_fd, relative)
            _validate_result_document(
                raw,
                relative=relative,
                expected_schema=expected_schema,
            )
            if relative == profile.scorer_terminal_index_path:
                _validate_reaped_scorer_bytes(raw, profile=profile)
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            total_file_bytes += metadata.st_size
            if total_file_bytes > _MAX_RESULT_BYTES:
                _fail("profiled result files exceed aggregate byte limit")
            if relative in anchors and anchors[relative] != actual_sha256:
                _fail(f"profiled terminal result anchor differs for {relative}")
            file_records.append(
                {
                    "path": relative,
                    "mode": "0400",
                    "size": metadata.st_size,
                    "sha256": actual_sha256,
                }
            )
        inventory = {
            "schema": RESULT_INVENTORY_V2_SCHEMA,
            "hash_domain": RESULT_INVENTORY_V2_HASH_DOMAIN,
            "root": canonical_root,
            "environment": profile.environment,
            "profile_id": profile.profile_id,
            "self_excluded": {
                "path": RESULT_INVENTORY_V2_FILENAME,
                "policy": "excluded-from-files-and-totals",
            },
            "directories": [
                {"mode": "0555", "path": relative}
                for relative in profile.result_directories
            ],
            "files": file_records,
            "totals": {
                "directory_count": len(profile.result_directories),
                "file_count": len(profile.result_files),
                "file_bytes": total_file_bytes,
            },
            "publication": {
                "directories_mode": "0555",
                "files_mode": "0400",
                "nofollow_reverified": True,
                "order": "inventory-then-bottom-up-directory-seal-then-rehash",
            },
        }
        payload = _canonical_json(inventory)
        _write_inventory_v2(root_fd, payload)
        inventory_sha256 = hashlib.sha256(payload).hexdigest()

        child_fd = _open_child_directory(
            root_fd,
            "strict_gym_child_runtime",
            mode=0o700,
        )
        try:
            os.fchmod(child_fd, 0o555)
            os.fsync(child_fd)
        finally:
            os.close(child_fd)
        os.fchmod(root_fd, 0o555)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)

    verified = verify_sealed_result_v2(
        result_root=canonical_root,
        expected_inventory_sha256=inventory_sha256,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    consume_verified_sealed_result_v2(
        verified,
        expected_result_root=canonical_root,
        expected_inventory_sha256=inventory_sha256,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    return f"{canonical_root}/{RESULT_INVENTORY_V2_FILENAME}", inventory_sha256


def verify_sealed_result_v2(
    *,
    result_root: str,
    expected_inventory_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> VerifiedSealedResultV2:
    """Verify one explicit profile and mint its distinct V2 authority."""
    profile = _strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    canonical_root = _canonical_absolute(result_root, name="result root")
    expected_sha256 = _digest(
        expected_inventory_sha256,
        name="expected profiled result inventory SHA-256",
    )
    root_fd = _open_absolute_directory(canonical_root)
    try:
        _require_directory(
            root_fd, name="sealed profiled result root", exact_mode=0o555
        )
        _validate_live_inventory_v2(
            root_fd,
            profile=profile,
            directory_mode=0o555,
        )
        inventory_raw, _ = _read_regular_at(
            root_fd,
            RESULT_INVENTORY_V2_FILENAME,
            relative=RESULT_INVENTORY_V2_FILENAME,
            exact_mode=0o400,
            maximum=_MAX_INVENTORY_BYTES,
        )
        actual_inventory_sha256 = hashlib.sha256(inventory_raw).hexdigest()
        if actual_inventory_sha256 != expected_sha256:
            _fail("profiled result inventory differs from caller-carried SHA-256")
        inventory = _load_inventory(inventory_raw)
        indexed = _validate_inventory_shape_v2(
            inventory,
            result_root=canonical_root,
            profile=profile,
        )
        verified_files: list[tuple[str, bytes]] = []
        verified_file_bytes = 0
        for relative, expected_schema in zip(
            profile.result_files,
            profile.result_file_schemas,
            strict=True,
        ):
            raw, metadata = (
                _read_relative_file(root_fd, relative)
                if "/" not in relative
                else _read_nested_sealed(root_fd, relative)
            )
            record = indexed[relative]
            if (
                metadata.st_size != record["size"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]
            ):
                _fail(f"sealed profiled result file differs from inventory: {relative}")
            _validate_result_document(
                raw,
                relative=relative,
                expected_schema=expected_schema,
            )
            if relative == profile.scorer_terminal_index_path:
                _validate_reaped_scorer_bytes(raw, profile=profile)
            verified_file_bytes += len(raw)
            if verified_file_bytes > _MAX_RESULT_BYTES:
                _fail("sealed profiled result exceeds aggregate byte limit")
            verified_files.append((relative, raw))
        if verified_file_bytes != inventory["totals"]["file_bytes"]:
            _fail("sealed profiled result aggregate byte count differs")
        _validate_live_inventory_v2(
            root_fd,
            profile=profile,
            directory_mode=0o555,
        )
        _require_directory(
            root_fd,
            name="sealed profiled result root",
            exact_mode=0o555,
        )
    finally:
        os.close(root_fd)
    return VerifiedSealedResultV2(
        _mint_token=_VERIFIED_RESULT_V2_MINT_TOKEN,
        result_root=canonical_root,
        inventory_sha256=actual_inventory_sha256,
        environment=profile.environment,
        profile_id=profile.profile_id,
        inventory_raw=inventory_raw,
        files=tuple(verified_files),
    )


def consume_verified_sealed_result_v2(
    value: Any,
    *,
    expected_result_root: str,
    expected_inventory_sha256: str,
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[tuple[str, bytes], ...]:
    """Return inert bytes only for a matching verifier-minted V2 authority."""
    profile = _strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    canonical_root = _canonical_absolute(
        expected_result_root,
        name="expected result root",
    )
    expected_sha256 = _digest(
        expected_inventory_sha256,
        name="expected profiled result inventory SHA-256",
    )
    if type(value) is not VerifiedSealedResultV2:
        _fail(
            "profiled sealed-result consumer requires exact V2 verifier-minted authority"
        )
    token = object.__getattribute__(value, "_VerifiedSealedResultV2__mint_token")
    result_root = object.__getattribute__(value, "_VerifiedSealedResultV2__result_root")
    inventory_sha256 = object.__getattribute__(
        value,
        "_VerifiedSealedResultV2__inventory_sha256",
    )
    environment = object.__getattribute__(
        value,
        "_VerifiedSealedResultV2__environment",
    )
    profile_id = object.__getattribute__(
        value,
        "_VerifiedSealedResultV2__profile_id",
    )
    inventory_raw = object.__getattribute__(
        value,
        "_VerifiedSealedResultV2__inventory_raw",
    )
    files = object.__getattribute__(value, "_VerifiedSealedResultV2__files")
    if (
        token is not _VERIFIED_RESULT_V2_MINT_TOKEN
        or type(result_root) is not str
        or result_root != canonical_root
        or type(inventory_sha256) is not str
        or inventory_sha256 != expected_sha256
        or type(environment) is not str
        or environment != profile.environment
        or type(profile_id) is not str
        or profile_id != profile.profile_id
        or type(inventory_raw) is not bytes
        or hashlib.sha256(inventory_raw).hexdigest() != expected_sha256
        or type(files) is not tuple
        or len(files) != len(profile.result_files)
    ):
        _fail("verified profiled sealed-result authority identity differs")
    inventory = _load_inventory(inventory_raw)
    indexed = _validate_inventory_shape_v2(
        inventory,
        result_root=canonical_root,
        profile=profile,
    )
    consumed: list[tuple[str, bytes]] = []
    consumed_bytes = 0
    for item, relative, expected_schema in zip(
        files,
        profile.result_files,
        profile.result_file_schemas,
        strict=True,
    ):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != relative
            or type(item[1]) is not bytes
            or not 1 <= len(item[1]) <= _MAX_FILE_BYTES
            or len(item[1]) != indexed[relative]["size"]
            or hashlib.sha256(item[1]).hexdigest() != indexed[relative]["sha256"]
        ):
            _fail(f"verified profiled sealed-result payload differs for {relative}")
        _validate_result_document(
            item[1],
            relative=relative,
            expected_schema=expected_schema,
        )
        consumed_bytes += len(item[1])
        if consumed_bytes > _MAX_RESULT_BYTES:
            _fail(
                "verified profiled sealed-result payload exceeds aggregate byte limit"
            )
        consumed.append((relative, item[1]))
    if consumed_bytes != inventory["totals"]["file_bytes"]:
        _fail("verified profiled sealed-result aggregate byte count differs")
    return tuple(consumed)


__all__ = [
    "RESULT_ANCHOR_ALLOWLIST",
    "RESULT_DIRECTORY_ALLOWLIST",
    "RESULT_FILE_ALLOWLIST",
    "RESULT_FILE_SCHEMA_ALLOWLIST",
    "RESULT_INVENTORY_FILENAME",
    "RESULT_INVENTORY_HASH_DOMAIN",
    "RESULT_INVENTORY_SCHEMA",
    "RESULT_INVENTORY_V2_FILENAME",
    "RESULT_INVENTORY_V2_HASH_DOMAIN",
    "RESULT_INVENTORY_V2_SCHEMA",
    "StrictCapturedReplaySealError",
    "VerifiedSealedResultV1",
    "VerifiedSealedResultV2",
    "consume_verified_sealed_result",
    "consume_verified_sealed_result_v2",
    "publish_sealed_result",
    "publish_sealed_result_v2",
    "verify_sealed_result",
    "verify_sealed_result_v2",
]
