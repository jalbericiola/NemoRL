#!/usr/bin/env python3
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

"""Capture and compose post-job Slurm terminal truth for one strict Pair.

Run ``capture`` once per arm, concurrently with the jobs. Each process polls
only its submission-authenticated job ID and exclusively seals the first exact
COMPLETED/SUCCESS/0 observation. This is deliberately per-arm because HSG may
purge a completed job before its slower peer finishes. Run ``compose`` only
after both immutable arm captures exist. No login-authored scheduler JSON is
accepted as input.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

PAIR_SCHEMA = "nemo-rl-strict-single-env-pair-v2"
SUBMISSION_SCHEMA = "nemo-rl-strict-pair-submission-receipt-v2"
JOB_RECEIPT_SCHEMA = "nemo-rl-strict-pair-job-receipt-v2"
ARM_CAPTURE_SCHEMA = "nemo-rl-strict-single-env-terminal-arm-capture-v1"
PAIR_RECEIPT_SCHEMA = "nemo-rl-strict-single-env-terminal-pair-receipt-v1"
CAPTURE_METHOD = "exact-id-scontrol-show-job-json-poll-v1"
MAX_SCONTROL_BYTES = 4 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_AUTHENTICATED_FILE_BYTES = 32 * 1024 * 1024
MAX_EXACT_INTEGER = 1 << 53
ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "STAGE_OUT",
        "SUSPENDED",
        "RESIZING",
        "REQUEUED",
    }
)
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
UNIX_NS_RE = re.compile(r"[1-9][0-9]{0,19}\Z")
HSG_SLURM_CONF = {
    "path": "/cm/shared/apps/slurm/etc/oci-hsg-cs-001/slurm.conf",
    "sha256": "2f81094a7a631b921d33513e6a3d74b96360510b5f9766f75b0cf45ebd95a410",
}
HSG_SHARED_PROJECT_ANCESTOR = {
    # HSG's one administered shared-project crossing is group-writable. Every
    # other component remains non-writable by group/other and is opened with
    # O_NOFOLLOW; any ownership, group, mode, or path drift fails closed.
    "path": "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text",
    "uid": 0,
    "gid": 20330,
    "mode": 0o775,
}


class TerminalCollectionError(RuntimeError):
    """Terminal scheduler evidence could not be authenticated or captured."""


@dataclass(frozen=True)
class Document:
    value: Any
    raw: bytes
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TerminalCollectionError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise TerminalCollectionError(f"non-finite JSON constant {value}")


def _parse_integer(value: str) -> int:
    if value == "-0":
        raise TerminalCollectionError("negative-zero JSON integer")
    return int(value, 10)


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed == 0.0 and value.startswith("-")):
        raise TerminalCollectionError(f"non-finite or negative-zero JSON float: {value}")
    return parsed


def _parse_json(raw: bytes, label: str) -> Any:
    if b"\0" in raw or b"\r" in raw:
        raise TerminalCollectionError(f"{label} has forbidden framing")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TerminalCollectionError(f"{label} is not UTF-8") from error
    try:
        arguments = {
            "object_pairs_hook": _reject_duplicate_keys,
            "parse_constant": _reject_constant,
            "parse_float": _parse_float,
            "parse_int": _parse_integer,
        }
        return json.loads(text, **arguments)
    except (json.JSONDecodeError, ValueError) as error:
        raise TerminalCollectionError(f"{label} is not strict JSON: {error}") from error


def _exact_json_tree(value: Any, label: str) -> None:
    if value is None or type(value) in {bool, str}:
        return
    if type(value) is int:
        if not -MAX_EXACT_INTEGER <= value <= MAX_EXACT_INTEGER:
            raise TerminalCollectionError(f"{label} integer is outside the exact JSON range")
        return
    if type(value) is float:
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0.0):
            raise TerminalCollectionError(f"{label} float is non-finite or negative zero")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _exact_json_tree(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TerminalCollectionError(f"{label} has a non-string key")
            _exact_json_tree(item, f"{label}.{key}")
        return
    raise TerminalCollectionError(f"{label} has an unsupported JSON type")


def canonical_json_bytes(value: Any, label: str) -> bytes:
    _exact_json_tree(value, label)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise TerminalCollectionError(f"{label} is not canonical JSON") from error


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise TerminalCollectionError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, " f"extra={sorted(set(value) - expected)}"
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TerminalCollectionError(f"{label} must be an exact JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or HEX64_RE.fullmatch(value) is None or set(value) == {"0"}:
        raise TerminalCollectionError(f"{label} must be a populated lowercase SHA-256")
    return value


def _safe_id(value: Any, label: str) -> str:
    if type(value) is not str or SAFE_ID_RE.fullmatch(value) is None:
        raise TerminalCollectionError(f"{label} must be filesystem-safe ASCII")
    return value


def _job_id(value: Any, label: str) -> str:
    if type(value) is not str or JOB_ID_RE.fullmatch(value) is None:
        raise TerminalCollectionError(f"{label} must be canonical positive decimal ASCII")
    return value


def _unix_ns(value: Any, label: str, *, from_clock: bool = False) -> str:
    if from_clock and type(value) is int:
        value = str(value)
    if type(value) is not str or UNIX_NS_RE.fullmatch(value) is None:
        raise TerminalCollectionError(f"{label} must be canonical positive decimal nanoseconds")
    return value


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _trusted_ancestor_metadata(path: Path, metadata: os.stat_result) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    shared_exception = (
        str(path) == HSG_SHARED_PROJECT_ANCESTOR["path"]
        and metadata.st_uid == HSG_SHARED_PROJECT_ANCESTOR["uid"]
        and metadata.st_gid == HSG_SHARED_PROJECT_ANCESTOR["gid"]
        and mode == HSG_SHARED_PROJECT_ANCESTOR["mode"]
    )
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid in {0, os.geteuid()}
        and (mode & 0o022 == 0 or shared_exception)
    )


def _parent_and_leaf(
    path: Path,
    label: str,
    *,
    immediate_parent_mode: int | None = None,
) -> tuple[int, str, Path]:
    if not path.is_absolute() or ".." in path.parts:
        raise TerminalCollectionError(f"{label} path must be supplied as lexical-canonical absolute")
    absolute = Path(os.path.normpath(str(path)))
    if absolute != path or absolute.name in {"", ".", ".."}:
        raise TerminalCollectionError(f"{label} path is not canonical absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        if not _trusted_ancestor_metadata(Path("/"), os.fstat(descriptor)):
            raise TerminalCollectionError(f"{label} root ancestry differs from the closed path policy")
        components = absolute.parts[1:-1]
        traversed = Path("/")
        for index, component in enumerate(components):
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            traversed /= component
            mode = stat.S_IMODE(metadata.st_mode)
            if not _trusted_ancestor_metadata(traversed, metadata):
                raise TerminalCollectionError(f"{label} ancestry differs from the closed path policy")
            if immediate_parent_mode is not None and index == len(components) - 1:
                if metadata.st_uid != os.geteuid() or mode != immediate_parent_mode:
                    raise TerminalCollectionError(
                        f"{label} immediate parent must be current-user mode {immediate_parent_mode:04o}"
                    )
        return descriptor, absolute.name, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _stable_file_bytes(
    path: Path,
    label: str,
    *,
    maximum: int,
    exact_mode: int | None,
    expected_owner: int,
    expected_group: int | None,
    require_executable: bool,
) -> bytes:
    parent_fd, leaf, _ = _parent_and_leaf(path, label)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != expected_owner
            or (expected_group is not None and before.st_gid != expected_group)
            or before.st_nlink != 1
            or mode & 0o022
            or not 1 <= before.st_size <= maximum
            or (exact_mode is not None and mode != exact_mode)
            or (require_executable and mode & 0o111 == 0)
        ):
            raise TerminalCollectionError(f"{label} metadata differs from the closed file policy")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise TerminalCollectionError(f"{label} ended while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TerminalCollectionError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        if _metadata_fingerprint(before) != _metadata_fingerprint(after):
            raise TerminalCollectionError(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise TerminalCollectionError(f"cannot read {label}: {error}") from error
    finally:
        os.close(parent_fd)
        if descriptor is not None:
            os.close(descriptor)


def load_document(path: Path, label: str, *, trailing_lf: bool = True) -> Document:
    raw = _stable_file_bytes(
        path,
        label,
        maximum=MAX_DOCUMENT_BYTES,
        exact_mode=0o400,
        expected_owner=os.geteuid(),
        # HSG result directories are setgid to the administered project group,
        # which need not be the invoking user's effective group. Authenticate
        # user-owned artifacts by UID, exact mode, link count, and stable bytes.
        expected_group=None,
        require_executable=False,
    )
    value = _parse_json(raw, label)
    expected = canonical_json_bytes(value, label) + (b"\n" if trailing_lf else b"")
    if raw != expected:
        raise TerminalCollectionError(f"{label} is not exact canonical ASCII JSON framing")
    return Document(value=value, raw=raw, sha256=hashlib.sha256(raw).hexdigest())


def _validate_document_integrity(document: Document, label: str) -> None:
    parameters = getattr(type(document), "__dataclass_params__", None)
    if (
        not is_dataclass(document)
        or parameters is None
        or not parameters.frozen
        or {field.name for field in fields(document)} != {"value", "raw", "sha256"}
        or type(document.raw) is not bytes
        or type(document.sha256) is not str
    ):
        raise TerminalCollectionError(f"{label} is not an exact authenticated document")
    canonical = canonical_json_bytes(document.value, label) + b"\n"
    actual_sha256 = hashlib.sha256(document.raw).hexdigest()
    if document.raw != canonical or document.sha256 != actual_sha256:
        raise TerminalCollectionError(f"{label} framing or internal digest differs")


def _authenticate(document: Document, expected_sha256: str, label: str) -> None:
    _validate_document_integrity(document, label)
    if document.sha256 != _digest(expected_sha256, f"trusted {label} SHA-256"):
        raise TerminalCollectionError(f"{label} differs from its trusted SHA-256")


def _ascii(value: Any, label: str, maximum: int) -> str:
    if (
        type(value) is not str
        or not 0 <= len(value) <= maximum
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise TerminalCollectionError(f"{label} must be bounded control-free ASCII")
    return value


def _submission_context_unchecked(
    pair_document: Document,
    submission_document: Document,
    *,
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    arm: str,
) -> dict[str, Any]:
    _authenticate(pair_document, expected_pair_sha256, "Pair manifest")
    _authenticate(submission_document, expected_submission_sha256, "submission receipt")
    pair = _mapping(pair_document.value, "Pair manifest")
    submission = _mapping(submission_document.value, "submission receipt")
    if pair.get("schema") != PAIR_SCHEMA:
        raise TerminalCollectionError("unexpected Pair schema")
    if submission.get("schema") != SUBMISSION_SCHEMA:
        raise TerminalCollectionError("unexpected submission schema")
    if arm not in {"off", "on"}:
        raise TerminalCollectionError("arm must be off or on")
    pair_id = _safe_id(pair.get("pair_id"), "Pair ID")
    environment = _safe_id(_mapping(pair.get("selection"), "Pair selection").get("environment"), "environment")
    if _mapping(pair.get("arms"), "Pair arms") != {"off": "observe", "on": "train"}:
        raise TerminalCollectionError("Pair arms differ from exact OFF/ON modes")
    if submission.get("outcome") != "released" or submission.get("stage") != "complete":
        raise TerminalCollectionError("submission receipt is not a completed release")
    submission_pair = _mapping(submission.get("pair"), "submission Pair")
    if submission_pair.get("id") != pair_id:
        raise TerminalCollectionError("submission Pair ID differs")
    if _mapping(submission_pair.get("manifest"), "submission Pair manifest").get("sha256") != pair_document.sha256:
        raise TerminalCollectionError("submission does not bind the Pair bytes")
    if submission.get("wandb") != pair.get("wandb"):
        raise TerminalCollectionError("submission W&B identity differs from Pair")

    held = _mapping(submission.get("held_submissions"), "held submissions")
    authenticated = _mapping(submission.get("authenticated_jobs"), "authenticated jobs")
    _exact_keys(held, {"off", "on"}, "held submissions")
    _exact_keys(authenticated, {"off", "on"}, "authenticated jobs")
    job_ids = {
        selected: _job_id(
            _mapping(held[selected], f"{selected} held submission").get("candidate_job_id"),
            f"{selected} job ID",
        )
        for selected in ("off", "on")
    }
    if job_ids["off"] == job_ids["on"]:
        raise TerminalCollectionError("OFF/ON scheduler IDs alias")
    records: dict[str, dict[str, Any]] = {}
    for selected in ("off", "on"):
        values = authenticated[selected]
        if type(values) is not list or len(values) != 1:
            raise TerminalCollectionError(f"{selected} must have exactly one authenticated job")
        record = _mapping(values[0], f"{selected} authenticated job")
        if record.get("job_id") != job_ids[selected]:
            raise TerminalCollectionError(f"{selected} authenticated job ID differs")
        expected_name = pair["wandb"]["arms"][selected]["name"]
        if record.get("job_name") != expected_name:
            raise TerminalCollectionError(f"{selected} authenticated job name differs")
        expected_comment = (
            f"nemo-rl-strict-pair-v1:{selected}:" f"{submission['submission_nonce']}:{pair_document.sha256}"
        )
        if record.get("comment") != expected_comment:
            raise TerminalCollectionError(f"{selected} authenticated job comment differs")
        user_id = record.get("user_id")
        if type(user_id) is not str or not user_id.isdecimal() or str(int(user_id)) != user_id:
            raise TerminalCollectionError(f"{selected} authenticated user ID differs")
        records[selected] = record

    runtime_tools = _mapping(pair.get("runtime_tools"), "Pair runtime tools")
    runtime_tool_manifest = _mapping(runtime_tools.get("manifest"), "Pair runtime-tool manifest")
    _exact_keys(runtime_tool_manifest, {"path", "sha256"}, "Pair runtime-tool manifest")
    _digest(runtime_tool_manifest["sha256"], "Pair runtime-tool manifest SHA-256")
    host = _mapping(_mapping(runtime_tools.get("document"), "runtime-tool document").get("host"), "host tools")
    scontrol = _mapping(host.get("scontrol"), "Pair scontrol tool")
    _exact_keys(scontrol, {"path", "sha256"}, "Pair scontrol tool")
    _digest(scontrol["sha256"], "Pair scontrol SHA-256")
    if type(scontrol["path"]) is not str or not scontrol["path"].startswith("/"):
        raise TerminalCollectionError("Pair scontrol path is not absolute")
    snapshots = _mapping(_mapping(pair.get("source"), "Pair source").get("snapshots"), "Pair snapshots")
    work_dir = _mapping(snapshots.get(arm), f"Pair {arm} snapshot").get("path")
    if type(work_dir) is not str or not work_dir.startswith("/"):
        raise TerminalCollectionError("Pair snapshot path is not absolute")
    submission_contract = _mapping(
        _mapping(pair.get("scheduler_submission"), "Pair scheduler submission").get("contract"),
        "Pair submission contract",
    )
    _exact_keys(submission_contract, {"path", "sha256"}, "Pair submission contract")
    _digest(submission_contract["sha256"], "Pair submission-contract SHA-256")
    results_dir = _mapping(
        _mapping(pair.get("execution_environment"), "Pair execution environment").get("arms"),
        "Pair execution-environment arms",
    ).get(arm)
    results_dir = _mapping(results_dir, f"Pair {arm} execution environment").get("results_dir")
    if type(results_dir) is not str or not results_dir.startswith("/"):
        raise TerminalCollectionError("Pair result path is not absolute")
    receipt_directory = pair.get("determinism_receipt_dir")
    if receipt_directory != "shared_prefix_determinism_receipts":
        raise TerminalCollectionError("Pair runtime-attestation directory differs")
    return {
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "job_id": job_ids[arm],
        "job_name": records[arm]["job_name"],
        "comment": records[arm]["comment"],
        "user_id": records[arm]["user_id"],
        "work_dir": work_dir,
        "results_dir": results_dir,
        "receipt_directory": receipt_directory,
        "scontrol": dict(scontrol),
        "runtime_tool_manifest_sha256": runtime_tool_manifest["sha256"],
        "submission_contract": dict(submission_contract),
    }


def _submission_context(
    pair_document: Document,
    submission_document: Document,
    *,
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    arm: str,
) -> dict[str, Any]:
    try:
        return _submission_context_unchecked(
            pair_document,
            submission_document,
            expected_pair_sha256=expected_pair_sha256,
            expected_submission_sha256=expected_submission_sha256,
            arm=arm,
        )
    except TerminalCollectionError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise TerminalCollectionError("Pair or submission structure is incomplete") from error


def _validate_exit_receipt_unchecked(
    document: Document,
    *,
    expected_sha256: str,
    context: Mapping[str, Any],
    pair_document: Document,
    submission_document: Document,
) -> None:
    _authenticate(document, expected_sha256, f"{context['arm']} EXIT receipt")
    value = _mapping(document.value, f"{context['arm']} EXIT receipt")
    pair = _mapping(pair_document.value, "Pair manifest")
    expected = {
        "schema": JOB_RECEIPT_SCHEMA,
        "phase": "EXIT",
        "post_verified": True,
        "driver_exit_code": 0,
        "pair_id": context["pair_id"],
        "environment": context["environment"],
        "arm": context["arm"],
        "job_id": context["job_id"],
        "job_name": context["job_name"],
        "restart_count": 0,
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "submission_receipt_path": pair["scheduler_submission"]["receipt"]["path"],
        "submission_contract_path": context["submission_contract"]["path"],
        "submission_contract_sha256": context["submission_contract"]["sha256"],
        "runtime_tool_manifest_sha256": context["runtime_tool_manifest_sha256"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value or type(value[key]) is not type(expected_value):
            raise TerminalCollectionError(f"{context['arm']} EXIT receipt {key} differs")
    for key in ("selection", "source", "execution_environment"):
        if value[key] != pair[key] or type(value[key]) is not type(pair[key]):
            raise TerminalCollectionError(f"{context['arm']} EXIT receipt {key} differs from Pair")
    campaign = _mapping(pair["campaign"], "Pair campaign")
    slurm = _mapping(campaign["slurm"], "Pair campaign Slurm")
    if value["job_account"] != slurm["account"] or value["job_partition"] != slurm["partition"]:
        raise TerminalCollectionError(f"{context['arm']} EXIT scheduler placement differs")
    if type(value["job_num_nodes"]) is not int or value["job_num_nodes"] != 1:
        raise TerminalCollectionError(f"{context['arm']} EXIT node count differs")
    if type(value["gpus_per_node"]) is not int or value["gpus_per_node"] != 4:
        raise TerminalCollectionError(f"{context['arm']} EXIT GPU count differs")
    expected_runtime_dir = f"{context['results_dir']}/{context['receipt_directory']}/" f"{context['job_id']}-0"
    if value["runtime_attestation_receipt_dir"] != expected_runtime_dir:
        raise TerminalCollectionError(f"{context['arm']} EXIT result/job path differs")


def _validate_exit_receipt(
    document: Document,
    *,
    expected_sha256: str,
    context: Mapping[str, Any],
    pair_document: Document,
    submission_document: Document,
) -> None:
    try:
        _validate_exit_receipt_unchecked(
            document,
            expected_sha256=expected_sha256,
            context=context,
            pair_document=pair_document,
            submission_document=submission_document,
        )
    except TerminalCollectionError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        arm = context.get("arm", "unknown") if isinstance(context, Mapping) else "unknown"
        raise TerminalCollectionError(f"{arm} EXIT receipt structure is incomplete") from error


def _typed_time(value: Any, label: str) -> int:
    record = _mapping(value, label)
    _exact_keys(record, {"set", "infinite", "number"}, label)
    if record["set"] is not True or record["infinite"] is not False:
        raise TerminalCollectionError(f"{label} must be set and finite")
    if type(record["number"]) is not int or record["number"] <= 0:
        raise TerminalCollectionError(f"{label}.number must be a positive exact integer")
    return record["number"]


def _typed_root_time(value: Any, label: str, *, zero_allowed: bool) -> int:
    record = _mapping(value, label)
    _exact_keys(record, {"set", "infinite", "number"}, label)
    if record["set"] is not True or record["infinite"] is not False:
        raise TerminalCollectionError(f"{label} must be set and finite")
    number = record["number"]
    minimum = 0 if zero_allowed else 1
    if type(number) is not int or number < minimum:
        raise TerminalCollectionError(f"{label}.number must be an exact integer >= {minimum}")
    return number


def _validate_scontrol_meta(value: Any) -> dict[str, Any]:
    meta = _mapping(value, "scontrol meta")
    _exact_keys(meta, {"plugin", "client", "command", "slurm"}, "scontrol meta")
    plugin = _mapping(meta["plugin"], "scontrol meta.plugin")
    _exact_keys(
        plugin,
        {"type", "name", "data_parser", "accounting_storage"},
        "scontrol meta.plugin",
    )
    expected_plugin = {
        "type": "",
        "name": "",
        "data_parser": "data_parser/v0.0.44",
        "accounting_storage": "accounting_storage/slurmdbd",
    }
    if plugin != expected_plugin:
        raise TerminalCollectionError("scontrol metadata plugin identity differs")
    client = _mapping(meta["client"], "scontrol meta.client")
    _exact_keys(client, {"source", "user", "group"}, "scontrol meta.client")
    if client["source"] != "":
        raise TerminalCollectionError("scontrol metadata client source differs")
    _ascii(client["user"], "scontrol metadata client user", 128)
    _ascii(client["group"], "scontrol metadata client group", 128)
    if not client["user"] or not client["group"]:
        raise TerminalCollectionError("scontrol metadata client identity is empty")
    if meta["command"] != ["show", "job"]:
        raise TerminalCollectionError("scontrol metadata command differs from exact-ID query")
    slurm = _mapping(meta["slurm"], "scontrol meta.slurm")
    _exact_keys(slurm, {"version", "release", "cluster"}, "scontrol meta.slurm")
    version = _mapping(slurm["version"], "scontrol meta.slurm.version")
    if version != {"major": "25", "minor": "11", "micro": "6"}:
        raise TerminalCollectionError("scontrol Slurm version differs")
    if slurm["release"] != "25.11.6" or slurm["cluster"] != "oci-hsg-cs-001":
        raise TerminalCollectionError("scontrol release or cluster differs")
    return meta


def _typed_exit_success(value: Any) -> None:
    exit_code = _mapping(value, "terminal exit_code")
    _exact_keys(exit_code, {"status", "return_code", "signal"}, "terminal exit_code")
    if exit_code["status"] != ["SUCCESS"]:
        raise TerminalCollectionError("terminal exit status is not exact SUCCESS")
    returned = _mapping(exit_code["return_code"], "terminal return_code")
    _exact_keys(returned, {"set", "infinite", "number"}, "terminal return_code")
    if (
        returned["set"] is not True
        or returned["infinite"] is not False
        or type(returned["number"]) is not int
        or returned["number"] != 0
    ):
        raise TerminalCollectionError("terminal return_code is not exact finite zero")
    signal_record = _mapping(exit_code["signal"], "terminal signal")
    _exact_keys(signal_record, {"id", "name"}, "terminal signal")
    signal_id = _mapping(signal_record["id"], "terminal signal ID")
    _exact_keys(signal_id, {"set", "infinite", "number"}, "terminal signal ID")
    if (
        signal_id["set"] is not False
        or signal_id["infinite"] is not False
        or type(signal_id["number"]) is not int
        or signal_id["number"] != 0
        or signal_record["name"] != ""
    ):
        raise TerminalCollectionError("terminal signal is not exact unset zero/empty")


def normalize_scontrol_terminal(raw: bytes, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a strict terminal record, None while active, or fail closed."""
    if not raw.endswith(b"\n") or len(raw) > MAX_SCONTROL_BYTES:
        raise TerminalCollectionError("scontrol output framing or size differs")
    if not raw.startswith(b"{") or not raw.endswith(b"}\n"):
        raise TerminalCollectionError("scontrol output must have exact object-plus-one-LF framing")
    root = _mapping(_parse_json(raw, "scontrol output"), "scontrol output")
    _exact_keys(root, {"errors", "jobs", "last_backfill", "last_update", "meta", "warnings"}, "scontrol root")
    if root["errors"] != [] or type(root["errors"]) is not list:
        raise TerminalCollectionError("scontrol errors must be the exact empty list")
    if root["warnings"] != [] or type(root["warnings"]) is not list:
        raise TerminalCollectionError("scontrol warnings must be the exact empty list")
    _typed_root_time(root["last_backfill"], "scontrol last_backfill", zero_allowed=False)
    _typed_root_time(root["last_update"], "scontrol last_update", zero_allowed=True)
    _validate_scontrol_meta(root["meta"])
    jobs = root["jobs"]
    if type(jobs) is not list or len(jobs) != 1 or type(jobs[0]) is not dict:
        raise TerminalCollectionError("exact scheduler job disappeared or duplicated before capture")
    job = jobs[0]
    required = {
        "comment",
        "current_working_directory",
        "end_time",
        "exit_code",
        "hold",
        "job_id",
        "job_state",
        "name",
        "restart_cnt",
        "start_time",
        "state_reason",
        "user_id",
    }
    if not required.issubset(job):
        raise TerminalCollectionError(f"scontrol job lacks fields {sorted(required - set(job))}")
    if type(job["job_id"]) is not int or str(job["job_id"]) != expected["job_id"]:
        raise TerminalCollectionError("scontrol job ID differs")
    states = job["job_state"]
    if type(states) is not list or len(states) != 1 or type(states[0]) is not str:
        raise TerminalCollectionError("scontrol job_state must be one exact string")
    state = _ascii(states[0], "scontrol job state", 64)
    if state in ACTIVE_STATES:
        return None
    if state != "COMPLETED":
        raise TerminalCollectionError(f"scheduler reached non-success terminal state {state!r}")
    if job["name"] != expected["job_name"] or job["comment"] != expected["comment"]:
        raise TerminalCollectionError("terminal scheduler name/comment differs from submission")
    if type(job["user_id"]) is not int or str(job["user_id"]) != expected["user_id"]:
        raise TerminalCollectionError("terminal scheduler user ID differs from submission")
    if job["current_working_directory"] != expected["work_dir"]:
        raise TerminalCollectionError("terminal scheduler working directory differs from Pair snapshot")
    if type(job["restart_cnt"]) is not int or job["restart_cnt"] != 0:
        raise TerminalCollectionError("terminal scheduler restart count is not exact zero")
    if job["hold"] is not False:
        raise TerminalCollectionError("terminal scheduler hold is not exact false")
    if job["state_reason"] != "None":
        raise TerminalCollectionError("terminal scheduler reason is not exact None")
    start_time = _typed_time(job["start_time"], "terminal start_time")
    end_time = _typed_time(job["end_time"], "terminal end_time")
    if end_time < start_time:
        raise TerminalCollectionError("terminal end_time precedes start_time")
    _typed_exit_success(job["exit_code"])
    return {
        "job_id": expected["job_id"],
        "job_name": expected["job_name"],
        "comment": expected["comment"],
        "user_id": expected["user_id"],
        "work_dir": expected["work_dir"],
        "job_state": "COMPLETED",
        "state_reason": job["state_reason"],
        "restart_count": 0,
        "hold": False,
        "start_time": start_time,
        "end_time": end_time,
        "exit_status": "SUCCESS",
        "return_code": 0,
        "signal": {"set": False, "number": 0, "name": ""},
    }


def _file_sha256(
    path: Path,
    label: str,
    *,
    require_executable: bool,
    expected_owner: int,
    expected_group: int | None,
    exact_mode: int,
) -> str:
    raw = _stable_file_bytes(
        path,
        label,
        maximum=MAX_AUTHENTICATED_FILE_BYTES,
        exact_mode=exact_mode,
        expected_owner=expected_owner,
        expected_group=expected_group,
        require_executable=require_executable,
    )
    return hashlib.sha256(raw).hexdigest()


def _stage_exclusive(path: Path, raw: bytes) -> None:
    if not raw:
        raise TerminalCollectionError("terminal evidence cannot be empty")
    parent_fd, target, _ = _parent_and_leaf(
        path,
        "terminal evidence output",
        immediate_parent_mode=0o700,
    )
    temporary = f".{target}.candidate-{uuid.uuid4().hex}"
    descriptor: int | None = None
    published = False
    complete = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise TerminalCollectionError("short write while staging terminal evidence")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != len(raw)
        ):
            raise TerminalCollectionError("staged terminal evidence metadata differs")
        os.link(
            temporary,
            target,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        if os.fstat(descriptor).st_nlink != 2:
            raise TerminalCollectionError("published terminal evidence link count differs")
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        if os.fstat(descriptor).st_nlink != 1:
            raise TerminalCollectionError("sealed terminal evidence link count differs")
        os.fsync(parent_fd)
        complete = True
    except FileExistsError as error:
        raise TerminalCollectionError(f"output already exists: {path}") from error
    except OSError as error:
        raise TerminalCollectionError(f"cannot stage terminal evidence: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError:
            pass
        if published and not complete:
            try:
                os.unlink(target, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _subprocess_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "SLURM_CONF": HSG_SLURM_CONF["path"],
        },
    )


def capture_arm(
    *,
    pair_document: Document,
    submission_document: Document,
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    expected_collector_sha256: str,
    arm: str,
    output: Path,
    collector_path: Path,
    poll_interval_seconds: float,
    timeout_seconds: float,
    runner: Callable[[Sequence[str]], Any] = _subprocess_runner,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    time_ns: Callable[[], int] = time.time_ns,
    system_owner: int = 0,
    system_group: int = 0,
) -> dict[str, Any]:
    context = _submission_context(
        pair_document,
        submission_document,
        expected_pair_sha256=expected_pair_sha256,
        expected_submission_sha256=expected_submission_sha256,
        arm=arm,
    )
    if type(poll_interval_seconds) is not float or not 0.1 <= poll_interval_seconds <= 60.0:
        raise TerminalCollectionError("poll interval must be a float in [0.1,60]")
    if type(timeout_seconds) is not float or not 1.0 <= timeout_seconds <= 604800.0:
        raise TerminalCollectionError("timeout must be a float in [1,604800]")
    if type(system_owner) is not int or system_owner < 0 or type(system_group) is not int or system_group < 0:
        raise TerminalCollectionError("system owner/group must be exact nonnegative integers")
    collector_sha256 = _file_sha256(
        collector_path,
        "terminal collector",
        require_executable=True,
        expected_owner=os.geteuid(),
        expected_group=None,
        exact_mode=0o500,
    )
    if collector_sha256 != _digest(expected_collector_sha256, "expected terminal collector SHA-256"):
        raise TerminalCollectionError("terminal collector bytes differ from the trusted OOB SHA-256")
    scontrol_path = Path(context["scontrol"]["path"])
    if (
        _file_sha256(
            scontrol_path,
            "Pair-pinned scontrol",
            require_executable=True,
            expected_owner=system_owner,
            expected_group=system_group,
            exact_mode=0o755,
        )
        != context["scontrol"]["sha256"]
    ):
        raise TerminalCollectionError("scontrol bytes differ from the Pair pin")
    slurm_conf_path = Path(HSG_SLURM_CONF["path"])
    if (
        _file_sha256(
            slurm_conf_path,
            "HSG Slurm configuration",
            require_executable=False,
            expected_owner=system_owner,
            expected_group=system_group,
            exact_mode=0o644,
        )
        != HSG_SLURM_CONF["sha256"]
    ):
        raise TerminalCollectionError("HSG Slurm configuration bytes differ")
    argv = [str(scontrol_path), "show", "job", "--json", context["job_id"]]
    deadline = monotonic() + timeout_seconds
    while True:
        started = _unix_ns(time_ns(), "query start time", from_clock=True)
        result = runner(argv)
        finished = _unix_ns(time_ns(), "query finish time", from_clock=True)
        if int(finished) < int(started):
            raise TerminalCollectionError("exact-ID query finish time precedes its start")
        if type(result.returncode) is not int or result.returncode != 0:
            raise TerminalCollectionError("exact-ID scontrol query failed")
        if type(result.stdout) is not bytes or type(result.stderr) is not bytes:
            raise TerminalCollectionError("exact-ID scontrol query streams must be exact bytes")
        stdout = result.stdout
        stderr = result.stderr
        if stderr or len(stderr) > MAX_STDERR_BYTES:
            raise TerminalCollectionError("exact-ID scontrol query wrote stderr")
        record = normalize_scontrol_terminal(stdout, context)
        if record is not None:
            if (
                _file_sha256(
                    scontrol_path,
                    "Pair-pinned scontrol",
                    require_executable=True,
                    expected_owner=system_owner,
                    expected_group=system_group,
                    exact_mode=0o755,
                )
                != context["scontrol"]["sha256"]
            ):
                raise TerminalCollectionError("scontrol bytes changed during capture")
            capture = {
                "schema": ARM_CAPTURE_SCHEMA,
                "capture_method": CAPTURE_METHOD,
                "collector_sha256": collector_sha256,
                "pair_id": context["pair_id"],
                "environment": context["environment"],
                "arm": arm,
                "pair_manifest_sha256": pair_document.sha256,
                "submission_receipt_sha256": submission_document.sha256,
                "submission_contract_sha256": context["submission_contract"]["sha256"],
                "runtime_tool_manifest_sha256": context["runtime_tool_manifest_sha256"],
                "scheduler_tool": context["scontrol"],
                "slurm_conf": dict(HSG_SLURM_CONF),
                "query": {
                    "argv": argv,
                    "return_code": 0,
                    "started_at_unix_ns": started,
                    "finished_at_unix_ns": finished,
                    "raw_stdout_base64": base64.b64encode(stdout).decode("ascii"),
                    "raw_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                    "raw_stdout_byte_count": len(stdout),
                    "raw_stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                    "raw_stderr_byte_count": len(stderr),
                },
                "terminal_record": record,
            }
            raw = canonical_json_bytes(capture, "terminal arm capture") + b"\n"
            _stage_exclusive(output, raw)
            return capture
        if monotonic() >= deadline:
            raise TerminalCollectionError("terminal scheduler capture timed out")
        sleeper(poll_interval_seconds)


def validate_arm_capture(
    document: Document,
    *,
    context: Mapping[str, Any],
    pair_sha256: str,
    submission_sha256: str,
    collector_sha256: str,
) -> dict[str, Any]:
    _validate_document_integrity(document, "terminal arm capture")
    capture = _mapping(document.value, "terminal arm capture")
    _exact_keys(
        capture,
        {
            "schema",
            "capture_method",
            "collector_sha256",
            "pair_id",
            "environment",
            "arm",
            "pair_manifest_sha256",
            "submission_receipt_sha256",
            "submission_contract_sha256",
            "runtime_tool_manifest_sha256",
            "scheduler_tool",
            "slurm_conf",
            "query",
            "terminal_record",
        },
        "terminal arm capture",
    )
    expected_projection = {
        "schema": ARM_CAPTURE_SCHEMA,
        "capture_method": CAPTURE_METHOD,
        "collector_sha256": collector_sha256,
        "pair_id": context["pair_id"],
        "environment": context["environment"],
        "arm": context["arm"],
        "pair_manifest_sha256": pair_sha256,
        "submission_receipt_sha256": submission_sha256,
        "submission_contract_sha256": context["submission_contract"]["sha256"],
        "runtime_tool_manifest_sha256": context["runtime_tool_manifest_sha256"],
        "scheduler_tool": context["scontrol"],
        "slurm_conf": HSG_SLURM_CONF,
    }
    for key, expected in expected_projection.items():
        if capture[key] != expected or type(capture[key]) is not type(expected):
            raise TerminalCollectionError(f"terminal arm capture {key} differs")
    query = _mapping(capture["query"], "terminal arm query")
    _exact_keys(
        query,
        {
            "argv",
            "return_code",
            "started_at_unix_ns",
            "finished_at_unix_ns",
            "raw_stdout_base64",
            "raw_stdout_sha256",
            "raw_stdout_byte_count",
            "raw_stderr_sha256",
            "raw_stderr_byte_count",
        },
        "terminal arm query",
    )
    expected_argv = [context["scontrol"]["path"], "show", "job", "--json", context["job_id"]]
    if query["argv"] != expected_argv or type(query["return_code"]) is not int or query["return_code"] != 0:
        raise TerminalCollectionError("terminal arm query invocation differs")
    started = _unix_ns(query["started_at_unix_ns"], "terminal arm query started_at_unix_ns")
    finished = _unix_ns(query["finished_at_unix_ns"], "terminal arm query finished_at_unix_ns")
    if int(finished) < int(started):
        raise TerminalCollectionError("terminal arm query time ordering differs")
    try:
        raw = base64.b64decode(query["raw_stdout_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise TerminalCollectionError("terminal arm raw stdout is not strict base64") from error
    if (
        hashlib.sha256(raw).hexdigest() != _digest(query["raw_stdout_sha256"], "terminal stdout SHA-256")
        or type(query["raw_stdout_byte_count"]) is not int
        or query["raw_stdout_byte_count"] != len(raw)
        or query["raw_stderr_sha256"] != hashlib.sha256(b"").hexdigest()
        or type(query["raw_stderr_byte_count"]) is not int
        or query["raw_stderr_byte_count"] != 0
    ):
        raise TerminalCollectionError("terminal arm raw query receipt differs")
    normalized = normalize_scontrol_terminal(raw, context)
    if normalized is None or normalized != capture["terminal_record"]:
        raise TerminalCollectionError("terminal arm normalized record differs from raw scheduler output")
    return capture


def _validate_pair_receipt_unchecked(
    document: Document,
    *,
    pair_document: Document,
    submission_document: Document,
    exit_documents: Mapping[str, Document],
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    expected_exit_sha256s: Mapping[str, str],
    expected_collector_sha256: str,
) -> dict[str, Any]:
    """Purely replay and close one composed terminal Pair receipt."""
    contexts = {
        arm: _submission_context(
            pair_document,
            submission_document,
            expected_pair_sha256=expected_pair_sha256,
            expected_submission_sha256=expected_submission_sha256,
            arm=arm,
        )
        for arm in ("off", "on")
    }
    for arm in ("off", "on"):
        _validate_exit_receipt(
            exit_documents[arm],
            expected_sha256=expected_exit_sha256s[arm],
            context=contexts[arm],
            pair_document=pair_document,
            submission_document=submission_document,
        )
    receipt = _mapping(document.value, "terminal Pair receipt")
    _exact_keys(
        receipt,
        {
            "schema",
            "capture_method",
            "collector_sha256",
            "pair_id",
            "environment",
            "pair_manifest_sha256",
            "submission_receipt_sha256",
            "job_exit_receipt_sha256s",
            "submission_contract_sha256",
            "runtime_tool_manifest_sha256",
            "capture_sha256s",
            "composition_sha256",
            "captures",
        },
        "terminal Pair receipt",
    )
    canonical = canonical_json_bytes(receipt, "terminal Pair receipt") + b"\n"
    if document.raw != canonical or document.sha256 != hashlib.sha256(canonical).hexdigest():
        raise TerminalCollectionError("terminal Pair receipt framing or digest differs")
    collector_sha256 = _digest(expected_collector_sha256, "expected terminal collector SHA-256")
    expected_projection = {
        "schema": PAIR_RECEIPT_SCHEMA,
        "capture_method": CAPTURE_METHOD,
        "collector_sha256": collector_sha256,
        "pair_id": contexts["off"]["pair_id"],
        "environment": contexts["off"]["environment"],
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "submission_contract_sha256": contexts["off"]["submission_contract"]["sha256"],
        "runtime_tool_manifest_sha256": contexts["off"]["runtime_tool_manifest_sha256"],
    }
    for key, expected in expected_projection.items():
        if receipt[key] != expected or type(receipt[key]) is not type(expected):
            raise TerminalCollectionError(f"terminal Pair receipt {key} differs")
    expected_exit_digests = {arm: exit_documents[arm].sha256 for arm in ("off", "on")}
    if receipt["job_exit_receipt_sha256s"] != expected_exit_digests:
        raise TerminalCollectionError("terminal Pair EXIT-receipt digests differ")
    capture_digests = _mapping(receipt["capture_sha256s"], "terminal capture digests")
    captures = _mapping(receipt["captures"], "terminal captures")
    _exact_keys(capture_digests, {"off", "on"}, "terminal capture digests")
    _exact_keys(captures, {"off", "on"}, "terminal captures")
    for arm in ("off", "on"):
        embedded_raw = canonical_json_bytes(captures[arm], f"{arm} embedded capture") + b"\n"
        embedded = Document(
            value=captures[arm],
            raw=embedded_raw,
            sha256=hashlib.sha256(embedded_raw).hexdigest(),
        )
        if embedded.sha256 != _digest(capture_digests[arm], f"{arm} capture SHA-256"):
            raise TerminalCollectionError(f"{arm} embedded capture digest differs")
        validate_arm_capture(
            embedded,
            context=contexts[arm],
            pair_sha256=pair_document.sha256,
            submission_sha256=submission_document.sha256,
            collector_sha256=collector_sha256,
        )
    if captures["off"]["terminal_record"]["job_id"] == captures["on"]["terminal_record"]["job_id"]:
        raise TerminalCollectionError("terminal OFF/ON job IDs alias")
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "job_exit_receipt_sha256s": expected_exit_digests,
        "capture_sha256s": dict(capture_digests),
    }
    expected_composition = hashlib.sha256(canonical_json_bytes(composition, "terminal Pair composition")).hexdigest()
    if receipt["composition_sha256"] != expected_composition:
        raise TerminalCollectionError("terminal Pair composition digest differs")
    return receipt


def validate_pair_receipt(
    document: Document,
    *,
    pair_document: Document,
    submission_document: Document,
    exit_documents: Mapping[str, Document],
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    expected_exit_sha256s: Mapping[str, str],
    expected_collector_sha256: str,
) -> dict[str, Any]:
    """Purely replay and exact-close one composed terminal Pair receipt."""
    try:
        return _validate_pair_receipt_unchecked(
            document,
            pair_document=pair_document,
            submission_document=submission_document,
            exit_documents=exit_documents,
            expected_pair_sha256=expected_pair_sha256,
            expected_submission_sha256=expected_submission_sha256,
            expected_exit_sha256s=expected_exit_sha256s,
            expected_collector_sha256=expected_collector_sha256,
        )
    except TerminalCollectionError:
        raise
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise TerminalCollectionError("terminal Pair evidence structure is incomplete") from error


def compose_pair_receipt(
    *,
    pair_document: Document,
    submission_document: Document,
    exit_documents: Mapping[str, Document],
    expected_pair_sha256: str,
    expected_submission_sha256: str,
    expected_exit_sha256s: Mapping[str, str],
    expected_collector_sha256: str,
    capture_documents: Mapping[str, Document],
    output: Path,
    collector_path: Path,
) -> dict[str, Any]:
    collector_sha256 = _file_sha256(
        collector_path,
        "terminal collector",
        require_executable=True,
        expected_owner=os.geteuid(),
        expected_group=None,
        exact_mode=0o500,
    )
    if collector_sha256 != _digest(expected_collector_sha256, "expected terminal collector SHA-256"):
        raise TerminalCollectionError("terminal collector bytes differ from the trusted OOB SHA-256")
    contexts = {
        arm: _submission_context(
            pair_document,
            submission_document,
            expected_pair_sha256=expected_pair_sha256,
            expected_submission_sha256=expected_submission_sha256,
            arm=arm,
        )
        for arm in ("off", "on")
    }
    for arm in ("off", "on"):
        _validate_exit_receipt(
            exit_documents[arm],
            expected_sha256=expected_exit_sha256s[arm],
            context=contexts[arm],
            pair_document=pair_document,
            submission_document=submission_document,
        )
    captures = {
        arm: validate_arm_capture(
            capture_documents[arm],
            context=contexts[arm],
            pair_sha256=pair_document.sha256,
            submission_sha256=submission_document.sha256,
            collector_sha256=collector_sha256,
        )
        for arm in ("off", "on")
    }
    if captures["off"]["terminal_record"]["job_id"] == captures["on"]["terminal_record"]["job_id"]:
        raise TerminalCollectionError("terminal OFF/ON job IDs alias")
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "job_exit_receipt_sha256s": {arm: exit_documents[arm].sha256 for arm in ("off", "on")},
        "capture_sha256s": {arm: capture_documents[arm].sha256 for arm in ("off", "on")},
    }
    receipt = {
        "schema": PAIR_RECEIPT_SCHEMA,
        "capture_method": CAPTURE_METHOD,
        "collector_sha256": collector_sha256,
        "pair_id": contexts["off"]["pair_id"],
        "environment": contexts["off"]["environment"],
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "job_exit_receipt_sha256s": {arm: exit_documents[arm].sha256 for arm in ("off", "on")},
        "submission_contract_sha256": contexts["off"]["submission_contract"]["sha256"],
        "runtime_tool_manifest_sha256": contexts["off"]["runtime_tool_manifest_sha256"],
        "capture_sha256s": {arm: capture_documents[arm].sha256 for arm in ("off", "on")},
        "composition_sha256": hashlib.sha256(
            canonical_json_bytes(composition, "terminal Pair composition")
        ).hexdigest(),
        "captures": captures,
    }
    raw = canonical_json_bytes(receipt, "terminal Pair receipt") + b"\n"
    validated = validate_pair_receipt(
        Document(value=receipt, raw=raw, sha256=hashlib.sha256(raw).hexdigest()),
        pair_document=pair_document,
        submission_document=submission_document,
        exit_documents=exit_documents,
        expected_pair_sha256=expected_pair_sha256,
        expected_submission_sha256=expected_submission_sha256,
        expected_exit_sha256s=expected_exit_sha256s,
        expected_collector_sha256=collector_sha256,
    )
    _stage_exclusive(output, raw)
    return validated


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--expected-pair-manifest-sha256", required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--expected-submission-receipt-sha256", required=True)
    parser.add_argument("--expected-collector-sha256", required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture_parser = subparsers.add_parser("capture")
    _common_arguments(capture_parser)
    capture_parser.add_argument("--arm", choices=("off", "on"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    capture_parser.add_argument("--timeout-seconds", type=float, default=86400.0)
    compose_parser = subparsers.add_parser("compose")
    _common_arguments(compose_parser)
    compose_parser.add_argument("--off-capture", type=Path, required=True)
    compose_parser.add_argument("--on-capture", type=Path, required=True)
    compose_parser.add_argument("--off-exit-receipt", type=Path, required=True)
    compose_parser.add_argument("--on-exit-receipt", type=Path, required=True)
    compose_parser.add_argument("--expected-off-exit-receipt-sha256", required=True)
    compose_parser.add_argument("--expected-on-exit-receipt-sha256", required=True)
    compose_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    collector_path = Path(__file__).resolve(strict=True)
    try:
        pair = load_document(args.pair_manifest, "Pair manifest")
        submission = load_document(args.submission_receipt, "submission receipt")
        if args.mode == "capture":
            value = capture_arm(
                pair_document=pair,
                submission_document=submission,
                expected_pair_sha256=args.expected_pair_manifest_sha256,
                expected_submission_sha256=args.expected_submission_receipt_sha256,
                expected_collector_sha256=args.expected_collector_sha256,
                arm=args.arm,
                output=args.output,
                collector_path=collector_path,
                poll_interval_seconds=args.poll_interval_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            exits = {
                "off": load_document(args.off_exit_receipt, "OFF EXIT receipt"),
                "on": load_document(args.on_exit_receipt, "ON EXIT receipt"),
            }
            value = compose_pair_receipt(
                pair_document=pair,
                submission_document=submission,
                exit_documents=exits,
                expected_pair_sha256=args.expected_pair_manifest_sha256,
                expected_submission_sha256=args.expected_submission_receipt_sha256,
                expected_exit_sha256s={
                    "off": args.expected_off_exit_receipt_sha256,
                    "on": args.expected_on_exit_receipt_sha256,
                },
                expected_collector_sha256=args.expected_collector_sha256,
                capture_documents={
                    "off": load_document(args.off_capture, "OFF terminal capture"),
                    "on": load_document(args.on_capture, "ON terminal capture"),
                },
                output=args.output,
                collector_path=collector_path,
            )
    except (TerminalCollectionError, OSError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "schema": value["schema"],
                "path": str(args.output.resolve(strict=True)),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
