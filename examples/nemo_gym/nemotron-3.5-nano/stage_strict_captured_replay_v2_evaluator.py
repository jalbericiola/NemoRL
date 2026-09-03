#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage the exact source closure for the isolated replay V2 evaluator.

The stager accepts caller-carried SHA-256 values for one Python executable,
the evaluator source, and the ten Python modules needed by the public replay
V2 consumer.  It stable-reads those files without following symlinks, copies
only Python source into a new fixed-layout directory, writes a canonical source
manifest, and seals every staged file to mode 0400 and directory to mode 0555.

The resulting manifest is not self-authenticating.  Its path and digest are an
out-of-band input to the evaluator, which independently verifies this complete
closure before importing any repository module.
"""

from __future__ import annotations

import argparse
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
from typing import Any

PROGRAM_MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-source-manifest-v1"
STAGE_REPORT_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-stage-report-v1"
HASH_DOMAIN = "sha256-canonical-ascii-json-no-lf-v1"
PROGRAM_MANIFEST_FILENAME = "evaluator-source-manifest-v1.json"
STAGED_EVALUATOR_FILENAME = "evaluate_strict_captured_replay_v2.py"
BOOTSTRAP_CONFIG_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-bootstrap-config-v1"
EXECUTION_REPORT_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-execution-report-v1"
EVALUATOR_REQUEST_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-request-v1"
EVALUATOR_REPORT_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-report-v1"
SNAPSHOT_SCHEMA = "nemo-rl-strict-captured-replay-authenticated-result-snapshot-v2"
PARITY_SCHEMA = "nemo-rl-strict-captured-replay-v2-parity-v1"
REQUEST_HASH_DOMAIN = b"nemo-rl-strict-captured-replay-v2-evaluator-request-v1\0"
PARITY_HASH_DOMAIN = b"nemo-rl-strict-captured-replay-v2-evaluator-parity-v1\0"
STAGER_REPOSITORY_PATH = "examples/nemo_gym/nemotron-3.5-nano/" "stage_strict_captured_replay_v2_evaluator.py"
EVALUATOR_REPOSITORY_PATH = "examples/nemo_gym/nemotron-3.5-nano/" "evaluate_strict_captured_replay_v2.py"
GIT_EXECUTABLE = "/usr/bin/git"
GIT_EXECUTABLE_SHA256 = "aa6540695d076182256dd6e96c8b302e4d56381e3000bbfd5c71bbdfe94a4942"
RELEASE_SIGNER = "jalbericiola@nvidia.com"
RELEASE_KEY_FINGERPRINT = "SHA256:FB2kWn6vFC7R7ANXV6rCryx6QYcXWmGbJcF5tE251Cc"
RELEASE_DCO_LINE = "Signed-off-by: Jorge Albericio <jalbericiola@nvidia.com>"
RELEASE_ALLOWED_SIGNER = (
    "jalbericiola@nvidia.com ssh-ed25519 " "AAAAC3NzaC1lZDI1NTE5AAAAIN4uOgShtxlNBlO+AevzxHkddsJRzG34GNpsdB1PbHl8"
)

ATTEMPT_NAMES = ("replay-1", "replay-2")
PROFILE_BY_ENVIRONMENT = {
    "citation": "citation-string-match-v1",
    "freeform": "freeform-regex-v1",
    "reasoning_gym": "reasoning-gym-exact-match-v1",
}
MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
SUBMISSION_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v5"
PRE_SCHEMA = "nemo-rl-strict-captured-replay-job-pre-receipt-v3"
EXIT_SCHEMA = "nemo-rl-strict-captured-replay-job-exit-receipt-v6"
FINAL_SCHEMA = "nemo-rl-strict-captured-replay-result-final-receipt-v1"
INVENTORY_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
INDEX_SCHEMA = "nemo-rl-strict-captured-replay-evidence-index-v4"
OUTPUT_SCHEMAS = {
    "scorer_call_index": "nemo-rl-strict-format-verification-call-index-v1",
    "transport_consumption": "nemo-rl-strict-model-transport-replay-consumption-v3",
    "transcript_bundle": "nemo-rl-strict-step1-transcript-bundle-v4",
    "replay_ledger": "nemo-rl-strict-captured-replay-step1-ledger-v5",
}
OUTPUT_PATHS = {
    "scorer_call_index": "strict_gym_child_runtime/format-verification-call-index.json",
    "transport_consumption": "model-transport-replay-consumption.json",
    "transcript_bundle": "transcript-bundle.json",
    "replay_ledger": "replay-ledger.json",
}
SCORER_INDEX_SCHEMA_BY_ENVIRONMENT = {
    "citation": OUTPUT_SCHEMAS["scorer_call_index"],
    "freeform": OUTPUT_SCHEMAS["scorer_call_index"],
    "reasoning_gym": "nemo-rl-strict-reasoning-score-call-index-v1",
}
SCORER_INDEX_PATH_BY_ENVIRONMENT = {
    "citation": OUTPUT_PATHS["scorer_call_index"],
    "freeform": OUTPUT_PATHS["scorer_call_index"],
    "reasoning_gym": "strict_gym_child_runtime/reasoning-score-call-index.json",
}

COMPANION_SOURCES: dict[str, tuple[str, str]] = {
    "coordinator": (
        "nemo_rl.utils.strict_captured_replay_coordinator_v2",
        "nemo_rl/utils/strict_captured_replay_coordinator_v2.py",
    ),
    "evidence": (
        "nemo_rl.utils.strict_captured_replay_evidence",
        "nemo_rl/utils/strict_captured_replay_evidence.py",
    ),
    "evidence_v2": (
        "nemo_rl.utils.strict_captured_replay_evidence_v2",
        "nemo_rl/utils/strict_captured_replay_evidence_v2.py",
    ),
    "gym_child_runtime_v2": (
        "nemo_rl.environments.strict_gym_child_runtime_v2",
        "nemo_rl/environments/strict_gym_child_runtime_v2.py",
    ),
    "main_step_ledger": (
        "nemo_rl.utils.strict_main_step_ledger",
        "nemo_rl/utils/strict_main_step_ledger.py",
    ),
    "manifest_v2": (
        "nemo_rl.utils.strict_captured_replay_manifest_v2",
        "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
    ),
    "model_transport": (
        "nemo_rl.utils.strict_model_transport",
        "nemo_rl/utils/strict_model_transport.py",
    ),
    "model_transport_replay_v3": (
        "nemo_rl.utils.strict_model_transport_replay_v3",
        "nemo_rl/utils/strict_model_transport_replay_v3.py",
    ),
    "profiles": (
        "nemo_rl.utils.strict_captured_replay_profiles",
        "nemo_rl/utils/strict_captured_replay_profiles.py",
    ),
    "seal_v2": (
        "nemo_rl.utils.strict_captured_replay_seal_v2",
        "nemo_rl/utils/strict_captured_replay_seal_v2.py",
    ),
}

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_GIT_OID_RE = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}\Z", re.ASCII)
_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.ASCII,
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_PYTHON_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 128 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_GIT_OUTPUT_BYTES = 1024 * 1024
_MAX_COMMIT_BYTES = 4 * 1024 * 1024
_EVALUATOR_TIMEOUT_SECONDS = 900.0
_GIT_TIMEOUT_SECONDS = 60.0
_EXACT_CHILD_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
_MODULE_LOAD_ORDER = (
    "evidence",
    "evidence_v2",
    "profiles",
    "manifest_v2",
    "main_step_ledger",
    "model_transport",
    "model_transport_replay_v3",
    "gym_child_runtime_v2",
    "seal_v2",
    "coordinator",
)


class EvaluatorStageError(ValueError):
    """A source or publication boundary failed closed."""


_ISOLATED_BOOTSTRAP = r"""
import hashlib
import importlib.machinery
import json
import os
import stat
import sys
import types

CONFIG_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-bootstrap-config-v1"
MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-source-manifest-v1"
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
COMPANION_KEYS = frozenset({
    "coordinator", "evidence", "evidence_v2", "gym_child_runtime_v2",
    "main_step_ledger", "manifest_v2", "model_transport",
    "model_transport_replay_v3", "profiles", "seal_v2",
})
LOAD_ORDER = (
    "evidence", "evidence_v2", "profiles", "manifest_v2", "main_step_ledger",
    "model_transport", "model_transport_replay_v3", "gym_child_runtime_v2",
    "seal_v2", "coordinator",
)
READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)

def die():
    os._exit(70)

def pairs(items):
    result = {}
    for key, value in items:
        if type(key) is not str or key in result:
            die()
        result[key] = value
    return result

def reject_constant(_value):
    die()

def validate_json(value):
    kind = type(value)
    if value is None or kind in {bool, int, str}:
        return
    if kind is list:
        for item in value:
            validate_json(item)
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                die()
            validate_json(item)
        return
    die()

def canonical(value):
    validate_json(value)
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except BaseException:
        die()

def parse(raw):
    try:
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except BaseException:
        die()
    if type(value) is not dict or canonical(value) != raw:
        die()
    return value

def exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        die()
    return value

def digest(value):
    if (
        type(value) is not str or len(value) != 64 or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        die()
    return value

def read_fd(record):
    exact(record, {"fd", "path", "sha256", "size"})
    descriptor = record["fd"]
    size = record["size"]
    if type(descriptor) is not int or descriptor < 3 or type(size) is not int or size < 1:
        die()
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_size != size
    ):
        die()
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(1 << 20, remaining))
        if not chunk:
            die()
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        die()
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != digest(record["sha256"]):
        die()
    return raw

def verify_python(reference):
    exact(reference, {"path", "sha256", "size"})
    if type(reference["path"]) is not str or reference["path"] != sys.executable:
        die()
    descriptor = os.open(reference["path"], READ_FLAGS)
    try:
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_uid != 0 or metadata.st_gid != 0 or mode != 0o755
            or type(reference["size"]) is not int
            or metadata.st_size != reference["size"]
        ):
            die()
        raw = b""
        while len(raw) < metadata.st_size:
            chunk = os.read(descriptor, min(1 << 20, metadata.st_size - len(raw)))
            if not chunk:
                die()
            raw += chunk
        if os.read(descriptor, 1) or hashlib.sha256(raw).hexdigest() != digest(reference["sha256"]):
            die()
    finally:
        os.close(descriptor)
    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != 0 or metadata.st_gid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            die()
        for component in reference["path"].split("/")[1:-1]:
            child = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            if metadata.st_uid != 0 or metadata.st_gid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
                die()
    finally:
        os.close(descriptor)

def install_package(name):
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = ()
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], child_name, module)

def install_module(name, path, raw):
    if name in sys.modules:
        die()
    module = types.ModuleType(name)
    module.__file__ = path
    module.__package__ = name.rpartition(".")[0]
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=path)
    sys.modules[name] = module
    parent_name, child_name = name.rsplit(".", 1)
    setattr(sys.modules[parent_name], child_name, module)
    try:
        exec(compile(raw, path, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        die()
    return module

def run():
    if (
        len(sys.argv) != 2 or not sys.flags.isolated or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode or os.getcwd() != "/"
        or dict(os.environ) != EXACT_ENVIRONMENT
        or any(name == "nemo_rl" or name.startswith("nemo_rl.") for name in sys.modules)
    ):
        die()
    config_raw = sys.argv[1].encode("ascii")
    config = exact(
        parse(config_raw),
        {"schema", "program", "manifest", "evaluator", "companions"},
    )
    if config["schema"] != CONFIG_SCHEMA:
        die()
    program = exact(config["program"], {"path", "sha256"})
    manifest_raw = read_fd(config["manifest"])
    if hashlib.sha256(manifest_raw).hexdigest() != digest(program["sha256"]):
        die()
    manifest = exact(
        parse(manifest_raw),
        {"schema", "hash_domain", "release", "python", "evaluator", "companions"},
    )
    if manifest["schema"] != MANIFEST_SCHEMA or config["manifest"]["path"] != program["path"]:
        die()
    release = exact(
        manifest["release"],
        {"commit", "tree", "signer", "key_fingerprint", "stager_sha256"},
    )
    if (
        type(release["commit"]) is not str or len(release["commit"]) != 40
        or type(release["tree"]) is not str or len(release["tree"]) != 40
        or type(release["signer"]) is not str
        or type(release["key_fingerprint"]) is not str
        or digest(release["stager_sha256"]) != release["stager_sha256"]
        or release["commit"] == "0" * 40 or release["tree"] == "0" * 40
        or any(character not in "0123456789abcdef" for character in release["commit"])
        or any(character not in "0123456789abcdef" for character in release["tree"])
    ):
        die()
    verify_python(manifest["python"])
    evaluator_record = exact(config["evaluator"], {"fd", "path", "sha256", "size"})
    if {key: evaluator_record[key] for key in ("path", "sha256", "size")} != manifest["evaluator"]:
        die()
    evaluator_raw = read_fd(evaluator_record)
    companions = exact(config["companions"], COMPANION_KEYS)
    manifest_companions = exact(manifest["companions"], COMPANION_KEYS)
    source = {}
    for key in COMPANION_KEYS:
        record = exact(companions[key], {"fd", "module", "path", "sha256", "size"})
        declared = exact(manifest_companions[key], {"module", "path", "sha256", "size"})
        if {name: record[name] for name in ("module", "path", "sha256", "size")} != declared:
            die()
        source[key] = read_fd({name: record[name] for name in ("fd", "path", "sha256", "size")})
    for package in ("nemo_rl", "nemo_rl.utils", "nemo_rl.environments"):
        install_package(package)
    loaded = {}
    for key in LOAD_ORDER:
        record = companions[key]
        loaded[key] = install_module(record["module"], record["path"], source[key])
    coordinator = loaded["coordinator"]
    result_type = vars(coordinator).get("ConsumedReplayResult")
    consume = vars(coordinator).get("consume_replay_result")
    if (
        type(result_type) is not type
        or result_type.__module__ != companions["coordinator"]["module"]
        or result_type.__name__ != "ConsumedReplayResult"
        or type(consume) is not types.FunctionType
        or consume.__module__ != companions["coordinator"]["module"]
        or consume.__name__ != "consume_replay_result"
    ):
        die()
    snapshot_descriptor = vars(result_type).get("snapshot")
    if (
        type(snapshot_descriptor) is not types.MemberDescriptorType
        or snapshot_descriptor.__objclass__ is not result_type
        or snapshot_descriptor.__name__ != "snapshot"
    ):
        die()
    sys.argv = [evaluator_record["path"]]
    globals_dict = {
        "__name__": "__main__",
        "__file__": evaluator_record["path"],
        "__package__": "",
        "__builtins__": __builtins__,
        "_NEMO_RL_V2_EVALUATOR_PROGRAM_REFERENCE": dict(program),
        "_NEMO_RL_V2_COORDINATOR_API": (result_type, consume, snapshot_descriptor),
    }
    exec(compile(evaluator_raw, evaluator_record["path"], "exec", dont_inherit=True), globals_dict)
    die()

try:
    run()
except SystemExit as error:
    if type(error.code) is not int or error.code not in {0, 1}:
        die()
    os._exit(error.code)
except BaseException:
    die()
"""


def _fail(message: str) -> None:
    raise EvaluatorStageError(message)


def _exact_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{name} must be one nonempty exact string")
    return value


def _digest(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if _DIGEST_RE.fullmatch(result) is None or result == "0" * 64:
        _fail(f"{name} must be one nonzero lowercase SHA-256 digest")
    return result


def _git_oid(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if _GIT_OID_RE.fullmatch(result) is None or result == "0" * 40:
        _fail(f"{name} must be one nonzero lowercase 40-hex Git object ID")
    return result


def _normalize_release(value: Any) -> dict[str, str]:
    release = _exact_dict(
        value,
        {"commit", "tree", "signer", "key_fingerprint", "stager_sha256"},
        name="evaluator release",
    )
    result = {
        "commit": _git_oid(release["commit"], name="release commit"),
        "tree": _git_oid(release["tree"], name="release tree"),
        "signer": _exact_string(release["signer"], name="release signer"),
        "key_fingerprint": _exact_string(
            release["key_fingerprint"],
            name="release key fingerprint",
        ),
        "stager_sha256": _digest(
            release["stager_sha256"],
            name="release stager SHA-256",
        ),
    }
    if result["signer"] != RELEASE_SIGNER or result["key_fingerprint"] != RELEASE_KEY_FINGERPRINT:
        _fail("release signer authority differs from the pinned policy")
    return result


def _require_outside_repository(path: str, *, repository_root: str, name: str) -> str:
    result = _canonical_absolute(path, name=name)
    repository = _canonical_absolute(repository_root, name="repository root")
    if result == repository or result.startswith(repository + "/"):
        _fail(f"{name} must remain outside the authenticated repository")
    return result


def _canonical_absolute(value: Any, *, name: str) -> str:
    path = _exact_string(value, name=name)
    try:
        encoded = path.encode("ascii")
    except UnicodeEncodeError:
        _fail(f"{name} must contain only ASCII")
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        _fail(f"{name} must contain only printable ASCII")
    components = path.split("/")
    if (
        not path.startswith("/")
        or path.startswith("//")
        or path.endswith("/")
        or len(components) < 2
        or components[0] != ""
        or any(component in {"", ".", ".."} for component in components[1:])
        or posixpath.normpath(path) != path
    ):
        _fail(f"{name} must be one canonical absolute path")
    return path


def _evaluator_path(value: Any, *, name: str) -> str:
    path = _exact_string(value, name=name)
    if "\x00" in path or "\n" in path or "\r" in path:
        _fail(f"{name} must be one nonempty line without NUL")
    if not path.startswith("/") or path.startswith("//") or path.endswith("/") or posixpath.normpath(path) != path:
        _fail(f"{name} must be one canonical absolute evaluator path")
    return path


def _canonical_json(value: Any, *, allow_float: bool = False) -> bytes:
    def validate(item: Any, *, name: str) -> None:
        item_type = type(item)
        if item is None or item_type in {bool, int, str}:
            return
        if item_type is float:
            if not allow_float or not math.isfinite(item) or (item == 0.0 and math.copysign(1.0, item) < 0):
                _fail(f"{name} contains a forbidden float")
            return
        if item_type is list:
            for index, child in enumerate(item):
                validate(child, name=f"{name}[{index}]")
            return
        if item_type is dict:
            for key, child in item.items():
                if type(key) is not str:
                    _fail(f"{name} contains a non-string key")
                validate(child, name=f"{name}.{key}")
            return
        _fail(f"{name} contains a forbidden JSON type")

    validate(value, name="document")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        _fail(f"cannot encode canonical manifest JSON: {error}")


def _parse_canonical_json(
    raw: bytes,
    *,
    name: str,
    allow_float: bool = False,
) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        _fail(f"{name} contains forbidden JSON constant {value!r}")

    def pairs_to_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                _fail(f"{name} contains a duplicate or non-string key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs_to_dict,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        _fail(f"{name} is not strict ASCII JSON: {error}")
    try:
        canonical = _canonical_json(value, allow_float=allow_float)
    except RecursionError as error:
        _fail(f"{name} exceeds the canonical JSON nesting limit: {error}")
    if type(value) is not dict or canonical != raw:
        _fail(f"{name} is not one canonical JSON object without LF")
    return value


def _exact_dict(value: Any, keys: set[str], *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{name} has the wrong exact key set")
    return value


def _positive_size(value: Any, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(f"{name} must be one bounded positive exact integer")
    return value


def _open_directory(path: str, *, name: str) -> int:
    canonical = _canonical_absolute(path, name=name)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in canonical.split("/")[1:]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{name} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _parent_and_leaf(path: str, *, name: str) -> tuple[int, str]:
    canonical = _canonical_absolute(path, name=name)
    parent, leaf = posixpath.split(canonical)
    if not leaf:
        _fail(f"{name} has no final component")
    return _open_directory(parent, name=f"{name} parent"), leaf


def _metadata_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_read(
    path: str,
    *,
    expected_sha256: str,
    name: str,
    maximum: int,
    executable: bool = False,
    exact_mode: int | None = None,
    root_owned: bool = False,
) -> bytes:
    raw, descriptor, _ = _stable_open(
        path,
        expected_sha256=expected_sha256,
        name=name,
        maximum=maximum,
        executable=executable,
        exact_mode=exact_mode,
        root_owned=root_owned,
    )
    os.close(descriptor)
    return raw


def _stable_open(
    path: str,
    *,
    expected_sha256: str,
    name: str,
    maximum: int,
    executable: bool = False,
    exact_mode: int | None = None,
    root_owned: bool = False,
) -> tuple[bytes, int, tuple[int, ...]]:
    canonical = _canonical_absolute(path, name=name)
    expected = _digest(expected_sha256, name=f"{name} SHA-256")
    parent_fd, leaf = _parent_and_leaf(canonical, name=name)
    try:
        descriptor = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        _fail(f"cannot open {name} without following links: {error}")
    finally:
        if "descriptor" in locals():
            os.close(parent_fd)
    try:
        initial = os.fstat(descriptor)
        mode = stat.S_IMODE(initial.st_mode)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != 1
            or initial.st_uid not in {0, os.geteuid()}
            or mode & 0o022
            or not 1 <= initial.st_size <= maximum
            or (executable and mode & 0o111 == 0)
            or (exact_mode is not None and mode != exact_mode)
            or (root_owned and (initial.st_uid != 0 or initial.st_gid != 0))
        ):
            _fail(f"{name} metadata differs from the closed source policy")
        chunks: list[bytes] = []
        remaining = initial.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                _fail(f"{name} ended before its authenticated size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{name} exceeds its authenticated size")
        final = os.fstat(descriptor)
        if _metadata_fingerprint(initial) != _metadata_fingerprint(final):
            _fail(f"{name} changed while it was read")
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected:
            _fail(f"{name} differs from its caller-carried SHA-256")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return raw, descriptor, _metadata_fingerprint(initial)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_root_owned_ancestors(path: str, *, name: str) -> None:
    canonical = _canonical_absolute(path, name=name)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        root_metadata = os.fstat(descriptor)
        if root_metadata.st_uid != 0 or root_metadata.st_gid != 0 or stat.S_IMODE(root_metadata.st_mode) & 0o022:
            _fail(f"{name} root ancestry differs from the trusted policy")
        for component in canonical.split("/")[1:-1]:
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != 0
                or metadata.st_gid != 0
                or stat.S_IMODE(metadata.st_mode) & 0o022
            ):
                _fail(f"{name} ancestry differs from the trusted policy")
    finally:
        os.close(descriptor)


def _recheck_open_file(
    descriptor: int,
    *,
    expected_raw: bytes,
    expected_fingerprint: tuple[int, ...],
    name: str,
) -> None:
    if _metadata_fingerprint(os.fstat(descriptor)) != expected_fingerprint:
        _fail(f"{name} changed while the isolated evaluator ran")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = len(expected_raw)
    while remaining:
        chunk = os.read(descriptor, min(1 << 20, remaining))
        if not chunk:
            _fail(f"{name} ended during post-execution verification")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1) or b"".join(chunks) != expected_raw:
        _fail(f"{name} bytes changed while the isolated evaluator ran")


def _mkdir_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot exclusively create staged directory {name!r}: {error}")
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(descriptor)
        _fail(f"new staged directory {name!r} metadata differs")
    return descriptor


def _write_at(parent_fd: int, name: str, raw: bytes) -> None:
    complete = False
    try:
        descriptor = os.open(name, _WRITE_FLAGS, 0o400, dir_fd=parent_fd)
    except OSError as error:
        _fail(f"cannot exclusively create staged file {name!r}: {error}")
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail(f"short write while staging {name!r}")
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
            _fail(f"staged file {name!r} metadata differs")
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            try:
                os.unlink(name, dir_fd=parent_fd)
            except OSError:
                pass


def _publish_report_at(
    parent_fd: int,
    name: str,
    raw: bytes,
    *,
    nonce: str,
) -> None:
    temporary_name = f".strict-replay-v2-report-{_digest(nonce, name='report nonce')}.tmp"
    linked = False
    complete = False
    descriptor: int | None = None
    try:
        _write_at(parent_fd, temporary_name, raw)
        descriptor = os.open(temporary_name, _READ_FLAGS, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != len(raw)
        ):
            _fail("sealed evaluator report temporary metadata differs")
        chunks: list[bytes] = []
        remaining = len(raw)
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                _fail("sealed evaluator report temporary ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) or b"".join(chunks) != raw:
            _fail("sealed evaluator report temporary bytes differ")
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            _fail(f"cannot atomically publish evaluator report: {error}")
        linked = True
        if os.fstat(descriptor).st_nlink != 2:
            _fail("published evaluator report link count differs")
        os.unlink(temporary_name, dir_fd=parent_fd)
        if os.fstat(descriptor).st_nlink != 1:
            _fail("sealed evaluator report final link count differs")
        os.fsync(parent_fd)
        complete = True
    finally:
        if not complete:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
            if linked and descriptor is not None:
                try:
                    output_metadata = os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if output_metadata.st_ino == os.fstat(descriptor).st_ino:
                        os.unlink(name, dir_fd=parent_fd)
                except OSError:
                    pass
        if descriptor is not None:
            os.close(descriptor)


def _companion_digests(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != set(COMPANION_SOURCES):
        _fail("companion SHA-256 map has the wrong exact key set")
    return {name: _digest(value[name], name=f"{name} SHA-256") for name in COMPANION_SOURCES}


def _validate_directory(
    path: str,
    *,
    name: str,
    entries: set[str],
) -> tuple[int, tuple[int, ...], str, str, frozenset[str]]:
    descriptor = _open_directory(path, name=name)
    try:
        metadata = os.fstat(descriptor)
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o555
            or set(os.listdir(descriptor)) != entries
        ):
            _fail(f"{name} metadata or exact inventory differs")
        return (
            descriptor,
            _metadata_fingerprint(metadata),
            path,
            name,
            frozenset(entries),
        )
    except BaseException:
        os.close(descriptor)
        raise


def _validate_staged_inventory(
    root: str,
) -> list[tuple[int, tuple[int, ...], str, str, frozenset[str]]]:
    utils_files = {
        posixpath.basename(relative)
        for _, relative in COMPANION_SOURCES.values()
        if relative.startswith("nemo_rl/utils/")
    }
    environment_files = {
        posixpath.basename(relative)
        for _, relative in COMPANION_SOURCES.values()
        if relative.startswith("nemo_rl/environments/")
    }
    records: list[tuple[int, tuple[int, ...], str, str, frozenset[str]]] = []
    try:
        for path, name, entries in (
            (
                root,
                "staged evaluator root",
                {STAGED_EVALUATOR_FILENAME, PROGRAM_MANIFEST_FILENAME, "modules"},
            ),
            (f"{root}/modules", "staged modules directory", {"nemo_rl"}),
            (
                f"{root}/modules/nemo_rl",
                "staged nemo_rl directory",
                {"utils", "environments"},
            ),
            (
                f"{root}/modules/nemo_rl/utils",
                "staged nemo_rl.utils directory",
                utils_files,
            ),
            (
                f"{root}/modules/nemo_rl/environments",
                "staged nemo_rl.environments directory",
                environment_files,
            ),
        ):
            records.append(_validate_directory(path, name=name, entries=entries))
        return records
    except BaseException:
        for descriptor, _, _, _, _ in records:
            os.close(descriptor)
        raise


def _recheck_directory(
    record: tuple[int, tuple[int, ...], str, str, frozenset[str]],
) -> None:
    descriptor, fingerprint, path, name, entries = record
    metadata = os.fstat(descriptor)
    if _metadata_fingerprint(metadata) != fingerprint or frozenset(os.listdir(descriptor)) != entries:
        _fail(f"{name} changed while the isolated evaluator ran")
    _check_named_open_identity(
        path,
        descriptor=descriptor,
        fingerprint=fingerprint,
        name=name,
    )


def _program_file_reference(
    value: Any,
    *,
    name: str,
    maximum: int,
) -> dict[str, Any]:
    reference = _exact_dict(
        value,
        {"path", "sha256", "size"},
        name=name,
    )
    return {
        "path": _canonical_absolute(reference["path"], name=f"{name} path"),
        "sha256": _digest(reference["sha256"], name=f"{name} SHA-256"),
        "size": _positive_size(reference["size"], name=f"{name} size", maximum=maximum),
    }


def _validate_program_manifest(
    value: Any,
    *,
    manifest_path: str,
    expected_release: Mapping[str, str],
) -> dict[str, Any]:
    document = _exact_dict(
        value,
        {"schema", "hash_domain", "release", "python", "evaluator", "companions"},
        name="evaluator source manifest",
    )
    if document["schema"] != PROGRAM_MANIFEST_SCHEMA or document["hash_domain"] != HASH_DOMAIN:
        _fail("evaluator source manifest identity differs")
    root = posixpath.dirname(manifest_path)
    if manifest_path != f"{root}/{PROGRAM_MANIFEST_FILENAME}":
        _fail("evaluator source manifest path differs from the fixed deployment path")
    release = _normalize_release(document["release"])
    if release != _normalize_release(expected_release):
        _fail("evaluator source manifest release differs from OOB authority")
    python = _program_file_reference(
        document["python"],
        name="evaluator Python",
        maximum=_MAX_PYTHON_BYTES,
    )
    evaluator = _program_file_reference(
        document["evaluator"],
        name="evaluator source",
        maximum=_MAX_SOURCE_BYTES,
    )
    if evaluator["path"] != f"{root}/{STAGED_EVALUATOR_FILENAME}":
        _fail("evaluator source path differs from the fixed deployment path")
    supplied = _exact_dict(
        document["companions"],
        set(COMPANION_SOURCES),
        name="evaluator companion closure",
    )
    companions: dict[str, dict[str, Any]] = {}
    total = evaluator["size"]
    for key, (module, relative) in COMPANION_SOURCES.items():
        reference = _exact_dict(
            supplied[key],
            {"module", "path", "sha256", "size"},
            name=f"evaluator companion {key}",
        )
        normalized = _program_file_reference(
            {name: reference[name] for name in ("path", "sha256", "size")},
            name=f"evaluator companion {key}",
            maximum=_MAX_SOURCE_BYTES,
        )
        if type(reference["module"]) is not str or reference["module"] != module:
            _fail(f"evaluator companion {key} module name differs")
        expected_path = f"{root}/modules/{relative}"
        if normalized["path"] != expected_path:
            _fail(f"evaluator companion {key} path differs")
        total += normalized["size"]
        if total > _MAX_TOTAL_SOURCE_BYTES:
            _fail("evaluator source closure exceeds the aggregate byte limit")
        companions[key] = {"module": module, **normalized}
    return {
        "schema": PROGRAM_MANIFEST_SCHEMA,
        "hash_domain": HASH_DOMAIN,
        "release": release,
        "python": python,
        "evaluator": evaluator,
        "companions": companions,
    }


def stage_evaluator_program(
    *,
    repository_root: str,
    release: Mapping[str, str],
    output_root: str,
    python_path: str,
    python_sha256: str,
    evaluator_path: str,
    evaluator_sha256: str,
    producer_root: str,
    companion_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Stage and seal one exact evaluator source closure."""
    repository = _canonical_absolute(repository_root, name="repository root")
    authenticated_release, release_sources = _authenticate_release(
        repository_root=repository,
        expected_release=release,
    )
    root = _require_outside_repository(
        output_root,
        repository_root=repository,
        name="output root",
    )
    interpreter = _canonical_absolute(python_path, name="Python executable")
    evaluator_source = _canonical_absolute(evaluator_path, name="evaluator source")
    producer = _canonical_absolute(producer_root, name="producer root")
    if evaluator_source != f"{repository}/{EVALUATOR_REPOSITORY_PATH}":
        _fail("evaluator source must be the fixed authenticated release member")
    if producer != repository:
        _fail("producer root must equal the authenticated repository root")
    digests = _companion_digests(companion_sha256)

    python_raw = _stable_read(
        interpreter,
        expected_sha256=python_sha256,
        name="Python executable",
        maximum=_MAX_PYTHON_BYTES,
        executable=True,
        exact_mode=0o755,
        root_owned=True,
    )
    _validate_root_owned_ancestors(interpreter, name="Python executable")
    evaluator_raw = release_sources[EVALUATOR_REPOSITORY_PATH]
    if hashlib.sha256(evaluator_raw).hexdigest() != _digest(
        evaluator_sha256,
        name="evaluator source SHA-256",
    ):
        _fail("evaluator source differs from its caller-carried SHA-256")
    companion_raw: dict[str, bytes] = {}
    total = len(evaluator_raw)
    for name, (_, relative) in COMPANION_SOURCES.items():
        raw = release_sources[relative]
        if hashlib.sha256(raw).hexdigest() != digests[name]:
            _fail(f"companion {name} differs from its caller-carried SHA-256")
        total += len(raw)
        if total > _MAX_TOTAL_SOURCE_BYTES:
            _fail("evaluator source closure exceeds the aggregate byte limit")
        companion_raw[name] = raw

    root_parent_fd, root_leaf = _parent_and_leaf(root, name="output root")
    try:
        root_fd = _mkdir_at(root_parent_fd, root_leaf)
    finally:
        os.close(root_parent_fd)
    try:
        _write_at(root_fd, STAGED_EVALUATOR_FILENAME, evaluator_raw)
        modules_fd = _mkdir_at(root_fd, "modules")
        try:
            nemo_fd = _mkdir_at(modules_fd, "nemo_rl")
            try:
                utils_fd = _mkdir_at(nemo_fd, "utils")
                try:
                    environments_fd = _mkdir_at(nemo_fd, "environments")
                    try:
                        for name, (_, relative) in COMPANION_SOURCES.items():
                            parent_name = relative.split("/")[1]
                            destination_fd = utils_fd if parent_name == "utils" else environments_fd
                            _write_at(
                                destination_fd,
                                posixpath.basename(relative),
                                companion_raw[name],
                            )
                        os.fchmod(utils_fd, 0o555)
                        os.fsync(utils_fd)
                        os.fchmod(environments_fd, 0o555)
                        os.fsync(environments_fd)
                    finally:
                        os.close(environments_fd)
                finally:
                    os.close(utils_fd)
                os.fchmod(nemo_fd, 0o555)
                os.fsync(nemo_fd)
            finally:
                os.close(nemo_fd)
            os.fchmod(modules_fd, 0o555)
            os.fsync(modules_fd)
        finally:
            os.close(modules_fd)

        staged_evaluator_path = f"{root}/{STAGED_EVALUATOR_FILENAME}"
        companions: dict[str, dict[str, Any]] = {}
        for name, (module, relative) in COMPANION_SOURCES.items():
            raw = companion_raw[name]
            companions[name] = {
                "module": module,
                "path": f"{root}/modules/{relative}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        manifest = {
            "schema": PROGRAM_MANIFEST_SCHEMA,
            "hash_domain": HASH_DOMAIN,
            "release": authenticated_release,
            "python": {
                "path": interpreter,
                "sha256": hashlib.sha256(python_raw).hexdigest(),
                "size": len(python_raw),
            },
            "evaluator": {
                "path": staged_evaluator_path,
                "sha256": hashlib.sha256(evaluator_raw).hexdigest(),
                "size": len(evaluator_raw),
            },
            "companions": companions,
        }
        manifest_raw = _canonical_json(manifest)
        _write_at(root_fd, PROGRAM_MANIFEST_FILENAME, manifest_raw)
        os.fchmod(root_fd, 0o555)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)

    final_release, final_sources = _authenticate_release(
        repository_root=repository,
        expected_release=authenticated_release,
    )
    if final_release != authenticated_release or final_sources != release_sources:
        _fail("authenticated release changed while the evaluator was staged")

    manifest_path = f"{root}/{PROGRAM_MANIFEST_FILENAME}"
    return {
        "schema": STAGE_REPORT_SCHEMA,
        "manifest": {
            "path": manifest_path,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    # Never signal a process group after its leader has been reaped.  Until the
    # leader is reaped its PID cannot be recycled, so process.returncode is the
    # one safe, non-reaping guard for this cleanup path.
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            return


def _wait_for_process_exit_unreaped(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> None:
    """Wait for the session leader to exit without releasing its PID/PGID."""
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            observation = os.waitid(os.P_PID, process.pid, flags)
        except ChildProcessError:
            _fail("isolated process was reaped before process-group cleanup")
        if observation is not None:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            _fail("isolated evaluator timed out")
        time.sleep(min(remaining, 0.01))


def _collect_process(
    process: subprocess.Popen[bytes],
    *,
    stdout_limit: int = _MAX_REPORT_BYTES,
    stderr_limit: int = 16 * 1024,
    timeout: float = _EVALUATOR_TIMEOUT_SECONDS,
) -> tuple[int, bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        _fail("isolated evaluator pipes are absent")
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout, stdout_limit),
        process.stderr.fileno(): ("stderr", process.stderr, stderr_limit),
    }
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        for descriptor, (name, stream, _) in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, (name, stream))
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                _fail("isolated evaluator timed out")
            events = selector.select(min(remaining, 1.0))
            if not events:
                continue
            for key, _ in events:
                name, stream = key.data
                try:
                    chunk = os.read(key.fd, 1 << 20)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    stream.close()
                    continue
                buffers[name].extend(chunk)
                limit = streams[key.fd][2]
                if len(buffers[name]) > limit:
                    _terminate_process(process)
                    _fail(f"isolated evaluator {name} exceeded its byte limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            _fail("isolated evaluator timed out")
        _wait_for_process_exit_unreaped(process, deadline=deadline)
        # The exited, unreaped leader keeps its PID/PGID reserved while every
        # surviving descendant in the dedicated session is killed.  Reap only
        # after that signal, closing the PID-reuse window.
        _terminate_process(process)
        if process.returncode is None:
            _fail("isolated evaluator leader was not reaped")
        return_code = process.returncode
    finally:
        selector.close()
        for _, stream, _ in streams.values():
            if not stream.closed:
                stream.close()
    return return_code, bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _release_source_paths() -> tuple[str, ...]:
    paths = (
        STAGER_REPOSITORY_PATH,
        EVALUATOR_REPOSITORY_PATH,
        *(relative for _, relative in COMPANION_SOURCES.values()),
    )
    if len(paths) != len(set(paths)):
        _fail("release source closure contains duplicate paths")
    return paths


def _check_named_open_identity(
    path: str,
    *,
    descriptor: int,
    fingerprint: tuple[int, ...],
    name: str,
) -> None:
    canonical = _canonical_absolute(path, name=name)
    try:
        lexical = os.lstat(canonical)
        named = os.stat(canonical, follow_symlinks=False)
    except OSError as error:
        _fail(f"cannot reauthenticate {name} pathname: {error}")
    if (
        stat.S_ISLNK(lexical.st_mode)
        or _metadata_fingerprint(lexical) != fingerprint
        or _metadata_fingerprint(named) != fingerprint
        or _metadata_fingerprint(os.fstat(descriptor)) != fingerprint
    ):
        _fail(f"{name} pathname identity changed")


def _git_command_prefix(*, repository_root: str, allowed_signers: str) -> list[str]:
    return [
        GIT_EXECUTABLE,
        "--no-replace-objects",
        "-c",
        "gpg.format=ssh",
        "-c",
        f"gpg.ssh.allowedSignersFile={allowed_signers}",
        "-c",
        "gpg.ssh.program=/usr/bin/ssh-keygen",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.filemode=true",
        "-C",
        repository_root,
    ]


def _run_git_command(
    *,
    repository_root: str,
    allowed_signers: str,
    private_home: str,
    arguments: Sequence[str],
    stdout_limit: int = _MAX_GIT_OUTPUT_BYTES,
) -> tuple[bytes, bytes]:
    if type(arguments) not in {list, tuple} or not arguments:
        _fail("Git command arguments differ")
    command = _git_command_prefix(
        repository_root=repository_root,
        allowed_signers=allowed_signers,
    ) + [_exact_string(value, name="Git argument") for value in arguments]
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": private_home,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
        "XDG_CACHE_HOME": private_home,
        "XDG_CONFIG_HOME": private_home,
    }
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/",
        env=environment,
        close_fds=True,
        start_new_session=True,
    )
    try:
        return_code, stdout, stderr = _collect_process(
            process,
            stdout_limit=stdout_limit,
            stderr_limit=_MAX_GIT_OUTPUT_BYTES,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    finally:
        _terminate_process(process)
    if return_code != 0:
        _fail(f"authenticated Git command exited with status {return_code}")
    return stdout, stderr


def _require_git_line(
    raw: bytes,
    *,
    expected: str,
    name: str,
) -> None:
    try:
        expected_raw = expected.encode("ascii") + b"\n"
    except UnicodeEncodeError:
        _fail(f"{name} expected value is not ASCII")
    if raw != expected_raw:
        _fail(f"{name} differs from release authority")


def _validate_commit_body(raw_commit: bytes) -> None:
    if (
        type(raw_commit) is not bytes
        or not raw_commit
        or b"\r" in raw_commit
        or b"\x00" in raw_commit
        or b"\n\n" not in raw_commit
    ):
        _fail("release commit framing differs")
    _, body = raw_commit.split(b"\n\n", 1)
    expected = RELEASE_DCO_LINE.encode("ascii")
    dco_lines = [line for line in body.split(b"\n") if b"Signed-off-by:" in line]
    if dco_lines != [expected]:
        _fail("release commit must contain exactly one pinned complete DCO line")


def _authenticate_release(
    *,
    repository_root: str,
    expected_release: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, bytes]]:
    repository = _canonical_absolute(repository_root, name="repository root")
    release = _normalize_release(expected_release)
    repository_fd: int | None = None
    git_fd: int | None = None
    source_records: dict[str, tuple[bytes, int, tuple[int, ...]]] = {}
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        repository_fd = _open_directory(repository, name="repository root")
        repository_fingerprint = _metadata_fingerprint(os.fstat(repository_fd))
        git_raw, git_fd, git_fingerprint = _stable_open(
            GIT_EXECUTABLE,
            expected_sha256=GIT_EXECUTABLE_SHA256,
            name="Git executable",
            maximum=_MAX_PYTHON_BYTES,
            executable=True,
            exact_mode=0o755,
            root_owned=True,
        )
        _validate_root_owned_ancestors(GIT_EXECUTABLE, name="Git executable")
        for relative in _release_source_paths():
            maximum = _MAX_SOURCE_BYTES
            expected = release["stager_sha256"] if relative == STAGER_REPOSITORY_PATH else None
            path = f"{repository}/{relative}"
            if expected is None:
                parent_fd, leaf = _parent_and_leaf(path, name=f"release source {relative}")
                try:
                    descriptor = os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
                try:
                    metadata = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_nlink != 1
                        or metadata.st_uid not in {0, os.geteuid()}
                        or stat.S_IMODE(metadata.st_mode) & 0o022
                        or not 1 <= metadata.st_size <= maximum
                    ):
                        _fail(f"release source {relative} metadata differs")
                    chunks: list[bytes] = []
                    remaining = metadata.st_size
                    while remaining:
                        chunk = os.read(descriptor, min(1 << 20, remaining))
                        if not chunk:
                            _fail(f"release source {relative} ended early")
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    if os.read(descriptor, 1):
                        _fail(f"release source {relative} exceeds its size")
                    final = os.fstat(descriptor)
                    if _metadata_fingerprint(metadata) != _metadata_fingerprint(final):
                        _fail(f"release source {relative} changed while read")
                    raw = b"".join(chunks)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    fingerprint = _metadata_fingerprint(metadata)
                except BaseException:
                    os.close(descriptor)
                    raise
            else:
                raw, descriptor, fingerprint = _stable_open(
                    path,
                    expected_sha256=expected,
                    name="release stager source",
                    maximum=maximum,
                )
            source_records[relative] = (raw, descriptor, fingerprint)

        running_stager = _canonical_absolute(__file__, name="running stager source")
        if running_stager != f"{repository}/{STAGER_REPOSITORY_PATH}":
            _fail("running stager is not the fixed authenticated release member")

        temporary = tempfile.TemporaryDirectory(prefix="strict-replay-v2-release-")
        private_home = _canonical_absolute(
            os.path.realpath(temporary.name),
            name="release verifier private directory",
        )
        private_fd = _open_directory(private_home, name="release verifier private directory")
        try:
            private_metadata = os.fstat(private_fd)
            if private_metadata.st_uid != os.geteuid() or stat.S_IMODE(private_metadata.st_mode) != 0o700:
                _fail("release verifier private directory metadata differs")
            _write_at(
                private_fd,
                "allowed-signers",
                RELEASE_ALLOWED_SIGNER.encode("ascii") + b"\n",
            )
        finally:
            os.close(private_fd)
        allowed_signers = f"{private_home}/allowed-signers"

        def git(*arguments: str, limit: int = _MAX_GIT_OUTPUT_BYTES) -> tuple[bytes, bytes]:
            return _run_git_command(
                repository_root=repository,
                allowed_signers=allowed_signers,
                private_home=private_home,
                arguments=arguments,
                stdout_limit=limit,
            )

        stdout, stderr = git("rev-parse", "--show-toplevel")
        if stderr:
            _fail("Git top-level query wrote to stderr")
        _require_git_line(stdout, expected=repository, name="Git top-level")
        for revision, expected, name in (
            ("HEAD^{commit}", release["commit"], "release HEAD"),
            ("HEAD^{tree}", release["tree"], "release tree"),
        ):
            stdout, stderr = git("rev-parse", "--verify", revision)
            if stderr:
                _fail(f"{name} query wrote to stderr")
            _require_git_line(stdout, expected=expected, name=name)

        _, _ = git("verify-commit", release["commit"])
        signature, stderr = git(
            "show",
            "-s",
            "--format=%G?%x00%GS%x00%GF%x00",
            release["commit"],
        )
        if stderr or signature != (
            b"G\x00" + RELEASE_SIGNER.encode("ascii") + b"\x00" + RELEASE_KEY_FINGERPRINT.encode("ascii") + b"\x00\n"
        ):
            _fail("release commit signature identity differs")
        commit_raw, stderr = git(
            "cat-file",
            "commit",
            release["commit"],
            limit=_MAX_COMMIT_BYTES,
        )
        if stderr:
            _fail("Git commit read wrote to stderr")
        _validate_commit_body(commit_raw)

        for relative, (source_raw, _, _) in source_records.items():
            committed_raw, stderr = git(
                "cat-file",
                "blob",
                f"{release['commit']}:{relative}",
                limit=_MAX_SOURCE_BYTES,
            )
            if stderr or committed_raw != source_raw:
                _fail(f"release source {relative} differs from its committed blob")

        for _ in range(2):
            status, stderr = git(
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            )
            if stderr or status:
                _fail("authenticated repository must be exactly clean")
            for revision, expected, name in (
                ("HEAD^{commit}", release["commit"], "release HEAD"),
                ("HEAD^{tree}", release["tree"], "release tree"),
            ):
                stdout, stderr = git("rev-parse", "--verify", revision)
                if stderr:
                    _fail(f"{name} post-query wrote to stderr")
                _require_git_line(stdout, expected=expected, name=name)

        _recheck_open_file(
            git_fd,
            expected_raw=git_raw,
            expected_fingerprint=git_fingerprint,
            name="Git executable",
        )
        _check_named_open_identity(
            GIT_EXECUTABLE,
            descriptor=git_fd,
            fingerprint=git_fingerprint,
            name="Git executable",
        )
        if _metadata_fingerprint(os.fstat(repository_fd)) != repository_fingerprint:
            _fail("repository root changed while authenticating release")
        _check_named_open_identity(
            repository,
            descriptor=repository_fd,
            fingerprint=repository_fingerprint,
            name="repository root",
        )
        for relative, (raw, descriptor, fingerprint) in source_records.items():
            _recheck_open_file(
                descriptor,
                expected_raw=raw,
                expected_fingerprint=fingerprint,
                name=f"release source {relative}",
            )
            _check_named_open_identity(
                f"{repository}/{relative}",
                descriptor=descriptor,
                fingerprint=fingerprint,
                name=f"release source {relative}",
            )
        return release, {relative: record[0] for relative, record in source_records.items()}
    finally:
        for _, descriptor, _ in source_records.values():
            os.close(descriptor)
        if git_fd is not None:
            os.close(git_fd)
        if repository_fd is not None:
            os.close(repository_fd)
        if temporary is not None:
            temporary.cleanup()


def _open_authenticated_program(
    *,
    program_manifest_path: str,
    program_manifest_sha256: str,
    expected_release: Mapping[str, str],
    release_sources: Mapping[str, bytes],
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    list[tuple[int, bytes, tuple[int, ...], str]],
    list[tuple[int, tuple[int, ...], str, str, frozenset[str]]],
]:
    path = _canonical_absolute(
        program_manifest_path,
        name="evaluator source manifest path",
    )
    expected_sha256 = _digest(
        program_manifest_sha256,
        name="evaluator source manifest SHA-256",
    )
    retained: list[tuple[int, bytes, tuple[int, ...], str]] = []
    directories: list[tuple[int, tuple[int, ...], str, str, frozenset[str]]] = []
    try:
        manifest_raw, manifest_fd, manifest_fingerprint = _stable_open(
            path,
            expected_sha256=expected_sha256,
            name="evaluator source manifest",
            maximum=_MAX_MANIFEST_BYTES,
            exact_mode=0o400,
        )
        retained.append(
            (
                manifest_fd,
                manifest_raw,
                manifest_fingerprint,
                "evaluator source manifest",
            )
        )
        manifest = _validate_program_manifest(
            _parse_canonical_json(manifest_raw, name="evaluator source manifest"),
            manifest_path=path,
            expected_release=expected_release,
        )
        root = posixpath.dirname(path)
        directories = _validate_staged_inventory(root)

        python_raw, python_fd, python_fingerprint = _stable_open(
            manifest["python"]["path"],
            expected_sha256=manifest["python"]["sha256"],
            name="evaluator Python",
            maximum=_MAX_PYTHON_BYTES,
            executable=True,
            exact_mode=0o755,
            root_owned=True,
        )
        retained.append((python_fd, python_raw, python_fingerprint, "evaluator Python"))
        _validate_root_owned_ancestors(
            manifest["python"]["path"],
            name="evaluator Python",
        )
        if len(python_raw) != manifest["python"]["size"]:
            _fail("evaluator Python size differs from the source manifest")

        evaluator_raw, evaluator_fd, evaluator_fingerprint = _stable_open(
            manifest["evaluator"]["path"],
            expected_sha256=manifest["evaluator"]["sha256"],
            name="evaluator source",
            maximum=_MAX_SOURCE_BYTES,
            exact_mode=0o400,
        )
        retained.append((evaluator_fd, evaluator_raw, evaluator_fingerprint, "evaluator source"))
        if len(evaluator_raw) != manifest["evaluator"]["size"]:
            _fail("evaluator source size differs from the source manifest")
        if evaluator_raw != release_sources.get(EVALUATOR_REPOSITORY_PATH):
            _fail("staged evaluator source differs from the authenticated release")

        companion_records: dict[str, dict[str, Any]] = {}
        for key in COMPANION_SOURCES:
            reference = manifest["companions"][key]
            raw, descriptor, fingerprint = _stable_open(
                reference["path"],
                expected_sha256=reference["sha256"],
                name=f"evaluator companion {key}",
                maximum=_MAX_SOURCE_BYTES,
                exact_mode=0o400,
            )
            retained.append((descriptor, raw, fingerprint, f"evaluator companion {key}"))
            if len(raw) != reference["size"]:
                _fail(f"evaluator companion {key} size differs from the source manifest")
            if raw != release_sources.get(COMPANION_SOURCES[key][1]):
                _fail(f"evaluator companion {key} differs from the authenticated release")
            companion_records[key] = {
                **reference,
                "fd": descriptor,
            }
        program = {"path": path, "sha256": expected_sha256}
        config = {
            "schema": BOOTSTRAP_CONFIG_SCHEMA,
            "program": program,
            "manifest": {
                "fd": manifest_fd,
                "path": path,
                "sha256": expected_sha256,
                "size": len(manifest_raw),
            },
            "evaluator": {
                "fd": evaluator_fd,
                **manifest["evaluator"],
            },
            "companions": companion_records,
        }
        return manifest, program, config, retained, directories
    except BaseException:
        for descriptor, _, _, _ in retained:
            os.close(descriptor)
        for descriptor, _, _, _, _ in directories:
            os.close(descriptor)
        raise


def _validate_evaluator_request(
    raw: bytes,
    *,
    program: dict[str, str],
) -> dict[str, Any]:
    request = _parse_canonical_json(raw, name="evaluator request")
    request = _exact_dict(
        request,
        {"schema", "nonce", "request_sha256", "evaluator_program", "pair", "attempts"},
        name="evaluator request",
    )
    if type(request["schema"]) is not str or request["schema"] != EVALUATOR_REQUEST_SCHEMA:
        _fail("evaluator request schema differs")
    _digest(request["nonce"], name="evaluator request nonce")
    supplied_request_sha256 = _digest(
        request["request_sha256"],
        name="evaluator request projection SHA-256",
    )
    projection = dict(request)
    projection.pop("request_sha256")
    expected_request_sha256 = hashlib.sha256(REQUEST_HASH_DOMAIN + _canonical_json(projection)).hexdigest()
    if supplied_request_sha256 != expected_request_sha256:
        _fail("evaluator request projection SHA-256 differs")
    reference = _exact_dict(
        request["evaluator_program"],
        {"path", "sha256"},
        name="evaluator request program reference",
    )
    normalized = {
        "path": _canonical_absolute(
            reference["path"],
            name="evaluator request program path",
        ),
        "sha256": _digest(
            reference["sha256"],
            name="evaluator request program SHA-256",
        ),
    }
    if normalized != program:
        _fail("evaluator request selects a different source manifest")

    pair = _exact_dict(
        request["pair"],
        {
            "pair_id",
            "environment",
            "profile_id",
            "manifest",
            "submission_receipt",
            "off_exit_receipt",
        },
        name="evaluator request Pair authority",
    )
    pair_id = _exact_string(pair["pair_id"], name="evaluator request Pair ID")
    if _PAIR_ID_RE.fullmatch(pair_id) is None:
        _fail("evaluator request Pair ID differs")
    environment = _exact_string(
        pair["environment"],
        name="evaluator request environment",
    )
    profile_id = _exact_string(
        pair["profile_id"],
        name="evaluator request profile ID",
    )
    if PROFILE_BY_ENVIRONMENT.get(environment) != profile_id:
        _fail("evaluator request environment/profile dispatch differs")
    pair_references = {
        key: _artifact_reference(pair[key], name=f"request Pair {key}")
        for key in ("manifest", "submission_receipt", "off_exit_receipt")
    }
    for member in ("path", "sha256"):
        if len({reference[member] for reference in pair_references.values()}) != 3:
            _fail(f"evaluator request Pair authority {member}s must be distinct")

    attempts = _exact_dict(
        request["attempts"],
        set(ATTEMPT_NAMES),
        name="evaluator request attempts",
    )
    candidates: set[str] = set()
    paths: set[str] = set()
    digest_sets = {name: set() for name in ("manifest", "submission", "final")}
    for attempt in ATTEMPT_NAMES:
        item = _exact_dict(
            attempts[attempt],
            {
                "replay_execution_manifest",
                "submission_receipt_sha256",
                "candidate_job_id",
                "result_final_receipt",
            },
            name=f"evaluator request {attempt}",
        )
        attempt_manifest = _artifact_reference(
            item["replay_execution_manifest"],
            name=f"evaluator request {attempt} manifest",
        )
        submission_sha256 = _digest(
            item["submission_receipt_sha256"],
            name=f"evaluator request {attempt} submission SHA-256",
        )
        candidate = _job_id(
            item["candidate_job_id"],
            name=f"evaluator request {attempt} candidate job ID",
        )
        final = _artifact_reference(
            item["result_final_receipt"],
            name=f"evaluator request {attempt} FINAL receipt",
        )
        if candidate in candidates:
            _fail("evaluator request candidate job IDs must be distinct")
        candidates.add(candidate)
        for digest, key in (
            (attempt_manifest["sha256"], "manifest"),
            (submission_sha256, "submission"),
            (final["sha256"], "final"),
        ):
            if digest in digest_sets[key]:
                _fail(f"evaluator request {key} digests must be distinct")
            digest_sets[key].add(digest)
        for path in (attempt_manifest["path"], final["path"]):
            if path in paths:
                _fail("evaluator request replay authority paths must be distinct")
            paths.add(path)
    return request


def _artifact_reference(
    value: Any,
    *,
    name: str,
    schema: str | None = None,
) -> dict[str, str]:
    keys = {"path", "sha256"} if schema is None else {"path", "schema", "sha256"}
    reference = _exact_dict(value, keys, name=name)
    result = {
        "path": _evaluator_path(reference["path"], name=f"{name} path"),
        "sha256": _digest(reference["sha256"], name=f"{name} SHA-256"),
    }
    if schema is not None:
        if type(reference["schema"]) is not str or reference["schema"] != schema:
            _fail(f"{name} schema differs")
        result["schema"] = schema
    return result


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{name} must be one exact bounded integer")
    return value


def _job_id(value: Any, *, name: str) -> str:
    result = _exact_string(value, name=name)
    if _JOB_ID_RE.fullmatch(result) is None or len(result) > 19 or int(result) > (1 << 63) - 1:
        _fail(f"{name} must be one canonical positive decimal signed-64 ID")
    return result


def _require_artifact(
    value: Any,
    *,
    name: str,
    path: str,
    schema: str,
    sha256: str | None = None,
) -> dict[str, str]:
    result = _artifact_reference(value, name=name, schema=schema)
    if result["path"] != path or (sha256 is not None and result["sha256"] != sha256):
        _fail(f"{name} authority differs")
    return result


def _validate_report_processes(snapshot: dict[str, Any], *, attempt: str) -> None:
    driver = _exact_dict(
        snapshot["driver_process"],
        {"boot_id_sha256", "pid", "start_time_ticks"},
        name=f"{attempt} driver process",
    )
    driver_boot = _digest(
        driver["boot_id_sha256"],
        name=f"{attempt} driver boot ID SHA-256",
    )
    driver_pid = _bounded_int(
        driver["pid"],
        name=f"{attempt} driver PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    _bounded_int(
        driver["start_time_ticks"],
        name=f"{attempt} driver start ticks",
        minimum=1,
        maximum=(1 << 63) - 1,
    )
    scorer = _exact_dict(
        snapshot["scorer_process_identity"],
        {"boot_id", "hostname", "pid", "start_ticks"},
        name=f"{attempt} scorer process",
    )
    boot_id = _exact_string(scorer["boot_id"], name=f"{attempt} scorer boot ID")
    if _BOOT_ID_RE.fullmatch(boot_id) is None:
        _fail(f"{attempt} scorer boot ID differs")
    hostname = _exact_string(
        scorer["hostname"],
        name=f"{attempt} scorer hostname",
    )
    try:
        hostname_raw = hostname.encode("ascii")
    except UnicodeEncodeError:
        _fail(f"{attempt} scorer hostname must be ASCII")
    if len(hostname_raw) > 255 or any(byte in {0, 10, 13} for byte in hostname_raw):
        _fail(f"{attempt} scorer hostname differs")
    scorer_pid = _bounded_int(
        scorer["pid"],
        name=f"{attempt} scorer PID",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    _bounded_int(
        scorer["start_ticks"],
        name=f"{attempt} scorer start ticks",
        minimum=1,
        maximum=(1 << 63) - 1,
    )
    if driver_boot != hashlib.sha256((boot_id + "\n").encode("ascii")).hexdigest():
        _fail(f"{attempt} driver and scorer boot identities differ")
    if driver_pid == scorer_pid:
        _fail(f"{attempt} driver and scorer process identities alias")


def _validate_report_samples(
    samples: Any,
    *,
    environment: str,
    attempt: str,
) -> list[dict[str, Any]]:
    if type(samples) is not list or len(samples) != 4:
        _fail(f"{attempt} must contain exact K=4 samples")
    sample_keys = {
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
    for index, value in enumerate(samples):
        sample = _exact_dict(value, sample_keys, name=f"{attempt} sample {index}")
        for field, expected in (
            ("sample_index", index),
            ("fixture_row_index", 0),
            ("rollout_index", index),
        ):
            if type(sample[field]) is not int or sample[field] != expected:
                _fail(f"{attempt} sample {index} {field} differs")
        _bounded_int(
            sample["generation_seed"],
            name=f"{attempt} sample {index} generation seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        for field in (
            "model_transport_entry_sha256",
            "model_transport_request_body_sha256",
            "model_transport_response_body_sha256",
            "model_response_sha256",
        ):
            _digest(sample[field], name=f"{attempt} sample {index} {field}")
        details = sample["match_details"]
        if environment == "citation":
            details = _exact_dict(
                details,
                {"expected", "missing", "spurious", "passed"},
                name=f"{attempt} citation sample {index} details",
            )
            for field in ("expected", "missing", "spurious"):
                items = details[field]
                if type(items) is not list or any(type(item) is not str for item in items):
                    _fail(f"{attempt} citation sample {index} {field} differs")
            passed = details["passed"]
            if type(passed) is not bool or passed is not (not details["missing"] and not details["spurious"]):
                _fail(f"{attempt} citation sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif environment == "freeform":
            details = _exact_dict(
                details,
                {"matching_lines", "min_matches", "passed"},
                name=f"{attempt} freeform sample {index} details",
            )
            matching = _bounded_int(
                details["matching_lines"],
                name=f"{attempt} freeform sample {index} matches",
                minimum=0,
                maximum=(1 << 31) - 1,
            )
            minimum = _bounded_int(
                details["min_matches"],
                name=f"{attempt} freeform sample {index} minimum",
                minimum=0,
                maximum=(1 << 31) - 1,
            )
            passed = details["passed"]
            if type(passed) is not bool or passed is not (matching >= minimum):
                _fail(f"{attempt} freeform sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif environment == "reasoning_gym":
            details = _exact_dict(
                details,
                {"task_name", "score", "extracted_answer"},
                name=f"{attempt} reasoning sample {index} details",
            )
            if type(details["task_name"]) is not str or details["task_name"] != "knights_knaves":
                _fail(f"{attempt} reasoning sample {index} task name differs")
            score = details["score"]
            if (
                type(score) is not float
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
                or (score == 0.0 and math.copysign(1.0, score) < 0.0)
            ):
                _fail(f"{attempt} reasoning sample {index} score differs")
            if type(details["extracted_answer"]) is not str:
                _fail(f"{attempt} reasoning sample {index} extracted answer differs")
            expected_reward = score
        else:
            _fail(f"{attempt} environment differs")
        reward = sample["raw_environment_reward"]
        if (
            type(reward) is not float
            or not math.isfinite(reward)
            or not 0.0 <= reward <= 1.0
            or (reward == 0.0 and math.copysign(1.0, reward) < 0)
            or reward != expected_reward
        ):
            _fail(f"{attempt} sample {index} reward differs")
    return samples


def _validate_report_snapshot(
    value: Any,
    *,
    attempt: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    snapshot_keys = {
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
    snapshot = _exact_dict(value, snapshot_keys, name=f"{attempt} snapshot")
    if type(snapshot["schema"]) is not str or snapshot["schema"] != SNAPSHOT_SCHEMA:
        _fail(f"{attempt} snapshot schema differs")
    pair = request["pair"]
    requested = request["attempts"][attempt]
    expected_identity = {
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "profile_id": pair["profile_id"],
        "attempt_id": attempt,
        "candidate_job_id": requested["candidate_job_id"],
        "authenticated_job_id": requested["candidate_job_id"],
    }
    if any(type(snapshot[key]) is not str or snapshot[key] != expected for key, expected in expected_identity.items()):
        _fail(f"{attempt} snapshot identity differs")
    expected_run_id = hashlib.sha256(
        f"nemo-rl-strict-replay-v2:{pair['environment']}:{pair['pair_id']}:{attempt}".encode("ascii")
    ).hexdigest()
    if type(snapshot["run_id"]) is not str or snapshot["run_id"] != expected_run_id:
        _fail(f"{attempt} snapshot run ID differs")
    _validate_report_processes(snapshot, attempt=attempt)
    result_root = _evaluator_path(snapshot["result_root"], name=f"{attempt} result root")
    suffix = f"/captured_replay/{attempt}"
    if not result_root.endswith(suffix) or len(result_root) <= len(suffix):
        _fail(f"{attempt} result root differs")
    results_root = result_root[: -len(suffix)]
    manifest_path = f"{results_root}/captured_replay/manifests/{pair['pair_id']}/{attempt}.json"
    submission_path = (
        f"{results_root}/captured_replay/replay_submission_state/"
        f"{pair['pair_id']}/{attempt}/submission-receipt.json"
    )
    receipt_root = (
        f"{results_root}/captured_replay/replay_job_state/{pair['pair_id']}/{attempt}/"
        f"{requested['candidate_job_id']}-0/receipts"
    )
    _require_artifact(
        snapshot["manifest"],
        name=f"{attempt} manifest",
        path=manifest_path,
        schema=MANIFEST_SCHEMA,
        sha256=requested["replay_execution_manifest"]["sha256"],
    )
    if requested["replay_execution_manifest"]["path"] != manifest_path:
        _fail(f"{attempt} request manifest path differs")
    _require_artifact(
        snapshot["submission_receipt"],
        name=f"{attempt} submission receipt",
        path=submission_path,
        schema=SUBMISSION_SCHEMA,
        sha256=requested["submission_receipt_sha256"],
    )
    _require_artifact(
        snapshot["pre_receipt"], name=f"{attempt} PRE", path=f"{receipt_root}/PRE.json", schema=PRE_SCHEMA
    )
    _require_artifact(
        snapshot["exit_receipt"], name=f"{attempt} EXIT", path=f"{receipt_root}/EXIT.json", schema=EXIT_SCHEMA
    )
    final_path = f"{receipt_root}/FINAL.json"
    _require_artifact(
        snapshot["result_final_receipt"],
        name=f"{attempt} FINAL",
        path=final_path,
        schema=FINAL_SCHEMA,
        sha256=requested["result_final_receipt"]["sha256"],
    )
    if requested["result_final_receipt"]["path"] != final_path:
        _fail(f"{attempt} request FINAL path differs")
    _require_artifact(
        snapshot["result_inventory"],
        name=f"{attempt} inventory",
        path=f"{result_root}/result-inventory-v2.json",
        schema=INVENTORY_SCHEMA,
    )
    _require_artifact(
        snapshot["evidence_index"],
        name=f"{attempt} evidence index",
        path=f"{result_root}/evidence-index.json",
        schema=INDEX_SCHEMA,
    )
    outputs = _exact_dict(snapshot["outputs"], set(OUTPUT_SCHEMAS), name=f"{attempt} outputs")
    output_schemas = dict(OUTPUT_SCHEMAS)
    output_paths = dict(OUTPUT_PATHS)
    output_schemas["scorer_call_index"] = SCORER_INDEX_SCHEMA_BY_ENVIRONMENT[pair["environment"]]
    output_paths["scorer_call_index"] = SCORER_INDEX_PATH_BY_ENVIRONMENT[pair["environment"]]
    for output_name, schema in output_schemas.items():
        _require_artifact(
            outputs[output_name],
            name=f"{attempt} output {output_name}",
            path=f"{result_root}/{output_paths[output_name]}",
            schema=schema,
        )
    _validate_report_samples(snapshot["samples"], environment=pair["environment"], attempt=attempt)
    return snapshot


def _validate_evaluator_report(
    raw: bytes,
    *,
    request: dict[str, Any],
    program: dict[str, str],
) -> dict[str, Any]:
    report = _parse_canonical_json(raw, name="evaluator report", allow_float=True)
    report = _exact_dict(
        report,
        {
            "schema",
            "status",
            "nonce",
            "request_sha256",
            "evaluator_program",
            "pair",
            "attempts",
            "parity",
        },
        name="authenticated evaluator report",
    )
    if (
        type(report["schema"]) is not str
        or report["schema"] != EVALUATOR_REPORT_SCHEMA
        or type(report["status"]) is not str
        or report["status"] != "authenticated"
        or report["nonce"] != request["nonce"]
        or report["request_sha256"] != request["request_sha256"]
        or report["evaluator_program"] != program
    ):
        _fail("authenticated evaluator report does not bind its request and program")
    expected_pair = {
        "pair_id": request["pair"]["pair_id"],
        "environment": request["pair"]["environment"],
        "profile_id": request["pair"]["profile_id"],
        "manifest_sha256": request["pair"]["manifest"]["sha256"],
        "submission_receipt_sha256": request["pair"]["submission_receipt"]["sha256"],
        "off_exit_receipt_sha256": request["pair"]["off_exit_receipt"]["sha256"],
    }
    if type(report["pair"]) is not dict or report["pair"] != expected_pair:
        _fail("authenticated evaluator report Pair echo differs")
    attempts = _exact_dict(report["attempts"], set(ATTEMPT_NAMES), name="authenticated report attempts")
    snapshots = {
        attempt: _validate_report_snapshot(
            attempts[attempt],
            attempt=attempt,
            request=request,
        )
        for attempt in ATTEMPT_NAMES
    }
    first = snapshots["replay-1"]
    second = snapshots["replay-2"]
    for field in (
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
    ):
        if _canonical_json(first[field], allow_float=True) == _canonical_json(second[field], allow_float=True):
            _fail(f"authenticated replay attempts reuse {field}")
    for field in (
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_inventory",
        "evidence_index",
    ):
        for member in ("path", "sha256"):
            if first[field][member] == second[field][member]:
                _fail(f"authenticated replay attempts reuse {field}.{member}")
    if first["result_root"] == second["result_root"]:
        _fail("authenticated replay attempts reuse result root")
    for output_name in OUTPUT_SCHEMAS:
        for member in ("path", "sha256"):
            if first["outputs"][output_name][member] == second["outputs"][output_name][member]:
                _fail(f"authenticated replay attempts reuse outputs.{output_name}.{member}")
    first_samples = _canonical_json(first["samples"], allow_float=True)
    second_samples = _canonical_json(second["samples"], allow_float=True)
    if first_samples != second_samples:
        _fail("authenticated replay sample evidence differs")
    rewards = [sample["raw_environment_reward"] for sample in first["samples"]]
    expected_parity = {
        "schema": PARITY_SCHEMA,
        "status": "exact-match",
        "samples_sha256": hashlib.sha256(PARITY_HASH_DOMAIN + first_samples).hexdigest(),
        "reward_vector": rewards,
    }
    parity = _exact_dict(
        report["parity"],
        {"schema", "status", "samples_sha256", "reward_vector"},
        name="authenticated evaluator parity",
    )
    vector = parity["reward_vector"]
    if (
        type(vector) is not list
        or len(vector) != 4
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            or (value == 0.0 and math.copysign(1.0, value) < 0.0)
            for value in vector
        )
        or _canonical_json(parity, allow_float=True) != _canonical_json(expected_parity, allow_float=True)
    ):
        _fail("authenticated evaluator report parity differs")
    return report


def verify_and_run_evaluator(
    *,
    repository_root: str,
    release: Mapping[str, str],
    program_manifest_path: str,
    program_manifest_sha256: str,
    request_path: str,
    request_sha256: str,
    report_path: str,
) -> dict[str, Any]:
    """Authenticate a staged program and run its worker in a fresh interpreter."""
    repository = _canonical_absolute(repository_root, name="repository root")
    authenticated_release, release_sources = _authenticate_release(
        repository_root=repository,
        expected_release=release,
    )
    manifest, program, config, retained, directories = _open_authenticated_program(
        program_manifest_path=program_manifest_path,
        program_manifest_sha256=program_manifest_sha256,
        expected_release=authenticated_release,
        release_sources=release_sources,
    )
    process: subprocess.Popen[bytes] | None = None
    request_fd: int | None = None
    request_fingerprint: tuple[int, ...] | None = None
    output_parent_fd: int | None = None
    output_parent_fingerprint: tuple[int, ...] | None = None
    try:
        request_raw, request_fd, request_fingerprint = _stable_open(
            request_path,
            expected_sha256=request_sha256,
            name="evaluator request",
            maximum=_MAX_REQUEST_BYTES,
            exact_mode=0o400,
        )
        request = _validate_evaluator_request(request_raw, program=program)
        output = _require_outside_repository(
            report_path,
            repository_root=repository,
            name="evaluator report output",
        )
        output_parent_fd, output_leaf = _parent_and_leaf(
            output,
            name="evaluator report output",
        )
        output_parent_fingerprint = _metadata_fingerprint(os.fstat(output_parent_fd))
        try:
            os.stat(output_leaf, dir_fd=output_parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail("evaluator report output must be absent")

        pass_fds = tuple(record[0] for record in retained if record[3] != "evaluator Python")
        config_raw = _canonical_json(config).decode("ascii")
        with tempfile.TemporaryFile(mode="w+b") as request_stream:
            request_stream.write(request_raw)
            request_stream.flush()
            request_stream.seek(0)
            process = subprocess.Popen(
                [
                    manifest["python"]["path"],
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _ISOLATED_BOOTSTRAP,
                    config_raw,
                ],
                stdin=request_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/",
                env=dict(_EXACT_CHILD_ENVIRONMENT),
                close_fds=True,
                pass_fds=pass_fds,
                start_new_session=True,
            )
            return_code, stdout, stderr = _collect_process(process)
            process = None

        for descriptor, raw, fingerprint, name in retained:
            _recheck_open_file(
                descriptor,
                expected_raw=raw,
                expected_fingerprint=fingerprint,
                name=name,
            )
        for directory in directories:
            _recheck_directory(directory)
        _recheck_open_file(
            request_fd,
            expected_raw=request_raw,
            expected_fingerprint=request_fingerprint,
            name="evaluator request",
        )
        _check_named_open_identity(
            _canonical_absolute(request_path, name="evaluator request path"),
            descriptor=request_fd,
            fingerprint=request_fingerprint,
            name="evaluator request",
        )
        _validate_root_owned_ancestors(
            manifest["python"]["path"],
            name="evaluator Python",
        )
        if stderr:
            _fail("isolated evaluator wrote to stderr")
        if return_code != 0:
            _fail(f"isolated evaluator exited with status {return_code}")
        report = _validate_evaluator_report(
            stdout,
            request=request,
            program=program,
        )
        report_raw = _canonical_json(report, allow_float=True)
        final_release, final_sources = _authenticate_release(
            repository_root=repository,
            expected_release=authenticated_release,
        )
        if final_release != authenticated_release or final_sources != release_sources:
            _fail("authenticated release changed while the evaluator ran")
        if (
            output_parent_fd is None
            or output_parent_fingerprint is None
            or _metadata_fingerprint(os.fstat(output_parent_fd)) != output_parent_fingerprint
        ):
            _fail("evaluator report parent changed while the evaluator ran")
        _check_named_open_identity(
            posixpath.dirname(output),
            descriptor=output_parent_fd,
            fingerprint=output_parent_fingerprint,
            name="evaluator report parent",
        )
        _publish_report_at(
            output_parent_fd,
            output_leaf,
            report_raw,
            nonce=request["nonce"],
        )
        return {
            "schema": EXECUTION_REPORT_SCHEMA,
            "status": "authenticated",
            "child_exit_code": 0,
            "evaluator_program": dict(program),
            "request": {
                "path": _canonical_absolute(request_path, name="evaluator request path"),
                "sha256": _digest(request_sha256, name="evaluator request SHA-256"),
            },
            "report": {
                "path": output,
                "sha256": hashlib.sha256(report_raw).hexdigest(),
            },
        }
    finally:
        if process is not None:
            _terminate_process(process)
        for descriptor, _, _, _ in retained:
            os.close(descriptor)
        for descriptor, _, _, _, _ in directories:
            os.close(descriptor)
        if output_parent_fd is not None:
            os.close(output_parent_fd)
        if request_fd is not None:
            os.close(request_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", allow_abbrev=False)
    verify = commands.add_parser("verify-run", allow_abbrev=False)

    for command in (stage, verify):
        command.add_argument("--repository-root", required=True)
        command.add_argument("--expected-release-commit", required=True)
        command.add_argument("--expected-release-tree", required=True)
        command.add_argument("--expected-release-signer", required=True)
        command.add_argument("--expected-release-key-fingerprint", required=True)
        command.add_argument("--expected-stager-sha256", required=True)

    stage.add_argument("--output-root", required=True)
    stage.add_argument("--python", required=True)
    stage.add_argument("--expected-python-sha256", required=True)
    stage.add_argument("--evaluator", required=True)
    stage.add_argument("--expected-evaluator-sha256", required=True)
    stage.add_argument("--producer-root", required=True)
    for name in COMPANION_SOURCES:
        stage.add_argument(
            f"--expected-{name.replace('_', '-')}-sha256",
            dest=f"expected_{name}_sha256",
            required=True,
        )
    verify.add_argument("--program-manifest", required=True)
    verify.add_argument("--expected-program-manifest-sha256", required=True)
    verify.add_argument("--request", required=True)
    verify.add_argument("--expected-request-sha256", required=True)
    verify.add_argument("--report", required=True)
    return parser


def _release_from_arguments(arguments: argparse.Namespace) -> dict[str, str]:
    return {
        "commit": arguments.expected_release_commit,
        "tree": arguments.expected_release_tree,
        "signer": arguments.expected_release_signer,
        "key_fingerprint": arguments.expected_release_key_fingerprint,
        "stager_sha256": arguments.expected_stager_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        release = _release_from_arguments(arguments)
        if arguments.command == "stage":
            report = stage_evaluator_program(
                repository_root=arguments.repository_root,
                release=release,
                output_root=arguments.output_root,
                python_path=arguments.python,
                python_sha256=arguments.expected_python_sha256,
                evaluator_path=arguments.evaluator,
                evaluator_sha256=arguments.expected_evaluator_sha256,
                producer_root=arguments.producer_root,
                companion_sha256={name: getattr(arguments, f"expected_{name}_sha256") for name in COMPANION_SOURCES},
            )
        elif arguments.command == "verify-run":
            report = verify_and_run_evaluator(
                repository_root=arguments.repository_root,
                release=release,
                program_manifest_path=arguments.program_manifest,
                program_manifest_sha256=arguments.expected_program_manifest_sha256,
                request_path=arguments.request,
                request_sha256=arguments.expected_request_sha256,
                report_path=arguments.report,
            )
        else:
            raise AssertionError("unreachable command")
    except (EvaluatorStageError, OSError) as error:
        print(f"strict replay V2 evaluator operation failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(_canonical_json(report) + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
