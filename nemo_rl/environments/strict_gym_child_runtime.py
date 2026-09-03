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

"""Process-bound runtime evidence for strict NeMo-Gym verifier children.

The ordinary :class:`nemo_gym.cli.env.RunHelper` launches each server through
an intermediate shell.  Inspecting a second process in the selected venv is
therefore useful preflight, but it does not prove which interpreter, packages,
or source the server that accepted ``/run`` or ``/verify`` actually used.

This module installs a sealed ``sitecustomize`` path for the duration of
``RunHelper.start``.  Python imports that hook in the actual SimpleAgent and
resource-server processes before it executes ``app.py``.  Each selected child
then publishes an immutable receipt.  Once RunHelper reports the servers live,
this module joins those receipts to RunHelper's selected config, wrapper PID,
the live child process, and the child-owned listening socket.

The hook is intentionally a standalone stdlib-only file.  It must run in the
per-server venv, not in the NeMo-RL actor venv.  The strict deployment mounts
the authenticated NeMo-RL and Gym snapshots read-only; this module additionally
requires an exclusive one-file bootstrap directory and exact pinned Gym paths
and source hashes.
"""

from __future__ import annotations

import hashlib
import ctypes
import json
import math
import os
import re
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STRICT_GYM_CHILD_SPEC_SCHEMA = "nemo-rl-strict-gym-child-spec-v1"
STRICT_GYM_CHILD_RECEIPT_SCHEMA = "nemo-rl-strict-gym-child-receipt-v1"
STRICT_GYM_CHILD_INDEX_SCHEMA = "nemo-rl-strict-gym-child-index-v1"
STRICT_GYM_SCORE_CALL_SCHEMA = "nemo-rl-strict-reasoning-score-call-v1"
STRICT_GYM_SCORE_CLOSED_SCHEMA = "nemo-rl-strict-reasoning-score-closed-v1"
STRICT_GYM_SCORE_CALL_INDEX_SCHEMA = "nemo-rl-strict-reasoning-score-call-index-v1"
STRICT_GYM_CHILD_HASH_DOMAIN = "sha256-canonical-ascii-json-no-lf-v1"
STRICT_GYM_SCORE_FINALIZER_SAFE = True

STRICT_GYM_GIT_COMMIT = "354babf7e3554fcd006807c86e80ef476aec9408"
STRICT_GYM_TREE = "f24e1ff729c3aed1957df382364c516097218fe0"
STRICT_GYM_ROOT = Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym")
STRICT_GYM_VENV_ROOT = Path("/opt/gym_venvs")
STRICT_RESULTS_DIRECTORY_NAME = "strict_gym_child_runtime"

# Scorer-only children execute with ``-I -S`` and therefore never execute these
# image-owned import hooks.  We nevertheless bind the exact Q-image inventory,
# including the two uv archive links and their resolved bytes.  An empty
# inventory is the sole safer normalization.  Main-scope children do not have
# the no-site guarantee and continue to reject every .pth file.
_SCORER_NO_SITE_PTH_FILES = {
    "__editable__.nemo_gym-0.5.0rc0.pth": {
        "kind": "symlink",
        "link_target": (
            "/root/.cache/uv/archive-v0/6y4mmc6Y7xD0Sgua/"
            "__editable__.nemo_gym-0.5.0rc0.pth"
        ),
        "size": 93,
        "sha256": "debb90ea383877803a356178d2fab9daafa3a1b86cae9845808f6cd99e1acd18",
    },
    "_virtualenv.pth": {
        "kind": "regular",
        "size": 18,
        "sha256": "69ac3d8f27e679c81b94ab30b3b56e9cd138219b1ba94a1fa3606d5a76a1433d",
    },
    "a1_coverage.pth": {
        "kind": "symlink",
        "link_target": "/root/.cache/uv/archive-v0/sf9_c48Qm-JpaoWQ/a1_coverage.pth",
        "size": 205,
        "sha256": "ef2ed06d19867ec669c09a804060666a9cd5e383af0a9d11aa2de79b77d448e8",
    },
}

_SPEC_ENV = "NRL_STRICT_GYM_CHILD_SPEC_PATH"
_SPEC_SHA_ENV = "NRL_STRICT_GYM_CHILD_SPEC_SHA256"
_BOOTSTRAP_ENV = "NRL_STRICT_GYM_CHILD_BOOTSTRAP_ROOT"
_BOOTSTRAP_SHA_ENV = "NRL_STRICT_GYM_CHILD_BOOTSTRAP_SHA256"
_DIRECT_RUNNER_ENV = "NRL_STRICT_GYM_DIRECT_RUNNER"

_PRE_PYTHON_INJECTION_ENV_NAMES = {
    "BASH_ENV",
    "BASHOPTS",
    "BASH_XTRACEFD",
    "CDPATH",
    "ENV",
    "GCONV_PATH",
    "GLOBIGNORE",
    "IFS",
    "LOCPATH",
    "NEMO_GYM_EXTRA_ROOTS",
    "NLSPATH",
    "POSIXLY_CORRECT",
    "PROMPT_COMMAND",
    "PS4",
    "SHELLOPTS",
}
_PRE_PYTHON_INJECTION_ENV_PREFIXES = (
    "BASH",
    "DYLD_",
    "LD_",
    "PYTHON",
)

_GYM_SOURCE_PINS = {
    "nemo_gym/__init__.py": (
        "a7d495b7057874ea7c6ea849623d789a1d2939c7a1aac47c0aeabaf849490188"
    ),
    "nemo_gym/global_config.py": (
        "5e8e7457e6c3b9ae2cc5b124fb3b42f1920ba6c17707519da7b1d560e0a1ea70"
    ),
    "nemo_gym/cli/setup_command.py": (
        "6e976fe8491e8ddc9770dc553f93ef46a5545760cf9f3ce5396369c0e9945a71"
    ),
    "nemo_gym/cli/env.py": (
        "e6f468072e6b6627624dbc9270f60ec792ff0fa2aed681f9ce338b714880188a"
    ),
}

_SIMPLE_AGENT = {
    "role": "simple_agent",
    "server_type": "responses_api_agents",
    "server_name": "simple_agent",
    "component_relative": "responses_api_agents/simple_agent",
    "entrypoint": "app.py",
    "source_relative": "responses_api_agents/simple_agent/app.py",
    "source_sha256": (
        "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a"
    ),
    "distribution_versions": {
        "nemo-gym": "0.5.0rc0",
        "openai": "2.6.1",
        "pydantic": "2.13.4",
        "pydantic-core": "2.46.4",
    },
    "module_versions": {"nemo_gym": "0.5.1"},
    "scorer": None,
}

_RESOURCE_TARGETS = {
    "reasoning_gym": {
        "config_path": "reasoning_gym",
        "role": "resource",
        "server_type": "resources_servers",
        "server_name": "reasoning_gym",
        "component_relative": "resources_servers/reasoning_gym",
        "entrypoint": "app.py",
        "source_relative": "resources_servers/reasoning_gym/app.py",
        "source_sha256": (
            "3a35c5d27392dae05499ceefac04e9c32ad963b51a54d77bb470ee59b1fe3127"
        ),
        "config_relative": (
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml"
        ),
        "config_sha256": (
            "bdbb459a4a920bc47cf84b1d7dc30aeaa9be35cf0dfac09c77879e45b62a52ab"
        ),
        "requirements_relative": "resources_servers/reasoning_gym/requirements.txt",
        "requirements_sha256": (
            "b00b45db433d797d8a5c5c5602f24ab94d9d5620d83b4bef21fbee851287d411"
        ),
        "distribution_versions": {
            "nemo-gym": "0.5.0rc0",
            "openai": "2.6.1",
            "pydantic": "2.13.4",
            "pydantic-core": "2.46.4",
            "ray": "2.56.1",
            "reasoning-gym": "0.1.25",
        },
        # reasoning-gym 0.1.25 intentionally ships a stale internal version
        # literal.  Metadata and module literals are evidence, not an equality
        # invariant.
        "module_versions": {
            "nemo_gym": "0.5.1",
            "ray": "2.56.1",
            "reasoning_gym": "0.1.19",
        },
        "scorer": {
            "module": "reasoning_gym.logic.knights_knaves",
            "callable": (
                "reasoning_gym.logic.knights_knaves.KnightsKnavesDataset.score_answer"
            ),
            "origin_relative_to_purelib": ("reasoning_gym/logic/knights_knaves.py"),
            "sha256": (
                "8837a3c6dfc72bb40db168b82ad6b3da45a08a4000a006fc306368b77b622705"
            ),
            "resolver_module": "reasoning_gym.factory",
            "resolver_origin_relative_to_purelib": "reasoning_gym/factory.py",
            "resolver_sha256": (
                "fc651cc93205fafab926526a7d2c7a88c9dfa569af6e895135a1db48765e75bf"
            ),
            "module_origin_relative_to_purelib": "reasoning_gym/__init__.py",
            "module_sha256": (
                "ce1e4c7b2d0f61ea2021395ffb228cb3bdf4a5db21cae2e72882afb4cc4b64c6"
            ),
            "package_tree_hash_domain": "sha256-canonical-package-files-v1",
            "package_tree_sha256": (
                "17c69662d248b5fc6017f128a152579e209d7dda31a94edb8b9141737917442c"
            ),
            "package_file_count": 200,
            "package_total_bytes": 17871220,
            "wheel_sha256": (
                "7f17a3eddb13c015d7d4a755ed576a061df889faf9468bcc2cca334ebe9e0435"
            ),
        },
    },
    "citation": {
        "config_path": "citation_format",
        "role": "resource",
        "server_type": "resources_servers",
        "server_name": "format_verification",
        "component_relative": "resources_servers/format_verification",
        "entrypoint": "app.py",
        "source_relative": "resources_servers/format_verification/app.py",
        "source_sha256": (
            "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
        ),
        "config_relative": (
            "resources_servers/format_verification/configs/citation_format.yaml"
        ),
        "config_sha256": (
            "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
        ),
        "requirements_relative": (
            "resources_servers/format_verification/requirements.txt"
        ),
        "requirements_sha256": (
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        "distribution_versions": {
            "nemo-gym": "0.5.0rc0",
            "openai": "2.6.1",
            "pydantic": "2.13.4",
            "pydantic-core": "2.46.4",
        },
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
    },
    "freeform": {
        "config_path": "freeform_formatting",
        "role": "resource",
        "server_type": "resources_servers",
        "server_name": "format_verification",
        "component_relative": "resources_servers/format_verification",
        "entrypoint": "app.py",
        "source_relative": "resources_servers/format_verification/app.py",
        "source_sha256": (
            "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
        ),
        "config_relative": (
            "resources_servers/format_verification/configs/freeform_formatting.yaml"
        ),
        "config_sha256": (
            "92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa"
        ),
        "requirements_relative": (
            "resources_servers/format_verification/requirements.txt"
        ),
        "requirements_sha256": (
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        "distribution_versions": {
            "nemo-gym": "0.5.0rc0",
            "openai": "2.6.1",
            "pydantic": "2.13.4",
            "pydantic-core": "2.46.4",
        },
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
    },
}

_AGENT_CONFIG_PATHS = {
    "reasoning_gym": "reasoning_gym_simple_agent",
    "citation": "citation_format_simple_agent",
    "freeform": "freeform_formatting_simple_agent",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_JOB_ID_RE = re.compile(r"[1-9][0-9]{0,31}\Z")
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIR_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
_CREATE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_ascii_json(value: Any) -> bytes:
    """Encode the only JSON representation admitted by this evidence path."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare canonical-JSON values without Python's bool/int aliases."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stat_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_all(fd: int, *, maximum: int = 1 << 20) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(65536, maximum + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise ValueError("strict Gym child artifact exceeds size limit")


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"strict Gym source must not be a symlink: {path}")
    fd = os.open(path, _READ_FLAGS)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"strict Gym source must be one regular inode: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        path_info = path.lstat()
        if _stat_fingerprint(before) != _stat_fingerprint(after) or _stat_fingerprint(
            after
        ) != _stat_fingerprint(path_info):
            raise ValueError(f"strict Gym source changed while it was read: {path}")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _package_tree_identity(package_root: Path) -> tuple[Path, str, int, int]:
    """Recompute the pinned wheel package-tree identity without importing it."""
    resolved_root = package_root.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(
        resolved_root, followlinks=False
    ):
        if "__pycache__" in directory_names or any(
            name.endswith(".pyc") for name in file_names
        ):
            raise ValueError(
                "reasoning-gym package contains executable unpinned bytecode"
            )
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        for name in directory_names:
            if (base / name).is_symlink():
                raise ValueError(
                    "reasoning-gym package contains a nested directory symlink"
                )
        for name in file_names:
            path = base / name
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
            ):
                raise ValueError("reasoning-gym package contains an untrusted inode")
            relative = path.relative_to(resolved_root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "sha256": _sha256_regular_file(path),
                    "size": info.st_size,
                }
            )
            total_bytes += info.st_size
    entries.sort(key=lambda item: item["path"])
    return (
        resolved_root,
        _sha256_bytes(canonical_ascii_json(entries)),
        len(entries),
        total_bytes,
    )


def _require_exact_keys(value: Any, keys: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{name} has the wrong keyset: {actual!r}")
    return value


def _load_canonical_document(
    path: Path, *, expected_mode: int = 0o400
) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"strict Gym child artifact is a symlink: {path}")
    fd = os.open(path, _READ_FLAGS)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
        ):
            raise ValueError(f"strict Gym child artifact inode is invalid: {path}")
        payload = _read_all(fd)
        after = os.fstat(fd)
        path_info = path.lstat()
        if _stat_fingerprint(before) != _stat_fingerprint(after) or _stat_fingerprint(
            after
        ) != _stat_fingerprint(path_info):
            raise ValueError(f"strict Gym child artifact changed while read: {path}")
    finally:
        os.close(fd)
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"strict Gym child artifact is not strict JSON: {path}"
        ) from error
    if not isinstance(value, dict) or canonical_ascii_json(value) != payload:
        raise ValueError(f"strict Gym child artifact is not canonical JSON: {path}")
    return value, payload


def _publish_document(
    root_fd: int, filename: str, document: Mapping[str, Any]
) -> tuple[str, str]:
    if not filename or filename in {".", ".."} or "/" in filename:
        raise ValueError("strict Gym child artifact filename is invalid")
    payload = canonical_ascii_json(document)
    fd = os.open(filename, _CREATE_FLAGS, stat.S_IRUSR, dir_fd=root_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while publishing strict Gym child artifact")
            offset += written
        os.fchmod(fd, stat.S_IRUSR)
        os.fsync(fd)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o400
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size != len(payload)
        ):
            raise RuntimeError(
                "strict Gym child artifact publication validation failed"
            )
    finally:
        os.close(fd)
    os.fsync(root_fd)
    return filename, _sha256_bytes(payload)


def _require_private_canonical_directory(path: Path, *, name: str) -> Path:
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError(f"{name} must be absolute, canonical, and non-symlink")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{name} must be lexical-canonical")
    info = path.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
    ):
        raise ValueError(f"{name} must be an owned mode-0700 directory")
    return resolved


def _require_sealed_bootstrap_root() -> tuple[Path, str]:
    root = Path(__file__).resolve(strict=True).parent / "_strict_gym_child_bootstrap"
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("strict Gym child bootstrap root is not canonical")
    entries = sorted(item.name for item in root.iterdir())
    if entries != ["sitecustomize.py"]:
        raise ValueError(
            "strict Gym child bootstrap root must contain only sitecustomize.py; "
            f"found {entries!r}"
        )
    root_info = root.stat()
    source = root / "sitecustomize.py"
    source_info = source.lstat()
    root_mode = stat.S_IMODE(root_info.st_mode)
    source_mode = stat.S_IMODE(source_info.st_mode)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_mode not in {0o500, 0o555}
        or root_info.st_uid != os.geteuid()
        or not stat.S_ISREG(source_info.st_mode)
        or source_mode not in {0o400, 0o444}
        or source_info.st_uid != os.geteuid()
        or source_info.st_nlink != 1
    ):
        raise ValueError(
            "strict Gym child bootstrap must be owned and non-writable with "
            "directory mode 0500/0555 and source mode 0400/0444"
        )
    return root, _sha256_regular_file(source)


def _target_matrix(
    environment: str, gym_root: Path, *, scope: str = "main"
) -> list[dict[str, Any]]:
    if environment not in _RESOURCE_TARGETS:
        raise ValueError("strict Gym environment is not admitted")
    if scope not in {"main", "scorer-only"}:
        raise ValueError("strict Gym child attestation scope is not admitted")
    resource = dict(_RESOURCE_TARGETS[environment])
    if scope == "scorer-only" and environment == "reasoning_gym":
        resource["config_relative"] = (
            "resources_servers/reasoning_gym/configs/resources_only.yaml"
        )
        resource["config_sha256"] = (
            "e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68"
        )
    agent = dict(_SIMPLE_AGENT)
    agent.update(
        {
            "config_path": _AGENT_CONFIG_PATHS[environment],
            "config_relative": resource["config_relative"],
            "config_sha256": resource["config_sha256"],
            "requirements_relative": (
                "responses_api_agents/simple_agent/requirements.txt"
            ),
            # Runtime metadata is the effective pin for this skip-venv path;
            # retain the exact requirements bytes as source provenance too.
            "requirements_sha256": (
                "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
            ),
            "resource_config_path": resource["config_path"],
        }
    )
    resource["resource_config_path"] = None

    targets: list[dict[str, Any]] = []
    templates = (resource, agent) if scope == "main" else (resource,)
    for template in templates:
        target = dict(template)
        component_dir = gym_root / target["component_relative"]
        venv = (
            STRICT_GYM_VENV_ROOT
            / target["server_type"]
            / target["server_name"]
            / ".venv"
        )
        target["component_dir"] = str(component_dir)
        target["source_path"] = str(gym_root / target["source_relative"])
        target["config_path_source"] = str(gym_root / target["config_relative"])
        target["requirements_path"] = str(gym_root / target["requirements_relative"])
        target["venv"] = str(venv)
        target["interpreter"] = str(venv / "bin" / "python")
        target["receipt_filename"] = f"{target['role']}.json"
        targets.append(target)
    return targets


def _validate_purelib_pth_inventory(purelib: Path, *, scope: str) -> None:
    pth_files = sorted(purelib.glob("*.pth"), key=lambda item: item.name)
    purelib_info = purelib.stat()
    if (
        not stat.S_ISDIR(purelib_info.st_mode)
        or stat.S_IMODE(purelib_info.st_mode) != 0o755
        or purelib_info.st_uid != os.geteuid()
        or purelib_info.st_gid != os.getegid()
    ):
        raise ValueError("selected Gym scorer-only purelib identity differs")
    if not pth_files:
        return
    if scope != "scorer-only" or [item.name for item in pth_files] != list(
        _SCORER_NO_SITE_PTH_FILES
    ):
        raise ValueError(
            "selected Gym child venv has untrusted pre-sitecustomize .pth code"
        )
    for pth_file in pth_files:
        identity = _SCORER_NO_SITE_PTH_FILES[pth_file.name]
        before_link = pth_file.lstat()
        if identity["kind"] == "regular":
            if (
                not stat.S_ISREG(before_link.st_mode)
                # The strict HSG wrapper enters Pyxis under umask 0077.  Enroot
                # materializes regular image files as 0600 in that context;
                # symlink modes remain 0777.  Bind the effective runtime
                # identity rather than the squashfs listing's pre-umask 0644.
                or stat.S_IMODE(before_link.st_mode) != 0o600
                or before_link.st_uid != os.geteuid()
                or before_link.st_gid != os.getegid()
                or before_link.st_nlink != 1
                or before_link.st_size != identity["size"]
                or pth_file.resolve(strict=True) != pth_file
                or _sha256_regular_file(pth_file) != identity["sha256"]
                or _stat_fingerprint(pth_file.lstat()) != _stat_fingerprint(before_link)
            ):
                raise ValueError("selected Gym scorer-only venv .pth identity differs")
            continue

        if not stat.S_ISLNK(before_link.st_mode):
            raise ValueError("selected Gym scorer-only venv .pth identity differs")
        link_target = os.readlink(pth_file)
        expected_target = Path(identity["link_target"])
        if (
            type(link_target) is not str
            or "\x00" in link_target
            or not Path(link_target).is_absolute()
            or ".." in Path(link_target).parts
            or link_target != identity["link_target"]
            or (sys.platform == "linux" and stat.S_IMODE(before_link.st_mode) != 0o777)
            or before_link.st_uid != os.geteuid()
            or before_link.st_gid != os.getegid()
            or before_link.st_nlink != 1
            or before_link.st_size != len(os.fsencode(link_target))
        ):
            raise ValueError("selected Gym scorer-only venv .pth identity differs")
        resolved_target = pth_file.resolve(strict=True)
        before_target = resolved_target.lstat()
        if (
            resolved_target != expected_target
            or not stat.S_ISREG(before_target.st_mode)
            or stat.S_IMODE(before_target.st_mode) != 0o600
            or before_target.st_uid != os.geteuid()
            or before_target.st_gid != os.getegid()
            or before_target.st_nlink != 1
            or before_target.st_size != identity["size"]
            or _sha256_regular_file(resolved_target) != identity["sha256"]
            or _stat_fingerprint(resolved_target.lstat())
            != _stat_fingerprint(before_target)
            or _stat_fingerprint(pth_file.lstat()) != _stat_fingerprint(before_link)
            or os.readlink(pth_file) != link_target
        ):
            raise ValueError("selected Gym scorer-only venv .pth identity differs")


def _validate_pinned_gym_root(
    gym_root: Path, targets: list[dict[str, Any]], *, scope: str
) -> None:
    if (
        gym_root != STRICT_GYM_ROOT
        or gym_root.is_symlink()
        or gym_root.resolve(strict=True) != gym_root
    ):
        raise ValueError(
            "strict Gym root must be the authenticated deployment mount "
            f"{STRICT_GYM_ROOT}"
        )
    for relative, expected in _GYM_SOURCE_PINS.items():
        actual = _sha256_regular_file(gym_root / relative)
        if actual != expected:
            raise ValueError(f"pinned Gym source hash differs: {relative}")

    from nemo_gym import PARENT_DIR
    from nemo_gym.cli.setup_command import get_venv_path
    from omegaconf import DictConfig

    if Path(PARENT_DIR).resolve(strict=True) != gym_root:
        raise ValueError(
            "imported nemo_gym.PARENT_DIR is not the authenticated Gym root"
        )
    probe_config = DictConfig({"uv_venv_dir": str(STRICT_GYM_VENV_ROOT)})
    for target in targets:
        component = Path(target["component_dir"])
        if component.is_symlink() or component.resolve(strict=True) != component:
            raise ValueError(
                "selected Gym component is not below the authenticated root"
            )
        try:
            component.relative_to(gym_root)
        except ValueError as error:
            raise ValueError(
                "selected Gym component escaped the authenticated root"
            ) from error
        if (component / "sitecustomize.py").exists() or (
            component / "sitecustomize"
        ).exists():
            raise ValueError("selected Gym component shadows strict sitecustomize")
        for path_key, digest_key in (
            ("source_path", "source_sha256"),
            ("config_path_source", "config_sha256"),
            ("requirements_path", "requirements_sha256"),
        ):
            if _sha256_regular_file(Path(target[path_key])) != target[digest_key]:
                raise ValueError(f"selected Gym target differs: {target[path_key]}")
        derived_venv = Path(get_venv_path(component, probe_config))
        if derived_venv != Path(target["venv"]):
            raise ValueError(
                "pinned Gym get_venv_path selected an unexpected child venv"
            )
        interpreter = Path(target["interpreter"])
        resolved_interpreter = interpreter.resolve(strict=True)
        if not stat.S_ISREG(resolved_interpreter.stat().st_mode) or not os.access(
            interpreter, os.X_OK
        ):
            raise ValueError("selected Gym child interpreter is not executable")
        purelib = Path(target["venv"]) / "lib" / "python3.13" / "site-packages"
        if purelib.resolve(strict=True) != purelib:
            raise ValueError("selected Gym child venv purelib is not canonical")
        _validate_purelib_pth_inventory(purelib, scope=scope)


def _strict_environment_value(
    name: str, *, pattern: re.Pattern[str] | None = None
) -> str:
    value = os.environ.get(name)
    if (
        not isinstance(value, str)
        or not value
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise ValueError(f"strict Gym child runtime requires valid {name}")
    return value


def _build_spec(
    *,
    environment: str,
    scope: str,
    pair_id: str,
    job_id: str,
    results_dir: Path,
    receipt_root: Path,
    bootstrap_root: Path,
    bootstrap_sha256: str,
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": STRICT_GYM_CHILD_SPEC_SCHEMA,
        "hash_domain": STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": environment,
        "scope": scope,
        "pair_id": pair_id,
        "job_id": job_id,
        "gym": {
            "git_commit": STRICT_GYM_GIT_COMMIT,
            "tree": STRICT_GYM_TREE,
            "root": str(STRICT_GYM_ROOT),
            "venv_root": str(STRICT_GYM_VENV_ROOT),
            "sources": dict(_GYM_SOURCE_PINS),
        },
        "results_dir": str(results_dir),
        "receipt_root": str(receipt_root),
        "bootstrap": {
            "root": str(bootstrap_root),
            "filename": "sitecustomize.py",
            "sha256": bootstrap_sha256,
        },
        "targets": targets,
    }


def _process_stat(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, int]:
    """Return ``(ppid, start_ticks)`` from one Linux proc stat record."""
    payload = (proc_root / str(pid) / "stat").read_bytes()
    closing = payload.rfind(b") ")
    if closing < 0:
        raise ValueError("process stat has no comm terminator")
    fields = payload[closing + 2 :].split()
    if len(fields) <= 19:
        raise ValueError("process stat is truncated")
    return int(fields[1]), int(fields[19])


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _process_argv(pid: int) -> list[str]:
    payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    if not payload.endswith(b"\0") or len(payload) > 65536:
        raise ValueError("process cmdline is malformed")
    return [part.decode("utf-8", "strict") for part in payload[:-1].split(b"\0")]


def _process_is_descendant(pid: int, ancestor: int, *, maximum_depth: int = 32) -> bool:
    current = pid
    seen: set[int] = set()
    for _ in range(maximum_depth):
        if current == ancestor:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        current, _ = _process_stat(current)
    return False


def _proc_status_tgid(path: Path) -> int:
    payload = path.read_bytes()
    if len(payload) > 1 << 20:
        raise ValueError("strict Gym child proc status is too large")
    values = [
        line.split()[1] for line in payload.splitlines() if line.startswith(b"Tgid:")
    ]
    if len(values) != 1:
        raise ValueError("strict Gym child proc status has no exact TGID")
    try:
        tgid = int(values[0])
    except ValueError as error:
        raise ValueError("strict Gym child proc TGID is malformed") from error
    if tgid <= 1:
        raise ValueError("strict Gym child proc TGID is invalid")
    return tgid


def _process_descendant_identities(
    pid: int, *, proc_root: Path = Path("/proc")
) -> list[tuple[int, int]]:
    """Return child identities found across every task in each process."""
    pending = [pid]
    seen = {pid}
    descendants: list[tuple[int, int]] = []
    while pending:
        parent = pending.pop()
        task_root = proc_root / str(parent) / "task"

        def task_ids() -> list[int]:
            entries = list(task_root.iterdir())
            if len(entries) > 4096:
                raise ValueError("strict Gym child process has too many tasks")
            if any(
                not item.name.isascii() or not item.name.isdecimal() for item in entries
            ):
                raise ValueError("strict Gym child task directory is malformed")
            return sorted(int(item.name) for item in entries)

        tasks_before = task_ids()
        if parent not in tasks_before or any(task <= 1 for task in tasks_before):
            raise ValueError("strict Gym child task membership is invalid")
        children: set[int] = set()
        for task in tasks_before:
            task_dir = task_root / str(task)
            if _proc_status_tgid(task_dir / "status") != parent:
                raise RuntimeError("strict Gym child task membership changed")
            payload = (task_dir / "children").read_bytes()
            if len(payload) > 65536:
                raise ValueError("strict Gym child process-tree record is too large")
            try:
                children.update(int(item) for item in payload.split())
            except ValueError as error:
                raise ValueError(
                    "strict Gym child process-tree record is malformed"
                ) from error
        if task_ids() != tasks_before:
            raise RuntimeError("strict Gym child task set changed during inspection")
        for child in sorted(children):
            if child <= 1 or child in seen:
                raise ValueError("strict Gym child process tree is invalid")
            if _proc_status_tgid(proc_root / str(child) / "status") != child:
                raise RuntimeError("strict Gym child descendant TGID differs")
            observed_parent, start_ticks = _process_stat(child, proc_root=proc_root)
            if observed_parent != parent or start_ticks <= 0:
                raise RuntimeError(
                    "strict Gym child process tree changed during inspection"
                )
            seen.add(child)
            pending.append(child)
            descendants.append((child, start_ticks))
    return sorted(descendants)


def _listening_socket_inodes(pid: int, host: str, port: int) -> list[int]:
    if host != "127.0.0.1":
        raise ValueError("strict Gym child listener host is not pinned loopback")
    owned: set[int] = set()
    for fd_path in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd_path)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match:
            owned.add(int(match.group(1)))

    listeners: set[int] = set()
    for table_name, expected_address in (("tcp", "0100007F"),):
        table = Path(f"/proc/{pid}/net/{table_name}")
        if not table.exists():
            continue
        lines = table.read_text(encoding="ascii").splitlines()
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_address, encoded_port = fields[1].rsplit(":", 1)
                local_port = int(encoded_port, 16)
                inode = int(fields[9])
            except (IndexError, ValueError):
                continue
            if (
                local_address == expected_address
                and local_port == port
                and inode in owned
            ):
                listeners.add(inode)
    return sorted(listeners)


_LINUX_PIDFD_SYSCALL_NUMBERS = {
    "aarch64": {"pidfd_send_signal": 424, "pidfd_open": 434},
    "x86_64": {"pidfd_send_signal": 424, "pidfd_open": 434},
}


def _linux_pidfd_syscall_number(name: str) -> int:
    if sys.platform != "linux":
        raise RuntimeError("strict Gym child quiescence requires Linux pidfd support")
    architecture = os.uname().machine
    numbers = _LINUX_PIDFD_SYSCALL_NUMBERS.get(architecture)
    if numbers is None or name not in numbers:
        raise RuntimeError(
            "strict Gym child quiescence has no pinned pidfd syscall ABI for "
            f"{architecture!r}"
        )
    return numbers[name]


def _linux_pidfd_open_syscall(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(
        ctypes.c_long(_linux_pidfd_syscall_number("pidfd_open")),
        ctypes.c_int(pid),
        ctypes.c_uint(0),
    )
    if result == -1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if result < 0:
        raise RuntimeError("strict Gym child pidfd_open returned an invalid descriptor")
    return int(result)


def _linux_pidfd_send_signal_syscall(pidfd: int, signal_number: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(
        ctypes.c_long(_linux_pidfd_syscall_number("pidfd_send_signal")),
        ctypes.c_int(pidfd),
        ctypes.c_int(signal_number),
        ctypes.c_void_p(),
        ctypes.c_uint(0),
    )
    if result == -1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if result != 0:
        raise RuntimeError("strict Gym child pidfd_send_signal returned invalid status")


def _pidfd_open(pid: int) -> int:
    pidfd_open = getattr(os, "pidfd_open", None)
    if callable(pidfd_open):
        return pidfd_open(pid, 0)
    return _linux_pidfd_open_syscall(pid)


def _pidfd_send_signal(pidfd: int, signal_number: int) -> None:
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_send_signal):
        pidfd_send_signal(pidfd, signal_number, None, 0)
        return
    _linux_pidfd_send_signal_syscall(pidfd, signal_number)


def _terminate_authenticated_process(
    pid: int,
    start_ticks: int,
    *,
    interrupt_timeout_ms: int = 5_000,
    terminate_timeout_ms: int = 5_000,
    kill_timeout_ms: int = 5_000,
) -> str:
    """Stop the exact authenticated Linux process without a PID-reuse race."""
    if (
        type(pid) is not int
        or pid <= 1
        or type(start_ticks) is not int
        or start_ticks <= 0
    ):
        raise ValueError("strict Gym child process identity is invalid")
    if type(interrupt_timeout_ms) is not int or interrupt_timeout_ms <= 0:
        raise ValueError("strict Gym child interrupt timeout is invalid")
    if type(terminate_timeout_ms) is not int or terminate_timeout_ms <= 0:
        raise ValueError("strict Gym child terminate timeout is invalid")
    if type(kill_timeout_ms) is not int or kill_timeout_ms <= 0:
        raise ValueError("strict Gym child kill timeout is invalid")
    pidfd = _pidfd_open(pid)
    try:
        _, observed_start = _process_stat(pid)
        if observed_start != start_ticks:
            raise RuntimeError("strict Gym child PID changed before pidfd termination")
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)

        _pidfd_send_signal(pidfd, signal.SIGINT)
        if poller.poll(interrupt_timeout_ms):
            return "SIGINT"

        # The pidfd still names the original process. Rechecking the proc
        # start time makes a surprising procfs view or early PID reuse fatal
        # before each escalation.
        _, observed_start = _process_stat(pid)
        if observed_start != start_ticks:
            raise RuntimeError("strict Gym child PID changed during termination")
        _pidfd_send_signal(pidfd, signal.SIGTERM)
        if poller.poll(terminate_timeout_ms):
            return "SIGTERM"

        _, observed_start = _process_stat(pid)
        if observed_start != start_ticks:
            raise RuntimeError("strict Gym child PID changed during termination")
        _pidfd_send_signal(pidfd, signal.SIGKILL)
        if not poller.poll(kill_timeout_ms):
            raise RuntimeError(
                "strict Gym child did not exit after authenticated SIGKILL"
            )
        return "SIGKILL"
    finally:
        os.close(pidfd)


def _validate_receipt(
    document: Any,
    *,
    spec: Mapping[str, Any],
    target: Mapping[str, Any],
    instance: Any,
    wrapper_pid: int,
) -> dict[str, Any]:
    keys = {
        "schema",
        "hash_domain",
        "environment",
        "pair_id",
        "job_id",
        "stage",
        "spec_sha256",
        "target",
        "server",
        "process",
        "distribution_versions",
        "module_versions",
        "scorer",
    }
    receipt = _require_exact_keys(document, keys, name="strict child receipt")
    if (
        receipt["schema"] != STRICT_GYM_CHILD_RECEIPT_SCHEMA
        or receipt["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
    ):
        raise ValueError("strict Gym child receipt schema mismatch")
    for name in ("environment", "pair_id", "job_id"):
        if receipt[name] != spec[name]:
            raise ValueError(f"strict Gym child receipt {name} mismatch")
    expected_stage = (
        "isolated-runner-pre-entrypoint"
        if spec["scope"] == "scorer-only"
        else "sitecustomize-pre-entrypoint"
    )
    if receipt["stage"] != expected_stage:
        raise ValueError("strict Gym child receipt stage mismatch")
    expected_spec_sha = _sha256_bytes(canonical_ascii_json(spec))
    if receipt["spec_sha256"] != expected_spec_sha:
        raise ValueError("strict Gym child receipt spec hash mismatch")

    target_record = _require_exact_keys(
        receipt["target"],
        {
            "role",
            "config_path",
            "server_type",
            "server_name",
            "component_dir",
            "entrypoint",
            "source_path",
            "source_sha256",
            "config_path_source",
            "config_sha256",
            "requirements_path",
            "requirements_sha256",
            "venv",
            "interpreter",
        },
        name="strict child receipt target",
    )
    for name in target_record:
        if target_record[name] != target[name]:
            raise ValueError(f"strict Gym child receipt target {name} mismatch")

    server = _require_exact_keys(
        receipt["server"],
        {
            "config_path",
            "server_type",
            "server_name",
            "entrypoint",
            "host",
            "port",
            "num_workers",
        },
        name="strict child receipt server",
    )
    for receipt_key, instance_key in (
        ("config_path", "config_path"),
        ("server_type", "server_type"),
        ("server_name", "name"),
        ("entrypoint", "entrypoint"),
        ("host", "host"),
        ("port", "port"),
    ):
        if server[receipt_key] != getattr(instance, instance_key, None):
            raise ValueError(f"strict Gym child RunHelper {receipt_key} mismatch")
    if type(server["port"]) is not int or not 5000 <= server["port"] <= 5999:
        raise ValueError("strict Gym child server port is outside the pinned range")
    if server["host"] != "127.0.0.1":
        raise ValueError("strict Gym child server host is not pinned loopback")
    if server["num_workers"] is not None and (
        type(server["num_workers"]) is not int or server["num_workers"] != 1
    ):
        raise ValueError("strict Gym child attestation requires one uvicorn process")

    process = _require_exact_keys(
        receipt["process"],
        {
            "pid",
            "ppid",
            "uid",
            "gid",
            "cwd",
            "sys_executable",
            "sys_prefix",
            "sys_base_prefix",
            "proc_exe",
            "sys_argv",
            "proc_argv",
            "start_ticks",
            "boot_id",
            "hostname",
        },
        name="strict child receipt process",
    )
    pid = process["pid"]
    if type(pid) is not int or pid <= 1:
        raise ValueError("strict Gym child PID is invalid")
    if type(process["ppid"]) is not int or process["ppid"] <= 0:
        raise ValueError("strict Gym child PPID is invalid")
    if type(process["start_ticks"]) is not int or process["start_ticks"] <= 0:
        raise ValueError("strict Gym child start ticks are invalid")
    if type(process["uid"]) is not int or process["uid"] != os.geteuid():
        raise ValueError("strict Gym child UID differs from the attestor")
    if type(process["gid"]) is not int or process["gid"] != os.getegid():
        raise ValueError("strict Gym child GID differs from the attestor")
    if (
        type(process["boot_id"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            process["boot_id"],
        )
        is None
        or process["boot_id"] != _boot_id()
    ):
        raise ValueError("strict Gym child boot identity is invalid")
    if (
        type(process["hostname"]) is not str
        or process["hostname"] != socket.gethostname()
    ):
        raise ValueError("strict Gym child hostname differs from the attestor")
    current_ppid, current_start_ticks = _process_stat(pid)
    if current_start_ticks != process["start_ticks"]:
        raise ValueError("strict Gym child PID was reused")
    if current_ppid != process["ppid"]:
        raise ValueError("strict Gym child parent changed before validation")
    if not _process_is_descendant(pid, wrapper_pid):
        raise ValueError("strict Gym child is not descended from RunHelper process")
    if spec["scope"] == "scorer-only" and pid != wrapper_pid:
        raise ValueError("isolated scorer PID differs from RunHelper Popen PID")
    if _process_argv(pid) != process["proc_argv"]:
        raise ValueError("strict Gym child argv changed")
    if os.readlink(f"/proc/{pid}/exe") != process["proc_exe"]:
        raise ValueError("strict Gym child executable changed")
    if os.readlink(f"/proc/{pid}/cwd") != process["cwd"]:
        raise ValueError("strict Gym child cwd changed")
    if process["cwd"] != target["component_dir"]:
        raise ValueError("strict Gym child cwd is not the authenticated component")
    if process["sys_executable"] != target["interpreter"]:
        raise ValueError(
            "strict Gym child interpreter differs from pinned get_venv_path"
        )
    if process["sys_prefix"] != target["venv"]:
        raise ValueError("strict Gym child sys.prefix differs from selected venv")
    if process["sys_base_prefix"] == process["sys_prefix"]:
        raise ValueError("strict Gym child did not execute in a venv")
    if spec["scope"] == "scorer-only":
        bootstrap_path = str(
            Path(spec["bootstrap"]["root"]) / spec["bootstrap"]["filename"]
        )
        expected_sys_argv = [bootstrap_path, target["source_path"]]
        expected_proc_argv = [
            target["interpreter"],
            "-I",
            "-S",
            "-B",
            bootstrap_path,
            target["source_path"],
        ]
    else:
        expected_sys_argv = [target["entrypoint"]]
        expected_proc_argv = ["python", target["entrypoint"]]
    if (
        process["sys_argv"] != expected_sys_argv
        or process["proc_argv"] != expected_proc_argv
    ):
        raise ValueError(
            "strict Gym child entrypoint argv differs from pinned RunHelper"
        )
    if _process_descendant_identities(pid):
        raise ValueError("strict Gym child unexpectedly owns descendant processes")

    if receipt["distribution_versions"] != target["distribution_versions"]:
        raise ValueError(
            "strict Gym child distribution metadata differs from pinned runtime"
        )
    if receipt["module_versions"] != target["module_versions"]:
        raise ValueError("strict Gym child module versions differ from pinned runtime")
    expected_scorer = target["scorer"]
    observed_scorer = receipt["scorer"]
    if expected_scorer is None:
        if observed_scorer is not None:
            raise ValueError("non-reasoning child unexpectedly reported a scorer")
    else:
        dynamic_scorer_keys = {
            "package_root",
            "package_resolved_root",
            "module_origin",
            "module_resolved_origin",
            "resolver_origin",
            "resolver_resolved_origin",
            "origin",
            "resolved_origin",
        }
        scorer = _require_exact_keys(
            observed_scorer,
            set(expected_scorer) | dynamic_scorer_keys,
            name="strict child scorer",
        )
        for name, value in expected_scorer.items():
            if scorer[name] != value:
                raise ValueError(f"strict Gym child scorer {name} mismatch")
        purelib = Path(target["venv"]) / "lib" / "python3.13" / "site-packages"
        package_root = purelib / "reasoning_gym"
        resolved_package, tree_sha256, file_count, total_bytes = _package_tree_identity(
            package_root
        )
        expected_module_origin = (
            purelib / expected_scorer["module_origin_relative_to_purelib"]
        )
        expected_resolver_origin = (
            purelib / expected_scorer["resolver_origin_relative_to_purelib"]
        )
        expected_origin = purelib / expected_scorer["origin_relative_to_purelib"]
        if (
            scorer["package_root"] != str(package_root)
            or scorer["package_resolved_root"] != str(resolved_package)
            or scorer["module_origin"] != str(expected_module_origin)
            or scorer["module_resolved_origin"]
            != str(expected_module_origin.resolve(strict=True))
            or scorer["resolver_origin"] != str(expected_resolver_origin)
            or scorer["resolver_resolved_origin"]
            != str(expected_resolver_origin.resolve(strict=True))
            or scorer["origin"] != str(expected_origin)
            or scorer["resolved_origin"] != str(expected_origin.resolve(strict=True))
            or _sha256_regular_file(expected_module_origin)
            != expected_scorer["module_sha256"]
            or _sha256_regular_file(expected_resolver_origin)
            != expected_scorer["resolver_sha256"]
            or _sha256_regular_file(expected_origin) != expected_scorer["sha256"]
            or (tree_sha256, file_count, total_bytes)
            != (
                expected_scorer["package_tree_sha256"],
                expected_scorer["package_file_count"],
                expected_scorer["package_total_bytes"],
            )
        ):
            raise ValueError(
                "strict Gym child scorer package origins/tree differ from pinned runtime"
            )

    listener_inodes = _listening_socket_inodes(pid, server["host"], server["port"])
    if not listener_inodes:
        raise ValueError("strict Gym child does not own its selected listening port")
    return {
        "pid": pid,
        "start_ticks": process["start_ticks"],
        "wrapper_pid": wrapper_pid,
        "host": server["host"],
        "port": server["port"],
        "listener_socket_inodes": listener_inodes,
    }


def reasoning_score_call_expectation(
    *, task_name: str, answer: Any, entry: Mapping[str, Any], float_result: float
) -> dict[str, Any]:
    """Build the exact input/result binding consumed by the K4 finalizer."""
    if type(task_name) is not str or task_name != "knights_knaves":
        raise ValueError("strict reasoning replay task must be knights_knaves")
    if not isinstance(entry, dict):
        raise TypeError("strict reasoning replay entry must be an exact dict")
    if (
        type(float_result) is not float
        or not math.isfinite(float_result)
        or not 0.0 <= float_result <= 1.0
        or (float_result == 0.0 and math.copysign(1.0, float_result) < 0.0)
    ):
        raise ValueError(
            "strict reasoning replay reward must be a finite float in [0,1]"
        )
    return {
        "task_name": task_name,
        "answer_sha256": _sha256_bytes(canonical_ascii_json(answer)),
        "entry_sha256": _sha256_bytes(canonical_ascii_json(entry)),
        "float_result": float_result,
    }


def _validate_score_call(
    document: Any,
    *,
    spec: Mapping[str, Any],
    sequence: int,
    expected: Mapping[str, Any],
    process: Mapping[str, Any],
) -> None:
    call = _require_exact_keys(
        document,
        {
            "schema",
            "hash_domain",
            "environment",
            "pair_id",
            "job_id",
            "spec_sha256",
            "process",
            "sequence",
            "task_name",
            "input",
            "outcome",
        },
        name=f"reasoning score call {sequence}",
    )
    if (
        call["schema"] != STRICT_GYM_SCORE_CALL_SCHEMA
        or call["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or call["environment"] != spec["environment"]
        or call["pair_id"] != spec["pair_id"]
        or call["job_id"] != spec["job_id"]
        or call["spec_sha256"] != _sha256_bytes(canonical_ascii_json(spec))
        or type(call["sequence"]) is not int
        or call["sequence"] != sequence
    ):
        raise ValueError(f"reasoning score call {sequence} identity mismatch")
    call_process = _require_exact_keys(
        call["process"], {"pid", "start_ticks"}, name="reasoning score process"
    )
    if (
        type(call_process["pid"]) is not int
        or type(call_process["start_ticks"]) is not int
        or call_process
        != {
            "pid": process["pid"],
            "start_ticks": process["start_ticks"],
        }
    ):
        raise ValueError(f"reasoning score call {sequence} process mismatch")
    if call["task_name"] != expected["task_name"]:
        raise ValueError(f"reasoning score call {sequence} task mismatch")
    call_input = _require_exact_keys(
        call["input"],
        {"answer_sha256", "entry_sha256"},
        name="reasoning score input",
    )
    if call_input != {
        "answer_sha256": expected["answer_sha256"],
        "entry_sha256": expected["entry_sha256"],
    }:
        raise ValueError(f"reasoning score call {sequence} input mismatch")
    outcome = _require_exact_keys(
        call["outcome"],
        {"kind", "float_result"},
        name="reasoning score outcome",
    )
    result = outcome.get("float_result")
    if (
        type(result) is not float
        or not math.isfinite(result)
        or (result == 0.0 and math.copysign(1.0, result) < 0.0)
        or outcome != {"kind": "returned", "float_result": expected["float_result"]}
    ):
        raise ValueError(
            f"reasoning score call {sequence} did not return the expected reward"
        )


@dataclass(frozen=True)
class StrictGymChildRuntimeSession:
    """Prepared immutable launch inputs and their exclusive receipt root."""

    environment: str
    scope: str
    receipt_root: Path
    spec_path: Path
    spec_sha256: str
    bootstrap_root: Path
    bootstrap_sha256: str
    spec: dict[str, Any]
    _started_index: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _started_index_sha256: str | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _score_finalized: bool = field(default=False, init=False, repr=False, compare=False)

    @contextmanager
    def launch_environment(self) -> Iterator[None]:
        """Expose only the sealed hook while RunHelper creates its children."""
        if self.scope == "scorer-only":
            import ray

            if not ray.is_initialized():
                raise RuntimeError(
                    "scorer-only Gym launch requires Ray initialized before hook injection"
                )

            # The pinned RunHelper normally renders a Bash command that sources
            # a persistent venv activation script and then invokes bare
            # ``python``.  That creates executable boundaries before Python's
            # sitecustomize hook.  Replace only the two pinned module globals
            # used by RunHelper.start and launch the already-attested
            # interpreter directly in isolated/no-site mode instead.  The
            # returned Popen remains the object owned by RunHelper, so its
            # readiness and shutdown bookkeeping are unchanged.
            import nemo_gym.cli.env as gym_env
            import nemo_gym.cli.setup_command as setup_command
            from omegaconf import OmegaConf

            if (
                gym_env.setup_env_command is not setup_command.setup_env_command
                or gym_env.run_command is not setup_command.run_command
            ):
                raise RuntimeError("pinned Gym launch functions were already replaced")
            if len(self.spec.get("targets", [])) != 1:
                raise RuntimeError("scorer-only direct launch requires one target")
            target = self.spec["targets"][0]
            component = Path(target["component_dir"])
            entrypoint = component / target["entrypoint"]
            bootstrap_source = self.bootstrap_root / "sitecustomize.py"
            sentinel = "NRL_STRICT_DIRECT_SCORER_LAUNCH_V1"
            setup_state: dict[str, Any] = {}

            def strict_setup_env_command(
                dir_path: Path, global_config_dict: Any, prefix: str
            ) -> str:
                if setup_state:
                    raise RuntimeError(
                        "strict scorer setup was rendered more than once"
                    )
                if (
                    Path(dir_path).resolve(strict=True) != component
                    or prefix != target["config_path"]
                    or global_config_dict.get("skip_venv_if_present") is not True
                    or global_config_dict.get("dry_run") is not False
                    or "nemo_gym_log_dir" in global_config_dict
                    or Path(global_config_dict.get("uv_venv_dir", ""))
                    != STRICT_GYM_VENV_ROOT
                    or Path(setup_command.get_venv_path(dir_path, global_config_dict))
                    != Path(target["venv"])
                ):
                    raise RuntimeError("strict scorer RunHelper setup differs")
                setup_state.update(
                    {
                        "config": global_config_dict,
                        "config_path": prefix,
                        "yaml": OmegaConf.to_yaml(global_config_dict),
                    }
                )
                return sentinel

            def strict_run_command(
                command: str,
                working_dir_path: Path,
                server_name: str = "",
                project_root: Path | None = None,
                *,
                global_config_dict: Any = None,
                stdout_target: Any = None,
                stderr_target: Any = None,
            ) -> subprocess.Popen[bytes]:
                if setup_state.get("process_started") is not None:
                    raise RuntimeError(
                        "strict scorer process was launched more than once"
                    )
                if set(setup_state) != {"config", "config_path", "yaml"}:
                    raise RuntimeError("strict scorer setup was not authenticated")
                escaped_yaml = shlex.quote(setup_state["yaml"])
                expected_command = (
                    f"{sentinel} \\\n"
                    f"    && NEMO_GYM_CONFIG_DICT={escaped_yaml} \\\n"
                    f"    NEMO_GYM_CONFIG_PATH={shlex.quote(target['config_path'])} \\\n"
                    f"    python {target['entrypoint']}"
                )
                if (
                    command != expected_command
                    or Path(working_dir_path).resolve(strict=True) != component
                    or server_name != target["config_path"]
                    or project_root is not None
                    or global_config_dict is not None
                    or stdout_target is not None
                    or stderr_target is not None
                ):
                    raise RuntimeError("strict scorer rendered launch command differs")

                child_environment = {
                    "EXPECTED_GYM_GITLINK_COMMIT": STRICT_GYM_GIT_COMMIT,
                    "EXPECTED_GYM_TREE": STRICT_GYM_TREE,
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "NEMO_GYM_CONFIG_DICT": setup_state["yaml"],
                    "NEMO_GYM_CONFIG_PATH": target["config_path"],
                    "PAIR_ID": self.spec["pair_id"],
                    "PATH": "/usr/bin:/bin",
                    "RESULTS_DIR": self.spec["results_dir"],
                    "STRICT_PAIR_BOUND_JOB_ID": self.spec["job_id"],
                    "STRICT_PAIR_ENVIRONMENT": self.spec["environment"],
                    "TZ": "UTC",
                    _BOOTSTRAP_ENV: str(self.bootstrap_root),
                    _BOOTSTRAP_SHA_ENV: self.bootstrap_sha256,
                    _DIRECT_RUNNER_ENV: "1",
                    _SPEC_ENV: str(self.spec_path),
                    _SPEC_SHA_ENV: self.spec_sha256,
                }
                argv = [
                    target["interpreter"],
                    "-I",
                    "-S",
                    "-B",
                    str(bootstrap_source),
                    str(entrypoint),
                ]
                process = subprocess.Popen(
                    argv,
                    cwd=component,
                    env=child_environment,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
                setup_state["process_started"] = process.pid
                return process

            original_setup = gym_env.setup_env_command
            original_run = gym_env.run_command
            gym_env.setup_env_command = strict_setup_env_command
            gym_env.run_command = strict_run_command
            primary_failure = False
            try:
                yield
            except BaseException:
                primary_failure = True
                raise
            finally:
                replacement_changed = (
                    gym_env.setup_env_command is not strict_setup_env_command
                    or gym_env.run_command is not strict_run_command
                )
                gym_env.setup_env_command = original_setup
                gym_env.run_command = original_run
                if replacement_changed and not primary_failure:
                    raise RuntimeError(
                        "strict scorer launch functions changed during RunHelper.start"
                    )
            if setup_state.get("process_started") is None:
                raise RuntimeError("strict scorer RunHelper launched no process")
            return

        disabled_pycache = self.bootstrap_root / "__pycache_disabled__"
        if disabled_pycache.exists():
            raise RuntimeError("strict Gym disabled-pycache path unexpectedly exists")
        controlled = {
            "PYTHONPATH": str(self.bootstrap_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPYCACHEPREFIX": str(disabled_pycache),
            _SPEC_ENV: str(self.spec_path),
            _SPEC_SHA_ENV: self.spec_sha256,
            _BOOTSTRAP_ENV: str(self.bootstrap_root),
            _BOOTSTRAP_SHA_ENV: self.bootstrap_sha256,
        }
        scrubbed_names = {
            name
            for name in os.environ
            if name in _PRE_PYTHON_INJECTION_ENV_NAMES
            or name.startswith(_PRE_PYTHON_INJECTION_ENV_PREFIXES)
        }
        previous = {
            name: os.environ.get(name) for name in scrubbed_names | set(controlled)
        }
        for name in (_SPEC_ENV, _SPEC_SHA_ENV, _BOOTSTRAP_ENV, _BOOTSTRAP_SHA_ENV):
            if previous[name] not in (None, ""):
                raise RuntimeError(
                    f"strict Gym child control variable was pre-set: {name}"
                )
        try:
            for name in scrubbed_names:
                os.environ.pop(name, None)
            os.environ.update(controlled)
            yield
        finally:
            for name in tuple(os.environ):
                if (
                    name in _PRE_PYTHON_INJECTION_ENV_NAMES
                    or name.startswith(_PRE_PYTHON_INJECTION_ENV_PREFIXES)
                    or name in controlled
                ):
                    os.environ.pop(name, None)
            for name, value in previous.items():
                if value is not None:
                    os.environ[name] = value

    def attest_started(self, run_helper: Any) -> tuple[dict[str, Any], str]:
        """Join in-process receipts to live RunHelper processes and sockets."""
        if self._started_index is not None or self._started_index_sha256 is not None:
            raise RuntimeError("strict Gym children were already attested")
        spec, payload = _load_canonical_document(self.spec_path)
        if spec != self.spec or _sha256_bytes(payload) != self.spec_sha256:
            raise RuntimeError("strict Gym child spec changed after publication")
        instances = getattr(run_helper, "_server_instance_display_configs", None)
        processes = getattr(run_helper, "_processes", None)
        if not isinstance(instances, list) or not isinstance(processes, dict):
            raise RuntimeError(
                "strict Gym child attestation requires RunHelper metadata"
            )

        expected_files = {"spec.json"}
        records: list[dict[str, Any]] = []
        for target in self.spec["targets"]:
            matches = [
                item
                for item in instances
                if getattr(item, "config_path", None) == target["config_path"]
                and getattr(item, "server_type", None) == target["server_type"]
                and getattr(item, "name", None) == target["server_name"]
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "strict Gym child attestation requires exactly one selected "
                    f"{target['role']} instance; found {len(matches)}"
                )
            instance = matches[0]
            if Path(getattr(instance, "dir_path", "")).resolve(strict=True) != Path(
                target["component_dir"]
            ):
                raise RuntimeError(
                    "RunHelper selected a shadow Gym component directory"
                )
            process = processes.get(target["config_path"])
            wrapper_pid = getattr(process, "pid", None)
            if (
                process is None
                or type(wrapper_pid) is not int
                or process.poll() is not None
            ):
                raise RuntimeError("selected RunHelper wrapper process is not live")
            if getattr(instance, "pid", None) != wrapper_pid:
                raise RuntimeError("RunHelper process table and display PID differ")

            filename = target["receipt_filename"]
            expected_files.add(filename)
            receipt, receipt_payload = _load_canonical_document(
                self.receipt_root / filename
            )
            observation = _validate_receipt(
                receipt,
                spec=self.spec,
                target=target,
                instance=instance,
                wrapper_pid=wrapper_pid,
            )
            records.append(
                {
                    "role": target["role"],
                    "config_path": target["config_path"],
                    "receipt": {
                        "path": str(self.receipt_root / filename),
                        "sha256": _sha256_bytes(receipt_payload),
                        "schema": STRICT_GYM_CHILD_RECEIPT_SCHEMA,
                    },
                    "observation": observation,
                }
            )

        actual_files = {item.name for item in self.receipt_root.iterdir()}
        if actual_files != expected_files:
            raise RuntimeError(
                "strict Gym child receipt root has an unexpected inventory: "
                f"{sorted(actual_files)!r}"
            )
        index = {
            "schema": STRICT_GYM_CHILD_INDEX_SCHEMA,
            "hash_domain": STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": self.spec["environment"],
            "scope": self.spec["scope"],
            "pair_id": self.spec["pair_id"],
            "job_id": self.spec["job_id"],
            "gym": self.spec["gym"],
            "spec": {
                "path": str(self.spec_path),
                "sha256": self.spec_sha256,
                "schema": STRICT_GYM_CHILD_SPEC_SCHEMA,
            },
            "children": records,
        }
        root_fd = os.open(self.receipt_root, _DIR_FLAGS)
        try:
            _, digest = _publish_document(root_fd, "index.json", index)
        finally:
            os.close(root_fd)
        object.__setattr__(self, "_started_index", index)
        object.__setattr__(self, "_started_index_sha256", digest)
        return index, digest

    def finalize_score_calls(
        self,
        expected_calls: Sequence[Mapping[str, Any]],
        *,
        run_helper: Any,
    ) -> tuple[dict[str, Any], str]:
        """Close scorer-only Reasoning Gym evidence after exactly four calls.

        The selected resource app catches scorer exceptions and converts them
        to reward ``0.0``.  The in-process hook records every resolver/scorer
        attempt before that catch.  This finalizer admits only four contiguous
        successful call receipts whose ordered inputs and rewards match the
        authenticated replay driver.
        """
        if self.scope != "scorer-only" or self.environment != "reasoning_gym":
            raise RuntimeError(
                "score-call finalization is only valid for scorer-only reasoning replay"
            )
        if self._score_finalized:
            raise RuntimeError("strict reasoning score calls were already finalized")
        if isinstance(expected_calls, (str, bytes)) or len(expected_calls) != 4:
            raise ValueError("strict reasoning score finalization requires exact K=4")
        normalized: list[dict[str, Any]] = []
        for index, value in enumerate(expected_calls, start=1):
            expected = _require_exact_keys(
                value,
                {"task_name", "answer_sha256", "entry_sha256", "float_result"},
                name=f"expected reasoning score call {index}",
            )
            if expected["task_name"] != "knights_knaves":
                raise ValueError("expected reasoning task differs from pinned scorer")
            for name in ("answer_sha256", "entry_sha256"):
                if (
                    type(expected[name]) is not str
                    or _SHA256_RE.fullmatch(expected[name]) is None
                ):
                    raise ValueError(f"expected reasoning score {name} is invalid")
            reward = expected["float_result"]
            if (
                type(reward) is not float
                or not math.isfinite(reward)
                or not 0.0 <= reward <= 1.0
                or (reward == 0.0 and math.copysign(1.0, reward) < 0.0)
            ):
                raise ValueError(
                    "expected reasoning reward must be finite float in [0,1]"
                )
            normalized.append(dict(expected))

        spec, spec_payload = _load_canonical_document(self.spec_path)
        if spec != self.spec or _sha256_bytes(spec_payload) != self.spec_sha256:
            raise RuntimeError(
                "strict Gym child spec changed before score finalization"
            )
        child_index_path = self.receipt_root / "index.json"
        child_index, child_index_payload = _load_canonical_document(child_index_path)
        if (
            self._started_index is None
            or self._started_index_sha256 is None
            or child_index != self._started_index
            or _sha256_bytes(child_index_payload) != self._started_index_sha256
        ):
            raise RuntimeError(
                "strict Gym child index differs from the retained startup attestation"
            )
        _require_exact_keys(
            child_index,
            {
                "schema",
                "hash_domain",
                "environment",
                "scope",
                "pair_id",
                "job_id",
                "gym",
                "spec",
                "children",
            },
            name="strict Gym child index",
        )
        if (
            child_index["schema"] != STRICT_GYM_CHILD_INDEX_SCHEMA
            or child_index["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
            or child_index["environment"] != self.environment
            or child_index["scope"] != self.scope
            or child_index["pair_id"] != spec["pair_id"]
            or child_index["job_id"] != spec["job_id"]
            or child_index["gym"] != spec["gym"]
            or child_index["spec"]
            != {
                "path": str(self.spec_path),
                "sha256": self.spec_sha256,
                "schema": STRICT_GYM_CHILD_SPEC_SCHEMA,
            }
            or not isinstance(child_index["children"], list)
            or len(child_index["children"]) != 1
            or child_index["children"][0].get("role") != "resource"
        ):
            raise ValueError("strict Gym child index differs at score finalization")

        child = _require_exact_keys(
            child_index["children"][0],
            {"role", "config_path", "receipt", "observation"},
            name="strict reasoning child-index resource",
        )
        target = spec["targets"][0]
        if child["role"] != "resource" or child["config_path"] != target["config_path"]:
            raise ValueError("strict reasoning child-index target differs")
        resource_path = self.receipt_root / "resource.json"
        resource, resource_payload = _load_canonical_document(resource_path)
        resource_ref = _require_exact_keys(
            child["receipt"],
            {"path", "sha256", "schema"},
            name="strict reasoning child-index receipt ref",
        )
        if not _exact_json_equal(
            resource_ref,
            {
                "path": str(resource_path),
                "sha256": _sha256_bytes(resource_payload),
                "schema": STRICT_GYM_CHILD_RECEIPT_SCHEMA,
            },
        ):
            raise ValueError("strict reasoning child-index receipt ref differs")
        observation = _require_exact_keys(
            child["observation"],
            {
                "pid",
                "start_ticks",
                "wrapper_pid",
                "host",
                "port",
                "listener_socket_inodes",
            },
            name="strict reasoning child-index observation",
        )
        if (
            type(observation["pid"]) is not int
            or observation["pid"] <= 1
            or type(observation["start_ticks"]) is not int
            or observation["start_ticks"] <= 0
            or type(observation["wrapper_pid"]) is not int
            or observation["wrapper_pid"] <= 1
            or observation["host"] != "127.0.0.1"
            or type(observation["port"]) is not int
            or not 5000 <= observation["port"] <= 5999
            or not isinstance(observation["listener_socket_inodes"], list)
            or not observation["listener_socket_inodes"]
            or any(
                type(item) is not int or item <= 0
                for item in observation["listener_socket_inodes"]
            )
        ):
            raise ValueError(
                "strict reasoning child-index observation types are invalid"
            )
        server = resource.get("server")
        if not isinstance(server, dict):
            raise ValueError("strict reasoning resource server receipt is invalid")
        instances = getattr(run_helper, "_server_instance_display_configs", None)
        processes = getattr(run_helper, "_processes", None)
        if not isinstance(instances, list) or not isinstance(processes, dict):
            raise RuntimeError("score finalization requires the attested RunHelper")
        matches = [
            item
            for item in instances
            if getattr(item, "config_path", None) == target["config_path"]
            and getattr(item, "server_type", None) == target["server_type"]
            and getattr(item, "name", None) == target["server_name"]
        ]
        wrapper_process = processes.get(target["config_path"])
        wrapper_pid = getattr(wrapper_process, "pid", None)
        if (
            len(matches) != 1
            or Path(getattr(matches[0], "dir_path", "")).resolve(strict=True)
            != Path(target["component_dir"])
            or type(wrapper_pid) is not int
            or wrapper_pid != observation["wrapper_pid"]
            or wrapper_process.poll() is not None
        ):
            raise RuntimeError(
                "score finalization RunHelper differs from startup attestation"
            )
        revalidated_observation = _validate_receipt(
            resource,
            spec=spec,
            target=target,
            instance=matches[0],
            wrapper_pid=wrapper_pid,
        )
        if not _exact_json_equal(revalidated_observation, observation):
            raise ValueError("strict reasoning live observation changed")
        process = _require_exact_keys(
            resource["process"],
            {
                "pid",
                "ppid",
                "uid",
                "gid",
                "cwd",
                "sys_executable",
                "sys_prefix",
                "sys_base_prefix",
                "proc_exe",
                "sys_argv",
                "proc_argv",
                "start_ticks",
                "boot_id",
                "hostname",
            },
            name="strict reasoning resource process",
        )
        current_ppid, current_start = _process_stat(process["pid"])
        if (
            observation.get("pid") != process["pid"]
            or observation.get("start_ticks") != process["start_ticks"]
            or current_start != process["start_ticks"]
            or current_ppid != process["ppid"]
            or process["boot_id"] != _boot_id()
            or process["hostname"] != socket.gethostname()
            or not _process_is_descendant(
                process["pid"], observation.get("wrapper_pid", -1)
            )
            or _process_descendant_identities(process["pid"])
            or not _listening_socket_inodes(
                process["pid"], observation.get("host", ""), observation.get("port", -1)
            )
        ):
            raise RuntimeError(
                "strict reasoning scorer process is not the attested live resource"
            )

        expected_inventory = {
            "spec.json",
            "resource.json",
            "index.json",
            "reasoning-score-closed.json",
        } | {f"reasoning-score-call-{sequence:08d}.json" for sequence in range(1, 5)}
        actual_inventory = {item.name for item in self.receipt_root.iterdir()}
        if actual_inventory != expected_inventory:
            raise RuntimeError(
                "strict reasoning score-call inventory has an extra, gap, or missing call: "
                f"{sorted(actual_inventory)!r}"
            )
        closed_path = self.receipt_root / "reasoning-score-closed.json"
        closed, closed_payload = _load_canonical_document(closed_path)
        closed_record = _require_exact_keys(
            closed,
            {
                "schema",
                "hash_domain",
                "environment",
                "pair_id",
                "job_id",
                "spec_sha256",
                "process",
                "call_count",
                "calls",
            },
            name="strict reasoning score closed receipt",
        )
        closed_process = _require_exact_keys(
            closed_record["process"],
            {"pid", "start_ticks"},
            name="strict reasoning score closed process",
        )
        closed_refs = closed_record["calls"]
        if (
            closed_record["schema"] != STRICT_GYM_SCORE_CLOSED_SCHEMA
            or closed_record["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
            or closed_record["environment"] != spec["environment"]
            or closed_record["pair_id"] != spec["pair_id"]
            or closed_record["job_id"] != spec["job_id"]
            or closed_record["spec_sha256"] != self.spec_sha256
            or type(closed_process["pid"]) is not int
            or type(closed_process["start_ticks"]) is not int
            or not _exact_json_equal(
                closed_process,
                {"pid": process["pid"], "start_ticks": process["start_ticks"]},
            )
            or type(closed_record["call_count"]) is not int
            or closed_record["call_count"] != 4
            or not isinstance(closed_refs, list)
            or len(closed_refs) != 4
        ):
            raise ValueError("strict reasoning score closed receipt differs")
        records: list[dict[str, Any]] = []
        call_payloads: list[bytes] = []
        for sequence, expected in enumerate(normalized, start=1):
            filename = f"reasoning-score-call-{sequence:08d}.json"
            document, payload = _load_canonical_document(self.receipt_root / filename)
            _validate_score_call(
                document,
                spec=spec,
                sequence=sequence,
                expected=expected,
                process=process,
            )
            closed_ref = _require_exact_keys(
                closed_refs[sequence - 1],
                {"sequence", "path", "sha256", "schema"},
                name=f"strict reasoning score closed call ref {sequence}",
            )
            if not _exact_json_equal(
                closed_ref,
                {
                    "sequence": sequence,
                    "path": str(self.receipt_root / filename),
                    "sha256": _sha256_bytes(payload),
                    "schema": STRICT_GYM_SCORE_CALL_SCHEMA,
                },
            ):
                raise ValueError(
                    f"strict reasoning score closed call ref {sequence} differs"
                )
            call_payloads.append(payload)
            records.append(
                {
                    "sequence": sequence,
                    "task_name": expected["task_name"],
                    "input": {
                        "answer_sha256": expected["answer_sha256"],
                        "entry_sha256": expected["entry_sha256"],
                    },
                    "float_result": expected["float_result"],
                    "receipt": {
                        "path": str(self.receipt_root / filename),
                        "sha256": _sha256_bytes(payload),
                        "schema": STRICT_GYM_SCORE_CALL_SCHEMA,
                    },
                }
            )

        # The process-owned CLOSED receipt prevents any fifth score from
        # entering the trusted callable. Signal the exact authenticated
        # resource through a pidfd first: RunHelper owns only its Bash wrapper
        # and, when logging is enabled, that wrapper may head a tee pipeline.
        # Then reap the wrapper and re-read all evidence offline.
        child_termination_signal = _terminate_authenticated_process(
            process["pid"], process["start_ticks"]
        )
        run_helper.shutdown()
        wrapper_returncode = wrapper_process.poll()
        if (
            type(wrapper_returncode) is not int
            or getattr(run_helper, "_processes", None) != {}
        ):
            raise RuntimeError(
                "strict reasoning RunHelper did not reap its resource wrapper"
            )
        try:
            _, post_shutdown_start = _process_stat(process["pid"])
        except (FileNotFoundError, ProcessLookupError):
            post_shutdown_start = None
        if post_shutdown_start == process["start_ticks"]:
            raise RuntimeError(
                "strict reasoning scorer child remained live after shutdown"
            )

        offline_spec, offline_spec_payload = _load_canonical_document(self.spec_path)
        offline_index, offline_index_payload = _load_canonical_document(
            child_index_path
        )
        offline_resource, offline_resource_payload = _load_canonical_document(
            resource_path
        )
        offline_closed, offline_closed_payload = _load_canonical_document(closed_path)
        if (
            offline_spec != spec
            or offline_spec_payload != spec_payload
            or offline_index != child_index
            or offline_index_payload != child_index_payload
            or offline_resource != resource
            or offline_resource_payload != resource_payload
            or offline_closed != closed
            or offline_closed_payload != closed_payload
        ):
            raise RuntimeError(
                "strict reasoning evidence changed across scorer shutdown"
            )
        for sequence, original_payload in enumerate(call_payloads, start=1):
            _, offline_payload = _load_canonical_document(
                self.receipt_root / f"reasoning-score-call-{sequence:08d}.json"
            )
            if offline_payload != original_payload:
                raise RuntimeError(
                    "strict reasoning call evidence changed across scorer shutdown"
                )
        if {item.name for item in self.receipt_root.iterdir()} != expected_inventory:
            raise RuntimeError(
                "strict reasoning score-call inventory changed across scorer shutdown"
            )

        terminal = {
            "schema": STRICT_GYM_SCORE_CALL_INDEX_SCHEMA,
            "hash_domain": STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": self.environment,
            "scope": self.scope,
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec": {
                "path": str(self.spec_path),
                "sha256": self.spec_sha256,
                "schema": STRICT_GYM_CHILD_SPEC_SCHEMA,
            },
            "child_index": {
                "path": str(child_index_path),
                "sha256": _sha256_bytes(child_index_payload),
                "schema": STRICT_GYM_CHILD_INDEX_SCHEMA,
            },
            "resource_receipt": {
                "path": str(resource_path),
                "sha256": _sha256_bytes(resource_payload),
                "schema": STRICT_GYM_CHILD_RECEIPT_SCHEMA,
            },
            "score_closed": {
                "path": str(closed_path),
                "sha256": _sha256_bytes(closed_payload),
                "schema": STRICT_GYM_SCORE_CLOSED_SCHEMA,
            },
            "quiescence": {
                "pid": process["pid"],
                "start_ticks": process["start_ticks"],
                "child_termination_signal": child_termination_signal,
                "wrapper_pid": wrapper_pid,
                "wrapper_returncode": wrapper_returncode,
                "original_process_reaped": True,
            },
            "call_count": 4,
            "calls": records,
        }
        root_fd = os.open(self.receipt_root, _DIR_FLAGS)
        try:
            _, digest = _publish_document(
                root_fd, "reasoning-score-call-index.json", terminal
            )
        finally:
            os.close(root_fd)
        final_inventory = {item.name for item in self.receipt_root.iterdir()}
        if final_inventory != expected_inventory | {"reasoning-score-call-index.json"}:
            raise RuntimeError("strict reasoning score-call terminal inventory changed")
        admitted_terminal, admitted_digest = load_finalized_reasoning_score_call_index(
            self.receipt_root / "reasoning-score-call-index.json",
            expected_sha256=digest,
            expected_receipt_root=self.receipt_root,
            expected_pair_id=spec["pair_id"],
            expected_job_id=spec["job_id"],
        )
        if admitted_terminal != terminal or admitted_digest != digest:
            raise RuntimeError("strict reasoning score-call offline admission differs")
        object.__setattr__(self, "_score_finalized", True)
        return terminal, digest


def load_finalized_reasoning_score_call_index(
    path: Path,
    *,
    expected_sha256: str,
    expected_receipt_root: Path,
    expected_pair_id: str,
    expected_job_id: str,
) -> tuple[dict[str, Any], str]:
    """Load a terminal K=4 scorer graph from an externally anchored digest.

    This is the post-shutdown admission boundary used by replay EXIT.  It is
    deliberately independent of :class:`StrictGymChildRuntimeSession`: every
    artifact is loaded again from the caller-authenticated receipt root, all
    paths and references are exact, and the complete graph is stable-read a
    second time before returning.
    """
    if (
        type(expected_sha256) is not str
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or expected_sha256 == "0" * 64
    ):
        raise ValueError("expected score-call index SHA256 is invalid")
    if (
        type(expected_pair_id) is not str
        or _PAIR_ID_RE.fullmatch(expected_pair_id) is None
    ):
        raise ValueError("expected score-call pair ID is invalid")
    if (
        type(expected_job_id) is not str
        or _JOB_ID_RE.fullmatch(expected_job_id) is None
    ):
        raise ValueError("expected score-call job ID is invalid")
    root = _require_private_canonical_directory(
        Path(expected_receipt_root), name="finalized score-call receipt root"
    )
    terminal_path = root / "reasoning-score-call-index.json"
    if not isinstance(path, Path) or path != terminal_path:
        raise ValueError("score-call index path is not the exact expected path")

    filenames = {
        "spec.json",
        "resource.json",
        "index.json",
        "reasoning-score-closed.json",
        "reasoning-score-call-index.json",
    } | {f"reasoning-score-call-{sequence:08d}.json" for sequence in range(1, 5)}
    root_before = root.lstat()
    if {item.name for item in root.iterdir()} != filenames:
        raise ValueError("finalized score-call receipt inventory differs")
    documents: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for filename in sorted(filenames):
        document, payload = _load_canonical_document(root / filename)
        documents[filename] = document
        payloads[filename] = payload
    if _sha256_bytes(payloads[terminal_path.name]) != expected_sha256:
        raise ValueError("score-call index differs from caller-carried SHA256")

    def exact_ref(
        value: Any, *, filename: str, schema: str, name: str
    ) -> Mapping[str, Any]:
        ref = _require_exact_keys(value, {"path", "sha256", "schema"}, name=name)
        expected = {
            "path": str(root / filename),
            "sha256": _sha256_bytes(payloads[filename]),
            "schema": schema,
        }
        if not _exact_json_equal(ref, expected):
            raise ValueError(f"{name} differs from the exact artifact")
        return ref

    spec = _require_exact_keys(
        documents["spec.json"],
        {
            "schema",
            "hash_domain",
            "environment",
            "scope",
            "pair_id",
            "job_id",
            "gym",
            "results_dir",
            "receipt_root",
            "bootstrap",
            "targets",
        },
        name="finalized score-call spec",
    )
    expected_gym = {
        "git_commit": STRICT_GYM_GIT_COMMIT,
        "tree": STRICT_GYM_TREE,
        "root": str(STRICT_GYM_ROOT),
        "venv_root": str(STRICT_GYM_VENV_ROOT),
        "sources": dict(_GYM_SOURCE_PINS),
    }
    expected_targets = _target_matrix(
        "reasoning_gym", STRICT_GYM_ROOT, scope="scorer-only"
    )
    bootstrap = _require_exact_keys(
        spec["bootstrap"], {"root", "filename", "sha256"}, name="score-call bootstrap"
    )
    bootstrap_root = bootstrap.get("root")
    sealed_bootstrap_root, sealed_bootstrap_sha256 = _require_sealed_bootstrap_root()
    if (
        spec["schema"] != STRICT_GYM_CHILD_SPEC_SCHEMA
        or spec["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or spec["environment"] != "reasoning_gym"
        or spec["scope"] != "scorer-only"
        or spec["pair_id"] != expected_pair_id
        or spec["job_id"] != expected_job_id
        or not _exact_json_equal(spec["gym"], expected_gym)
        or spec["results_dir"] != str(root.parent)
        or spec["receipt_root"] != str(root)
        or not _exact_json_equal(spec["targets"], expected_targets)
        or type(bootstrap_root) is not str
        or bootstrap_root != str(sealed_bootstrap_root)
        or bootstrap["filename"] != "sitecustomize.py"
        or type(bootstrap["sha256"]) is not str
        or _SHA256_RE.fullmatch(bootstrap["sha256"]) is None
        or bootstrap["sha256"] != sealed_bootstrap_sha256
    ):
        raise ValueError("finalized score-call spec differs")
    spec_sha256 = _sha256_bytes(payloads["spec.json"])
    target = expected_targets[0]

    child_index = _require_exact_keys(
        documents["index.json"],
        {
            "schema",
            "hash_domain",
            "environment",
            "scope",
            "pair_id",
            "job_id",
            "gym",
            "spec",
            "children",
        },
        name="finalized score-call child index",
    )
    children = child_index["children"]
    if (
        child_index["schema"] != STRICT_GYM_CHILD_INDEX_SCHEMA
        or child_index["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or child_index["environment"] != "reasoning_gym"
        or child_index["scope"] != "scorer-only"
        or child_index["pair_id"] != expected_pair_id
        or child_index["job_id"] != expected_job_id
        or not _exact_json_equal(child_index["gym"], expected_gym)
        or not isinstance(children, list)
        or len(children) != 1
    ):
        raise ValueError("finalized score-call child index differs")
    exact_ref(
        child_index["spec"],
        filename="spec.json",
        schema=STRICT_GYM_CHILD_SPEC_SCHEMA,
        name="finalized score-call child-index spec ref",
    )
    child = _require_exact_keys(
        children[0],
        {"role", "config_path", "receipt", "observation"},
        name="finalized score-call child",
    )
    exact_ref(
        child["receipt"],
        filename="resource.json",
        schema=STRICT_GYM_CHILD_RECEIPT_SCHEMA,
        name="finalized score-call resource ref",
    )
    observation = _require_exact_keys(
        child["observation"],
        {
            "pid",
            "start_ticks",
            "wrapper_pid",
            "host",
            "port",
            "listener_socket_inodes",
        },
        name="finalized score-call observation",
    )
    listener_inodes = observation["listener_socket_inodes"]
    if (
        child["role"] != "resource"
        or child["config_path"] != target["config_path"]
        or type(observation["pid"]) is not int
        or observation["pid"] <= 1
        or type(observation["start_ticks"]) is not int
        or observation["start_ticks"] <= 0
        or type(observation["wrapper_pid"]) is not int
        or observation["wrapper_pid"] != observation["pid"]
        or observation["host"] != "127.0.0.1"
        or type(observation["port"]) is not int
        or not 5000 <= observation["port"] <= 5999
        or not isinstance(listener_inodes, list)
        or not listener_inodes
        or any(type(item) is not int or item <= 0 for item in listener_inodes)
        or len(set(listener_inodes)) != len(listener_inodes)
    ):
        raise ValueError("finalized score-call observation differs")

    resource = _require_exact_keys(
        documents["resource.json"],
        {
            "schema",
            "hash_domain",
            "environment",
            "pair_id",
            "job_id",
            "stage",
            "spec_sha256",
            "target",
            "server",
            "process",
            "distribution_versions",
            "module_versions",
            "scorer",
        },
        name="finalized score-call resource receipt",
    )
    target_record_keys = {
        "role",
        "config_path",
        "server_type",
        "server_name",
        "component_dir",
        "entrypoint",
        "source_path",
        "source_sha256",
        "config_path_source",
        "config_sha256",
        "requirements_path",
        "requirements_sha256",
        "venv",
        "interpreter",
    }
    target_record = _require_exact_keys(
        resource["target"], target_record_keys, name="score-call resource target"
    )
    server = _require_exact_keys(
        resource["server"],
        {
            "config_path",
            "server_type",
            "server_name",
            "entrypoint",
            "host",
            "port",
            "num_workers",
        },
        name="score-call resource server",
    )
    process = _require_exact_keys(
        resource["process"],
        {
            "pid",
            "ppid",
            "uid",
            "gid",
            "cwd",
            "sys_executable",
            "sys_prefix",
            "sys_base_prefix",
            "proc_exe",
            "sys_argv",
            "proc_argv",
            "start_ticks",
            "boot_id",
            "hostname",
        },
        name="score-call resource process",
    )
    bootstrap_source = str(Path(bootstrap_root) / "sitecustomize.py")
    expected_sys_argv = [bootstrap_source, target["source_path"]]
    expected_proc_argv = [
        target["interpreter"],
        "-I",
        "-S",
        "-B",
        bootstrap_source,
        target["source_path"],
    ]
    if (
        resource["schema"] != STRICT_GYM_CHILD_RECEIPT_SCHEMA
        or resource["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or resource["environment"] != "reasoning_gym"
        or resource["pair_id"] != expected_pair_id
        or resource["job_id"] != expected_job_id
        or resource["stage"] != "isolated-runner-pre-entrypoint"
        or resource["spec_sha256"] != spec_sha256
        or not _exact_json_equal(
            target_record, {name: target[name] for name in target_record_keys}
        )
        or not _exact_json_equal(
            resource["distribution_versions"], target["distribution_versions"]
        )
        or not _exact_json_equal(resource["module_versions"], target["module_versions"])
        or server["config_path"] != target["config_path"]
        or server["server_type"] != target["server_type"]
        or server["server_name"] != target["server_name"]
        or server["entrypoint"] != target["entrypoint"]
        or server["host"] != observation["host"]
        or type(server["port"]) is not int
        or server["port"] != observation["port"]
        or (
            server["num_workers"] is not None
            and (type(server["num_workers"]) is not int or server["num_workers"] != 1)
        )
        or type(process["pid"]) is not int
        or process["pid"] != observation["pid"]
        or type(process["ppid"]) is not int
        or process["ppid"] <= 0
        or type(process["uid"]) is not int
        or process["uid"] != os.geteuid()
        or type(process["gid"]) is not int
        or process["gid"] != os.getegid()
        or type(process["start_ticks"]) is not int
        or process["start_ticks"] != observation["start_ticks"]
        or process["cwd"] != target["component_dir"]
        or process["sys_executable"] != target["interpreter"]
        or process["sys_prefix"] != target["venv"]
        or type(process["sys_base_prefix"]) is not str
        or not Path(process["sys_base_prefix"]).is_absolute()
        or ".." in Path(process["sys_base_prefix"]).parts
        or Path(process["sys_base_prefix"]).resolve(strict=True)
        != Path(process["sys_base_prefix"])
        or process["sys_base_prefix"] == process["sys_prefix"]
        or type(process["proc_exe"]) is not str
        or not process["proc_exe"].startswith("/")
        or process["sys_argv"] != expected_sys_argv
        or process["proc_argv"] != expected_proc_argv
        or type(process["boot_id"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            process["boot_id"],
        )
        is None
        or type(process["hostname"]) is not str
        or not process["hostname"]
    ):
        raise ValueError("finalized score-call resource receipt differs")
    expected_scorer = target["scorer"]
    dynamic_scorer_keys = {
        "package_root",
        "package_resolved_root",
        "module_origin",
        "module_resolved_origin",
        "resolver_origin",
        "resolver_resolved_origin",
        "origin",
        "resolved_origin",
    }
    scorer = _require_exact_keys(
        resource["scorer"],
        set(expected_scorer) | dynamic_scorer_keys,
        name="finalized score-call scorer",
    )
    if any(
        not _exact_json_equal(scorer[name], value)
        for name, value in expected_scorer.items()
    ):
        raise ValueError("finalized score-call scorer static identity differs")
    purelib = Path(target["venv"]) / "lib/python3.13/site-packages"
    expected_scorer_paths = {
        "package_root": str(purelib / "reasoning_gym"),
        "module_origin": str(
            purelib / expected_scorer["module_origin_relative_to_purelib"]
        ),
        "resolver_origin": str(
            purelib / expected_scorer["resolver_origin_relative_to_purelib"]
        ),
        "origin": str(purelib / expected_scorer["origin_relative_to_purelib"]),
    }
    expected_resolved_scorer_paths = {
        "package_resolved_root": str(
            Path(expected_scorer_paths["package_root"]).resolve(strict=True)
        ),
        "module_resolved_origin": str(
            Path(expected_scorer_paths["module_origin"]).resolve(strict=True)
        ),
        "resolver_resolved_origin": str(
            Path(expected_scorer_paths["resolver_origin"]).resolve(strict=True)
        ),
        "resolved_origin": str(
            Path(expected_scorer_paths["origin"]).resolve(strict=True)
        ),
    }
    if any(
        scorer[name] != value for name, value in expected_scorer_paths.items()
    ) or any(
        scorer[name] != value for name, value in expected_resolved_scorer_paths.items()
    ):
        raise ValueError("finalized score-call scorer paths differ")

    closed = _require_exact_keys(
        documents["reasoning-score-closed.json"],
        {
            "schema",
            "hash_domain",
            "environment",
            "pair_id",
            "job_id",
            "spec_sha256",
            "process",
            "call_count",
            "calls",
        },
        name="finalized score-call closed receipt",
    )
    closed_process = _require_exact_keys(
        closed["process"], {"pid", "start_ticks"}, name="score-call closed process"
    )
    if (
        closed["schema"] != STRICT_GYM_SCORE_CLOSED_SCHEMA
        or closed["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or closed["environment"] != "reasoning_gym"
        or closed["pair_id"] != expected_pair_id
        or closed["job_id"] != expected_job_id
        or closed["spec_sha256"] != spec_sha256
        or type(closed_process["pid"]) is not int
        or type(closed_process["start_ticks"]) is not int
        or not _exact_json_equal(
            closed_process,
            {"pid": process["pid"], "start_ticks": process["start_ticks"]},
        )
        or type(closed["call_count"]) is not int
        or closed["call_count"] != 4
        or not isinstance(closed["calls"], list)
        or len(closed["calls"]) != 4
    ):
        raise ValueError("finalized score-call closed receipt differs")

    terminal = _require_exact_keys(
        documents[terminal_path.name],
        {
            "schema",
            "hash_domain",
            "environment",
            "scope",
            "pair_id",
            "job_id",
            "spec",
            "child_index",
            "resource_receipt",
            "score_closed",
            "quiescence",
            "call_count",
            "calls",
        },
        name="finalized score-call index",
    )
    if (
        terminal["schema"] != STRICT_GYM_SCORE_CALL_INDEX_SCHEMA
        or terminal["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
        or terminal["environment"] != "reasoning_gym"
        or terminal["scope"] != "scorer-only"
        or terminal["pair_id"] != expected_pair_id
        or terminal["job_id"] != expected_job_id
        or type(terminal["call_count"]) is not int
        or terminal["call_count"] != 4
        or not isinstance(terminal["calls"], list)
        or len(terminal["calls"]) != 4
    ):
        raise ValueError("finalized score-call index identity differs")
    exact_ref(
        terminal["spec"],
        filename="spec.json",
        schema=STRICT_GYM_CHILD_SPEC_SCHEMA,
        name="finalized score-call spec ref",
    )
    exact_ref(
        terminal["child_index"],
        filename="index.json",
        schema=STRICT_GYM_CHILD_INDEX_SCHEMA,
        name="finalized score-call child-index ref",
    )
    exact_ref(
        terminal["resource_receipt"],
        filename="resource.json",
        schema=STRICT_GYM_CHILD_RECEIPT_SCHEMA,
        name="finalized score-call resource-receipt ref",
    )
    exact_ref(
        terminal["score_closed"],
        filename="reasoning-score-closed.json",
        schema=STRICT_GYM_SCORE_CLOSED_SCHEMA,
        name="finalized score-call closed ref",
    )

    for sequence in range(1, 5):
        filename = f"reasoning-score-call-{sequence:08d}.json"
        record = _require_exact_keys(
            terminal["calls"][sequence - 1],
            {"sequence", "task_name", "input", "float_result", "receipt"},
            name=f"finalized score-call record {sequence}",
        )
        call_input = _require_exact_keys(
            record["input"],
            {"answer_sha256", "entry_sha256"},
            name=f"finalized score-call input {sequence}",
        )
        reward = record["float_result"]
        if (
            type(record["sequence"]) is not int
            or record["sequence"] != sequence
            or record["task_name"] != "knights_knaves"
            or any(
                type(call_input[name]) is not str
                or _SHA256_RE.fullmatch(call_input[name]) is None
                or call_input[name] == "0" * 64
                for name in ("answer_sha256", "entry_sha256")
            )
            or type(reward) is not float
            or not math.isfinite(reward)
            or not 0.0 <= reward <= 1.0
            or (reward == 0.0 and math.copysign(1.0, reward) < 0.0)
        ):
            raise ValueError(f"finalized score-call record {sequence} differs")
        exact_ref(
            record["receipt"],
            filename=filename,
            schema=STRICT_GYM_SCORE_CALL_SCHEMA,
            name=f"finalized score-call receipt ref {sequence}",
        )
        closed_ref = _require_exact_keys(
            closed["calls"][sequence - 1],
            {"sequence", "path", "sha256", "schema"},
            name=f"finalized score-call closed ref {sequence}",
        )
        expected_closed_ref = {
            "sequence": sequence,
            "path": str(root / filename),
            "sha256": _sha256_bytes(payloads[filename]),
            "schema": STRICT_GYM_SCORE_CALL_SCHEMA,
        }
        if not _exact_json_equal(closed_ref, expected_closed_ref):
            raise ValueError(f"finalized score-call closed ref {sequence} differs")
        _validate_score_call(
            documents[filename],
            spec=spec,
            sequence=sequence,
            expected={
                "task_name": record["task_name"],
                "answer_sha256": call_input["answer_sha256"],
                "entry_sha256": call_input["entry_sha256"],
                "float_result": reward,
            },
            process=process,
        )

    quiescence = _require_exact_keys(
        terminal["quiescence"],
        {
            "pid",
            "start_ticks",
            "child_termination_signal",
            "wrapper_pid",
            "wrapper_returncode",
            "original_process_reaped",
        },
        name="finalized score-call quiescence",
    )
    child_signal = quiescence["child_termination_signal"]
    if (
        type(quiescence["pid"]) is not int
        or quiescence["pid"] != process["pid"]
        or type(quiescence["start_ticks"]) is not int
        or quiescence["start_ticks"] != process["start_ticks"]
        or type(quiescence["wrapper_pid"]) is not int
        or quiescence["wrapper_pid"] != process["pid"]
        or type(child_signal) is not str
        or child_signal not in {"SIGINT", "SIGTERM", "SIGKILL"}
        or type(quiescence["wrapper_returncode"]) is not int
        or (
            child_signal == "SIGKILL"
            and quiescence["wrapper_returncode"] != -signal.SIGKILL
        )
        or quiescence["original_process_reaped"] is not True
    ):
        raise ValueError("finalized score-call quiescence differs")

    if {item.name for item in root.iterdir()} != filenames:
        raise RuntimeError("finalized score-call inventory changed during validation")
    for filename in sorted(filenames):
        second_document, second_payload = _load_canonical_document(root / filename)
        if (
            second_document != documents[filename]
            or second_payload != payloads[filename]
        ):
            raise RuntimeError(f"finalized score-call artifact changed: {filename}")
    root_after = root.lstat()
    if (
        _stat_fingerprint(root_before) != _stat_fingerprint(root_after)
        or {item.name for item in root.iterdir()} != filenames
    ):
        raise RuntimeError(
            "finalized score-call receipt root changed during validation"
        )
    return dict(terminal), expected_sha256


def prepare_strict_gym_child_runtime(
    *, scope: str = "main"
) -> StrictGymChildRuntimeSession:
    """Create the exclusive pre-launch spec for one strict Gym actor."""
    if scope not in {"main", "scorer-only"}:
        raise ValueError("strict Gym child attestation scope is not admitted")
    environment = _strict_environment_value("STRICT_PAIR_ENVIRONMENT")
    if environment not in _RESOURCE_TARGETS:
        raise ValueError("STRICT_PAIR_ENVIRONMENT is not admitted")
    pair_id = _strict_environment_value("PAIR_ID", pattern=_PAIR_ID_RE)
    job_id = _strict_environment_value("STRICT_PAIR_BOUND_JOB_ID", pattern=_JOB_ID_RE)
    if (
        _strict_environment_value("EXPECTED_GYM_GITLINK_COMMIT")
        != STRICT_GYM_GIT_COMMIT
    ):
        raise ValueError("EXPECTED_GYM_GITLINK_COMMIT differs from pinned Gym")
    if _strict_environment_value("EXPECTED_GYM_TREE") != STRICT_GYM_TREE:
        raise ValueError("EXPECTED_GYM_TREE differs from pinned Gym")

    raw_results = _strict_environment_value("RESULTS_DIR")
    results_dir = _require_private_canonical_directory(
        Path(raw_results), name="RESULTS_DIR"
    )
    bootstrap_root, bootstrap_sha256 = _require_sealed_bootstrap_root()
    targets = _target_matrix(environment, STRICT_GYM_ROOT, scope=scope)
    _validate_pinned_gym_root(STRICT_GYM_ROOT, targets, scope=scope)

    results_fd = os.open(results_dir, _DIR_FLAGS)
    try:
        try:
            os.mkdir(STRICT_RESULTS_DIRECTORY_NAME, 0o700, dir_fd=results_fd)
        except FileExistsError as error:
            raise RuntimeError(
                "strict Gym child receipt root already exists"
            ) from error
        os.fsync(results_fd)
    finally:
        os.close(results_fd)
    receipt_root = _require_private_canonical_directory(
        results_dir / STRICT_RESULTS_DIRECTORY_NAME,
        name="strict Gym child receipt root",
    )
    spec = _build_spec(
        environment=environment,
        scope=scope,
        pair_id=pair_id,
        job_id=job_id,
        results_dir=results_dir,
        receipt_root=receipt_root,
        bootstrap_root=bootstrap_root,
        bootstrap_sha256=bootstrap_sha256,
        targets=targets,
    )
    root_fd = os.open(receipt_root, _DIR_FLAGS)
    try:
        _, spec_sha256 = _publish_document(root_fd, "spec.json", spec)
    finally:
        os.close(root_fd)
    return StrictGymChildRuntimeSession(
        environment=environment,
        scope=scope,
        receipt_root=receipt_root,
        spec_path=receipt_root / "spec.json",
        spec_sha256=spec_sha256,
        bootstrap_root=bootstrap_root,
        bootstrap_sha256=bootstrap_sha256,
        spec=spec,
    )
