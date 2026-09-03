"""Fail-closed startup hook for selected strict NeMo-Gym child processes.

This file must remain stdlib-only and the sole entry in its directory.  It is
loaded by Python itself, before the selected ``app.py`` entrypoint.  Any error
in a selected child terminates the process with ``os._exit`` because Python's
normal sitecustomize loader merely prints and suppresses import exceptions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import socket
import stat
import sys
import sysconfig
import threading
from copy import deepcopy
from pathlib import Path

_SYS_GETTRACE = sys.gettrace
_SYS_SETTRACE = sys.settrace

_SPEC_ENV = "NRL_STRICT_GYM_CHILD_SPEC_PATH"
_SPEC_SHA_ENV = "NRL_STRICT_GYM_CHILD_SPEC_SHA256"
_BOOTSTRAP_ENV = "NRL_STRICT_GYM_CHILD_BOOTSTRAP_ROOT"
_BOOTSTRAP_SHA_ENV = "NRL_STRICT_GYM_CHILD_BOOTSTRAP_SHA256"
_DIRECT_RUNNER_ENV = "NRL_STRICT_GYM_DIRECT_RUNNER"
_SPEC_SCHEMA = "nemo-rl-strict-gym-child-spec-v1"
_RECEIPT_SCHEMA = "nemo-rl-strict-gym-child-receipt-v1"
_SCORE_CALL_SCHEMA = "nemo-rl-strict-reasoning-score-call-v1"
_SCORE_CLOSED_SCHEMA = "nemo-rl-strict-reasoning-score-closed-v1"
_HASH_DOMAIN = "sha256-canonical-ascii-json-no-lf-v1"
_GYM_ROOT = Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym")
_VENV_ROOT = Path("/opt/gym_venvs")
_GYM_COMMIT = "354babf7e3554fcd006807c86e80ef476aec9408"
_GYM_TREE = "f24e1ff729c3aed1957df382364c516097218fe0"
_POLICY_PROXY = {
    "config_path": "policy_model",
    "component_dir": str(_GYM_ROOT / "responses_api_models/vllm_model"),
    "entrypoint": "app.py",
    "source_path": str(_GYM_ROOT / "responses_api_models/vllm_model/app.py"),
    "source_sha256": (
        "730cd0e60135bf7981d85e0d1e79933f378fa86867983d9c409922638eed9795"
    ),
}

_GYM_SOURCES = {
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

_COMMON_DISTRIBUTIONS = {
    "nemo-gym": "0.5.0rc0",
    "openai": "2.6.1",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
}

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

_SCORER_RAY_VERSION = "2.56.1"
_SCORER_RAY_ARCHIVE_ROOT = Path("/root/.cache/uv/archive-v0/h3_h3tkzE2mqPI10/ray")
_SCORER_RAY_SOURCES = {
    "__init__.py": {
        "size": 7559,
        "sha256": "a3845ca44927ca669778ca7489a522f4f5138e8303c7efe6818a193e1b15d376",
    },
    "_version.py": {
        "size": 199,
        "sha256": "68b02abec4e1338c8bb21772687e0d05005965159483aae42c44c01e60e203ab",
    },
}

_STATIC_TARGETS = {
    "reasoning_gym": {
        "environment": "reasoning_gym",
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
            **_COMMON_DISTRIBUTIONS,
            "ray": _SCORER_RAY_VERSION,
            "reasoning-gym": "0.1.25",
        },
        "module_versions": {
            "nemo_gym": "0.5.1",
            "ray": _SCORER_RAY_VERSION,
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
        "resource_config_path": None,
        "receipt_filename": "resource.json",
    },
    "citation_format": {
        "environment": "citation",
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
        "distribution_versions": _COMMON_DISTRIBUTIONS,
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
        "resource_config_path": None,
        "receipt_filename": "resource.json",
    },
    "freeform_formatting": {
        "environment": "freeform",
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
        "distribution_versions": _COMMON_DISTRIBUTIONS,
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
        "resource_config_path": None,
        "receipt_filename": "resource.json",
    },
    "reasoning_gym_simple_agent": {
        "environment": "reasoning_gym",
        "role": "simple_agent",
        "server_type": "responses_api_agents",
        "server_name": "simple_agent",
        "component_relative": "responses_api_agents/simple_agent",
        "entrypoint": "app.py",
        "source_relative": "responses_api_agents/simple_agent/app.py",
        "source_sha256": (
            "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a"
        ),
        "config_relative": (
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml"
        ),
        "config_sha256": (
            "bdbb459a4a920bc47cf84b1d7dc30aeaa9be35cf0dfac09c77879e45b62a52ab"
        ),
        "requirements_relative": "responses_api_agents/simple_agent/requirements.txt",
        "requirements_sha256": (
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        "distribution_versions": _COMMON_DISTRIBUTIONS,
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
        "resource_config_path": "reasoning_gym",
        "receipt_filename": "simple_agent.json",
    },
    "citation_format_simple_agent": {
        "environment": "citation",
        "role": "simple_agent",
        "server_type": "responses_api_agents",
        "server_name": "simple_agent",
        "component_relative": "responses_api_agents/simple_agent",
        "entrypoint": "app.py",
        "source_relative": "responses_api_agents/simple_agent/app.py",
        "source_sha256": (
            "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a"
        ),
        "config_relative": (
            "resources_servers/format_verification/configs/citation_format.yaml"
        ),
        "config_sha256": (
            "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
        ),
        "requirements_relative": "responses_api_agents/simple_agent/requirements.txt",
        "requirements_sha256": (
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        "distribution_versions": _COMMON_DISTRIBUTIONS,
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
        "resource_config_path": "citation_format",
        "receipt_filename": "simple_agent.json",
    },
    "freeform_formatting_simple_agent": {
        "environment": "freeform",
        "role": "simple_agent",
        "server_type": "responses_api_agents",
        "server_name": "simple_agent",
        "component_relative": "responses_api_agents/simple_agent",
        "entrypoint": "app.py",
        "source_relative": "responses_api_agents/simple_agent/app.py",
        "source_sha256": (
            "ea8179439c54962fdd48de3b0f64caed61049848a7801f1a63d0c1d0fd0ab97a"
        ),
        "config_relative": (
            "resources_servers/format_verification/configs/freeform_formatting.yaml"
        ),
        "config_sha256": (
            "92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa"
        ),
        "requirements_relative": "responses_api_agents/simple_agent/requirements.txt",
        "requirements_sha256": (
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        "distribution_versions": _COMMON_DISTRIBUTIONS,
        "module_versions": {"nemo_gym": "0.5.1"},
        "scorer": None,
        "resource_config_path": "freeform_formatting",
        "receipt_filename": "simple_agent.json",
    },
}

_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIR_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
_CREATE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _fingerprint(info):
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


def _read_file(path, *, expected_mode=None, expected_nlink=1, maximum=1 << 20):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlink file")
    fd = os.open(path, _READ_FLAGS)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != expected_nlink:
            raise ValueError("not expected regular inode")
        if expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode:
            raise ValueError("wrong file mode")
        if before.st_uid != os.geteuid():
            raise ValueError("wrong file owner")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ValueError("file too large")
        after = os.fstat(fd)
        path_info = path.lstat()
        if _fingerprint(before) != _fingerprint(after) or _fingerprint(
            after
        ) != _fingerprint(path_info):
            raise ValueError("file changed while read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sha_file(path, *, expected_nlink=1):
    return hashlib.sha256(
        _read_file(path, expected_nlink=expected_nlink, maximum=1 << 24)
    ).hexdigest()


def _package_tree_identity(package_root):
    """Hash every wheel-owned package byte before importing it.

    ``uv`` installs the package directory as one symlink into its archive
    cache.  We bind both that lexical venv path and the image-specific resolved
    cache path and reject nested links, special files, and all bytecode. A
    sourceless legacy ``.pyc`` remains importable even when cache writes are
    disabled, so bytecode must be absent rather than ignored.
    """
    lexical_root = Path(package_root)
    resolved_root = lexical_root.resolve(strict=True)
    entries = []
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
            payload = _read_file(path, maximum=1 << 24)
            relative = path.relative_to(resolved_root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
            total_bytes += len(payload)
    entries.sort(key=lambda item: item["path"])
    return (
        resolved_root,
        hashlib.sha256(_canonical(entries)).hexdigest(),
        len(entries),
        total_bytes,
    )


def _load_spec():
    spec_path_text = os.environ.get(_SPEC_ENV)
    expected_sha = os.environ.get(_SPEC_SHA_ENV)
    results_text = os.environ.get("RESULTS_DIR")
    if (
        not spec_path_text
        or not results_text
        or not expected_sha
        or not _SHA_RE.fullmatch(expected_sha)
    ):
        raise ValueError("missing strict spec environment")
    results = Path(results_text)
    spec_path = Path(spec_path_text)
    expected_path = results / "strict_gym_child_runtime" / "spec.json"
    if (
        not results.is_absolute()
        or results.resolve(strict=True) != results
        or spec_path != expected_path
        or spec_path.parent.resolve(strict=True) != spec_path.parent
    ):
        raise ValueError("non-canonical strict spec path")
    root_info = spec_path.parent.stat()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_IMODE(root_info.st_mode) != 0o700
        or root_info.st_uid != os.geteuid()
    ):
        raise ValueError("invalid strict receipt root")
    payload = _read_file(spec_path, expected_mode=0o400)
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("strict spec hash mismatch")
    value = json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise ValueError("strict spec is not canonical JSON")
    return value, spec_path, expected_sha


def _validate_bootstrap(spec):
    root_text = os.environ.get(_BOOTSTRAP_ENV)
    sha = os.environ.get(_BOOTSTRAP_SHA_ENV)
    if not root_text or not sha or not _SHA_RE.fullmatch(sha):
        raise ValueError("missing bootstrap identity")
    root = Path(root_text)
    source = Path(__file__)
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or source.resolve(strict=True) != source
        or source.parent != root
        or sorted(item.name for item in root.iterdir()) != ["sitecustomize.py"]
        or _sha_file(source) != sha
        or spec.get("bootstrap")
        != {"root": str(root), "filename": "sitecustomize.py", "sha256": sha}
    ):
        raise ValueError("bootstrap identity mismatch")
    root_info = root.stat()
    source_info = source.stat()
    if (
        stat.S_IMODE(root_info.st_mode) not in {0o500, 0o555}
        or stat.S_IMODE(source_info.st_mode) not in {0o400, 0o444}
        or root_info.st_uid != os.geteuid()
        or source_info.st_uid != os.geteuid()
    ):
        raise ValueError(
            "bootstrap must be owned, readable, executable, and non-writable"
        )


def _validate_spec(spec):
    expected_keys = {
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
    }
    if set(spec) != expected_keys:
        raise ValueError("strict spec keyset mismatch")
    if spec["schema"] != _SPEC_SCHEMA or spec["hash_domain"] != _HASH_DOMAIN:
        raise ValueError("strict spec schema mismatch")
    if spec["environment"] != os.environ.get("STRICT_PAIR_ENVIRONMENT"):
        raise ValueError("strict environment mismatch")
    if spec["pair_id"] != os.environ.get("PAIR_ID") or spec["job_id"] != os.environ.get(
        "STRICT_PAIR_BOUND_JOB_ID"
    ):
        raise ValueError("strict job identity mismatch")
    if spec["scope"] not in {"main", "scorer-only"}:
        raise ValueError("strict scope mismatch")
    if spec["gym"] != {
        "git_commit": _GYM_COMMIT,
        "tree": _GYM_TREE,
        "root": str(_GYM_ROOT),
        "venv_root": str(_VENV_ROOT),
        "sources": _GYM_SOURCES,
    }:
        raise ValueError("strict Gym identity mismatch")
    if (
        os.environ.get("EXPECTED_GYM_GITLINK_COMMIT") != _GYM_COMMIT
        or os.environ.get("EXPECTED_GYM_TREE") != _GYM_TREE
    ):
        raise ValueError("strict Gym boundary environment mismatch")
    if spec["results_dir"] != os.environ.get("RESULTS_DIR"):
        raise ValueError("strict results identity mismatch")
    expected_root = str(Path(spec["results_dir"]) / "strict_gym_child_runtime")
    if spec["receipt_root"] != expected_root:
        raise ValueError("strict receipt root mismatch")
    if not isinstance(spec["targets"], list):
        raise ValueError("strict targets are not a list")
    expected_count = 2 if spec["scope"] == "main" else 1
    if len(spec["targets"]) != expected_count:
        raise ValueError("strict target count mismatch")
    configs = [
        target.get("config_path")
        for target in spec["targets"]
        if isinstance(target, dict)
    ]
    resource_config = {
        "reasoning_gym": "reasoning_gym",
        "citation": "citation_format",
        "freeform": "freeform_formatting",
    }[spec["environment"]]
    expected_configs = [resource_config]
    if spec["scope"] == "main":
        expected_configs.append(
            {
                "reasoning_gym": "reasoning_gym_simple_agent",
                "citation": "citation_format_simple_agent",
                "freeform": "freeform_formatting_simple_agent",
            }[spec["environment"]]
        )
    if configs != expected_configs:
        raise ValueError("strict target selection mismatch")


def _validate_target(spec, target):
    config_path = target.get("config_path")
    static_expected = _STATIC_TARGETS.get(config_path)
    expected = dict(static_expected) if static_expected is not None else None
    if expected is None or expected["environment"] != spec["environment"]:
        raise ValueError("unknown strict target")
    if spec["scope"] == "scorer-only" and config_path == "reasoning_gym":
        expected["config_relative"] = (
            "resources_servers/reasoning_gym/configs/resources_only.yaml"
        )
        expected["config_sha256"] = (
            "e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68"
        )
    for key, value in expected.items():
        if key != "environment" and target.get(key) != value:
            raise ValueError(f"strict target field differs: {key}")
    component = _GYM_ROOT / expected["component_relative"]
    venv = _VENV_ROOT / expected["server_type"] / expected["server_name"] / ".venv"
    dynamic = {
        "component_dir": str(component),
        "source_path": str(_GYM_ROOT / expected["source_relative"]),
        "config_path_source": str(_GYM_ROOT / expected["config_relative"]),
        "requirements_path": str(_GYM_ROOT / expected["requirements_relative"]),
        "venv": str(venv),
        "interpreter": str(venv / "bin" / "python"),
    }
    for key, value in dynamic.items():
        if target.get(key) != value:
            raise ValueError(f"strict target path differs: {key}")
    for path_key, hash_key in (
        ("source_path", "source_sha256"),
        ("config_path_source", "config_sha256"),
        ("requirements_path", "requirements_sha256"),
    ):
        if _sha_file(target[path_key]) != target[hash_key]:
            raise ValueError(f"strict target source differs: {path_key}")
    return expected


def _proc_stat(pid):
    payload = Path(f"/proc/{pid}/stat").read_bytes()
    closing = payload.rfind(b") ")
    fields = payload[closing + 2 :].split()
    if closing < 0 or len(fields) <= 19:
        raise ValueError("malformed proc stat")
    return int(fields[1]), int(fields[19])


def _proc_argv(pid):
    payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    if not payload.endswith(b"\0") or len(payload) > 65536:
        raise ValueError("malformed proc argv")
    return [part.decode("utf-8", "strict") for part in payload[:-1].split(b"\0")]


def _publish(root, filename, document):
    payload = _canonical(document)
    root_fd = os.open(root, _DIR_FLAGS)
    try:
        info = os.fstat(root_fd)
        if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
            raise ValueError("invalid receipt root at publication")
        fd = os.open(filename, _CREATE_FLAGS, stat.S_IRUSR, dir_fd=root_fd)
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("short receipt write")
                offset += written
            os.fchmod(fd, stat.S_IRUSR)
            os.fsync(fd)
            final = os.fstat(fd)
            if (
                not stat.S_ISREG(final.st_mode)
                or stat.S_IMODE(final.st_mode) != 0o400
                or final.st_uid != os.geteuid()
                or final.st_nlink != 1
                or final.st_size != len(payload)
            ):
                raise ValueError("invalid published receipt inode")
        finally:
            os.close(fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return hashlib.sha256(payload).hexdigest()


def _install_reasoning_score_evidence(
    reasoning_gym,
    *,
    spec,
    spec_sha,
    process,
    frozen_score_fn,
    frozen_normalize_fn,
    frozen_score_code=None,
    frozen_normalize_code=None,
    frozen_valid_roles=None,
    frozen_scorer_globals=None,
):
    score_function = getattr(frozen_score_fn, "__func__", frozen_score_fn)
    scorer_instance = getattr(frozen_score_fn, "__self__", None)
    if frozen_score_code is None:
        frozen_score_code = score_function.__code__
    if frozen_normalize_code is None:
        frozen_normalize_code = frozen_normalize_fn.__code__

    def validate_frozen_scorer_semantics():
        if (
            score_function.__code__ is not frozen_score_code
            or frozen_normalize_fn.__code__ is not frozen_normalize_code
            or (
                scorer_instance is not None
                and scorer_instance._normalize_answer is not frozen_normalize_fn
            )
            or (
                frozen_scorer_globals is not None
                and frozen_scorer_globals.get("VALID_ROLES") is not frozen_valid_roles
            )
        ):
            raise RuntimeError("authenticated reasoning scorer semantics changed")

    validate_frozen_scorer_semantics()
    counter = 0
    closed = False
    call_refs = []
    lock = threading.Lock()

    def publish(sequence, task_name, answer_sha256, entry_sha256, outcome):
        document = {
            "schema": _SCORE_CALL_SCHEMA,
            "hash_domain": _HASH_DOMAIN,
            "environment": spec["environment"],
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec_sha256": spec_sha,
            "process": {
                "pid": process["pid"],
                "start_ticks": process["start_ticks"],
            },
            "sequence": sequence,
            "task_name": task_name,
            "input": {
                "answer_sha256": answer_sha256,
                "entry_sha256": entry_sha256,
            },
            "outcome": outcome,
        }
        filename = f"reasoning-score-call-{sequence:08d}.json"
        digest = _publish(
            spec["receipt_root"],
            filename,
            document,
        )
        return {
            "sequence": sequence,
            "path": str(Path(spec["receipt_root"]) / filename),
            "sha256": digest,
            "schema": _SCORE_CALL_SCHEMA,
        }

    def close_after_four():
        nonlocal closed
        if counter != 4:
            return
        if len(call_refs) != 4 or [item["sequence"] for item in call_refs] != [
            1,
            2,
            3,
            4,
        ]:
            os._exit(79)
        closed_document = {
            "schema": _SCORE_CLOSED_SCHEMA,
            "hash_domain": _HASH_DOMAIN,
            "environment": spec["environment"],
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec_sha256": spec_sha,
            "process": {
                "pid": process["pid"],
                "start_ticks": process["start_ticks"],
            },
            "call_count": 4,
            "calls": list(call_refs),
        }
        try:
            _publish(
                spec["receipt_root"], "reasoning-score-closed.json", closed_document
            )
        except BaseException:
            os._exit(79)
        closed = True

    def wrapped_get_score_answer_fn(task_name):
        with lock:
            if (
                closed
                or counter >= 4
                or type(task_name) is not str
                or task_name != "knights_knaves"
            ):
                os._exit(80)

        def score_with_evidence(*args, **kwargs):
            nonlocal counter
            with lock:
                if closed or counter >= 4:
                    os._exit(80)
                counter += 1
                sequence = counter
                answer_sha256 = hashlib.sha256(_canonical(None)).hexdigest()
                entry_sha256 = hashlib.sha256(_canonical(None)).hexdigest()
                try:
                    if args or set(kwargs) != {"answer", "entry"}:
                        raise TypeError("pinned reasoning scorer invocation differs")
                    answer_snapshot = deepcopy(kwargs["answer"])
                    entry_snapshot = deepcopy(kwargs["entry"])
                    answer_sha256 = hashlib.sha256(
                        _canonical(answer_snapshot)
                    ).hexdigest()
                    entry_sha256 = hashlib.sha256(
                        _canonical(entry_snapshot)
                    ).hexdigest()
                    if type(kwargs["answer"]) is not str:
                        raise TypeError(
                            "pinned reasoning answer is not an exact string"
                        )
                    if (
                        type(kwargs["entry"]) is not dict
                        or "answer" not in kwargs["entry"]
                        or type(kwargs["entry"]["answer"]) is not str
                    ):
                        raise TypeError(
                            "pinned reasoning entry answer is not an exact string"
                        )
                    validate_frozen_scorer_semantics()
                    oracle_assignments = frozen_normalize_fn(kwargs["entry"]["answer"])
                    answer_assignments = frozen_normalize_fn(kwargs["answer"])
                    validate_frozen_scorer_semantics()
                    for normalized in (oracle_assignments, answer_assignments):
                        if type(normalized) is not set or any(
                            type(item) is not tuple
                            or len(item) != 2
                            or any(type(part) is not str for part in item)
                            for item in normalized
                        ):
                            raise TypeError(
                                "pinned reasoning normalization result differs"
                            )
                    expected_result = 0.0
                    if kwargs["answer"]:
                        if oracle_assignments == answer_assignments:
                            expected_result = 1.0
                        elif len(oracle_assignments) == len(answer_assignments):
                            matching = len(
                                oracle_assignments.intersection(answer_assignments)
                            )
                            if matching > 0:
                                expected_result = 0.3 + 0.7 * matching / len(
                                    oracle_assignments
                                )
                    if _SYS_GETTRACE() is not None:
                        raise RuntimeError(
                            "reasoning scorer inherited an untrusted trace function"
                        )
                    scorer_exception = False

                    def trace_score(frame, event, arg):
                        nonlocal scorer_exception
                        del arg
                        if frame.f_code is frozen_score_code:
                            if event == "exception":
                                scorer_exception = True
                            return trace_score
                        return None

                    _SYS_SETTRACE(trace_score)
                    try:
                        if _SYS_GETTRACE() is not trace_score:
                            raise RuntimeError(
                                "reasoning scorer trace function was not installed"
                            )
                        result = frozen_score_fn(
                            answer=kwargs["answer"], entry=kwargs["entry"]
                        )
                    finally:
                        _SYS_SETTRACE(None)
                    if _SYS_GETTRACE() is not None:
                        os._exit(79)
                    if scorer_exception:
                        raise RuntimeError(
                            "pinned reasoning scorer raised a caught exception"
                        )
                    validate_frozen_scorer_semantics()
                    if (
                        type(result) is not float
                        or not math.isfinite(result)
                        or not 0.0 <= result <= 1.0
                        or (result == 0.0 and math.copysign(1.0, result) < 0.0)
                    ):
                        raise ValueError(
                            "reasoning score is not an admitted exact float"
                        )
                    if result != expected_result:
                        raise ValueError(
                            "pinned reasoning scorer returned a result inconsistent "
                            "with authenticated normalization"
                        )
                    call_ref = publish(
                        sequence,
                        task_name,
                        answer_sha256,
                        entry_sha256,
                        {"kind": "returned", "float_result": result},
                    )
                except BaseException as error:
                    try:
                        call_ref = publish(
                            sequence,
                            task_name,
                            answer_sha256,
                            entry_sha256,
                            {
                                "kind": "exception",
                                "phase": "score",
                                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                            },
                        )
                    except BaseException:
                        os._exit(79)
                    call_refs.append(call_ref)
                    close_after_four()
                    raise
                call_refs.append(call_ref)
                close_after_four()
                return result

        return score_with_evidence

    reasoning_gym.get_score_answer_fn = wrapped_get_score_answer_fn


def _validate_scorer_no_site_pth_inventory(purelib):
    pth_files = sorted(purelib.glob("*.pth"), key=lambda item: item.name)
    purelib_info = purelib.stat()
    if (
        not stat.S_ISDIR(purelib_info.st_mode)
        or stat.S_IMODE(purelib_info.st_mode) != 0o755
        or purelib_info.st_uid != os.geteuid()
        or purelib_info.st_gid != os.getegid()
    ):
        raise ValueError("selected scorer purelib identity differs")
    if not pth_files:
        return
    if [item.name for item in pth_files] != list(_SCORER_NO_SITE_PTH_FILES):
        raise ValueError("selected scorer venv .pth inventory differs")
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
                or _sha_file(pth_file) != identity["sha256"]
                or _fingerprint(pth_file.lstat()) != _fingerprint(before_link)
            ):
                raise ValueError("selected scorer venv .pth identity differs")
            continue

        if not stat.S_ISLNK(before_link.st_mode):
            raise ValueError("selected scorer venv .pth identity differs")
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
            raise ValueError("selected scorer venv .pth identity differs")
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
            or _sha_file(resolved_target) != identity["sha256"]
            or _fingerprint(resolved_target.lstat()) != _fingerprint(before_target)
            or _fingerprint(pth_file.lstat()) != _fingerprint(before_link)
            or os.readlink(pth_file) != link_target
        ):
            raise ValueError("selected scorer venv .pth identity differs")


def _seal_direct_runner_sys_path(isolated_path, purelib, cwd, nemo_gym):
    """Remove the pinned Gym import's deliberate cwd path insertion."""
    isolated = list(isolated_path)
    safe_path = [*isolated, str(purelib), str(_GYM_ROOT)]
    expected_after_gym_import = [*safe_path[:-1], str(cwd), str(_GYM_ROOT)]
    if sys.path != expected_after_gym_import:
        raise ValueError(
            "nemo_gym changed the isolated scorer import path unexpectedly"
        )
    sys.path[:] = safe_path

    def locked_augment_sys_path():
        if sys.path != safe_path:
            raise ValueError("isolated scorer import path changed after authentication")

    nemo_gym._augment_sys_path = locked_augment_sys_path
    locked_augment_sys_path()
    return tuple(safe_path)


def _authenticate_direct_runner_ray_inputs(purelib):
    """Bind the image-owned Ray import that mutates ``sys.path``."""
    ray_root = purelib / "ray"
    thirdparty = ray_root / "thirdparty_files"
    if (
        ray_root.is_symlink()
        or not ray_root.is_dir()
        or ray_root.resolve(strict=True) != ray_root
        or thirdparty.is_symlink()
        or not thirdparty.is_dir()
        or thirdparty.resolve(strict=True) != thirdparty
        or importlib.metadata.version("ray") != _SCORER_RAY_VERSION
    ):
        raise ValueError("selected scorer Ray installation differs")
    source_fingerprints = {}
    for relative, identity in _SCORER_RAY_SOURCES.items():
        source = ray_root / relative
        before_link = source.lstat()
        expected_target = _SCORER_RAY_ARCHIVE_ROOT / relative
        link_target = os.readlink(source)
        if (
            not stat.S_ISLNK(before_link.st_mode)
            or not Path(link_target).is_absolute()
            or ".." in Path(link_target).parts
            or Path(link_target) != expected_target
            or (sys.platform == "linux" and stat.S_IMODE(before_link.st_mode) != 0o777)
            or before_link.st_uid != os.geteuid()
            or before_link.st_gid != os.getegid()
            or before_link.st_nlink != 1
            or before_link.st_size != len(os.fsencode(link_target))
        ):
            raise ValueError("selected scorer Ray source link differs")
        resolved = source.resolve(strict=True)
        before_target = resolved.lstat()
        if (
            resolved != expected_target
            or not stat.S_ISREG(before_target.st_mode)
            or stat.S_IMODE(before_target.st_mode) != 0o600
            or before_target.st_uid != os.geteuid()
            or before_target.st_gid != os.getegid()
            or before_target.st_nlink != 2
            or before_target.st_size != identity["size"]
            or _sha_file(resolved, expected_nlink=2) != identity["sha256"]
            or _fingerprint(source.lstat()) != _fingerprint(before_link)
            or _fingerprint(resolved.lstat()) != _fingerprint(before_target)
            or os.readlink(source) != link_target
        ):
            raise ValueError("selected scorer Ray source identity differs")
        source_fingerprints[relative] = {
            "link": _fingerprint(before_link),
            "target": _fingerprint(before_target),
        }
    return {
        "root": ray_root,
        "thirdparty": thirdparty,
        "sources": source_fingerprints,
    }


def _seal_direct_runner_ray_sys_path(safe_path, purelib, ray_inputs):
    """Authenticate and remove Ray's exact bundled third-party path prefix."""
    if _authenticate_direct_runner_ray_inputs(purelib) != ray_inputs:
        raise ValueError("selected scorer Ray inputs changed during import")
    expected = [str(ray_inputs["thirdparty"]), *safe_path]
    ray_module = sys.modules.get("ray")
    ray_origin = getattr(ray_module, "__file__", None)
    if (
        sys.path != expected
        or type(ray_origin) is not str
        or Path(ray_origin).resolve(strict=True)
        != _SCORER_RAY_ARCHIVE_ROOT / "__init__.py"
        or getattr(ray_module, "__version__", None) != _SCORER_RAY_VERSION
    ):
        raise ValueError("Ray changed the isolated scorer import path unexpectedly")
    sys.path[:] = safe_path


def _attest():
    spec, _spec_path, spec_sha = _load_spec()
    _validate_spec(spec)
    _validate_bootstrap(spec)
    direct_runner = os.environ.get(_DIRECT_RUNNER_ENV) == "1"
    config_path = os.environ.get("NEMO_GYM_CONFIG_PATH")
    matching = [
        target for target in spec["targets"] if target.get("config_path") == config_path
    ]
    if not matching:
        # The main training RunHelper also launches the authenticated policy
        # proxy.  It must inherit the hook path so its environment cannot
        # create a bypass, but it is not a verifier and must publish nothing.
        if (
            direct_runner
            or spec["scope"] != "main"
            or config_path != _POLICY_PROXY["config_path"]
        ):
            raise ValueError("unexpected non-target Gym child")
        proxy_cwd = Path.cwd().resolve(strict=True)
        bootstrap = Path(os.environ[_BOOTSTRAP_ENV])
        if (
            proxy_cwd != Path(_POLICY_PROXY["component_dir"])
            or sys.argv != [_POLICY_PROXY["entrypoint"]]
            or _proc_argv(os.getpid()) != ["python", _POLICY_PROXY["entrypoint"]]
            or os.environ.get("PYTHONPATH", "").split(os.pathsep)
            != [str(proxy_cwd), str(bootstrap)]
            or _sha_file(_POLICY_PROXY["source_path"]) != _POLICY_PROXY["source_sha256"]
        ):
            raise ValueError("policy proxy no-op identity differs")
        return
    if len(matching) != 1:
        raise ValueError("ambiguous selected strict target")
    target = matching[0]
    expected = _validate_target(spec, target)

    cwd = Path.cwd().resolve(strict=True)
    bootstrap = Path(os.environ[_BOOTSTRAP_ENV])
    if cwd != Path(target["component_dir"]):
        raise ValueError("child cwd is not the selected component")
    proc_exe = os.readlink("/proc/self/exe")
    if Path(proc_exe) != Path(target["interpreter"]).resolve(strict=True):
        raise ValueError("child /proc executable differs from selected interpreter")
    if direct_runner:
        isolated_sys_path = tuple(sys.path)
        bootstrap_source = bootstrap / "sitecustomize.py"
        app_path = Path(target["source_path"])
        expected_proc_argv = [
            target["interpreter"],
            "-I",
            "-S",
            "-B",
            str(bootstrap_source),
            str(app_path),
        ]
        if (
            spec["scope"] != "scorer-only"
            or sys.argv != [str(bootstrap_source), str(app_path)]
            or _proc_argv(os.getpid()) != expected_proc_argv
            or sys.executable != target["interpreter"]
            or "site" in sys.modules
            or "sitecustomize" in sys.modules
            or not sys.flags.isolated
            or not sys.flags.ignore_environment
            or not sys.flags.no_site
            or not sys.flags.no_user_site
            or not sys.flags.safe_path
            or not sys.dont_write_bytecode
            or sys.pycache_prefix is not None
            or "PYTHONPATH" in os.environ
        ):
            raise ValueError("isolated scorer runner startup differs")
        forbidden_roots = {
            str(cwd),
            str(bootstrap),
            str(_GYM_ROOT),
            target["venv"],
        }
        if any(item in forbidden_roots for item in sys.path):
            raise ValueError("isolated scorer runner inherited an executable root")
        purelib = Path(target["venv"]) / "lib/python3.13/site-packages"
        if (
            purelib.is_symlink()
            or purelib.resolve(strict=True) != purelib
            or not purelib.is_dir()
        ):
            raise ValueError("selected scorer purelib is not canonical")
        _validate_scorer_no_site_pth_inventory(purelib)
        # ``-S`` prevents all .pth/sitecustomize processing. Append the
        # selected venv package directory and authenticated Gym checkout only
        # after the stdlib-only checks.  Keeping both behind the isolated
        # interpreter's stdlib prevents either tree from shadowing Python;
        # keeping purelib first supports both a materialized nemo_gym install
        # and the pinned editable-checkout layout without executing .pth code.
        sys.path.extend([str(purelib), str(_GYM_ROOT)])
        sys.prefix = target["venv"]
        sys.exec_prefix = target["venv"]
    else:
        if os.environ.get("NEMO_GYM_EXTRA_ROOTS") not in (
            None,
            "",
        ) or os.environ.get("PYTHONHOME") not in (None, ""):
            raise ValueError("untrusted Python/Gym root injection")
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if python_path != [str(cwd), str(bootstrap)]:
            raise ValueError("child PYTHONPATH is not the exclusive pinned path")
        if (cwd / "sitecustomize.py").exists() or (cwd / "sitecustomize").exists():
            raise ValueError("component shadows strict sitecustomize")
        if sys.argv != [target["entrypoint"]] or _proc_argv(os.getpid()) != [
            "python",
            target["entrypoint"],
        ]:
            raise ValueError("child argv differs from pinned RunHelper command")
        if (
            sys.executable != target["interpreter"]
            or sys.prefix != target["venv"]
            or sys.base_prefix == sys.prefix
        ):
            raise ValueError(
                "child did not use the selected canonical venv interpreter"
            )
        purelib = Path(sysconfig.get_path("purelib"))
        if purelib.parent.parent.parent != Path(target["venv"]):
            raise ValueError("child purelib is outside selected venv")
        if list(purelib.glob("*.pth")):
            raise ValueError("selected child venv contains pre-sitecustomize .pth code")
        disabled_pycache = bootstrap / "__pycache_disabled__"
        if (
            os.environ.get("PYTHONNOUSERSITE") != "1"
            or os.environ.get("PYTHONSAFEPATH") != "1"
            or os.environ.get("PYTHONPYCACHEPREFIX") != str(disabled_pycache)
            or disabled_pycache.exists()
            or not sys.dont_write_bytecode
            or not sys.flags.no_user_site
            or not sys.flags.safe_path
            or sys.pycache_prefix != str(disabled_pycache)
        ):
            raise ValueError("child Python startup isolation differs")

    for relative, digest in _GYM_SOURCES.items():
        if _sha_file(_GYM_ROOT / relative) != digest:
            raise ValueError(f"pinned Gym source differs: {relative}")
    gym_spec = importlib.util.find_spec("nemo_gym")
    if (
        gym_spec is None
        or gym_spec.origin is None
        or Path(gym_spec.origin).resolve(strict=True)
        != _GYM_ROOT / "nemo_gym/__init__.py"
    ):
        raise ValueError("nemo_gym import origin is not the authenticated root")
    import nemo_gym

    if direct_runner:
        direct_safe_path = _seal_direct_runner_sys_path(
            isolated_sys_path, purelib, cwd, nemo_gym
        )
        direct_ray_inputs = _authenticate_direct_runner_ray_inputs(purelib)
    from nemo_gym.cli.setup_command import get_venv_path
    from nemo_gym.global_config import (
        get_first_server_config_dict,
        get_global_config_dict,
    )

    if direct_runner:
        _seal_direct_runner_ray_sys_path(direct_safe_path, purelib, direct_ray_inputs)

    if Path(nemo_gym.PARENT_DIR).resolve(strict=True) != _GYM_ROOT:
        raise ValueError("nemo_gym.PARENT_DIR is not authenticated")
    global_config = get_global_config_dict()
    if (
        global_config.get("skip_venv_if_present") is not True
        or global_config.get("dry_run") is not False
    ):
        raise ValueError("strict child requires skip-existing live server semantics")
    if Path(global_config["uv_venv_dir"]) != _VENV_ROOT:
        raise ValueError("strict child venv root differs")
    if Path(get_venv_path(cwd, global_config)) != Path(target["venv"]):
        raise ValueError("pinned get_venv_path differs from selected interpreter")
    top = global_config[config_path]
    if list(top.keys()) != [expected["server_type"]]:
        raise ValueError("selected server type differs")
    middle = top[expected["server_type"]]
    if list(middle.keys()) != [expected["server_name"]]:
        raise ValueError("selected server name differs")
    server_config = get_first_server_config_dict(global_config, config_path)
    host = server_config.get("host")
    port = server_config.get("port")
    num_workers = server_config.get("num_workers")
    if server_config.get("entrypoint") != target["entrypoint"] or host != "127.0.0.1":
        raise ValueError("selected server endpoint differs")
    if type(port) is not int or not 5000 <= port <= 5999:
        raise ValueError("selected server port differs")
    if num_workers is not None and (type(num_workers) is not int or num_workers != 1):
        raise ValueError("strict attestation requires one uvicorn process")
    if expected["resource_config_path"] is not None:
        resource_ref = server_config.get("resources_server")
        if (
            resource_ref is None
            or resource_ref.get("type") != "resources_servers"
            or resource_ref.get("name") != expected["resource_config_path"]
        ):
            raise ValueError("SimpleAgent resource selection differs")

    distributions = {
        name: importlib.metadata.version(name)
        for name in sorted(target["distribution_versions"])
    }
    if distributions != target["distribution_versions"]:
        raise ValueError("installed distribution metadata differs")
    module_versions = {"nemo_gym": nemo_gym.__version__}
    if target["scorer"] is not None:
        import ray

        module_versions["ray"] = ray.__version__
    scorer = None
    reasoning_gym = None
    frozen_score_fn = None
    frozen_normalize_fn = None
    frozen_score_code = None
    frozen_normalize_code = None
    frozen_valid_roles = None
    frozen_scorer_globals = None
    if target["scorer"] is not None:
        scorer_pin = target["scorer"]
        package_root = purelib / "reasoning_gym"
        resolved_package, *package_tree = _package_tree_identity(package_root)
        if tuple(package_tree) != (
            scorer_pin["package_tree_sha256"],
            scorer_pin["package_file_count"],
            scorer_pin["package_total_bytes"],
        ):
            raise ValueError("reasoning-gym package tree differs from pinned wheel")
        import reasoning_gym as imported_reasoning_gym
        import reasoning_gym.factory as reasoning_factory
        import reasoning_gym.logic.knights_knaves as knights_knaves

        reasoning_gym = imported_reasoning_gym
        module_versions["reasoning_gym"] = reasoning_gym.__version__
        module_origin = Path(reasoning_gym.__file__)
        resolver_origin = Path(reasoning_factory.__file__)
        scorer_origin = Path(knights_knaves.__file__)
        expected_module_origin = (
            purelib / scorer_pin["module_origin_relative_to_purelib"]
        )
        expected_resolver_origin = (
            purelib / scorer_pin["resolver_origin_relative_to_purelib"]
        )
        expected_scorer_origin = purelib / scorer_pin["origin_relative_to_purelib"]
        if (
            module_origin != expected_module_origin
            or resolver_origin != expected_resolver_origin
            or scorer_origin != expected_scorer_origin
            or module_origin.resolve(strict=True) != resolved_package / "__init__.py"
            or resolver_origin.resolve(strict=True) != resolved_package / "factory.py"
            or scorer_origin.resolve(strict=True)
            != resolved_package / "logic/knights_knaves.py"
            or _sha_file(module_origin) != scorer_pin["module_sha256"]
            or _sha_file(resolver_origin) != scorer_pin["resolver_sha256"]
            or _sha_file(scorer_origin) != scorer_pin["sha256"]
            or reasoning_gym.get_score_answer_fn
            is not reasoning_factory.get_score_answer_fn
        ):
            raise ValueError("reasoning scorer package origins/hashes differ")
        frozen_score_fn = reasoning_gym.get_score_answer_fn("knights_knaves")
        callable_name = f"{frozen_score_fn.__module__}.{frozen_score_fn.__qualname__}"
        if (
            callable_name != scorer_pin["callable"]
            or getattr(frozen_score_fn, "__func__", None)
            is not knights_knaves.KnightsKnavesDataset.score_answer
            or type(getattr(frozen_score_fn, "__self__", None))
            is not knights_knaves.KnightsKnavesDataset
            or frozen_score_fn.__self__._normalize_answer
            is not knights_knaves.KnightsKnavesDataset._normalize_answer
        ):
            raise ValueError("reasoning scorer callable differs")
        frozen_normalize_fn = frozen_score_fn.__self__._normalize_answer
        frozen_score_code = frozen_score_fn.__func__.__code__
        frozen_normalize_code = frozen_normalize_fn.__code__
        frozen_valid_roles = frozenset(
            {
                "altruist",
                "angel",
                "devil",
                "egoist",
                "fool",
                "hero",
                "knave",
                "knight",
                "laggard",
                "pioneer",
                "sage",
                "saint",
                "sinner",
                "villain",
            }
        )
        if (
            type(knights_knaves.VALID_ROLES) is not set
            or frozenset(knights_knaves.VALID_ROLES) != frozen_valid_roles
        ):
            raise ValueError("reasoning scorer role vocabulary differs")
        knights_knaves.VALID_ROLES = frozen_valid_roles
        frozen_scorer_globals = frozen_normalize_fn.__globals__
        if frozen_scorer_globals.get("VALID_ROLES") is not frozen_valid_roles:
            raise ValueError("reasoning scorer role vocabulary was not frozen")
        scorer = dict(scorer_pin)
        scorer.update(
            {
                "package_root": str(package_root),
                "package_resolved_root": str(resolved_package),
                "module_origin": str(module_origin),
                "module_resolved_origin": str(module_origin.resolve(strict=True)),
                "resolver_origin": str(resolver_origin),
                "resolver_resolved_origin": str(resolver_origin.resolve(strict=True)),
                "origin": str(scorer_origin),
                "resolved_origin": str(scorer_origin.resolve(strict=True)),
            }
        )
    if module_versions != target["module_versions"]:
        raise ValueError("module version literals differ")

    if direct_runner:
        # Catch any later dependency that tried to reopen an executable search
        # path after the two admitted Gym/Ray mutations were stripped.
        nemo_gym._augment_sys_path()

    ppid, start_ticks = _proc_stat(os.getpid())
    process = {
        "pid": os.getpid(),
        "ppid": ppid,
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "cwd": str(cwd),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "proc_exe": proc_exe,
        "sys_argv": list(sys.argv),
        "proc_argv": _proc_argv(os.getpid()),
        "start_ticks": start_ticks,
        "boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        "hostname": socket.gethostname(),
    }
    target_record = {
        name: target[name]
        for name in (
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
        )
    }
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "hash_domain": _HASH_DOMAIN,
        "environment": spec["environment"],
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "stage": (
            "isolated-runner-pre-entrypoint"
            if direct_runner
            else "sitecustomize-pre-entrypoint"
        ),
        "spec_sha256": spec_sha,
        "target": target_record,
        "server": {
            "config_path": config_path,
            "server_type": expected["server_type"],
            "server_name": expected["server_name"],
            "entrypoint": target["entrypoint"],
            "host": host,
            "port": port,
            "num_workers": num_workers,
        },
        "process": process,
        "distribution_versions": distributions,
        "module_versions": module_versions,
        "scorer": scorer,
    }
    _publish(spec["receipt_root"], expected["receipt_filename"], receipt)
    if reasoning_gym is not None and spec["scope"] == "scorer-only":
        _install_reasoning_score_evidence(
            reasoning_gym,
            spec=spec,
            spec_sha=spec_sha,
            process=process,
            frozen_score_fn=frozen_score_fn,
            frozen_normalize_fn=frozen_normalize_fn,
            frozen_score_code=frozen_score_code,
            frozen_normalize_code=frozen_normalize_code,
            frozen_valid_roles=frozen_valid_roles,
            frozen_scorer_globals=frozen_scorer_globals,
        )
    if direct_runner:
        return target, app_path
    return None


def _abort(error):
    message = (
        "STRICT_GYM_CHILD_ATTESTATION_FAILED "
        f"type={type(error).__module__}.{type(error).__qualname__} "
        f"detail={str(error)[:512]!r}\n"
    ).encode("ascii", "backslashreplace")
    try:
        os.write(2, message)
    finally:
        os._exit(78)


if os.environ.get(_SPEC_ENV):
    try:
        _strict_gym_attested_target = _attest()
        if os.environ.get(_DIRECT_RUNNER_ENV) == "1":
            if _strict_gym_attested_target is None:
                raise ValueError("isolated scorer runner did not select a target")
            _strict_gym_target, _strict_gym_app_path = _strict_gym_attested_target
            _strict_gym_app_payload = _read_file(_strict_gym_app_path, maximum=1 << 24)
            if (
                hashlib.sha256(_strict_gym_app_payload).hexdigest()
                != _strict_gym_target["source_sha256"]
            ):
                raise ValueError("isolated scorer app changed after attestation")
            sys.argv = [_strict_gym_target["entrypoint"]]
            _strict_gym_globals = {
                "__name__": "__main__",
                "__file__": str(_strict_gym_app_path),
                "__package__": None,
                "__cached__": None,
            }
            exec(
                compile(
                    _strict_gym_app_payload,
                    str(_strict_gym_app_path),
                    "exec",
                    dont_inherit=True,
                ),
                _strict_gym_globals,
            )
    except BaseException as _strict_gym_error:
        _abort(_strict_gym_error)
