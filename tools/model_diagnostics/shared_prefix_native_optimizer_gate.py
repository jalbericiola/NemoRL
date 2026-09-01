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

"""Gate the production MCore precision-aware state initializer on a NeMo-RL GPU step.

This diagnostic intentionally does not call Transformer Engine's state initializer
or the captured-parity diagnostic initializer.  It calls the ``init_state_fn``
closure installed on the real MCore optimizer, validates the exact BF16/INT16
remainder representation, and then lets ``MegatronPolicyWorkerImpl`` finish its
native optimizer step.

Run ``run`` under ``torchrun`` with the same four-GPU TP2/CP2/SP/EP4/MTP5 topology
as captured citation parity.  The builder must supply independent outer-overlay and
materialized-runtime digests via ``OUTER_PACKAGE_EXPECTED`` and
``RUNTIME_SOURCE_PACKAGE_EXPECTED``.  ``self-test-source-contract`` is CPU-only and
rejects any future direct call to either forbidden initializer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "nemorl-shared-prefix-native-optimizer-gate-v2"
BRIDGE_ROOT_RELATIVE = Path("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge")
MCORE_ROOT_RELATIVE = BRIDGE_ROOT_RELATIVE / "3rdparty/Megatron-LM"
BRIDGE_PACKAGE_RELATIVE = BRIDGE_ROOT_RELATIVE / "src/megatron/bridge/__init__.py"
MCORE_OPTIMIZER_RELATIVE = MCORE_ROOT_RELATIVE / "megatron/core/optimizer/__init__.py"
DEPLOYMENT_METADATA_ROOT_RELATIVE = Path(".rlvr41-deployment")
OUTER_MANIFEST_RELATIVE = DEPLOYMENT_METADATA_ROOT_RELATIVE / "PACKAGE.sha256"
OUTER_EXPECTED_RELATIVE = DEPLOYMENT_METADATA_ROOT_RELATIVE / "PACKAGE_EXPECTED.txt"
SOURCE_IDENTITIES_RELATIVE = DEPLOYMENT_METADATA_ROOT_RELATIVE / "SOURCE_IDENTITIES.json"
RUNTIME_MATERIALIZATION_RELATIVE = DEPLOYMENT_METADATA_ROOT_RELATIVE / "RUNTIME_MATERIALIZATION.json"
RUNTIME_MATERIALIZATION_SCHEMA = "rlvr41-runtime-materialization-v1"
Q_BASE_ROOT = (
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/"
    "jalbericiola/rlvr41_spfx_validation/deployments/validated_shared_prefix_20260830q"
)
Q_BASE_READY = "441548f85b9779788d458a0d4deeefcca789ed8b46810a1f6a929492cd27d4cf"
Q_BASE_PACKAGE_SHA256 = "09e8875f73d76f13c03b69d1496ee9f486d53d0badfdaa10df578f94b1cfd269"
RUNTIME_LAYOUT = {
    "schema": "rlvr41-vendored-nemorl-runtime-layout-v1",
    "root_repository": "NemoRL",
    "repository_roots": {
        "NemoRL": ".",
        "Megatron-Bridge": BRIDGE_ROOT_RELATIVE.as_posix(),
        "Megatron-LM": MCORE_ROOT_RELATIVE.as_posix(),
    },
    "require_regular_read_only_files": True,
    "allow_repository_root_symlinks": False,
}
SOURCE_TREE_ROOTS = {
    "nemo_rl": Path("nemo_rl"),
    "megatron_bridge": BRIDGE_ROOT_RELATIVE,
    "megatron_core": MCORE_ROOT_RELATIVE,
    "model_diagnostics": Path("tools/model_diagnostics"),
}
IMPORTED_PREFIX_ROOTS = {
    "nemo_rl": Path("nemo_rl"),
    "megatron.bridge": BRIDGE_ROOT_RELATIVE / "src/megatron/bridge",
    "megatron.core": MCORE_ROOT_RELATIVE / "megatron/core",
    "megatron.training": MCORE_ROOT_RELATIVE / "megatron/training",
    "tools": Path("tools"),
}
MEGATRON_NAMESPACE_ROOTS = {
    BRIDGE_ROOT_RELATIVE / "src/megatron",
    MCORE_ROOT_RELATIVE / "megatron",
}
GATE_WORLD_SIZE = 4
GATE_BATCH_SIZE = 4
GATE_FIXTURE_SHA256 = "ef5a69c9ca579a55c940c7a1ae4cfd4ed12b666073563ae61d49138a8ec62e12"
GATE_SELECTED_TOKENS = 788
GATE_PACKING_TOKENS = 16384
GATE_REQUIRED_FAMILIES = {"attention", "mamba", "moe", "mtp"}
GATE_MTP_METRIC_KEYS = {
    *(f"mtp_{depth}_loss" for depth in range(1, 6)),
    *(f"mtp_{depth}_acceptance_rate" for depth in range(1, 6)),
    "grad_norm",
}
GATE_TOPOLOGY = {
    "world_size": GATE_WORLD_SIZE,
    "tensor_parallel_size": 2,
    "context_parallel_size": 2,
    "sequence_parallel": True,
    "expert_parallel_size": 4,
    "expert_tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "mtp_num_layers": 5,
    "mtp_use_repeated_layer": True,
    "mtp_detach_heads": True,
}


class NativeOptimizerGateError(RuntimeError):
    """The production initializer or native optimizer-step contract is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_regular_file_bytes(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    """Read one regular file through O_NOFOLLOW and prove its inode stayed stable."""
    try:
        before = path.lstat()
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot stat {label} {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise NativeOptimizerGateError(f"{label} is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot safely open {label} {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise NativeOptimizerGateError(f"{label} changed while being opened: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(opened):
            raise NativeOptimizerGateError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    try:
        final_path = path.lstat()
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot restat {label} {path}: {error}") from error
    if _stat_identity(final_path) != _stat_identity(opened):
        raise NativeOptimizerGateError(f"{label} path was replaced while being read: {path}")
    value = b"".join(chunks)
    if len(value) != opened.st_size:
        raise NativeOptimizerGateError(f"short read for {label}: {path}")
    return value, opened


def _file_sha256(path: Path) -> str:
    value, _ = _read_regular_file_bytes(path, label="SHA256 input")
    return _bytes_sha256(value)


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _BoundSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile only descriptor-read source bytes and retain their execution digest."""

    def __init__(self, fullname: str, path: str, *, repo_root: Path, audit: dict[str, dict[str, Any]]):
        super().__init__(fullname, path)
        self._repo_root = repo_root
        self._audit = audit

    def get_data(self, path: str) -> bytes:
        candidate = Path(path)
        if candidate.suffix in {".pyc", ".pyo"}:
            try:
                candidate.lstat()
            except FileNotFoundError:
                raise
            except OSError as error:
                raise NativeOptimizerGateError(f"cannot stat cached import path {candidate}: {error}") from error
            raise NativeOptimizerGateError(f"bound import attempted cached bytecode: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise NativeOptimizerGateError(f"cannot resolve bound import source {candidate}: {error}") from error
        source_path = Path(self.path)
        if resolved != source_path:
            raise NativeOptimizerGateError(
                f"bound import loader requested an unexpected file: module={self.name} "
                f"requested={resolved} expected={source_path}"
            )
        value, metadata = _read_regular_file_bytes(resolved, label=f"executed import {self.name}")
        try:
            relative_path = resolved.relative_to(self._repo_root).as_posix()
        except ValueError as error:
            raise NativeOptimizerGateError(f"executed import {self.name} is outside --repo-root: {resolved}") from error
        record = {
            "module": self.name,
            "relative_path": relative_path,
            "sha256": _bytes_sha256(value),
            "size": metadata.st_size,
        }
        previous = self._audit.get(self.name)
        if previous is not None and previous != record:
            raise NativeOptimizerGateError(f"executed source bytes changed across imports of {self.name}")
        self._audit[self.name] = record
        return value


class _BoundSourceFinder(importlib.abc.MetaPathFinder):
    """Force every gate-owned Python import through :class:`_BoundSourceLoader`."""

    def __init__(self, repo_root: Path, audit: dict[str, dict[str, Any]]):
        self._repo_root = repo_root
        self._audit = audit

    def find_spec(self, fullname: str, path: Sequence[str] | None, target: Any = None) -> Any:
        if fullname == "megatron":
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.origin is not None or spec.submodule_search_locations is None:
                raise NativeOptimizerGateError("top-level megatron must be the canonical vendored namespace")
            observed_roots = {Path(location).resolve(strict=True) for location in spec.submodule_search_locations}
            expected_roots = {self._repo_root / relative for relative in MEGATRON_NAMESPACE_ROOTS}
            if observed_roots != expected_roots:
                raise NativeOptimizerGateError(
                    "top-level megatron namespace search roots are not the canonical Bridge/MCore pair: "
                    f"observed={sorted(map(str, observed_roots))} expected={sorted(map(str, expected_roots))}"
                )
            return spec
        matching_prefixes = [
            prefix for prefix in IMPORTED_PREFIX_ROOTS if fullname == prefix or fullname.startswith(f"{prefix}.")
        ]
        if not matching_prefixes:
            if fullname.startswith("megatron."):
                raise NativeOptimizerGateError(f"unapproved megatron import subtree: {fullname}")
            return None
        if len(matching_prefixes) != 1:
            raise NativeOptimizerGateError(f"bound import prefix is ambiguous: {fullname}")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None
        if spec.origin is None:
            return spec
        if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise NativeOptimizerGateError(
                f"bound import {fullname} did not resolve to a Python source loader: "
                f"origin={spec.origin!r} loader={type(spec.loader).__name__}"
            )
        source_path = Path(spec.origin).resolve(strict=True)
        expected_root = self._repo_root / IMPORTED_PREFIX_ROOTS[matching_prefixes[0]]
        if source_path != expected_root and expected_root not in source_path.parents:
            raise NativeOptimizerGateError(
                f"bound import {fullname} is outside its canonical source root: "
                f"source={source_path} expected_root={expected_root}"
            )
        if source_path.suffix != ".py":
            raise NativeOptimizerGateError(f"bound import {fullname} is not Python source: {source_path}")
        spec.origin = str(source_path)
        spec.cached = None
        spec.loader = _BoundSourceLoader(
            fullname,
            str(source_path),
            repo_root=self._repo_root,
            audit=self._audit,
        )
        return spec


def _install_bound_source_importer(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Install the source-only loader before any NeMo/Bridge/MCore/tools import."""
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "megatron"
        or any(name == prefix or name.startswith(f"{prefix}.") for prefix in IMPORTED_PREFIX_ROOTS)
    )
    if preloaded:
        raise NativeOptimizerGateError(f"gate-owned modules were imported before source binding: {preloaded}")
    audit: dict[str, dict[str, Any]] = {}
    sys.meta_path.insert(0, _BoundSourceFinder(repo_root, audit))
    return audit


def _stable_directory_metadata(path: Path, *, label: str) -> os.stat_result:
    before = path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot safely open {label} directory {path}: {error}") from error
    try:
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final_path = path.lstat()
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _stat_identity(before) != _stat_identity(opened)
        or _stat_identity(final_path) != _stat_identity(opened)
    ):
        raise NativeOptimizerGateError(f"{label} directory changed while being inspected: {path}")
    return opened


def _require_immutable_directory(path: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    """Require one canonical, non-writable directory inside the sealed package root."""
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot resolve {label} directory {path}: {error}") from error
    if path.is_symlink() or path != resolved:
        raise NativeOptimizerGateError(f"{label} directory must be canonical and non-symlink: {path}")
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise NativeOptimizerGateError(
            f"{label} directory resolved outside --repo-root: directory={resolved} repo_root={repo_root}"
        ) from error
    metadata = _stable_directory_metadata(resolved, label=label)
    mode = metadata.st_mode
    if not stat.S_ISDIR(mode):
        raise NativeOptimizerGateError(f"{label} path is not a directory: {resolved}")
    if mode & 0o222:
        raise NativeOptimizerGateError(f"{label} directory is writable and permits source substitution: {resolved}")
    return {"relative_path": relative.as_posix() or ".", "mode": stat.S_IMODE(mode)}


def _require_immutable_parent_chain(path: Path, *, repo_root: Path, label: str) -> None:
    """Reject writable or replaceable package directories from a source up to the root."""
    current = path.parent
    while True:
        _require_immutable_directory(current, repo_root=repo_root, label=f"{label} parent")
        if current == repo_root:
            return
        current = current.parent


def _require_immutable_source_file(path: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    """Bind one regular, read-only source file beneath the declared repository root."""
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot resolve {label} source {path}: {error}") from error
    if path.is_symlink() or path != resolved:
        raise NativeOptimizerGateError(f"{label} source must be canonical and non-symlink: {path}")
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise NativeOptimizerGateError(
            f"{label} source resolved outside --repo-root: source={resolved} repo_root={repo_root}"
        ) from error
    value, metadata = _read_regular_file_bytes(resolved, label=f"{label} source")
    mode = metadata.st_mode
    if not stat.S_ISREG(mode):
        raise NativeOptimizerGateError(f"{label} source is not a regular file: {resolved}")
    if mode & 0o222:
        raise NativeOptimizerGateError(f"{label} source is writable and therefore not immutable: {resolved}")
    _require_immutable_parent_chain(resolved, repo_root=repo_root, label=label)
    return {
        "relative_path": relative.as_posix(),
        "sha256": _bytes_sha256(value),
        "size": metadata.st_size,
        "mode": stat.S_IMODE(mode),
    }


def _immutable_tree_fingerprint(tree_root: Path, *, repo_root: Path, label: str) -> dict[str, Any]:
    """Hash every non-cache regular file in one immutable source-package tree."""
    _require_immutable_directory(tree_root, repo_root=repo_root, label=f"{label} tree root")
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in sorted(tree_root.rglob("*")):
        relative_tree_path = path.relative_to(tree_root)
        if "__pycache__" in relative_tree_path.parts or path.suffix in {".pyc", ".pyo"}:
            raise NativeOptimizerGateError(f"{label} tree contains forbidden cached bytecode: {path}")
        if path.is_dir():
            directories.append(_require_immutable_directory(path, repo_root=repo_root, label=label))
            continue
        files.append(_require_immutable_source_file(path, repo_root=repo_root, label=label))
    if not files:
        raise NativeOptimizerGateError(f"{label} tree contains no immutable source files")
    manifest_sha256 = hashlib.sha256()
    total_bytes = 0
    root_record = _require_immutable_directory(tree_root, repo_root=repo_root, label=f"{label} tree root")
    manifest_sha256.update(_canonical_json_bytes({"kind": "directory", **root_record}))
    for item in directories:
        manifest_sha256.update(_canonical_json_bytes({"kind": "directory", **item}))
    for item in files:
        manifest_sha256.update(_canonical_json_bytes({"kind": "file", **item}))
        total_bytes += item["size"]
    return {
        "root": tree_root.relative_to(repo_root).as_posix(),
        "sha256": manifest_sha256.hexdigest(),
        "directory_count": len(directories) + 1,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _loaded_module_source_manifest(
    repo_root: Path,
    modules: dict[str, Any],
    *,
    prefixes: tuple[str, ...],
    expected_roots: dict[str, Path] | None = None,
    executed_sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove every loaded NeMo/Bridge/MCore file-backed module comes from --repo-root."""
    records: list[dict[str, Any]] = []
    namespace_modules: list[str] = []
    for name, module in sorted(modules.items()):
        if not any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            namespace_modules.append(name)
            continue
        item = _require_immutable_source_file(
            Path(source),
            repo_root=repo_root,
            label=f"imported module {name}",
        )
        if Path(source).suffix in {".pyc", ".pyo"}:
            raise NativeOptimizerGateError(f"imported module {name} executed cached bytecode: {source}")
        cached = getattr(module, "__cached__", None)
        if isinstance(cached, str) and Path(cached).exists():
            raise NativeOptimizerGateError(f"imported module {name} has forbidden cached bytecode: {cached}")
        if expected_roots is not None:
            matching_prefixes = [prefix for prefix in prefixes if name == prefix or name.startswith(f"{prefix}.")]
            if len(matching_prefixes) != 1:
                raise NativeOptimizerGateError(f"cannot resolve a unique source root for imported module {name}")
            expected_root = expected_roots[matching_prefixes[0]]
            relative_path = Path(item["relative_path"])
            if relative_path != expected_root and expected_root not in relative_path.parents:
                raise NativeOptimizerGateError(
                    f"imported module {name} is outside its canonical vendored root: "
                    f"source={relative_path} expected_root={expected_root}"
                )
        if executed_sources is not None:
            executed = executed_sources.get(name)
            expected_executed = {
                "module": name,
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
                "size": item["size"],
            }
            if executed != expected_executed:
                raise NativeOptimizerGateError(f"imported module {name} was not compiled from its bound source bytes")
        records.append({"module": name, **item})
    loaded_prefixes = {
        prefix
        for prefix in prefixes
        if any(record["module"] == prefix or record["module"].startswith(f"{prefix}.") for record in records)
    }
    missing_prefixes = set(prefixes) - loaded_prefixes
    if missing_prefixes:
        raise NativeOptimizerGateError(
            f"source binding did not observe required imported module prefixes: {sorted(missing_prefixes)}"
        )
    if executed_sources is not None and set(executed_sources) != {record["module"] for record in records}:
        raise NativeOptimizerGateError("executed/imported bound-module inventories differ")
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json_bytes(record))
    return {
        "prefixes": list(prefixes),
        "sha256": digest.hexdigest(),
        "file_backed_module_count": len(records),
        "namespace_modules": namespace_modules,
        "modules": records,
    }


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeOptimizerGateError(f"{label} is not a lowercase SHA256 digest")
    return value


def _validated_git_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NativeOptimizerGateError(f"{label} is not a lowercase 40-character Git SHA")
    return value


def _no_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeOptimizerGateError(f"duplicate JSON key in runtime materialization: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise NativeOptimizerGateError(f"invalid JSON constant in runtime materialization: {value}")


def _source_package_layout() -> dict[str, Any]:
    return {
        "bridge_package": BRIDGE_PACKAGE_RELATIVE.as_posix(),
        "mcore_optimizer": MCORE_OPTIMIZER_RELATIVE.as_posix(),
        "outer_manifest": OUTER_MANIFEST_RELATIVE.as_posix(),
        "outer_expected": OUTER_EXPECTED_RELATIVE.as_posix(),
        "source_identities": SOURCE_IDENTITIES_RELATIVE.as_posix(),
        "source_tree_roots": {name: relative.as_posix() for name, relative in sorted(SOURCE_TREE_ROOTS.items())},
    }


def _source_package_sha256(explicit_sources: dict[str, Any], source_trees: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(_source_package_layout()))
    digest.update(_canonical_json_bytes(explicit_sources))
    digest.update(_canonical_json_bytes(source_trees))
    return digest.hexdigest()


def _immutable_source_package(repo_root: Path) -> dict[str, Any]:
    """Fingerprint the complete canonical NeMo-RL, Bridge, and MCore source package."""
    _require_immutable_directory(repo_root, repo_root=repo_root, label="NeMo-RL repository root")
    explicit_paths = {
        "gate": Path("tools/model_diagnostics/shared_prefix_native_optimizer_gate.py"),
        "captured_parity_helpers": Path("tools/model_diagnostics/shared_prefix_captured_citation_parity.py"),
        "tools_package": Path("tools/__init__.py"),
        "nemo_rl_package": Path("nemo_rl/__init__.py"),
        "bridge_package": BRIDGE_PACKAGE_RELATIVE,
        "mcore_optimizer": MCORE_OPTIMIZER_RELATIVE,
        "outer_package_manifest": OUTER_MANIFEST_RELATIVE,
        "outer_package_expected": OUTER_EXPECTED_RELATIVE,
        "source_identities": SOURCE_IDENTITIES_RELATIVE,
        "recipe": Path("examples/nemo_gym/nemotron-3.5-nano/rlvr.yaml"),
        "pyproject": Path("pyproject.toml"),
        "lockfile": Path("uv.lock"),
    }
    explicit_sources = {
        name: _require_immutable_source_file(repo_root / relative, repo_root=repo_root, label=name)
        for name, relative in explicit_paths.items()
    }
    source_trees = {
        name: _immutable_tree_fingerprint(repo_root / relative, repo_root=repo_root, label=name)
        for name, relative in SOURCE_TREE_ROOTS.items()
    }
    return {
        "source_package_sha256": _source_package_sha256(explicit_sources, source_trees),
        "explicit_sources": explicit_sources,
        "source_trees": source_trees,
    }


def _parse_outer_payload_manifest(value: bytes) -> dict[str, str]:
    """Parse the builder's exact raw two-space-delimited PACKAGE.sha256 format."""
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as error:
        raise NativeOptimizerGateError("outer PACKAGE.sha256 is not ASCII") from error
    if not text.endswith("\n"):
        raise NativeOptimizerGateError("outer PACKAGE.sha256 lacks a final newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise NativeOptimizerGateError(f"malformed outer PACKAGE.sha256 line: {line!r}")
        digest = _validated_sha256(fields[0], label="outer package member sha256")
        relative = Path(fields[1])
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != fields[1]
        ):
            raise NativeOptimizerGateError(f"unsafe outer package member path: {fields[1]!r}")
        if fields[1] in result:
            raise NativeOptimizerGateError(f"duplicate outer package member: {fields[1]}")
        result[fields[1]] = digest
    if not result:
        raise NativeOptimizerGateError("outer PACKAGE.sha256 is empty")
    return result


def _outer_package_binding(repo_root: Path, *, outer_package_expected: Any) -> dict[str, str]:
    """Bind the raw builder overlay manifest embedded in the materialized runtime."""
    outer_package_expected = _validated_sha256(outer_package_expected, label="builder OUTER_PACKAGE_EXPECTED")
    manifest_path = repo_root / OUTER_MANIFEST_RELATIVE
    expected_path = repo_root / OUTER_EXPECTED_RELATIVE
    identities_path = repo_root / SOURCE_IDENTITIES_RELATIVE
    manifest_record = _require_immutable_source_file(manifest_path, repo_root=repo_root, label="outer package manifest")
    _require_immutable_source_file(expected_path, repo_root=repo_root, label="outer package expected digest")
    identities_record = _require_immutable_source_file(
        identities_path, repo_root=repo_root, label="outer source identities"
    )
    if manifest_record["sha256"] != outer_package_expected:
        raise NativeOptimizerGateError(
            "embedded outer PACKAGE.sha256 does not match builder OUTER_PACKAGE_EXPECTED: "
            f"observed={manifest_record['sha256']} expected={outer_package_expected}"
        )
    expected_bytes = f"package_manifest_sha256={outer_package_expected}\n".encode("ascii")
    expected_value, _ = _read_regular_file_bytes(expected_path, label="outer package expected digest")
    if expected_value != expected_bytes:
        raise NativeOptimizerGateError("embedded PACKAGE_EXPECTED.txt does not exactly bind OUTER_PACKAGE_EXPECTED")
    manifest_value, _ = _read_regular_file_bytes(manifest_path, label="outer package manifest")
    outer_members = _parse_outer_payload_manifest(manifest_value)
    if outer_members.get("SOURCE_IDENTITIES.json") != identities_record["sha256"]:
        raise NativeOptimizerGateError("embedded SOURCE_IDENTITIES.json is not bound by outer PACKAGE.sha256")
    return {
        "manifest_relative_path": OUTER_MANIFEST_RELATIVE.as_posix(),
        "manifest_sha256": outer_package_expected,
        "expected_relative_path": OUTER_EXPECTED_RELATIVE.as_posix(),
        "expected_sha256": outer_package_expected,
    }


def _validate_runtime_materialization_manifest(
    manifest: Any,
    *,
    repo_root: str,
    outer_package_sha256: str,
    runtime_source_package_sha256: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "base",
        "outer_package_manifest_sha256",
        "runtime_source_package_sha256",
        "runtime_layout",
        "repositories",
        "runtime_repo_root",
    }:
        raise NativeOptimizerGateError("runtime materialization manifest has invalid keys")
    if manifest["schema"] != RUNTIME_MATERIALIZATION_SCHEMA:
        raise NativeOptimizerGateError("runtime materialization schema is invalid")
    if manifest["base"] != {
        "root": Q_BASE_ROOT,
        "ready": Q_BASE_READY,
        "package_manifest_sha256": Q_BASE_PACKAGE_SHA256,
    }:
        raise NativeOptimizerGateError("runtime materialization base is not Deployment Q")
    if manifest["outer_package_manifest_sha256"] != outer_package_sha256:
        raise NativeOptimizerGateError("runtime materialization outer-package digest mismatch")
    if manifest["runtime_source_package_sha256"] != runtime_source_package_sha256:
        raise NativeOptimizerGateError("runtime materialization source-package digest mismatch")
    if manifest["runtime_layout"] != RUNTIME_LAYOUT:
        raise NativeOptimizerGateError("runtime materialization layout is not canonical")
    if manifest["runtime_repo_root"] != repo_root:
        raise NativeOptimizerGateError("runtime materialization repository root mismatch")
    repositories = manifest["repositories"]
    if not isinstance(repositories, dict) or set(repositories) != {
        "NemoRL",
        "Megatron-Bridge",
        "Megatron-LM",
    }:
        raise NativeOptimizerGateError("runtime materialization repository inventory is invalid")
    required_repository_fields = {
        "base_head",
        "head",
        "overlay_archive_sha256",
        "overlay_payload_manifest_sha256",
    }
    for name, repository in repositories.items():
        if not isinstance(repository, dict) or set(repository) != required_repository_fields:
            raise NativeOptimizerGateError(f"runtime materialization {name} binding is invalid")
        _validated_git_sha(repository["base_head"], label=f"runtime materialization {name} base head")
        _validated_git_sha(repository["head"], label=f"runtime materialization {name} head")
        _validated_sha256(
            repository["overlay_archive_sha256"],
            label=f"runtime materialization {name} overlay archive sha256",
        )
        _validated_sha256(
            repository["overlay_payload_manifest_sha256"],
            label=f"runtime materialization {name} overlay payload manifest sha256",
        )
    return manifest


def _runtime_materialization_binding(
    repo_root: Path,
    *,
    outer_package_sha256: str,
    runtime_source_package_sha256: str,
) -> dict[str, Any]:
    """Validate the builder-sealed materialization record without digest recursion."""
    path = repo_root / RUNTIME_MATERIALIZATION_RELATIVE
    source_record = _require_immutable_source_file(path, repo_root=repo_root, label="runtime materialization manifest")
    raw, _ = _read_regular_file_bytes(path, label="runtime materialization manifest")
    try:
        manifest = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeOptimizerGateError(f"invalid runtime materialization JSON: {error}") from error
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != raw:
        raise NativeOptimizerGateError(
            "runtime materialization must use exact canonical JSON bytes with no trailing newline"
        )
    _validate_runtime_materialization_manifest(
        manifest,
        repo_root=str(repo_root),
        outer_package_sha256=outer_package_sha256,
        runtime_source_package_sha256=runtime_source_package_sha256,
    )
    if source_record["sha256"] != _bytes_sha256(raw):
        raise NativeOptimizerGateError("runtime materialization changed while being read")
    return {
        "relative_path": RUNTIME_MATERIALIZATION_RELATIVE.as_posix(),
        "sha256": source_record["sha256"],
        "manifest": manifest,
    }


def _require_expected_runtime_source_package(observed: str, expected: Any) -> str:
    observed = _validated_sha256(observed, label="runtime source-package sha256")
    expected = _validated_sha256(expected, label="builder RUNTIME_SOURCE_PACKAGE_EXPECTED")
    if observed != expected:
        raise NativeOptimizerGateError(
            "runtime source package does not match builder RUNTIME_SOURCE_PACKAGE_EXPECTED: "
            f"observed={observed} expected={expected}"
        )
    return expected


def _validate_tree_binding(value: Any, *, name: str, expected_root: Path) -> None:
    required_fields = {"root", "sha256", "directory_count", "file_count", "total_bytes"}
    if not isinstance(value, dict) or set(value) != required_fields:
        raise NativeOptimizerGateError(f"runtime {name} tree binding is invalid")
    if value["root"] != expected_root.as_posix():
        raise NativeOptimizerGateError(f"runtime {name} tree root is not canonical")
    _validated_sha256(value["sha256"], label=f"runtime {name} tree sha256")
    if not isinstance(value["directory_count"], int) or value["directory_count"] <= 0:
        raise NativeOptimizerGateError(f"runtime {name} tree directory count is invalid")
    if not isinstance(value["file_count"], int) or value["file_count"] <= 0:
        raise NativeOptimizerGateError(f"runtime {name} tree file count is invalid")
    if not isinstance(value["total_bytes"], int) or value["total_bytes"] <= 0:
        raise NativeOptimizerGateError(f"runtime {name} tree byte count is invalid")


def _validate_source_binding_evidence(value: Any) -> dict[str, Any]:
    """Recompute every manifest digest and validate required imported-source bindings."""
    if not isinstance(value, dict) or set(value) != {
        "repo_root",
        "outer_package",
        "materialization",
        "runtime_source_package_sha256",
        "sha256",
        "explicit_sources",
        "source_trees",
        "imported_modules",
    }:
        raise NativeOptimizerGateError("runtime source binding has an invalid schema")
    repo_root = value["repo_root"]
    if not isinstance(repo_root, str) or not Path(repo_root).is_absolute():
        raise NativeOptimizerGateError("runtime source binding repo_root must be absolute")
    explicit_sources = value["explicit_sources"]
    required_sources = {
        "gate",
        "captured_parity_helpers",
        "tools_package",
        "nemo_rl_package",
        "bridge_package",
        "mcore_optimizer",
        "outer_package_manifest",
        "outer_package_expected",
        "source_identities",
        "recipe",
        "pyproject",
        "lockfile",
    }
    if not isinstance(explicit_sources, dict) or set(explicit_sources) != required_sources:
        raise NativeOptimizerGateError("runtime source binding lacks required explicit sources")
    for label, source in explicit_sources.items():
        if not isinstance(source, dict) or set(source) != {"relative_path", "sha256", "size", "mode"}:
            raise NativeOptimizerGateError(f"runtime source binding {label} record is invalid")
        relative_path = source["relative_path"]
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise NativeOptimizerGateError(f"runtime source binding {label} path is invalid")
        _validated_sha256(source["sha256"], label=f"runtime source binding {label} sha256")
        if not isinstance(source["size"], int) or source["size"] < 0:
            raise NativeOptimizerGateError(f"runtime source binding {label} size is invalid")
        if (
            not isinstance(source["mode"], int)
            or source["mode"] < 0
            or source["mode"] > 0o7777
            or source["mode"] & 0o222
        ):
            raise NativeOptimizerGateError(f"runtime source binding {label} mode is not immutable")
    if explicit_sources["gate"]["relative_path"] != ("tools/model_diagnostics/shared_prefix_native_optimizer_gate.py"):
        raise NativeOptimizerGateError("runtime source binding gate path is not canonical")
    if explicit_sources["tools_package"]["relative_path"] != "tools/__init__.py":
        raise NativeOptimizerGateError("runtime source binding tools package path is not canonical")
    if explicit_sources["nemo_rl_package"]["relative_path"] != "nemo_rl/__init__.py":
        raise NativeOptimizerGateError("runtime source binding NeMo-RL package path is not canonical")
    if explicit_sources["bridge_package"]["relative_path"] != BRIDGE_PACKAGE_RELATIVE.as_posix():
        raise NativeOptimizerGateError("runtime source binding Bridge package path is not canonical")
    if explicit_sources["mcore_optimizer"]["relative_path"] != MCORE_OPTIMIZER_RELATIVE.as_posix():
        raise NativeOptimizerGateError("runtime source binding MCore optimizer path is not canonical")
    expected_outer_source_paths = {
        "outer_package_manifest": OUTER_MANIFEST_RELATIVE.as_posix(),
        "outer_package_expected": OUTER_EXPECTED_RELATIVE.as_posix(),
        "source_identities": SOURCE_IDENTITIES_RELATIVE.as_posix(),
    }
    for label, expected_path in expected_outer_source_paths.items():
        if explicit_sources[label]["relative_path"] != expected_path:
            raise NativeOptimizerGateError(f"runtime source binding {label} path is not canonical")

    outer_package = value["outer_package"]
    if not isinstance(outer_package, dict) or set(outer_package) != {
        "manifest_relative_path",
        "manifest_sha256",
        "expected_relative_path",
        "expected_sha256",
    }:
        raise NativeOptimizerGateError("runtime outer-package binding is invalid")
    if outer_package["manifest_relative_path"] != OUTER_MANIFEST_RELATIVE.as_posix():
        raise NativeOptimizerGateError("runtime outer manifest path is not canonical")
    if outer_package["expected_relative_path"] != OUTER_EXPECTED_RELATIVE.as_posix():
        raise NativeOptimizerGateError("runtime outer expected path is not canonical")
    outer_sha256 = _validated_sha256(outer_package["manifest_sha256"], label="runtime outer package manifest sha256")
    if outer_package["expected_sha256"] != outer_sha256:
        raise NativeOptimizerGateError("runtime outer expected digest differs from manifest digest")
    if explicit_sources["outer_package_manifest"]["sha256"] != outer_sha256:
        raise NativeOptimizerGateError("runtime outer manifest source digest is inconsistent")
    expected_outer_text = f"package_manifest_sha256={outer_sha256}\n".encode("ascii")
    if explicit_sources["outer_package_expected"]["sha256"] != _bytes_sha256(expected_outer_text):
        raise NativeOptimizerGateError("runtime outer expected-file digest is inconsistent")

    source_trees = value["source_trees"]
    if not isinstance(source_trees, dict) or set(source_trees) != set(SOURCE_TREE_ROOTS):
        raise NativeOptimizerGateError("runtime source binding lacks required full source trees")
    for name, expected_root in SOURCE_TREE_ROOTS.items():
        _validate_tree_binding(source_trees[name], name=name, expected_root=expected_root)

    imported_modules = value["imported_modules"]
    if not isinstance(imported_modules, dict) or set(imported_modules) != {
        "prefixes",
        "sha256",
        "file_backed_module_count",
        "namespace_modules",
        "modules",
    }:
        raise NativeOptimizerGateError("runtime imported-module binding is invalid")
    if imported_modules["prefixes"] != [
        "nemo_rl",
        "megatron.bridge",
        "megatron.core",
        "megatron.training",
        "tools",
    ]:
        raise NativeOptimizerGateError("runtime imported-module prefixes are invalid")
    modules = imported_modules["modules"]
    if not isinstance(modules, list) or not modules:
        raise NativeOptimizerGateError("runtime imported-module manifest is empty")
    if imported_modules["file_backed_module_count"] != len(modules):
        raise NativeOptimizerGateError("runtime imported-module count is inconsistent")
    module_digest = hashlib.sha256()
    seen_module_names: set[str] = set()
    for module in modules:
        if not isinstance(module, dict) or set(module) != {
            "module",
            "relative_path",
            "sha256",
            "size",
            "mode",
        }:
            raise NativeOptimizerGateError("runtime imported-module record is invalid")
        module_name = module["module"]
        if not isinstance(module_name, str) or module_name in seen_module_names:
            raise NativeOptimizerGateError("runtime imported-module name is invalid or duplicated")
        seen_module_names.add(module_name)
        if not any(
            module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in imported_modules["prefixes"]
        ):
            raise NativeOptimizerGateError(f"runtime imported-module name is out of scope: {module_name}")
        relative_path = module["relative_path"]
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise NativeOptimizerGateError(f"runtime imported-module path is invalid: {relative_path!r}")
        matching_prefixes = [
            prefix
            for prefix in imported_modules["prefixes"]
            if module_name == prefix or module_name.startswith(f"{prefix}.")
        ]
        if len(matching_prefixes) != 1:
            raise NativeOptimizerGateError(f"runtime imported-module prefix is ambiguous: {module_name}")
        expected_root = IMPORTED_PREFIX_ROOTS[matching_prefixes[0]]
        relative_module_path = Path(relative_path)
        if relative_module_path != expected_root and expected_root not in relative_module_path.parents:
            raise NativeOptimizerGateError(f"runtime imported module {module_name} is outside canonical vendored root")
        _validated_sha256(module["sha256"], label=f"runtime imported module {module_name} sha256")
        if not isinstance(module["size"], int) or module["size"] < 0:
            raise NativeOptimizerGateError(f"runtime imported module {module_name} size is invalid")
        if not isinstance(module["mode"], int) or module["mode"] & 0o222:
            raise NativeOptimizerGateError(f"runtime imported module {module_name} is writable")
        module_digest.update(_canonical_json_bytes(module))
    if module_digest.hexdigest() != _validated_sha256(
        imported_modules["sha256"], label="runtime imported-module manifest sha256"
    ):
        raise NativeOptimizerGateError("runtime imported-module manifest digest mismatch")

    runtime_source_package_sha256 = _source_package_sha256(explicit_sources, source_trees)
    if runtime_source_package_sha256 != _validated_sha256(
        value["runtime_source_package_sha256"], label="runtime source-package sha256"
    ):
        raise NativeOptimizerGateError("runtime source-package digest mismatch")
    materialization = value["materialization"]
    if not isinstance(materialization, dict) or set(materialization) != {
        "relative_path",
        "sha256",
        "manifest",
    }:
        raise NativeOptimizerGateError("runtime materialization evidence is invalid")
    if materialization["relative_path"] != RUNTIME_MATERIALIZATION_RELATIVE.as_posix():
        raise NativeOptimizerGateError("runtime materialization evidence path is not canonical")
    materialization_sha256 = _validated_sha256(
        materialization["sha256"], label="runtime materialization evidence sha256"
    )
    if _bytes_sha256(_canonical_json_bytes(materialization["manifest"])) != materialization_sha256:
        raise NativeOptimizerGateError("runtime materialization evidence digest mismatch")
    _validate_runtime_materialization_manifest(
        materialization["manifest"],
        repo_root=repo_root,
        outer_package_sha256=outer_sha256,
        runtime_source_package_sha256=runtime_source_package_sha256,
    )
    binding_digest = hashlib.sha256()
    binding_digest.update(_canonical_json_bytes({"repo_root": repo_root}))
    binding_digest.update(_canonical_json_bytes(outer_package))
    binding_digest.update(bytes.fromhex(runtime_source_package_sha256))
    binding_digest.update(_canonical_json_bytes(explicit_sources))
    binding_digest.update(_canonical_json_bytes(source_trees))
    binding_digest.update(_canonical_json_bytes(materialization))
    binding_digest.update(_canonical_json_bytes(imported_modules))
    if binding_digest.hexdigest() != _validated_sha256(value["sha256"], label="runtime binding sha256"):
        raise NativeOptimizerGateError("runtime source-binding digest mismatch")
    return value


def _source_contract() -> dict[str, Any]:
    """Prove that this gate cannot silently fall back to either diagnostic API."""
    source_path = Path(__file__).resolve(strict=True)
    source_bytes, _ = _read_regular_file_bytes(source_path, label="native optimizer gate source")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NativeOptimizerGateError("native optimizer gate source is not UTF-8") from error
    tree = ast.parse(source_text, filename=str(source_path))
    forbidden_call_names = {
        "_initialize_optimizer_state_for_probe",
        "initialize_state",
    }
    forbidden_calls: list[tuple[str, int]] = []
    production_closure_call_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        else:
            call_name = ""
        if call_name in forbidden_call_names:
            forbidden_calls.append((call_name, node.lineno))
        if call_name == "production_initializer":
            production_closure_call_count += 1
    if forbidden_calls:
        raise NativeOptimizerGateError(
            f"native optimizer gate contains forbidden direct initializer calls: {forbidden_calls}"
        )
    if production_closure_call_count != 1:
        raise NativeOptimizerGateError(
            "native optimizer gate must call the resolved production MCore closure exactly once"
        )
    return {
        "schema": SCHEMA,
        "forbidden_direct_calls": 0,
        "production_closure_calls": production_closure_call_count,
    }


def _raw_optimizer_parts(optimizer: Any) -> list[tuple[Any, Any, list[Any]]]:
    """Resolve MCore wrapper, raw optimizer, and its unique parameters."""
    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    result: list[tuple[Any, Any, list[Any]]] = []
    seen_parameters: set[int] = set()
    for part in parity._optimizer_parts(optimizer):
        raw_optimizer = getattr(part, "optimizer", None)
        groups = getattr(raw_optimizer, "param_groups", None)
        state = getattr(raw_optimizer, "state", None)
        if not isinstance(groups, list) or not hasattr(state, "get"):
            raise NativeOptimizerGateError("MCore optimizer part has invalid raw optimizer state")
        parameters: list[Any] = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("params"), list):
                raise NativeOptimizerGateError("raw optimizer parameter group is invalid")
            for parameter in group["params"]:
                identity = id(parameter)
                if identity in seen_parameters:
                    raise NativeOptimizerGateError("raw optimizer parameter occurs more than once")
                seen_parameters.add(identity)
                parameters.append(parameter)
        if not parameters:
            raise NativeOptimizerGateError("MCore optimizer part has no raw parameters")
        result.append((part, raw_optimizer, parameters))
    if not result:
        raise NativeOptimizerGateError("worker exposes no MCore optimizer parts")
    return result


def _require_empty_optimizer_state(optimizer: Any) -> dict[str, int]:
    """Fail unless every raw optimizer parameter still has lazy, empty state."""
    parameter_count = 0
    for _, raw_optimizer, parameters in _raw_optimizer_parts(optimizer):
        nonempty = [parameter for parameter in parameters if raw_optimizer.state.get(parameter)]
        if nonempty:
            raise NativeOptimizerGateError(
                "production initializer gate requires empty optimizer state, "
                f"got {len(nonempty)} initialized parameters"
            )
        parameter_count += len(parameters)
    return {"parts": len(_raw_optimizer_parts(optimizer)), "parameters": parameter_count}


def _validate_initialized_state(optimizer: Any) -> dict[str, Any]:
    """Validate the exact precision-aware Adam state produced by MCore's closure."""
    import torch

    initialized_parameters = 0
    bf16_remainder_parameters = 0
    state_tensor_numel = 0
    for part, raw_optimizer, parameters in _raw_optimizer_parts(optimizer):
        config = getattr(part, "config", None)
        if getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", None) is not True:
            raise NativeOptimizerGateError("gate requires the recipe's precision-aware distributed optimizer")
        if getattr(config, "store_param_remainders", None) is not True:
            raise NativeOptimizerGateError("gate requires BF16/INT16 parameter-remainder storage")
        if getattr(raw_optimizer, "store_param_remainders", None) is not True:
            raise NativeOptimizerGateError("raw TE optimizer did not enable parameter remainders")
        for parameter in parameters:
            parameter_state = raw_optimizer.state.get(parameter)
            if not isinstance(parameter_state, dict) or set(parameter_state) != {
                "exp_avg",
                "exp_avg_sq",
                "master_param",
            }:
                keys = sorted(parameter_state) if isinstance(parameter_state, dict) else parameter_state
                raise NativeOptimizerGateError(f"production initializer produced unexpected state keys: {keys!r}")
            exp_avg = parameter_state["exp_avg"]
            exp_avg_sq = parameter_state["exp_avg_sq"]
            master_param = parameter_state["master_param"]
            if not all(isinstance(value, torch.Tensor) for value in (exp_avg, exp_avg_sq, master_param)):
                raise NativeOptimizerGateError("production initializer produced non-tensor state")
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise NativeOptimizerGateError(
                    "gate requires FP32 Adam moments, " f"got {exp_avg.dtype}/{exp_avg_sq.dtype}"
                )
            if torch.count_nonzero(exp_avg).item() or torch.count_nonzero(exp_avg_sq).item():
                raise NativeOptimizerGateError("production initializer produced nonzero Adam moments")
            if parameter.dtype == torch.bfloat16:
                if master_param.dtype != torch.int16 or torch.count_nonzero(master_param).item():
                    raise NativeOptimizerGateError("production initializer did not create a zero INT16 BF16 remainder")
                bf16_remainder_parameters += 1
            elif master_param.dtype != torch.float32:
                raise NativeOptimizerGateError(
                    f"non-BF16 precision-aware master must be FP32, got {master_param.dtype}"
                )
            initialized_parameters += 1
            state_tensor_numel += exp_avg.numel() + exp_avg_sq.numel() + master_param.numel()
    if initialized_parameters == 0 or bf16_remainder_parameters == 0:
        raise NativeOptimizerGateError("production initializer did not cover BF16 remainder state")
    return {
        "initialized_parameters": initialized_parameters,
        "bf16_remainder_parameters": bf16_remainder_parameters,
        "state_tensor_numel": state_tensor_numel,
    }


def _raw_optimizer_training_state_snapshot(optimizer: Any) -> dict[str, Any]:
    """Snapshot raw TE group counters and full-tensor Adam-moment reductions."""
    import math

    import torch

    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    parts: list[dict[str, Any]] = []
    for part_index, (_, raw_optimizer, parameters) in enumerate(_raw_optimizer_parts(optimizer)):
        parameter_locations = {id(parameter): index for index, parameter in enumerate(parameters)}
        group_steps: list[dict[str, Any]] = []
        for group_index, group in enumerate(raw_optimizer.param_groups):
            group_parameters = group["params"]
            if not group_parameters:
                continue
            step = group.get("step")
            if step is None:
                step_value = None
            elif isinstance(step, bool):
                raise NativeOptimizerGateError("raw optimizer group step must not be boolean")
            elif isinstance(step, int):
                step_value = step
            elif isinstance(step, torch.Tensor) and step.numel() == 1:
                step_value = int(step.detach().cpu().item())
            else:
                raise NativeOptimizerGateError(f"unsupported raw optimizer group step: {step!r}")
            group_steps.append(
                {
                    "group": group_index,
                    "parameter_count": len(group_parameters),
                    "present": "step" in group,
                    "value": step_value,
                }
            )

        moments: dict[str, dict[str, Any]] = {}
        for state_key in ("exp_avg", "exp_avg_sq"):
            digest = hashlib.sha256()
            tensor_count = 0
            numel = 0
            nonzero = 0
            squared_l2 = 0.0
            for parameter in parameters:
                parameter_state = raw_optimizer.state.get(parameter)
                if not isinstance(parameter_state, dict) or state_key not in parameter_state:
                    raise NativeOptimizerGateError(f"raw optimizer parameter lacks initialized {state_key} state")
                tensor = parameter_state[state_key]
                if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                    raise NativeOptimizerGateError(f"raw optimizer {state_key} must be an FP32 tensor")
                flattened = tensor.detach().reshape(-1)
                tensor_nonzero = int(torch.count_nonzero(flattened).item())
                tensor_l2 = float(torch.linalg.vector_norm(flattened, ord=2).item())
                if not math.isfinite(tensor_l2):
                    raise NativeOptimizerGateError(f"raw optimizer {state_key} contains non-finite values")
                indices = parity._sample_indices(
                    flattened.numel(),
                    device=flattened.device,
                    sample_count=parity.MASTER_SAMPLE_COUNT,
                )
                sample_sha256 = parity._tensor_sha256(flattened.index_select(0, indices))
                item = {
                    "parameter": parameter_locations[id(parameter)],
                    "numel": flattened.numel(),
                    "nonzero": tensor_nonzero,
                    "l2": tensor_l2,
                    "sample_sha256": sample_sha256,
                }
                digest.update(_canonical_json_bytes(item))
                tensor_count += 1
                numel += flattened.numel()
                nonzero += tensor_nonzero
                squared_l2 += tensor_l2 * tensor_l2
            aggregate_l2 = math.sqrt(squared_l2)
            if not math.isfinite(aggregate_l2):
                raise NativeOptimizerGateError(f"raw optimizer aggregate {state_key} L2 is non-finite")
            moments[state_key] = {
                "sha256": digest.hexdigest(),
                "tensor_count": tensor_count,
                "numel": numel,
                "nonzero": nonzero,
                "l2": aggregate_l2,
            }
        parts.append(
            {
                "part": part_index,
                "parameter_count": len(parameters),
                "group_steps": group_steps,
                "moments": moments,
            }
        )
    return {"parts": parts}


def _validate_raw_optimizer_step(before: Any, after: Any) -> dict[str, Any]:
    """Require one TE group-step increment and changed Adam moments in every optimizer part."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise NativeOptimizerGateError("raw optimizer snapshots must be dictionaries")
    before_parts = before.get("parts")
    after_parts = after.get("parts")
    if not isinstance(before_parts, list) or not before_parts or not isinstance(after_parts, list):
        raise NativeOptimizerGateError("raw optimizer snapshots have invalid parts")
    if len(before_parts) != len(after_parts):
        raise NativeOptimizerGateError("raw optimizer part count changed across native step")
    validated_parts: list[dict[str, Any]] = []
    for part_index, (before_part, after_part) in enumerate(zip(before_parts, after_parts, strict=True)):
        if before_part.get("part") != part_index or after_part.get("part") != part_index:
            raise NativeOptimizerGateError("raw optimizer part ordering changed across native step")
        if before_part.get("parameter_count") != after_part.get("parameter_count"):
            raise NativeOptimizerGateError(f"raw optimizer part {part_index} parameter count changed")
        before_steps = before_part.get("group_steps")
        after_steps = after_part.get("group_steps")
        if not isinstance(before_steps, list) or not before_steps or not isinstance(after_steps, list):
            raise NativeOptimizerGateError(f"raw optimizer part {part_index} has invalid group steps")
        if len(before_steps) != len(after_steps):
            raise NativeOptimizerGateError(f"raw optimizer part {part_index} group count changed")
        for before_step, after_step in zip(before_steps, after_steps, strict=True):
            if before_step.get("group") != after_step.get("group") or before_step.get(
                "parameter_count"
            ) != after_step.get("parameter_count"):
                raise NativeOptimizerGateError(f"raw optimizer part {part_index} group identity changed")
            if before_step.get("present") is not False or before_step.get("value") is not None:
                raise NativeOptimizerGateError(
                    f"raw optimizer part {part_index} counter existed before first native step"
                )
            if after_step.get("present") is not True or after_step.get("value") != 1:
                raise NativeOptimizerGateError(f"raw optimizer part {part_index} counter did not advance exactly 0->1")

        moment_changes: dict[str, Any] = {}
        for state_key in ("exp_avg", "exp_avg_sq"):
            before_moment = before_part.get("moments", {}).get(state_key)
            after_moment = after_part.get("moments", {}).get(state_key)
            if not isinstance(before_moment, dict) or not isinstance(after_moment, dict):
                raise NativeOptimizerGateError(f"raw optimizer part {part_index} lacks {state_key} snapshot")
            for invariant in ("tensor_count", "numel"):
                if before_moment.get(invariant) != after_moment.get(invariant):
                    raise NativeOptimizerGateError(f"raw optimizer part {part_index} {state_key} {invariant} changed")
            if (
                type(before_moment.get("nonzero")) is not int
                or before_moment["nonzero"] != 0
                or isinstance(before_moment.get("l2"), bool)
                or not isinstance(before_moment.get("l2"), (int, float))
                or before_moment["l2"] != 0.0
            ):
                raise NativeOptimizerGateError(
                    f"raw optimizer part {part_index} {state_key} was nonzero before native step"
                )
            before_sha256 = _validated_sha256(
                before_moment.get("sha256"),
                label=f"raw optimizer part {part_index} before {state_key} sha256",
            )
            after_sha256 = _validated_sha256(
                after_moment.get("sha256"),
                label=f"raw optimizer part {part_index} after {state_key} sha256",
            )
            if (
                type(after_moment.get("nonzero")) is not int
                or after_moment["nonzero"] <= 0
                or isinstance(after_moment.get("l2"), bool)
                or not isinstance(after_moment.get("l2"), (int, float))
                or after_moment["l2"] <= 0.0
                or before_sha256 == after_sha256
            ):
                raise NativeOptimizerGateError(
                    f"raw optimizer part {part_index} {state_key} did not change on native step"
                )
            moment_changes[state_key] = {
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "after_nonzero": after_moment["nonzero"],
                "after_l2": after_moment["l2"],
            }
        validated_parts.append(
            {
                "part": part_index,
                "group_count": len(before_steps),
                "counter_advance": "0->1",
                "moments": moment_changes,
            }
        )
    return {"parts": validated_parts}


def _invoke_mcore_production_initializer(model: Any, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    """Invoke only the MCore-installed closure and prove it is state-only."""
    import torch

    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    primary_before = parity._optimizer_raw_parameter_fingerprint(optimizer)
    gradients_before = parity._finalized_gradient_fingerprint(model)
    group_steps_before = parity._optimizer_group_step_metadata(optimizer)
    scheduler_steps_before = getattr(scheduler, "num_steps", None)
    if isinstance(scheduler_steps_before, bool) or not isinstance(scheduler_steps_before, int):
        raise NativeOptimizerGateError(f"scheduler num_steps must be an integer, got {scheduler_steps_before!r}")

    closures: list[dict[str, str]] = []
    for part, raw_optimizer, _ in _raw_optimizer_parts(optimizer):
        production_initializer = getattr(part, "init_state_fn", None)
        if not callable(production_initializer):
            raise NativeOptimizerGateError("MCore optimizer part has no production init_state_fn")
        module = getattr(production_initializer, "__module__", "")
        qualname = getattr(production_initializer, "__qualname__", "")
        if module != "megatron.core.optimizer" or not qualname.endswith(".<locals>.init_state_fn"):
            raise NativeOptimizerGateError(
                "optimizer state initializer is not the production MCore Adam closure: "
                f"module={module!r} qualname={qualname!r}"
            )
        signature = str(inspect.signature(production_initializer))
        if signature not in {"(opt, config=None)", "(opt, config = None)"}:
            raise NativeOptimizerGateError(f"production MCore init_state_fn has unexpected signature {signature}")
        production_initializer(raw_optimizer, getattr(part, "config", None))
        closures.append({"module": module, "qualname": qualname, "signature": signature})

    state_summary = _validate_initialized_state(optimizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    primary_after = parity._optimizer_raw_parameter_fingerprint(optimizer)
    gradients_after = parity._finalized_gradient_fingerprint(model)
    group_steps_after = parity._optimizer_group_step_metadata(optimizer)
    scheduler_steps_after = getattr(scheduler, "num_steps", None)
    if primary_after != primary_before:
        raise NativeOptimizerGateError("production initializer mutated a primary parameter shard")
    if gradients_after != gradients_before:
        raise NativeOptimizerGateError("production initializer mutated a finalized gradient")
    if group_steps_after != group_steps_before:
        raise NativeOptimizerGateError("production initializer advanced optimizer group steps")
    if scheduler_steps_after != scheduler_steps_before:
        raise NativeOptimizerGateError("production initializer advanced the scheduler")
    return {
        "method": "mcore-production-init-state-fn",
        "closures": closures,
        **state_summary,
        "primary_sha256": primary_before["sha256"],
        "gradient_sha256": gradients_before["sha256"],
        "group_steps": group_steps_before,
        "scheduler_steps": scheduler_steps_before,
        "unchanged": True,
    }


def _summarize_native_update(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Require every locally-owned required family to update on this rank."""
    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    families = sorted(set(before["families"]) | set(after["families"]))
    if set(before["parameters"]) != set(after["parameters"]):
        raise NativeOptimizerGateError("optimizer master parameter set changed across native step")
    summary: dict[str, Any] = {}
    changed_parameters = 0
    for name, before_value in before["parameters"].items():
        if before_value["sha256"] != after["parameters"][name]["sha256"]:
            changed_parameters += 1
    for family in families:
        before_family = before["families"].get(family)
        after_family = after["families"].get(family)
        if before_family is None or after_family is None:
            raise NativeOptimizerGateError(f"optimizer family {family!r} appeared or disappeared")
        changed = before_family["sha256"] != after_family["sha256"]
        summary[family] = {
            "changed": changed,
            "before_sha256": before_family["sha256"],
            "after_sha256": after_family["sha256"],
            "tensor_count": before_family["tensor_count"],
            "numel": before_family["numel"],
        }
    if changed_parameters == 0:
        raise NativeOptimizerGateError("native NeMo-RL optimizer step changed no logical FP32 masters")
    required_families = set(parity.REQUIRED_GRADIENT_FAMILIES)
    locally_present_required_families = required_families & set(families)
    unchanged_required_families = {
        family for family in locally_present_required_families if summary[family]["changed"] is not True
    }
    if unchanged_required_families:
        raise NativeOptimizerGateError(
            "native step did not update locally-present required model families: "
            f"{sorted(unchanged_required_families)}"
        )
    return {
        "changed_parameters": changed_parameters,
        "locally_present_required_families": sorted(locally_present_required_families),
        "families": summary,
    }


def _validate_per_rank_required_family_updates(
    per_rank: list[dict[str, Any]], *, required_families: set[str], world_size: int
) -> list[str]:
    """Reject union-based false greens and require each rank's local required families to change."""
    if len(per_rank) != world_size:
        raise NativeOptimizerGateError(f"native optimizer evidence requires {world_size} ranks, got {len(per_rank)}")
    globally_present: set[str] = set()
    seen_ranks: set[int] = set()
    reference_source_package: dict[str, Any] | None = None
    for expected_rank, item in enumerate(per_rank):
        if not isinstance(item, dict) or item.get("rank") != expected_rank:
            raise NativeOptimizerGateError(f"native optimizer evidence rank ordering mismatch at index {expected_rank}")
        if expected_rank in seen_ranks:
            raise NativeOptimizerGateError(f"duplicate native optimizer evidence rank {expected_rank}")
        seen_ranks.add(expected_rank)
        source_binding = item.get("source_binding")
        if not isinstance(source_binding, dict):
            raise NativeOptimizerGateError(f"rank {expected_rank} lacks source-binding evidence")
        source_binding = _validate_source_binding_evidence(source_binding)
        runtime_source_package_sha256 = source_binding["runtime_source_package_sha256"]
        source_package = {
            "repo_root": source_binding.get("repo_root"),
            "outer_package": source_binding.get("outer_package"),
            "materialization": source_binding.get("materialization"),
            "runtime_source_package_sha256": runtime_source_package_sha256,
            "explicit_sources": source_binding.get("explicit_sources"),
            "source_trees": source_binding.get("source_trees"),
            "imported_modules": source_binding.get("imported_modules"),
        }
        if reference_source_package is None:
            reference_source_package = source_package
        elif source_package != reference_source_package:
            raise NativeOptimizerGateError("immutable source package differs across ranks")
        native_step = item.get("native_step")
        if not isinstance(native_step, dict):
            raise NativeOptimizerGateError(f"rank {expected_rank} lacks native-step evidence")
        _validate_raw_optimizer_step(native_step.get("raw_optimizer_before"), native_step.get("raw_optimizer_after"))
        if not isinstance(native_step.get("raw_optimizer_step"), dict):
            raise NativeOptimizerGateError(f"rank {expected_rank} lacks validated raw optimizer-step evidence")
        update = native_step.get("optimizer_update")
        if not isinstance(update, dict):
            raise NativeOptimizerGateError(f"rank {expected_rank} lacks native optimizer update evidence")
        families = update.get("families")
        local_required = update.get("locally_present_required_families")
        if not isinstance(families, dict) or not isinstance(local_required, list):
            raise NativeOptimizerGateError(f"rank {expected_rank} has invalid required-family update evidence")
        for family, record in families.items():
            if (
                not isinstance(family, str)
                or not isinstance(record, dict)
                or set(record)
                != {
                    "changed",
                    "before_sha256",
                    "after_sha256",
                    "tensor_count",
                    "numel",
                }
            ):
                raise NativeOptimizerGateError(f"rank {expected_rank} optimizer family {family!r} record is invalid")
            before_sha256 = _validated_sha256(
                record["before_sha256"],
                label=f"rank {expected_rank} optimizer family {family} before sha256",
            )
            after_sha256 = _validated_sha256(
                record["after_sha256"],
                label=f"rank {expected_rank} optimizer family {family} after sha256",
            )
            _gate_integer(
                record["tensor_count"],
                label=f"rank {expected_rank} optimizer family {family} tensors",
                minimum=1,
            )
            _gate_integer(
                record["numel"],
                label=f"rank {expected_rank} optimizer family {family} numel",
                minimum=1,
            )
            if type(record["changed"]) is not bool or record["changed"] is not (before_sha256 != after_sha256):
                raise NativeOptimizerGateError(
                    f"rank {expected_rank} optimizer family {family!r} change flag is not derived"
                )
        derived_local_required = required_families & set(families)
        if set(local_required) != derived_local_required or len(local_required) != len(set(local_required)):
            raise NativeOptimizerGateError(
                f"rank {expected_rank} locally-present required-family declaration is inconsistent"
            )
        unchanged = {
            family
            for family in derived_local_required
            if not isinstance(families.get(family), dict) or families[family].get("changed") is not True
        }
        if unchanged:
            raise NativeOptimizerGateError(
                f"rank {expected_rank} did not update locally-present required families: {sorted(unchanged)}"
            )
        globally_present.update(derived_local_required)
    missing_families = required_families - globally_present
    if missing_families:
        raise NativeOptimizerGateError(
            f"required model families were absent from every rank: {sorted(missing_families)}"
        )
    return sorted(globally_present)


def _gate_integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise NativeOptimizerGateError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise NativeOptimizerGateError(f"{label} must be at least {minimum}")
    return value


def _validate_gate_batch(value: Any, *, rank: int) -> dict[str, Any]:
    required = {
        "sha256",
        "rows",
        "width",
        "input_lengths",
        "prompt_length",
        "rewards",
        "selected_tokens",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise NativeOptimizerGateError(f"rank {rank} batch evidence has invalid keys")
    if value["sha256"] != GATE_FIXTURE_SHA256:
        raise NativeOptimizerGateError(f"rank {rank} batch does not bind the captured K=4 fixture")
    if _gate_integer(value["rows"], label=f"rank {rank} batch rows") != GATE_BATCH_SIZE:
        raise NativeOptimizerGateError(f"rank {rank} batch must contain exactly four rows")
    width = _gate_integer(value["width"], label=f"rank {rank} batch width", minimum=1)
    prompt_length = _gate_integer(value["prompt_length"], label=f"rank {rank} batch prompt length", minimum=1)
    selected_tokens = _gate_integer(value["selected_tokens"], label=f"rank {rank} selected tokens", minimum=1)
    if selected_tokens != GATE_SELECTED_TOKENS:
        raise NativeOptimizerGateError(f"rank {rank} selected-token count is not the captured fixture")
    input_lengths = value["input_lengths"]
    if (
        not isinstance(input_lengths, list)
        or len(input_lengths) != GATE_BATCH_SIZE
        or any(type(length) is not int or not prompt_length < length <= width for length in input_lengths)
    ):
        raise NativeOptimizerGateError(f"rank {rank} batch input lengths are invalid")
    rewards = value["rewards"]
    if (
        not isinstance(rewards, list)
        or len(rewards) != GATE_BATCH_SIZE
        or any(isinstance(reward, bool) or not isinstance(reward, (int, float)) for reward in rewards)
        or len(set(rewards)) < 2
    ):
        raise NativeOptimizerGateError(f"rank {rank} batch rewards are invalid or degenerate")
    return value


def _validate_gate_batch_preparation(value: Any, *, batch: dict[str, Any], rank: int) -> dict[str, Any]:
    """Require the exact non-fallback K=4 shared-star plan for both worker stages."""
    if not isinstance(value, dict) or set(value) != {"logprob", "train"}:
        raise NativeOptimizerGateError(f"rank {rank} batch-preparation evidence has invalid stages")
    input_lengths = batch["input_lengths"]
    prompt_length = batch["prompt_length"]
    padding_multiple = GATE_TOPOLOGY["tensor_parallel_size"] * GATE_TOPOLOGY["context_parallel_size"] * 2
    padded_lengths = [
        ((length + padding_multiple - 1) // padding_multiple) * padding_multiple for length in input_lengths
    ]
    physical_length = prompt_length + sum(length - prompt_length for length in padded_lengths)
    row_order = sorted(
        range(GATE_BATCH_SIZE),
        key=lambda row: (
            -(padded_lengths[row] - prompt_length),
            -(input_lengths[row] - prompt_length),
            row,
        ),
    )
    if physical_length > GATE_PACKING_TOKENS:
        raise NativeOptimizerGateError(f"rank {rank} captured shared-star plan exceeds packing capacity")
    required_stage_keys = {
        "capacity",
        "microbatches",
        "source_order",
        "plan",
        "worker_num_microbatches",
    }
    required_plan_keys = {
        "execution_units",
        "row_indices",
        "slot_ids",
        "shared_layout",
        "physical_length",
        "capacity",
    }
    for stage in ("logprob", "train"):
        stage_value = value[stage]
        if not isinstance(stage_value, dict) or set(stage_value) != required_stage_keys:
            raise NativeOptimizerGateError(f"rank {rank} {stage} preparation has invalid keys")
        if (
            _gate_integer(stage_value["capacity"], label=f"rank {rank} {stage} capacity") != GATE_PACKING_TOKENS
            or stage_value["microbatches"] is not None
            or stage_value["source_order"] is not None
            or _gate_integer(
                stage_value["worker_num_microbatches"],
                label=f"rank {rank} {stage} worker microbatches",
            )
            != 1
        ):
            raise NativeOptimizerGateError(f"rank {rank} {stage} did not use the DP=1 shared-prefix path")
        plan = stage_value["plan"]
        if not isinstance(plan, dict) or set(plan) != required_plan_keys:
            raise NativeOptimizerGateError(f"rank {rank} {stage} shared-star plan has invalid keys")
        if (
            _gate_integer(plan["execution_units"], label=f"rank {rank} {stage} execution units") != 1
            or plan["row_indices"] != row_order
            or plan["slot_ids"] != [0] * GATE_BATCH_SIZE
            or plan["shared_layout"] is not True
            or _gate_integer(plan["physical_length"], label=f"rank {rank} {stage} physical length") != physical_length
            or _gate_integer(plan["capacity"], label=f"rank {rank} {stage} plan capacity") != GATE_PACKING_TOKENS
        ):
            raise NativeOptimizerGateError(f"rank {rank} {stage} is not the exact non-fallback K=4 shared star")
    return value


def _gate_finite_number(value: Any, *, label: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NativeOptimizerGateError(f"{label} must be a finite number")
    return float(value)


def _validate_gate_metrics(value: Any, *, rank: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"global_loss", "grad_norm", "mtp"}:
        raise NativeOptimizerGateError(f"rank {rank} native-step metric evidence has invalid keys")
    _gate_finite_number(value["global_loss"], label=f"rank {rank} global loss")
    _gate_finite_number(value["grad_norm"], label=f"rank {rank} gradient norm")
    mtp = value["mtp"]
    if not isinstance(mtp, dict) or set(mtp) != GATE_MTP_METRIC_KEYS:
        raise NativeOptimizerGateError(f"rank {rank} native-step metrics do not cover all MTP heads")
    for name, metric in mtp.items():
        _gate_finite_number(metric, label=f"rank {rank} {name}")
    return value


def _validate_gate_optimizer_update(value: Any, *, rank: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "changed_parameters",
        "locally_present_required_families",
        "families",
    }:
        raise NativeOptimizerGateError(f"rank {rank} optimizer update evidence is invalid")
    _gate_integer(value["changed_parameters"], label=f"rank {rank} changed parameters", minimum=1)
    families = value["families"]
    local_required = value["locally_present_required_families"]
    if not isinstance(families, dict) or not families or not isinstance(local_required, list):
        raise NativeOptimizerGateError(f"rank {rank} optimizer family evidence is invalid")
    for family, record in families.items():
        if not isinstance(family, str) or not family:
            raise NativeOptimizerGateError(f"rank {rank} optimizer family name is invalid")
        if not isinstance(record, dict) or set(record) != {
            "changed",
            "before_sha256",
            "after_sha256",
            "tensor_count",
            "numel",
        }:
            raise NativeOptimizerGateError(f"rank {rank} optimizer family {family!r} record is invalid")
        before_sha256 = _validated_sha256(
            record["before_sha256"], label=f"rank {rank} optimizer family {family} before sha256"
        )
        after_sha256 = _validated_sha256(
            record["after_sha256"], label=f"rank {rank} optimizer family {family} after sha256"
        )
        _gate_integer(record["tensor_count"], label=f"rank {rank} optimizer family {family} tensors", minimum=1)
        _gate_integer(record["numel"], label=f"rank {rank} optimizer family {family} numel", minimum=1)
        if type(record["changed"]) is not bool or record["changed"] is not (before_sha256 != after_sha256):
            raise NativeOptimizerGateError(f"rank {rank} optimizer family {family!r} change flag is not derived")
    derived_local_required = sorted(GATE_REQUIRED_FAMILIES & set(families))
    if local_required != derived_local_required:
        raise NativeOptimizerGateError(f"rank {rank} locally-present required-family declaration is inconsistent")
    return value


def _validate_production_initialization(value: Any, *, rank: int) -> dict[str, Any]:
    required = {
        "method",
        "closures",
        "initialized_parameters",
        "bf16_remainder_parameters",
        "state_tensor_numel",
        "primary_sha256",
        "gradient_sha256",
        "group_steps",
        "scheduler_steps",
        "unchanged",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise NativeOptimizerGateError(f"rank {rank} production-initializer evidence has invalid keys")
    if value["method"] != "mcore-production-init-state-fn" or value["unchanged"] is not True:
        raise NativeOptimizerGateError(f"rank {rank} did not use the production MCore initializer")
    closures = value["closures"]
    if not isinstance(closures, list) or not closures:
        raise NativeOptimizerGateError(f"rank {rank} production initializer has no closures")
    for closure in closures:
        if not isinstance(closure, dict) or set(closure) != {"module", "qualname", "signature"}:
            raise NativeOptimizerGateError(f"rank {rank} production closure record is invalid")
        if (
            closure["module"] != "megatron.core.optimizer"
            or not isinstance(closure["qualname"], str)
            or not closure["qualname"].endswith(".<locals>.init_state_fn")
        ):
            raise NativeOptimizerGateError(f"rank {rank} initializer closure is not native MCore")
        if closure["signature"] not in {"(opt, config=None)", "(opt, config = None)"}:
            raise NativeOptimizerGateError(f"rank {rank} initializer closure signature is invalid")
    initialized = _gate_integer(value["initialized_parameters"], label=f"rank {rank} initialized parameters", minimum=1)
    remainders = _gate_integer(value["bf16_remainder_parameters"], label=f"rank {rank} remainder parameters", minimum=1)
    if remainders > initialized:
        raise NativeOptimizerGateError(f"rank {rank} remainder parameter count exceeds initialized state")
    _gate_integer(value["state_tensor_numel"], label=f"rank {rank} state tensor numel", minimum=1)
    _validated_sha256(value["primary_sha256"], label=f"rank {rank} primary sha256")
    _validated_sha256(value["gradient_sha256"], label=f"rank {rank} gradient sha256")
    if not isinstance(value["group_steps"], list) or not value["group_steps"]:
        raise NativeOptimizerGateError(f"rank {rank} initializer group-step evidence is empty")
    _gate_integer(value["scheduler_steps"], label=f"rank {rank} initializer scheduler steps", minimum=0)
    return value


def _validate_rank_evidence_contract(value: Any, *, rank: int) -> dict[str, Any]:
    required = {
        "schema",
        "rank",
        "local_rank",
        "batch",
        "batch_preparation",
        "config_sha256",
        "source_binding",
        "topology",
        "shared_prefix_mode",
        "lazy_state_before_backward",
        "production_initialization",
        "native_step",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise NativeOptimizerGateError(f"rank {rank} evidence has invalid top-level keys")
    evidence_rank = _gate_integer(value["rank"], label=f"rank {rank} evidence rank")
    local_rank = _gate_integer(value["local_rank"], label=f"rank {rank} local rank")
    if value["schema"] != SCHEMA or evidence_rank != rank or local_rank != rank:
        raise NativeOptimizerGateError(f"rank {rank} evidence schema or rank identity is invalid")
    batch = _validate_gate_batch(value["batch"], rank=rank)
    _validate_gate_batch_preparation(value["batch_preparation"], batch=batch, rank=rank)
    _validated_sha256(value["config_sha256"], label=f"rank {rank} config sha256")
    _validate_source_binding_evidence(value["source_binding"])
    topology = value["topology"]
    if not isinstance(topology, dict) or set(topology) != set(GATE_TOPOLOGY):
        raise NativeOptimizerGateError(f"rank {rank} topology has invalid keys")
    if any(
        type(topology[key]) is not type(expected) or topology[key] != expected
        for key, expected in GATE_TOPOLOGY.items()
    ):
        raise NativeOptimizerGateError(f"rank {rank} topology is not TP2/CP2/SP/EP4/MTP5")
    if value["shared_prefix_mode"] != "train":
        raise NativeOptimizerGateError(f"rank {rank} did not run shared-prefix training")
    lazy_state = value["lazy_state_before_backward"]
    if (
        not isinstance(lazy_state, dict)
        or set(lazy_state) != {"parts", "parameters"}
        or _gate_integer(lazy_state["parts"], label=f"rank {rank} lazy optimizer parts", minimum=1) < 1
        or _gate_integer(lazy_state["parameters"], label=f"rank {rank} lazy optimizer parameters", minimum=1) < 1
    ):
        raise NativeOptimizerGateError(f"rank {rank} lazy optimizer evidence is invalid")
    initialization = _validate_production_initialization(value["production_initialization"], rank=rank)
    native_step = value["native_step"]
    required_native = {
        "scheduler_steps_before",
        "scheduler_steps_after",
        "raw_optimizer_before",
        "raw_optimizer_after",
        "raw_optimizer_step",
        "metrics",
        "optimizer_update",
    }
    if not isinstance(native_step, dict) or set(native_step) != required_native:
        raise NativeOptimizerGateError(f"rank {rank} native-step evidence has invalid keys")
    scheduler_before = _gate_integer(
        native_step["scheduler_steps_before"], label=f"rank {rank} scheduler before", minimum=0
    )
    scheduler_after = _gate_integer(
        native_step["scheduler_steps_after"], label=f"rank {rank} scheduler after", minimum=0
    )
    if scheduler_after != scheduler_before + GATE_BATCH_SIZE:
        raise NativeOptimizerGateError(f"rank {rank} scheduler did not advance by global batch size")
    if initialization["scheduler_steps"] != scheduler_before:
        raise NativeOptimizerGateError(f"rank {rank} initializer/native scheduler evidence differs")
    recomputed_raw_step = _validate_raw_optimizer_step(
        native_step["raw_optimizer_before"], native_step["raw_optimizer_after"]
    )
    if native_step["raw_optimizer_step"] != recomputed_raw_step:
        raise NativeOptimizerGateError(f"rank {rank} raw optimizer-step summary is not derived evidence")
    _validate_gate_metrics(native_step["metrics"], rank=rank)
    _validate_gate_optimizer_update(native_step["optimizer_update"], rank=rank)
    return value


def _load_evidence_json(path: Path, *, evidence_root: Path, label: str) -> dict[str, Any]:
    _require_immutable_source_file(path, repo_root=evidence_root, label=label)
    raw, _ = _read_regular_file_bytes(path, label=label)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeOptimizerGateError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise NativeOptimizerGateError(f"{label} must contain a JSON object")
    return value


def _validate_evidence_root(evidence_root: Path) -> dict[str, Any]:
    _require_immutable_directory(evidence_root, repo_root=evidence_root, label="native optimizer evidence root")
    rank_names = [f"native-optimizer.rank{rank}.json" for rank in range(GATE_WORLD_SIZE)]
    expected_inventory = {"EVIDENCE.sha256", *rank_names}
    if {path.name for path in evidence_root.iterdir()} != expected_inventory:
        raise NativeOptimizerGateError("native optimizer evidence root has an invalid inventory")
    manifest_path = evidence_root / "EVIDENCE.sha256"
    _require_immutable_source_file(manifest_path, repo_root=evidence_root, label="native optimizer evidence manifest")
    manifest_raw, _ = _read_regular_file_bytes(manifest_path, label="native optimizer evidence manifest")
    manifest = _parse_outer_payload_manifest(manifest_raw)
    if list(manifest) != rank_names:
        raise NativeOptimizerGateError("EVIDENCE.sha256 must contain sorted rank JSON entries only")
    per_rank: list[dict[str, Any]] = []
    for rank, name in enumerate(rank_names):
        path = evidence_root / name
        raw, _ = _read_regular_file_bytes(path, label=f"rank {rank} evidence")
        if _bytes_sha256(raw) != manifest[name]:
            raise NativeOptimizerGateError(f"rank {rank} evidence digest differs from EVIDENCE.sha256")
        evidence = _load_evidence_json(path, evidence_root=evidence_root, label=f"rank {rank} evidence")
        per_rank.append(_validate_rank_evidence_contract(evidence, rank=rank))

    reference = per_rank[0]
    cross_rank_fields = ("batch", "batch_preparation", "config_sha256", "topology")
    for rank, evidence in enumerate(per_rank[1:], start=1):
        for field in cross_rank_fields:
            if evidence[field] != reference[field]:
                raise NativeOptimizerGateError(f"rank {rank} {field} differs across ranks")
    _validate_per_rank_required_family_updates(
        per_rank,
        required_families=GATE_REQUIRED_FAMILIES,
        world_size=GATE_WORLD_SIZE,
    )
    source_binding = reference["source_binding"]
    result_sha256 = _bytes_sha256(_canonical_json_bytes(per_rank))
    return {
        "result_sha256": result_sha256,
        "outer_package_sha256": source_binding["outer_package"]["manifest_sha256"],
        "runtime_source_package_sha256": source_binding["runtime_source_package_sha256"],
        "source_binding_sha256": source_binding["sha256"],
        "materialization_sha256": source_binding["materialization"]["sha256"],
        "ranks": GATE_WORLD_SIZE,
        "selected_tokens": reference["batch"]["selected_tokens"],
    }


def _require_path(path_value: str, *, kind: str) -> Path:
    path = Path(path_value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise NativeOptimizerGateError(f"cannot resolve {kind} {path}: {error}") from error
    if not path.is_absolute() or path.is_symlink() or path != resolved:
        raise NativeOptimizerGateError(f"{kind} must be an absolute canonical non-symlink path: {path}")
    return path


def _runtime_source_binding(
    repo_root: Path,
    *,
    outer_package_expected: str,
    runtime_source_package_expected: str,
    executed_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind the live gate, NeMo-RL package tree, Bridge, and MCore imports."""
    import megatron.bridge
    import megatron.core.optimizer
    import megatron.training
    import nemo_rl

    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    expected_gate = repo_root / "tools/model_diagnostics/shared_prefix_native_optimizer_gate.py"
    gate_source = Path(__file__).resolve(strict=True)
    if gate_source != expected_gate:
        raise NativeOptimizerGateError(
            "executed gate does not resolve to the declared --repo-root: "
            f"gate={gate_source} expected={expected_gate}"
        )
    expected_parity = repo_root / "tools/model_diagnostics/shared_prefix_captured_citation_parity.py"
    parity_source = Path(inspect.getfile(parity)).resolve(strict=True)
    if parity_source != expected_parity:
        raise NativeOptimizerGateError(
            "imported captured-parity helpers do not resolve to --repo-root: "
            f"module={parity_source} expected={expected_parity}"
        )
    nemo_source = Path(inspect.getfile(nemo_rl)).resolve(strict=True)
    expected_nemo_source = repo_root / "nemo_rl/__init__.py"
    if nemo_source != expected_nemo_source:
        raise NativeOptimizerGateError(
            "imported NeMo-RL package does not resolve to --repo-root: "
            f"module={nemo_source} expected={expected_nemo_source}"
        )

    bridge_source = Path(inspect.getfile(megatron.bridge)).resolve(strict=True)
    expected_bridge_source = repo_root / BRIDGE_PACKAGE_RELATIVE
    if bridge_source != expected_bridge_source:
        raise NativeOptimizerGateError(
            "imported Megatron Bridge package does not resolve to its canonical vendored root: "
            f"module={bridge_source} expected={expected_bridge_source}"
        )
    mcore_source = Path(inspect.getfile(megatron.core.optimizer)).resolve(strict=True)
    expected_mcore_source = repo_root / MCORE_OPTIMIZER_RELATIVE
    if mcore_source != expected_mcore_source:
        raise NativeOptimizerGateError(
            "imported MCore optimizer does not resolve to its canonical vendored root: "
            f"module={mcore_source} expected={expected_mcore_source}"
        )

    outer_package = _outer_package_binding(repo_root, outer_package_expected=outer_package_expected)
    source_package = _immutable_source_package(repo_root)
    runtime_source_package_expected = _require_expected_runtime_source_package(
        source_package["source_package_sha256"], runtime_source_package_expected
    )
    materialization = _runtime_materialization_binding(
        repo_root,
        outer_package_sha256=outer_package["manifest_sha256"],
        runtime_source_package_sha256=runtime_source_package_expected,
    )
    imported_modules = _loaded_module_source_manifest(
        repo_root,
        sys.modules,
        prefixes=("nemo_rl", "megatron.bridge", "megatron.core", "megatron.training", "tools"),
        expected_roots=IMPORTED_PREFIX_ROOTS,
        executed_sources=executed_sources,
    )
    binding_digest = hashlib.sha256()
    binding_digest.update(_canonical_json_bytes({"repo_root": str(repo_root)}))
    binding_digest.update(_canonical_json_bytes(outer_package))
    binding_digest.update(bytes.fromhex(runtime_source_package_expected))
    binding_digest.update(_canonical_json_bytes(source_package["explicit_sources"]))
    binding_digest.update(_canonical_json_bytes(source_package["source_trees"]))
    binding_digest.update(_canonical_json_bytes(materialization))
    binding_digest.update(_canonical_json_bytes(imported_modules))
    result = {
        "repo_root": str(repo_root),
        "outer_package": outer_package,
        "materialization": materialization,
        "runtime_source_package_sha256": source_package["source_package_sha256"],
        "sha256": binding_digest.hexdigest(),
        "explicit_sources": source_package["explicit_sources"],
        "source_trees": source_package["source_trees"],
        "imported_modules": imported_modules,
    }
    _validate_source_binding_evidence(result)
    return result


def _run(arguments: argparse.Namespace) -> None:
    repo_root = _require_path(arguments.repo_root, kind="NeMo-RL repo root")
    if not repo_root.is_dir():
        raise NativeOptimizerGateError(f"NeMo-RL repo root is not a directory: {repo_root}")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or sys.dont_write_bytecode is not True:
        raise NativeOptimizerGateError("run requires PYTHONDONTWRITEBYTECODE=1 active from interpreter startup")
    outer_package_expected = _validated_sha256(
        os.environ.get("OUTER_PACKAGE_EXPECTED"), label="builder OUTER_PACKAGE_EXPECTED"
    )
    runtime_source_package_expected = _validated_sha256(
        os.environ.get("RUNTIME_SOURCE_PACKAGE_EXPECTED"),
        label="builder RUNTIME_SOURCE_PACKAGE_EXPECTED",
    )
    outer_package = _outer_package_binding(repo_root, outer_package_expected=outer_package_expected)
    preflight_source_package = _immutable_source_package(repo_root)
    _require_expected_runtime_source_package(
        preflight_source_package["source_package_sha256"], runtime_source_package_expected
    )
    _runtime_materialization_binding(
        repo_root,
        outer_package_sha256=outer_package["manifest_sha256"],
        runtime_source_package_sha256=runtime_source_package_expected,
    )
    executed_sources = _install_bound_source_importer(repo_root)

    import numpy as np
    import ray
    import torch

    from tools.model_diagnostics import shared_prefix_captured_citation_parity as parity

    if int(os.environ.get("WORLD_SIZE", "0")) != parity.WORLD_SIZE:
        raise NativeOptimizerGateError(f"run requires torchrun WORLD_SIZE={parity.WORLD_SIZE}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= local_rank < parity.WORLD_SIZE:
        raise NativeOptimizerGateError(f"invalid LOCAL_RANK={local_rank}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != parity.WORLD_SIZE:
        raise NativeOptimizerGateError(
            f"run requires exactly four visible CUDA devices, got {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise NativeOptimizerGateError("run requires CUDA BF16 support")

    model_path = _require_path(arguments.model_path, kind="HF model path")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise NativeOptimizerGateError(f"model path is not an HF checkpoint: {model_path}")
    output_dir = _require_path(arguments.output_dir, kind="output directory")
    if not output_dir.is_dir():
        raise NativeOptimizerGateError(f"output directory is not a directory: {output_dir}")

    rows, batch_summary = parity._read_captured_rows(
        Path(arguments.batch), expected_sha256=arguments.expected_batch_sha256
    )
    policy_config, loss_config, config_sha256 = parity._build_policy_and_loss_config(
        str(model_path), repo_root=repo_root, shared_prefix_mode="train"
    )
    batch_preparation = parity._preflight_batch_preparation(
        rows, policy_config=policy_config, shared_prefix_mode="train"
    )

    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.models.megatron.setup import destroy_parallel_state
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    ray.get_gpu_ids = lambda: [local_rank]
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    worker = None
    step_open = False
    try:
        worker = MegatronPolicyWorkerImpl(
            config=policy_config,
            tokenizer=tokenizer,
            init_optimizer=True,
            init_reference_model=False,
            worker_sharding_annotations=parity._worker_sharding(),
        )
        model_config = worker._get_model_config()
        topology = {
            "world_size": torch.distributed.get_world_size(),
            "tensor_parallel_size": model_config.tensor_model_parallel_size,
            "context_parallel_size": model_config.context_parallel_size,
            "sequence_parallel": model_config.sequence_parallel,
            "expert_parallel_size": model_config.expert_model_parallel_size,
            "expert_tensor_parallel_size": model_config.expert_tensor_parallel_size,
            "pipeline_parallel_size": model_config.pipeline_model_parallel_size,
            "mtp_num_layers": model_config.mtp_num_layers,
            "mtp_use_repeated_layer": model_config.mtp_use_repeated_layer,
            "mtp_detach_heads": model_config.mtp_detach_heads,
        }
        if topology != parity._topology():
            raise NativeOptimizerGateError(f"runtime topology mismatch: {topology}")
        if worker._shared_prefix_training_enabled is not True:
            raise NativeOptimizerGateError("worker did not activate shared-prefix training")

        lazy_state = _require_empty_optimizer_state(worker.optimizer)
        scheduler_steps_before = worker.scheduler.num_steps
        loss_fn = ClippedPGLossFn(loss_config)
        worker.begin_train_step(loss_fn=loss_fn, gbs=parity.BATCH_SIZE, mbs=1)
        step_open = True
        train_batch, _ = parity._prepare_batch_for_worker(
            parity._build_batch(rows),
            policy_config=policy_config,
            shared_prefix_mode="train",
            stage="train",
        )
        worker.train_microbatch(train_batch)
        torch.cuda.synchronize()

        production_initialization = _invoke_mcore_production_initializer(
            worker.model, worker.optimizer, worker.scheduler
        )
        raw_optimizer_before = _raw_optimizer_training_state_snapshot(worker.optimizer)
        masters_before = parity._snapshot_optimizer_masters(worker.model, worker.optimizer)
        metrics = worker.finish_train_step()
        step_open = False
        torch.cuda.synchronize()
        raw_optimizer_after = _raw_optimizer_training_state_snapshot(worker.optimizer)
        raw_optimizer_step = _validate_raw_optimizer_step(raw_optimizer_before, raw_optimizer_after)
        masters_after = parity._snapshot_optimizer_masters(worker.model, worker.optimizer)
        update = _summarize_native_update(masters_before, masters_after)
        source_binding = _runtime_source_binding(
            repo_root,
            outer_package_expected=outer_package_expected,
            runtime_source_package_expected=runtime_source_package_expected,
            executed_sources=executed_sources,
        )
        scheduler_steps_after = worker.scheduler.num_steps
        if scheduler_steps_after != scheduler_steps_before + parity.BATCH_SIZE:
            raise NativeOptimizerGateError(
                "native NeMo-RL step did not advance the scheduler by global batch size: "
                f"before={scheduler_steps_before} after={scheduler_steps_after}"
            )

        evidence = {
            "schema": SCHEMA,
            "rank": rank,
            "local_rank": local_rank,
            "batch": batch_summary,
            "batch_preparation": batch_preparation,
            "config_sha256": config_sha256,
            "source_binding": source_binding,
            "topology": topology,
            "shared_prefix_mode": "train",
            "lazy_state_before_backward": lazy_state,
            "production_initialization": production_initialization,
            "native_step": {
                "scheduler_steps_before": scheduler_steps_before,
                "scheduler_steps_after": scheduler_steps_after,
                "raw_optimizer_before": raw_optimizer_before,
                "raw_optimizer_after": raw_optimizer_after,
                "raw_optimizer_step": raw_optimizer_step,
                "metrics": parity._metric_evidence(metrics),
                "optimizer_update": update,
            },
        }
        parity._atomic_json(output_dir / f"native-optimizer.rank{rank}.json", evidence)
        torch.distributed.barrier()
        if rank == 0:
            per_rank = [
                json.loads(
                    _read_regular_file_bytes(
                        output_dir / f"native-optimizer.rank{peer_rank}.json",
                        label=f"rank {peer_rank} native optimizer evidence",
                    )[0],
                    object_pairs_hook=parity._no_duplicate_keys,
                    parse_constant=parity._reject_constant,
                )
                for peer_rank in range(parity.WORLD_SIZE)
            ]
            observed_families = _validate_per_rank_required_family_updates(
                per_rank,
                required_families=set(parity.REQUIRED_GRADIENT_FAMILIES),
                world_size=parity.WORLD_SIZE,
            )
            print(
                "NEMORL_SHARED_PREFIX_NATIVE_OPTIMIZER_GATE_GREEN "
                f"batch_sha256={batch_summary['sha256']} config_sha256={config_sha256} "
                f"ranks={parity.WORLD_SIZE} updated_families={','.join(sorted(observed_families))}",
                flush=True,
            )
    except Exception:
        if step_open and worker is not None:
            worker.abort_train_step()
        raise
    finally:
        if worker is not None:
            worker.shutdown()
        destroy_parallel_state()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--batch", required=True)
    run.add_argument("--expected-batch-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    fingerprint = subparsers.add_parser("fingerprint-package")
    fingerprint.add_argument("--repo-root", required=True)
    fingerprint.add_argument("--outer-package-expected", required=True)
    validate_evidence = subparsers.add_parser("validate-evidence")
    validate_evidence.add_argument("--evidence-root", required=True)
    subparsers.add_parser("self-test-source-contract")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            _run(arguments)
        elif arguments.command == "fingerprint-package":
            repo_root = _require_path(arguments.repo_root, kind="NeMo-RL repo root")
            _outer_package_binding(
                repo_root,
                outer_package_expected=arguments.outer_package_expected,
            )
            source_package = _immutable_source_package(repo_root)
            print(
                "NEMORL_SHARED_PREFIX_NATIVE_OPTIMIZER_PACKAGE_FINGERPRINT_GREEN "
                f"RUNTIME_SOURCE_PACKAGE_EXPECTED={source_package['source_package_sha256']}",
                flush=True,
            )
        elif arguments.command == "validate-evidence":
            evidence_root = _require_path(arguments.evidence_root, kind="native optimizer evidence root")
            result = _validate_evidence_root(evidence_root)
            print(
                "NEMORL_SHARED_PREFIX_NATIVE_OPTIMIZER_EVIDENCE_GREEN "
                f"result_sha256={result['result_sha256']} "
                f"outer_package_sha256={result['outer_package_sha256']} "
                f"runtime_source_package_sha256={result['runtime_source_package_sha256']} "
                f"source_binding_sha256={result['source_binding_sha256']} "
                f"materialization_sha256={result['materialization_sha256']} "
                f"ranks={result['ranks']} selected_tokens={result['selected_tokens']}",
                flush=True,
            )
        else:
            result = _source_contract()
            print(
                "NEMORL_SHARED_PREFIX_NATIVE_OPTIMIZER_SOURCE_CONTRACT_GREEN "
                f"forbidden_direct_calls={result['forbidden_direct_calls']} "
                f"production_closure_calls={result['production_closure_calls']}",
                flush=True,
            )
    except NativeOptimizerGateError as error:
        print(f"NATIVE_OPTIMIZER_GATE_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
