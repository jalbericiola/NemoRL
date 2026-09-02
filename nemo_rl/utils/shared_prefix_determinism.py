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

"""Dependency-neutral constants for shared-prefix deterministic execution."""

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


SHARED_PREFIX_DETERMINISM_ENV_VAR_VALUES: Mapping[str, str] = MappingProxyType(
    {
        "MAMBA_DETERMINISTIC": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
    }
)
SHARED_PREFIX_DETERMINISM_MODEL_OVERRIDE_VALUES: Mapping[str, bool] = MappingProxyType(
    {
        "deterministic_mode": True,
        "cross_entropy_loss_fusion": False,
        "tp_comm_overlap": False,
    }
)
SHARED_PREFIX_FORBIDDEN_DETERMINISM_ENV_VAR_NAMES = frozenset(
    {"TRITON_CACHE_AUTOTUNING"}
)
SHARED_PREFIX_FORBIDDEN_DETERMINISM_ENV_VAR_PREFIXES = ("TRITON_AUTOTUNE_BLOCK",)

SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME = "RESULTS_DIR"
SHARED_PREFIX_DETERMINISM_RECEIPT_DIR_ENV_VAR_NAME = (
    "NRL_SHARED_PREFIX_DETERMINISM_RECEIPT_DIR"
)
SHARED_PREFIX_DETERMINISM_RECEIPT_PATH_ENV_VAR_NAMES = (
    SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME,
    SHARED_PREFIX_DETERMINISM_RECEIPT_DIR_ENV_VAR_NAME,
)
SHARED_PREFIX_DETERMINISM_ATTESTATION_TEMPLATE = (
    "SHARED_PREFIX_DETERMINISM_ATTESTED mode={mode} env_controls=4 "
    "triton_autotune=absent model_overrides=3 torch_deterministic=true "
    "total_controls=8"
)

_SHARED_PREFIX_DETERMINISTIC_MODES = frozenset({"observe", "train"})
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_RECEIPT_OPEN_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
)


def validate_shared_prefix_determinism_receipt_paths(
    *,
    results_dir: str,
    receipt_dir: str,
) -> tuple[Path, Path]:
    """Validate canonical absolute receipt paths and strict containment.

    This check is filesystem-independent so it can run during policy validation.
    Receipt publication separately traverses both directories with ``O_NOFOLLOW``
    to close symlink and time-of-check/time-of-use escape routes.
    """
    results_path = _validate_canonical_absolute_path(
        results_dir,
        name=SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME,
    )
    if results_path == Path("/"):
        raise ValueError(f"{SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME} must not be root")
    receipt_path = _validate_canonical_absolute_path(
        receipt_dir,
        name=SHARED_PREFIX_DETERMINISM_RECEIPT_DIR_ENV_VAR_NAME,
    )
    try:
        relative_receipt_path = receipt_path.relative_to(results_path)
    except ValueError as error:
        raise ValueError(
            f"{SHARED_PREFIX_DETERMINISM_RECEIPT_DIR_ENV_VAR_NAME} must be "
            f"strictly below {SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME}"
        ) from error
    if not relative_receipt_path.parts:
        raise ValueError(
            f"{SHARED_PREFIX_DETERMINISM_RECEIPT_DIR_ENV_VAR_NAME} must be "
            f"strictly below {SHARED_PREFIX_RESULTS_DIR_ENV_VAR_NAME}, not equal to it"
        )
    return results_path, receipt_path


def publish_shared_prefix_determinism_receipt(
    *,
    results_dir: str,
    receipt_dir: str,
    mode: str,
    rank: str,
) -> str:
    """Atomically publish one immutable per-rank determinism receipt.

    The wrapper creates an empty, restart-specific receipt directory. This
    function opens every path component without following symlinks, writes and
    fsyncs a private ``O_EXCL`` temporary inode, then atomically hard-links that
    inode to the final mode/rank name without overwrite. A second publication
    for the same worker fails rather than replacing evidence.

    Returns:
        The exact ASCII marker written to the receipt, for identical logging.
    """
    if mode not in _SHARED_PREFIX_DETERMINISTIC_MODES:
        raise ValueError(
            "shared-prefix deterministic receipt mode must be 'observe' or "
            f"'train', got {mode!r}"
        )
    canonical_rank = _validate_canonical_rank(rank)
    results_path, receipt_path = validate_shared_prefix_determinism_receipt_paths(
        results_dir=results_dir,
        receipt_dir=receipt_dir,
    )
    relative_receipt_path = receipt_path.relative_to(results_path)
    marker = SHARED_PREFIX_DETERMINISM_ATTESTATION_TEMPLATE.format(mode=mode)
    marker_bytes = marker.encode("ascii")
    receipt_name = f"shared_prefix_determinism.{mode}.rank-{canonical_rank}.receipt"
    temporary_name = f".{receipt_name}.tmp"

    results_fd = _open_absolute_directory_without_symlinks(results_path)
    receipt_fd: int | None = None
    temporary_fd: int | None = None
    temporary_created = False
    try:
        receipt_fd = _open_relative_directory_without_symlinks(
            results_fd,
            relative_receipt_path,
        )
        temporary_fd = os.open(
            temporary_name,
            _RECEIPT_OPEN_FLAGS,
            stat.S_IRUSR,
            dir_fd=receipt_fd,
        )
        temporary_created = True
        _write_all(temporary_fd, marker_bytes)
        os.fchmod(temporary_fd, stat.S_IRUSR)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None

        try:
            os.link(
                temporary_name,
                receipt_name,
                src_dir_fd=receipt_fd,
                dst_dir_fd=receipt_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"shared-prefix deterministic receipt already exists: {receipt_name}"
            ) from error
        os.unlink(temporary_name, dir_fd=receipt_fd)
        temporary_created = False
        os.fsync(receipt_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_created and receipt_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=receipt_fd)
            except FileNotFoundError:
                pass
        if receipt_fd is not None:
            os.close(receipt_fd)
        os.close(results_fd)
    return marker


def _validate_canonical_absolute_path(value: str, *, name: str) -> Path:
    """Return one absolute normalized path without traversal components."""
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a null byte")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path, got {value!r}")
    if ".." in path.parts or str(path) != value:
        raise ValueError(
            f"{name} must be a canonical path without '.', '..', duplicate, or "
            f"trailing separators, got {value!r}"
        )
    return path


def _validate_canonical_rank(rank: str) -> str:
    """Validate the Ray-provided global rank used as a unique filename key."""
    if (
        type(rank) is not str
        or not rank
        or not rank.isascii()
        or not rank.isdecimal()
        or str(int(rank)) != rank
    ):
        raise ValueError(
            "shared-prefix deterministic receipt RANK must be a canonical "
            f"nonnegative decimal integer, got {rank!r}"
        )
    return rank


def _open_absolute_directory_without_symlinks(path: Path) -> int:
    """Open an absolute directory by walking every component with no-follow."""
    directory_fd = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
    except OSError:
        os.close(directory_fd)
        raise
    return directory_fd


def _open_relative_directory_without_symlinks(
    parent_fd: int,
    relative_path: Path,
) -> int:
    """Open a descendant directory from a trusted parent descriptor."""
    directory_fd = os.dup(parent_fd)
    try:
        for component in relative_path.parts:
            child_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
    except OSError:
        os.close(directory_fd)
        raise
    return directory_fd


def _write_all(file_descriptor: int, value: bytes) -> None:
    """Write all marker bytes or fail without publishing the final receipt."""
    remaining = memoryview(value)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written == 0:
            raise OSError("short write while publishing deterministic receipt")
        remaining = remaining[written:]
