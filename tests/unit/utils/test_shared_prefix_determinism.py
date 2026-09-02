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

import os
import stat

import pytest

from nemo_rl.utils import shared_prefix_determinism as determinism


@pytest.mark.parametrize("mode", ["observe", "train"])
def test_publish_determinism_receipt_is_exact_immutable_evidence(
    tmp_path,
    mode: str,
) -> None:
    results_dir = tmp_path / "results"
    receipt_dir = results_dir / "shared_prefix_determinism_receipts" / "123-0"
    receipt_dir.mkdir(parents=True, mode=0o700)

    marker = determinism.publish_shared_prefix_determinism_receipt(
        results_dir=str(results_dir),
        receipt_dir=str(receipt_dir),
        mode=mode,
        rank="2",
    )

    expected_marker = (
        "SHARED_PREFIX_DETERMINISM_ATTESTED "
        f"mode={mode} env_controls=4 triton_autotune=absent "
        "model_overrides=3 torch_deterministic=true total_controls=8"
    )
    receipt_path = receipt_dir / f"shared_prefix_determinism.{mode}.rank-2.receipt"
    assert marker == expected_marker
    assert receipt_path.read_bytes() == expected_marker.encode("ascii")
    assert not receipt_path.read_bytes().endswith(b"\n")
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert [path.name for path in receipt_dir.iterdir()] == [receipt_path.name]


def test_publish_determinism_receipt_uses_exclusive_atomic_operations(
    tmp_path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    receipt_dir = results_dir / "receipts" / "123-0"
    receipt_dir.mkdir(parents=True)
    real_open = determinism.os.open
    real_link = determinism.os.link
    temporary_open_flags: list[int] = []
    link_calls: list[tuple[str, str, bool]] = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        if str(path).endswith(".tmp"):
            temporary_open_flags.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def recording_link(
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=True,
    ):
        link_calls.append((source, destination, follow_symlinks))
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(determinism.os, "open", recording_open)
    monkeypatch.setattr(determinism.os, "link", recording_link)

    determinism.publish_shared_prefix_determinism_receipt(
        results_dir=str(results_dir),
        receipt_dir=str(receipt_dir),
        mode="train",
        rank="0",
    )

    assert len(temporary_open_flags) == 1
    assert temporary_open_flags[0] & os.O_EXCL
    assert temporary_open_flags[0] & os.O_NOFOLLOW
    assert link_calls == [
        (
            ".shared_prefix_determinism.train.rank-0.receipt.tmp",
            "shared_prefix_determinism.train.rank-0.receipt",
            False,
        )
    ]


def test_publish_determinism_receipt_collision_fails_without_overwrite(
    tmp_path,
) -> None:
    results_dir = tmp_path / "results"
    receipt_dir = results_dir / "receipts" / "123-0"
    receipt_dir.mkdir(parents=True)
    receipt_path = receipt_dir / "shared_prefix_determinism.train.rank-0.receipt"

    determinism.publish_shared_prefix_determinism_receipt(
        results_dir=str(results_dir),
        receipt_dir=str(receipt_dir),
        mode="train",
        rank="0",
    )
    original = receipt_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        determinism.publish_shared_prefix_determinism_receipt(
            results_dir=str(results_dir),
            receipt_dir=str(receipt_dir),
            mode="train",
            rank="0",
        )

    assert receipt_path.read_bytes() == original
    assert not list(receipt_dir.glob("*.tmp"))


def test_publish_determinism_receipt_write_failure_leaves_no_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    results_dir = tmp_path / "results"
    receipt_dir = results_dir / "receipts" / "123-0"
    receipt_dir.mkdir(parents=True)

    def fail_write(_file_descriptor, _value):
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(determinism.os, "write", fail_write)

    with pytest.raises(OSError, match="injected receipt write failure"):
        determinism.publish_shared_prefix_determinism_receipt(
            results_dir=str(results_dir),
            receipt_dir=str(receipt_dir),
            mode="train",
            rank="0",
        )

    assert list(receipt_dir.iterdir()) == []


def test_publish_determinism_receipt_rejects_symlinked_receipt_directory(
    tmp_path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (results_dir / "receipts").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(OSError):
        determinism.publish_shared_prefix_determinism_receipt(
            results_dir=str(results_dir),
            receipt_dir=str(results_dir / "receipts"),
            mode="train",
            rank="0",
        )

    assert list(outside_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("receipt_value", "expected_error"),
    [
        ("relative/receipts", "must be an absolute path"),
        ("{results}", "strictly below"),
        ("{outside}", "strictly below"),
        ("{results}/child/../receipts", "canonical path"),
        ("{results}/receipts/", "canonical path"),
    ],
)
def test_validate_determinism_receipt_paths_rejects_escape_or_alias(
    tmp_path,
    receipt_value: str,
    expected_error: str,
) -> None:
    results_dir = tmp_path / "results"
    outside_dir = tmp_path / "outside"
    value = receipt_value.format(results=results_dir, outside=outside_dir)

    with pytest.raises(ValueError, match=expected_error):
        determinism.validate_shared_prefix_determinism_receipt_paths(
            results_dir=str(results_dir),
            receipt_dir=value,
        )


def test_validate_determinism_receipt_paths_rejects_root_results_dir() -> None:
    with pytest.raises(ValueError, match="RESULTS_DIR must not be root"):
        determinism.validate_shared_prefix_determinism_receipt_paths(
            results_dir="/",
            receipt_dir="/tmp/receipts",
        )


@pytest.mark.parametrize("rank", ["", "-1", "01", "1.0", "１", None])
def test_publish_determinism_receipt_rejects_noncanonical_rank(
    tmp_path,
    rank: object,
) -> None:
    results_dir = tmp_path / "results"
    receipt_dir = results_dir / "receipts"
    receipt_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="RANK must be a canonical"):
        determinism.publish_shared_prefix_determinism_receipt(
            results_dir=str(results_dir),
            receipt_dir=str(receipt_dir),
            mode="train",
            rank=rank,
        )
