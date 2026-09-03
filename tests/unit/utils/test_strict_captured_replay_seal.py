# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter
from pathlib import Path

import pytest

from nemo_rl.utils import strict_captured_replay_seal as seal
from nemo_rl.utils.strict_captured_replay_seal import (
    RESULT_ANCHOR_ALLOWLIST,
    RESULT_FILE_ALLOWLIST,
    RESULT_FILE_SCHEMA_ALLOWLIST,
    RESULT_INVENTORY_FILENAME,
    RESULT_INVENTORY_SCHEMA,
    StrictCapturedReplaySealError,
    VerifiedSealedResultV1,
    consume_verified_sealed_result,
    publish_sealed_result,
    verify_sealed_result,
)


def _payload(relative: str, *, reaped: bool = True) -> bytes:
    schema = dict(zip(RESULT_FILE_ALLOWLIST, RESULT_FILE_SCHEMA_ALLOWLIST, strict=True))[relative]
    if relative.endswith("reasoning-score-call-index.json"):
        document = {
            "schema": schema,
            "quiescence": {
                "original_process_reaped": reaped,
                "wrapper_returncode": 0,
            },
        }
    else:
        document = {"fixture": relative, "schema": schema}
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")


def _tree(tmp_path: Path, *, reaped: bool = True) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "result"
    child = root / "strict_gym_child_runtime"
    child.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    child.chmod(0o700)
    anchors: dict[str, str] = {}
    for relative in RESULT_FILE_ALLOWLIST:
        path = root / relative
        raw = _payload(relative, reaped=reaped)
        path.write_bytes(raw)
        path.chmod(0o400)
        if relative in RESULT_ANCHOR_ALLOWLIST:
            anchors[relative] = hashlib.sha256(raw).hexdigest()
    return root, anchors


def _rewrite_inventory(root: Path, mutate) -> str:
    path = root / RESULT_INVENTORY_FILENAME
    root.chmod(0o755)
    path.chmod(0o600)
    document = json.loads(path.read_bytes())
    mutate(document)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    path.write_bytes(raw)
    path.chmod(0o400)
    root.chmod(0o555)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def _restore_modes(tmp_path: Path):
    yield
    for current, directories, files in os.walk(tmp_path, topdown=False):
        for filename in files:
            path = Path(current, filename)
            if not path.is_symlink():
                path.chmod(0o600)
        for directory in directories:
            path = Path(current, directory)
            if not path.is_symlink():
                path.chmod(0o700)
        Path(current).chmod(0o700)


def test_publish_and_offline_verify_exact_self_excluded_inventory(
    tmp_path: Path,
) -> None:
    root, anchors = _tree(tmp_path)
    path, digest = publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    capability = verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)
    files = consume_verified_sealed_result(
        capability,
        expected_result_root=str(root),
        expected_inventory_sha256=digest,
    )
    inventory = json.loads((root / RESULT_INVENTORY_FILENAME).read_bytes())

    assert path == f"{root}/{RESULT_INVENTORY_FILENAME}"
    assert type(capability) is VerifiedSealedResultV1
    assert [item[0] for item in files] == list(RESULT_FILE_ALLOWLIST)
    assert inventory["schema"] == RESULT_INVENTORY_SCHEMA
    assert inventory["self_excluded"] == {
        "path": RESULT_INVENTORY_FILENAME,
        "policy": "excluded-from-files-and-totals",
    }
    assert [record["path"] for record in inventory["files"]] == list(RESULT_FILE_ALLOWLIST)
    assert RESULT_INVENTORY_FILENAME not in {record["path"] for record in inventory["files"]}
    assert stat.S_IMODE(os.lstat(root).st_mode) == 0o555
    assert stat.S_IMODE(os.lstat(root / "strict_gym_child_runtime").st_mode) == 0o555
    assert all(stat.S_IMODE(os.lstat(root / relative).st_mode) == 0o400 for relative in RESULT_FILE_ALLOWLIST)


@pytest.mark.parametrize(
    ("poison", "message"),
    [
        ("extra", "extra, missing, or renamed"),
        ("symlink", "regular-file contract"),
        ("writable", "mode-0400"),
        ("hardlink", "single-link"),
        ("fifo", "regular-file contract"),
    ],
)
def test_publication_rejects_filesystem_poison(tmp_path: Path, poison: str, message: str) -> None:
    root, anchors = _tree(tmp_path)
    target = root / "replay-ledger.json"
    if poison == "extra":
        extra = root / "attacker-extra.json"
        extra.write_bytes(b"{}")
        extra.chmod(0o400)
    elif poison == "symlink":
        target.chmod(0o600)
        target.unlink()
        target.symlink_to(root / "transcript-bundle.json")
    elif poison == "writable":
        target.chmod(0o600)
    elif poison == "hardlink":
        target.chmod(0o600)
        target.unlink()
        os.link(root / "transcript-bundle.json", target)
    elif poison == "fifo":
        target.chmod(0o600)
        target.unlink()
        os.mkfifo(target, mode=0o400)
    else:  # pragma: no cover
        raise AssertionError(poison)

    with pytest.raises(StrictCapturedReplaySealError, match=message):
        publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    assert not (root / RESULT_INVENTORY_FILENAME).exists()


def test_publication_requires_terminal_anchor_and_reaped_scorer(tmp_path: Path) -> None:
    root, anchors = _tree(tmp_path, reaped=False)
    with pytest.raises(StrictCapturedReplaySealError, match="was reaped"):
        publish_sealed_result(result_root=str(root), anchored_sha256=anchors)

    root, anchors = _tree(tmp_path / "other")
    anchors["replay-ledger.json"] = "1" * 64
    with pytest.raises(StrictCapturedReplaySealError, match="anchor differs"):
        publish_sealed_result(result_root=str(root), anchored_sha256=anchors)


def test_offline_verifier_rejects_digest_extra_and_postseal_mutation(
    tmp_path: Path,
) -> None:
    root, anchors = _tree(tmp_path)
    _, digest = publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    with pytest.raises(StrictCapturedReplaySealError, match="caller-carried"):
        verify_sealed_result(result_root=str(root), expected_inventory_sha256="1" * 64)

    root.chmod(0o755)
    extra = root / "late-extra"
    extra.write_bytes(b"poison")
    extra.chmod(0o400)
    root.chmod(0o555)
    with pytest.raises(StrictCapturedReplaySealError, match="mode-0555|extra"):
        verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)


def test_verifier_rejects_non_ascii_root_before_filesystem_access() -> None:
    with pytest.raises(StrictCapturedReplaySealError, match="printable-ASCII"):
        verify_sealed_result(
            result_root="/tmp/caf\N{LATIN SMALL LETTER E WITH ACUTE}",
            expected_inventory_sha256="1" * 64,
        )


def test_inventory_aggregate_cap_rejects_before_any_member_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, anchors = _tree(tmp_path)
    publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    digest = _rewrite_inventory(
        root,
        lambda document: document["totals"].__setitem__("file_bytes", seal._MAX_RESULT_BYTES + 1),
    )
    monkeypatch.setattr(
        seal,
        "_read_relative_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("member read happened before aggregate rejection")
        ),
    )
    with pytest.raises(StrictCapturedReplaySealError, match="aggregate totals"):
        verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)


def test_inventory_rejects_boolean_integer_alias_in_publication(
    tmp_path: Path,
) -> None:
    root, anchors = _tree(tmp_path)
    publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    digest = _rewrite_inventory(
        root,
        lambda document: document["publication"].__setitem__("nofollow_reverified", 1),
    )
    with pytest.raises(StrictCapturedReplaySealError, match="publication contract"):
        verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)


def test_publication_rejects_callback_mapping_surface(tmp_path: Path) -> None:
    root, anchors = _tree(tmp_path)

    class ExecutableMapping(dict[str, str]):
        def keys(self):  # type: ignore[override]
            raise AssertionError("mapping callback executed")

    with pytest.raises(StrictCapturedReplaySealError, match="anchor keyset differs"):
        publish_sealed_result(result_root=str(root), anchored_sha256=ExecutableMapping(anchors))


def test_consumer_rejects_unminted_or_mismatched_authority(tmp_path: Path) -> None:
    root, anchors = _tree(tmp_path)
    _, digest = publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    capability = verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)
    with pytest.raises(StrictCapturedReplaySealError, match="verifier-minted"):
        consume_verified_sealed_result(
            tuple(),
            expected_result_root=str(root),
            expected_inventory_sha256=digest,
        )
    with pytest.raises(StrictCapturedReplaySealError, match="identity differs"):
        consume_verified_sealed_result(
            capability,
            expected_result_root=str(root),
            expected_inventory_sha256="1" * 64,
        )
    with pytest.raises(AttributeError, match="immutable"):
        capability.result_root = str(root)  # type: ignore[attr-defined]


def test_publication_rejects_noncanonical_or_wrong_schema_payload(
    tmp_path: Path,
) -> None:
    root, anchors = _tree(tmp_path)
    target = root / "replay-ledger.json"
    target.chmod(0o600)
    raw = b'{"schema":"wrong"}\n'
    target.write_bytes(raw)
    target.chmod(0o400)
    anchors["replay-ledger.json"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(StrictCapturedReplaySealError, match="canonical|schema differs"):
        publish_sealed_result(result_root=str(root), anchored_sha256=anchors)


def test_publisher_reads_each_result_file_once_without_quiescence_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, anchors = _tree(tmp_path)
    calls: Counter[str] = Counter()
    original = seal._read_regular_at

    def counted_read(*args, relative: str, **kwargs):
        calls[relative] += 1
        if calls[relative] > 1:
            raise AssertionError(f"publisher reopened {relative}")
        return original(*args, relative=relative, **kwargs)

    monkeypatch.setattr(seal, "_read_regular_at", counted_read)
    # Isolate publication: terminal offline verification has its own exact-read
    # regression below.
    monkeypatch.setattr(seal, "verify_sealed_result", lambda **kwargs: object())
    monkeypatch.setattr(seal, "consume_verified_sealed_result", lambda *args, **kwargs: ())
    publish_sealed_result(result_root=str(root), anchored_sha256=anchors)

    assert calls == Counter({relative: 1 for relative in RESULT_FILE_ALLOWLIST})


def test_verifier_reads_each_result_file_once_and_returns_same_owned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, anchors = _tree(tmp_path)
    _, digest = publish_sealed_result(result_root=str(root), anchored_sha256=anchors)
    calls: Counter[str] = Counter()
    original = seal._read_regular_at

    def counted_read(*args, relative: str, **kwargs):
        calls[relative] += 1
        if relative in RESULT_FILE_ALLOWLIST and calls[relative] > 1:
            raise AssertionError(f"verifier reopened {relative}")
        return original(*args, relative=relative, **kwargs)

    monkeypatch.setattr(seal, "_read_regular_at", counted_read)
    capability = verify_sealed_result(result_root=str(root), expected_inventory_sha256=digest)
    consumed = consume_verified_sealed_result(
        capability,
        expected_result_root=str(root),
        expected_inventory_sha256=digest,
    )

    assert calls[RESULT_INVENTORY_FILENAME] == 1
    assert all(calls[relative] == 1 for relative in RESULT_FILE_ALLOWLIST)
    score_index = "strict_gym_child_runtime/reasoning-score-call-index.json"
    assert dict(consumed)[score_index] == _payload(score_index)


def test_file_allowlist_is_closed_data_not_a_callback() -> None:
    assert type(RESULT_FILE_ALLOWLIST) is tuple
    assert type(RESULT_FILE_SCHEMA_ALLOWLIST) is tuple
    assert len(RESULT_FILE_SCHEMA_ALLOWLIST) == len(RESULT_FILE_ALLOWLIST)
    assert type(RESULT_ANCHOR_ALLOWLIST) is frozenset
    assert all(type(value) is str for value in RESULT_FILE_ALLOWLIST)
