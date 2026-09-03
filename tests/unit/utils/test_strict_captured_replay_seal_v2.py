# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import nemo_rl.utils.strict_captured_replay_seal_v2 as seal_module
from nemo_rl.utils.strict_captured_replay_profiles import (
    FORMAT_RESULT_FILE_SCHEMAS,
    FORMAT_RESULT_FILES,
    STRICT_CAPTURED_REPLAY_PROFILES,
    StrictCapturedReplayProfile,
    get_strict_captured_replay_profile,
)
from nemo_rl.utils.strict_captured_replay_seal_v2 import (
    RESULT_INVENTORY_FILENAME,
    RESULT_INVENTORY_V2_FILENAME,
    RESULT_INVENTORY_V2_SCHEMA,
    StrictCapturedReplaySealError,
    VerifiedSealedResultV2,
    consume_verified_sealed_result,
    consume_verified_sealed_result_v2,
    publish_sealed_result_v2,
    snapshot_verified_sealed_result_v2,
    verify_sealed_result_v2,
)

_FORMAT_PROFILE_PAIRS = (
    ("citation", "citation-string-match-v1"),
    ("freeform", "freeform-regex-v1"),
)
_REASONING_PROFILE_PAIR = ("reasoning_gym", "reasoning-gym-exact-match-v1")
_PROFILE_PAIRS = tuple((profile.environment, profile.profile_id) for profile in STRICT_CAPTURED_REPLAY_PROFILES)


def _profile_payload(
    profile: StrictCapturedReplayProfile,
    relative: str,
    *,
    reaped: bool = True,
) -> bytes:
    schema = dict(zip(profile.result_files, profile.result_file_schemas, strict=True))[relative]
    if relative == profile.scorer_terminal_index_path:
        document = {
            "environment": profile.environment,
            "schema": schema,
            "quiescence": {
                "original_process_reaped": reaped,
                "wrapper_returncode": 0,
            },
        }
        if profile.environment != "reasoning_gym":
            document["profile_id"] = profile.profile_id
    else:
        document = {
            "environment": profile.environment,
            "fixture": relative,
            "profile_id": profile.profile_id,
            "schema": schema,
        }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")


def _reasoning_profile() -> StrictCapturedReplayProfile:
    try:
        return get_strict_captured_replay_profile(
            expected_environment="reasoning_gym",
            expected_profile_id="reasoning-gym-exact-match-v1",
        )
    except ValueError:
        # Slice D is developed on the exact Slice-A base.  The composed branch
        # resolves this profile directly from Slice B's immutable registry.
        pass
    base = get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    return replace(
        base,
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
        call_schema="nemo-rl-strict-reasoning-score-call-v1",
        closed_schema="nemo-rl-strict-reasoning-score-closed-v1",
        call_index_schema="nemo-rl-strict-reasoning-score-call-index-v1",
        result_files=seal_module.RESULT_FILE_ALLOWLIST,
        result_file_schemas=(
            *FORMAT_RESULT_FILE_SCHEMAS[:3],
            *seal_module.RESULT_FILE_SCHEMA_ALLOWLIST[3:],
        ),
        result_anchor_paths=seal_module.RESULT_ANCHOR_ALLOWLIST,
        scorer_terminal_index_path=("strict_gym_child_runtime/reasoning-score-call-index.json"),
    )


def _profile_tree_from_profile(
    tmp_path: Path,
    *,
    profile: StrictCapturedReplayProfile,
    reaped: bool = True,
) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "result"
    child = root / "strict_gym_child_runtime"
    child.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    child.chmod(0o700)
    anchors: dict[str, str] = {}
    for relative in profile.result_files:
        path = root / relative
        raw = _profile_payload(profile, relative, reaped=reaped)
        path.write_bytes(raw)
        path.chmod(0o400)
        if relative in profile.result_anchor_paths:
            anchors[relative] = hashlib.sha256(raw).hexdigest()
    return root, anchors


def _profile_tree(
    tmp_path: Path,
    *,
    environment: str,
    profile_id: str,
    reaped: bool = True,
) -> tuple[Path, StrictCapturedReplayProfile, dict[str, str]]:
    profile = get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    root, anchors = _profile_tree_from_profile(
        tmp_path,
        profile=profile,
        reaped=reaped,
    )
    return root, profile, anchors


def _rewrite_inventory_v2(root: Path, mutate) -> str:
    path = root / RESULT_INVENTORY_V2_FILENAME
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


def test_profile_registry_is_closed_frozen_and_exact() -> None:
    assert type(STRICT_CAPTURED_REPLAY_PROFILES) is tuple
    assert _PROFILE_PAIRS in (
        _FORMAT_PROFILE_PAIRS,
        (*_FORMAT_PROFILE_PAIRS, _REASONING_PROFILE_PAIR),
    )
    assert len(FORMAT_RESULT_FILES) == 13
    assert len(FORMAT_RESULT_FILE_SCHEMAS) == len(FORMAT_RESULT_FILES)
    assert FORMAT_RESULT_FILE_SCHEMAS[:3] == (
        "nemo-rl-strict-captured-replay-evidence-index-v4",
        "nemo-rl-strict-model-transport-replay-consumption-v3",
        "nemo-rl-strict-captured-replay-step1-ledger-v5",
    )
    citation = get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    assert citation.resource_config_path_name == "citation_format"
    assert citation.disabled_config_path_name == "citation_format_simple_agent"
    assert citation.verifier_type == "string_match"
    assert citation.method == "_verify_string_match"
    assert citation.resource_app_sha256 == "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
    assert citation.resource_config_sha256 == "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
    assert citation.requirements_sha256 == "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
    with pytest.raises(FrozenInstanceError):
        citation.profile_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unsupported"):
        get_strict_captured_replay_profile(
            expected_environment="citation",
            expected_profile_id="freeform-regex-v1",
        )


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_publish_verify_consume_v2_uses_exact_profile_order(
    tmp_path: Path,
    environment: str,
    profile_id: str,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment=environment,
        profile_id=profile_id,
    )
    path, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    capability = verify_sealed_result_v2(
        result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    files = consume_verified_sealed_result_v2(
        capability,
        expected_result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    inventory = json.loads((root / RESULT_INVENTORY_V2_FILENAME).read_bytes())

    assert path == f"{root}/{RESULT_INVENTORY_V2_FILENAME}"
    assert type(capability) is VerifiedSealedResultV2
    assert [relative for relative, _ in files] == list(profile.result_files)
    assert inventory["schema"] == RESULT_INVENTORY_V2_SCHEMA
    assert inventory["environment"] == environment
    assert inventory["profile_id"] == profile_id
    assert [record["path"] for record in inventory["files"]] == list(profile.result_files)
    assert not (root / RESULT_INVENTORY_FILENAME).exists()
    assert stat.S_IMODE(os.lstat(root).st_mode) == 0o555
    assert stat.S_IMODE(os.lstat(root / "strict_gym_child_runtime").st_mode) == 0o555


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_snapshot_v2_returns_fresh_retained_bytes_without_filesystem_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    profile_id: str,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment=environment,
        profile_id=profile_id,
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    capability = verify_sealed_result_v2(
        result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    expected_inventory_raw = (root / RESULT_INVENTORY_V2_FILENAME).read_bytes()
    retained_inventory_raw = object.__getattribute__(
        capability,
        "_VerifiedSealedResultV2__inventory_raw",
    )
    retained_members = object.__getattribute__(
        capability,
        "_VerifiedSealedResultV2__files",
    )

    def forbidden_filesystem_io(*args, **kwargs):
        del args, kwargs
        raise AssertionError("retained-byte snapshot must not reopen the filesystem")

    monkeypatch.setattr(seal_module, "_open_absolute_directory", forbidden_filesystem_io)
    monkeypatch.setattr(seal_module, "_read_relative_file", forbidden_filesystem_io)
    monkeypatch.setattr(seal_module, "_read_nested_sealed", forbidden_filesystem_io)

    snapshots = [snapshot_verified_sealed_result_v2(capability) for _ in range(2)]
    first, second = snapshots
    assert type(first) is dict
    assert set(first) == {
        "environment",
        "profile_id",
        "result_root",
        "inventory",
        "members",
    }
    assert first["environment"] == environment
    assert first["profile_id"] == profile_id
    assert first["result_root"] == str(root)
    assert type(first["inventory"]) is dict
    assert first["inventory"] == {
        "path": f"{root}/{RESULT_INVENTORY_V2_FILENAME}",
        "schema": RESULT_INVENTORY_V2_SCHEMA,
        "sha256": digest,
        "raw": expected_inventory_raw,
    }
    assert type(first["inventory"]["raw"]) is bytes
    assert first["inventory"]["raw"] is not retained_inventory_raw
    assert hashlib.sha256(first["inventory"]["raw"]).hexdigest() == digest
    assert type(first["members"]) is tuple
    assert [relative for relative, _ in first["members"]] == list(profile.result_files)
    assert all(
        type(item) is tuple and len(item) == 2 and type(item[0]) is str and type(item[1]) is bytes
        for item in first["members"]
    )
    assert all(
        returned[1] == retained[1] and returned[1] is not retained[1]
        for returned, retained in zip(first["members"], retained_members, strict=True)
    )
    assert first is not second
    assert first["inventory"] is not second["inventory"]
    assert first["inventory"]["raw"] is not second["inventory"]["raw"]
    assert first["members"] is not second["members"]
    assert all(left[1] is not right[1] for left, right in zip(first["members"], second["members"], strict=True))

    first["inventory"]["path"] = "/changed"
    first["members"] = ()
    assert second["inventory"]["path"] == f"{root}/{RESULT_INVENTORY_V2_FILENAME}"
    assert len(second["members"]) == 13
    with pytest.raises(StrictCapturedReplaySealError, match="exact V2 verifier-minted"):
        snapshot_verified_sealed_result_v2(object())


def test_v2_verifier_stable_reads_inventory_and_every_member_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    calls: Counter[str] = Counter()
    original = seal_module._read_regular_at

    def counted_read(*args, relative: str, **kwargs):
        calls[relative] += 1
        return original(*args, relative=relative, **kwargs)

    monkeypatch.setattr(seal_module, "_read_regular_at", counted_read)
    verify_sealed_result_v2(
        result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )

    assert calls == Counter(
        {
            RESULT_INVENTORY_V2_FILENAME: 2,
            **{relative: 2 for relative in profile.result_files},
        }
    )


def test_v2_verifier_rejects_member_mutated_after_its_first_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    target_relative = profile.result_files[0]
    target = root / target_relative
    replacement = json.loads(target.read_bytes())
    replacement["race_mutation"] = True
    replacement_raw = json.dumps(
        replacement,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    original = seal_module._read_regular_at
    target_reads = 0

    def mutate_after_first_read(*args, relative: str, **kwargs):
        nonlocal target_reads
        raw, metadata = original(*args, relative=relative, **kwargs)
        if relative == target_relative:
            target_reads += 1
            if target_reads == 1:
                target.chmod(0o600)
                target.write_bytes(replacement_raw)
                target.chmod(0o400)
        return raw, metadata

    monkeypatch.setattr(seal_module, "_read_regular_at", mutate_after_first_read)
    with pytest.raises(
        StrictCapturedReplaySealError,
        match="changed between verification reads",
    ):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
    assert target_reads == 2


def test_v2_verifier_rejects_inventory_mutated_after_its_second_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    inventory_path = root / RESULT_INVENTORY_V2_FILENAME
    replacement = json.loads(inventory_path.read_bytes())
    replacement["race_mutation"] = True
    replacement_raw = json.dumps(
        replacement,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    target_relative = profile.result_files[0]
    original = seal_module._read_pinned_profile_member_v2
    target_reads = 0

    def mutate_after_inventory_second_read(root_fd, child_fd, relative):
        nonlocal target_reads
        raw, metadata = original(root_fd, child_fd, relative)
        if relative == target_relative:
            target_reads += 1
            if target_reads == 2:
                inventory_path.chmod(0o600)
                inventory_path.write_bytes(replacement_raw)
                inventory_path.chmod(0o400)
        return raw, metadata

    monkeypatch.setattr(
        seal_module,
        "_read_pinned_profile_member_v2",
        mutate_after_inventory_second_read,
    )
    with pytest.raises(
        StrictCapturedReplaySealError,
        match="inventory changed after its second stable read",
    ):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
    assert target_reads == 2


def test_v2_verifier_rejects_byte_identical_child_name_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    child = root / "strict_gym_child_runtime"
    replacement = tmp_path / "replacement-child"
    displaced = tmp_path / "displaced-child"
    shutil.copytree(child, replacement)
    original = seal_module._read_regular_at
    inventory_reads = 0

    def swap_after_inventory_read(*args, relative: str, **kwargs):
        nonlocal inventory_reads
        raw, metadata = original(*args, relative=relative, **kwargs)
        if relative == RESULT_INVENTORY_V2_FILENAME:
            inventory_reads += 1
            if inventory_reads == 1:
                root.chmod(0o755)
                in_root_displaced = root / "displaced-child"
                child.rename(in_root_displaced)
                replacement.chmod(0o755)
                replacement.rename(child)
                child.chmod(0o555)
                in_root_displaced.chmod(0o755)
                in_root_displaced.rename(displaced)
                displaced.chmod(0o555)
                root.chmod(0o555)
        return raw, metadata

    monkeypatch.setattr(seal_module, "_read_regular_at", swap_after_inventory_read)
    with pytest.raises(
        StrictCapturedReplaySealError,
        match="strict Gym result directory changed canonical name",
    ):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
    assert inventory_reads == 2


def test_v2_verifier_rejects_byte_identical_canonical_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    replacement = tmp_path / "replacement-root"
    displaced = tmp_path / "displaced-root"
    shutil.copytree(root, replacement)
    replacement.chmod(0o755)
    original = seal_module._read_regular_at
    inventory_reads = 0

    def swap_after_inventory_read(*args, relative: str, **kwargs):
        nonlocal inventory_reads
        raw, metadata = original(*args, relative=relative, **kwargs)
        if relative == RESULT_INVENTORY_V2_FILENAME:
            inventory_reads += 1
            if inventory_reads == 1:
                root.rename(displaced)
                replacement.rename(root)
                root.chmod(0o555)
        return raw, metadata

    monkeypatch.setattr(seal_module, "_read_regular_at", swap_after_inventory_read)
    with pytest.raises(
        StrictCapturedReplaySealError,
        match="result root changed canonical path",
    ):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
    assert inventory_reads == 2


def test_v2_never_autodetects_or_accepts_a_different_profile(tmp_path: Path) -> None:
    root, _, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    with pytest.raises(TypeError, match="expected_environment"):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
        )  # type: ignore[call-arg]
    with pytest.raises(StrictCapturedReplaySealError, match="identity"):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )


def test_v2_inventory_identity_and_capability_type_are_bound(
    tmp_path: Path,
) -> None:
    root, _, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    capability = verify_sealed_result_v2(
        result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    with pytest.raises(StrictCapturedReplaySealError, match="V2 verifier-minted"):
        consume_verified_sealed_result_v2(
            tuple(),
            expected_result_root=str(root),
            expected_inventory_sha256=digest,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )
    with pytest.raises(StrictCapturedReplaySealError, match="verifier-minted"):
        consume_verified_sealed_result(
            capability,
            expected_result_root=str(root),
            expected_inventory_sha256=digest,
        )
    changed_digest = _rewrite_inventory_v2(
        root,
        lambda document: document.__setitem__("profile_id", "freeform-regex-v1"),
    )
    with pytest.raises(StrictCapturedReplaySealError, match="identity"):
        verify_sealed_result_v2(
            result_root=str(root),
            expected_inventory_sha256=changed_digest,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_v2_rejects_unreaped_terminal_and_out_of_profile_file(
    tmp_path: Path,
) -> None:
    root, _, anchors = _profile_tree(
        tmp_path,
        environment="freeform",
        profile_id="freeform-regex-v1",
        reaped=False,
    )
    with pytest.raises(StrictCapturedReplaySealError, match="was reaped"):
        publish_sealed_result_v2(
            result_root=str(root),
            anchored_sha256=anchors,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )

    other_root, _, other_anchors = _profile_tree(
        tmp_path / "other",
        environment="freeform",
        profile_id="freeform-regex-v1",
    )
    extra = other_root / "strict_gym_child_runtime" / "reasoning-score-call-index.json"
    extra.write_bytes(b"{}")
    extra.chmod(0o400)
    with pytest.raises(StrictCapturedReplaySealError, match="exact allowlist"):
        publish_sealed_result_v2(
            result_root=str(other_root),
            anchored_sha256=other_anchors,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )


def test_v2_rejects_cross_profile_scorer_call_index(tmp_path: Path) -> None:
    root, profile, anchors = _profile_tree(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    terminal = root / profile.scorer_terminal_index_path
    terminal.chmod(0o600)
    document = json.loads(terminal.read_bytes())
    document["environment"] = "freeform"
    document["profile_id"] = "freeform-regex-v1"
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
    terminal.write_bytes(raw)
    terminal.chmod(0o400)
    anchors[profile.scorer_terminal_index_path] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(
        StrictCapturedReplaySealError,
        match="scorer call-index environment/profile identity differs",
    ):
        publish_sealed_result_v2(
            result_root=str(root),
            anchored_sha256=anchors,
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_v2_reasoning_terminal_uses_only_outer_profile_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _reasoning_profile()
    root, anchors = _profile_tree_from_profile(tmp_path, profile=profile)

    def admitted_profile(*, expected_environment: str, expected_profile_id: str):
        assert expected_environment == profile.environment
        assert expected_profile_id == profile.profile_id
        return profile

    monkeypatch.setattr(
        seal_module,
        "_strict_captured_replay_profile",
        admitted_profile,
    )
    _, digest = publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    verify_sealed_result_v2(
        result_root=str(root),
        expected_inventory_sha256=digest,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )

    inventory = json.loads((root / RESULT_INVENTORY_V2_FILENAME).read_bytes())
    terminal = json.loads((root / profile.scorer_terminal_index_path).read_bytes())
    assert inventory["environment"] == "reasoning_gym"
    assert inventory["profile_id"] == "reasoning-gym-exact-match-v1"
    assert terminal["environment"] == "reasoning_gym"
    assert terminal["schema"] == "nemo-rl-strict-reasoning-score-call-index-v1"
    assert "profile_id" not in terminal


def test_v2_scorer_terminal_dispatch_rejects_cross_shape_and_profile_poison() -> None:
    reasoning_profile = _reasoning_profile()
    format_profile = get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    quiescence = {"original_process_reaped": True, "wrapper_returncode": 0}
    reasoning_terminal = {
        "environment": "reasoning_gym",
        "schema": "nemo-rl-strict-reasoning-score-call-index-v1",
        "quiescence": quiescence,
    }
    format_terminal = {
        "environment": "citation",
        "profile_id": "citation-string-match-v1",
        "schema": "nemo-rl-strict-format-verification-call-index-v1",
        "quiescence": quiescence,
    }

    def payload(document: dict[str, object]) -> bytes:
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")

    seal_module._validate_reaped_scorer_bytes(
        payload(reasoning_terminal),
        profile=reasoning_profile,
    )
    seal_module._validate_reaped_scorer_bytes(
        payload(format_terminal),
        profile=format_profile,
    )

    reasoning_with_inner_profile = dict(reasoning_terminal)
    reasoning_with_inner_profile["profile_id"] = reasoning_profile.profile_id
    wrong_outer_reasoning_profile = replace(
        reasoning_profile,
        profile_id="citation-string-match-v1",
    )
    format_without_inner_profile = dict(format_terminal)
    del format_without_inner_profile["profile_id"]
    poisons = (
        (format_terminal, reasoning_profile),
        (reasoning_terminal, format_profile),
        (reasoning_with_inner_profile, reasoning_profile),
        (reasoning_terminal, wrong_outer_reasoning_profile),
        (format_without_inner_profile, format_profile),
    )
    for terminal, profile in poisons:
        with pytest.raises(StrictCapturedReplaySealError, match="profile|authority"):
            seal_module._validate_reaped_scorer_bytes(
                payload(terminal),
                profile=profile,
            )


@pytest.mark.parametrize(
    "raw",
    (
        b'{"environment":"reasoning_gym","quiescence":{"original_process_reaped":true,"wrapper_returncode":-0.0},"schema":"nemo-rl-strict-reasoning-score-call-index-v1"}',
        b'{"environment":"reasoning_gym","quiescence":{"original_process_reaped":true,"wrapper_returncode":0},"schema":"nemo-rl-strict-reasoning-score-call-index-v1"}\n',
    ),
)
def test_v2_reasoning_terminal_rejects_negative_zero_and_noncanonical_framing(
    raw: bytes,
) -> None:
    with pytest.raises(StrictCapturedReplaySealError):
        seal_module._validate_reaped_scorer_bytes(
            raw,
            profile=_reasoning_profile(),
        )
