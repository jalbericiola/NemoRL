# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

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
    verify_sealed_result_v2,
)

_PROFILE_PAIRS = (
    ("citation", "citation-string-match-v1"),
    ("freeform", "freeform-regex-v1"),
)


def _profile_payload(
    profile: StrictCapturedReplayProfile,
    relative: str,
    *,
    reaped: bool = True,
) -> bytes:
    schema = dict(zip(profile.result_files, profile.result_file_schemas, strict=True))[
        relative
    ]
    if relative == profile.scorer_terminal_index_path:
        document = {
            "environment": profile.environment,
            "profile_id": profile.profile_id,
            "schema": schema,
            "quiescence": {
                "original_process_reaped": reaped,
                "wrapper_returncode": 0,
            },
        }
    else:
        document = {
            "environment": profile.environment,
            "fixture": relative,
            "profile_id": profile.profile_id,
            "schema": schema,
        }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")


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
    assert (
        tuple(
            (profile.environment, profile.profile_id)
            for profile in STRICT_CAPTURED_REPLAY_PROFILES
        )
        == _PROFILE_PAIRS
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
    assert (
        citation.resource_app_sha256
        == "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
    )
    assert (
        citation.resource_config_sha256
        == "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
    )
    assert (
        citation.requirements_sha256
        == "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
    )
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
    assert [record["path"] for record in inventory["files"]] == list(
        profile.result_files
    )
    assert not (root / RESULT_INVENTORY_FILENAME).exists()
    assert stat.S_IMODE(os.lstat(root).st_mode) == 0o555
    assert stat.S_IMODE(os.lstat(root / "strict_gym_child_runtime").st_mode) == 0o555


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
