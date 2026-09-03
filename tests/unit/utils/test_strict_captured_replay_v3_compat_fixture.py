# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nemo_rl.environments.strict_gym_child_runtime import (
    load_finalized_reasoning_score_call_index,
)
from nemo_rl.utils.strict_captured_replay_evidence import (
    validate_captured_replay_evidence_index,
    validate_captured_replay_exit_receipt,
)
from nemo_rl.utils.strict_captured_replay_seal import (
    consume_verified_sealed_result,
    publish_sealed_result,
    verify_sealed_result,
)
from tests.unit.utils.strict_captured_replay_v3_compat_fixture import (
    build_strict_captured_replay_v3_compat_pair,
)


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


def test_two_attempts_share_authority_but_have_distinct_live_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = build_strict_captured_replay_v3_compat_pair(tmp_path, monkeypatch)
    first, second = pair.replay_1, pair.replay_2

    assert first.authenticated_source is pair.authenticated_source
    assert second.authenticated_source is pair.authenticated_source
    assert first.manifest.document["pair"] == second.manifest.document["pair"]
    assert [first.manifest.document["attempt_id"], second.manifest.document["attempt_id"]] == [
        "replay-1",
        "replay-2",
    ]
    assert [first.pre.document["authenticated_job_id"], second.pre.document["authenticated_job_id"]] == [
        "82001",
        "82002",
    ]
    assert first.result_root != second.result_root

    driver_identities = []
    scorer_identities = []
    inventory_digests = []
    for fixture in (first, second):
        score_ref = fixture.outputs["reasoning_score_call_index"]
        score_index, score_index_sha256 = load_finalized_reasoning_score_call_index(
            Path(score_ref["path"]),
            expected_sha256=score_ref["sha256"],
            expected_receipt_root=fixture.result_root / "strict_gym_child_runtime",
            expected_pair_id=fixture.manifest.document["pair_id"],
            expected_job_id=fixture.pre.document["authenticated_job_id"],
        )
        assert score_index_sha256 == score_ref["sha256"]
        assert score_index == json.loads(
            dict(fixture.result_roster)["strict_gym_child_runtime/reasoning-score-call-index.json"]
        )
        assert score_index["quiescence"]["original_process_reaped"] is True
        scorer_identities.append(
            (
                score_index["quiescence"]["pid"],
                score_index["quiescence"]["start_ticks"],
            )
        )
        driver_identities.append(fixture.evidence_index.document["identity"]["driver_process"])

        validate_captured_replay_exit_receipt(
            fixture.exit.document,
            replay_execution_manifest=fixture.manifest.document,
            submission_receipt=fixture.submission.document,
            pre_receipt=fixture.pre.document,
            authenticated_source=pair.authenticated_source,
        )
        validate_captured_replay_evidence_index(
            fixture.evidence_index.document,
            replay_execution_manifest=fixture.manifest.document,
            submission_receipt=fixture.submission.document,
            pre_receipt=fixture.pre.document,
            exit_receipt=fixture.exit.document,
            authenticated_source=pair.authenticated_source,
        )

        _, inventory_sha256 = publish_sealed_result(
            result_root=str(fixture.result_root),
            anchored_sha256=fixture.result_anchors,
        )
        capability = verify_sealed_result(
            result_root=str(fixture.result_root),
            expected_inventory_sha256=inventory_sha256,
        )
        assert (
            consume_verified_sealed_result(
                capability,
                expected_result_root=str(fixture.result_root),
                expected_inventory_sha256=inventory_sha256,
            )
            == fixture.result_roster
        )
        inventory_digests.append(inventory_sha256)

    assert len({json.dumps(item, sort_keys=True) for item in driver_identities}) == 2
    assert len(set(scorer_identities)) == 2
    assert len(set(inventory_digests)) == 2
