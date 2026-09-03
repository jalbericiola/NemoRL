# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

import nemo_rl.utils.strict_captured_replay_manifest_v2 as manifest_module
from nemo_rl.utils.strict_captured_replay_evidence import canonical_ascii_json
from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
    HASH_DOMAIN,
    OUTPUT_V2_KEYS,
    REPLAY_CONTRACT_V2_SCHEMA,
    REPLAY_EVIDENCE_INDEX_V2_SCHEMA,
    REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
    REPLAY_JOB_ARGV_TEMPLATE_V2,
    REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
    REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
    REPLAY_RESULT_INVENTORY_V2_SCHEMA,
    REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA,
    REPLAY_RUNTIME_REQUIREMENTS_V2_SCHEMA,
    REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
    REPLAY_TRANSPORT_CONSUMPTION_V2_SCHEMA,
    ROOT_V2_KEYS,
    AuthenticatedOffSourceCapture,
    AuthenticatedReplayStaticInputs,
    build_replay_execution_manifest_v2,
    load_replay_execution_manifest_v2,
    publish_replay_execution_manifest_v2,
    validate_replay_execution_manifest_v2,
)
from nemo_rl.utils.strict_captured_replay_profiles import (
    StrictCapturedReplayProfile,
    get_strict_captured_replay_profile,
)
from tests.unit.utils.test_strict_captured_replay_manifest import (
    _digest,
    _pair,
    _source_capture,
)

_PROFILE_PAIRS = (
    ("citation", "citation-string-match-v1"),
    ("freeform", "freeform-regex-v1"),
)

_PAIR79_V2_ADDITIONS = {
    "EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256",
    "EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256",
    "EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256",
    "STRICT_PAIR_HOST_PYTHON",
    "STRICT_PAIR_RUNTIME_TOOL_MANIFEST",
}


def test_v2_pair_export_roster_adds_exact_runtime_anchors_without_changing_v1() -> None:
    from nemo_rl.utils.strict_captured_replay_manifest import (
        SLURM_EXPORT_ALLOWED_NAMES as V1_ALLOWED_NAMES,
    )

    v1_names = set(V1_ALLOWED_NAMES)
    v2_names = set(manifest_module.SLURM_EXPORT_ALLOWED_NAMES)
    assert len(V1_ALLOWED_NAMES) == 74
    assert len(manifest_module.SLURM_EXPORT_ALLOWED_NAMES) == 79
    assert v2_names - v1_names == _PAIR79_V2_ADDITIONS
    assert v1_names - v2_names == set()


def _profiled_authority(
    tmp_path: Path,
    *,
    environment: str,
    profile_id: str,
) -> tuple[
    StrictCapturedReplayProfile,
    AuthenticatedOffSourceCapture,
    AuthenticatedReplayStaticInputs,
]:
    profile = get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    root = str(tmp_path)
    pair = _pair(root)
    pair["slurm_export_boundary"]["schema"] = manifest_module.PAIR_SLURM_EXPORT_SCHEMA
    pair["slurm_export_boundary"]["allowed_names"] = list(
        manifest_module.SLURM_EXPORT_ALLOWED_NAMES
    )
    pair["selection"]["environment"] = environment
    fixture_path = f"{pair['source']['root']}/{profile.fixture_path}"
    pair["artifacts"]["fixture"] = {
        "path": fixture_path,
        "rows": profile.fixture_rows,
        "sha256": profile.fixture_sha256,
    }
    pair["execution_environment"]["fixed"]["train_path"] = fixture_path
    pair["execution_environment"]["fixed"]["val_path"] = fixture_path
    pair["selection"]["gym_resources"] = {
        "config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "verifier_source": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
    }
    pair["campaign"] = {
        "generation_seed_base": 42,
        "generation": {
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "vllm_gpu_memory_utilization": 0.1,
        },
    }
    pair["scheduler_submission"]["identity"]["submitter_euid"] = os.geteuid()
    pair_sha256 = hashlib.sha256(canonical_ascii_json(pair) + b"\n").hexdigest()
    source_capture = _source_capture(root)
    source_capture["authenticated_job"] = {
        "comment": (
            f"nemo-rl-strict-pair-v1:off:"
            f"{pair['scheduler_submission']['nonce']}:{pair_sha256}"
        ),
        "job_id": "6787903",
        "job_name": f"off-{environment}-pair-abc",
        "user_id": str(os.geteuid()),
    }
    source = AuthenticatedOffSourceCapture(
        source_capture=source_capture,
        pair_manifest=pair,
        pair_manifest_sha256=pair_sha256,
        pair_submission_receipt={},
        pair_submission_receipt_sha256=_digest("pair-submission"),
        trusted_off_exit_receipt_path=(
            f"{root}/results/off/strict_pair_job_state/6787903-0/receipts/EXIT.json"
        ),
        trusted_off_exit_receipt_sha256=_digest("off-exit"),
        pre_receipt={},
        pre_receipt_sha256=_digest("off-pre"),
        exit_receipt={},
        exit_receipt_sha256=_digest("off-exit"),
        main_ledger={},
        transcript_bundle={},
        transport_bundle={},
        transport_manifest={},
        transport_records=(),
    )
    snapshot = copy.deepcopy(pair["source"]["snapshots"]["on"])
    static = AuthenticatedReplayStaticInputs(
        attempt_id="replay-1",
        container_asset={
            **pair["artifacts"]["container"],
            "owner_uid": 153493,
            "owner_gid": 30,
        },
        source_snapshot=snapshot,
        gym_source_root={
            "snapshot_relative_path": "3rdparty/Gym-workspace/Gym",
            "host_path": (f"{snapshot['path']}/3rdparty/Gym-workspace/Gym"),
            "container_path": "/opt/nemo-rl/3rdparty/Gym-workspace/Gym",
        },
        replay_program={
            name: {"path": path, "sha256": _digest(name)}
            for name, path in manifest_module.REPLAY_PROGRAM_V2_PATHS.items()
        },
        slurm_export_path=(
            f"{root}/results/captured_replay/slurm_exports/pair-abc/replay-1.env"
        ),
        slurm_export_sha256=_digest("replay-export"),
        slurm_export_values=(),
        submission_contract_path=(
            f"{root}/results/captured_replay/replay_submission_state/"
            "pair-abc/replay-1.submission-contract.json"
        ),
        submission_contract_sha256=_digest("replay-contract"),
        submission_contract={
            "submission_nonce": _digest("replay-nonce"),
            "submitter_euid": os.geteuid(),
        },
    )
    return profile, source, static


@pytest.fixture
def isolated_authority(monkeypatch: pytest.MonkeyPatch):
    def install(
        tmp_path: Path,
        *,
        environment: str,
        profile_id: str,
    ):
        profile, source, static = _profiled_authority(
            tmp_path,
            environment=environment,
            profile_id=profile_id,
        )
        monkeypatch.setattr(
            manifest_module,
            "_reload_authenticated_off_source_capture",
            lambda value: value,
        )
        monkeypatch.setattr(
            manifest_module,
            "_load_authenticated_replay_static_inputs_v2",
            lambda **kwargs: static,
        )
        return profile, source, static

    return install


@pytest.mark.parametrize(("environment", "profile_id"), _PROFILE_PAIRS)
def test_manifest_v2_binds_profile_lifecycle_and_inventory(
    tmp_path: Path,
    isolated_authority,
    environment: str,
    profile_id: str,
) -> None:
    profile, source, _ = isolated_authority(
        tmp_path,
        environment=environment,
        profile_id=profile_id,
    )
    document = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    validate_replay_execution_manifest_v2(
        document,
        authenticated_source=source,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )

    assert set(document) == set(ROOT_V2_KEYS)
    assert document["schema"] == REPLAY_EXECUTION_MANIFEST_V2_SCHEMA
    assert document["hash_domain"] == HASH_DOMAIN
    assert document["replay_contract"]["schema"] == REPLAY_CONTRACT_V2_SCHEMA
    assert document["slurm_export_boundary"]["job_argv_template"] == list(
        REPLAY_JOB_ARGV_TEMPLATE_V2
    )
    assert document["slurm_export_boundary"]["job_argv_template"][-4:] == [
        "--environment",
        "{environment}",
        "--profile-id",
        "{profile_id}",
    ]
    assert document["replay_contract"]["program"]["profile_registry"] == {
        "path": "nemo_rl/utils/strict_captured_replay_profiles.py",
        "sha256": _digest("profile_registry"),
    }
    assert document["scorer_profile"]["environment"] == environment
    assert document["scorer_profile"]["profile_id"] == profile_id
    assert document["scorer_profile"]["resource_app"] == {
        "path": profile.resource_app_path,
        "sha256": profile.resource_app_sha256,
    }
    assert document["scorer_profile"]["fixture"] == {
        "path": profile.fixture_path,
        "sha256": profile.fixture_sha256,
        "rows": 5,
    }
    assert set(document["artifacts"]["outputs"]) == set(OUTPUT_V2_KEYS)
    assert "reasoning_score_call_index" not in document["artifacts"]["outputs"]
    assert document["artifacts"]["outputs"]["scorer_call_index"] == {
        "path": (
            f"{tmp_path}/results/captured_replay/replay-1/"
            f"{profile.scorer_terminal_index_path}"
        ),
        "schema": profile.call_index_schema,
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
    }
    inventory = document["artifacts"]["outputs"]["result_inventory"]
    assert inventory["schema"] == REPLAY_RESULT_INVENTORY_V2_SCHEMA
    assert [item["path"] for item in inventory["directories"]] == list(
        profile.result_directories
    )
    assert [item["path"] for item in inventory["files"]] == list(profile.result_files)
    assert [item["schema"] for item in inventory["files"]] == list(
        profile.result_file_schemas
    )
    assert [item["path"] for item in inventory["anchors"]] == [
        relative
        for relative in profile.result_files
        if relative in profile.result_anchor_paths
    ]
    assert document["artifacts"]["outputs"]["evidence_index"]["schema"] == (
        REPLAY_EVIDENCE_INDEX_V2_SCHEMA
    )
    assert (
        document["artifacts"]["outputs"]["transport_consumption"]["schema"]
        == REPLAY_TRANSPORT_CONSUMPTION_V2_SCHEMA
    )
    runtime = document["runtime_attestation_requirements"]
    assert runtime["schema"] == REPLAY_RUNTIME_REQUIREMENTS_V2_SCHEMA
    assert runtime["lifecycle_schemas"] == {
        "submission_receipt": REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        "pre_receipt": REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        "exit_receipt": REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        "runtime_attestation": REPLAY_RUNTIME_ATTESTATION_V2_SCHEMA,
    }
    assert document["scheduler_submission"]["receipt"]["schema"] == (
        REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA
    )


def test_manifest_v2_publish_and_reload_require_same_explicit_pair(
    tmp_path: Path,
    isolated_authority,
) -> None:
    _, source, _ = isolated_authority(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    document = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    output = tmp_path / "manifest-v4.json"
    published, digest = publish_replay_execution_manifest_v2(
        output=output,
        document=document,
        authenticated_source=source,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    loaded, loaded_digest = load_replay_execution_manifest_v2(
        path=published,
        expected_sha256=digest,
        authenticated_source=source,
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    assert loaded == document
    assert loaded_digest == digest
    with pytest.raises(ValueError, match="Pair|profile|identity"):
        load_replay_execution_manifest_v2(
            path=published,
            expected_sha256=digest,
            authenticated_source=source,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )
    with pytest.raises(TypeError, match="expected_environment"):
        validate_replay_execution_manifest_v2(
            document,
            authenticated_source=source,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["scorer_profile"]["resource_app"].__setitem__(
                "sha256", "1" * 64
            ),
            "scorer_profile",
        ),
        (
            lambda value: value["scorer_profile"]["fixture"].__setitem__(
                "sha256", "1" * 64
            ),
            "scorer_profile",
        ),
        (
            lambda value: value["artifacts"]["outputs"]["result_inventory"][
                "files"
            ].reverse(),
            "output contract",
        ),
        (
            lambda value: value["runtime_attestation_requirements"][
                "lifecycle_schemas"
            ].__setitem__(
                "exit_receipt",
                "nemo-rl-strict-captured-replay-job-exit-receipt-v5",
            ),
            "runtime requirements",
        ),
        (
            lambda value: value["artifacts"]["outputs"][
                "scorer_call_index"
            ].__setitem__(
                "path",
                "/tmp/reasoning-score-call-index.json",
            ),
            "output contract",
        ),
        (
            lambda value: value["replay_contract"]["program"].pop("profile_registry"),
            "profiled replay program",
        ),
        (
            lambda value: value["slurm_export_boundary"]["job_argv_template"].pop(),
            "profiled replay Slurm job argv template",
        ),
        (
            lambda value: value["slurm_export_boundary"].__setitem__(
                "schema", "nemo-rl-strict-captured-replay-slurm-export-file-v1"
            ),
            "Slurm export schema",
        ),
    ],
)
def test_manifest_v2_rejects_profile_version_and_roster_poison(
    tmp_path: Path,
    isolated_authority,
    mutate,
    message: str,
) -> None:
    _, source, _ = isolated_authority(
        tmp_path,
        environment="freeform",
        profile_id="freeform-regex-v1",
    )
    document = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment="freeform",
        expected_profile_id="freeform-regex-v1",
    )
    mutate(document)
    with pytest.raises(ValueError, match=message):
        validate_replay_execution_manifest_v2(
            document,
            authenticated_source=source,
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )


class _FixtureDictSubclass(dict):
    pass


@pytest.mark.parametrize(
    "mutation",
    ("cross_profile", "path_subclass", "sha_subclass", "rows_bool", "dict_subclass"),
)
def test_manifest_v2_rejects_pair_fixture_profile_swap_and_subclasses(
    tmp_path: Path,
    isolated_authority,
    mutation: str,
) -> None:
    profile, source, _ = isolated_authority(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    fixture = source.pair_manifest["artifacts"]["fixture"]
    assert type(fixture) is dict
    if mutation == "cross_profile":
        other = get_strict_captured_replay_profile(
            expected_environment="freeform",
            expected_profile_id="freeform-regex-v1",
        )
        fixture["path"] = (
            f"{source.pair_manifest['source']['root']}/{other.fixture_path}"
        )
        fixture["sha256"] = other.fixture_sha256
    elif mutation == "path_subclass":

        class PathSubclass(str):
            pass

        fixture["path"] = PathSubclass(fixture["path"])
    elif mutation == "sha_subclass":

        class ShaSubclass(str):
            pass

        fixture["sha256"] = ShaSubclass(fixture["sha256"])
    elif mutation == "rows_bool":
        fixture["rows"] = True
    else:
        source.pair_manifest["artifacts"]["fixture"] = _FixtureDictSubclass(fixture)
    with pytest.raises(ValueError, match="fixture|types"):
        build_replay_execution_manifest_v2(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )


@pytest.mark.parametrize("mutation", ("pair74", "schema"))
def test_manifest_v2_rejects_legacy_or_cross_schema_source_pair_boundary(
    tmp_path: Path,
    isolated_authority,
    mutation: str,
) -> None:
    from nemo_rl.utils.strict_captured_replay_manifest import (
        SLURM_EXPORT_ALLOWED_NAMES as V1_ALLOWED_NAMES,
    )

    profile, source, _ = isolated_authority(
        tmp_path,
        environment="citation",
        profile_id="citation-string-match-v1",
    )
    boundary = source.pair_manifest["slurm_export_boundary"]
    if mutation == "pair74":
        boundary["allowed_names"] = list(V1_ALLOWED_NAMES)
    else:
        boundary["schema"] = "nemo-rl-strict-slurm-export-file-v2"
    pair_sha256 = hashlib.sha256(
        canonical_ascii_json(source.pair_manifest) + b"\n"
    ).hexdigest()
    source.source_capture["authenticated_job"]["comment"] = (
        f"nemo-rl-strict-pair-v1:off:"
        f"{source.pair_manifest['scheduler_submission']['nonce']}:{pair_sha256}"
    )
    source = replace(source, pair_manifest_sha256=pair_sha256)
    with pytest.raises(ValueError, match="Pair Slurm export|Pair79"):
        build_replay_execution_manifest_v2(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )
