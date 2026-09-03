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
    STRICT_CAPTURED_REPLAY_PROFILES,
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
    ("reasoning_gym", "reasoning-gym-exact-match-v1"),
)

_FORMAT_WIRE_SHA256 = {
    "citation": {
        "launcher": "fc58b59291668690fa69143b8e82b694c436344f717fce0f8b3aeb15ee8883a1",
        "runtime": "fa5ac912c62ad0b51b0c36ca087d3c76ccb70c4f375fab75e271738cd3272f53",
        "scorer_profile": "5b4e35fe3dcf039a907827a7f984ec26e400d82e616151f2b22a68f8318acbbe",
    },
    "freeform": {
        "launcher": "c6a0e7e3097b1fa4c53d61a72a366b555a4c2b8c5af411809633f76f8023ab06",
        "runtime": "fe5550af5e4597e5f5fabdf92bfa4af4e68c6e9e60d43e508f5d19f1f9449dde",
        "scorer_profile": "9feda72c66aeeb237c5e73d16e1b586dbb4ce3bce37cf1077bcae385862566ea",
    },
}

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


def test_reasoning_gym_profile_has_distinct_pair_and_scorer_config_authorities() -> None:
    assert tuple(
        (profile.environment, profile.profile_id)
        for profile in STRICT_CAPTURED_REPLAY_PROFILES
    ) == _PROFILE_PAIRS
    profile = get_strict_captured_replay_profile(
        expected_environment="reasoning_gym",
        expected_profile_id="reasoning-gym-exact-match-v1",
    )

    assert profile.verifier_type == "score_answer"
    assert profile.method == "KnightsKnavesDataset.score_answer"
    assert profile.disabled_config_path_name == "reasoning_gym_simple_agent"
    assert profile.resource_config_path_name == "reasoning_gym"
    assert profile.resource_config_path == ("resources_servers/reasoning_gym/configs/reasoning_gym.yaml")
    assert profile.resource_config_sha256 == ("bdbb459a4a920bc47cf84b1d7dc30aeaa9be35cf0dfac09c77879e45b62a52ab")
    assert profile.scorer_config_path_name == "resources_only"
    assert profile.scorer_config_path == ("resources_servers/reasoning_gym/configs/resources_only.yaml")
    assert profile.scorer_config_sha256 == ("e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68")
    assert profile.resource_app_sha256 == ("3a35c5d27392dae05499ceefac04e9c32ad963b51a54d77bb470ee59b1fe3127")
    assert profile.requirements_sha256 == ("b00b45db433d797d8a5c5c5602f24ab94d9d5620d83b4bef21fbee851287d411")
    assert profile.fixture_sha256 == ("da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce")
    assert profile.fixture_rows == 5
    assert len(profile.result_files) == 13
    assert profile.result_files[4:10] == (
        "strict_gym_child_runtime/reasoning-score-call-00000001.json",
        "strict_gym_child_runtime/reasoning-score-call-00000002.json",
        "strict_gym_child_runtime/reasoning-score-call-00000003.json",
        "strict_gym_child_runtime/reasoning-score-call-00000004.json",
        "strict_gym_child_runtime/reasoning-score-call-index.json",
        "strict_gym_child_runtime/reasoning-score-closed.json",
    )
    assert profile.result_file_schemas[:4] == (
        "nemo-rl-strict-captured-replay-evidence-index-v4",
        "nemo-rl-strict-model-transport-replay-consumption-v3",
        "nemo-rl-strict-captured-replay-step1-ledger-v5",
        "nemo-rl-strict-gym-child-index-v1",
    )
    assert profile.scorer_terminal_index_path == ("strict_gym_child_runtime/reasoning-score-call-index.json")


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    _PROFILE_PAIRS[:2],
)
def test_format_profile_wire_projections_are_byte_identical_to_slice_a(
    environment: str,
    profile_id: str,
) -> None:
    profile = get_strict_captured_replay_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    projections = {
        "scorer_profile": manifest_module._scorer_profile_v2(profile),
        "launcher": manifest_module._gym_scorer_launcher_v2(profile),
        "runtime": manifest_module._gym_scorer_runtime_v2(profile),
    }

    assert profile.scorer_config_path_name == profile.resource_config_path_name
    assert profile.scorer_config_path == profile.resource_config_path
    assert profile.scorer_config_sha256 == profile.resource_config_sha256
    assert set(projections["scorer_profile"]) == {
        "call_index_schema",
        "call_schema",
        "closed_schema",
        "disabled_config_path_name",
        "environment",
        "fixture",
        "method",
        "profile_id",
        "requirements",
        "resource_app",
        "resource_config",
        "resource_config_path_name",
        "verifier_type",
    }
    actual_sha256 = {
        name: hashlib.sha256(canonical_ascii_json(value)).hexdigest() for name, value in projections.items()
    }
    assert actual_sha256 == _FORMAT_WIRE_SHA256[environment]


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


def test_reasoning_gym_manifest_keeps_pair_config_full_and_launches_resources_only(
    tmp_path: Path,
    isolated_authority,
) -> None:
    profile, source, _ = isolated_authority(
        tmp_path,
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
    )
    document = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )

    full_config = {
        "path": profile.resource_config_path,
        "sha256": profile.resource_config_sha256,
    }
    scorer_config = {
        "path": profile.scorer_config_path,
        "sha256": profile.scorer_config_sha256,
    }
    scorer_profile = document["scorer_profile"]
    scorer = document["replay_contract"]["gym_scorer"]
    launcher = scorer["launcher"]
    runtime = scorer["runtime"]

    assert set(scorer_profile) == {
        "call_index_schema",
        "call_schema",
        "closed_schema",
        "disabled_config_path_name",
        "environment",
        "fixture",
        "method",
        "profile_id",
        "requirements",
        "resource_app",
        "resource_config",
        "resource_config_path_name",
        "verifier_type",
    }
    assert scorer_profile["resource_config_path_name"] == "reasoning_gym"
    assert scorer_profile["resource_config"] == full_config
    assert scorer["resources"]["config"] == full_config
    assert launcher["working_directory"] == ("/opt/nemo-rl/3rdparty/Gym-workspace/Gym/resources_servers/reasoning_gym")
    assert launcher["venv_directory"] == ("/opt/gym_venvs/resources_servers/reasoning_gym/.venv")
    assert launcher["config_path_name"] == "resources_only"
    assert launcher["resource_only_config"] == scorer_config
    assert runtime["selected_resource_config"] == scorer_config
    assert runtime["scorer_pin"] == {
        "distribution": "reasoning-gym",
        "required_distribution_version": "0.1.25",
        "module": "reasoning_gym.logic.knights_knaves",
        "module_internal_version_literal": "0.1.19",
        "module_relative_path": "reasoning_gym/logic/knights_knaves.py",
        "module_sha256": ("8837a3c6dfc72bb40db168b82ad6b3da45a08a4000a006fc306368b77b622705"),
        "score_function": "KnightsKnavesDataset.score_answer",
    }
    assert document["runtime_attestation_requirements"]["resource_scorer_child"] == runtime


@pytest.mark.parametrize(
    "mutation",
    (
        "full_swap",
        "full_missing",
        "full_name_mutation",
        "full_path_mutation",
        "full_sha_mutation",
        "effective_swap",
        "effective_missing",
        "effective_name_mutation",
        "effective_path_mutation",
        "effective_sha_mutation",
        "effective_runtime_swap",
        "effective_runtime_missing",
        "effective_runtime_path_mutation",
        "effective_runtime_mutation",
    ),
)
def test_reasoning_gym_manifest_rejects_independent_config_authority_poison(
    tmp_path: Path,
    isolated_authority,
    mutation: str,
) -> None:
    profile, source, _ = isolated_authority(
        tmp_path,
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
    )
    document = build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id="replay-1",
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    scorer = document["replay_contract"]["gym_scorer"]
    full_config = {
        "path": profile.resource_config_path,
        "sha256": profile.resource_config_sha256,
    }
    scorer_config = {
        "path": profile.scorer_config_path,
        "sha256": profile.scorer_config_sha256,
    }
    if mutation == "full_swap":
        document["scorer_profile"]["resource_config"] = scorer_config
    elif mutation == "full_missing":
        scorer["resources"].pop("config")
    elif mutation == "full_name_mutation":
        document["scorer_profile"]["resource_config_path_name"] = "resources_only"
    elif mutation == "full_path_mutation":
        scorer["resources"]["config"]["path"] = profile.scorer_config_path
    elif mutation == "full_sha_mutation":
        scorer["resources"]["config"]["sha256"] = "1" * 64
    elif mutation == "effective_swap":
        scorer["launcher"]["resource_only_config"] = full_config
    elif mutation == "effective_missing":
        scorer["launcher"]["resource_only_config"] = None
    elif mutation == "effective_name_mutation":
        scorer["launcher"]["config_path_name"] = "reasoning_gym"
    elif mutation == "effective_path_mutation":
        scorer["launcher"]["resource_only_config"]["path"] = (
            profile.resource_config_path
        )
    elif mutation == "effective_sha_mutation":
        scorer["launcher"]["resource_only_config"]["sha256"] = "2" * 64
    elif mutation == "effective_runtime_swap":
        scorer["runtime"]["selected_resource_config"] = full_config
    elif mutation == "effective_runtime_missing":
        scorer["runtime"].pop("selected_resource_config")
    elif mutation == "effective_runtime_path_mutation":
        scorer["runtime"]["selected_resource_config"]["path"] = (
            profile.resource_config_path
        )
    else:
        scorer["runtime"]["selected_resource_config"]["sha256"] = "3" * 64

    with pytest.raises(
        ValueError,
        match="scorer_profile|launcher|resource pins|runtime requirements",
    ):
        validate_replay_execution_manifest_v2(
            document,
            authenticated_source=source,
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
        )


def test_reasoning_gym_builder_rejects_resources_only_as_pair_full_config(
    tmp_path: Path,
    isolated_authority,
) -> None:
    profile, source, _ = isolated_authority(
        tmp_path,
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
    )
    source.pair_manifest["selection"]["gym_resources"]["config"] = {
        "path": profile.scorer_config_path,
        "sha256": profile.scorer_config_sha256,
    }
    pair_sha256 = hashlib.sha256(canonical_ascii_json(source.pair_manifest) + b"\n").hexdigest()
    source = replace(source, pair_manifest_sha256=pair_sha256)

    with pytest.raises(ValueError, match="resource pins"):
        build_replay_execution_manifest_v2(
            authenticated_source=source,
            attempt_id="replay-1",
            expected_environment=profile.environment,
            expected_profile_id=profile.profile_id,
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
