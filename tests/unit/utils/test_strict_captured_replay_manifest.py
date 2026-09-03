from __future__ import annotations

import hashlib
import json
import os
import stat
import types
import uuid
from pathlib import Path

import pytest

import nemo_rl.utils.strict_captured_replay_manifest as replay_manifest_module
from nemo_rl.utils.strict_captured_replay_evidence import (
    build_transcript_bundle,
    build_verifier_request_derivation,
    canonical_ascii_json,
    publish_evidence_document,
)
from nemo_rl.utils.strict_captured_replay_manifest import (
    HASH_DOMAIN,
    PAIR_PRE_RECEIPT_KEYS,
    REPLAY_EXECUTION_MANIFEST_SCHEMA,
    ROOT_KEYS,
    SLURM_EXPORT_ALLOWED_NAMES,
    AuthenticatedOffSourceCapture,
    build_replay_execution_manifest,
    build_replay_submission_contract,
    load_authenticated_off_source_capture,
    load_authenticated_replay_static_inputs,
    load_replay_execution_manifest,
    publish_replay_execution_manifest,
    publish_replay_submission_contract,
    validate_replay_execution_manifest,
)
from nemo_rl.utils.strict_main_step_ledger import build_main_step1_ledger
from nemo_rl.utils.strict_model_transport import (
    build_model_transport_bundle,
    build_model_transport_call,
    build_model_transport_manifest,
    build_model_transport_policy,
)

_REAL_STABLE_CONTAINER_ASSET_IDENTITY = replay_manifest_module._stable_container_asset_identity


@pytest.fixture(autouse=True)
def _restore_test_tree_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return sealed test snapshots to removable modes after every case."""
    calls: list[str] = []

    def trusted_test_container(pair: dict[str, object]) -> dict[str, object]:
        reference = pair["artifacts"]["container"]
        calls.append(reference["path"])
        return {
            **reference,
            "owner_uid": 153493,
            "owner_gid": 30,
        }

    monkeypatch.setattr(
        replay_manifest_module,
        "_stable_container_asset_identity",
        trusted_test_container,
    )
    yield calls
    for root, directories, files in os.walk(tmp_path, topdown=False):
        for name in files:
            path = Path(root, name)
            try:
                if not path.is_symlink():
                    path.chmod(0o600)
            except FileNotFoundError:
                pass
        for name in directories:
            path = Path(root, name)
            try:
                if not path.is_symlink():
                    path.chmod(0o700)
            except FileNotFoundError:
                pass
        try:
            Path(root).chmod(0o700)
        except FileNotFoundError:
            pass


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _file(path: str, label: str) -> dict[str, str]:
    return {"path": path, "sha256": _digest(label)}


def _artifact(path: str, schema: str, label: str) -> dict[str, str]:
    return {"path": path, "schema": schema, "sha256": _digest(label)}


def _pair(root: str) -> dict[str, object]:
    return {
        "schema": "nemo-rl-strict-single-env-pair-v2",
        "pair_id": "pair-abc",
        "arms": {"off": "observe", "on": "train"},
        "campaign": {"generation_seed_base": 42},
        "determinism_receipt_dir": "shared_prefix_determinism_receipts",
        "selection": {
            "environment": "reasoning_gym",
            "config": _file("examples/selected.yaml", "config"),
            "gym_resources": {
                "config": _file("resources/reasoning/config.yaml", "gym-config"),
                "requirements": _file("resources/reasoning/requirements.txt", "gym-requirements"),
                "verifier_source": _file("resources/reasoning/server.py", "verifier"),
            },
        },
        "paths": {
            "results_root": f"{root}/results",
            "cache_root": f"{root}/cache",
            "hf_home": f"{root}/hf",
            "snapshot_parent": f"{root}/snapshots",
        },
        "acceptance": {"policy_sha256": _digest("acceptance")},
        "model_transport": {"policy_sha256": _digest("transport-policy")},
        "pair_campaign_sha256": _digest("campaign"),
        "pair_campaign_reward_and_advantage_sha256": _digest("reward"),
        "artifacts": {
            "container": _file(f"{root}/containers/train.sqsh", "container"),
            "fixture": {
                "path": f"{root}/fixture.jsonl",
                "rows": 5,
                "sha256": _digest("fixture"),
            },
            "model": {
                "path": f"{root}/model",
                "tree_sha256_v1": _digest("model-tree"),
            },
            "sandbox_container": _file(f"{root}/containers/sandbox.sqsh", "sandbox"),
        },
        "execution_environment": {
            "schema": "nemo-rl-strict-execution-environment-v1",
            "arm_launcher": {
                "ambient_merge": False,
                "argv_prefix": ["-i"],
                "forbidden_caller_names": ["BASE_LOG_DIR"],
            },
            "fixed": {
                "cpus_per_worker": "144",
                "nemo_skills_sandbox_port": "6000",
                "ray_log_sync_frequency": "60",
                "sandbox_command": "/start-with-nginx.sh",
                "train_path": f"{root}/fixture.jsonl",
                "val_path": f"{root}/fixture.jsonl",
            },
            "arms": {
                "on": {
                    "setup_command": {
                        "byte_count": 12,
                        "sha256": _digest("setup"),
                    }
                }
            },
        },
        "wandb": {
            "entity": "nvidia",
            "project": "nano35-rlvr-convergence",
            "group": {
                "template": "{environment}-{pair_id}",
                "value": "reasoning_gym-pair-abc",
            },
            "resume": "never",
            "arms": {},
            "run_id_derivation": "pinned",
        },
        "scheduler_submission": {
            "nonce": _digest("pair-submission-nonce"),
            "identity": {"submitter_euid": 1000},
        },
        "slurm_export_boundary": {"allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES)},
        "runtime_attestation": {"schema": "pair-runtime"},
        "deployment": {
            "bridge_runnable_manifest_sha256": _digest("bridge"),
            "mcore_runnable_manifest_sha256": _digest("mcore"),
            "nemo_runnable_manifest_sha256": _digest("nemo"),
            "ready": _digest("ready"),
            "ready_file_sha256": _digest("ready-file"),
            "root": f"{root}/deployment",
        },
        "runtime_tools": {
            "bootstrap_sha256sum": _file("/usr/bin/sha256sum", "bootstrap"),
            "document": {
                "schema": "nemo-rl-strict-runtime-tools-v2",
                "host": {},
                "container": {},
            },
            "manifest": _file(f"{root}/runtime-tools.json", "runtime-tools"),
        },
        "container_entry_boundary": {
            "bash_args": ["-p"],
            "bash_path": "/bin/bash",
            "env_path": "/usr/bin/env",
            "sha256sum": _file("/usr/bin/sha256sum", "entry-sha"),
            "unset_environment": ["BASH_ENV", "ENV"],
        },
        "source": {
            "arm_wrapper_sha256": _digest("arm-wrapper"),
            "bridge": {"head": "1" * 40, "root": f"{root}/bridge", "tree": "2" * 40},
            "config_sha256": _digest("source-config"),
            "contract_sha256": _digest("contract"),
            "entrypoint_sha256": _digest("main-entrypoint"),
            "gym": {
                "gitlink_commit": "3" * 40,
                "path": f"{root}/gym",
                "tree": "4" * 40,
            },
            "head": "5" * 40,
            "job_wrapper": _file(f"{root}/strict_pair_job_wrapper.sh", "wrapper"),
            "launcher_sha256": _digest("launcher"),
            "mcore": {"head": "6" * 40, "root": f"{root}/mcore", "tree": "7" * 40},
            "parent_wrapper_sha256": _digest("parent-wrapper"),
            "root": f"{root}/nemo",
            "snapshots": {
                "off": {
                    "config_sha256": _digest("off-config"),
                    "entrypoint_sha256": _digest("off-entrypoint"),
                    "manifest_sha256": _digest("off-snapshot"),
                    "path": f"{root}/snapshots/off",
                },
                "on": {
                    "config_sha256": _digest("on-config"),
                    "entrypoint_sha256": _digest("on-entrypoint"),
                    "manifest_sha256": _digest("on-snapshot"),
                    "path": f"{root}/snapshots/on",
                },
            },
            "tree": "8" * 40,
        },
    }


def _source_capture(root: str) -> dict[str, object]:
    return {
        "arm": "off",
        "restart_count": 0,
        "authenticated_job": {
            "comment": "replay-source",
            "job_id": "6787903",
            "job_name": "off-job",
            "user_id": "1234",
        },
        "job_receipts": {
            "pre": _artifact(
                f"{root}/results/off/strict_pair_job_state/6787903-0/receipts/PRE.json",
                "nemo-rl-strict-pair-job-receipt-v2",
                "off-pre",
            ),
            "exit": _artifact(
                f"{root}/results/off/strict_pair_job_state/6787903-0/receipts/EXIT.json",
                "nemo-rl-strict-pair-job-receipt-v2",
                "off-exit",
            ),
        },
        "step1_evidence": {
            "schema": "nemo-rl-strict-step1-evidence-index-v4",
            "main_ledger": _artifact(
                f"{root}/results/off/strict_pair_step1_evidence/main-ledger.json",
                "nemo-rl-strict-main-step1-ledger-v5",
                "off-ledger",
            ),
            "transcript_bundle": _artifact(
                f"{root}/results/off/strict_pair_step1_evidence/transcript-bundle.json",
                "nemo-rl-strict-step1-transcript-bundle-v4",
                "off-transcript",
            ),
            "model_transport": {
                "schema": "nemo-rl-strict-model-transport-evidence-index-v1",
                "bundle": _artifact(
                    f"{root}/results/off/strict_model_transport/model-transport-bundle.json",
                    "nemo-rl-strict-model-transport-bundle-v1",
                    "off-bundle",
                ),
                "manifest": _artifact(
                    f"{root}/results/off/strict_model_transport/model-transport-manifest.json",
                    "nemo-rl-strict-model-transport-manifest-v1",
                    "off-manifest",
                ),
                "raw_log": {
                    "path": f"{root}/results/off/strict_model_transport/model-transport.jsonl",
                    "record_schema": "nemo-rl-strict-model-transport-call-v1",
                    "record_count": 4,
                    "sha256": _digest("off-log"),
                },
                "ordered_entries_sha256": _digest("off-ordered"),
            },
        },
    }


def _program() -> dict[str, dict[str, str]]:
    return {
        "entrypoint": _file("examples/nemo_gym/run_strict_captured_replay.py", "replay-entrypoint"),
        "evidence_utility": _file("nemo_rl/utils/strict_captured_replay_evidence.py", "evidence-utility"),
        "gym_child_bootstrap": _file(
            "nemo_rl/environments/_strict_gym_child_bootstrap/sitecustomize.py",
            "gym-child-bootstrap",
        ),
        "gym_child_runtime": _file(
            "nemo_rl/environments/strict_gym_child_runtime.py",
            "gym-child-runtime",
        ),
        "job_wrapper": _file("strict_pair_replay_job_wrapper.sh", "replay-job-wrapper"),
        "manifest_utility": _file("nemo_rl/utils/strict_captured_replay_manifest.py", "manifest-utility"),
        "raw_transport_owner": _file("nemo_rl/utils/strict_model_transport_replay.py", "transport-replay"),
        "result_sealer": _file("nemo_rl/utils/strict_captured_replay_seal.py", "result-sealer"),
        "runtime": _file("nemo_rl/algorithms/strict_captured_replay_runtime.py", "replay-runtime"),
        "submission_launcher": _file("strict_pair_replay_launch.sh", "replay-submission-launcher"),
    }


def _manifest(
    root: str,
) -> tuple[dict[str, object], dict[str, object], AuthenticatedOffSourceCapture]:
    source_kwargs = _authenticated_source_fixture(Path(root))
    source = load_authenticated_off_source_capture(**source_kwargs)
    pair = source.pair_manifest
    _prepare_replay_static(Path(root), source=source, attempt_id="replay-1")
    manifest = build_replay_execution_manifest(
        authenticated_source=source,
        attempt_id="replay-1",
    )
    return manifest, pair, source


def _prepare_replay_static(root: Path, *, source: AuthenticatedOffSourceCapture, attempt_id: str) -> None:
    pair = source.pair_manifest
    _seal_replay_export(root, pair=pair, attempt_id=attempt_id)
    contract = build_replay_submission_contract(
        authenticated_source=source,
        attempt_id=attempt_id,
        submission_nonce=_digest(f"replay-nonce-{attempt_id}"),
    )
    contract_parent = Path(root) / "results" / "captured_replay" / "replay_submission_state" / "pair-abc"
    contract_parent.mkdir(mode=0o700, parents=True)
    publish_replay_submission_contract(
        authenticated_source=source,
        attempt_id=attempt_id,
        document=contract,
    )
    load_authenticated_replay_static_inputs(authenticated_source=source, attempt_id=attempt_id)


def test_manifest_exactly_scopes_fresh_verifier_only_replay(tmp_path: Path) -> None:
    manifest, pair, source = _manifest(str(tmp_path))
    assert len(SLURM_EXPORT_ALLOWED_NAMES) == 74
    assert tuple(sorted(SLURM_EXPORT_ALLOWED_NAMES)) == SLURM_EXPORT_ALLOWED_NAMES
    assert manifest["slurm_export_boundary"]["allowed_names"] == list(SLURM_EXPORT_ALLOWED_NAMES)
    assert set(manifest) == set(ROOT_KEYS)
    assert manifest["schema"] == REPLAY_EXECUTION_MANIFEST_SCHEMA
    assert manifest["hash_domain"] == HASH_DOMAIN
    assert manifest["mode"] == "fresh_verifier_reward_replay"
    assert manifest["wandb"] == {
        "enabled": False,
        "mode": "disabled",
        "reason": "scorer-only-replay-no-wandb-credentials-or-output",
    }
    assert manifest["replay_contract"]["policy_execution"] == {
        "backward": False,
        "forward": False,
        "optimizer": False,
        "violation": "fail-closed",
    }
    assert manifest["replay_contract"]["source_generation"] == {
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "vllm_gpu_memory_utilization": 0.1,
    }
    assert manifest["replay_contract"]["gym_scorer"]["container"] == {
        **pair["artifacts"]["container"],
        "owner_uid": 153493,
        "owner_gid": 30,
    }
    assert source.transcript_bundle["generation"] == {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }
    assert manifest["runtime_attestation_requirements"]["shared_prefix_determinism"] == {
        "applicable": False,
        "reason": "verifier-only-no-policy-forward-backward-or-optimizer",
        "status": "not_applicable",
    }
    assert manifest["runtime_attestation_requirements"]["derived_request_runtime"] == {
        "required_openai_version": "2.6.1",
        "required_pydantic_version": "2.13.4",
        "algorithm": "pinned-simple-agent-model-dump-v1",
        "forbidden_endpoints": ["/run"],
        "required_attestation": ("replay-driver-importlib-metadata-before-first-verifier-request"),
        "required": True,
    }
    assert manifest["runtime_attestation_requirements"]["resource_scorer_child"]["scorer_pin"] == {
        "distribution": "reasoning-gym",
        "required_distribution_version": "0.1.25",
        "module": "reasoning_gym.logic.knights_knaves",
        "module_internal_version_literal": "0.1.19",
        "module_relative_path": "reasoning_gym/logic/knights_knaves.py",
        "module_sha256": ("8837a3c6dfc72bb40db168b82ad6b3da45a08a4000a006fc306368b77b622705"),
        "score_function": "KnightsKnavesDataset.score_answer",
    }
    scorer_runtime = manifest["runtime_attestation_requirements"]["resource_scorer_child"]
    assert scorer_runtime["required_common_distributions"] == {
        "nemo-gym": "0.5.0rc0",
        "openai": "2.6.1",
        "pydantic": "2.13.4",
        "pydantic-core": "2.46.4",
        "ray": "2.56.1",
    }
    assert scorer_runtime["required_module_versions"] == {
        "nemo_gym": "0.5.1",
        "ray": "2.56.1",
    }
    assert scorer_runtime["required_per_call_success_evidence"] == {
        "call_schema": "nemo-rl-strict-reasoning-score-call-v1",
        "expected_count": 4,
        "required_outcome_kind": "returned",
        "terminal_index_schema": "nemo-rl-strict-reasoning-score-call-index-v1",
    }
    assert manifest["artifacts"]["outputs"]["reasoning_score_call_index"] == {
        "path": (
            f"{tmp_path}/results/captured_replay/replay-1/" "strict_gym_child_runtime/reasoning-score-call-index.json"
        ),
        "schema": "nemo-rl-strict-reasoning-score-call-index-v1",
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
    }
    assert manifest["artifacts"]["outputs"]["evidence_index"] == {
        "path": f"{tmp_path}/results/captured_replay/replay-1/evidence-index.json",
        "schema": "nemo-rl-strict-captured-replay-evidence-index-v3",
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
    }
    assert manifest["artifacts"]["outputs"]["result_inventory"] == {
        "path": (f"{tmp_path}/results/captured_replay/replay-1/" "result-inventory-v1.json"),
        "schema": "nemo-rl-strict-captured-replay-result-inventory-v1",
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
        "self_excluded": True,
        "terminal_directory_mode": "0555",
    }
    assert manifest["replay_contract"]["gym_scorer"]["source_root"] == {
        "snapshot_relative_path": "3rdparty/Gym-workspace/Gym",
        "host_path": (f"{pair['source']['snapshots']['on']['path']}/3rdparty/Gym-workspace/Gym"),
        "container_path": "/opt/nemo-rl/3rdparty/Gym-workspace/Gym",
    }
    export_values = dict(
        load_authenticated_replay_static_inputs(authenticated_source=source, attempt_id="replay-1").slurm_export_values
    )
    assert {name for name, value in export_values.items() if value} == {
        "EXPECTED_GYM_GITLINK_COMMIT",
        "EXPECTED_GYM_TREE",
        "PAIR_ID",
        "RESULTS_DIR",
        "STRICT_PAIR_ENVIRONMENT",
        "STRICT_PREBUILT_SNAPSHOT_DIR",
    }
    validate_replay_execution_manifest(manifest, authenticated_source=source)


def test_generated_manifest_closes_runtime_direct_on_bindings(tmp_path: Path) -> None:
    from nemo_rl.algorithms.strict_captured_replay_runtime import _transcript_bindings
    from nemo_rl.utils.strict_captured_replay_evidence import replay_run_id

    manifest, pair, _ = _manifest(str(tmp_path))
    submission_sha256 = _digest("fresh-replay-submission")
    bindings = _transcript_bindings(
        manifest=manifest,
        submission_receipt_sha256=submission_sha256,
        authenticated_job_id="987654",
    )
    assert bindings["snapshot_manifest_sha256"] == pair["source"]["snapshots"]["on"]["manifest_sha256"]
    assert bindings["run_id"] == replay_run_id(environment="reasoning_gym", pair_id="pair-abc", attempt_id="replay-1")
    assert bindings["submission_receipt_sha256"] == submission_sha256
    assert manifest["wandb"]["enabled"] is False


def test_replay_export_names_equal_pair_shell_boundary() -> None:
    contract = (
        Path(__file__).resolve().parents[3] / "examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh"
    ).read_text(encoding="utf-8")
    body = contract.split("STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES=(\n", 1)[1].split("\n)", 1)[0]
    shell_names = tuple(line.strip() for line in body.splitlines() if line.strip())
    assert shell_names == SLURM_EXPORT_ALLOWED_NAMES


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["replay_contract"]["policy_execution"].__setitem__("forward", True),
            "forbid all policy execution",
        ),
        (
            lambda value: value["replay_contract"]["source_generation"].__setitem__("vllm_gpu_memory_utilization", 0.2),
            "source generation controls differ",
        ),
        (
            lambda value: value["replay_contract"]["gym_scorer"]["container"].__setitem__("owner_uid", True),
            "named publisher identity differs",
        ),
        (
            lambda value: value["replay_contract"]["gym_scorer"]["container"].__setitem__("owner_uid", 153494),
            "named publisher identity differs",
        ),
        (
            lambda value: value["replay_contract"]["gym_scorer"]["container"].__setitem__("owner_gid", 31),
            "named publisher identity differs",
        ),
        (
            lambda value: value["replay_contract"]["gym_scorer"]["container"].__setitem__("owner_gid", False),
            "named publisher identity differs",
        ),
        (
            lambda value: value["wandb"].__setitem__("enabled", True),
            "W&B disabled policy differs",
        ),
        (
            lambda value: value["runtime_attestation_requirements"]["shared_prefix_determinism"].__setitem__(
                "status", "complete"
            ),
            "runtime requirements contract",
        ),
        (
            lambda value: value["replay_contract"]["gym_scorer"]["source_root"].__setitem__(
                "host_path", "/tmp/ambient-gym"
            ),
            "Gym source root differs",
        ),
        (
            lambda value: value["source_capture"]["step1_evidence"]["model_transport"]["raw_log"].__setitem__(
                "record_count", 3
            ),
            "exact K4",
        ),
        (
            lambda value: value.__setitem__("candidate_job_id", "123"),
            "keyset mismatch",
        ),
        (
            lambda value: value["runtime_attestation_requirements"]["derived_request_runtime"].__setitem__(
                "required_openai_version", "2.7.2"
            ),
            "runtime requirements contract",
        ),
    ],
)
def test_manifest_rejects_training_auth_and_future_fact_drift(tmp_path: Path, mutate, message: str) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    mutate(manifest)
    with pytest.raises(ValueError, match=message):
        validate_replay_execution_manifest(manifest, authenticated_source=source)


@pytest.mark.parametrize(
    ("map_name", "poison_kind", "poisoned_value"),
    [
        ("required_common_distributions", "missing", None),
        ("required_common_distributions", "extra", "2.56.1"),
        ("required_common_distributions", "wrong-type", 2.56),
        ("required_common_distributions", "wrong-value", "2.56.0"),
        ("required_module_versions", "missing", None),
        ("required_module_versions", "extra", "2.56.1"),
        ("required_module_versions", "wrong-type", False),
        ("required_module_versions", "wrong-value", "2.56.0"),
    ],
)
def test_manifest_rejects_inexact_ray_r9_runtime_maps(
    tmp_path: Path,
    map_name: str,
    poison_kind: str,
    poisoned_value: object,
) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    versions = manifest["runtime_attestation_requirements"]["resource_scorer_child"][map_name]
    if poison_kind == "missing":
        versions.pop("ray")
    elif poison_kind == "extra":
        versions["ray-extra"] = poisoned_value
    else:
        versions["ray"] = poisoned_value
    with pytest.raises(ValueError, match="replay runtime requirements contract differs"):
        validate_replay_execution_manifest(manifest, authenticated_source=source)


def test_public_manifest_validator_rejects_coherent_static_contract_forgery(
    tmp_path: Path,
) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    scheduler = manifest["scheduler_submission"]
    scheduler["contract"] = {
        "path": (
            f"{tmp_path}/results/captured_replay/replay_submission_state/" "pair-abc/replay-2.submission-contract.json"
        ),
        "sha256": _digest("coherent-forged-contract"),
    }
    scheduler["nonce"] = _digest("coherent-forged-nonce")
    with pytest.raises(ValueError, match="submission contract differs"):
        validate_replay_execution_manifest(manifest, authenticated_source=source)


def test_manifest_has_acyclic_declarations_not_future_replay_facts(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _manifest(str(tmp_path))
    encoded = str(manifest)
    for forbidden in (
        "candidate_job_id",
        "authenticated_replay_job_id",
        "pre_receipt_sha256",
        "replay_exit_receipt_sha256",
        "replay_ledger_sha256",
        "transcript_bundle_sha256",
    ):
        assert forbidden not in encoded
    assert "authenticated_job" in manifest["source_capture"]
    assert manifest["source_capture"]["arm"] == "off"
    source_exit = manifest["source_capture"]["job_receipts"]["exit"]
    assert set(source_exit) == {"path", "schema", "sha256"}
    assert source_exit["schema"] == "nemo-rl-strict-pair-job-receipt-v2"
    assert source_exit["path"].endswith("/receipts/EXIT.json")


def test_manifest_publish_and_load_are_single_link_mode_0400(
    tmp_path: Path, _restore_test_tree_permissions: list[str]
) -> None:
    manifest, pair, source = _manifest(str(tmp_path))
    output_parent = tmp_path / "control"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "REPLAY_EXECUTION_MANIFEST.json"
    _restore_test_tree_permissions.clear()
    path, digest = publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)
    assert _restore_test_tree_permissions == [pair["artifacts"]["container"]["path"]]
    assert path.stat().st_mode & 0o777 == 0o400
    assert path.stat().st_nlink == 1
    loaded, actual = load_replay_execution_manifest(path=path, expected_sha256=digest, authenticated_source=source)
    assert _restore_test_tree_permissions == [
        pair["artifacts"]["container"]["path"],
        pair["artifacts"]["container"]["path"],
    ]
    assert loaded == manifest
    assert actual == digest
    with pytest.raises(FileExistsError):
        publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)


def test_manifest_build_reloads_named_container_owner(
    tmp_path: Path, _restore_test_tree_permissions: list[str]
) -> None:
    source = load_authenticated_off_source_capture(**_authenticated_source_fixture(tmp_path))
    _prepare_replay_static(tmp_path, source=source, attempt_id="replay-1")
    _restore_test_tree_permissions.clear()
    manifest = build_replay_execution_manifest(
        authenticated_source=source,
        attempt_id="replay-1",
    )
    expected_path = source.pair_manifest["artifacts"]["container"]["path"]
    assert _restore_test_tree_permissions == [expected_path]
    assert manifest["replay_contract"]["gym_scorer"]["container"] == {
        **source.pair_manifest["artifacts"]["container"],
        "owner_uid": 153493,
        "owner_gid": 30,
    }


def _container_stat(**changes: object) -> types.SimpleNamespace:
    values = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": stat.S_IFREG | 0o644,
        "st_nlink": 1,
        "st_uid": 153493,
        "st_gid": 30,
        "st_size": 85_700_000_000,
        "st_mtime_ns": 3,
        "st_ctime_ns": 4,
    }
    values.update(changes)
    return types.SimpleNamespace(**values)


def test_stable_container_identity_accepts_only_named_foreign_publisher() -> None:
    pair = _pair("/authority")
    metadata = _container_stat()
    actual = _REAL_STABLE_CONTAINER_ASSET_IDENTITY(
        pair,
        lstat=lambda path: metadata,
        access=lambda path, mode, *, effective_ids: False,
        geteuid=lambda: 14568,
    )
    assert actual == {
        **pair["artifacts"]["container"],
        "owner_uid": 153493,
        "owner_gid": 30,
    }


@pytest.mark.parametrize(
    ("metadata", "effective_uid", "effectively_writable", "message"),
    [
        (_container_stat(st_uid=153494), 14568, False, "publisher identity"),
        (_container_stat(st_gid=31), 14568, False, "publisher identity"),
        (_container_stat(st_nlink=2), 14568, False, "single-link"),
        (_container_stat(st_mode=stat.S_IFREG | 0o664), 14568, False, "writable"),
        (_container_stat(st_mode=stat.S_IFREG | 0o646), 14568, False, "writable"),
        (_container_stat(), 0, False, "distinct from the container publisher"),
        (_container_stat(), 153493, False, "distinct from the container publisher"),
        (_container_stat(), 14568, True, "writable by the replay submitter"),
    ],
)
def test_stable_container_identity_rejects_untrusted_or_writable_asset(
    metadata: types.SimpleNamespace,
    effective_uid: int,
    effectively_writable: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _REAL_STABLE_CONTAINER_ASSET_IDENTITY(
            _pair("/authority"),
            lstat=lambda path: metadata,
            access=lambda path, mode, *, effective_ids: effectively_writable,
            geteuid=lambda: effective_uid,
        )


def test_manifest_publish_reloads_program_export_and_contract_bytes(
    tmp_path: Path,
) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    output_parent = tmp_path / "control"
    output_parent.mkdir(mode=0o700)

    manifest["replay_contract"]["program"]["runtime"]["sha256"] = _digest("forged-runtime")
    with pytest.raises(ValueError, match="program differs from authenticated ON"):
        publish_replay_execution_manifest(
            output=output_parent / "program.json",
            document=manifest,
            authenticated_source=source,
        )

    (tmp_path / "export").mkdir()
    manifest, _, source = _manifest(str(tmp_path / "export"))
    export_path = Path(manifest["slurm_export_boundary"]["path"])
    payload = export_path.read_bytes().replace(b"COMMAND=\0", b"COMMAND=forged\0")
    export_path.chmod(0o600)
    export_path.write_bytes(payload)
    export_path.chmod(0o400)
    with pytest.raises(ValueError, match="COMMAND value differs"):
        publish_replay_execution_manifest(
            output=output_parent / "export.json",
            document=manifest,
            authenticated_source=source,
        )

    (tmp_path / "contract").mkdir()
    manifest, _, source = _manifest(str(tmp_path / "contract"))
    contract_path = Path(manifest["scheduler_submission"]["contract"]["path"])
    contract = load_authenticated_replay_static_inputs(
        authenticated_source=source, attempt_id="replay-1"
    ).submission_contract
    contract["job_wrapper"]["sha256"] = _digest("forged-wrapper")
    contract_path.chmod(0o600)
    contract_path.write_bytes(canonical_ascii_json(contract) + b"\n")
    contract_path.chmod(0o400)
    with pytest.raises(ValueError, match="job_wrapper differs"):
        publish_replay_execution_manifest(
            output=output_parent / "contract.json",
            document=manifest,
            authenticated_source=source,
        )


@pytest.mark.parametrize("phase", ["publish", "load"])
def test_manifest_publish_and_load_rehash_authenticated_sbatch(tmp_path: Path, phase: str) -> None:
    root = tmp_path / phase
    root.mkdir()
    manifest, _, source = _manifest(str(root))
    output_parent = root / "control"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "manifest.json"
    published = None
    if phase == "load":
        published = publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)
    sbatch = Path(source.pair_manifest["runtime_tools"]["document"]["host"]["sbatch"]["path"])
    sbatch.chmod(0o700)
    sbatch.write_bytes(b"forged-sbatch")
    sbatch.chmod(0o500)
    with pytest.raises(ValueError, match="sbatch program bytes differ"):
        if phase == "publish":
            publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)
        else:
            assert published is not None
            load_replay_execution_manifest(
                path=published[0],
                expected_sha256=published[1],
                authenticated_source=source,
            )


@pytest.mark.parametrize("phase", ["publish", "load"])
def test_manifest_publish_and_load_rehash_pair_on_program_bytes(tmp_path: Path, phase: str) -> None:
    root = tmp_path / phase
    root.mkdir()
    manifest, _, source = _manifest(str(root))
    output_parent = root / "control"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "manifest.json"
    published = None
    if phase == "load":
        published = publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)
    runtime = (
        Path(source.pair_manifest["source"]["snapshots"]["on"]["path"])
        / manifest["replay_contract"]["program"]["runtime"]["path"]
    )
    runtime.chmod(0o600)
    runtime.write_bytes(b"# substituted replay runtime\n")
    runtime.chmod(0o400)
    with pytest.raises(ValueError, match="snapshot member .* bytes differ"):
        if phase == "publish":
            publish_replay_execution_manifest(output=output, document=manifest, authenticated_source=source)
        else:
            assert published is not None
            load_replay_execution_manifest(
                path=published[0],
                expected_sha256=published[1],
                authenticated_source=source,
            )


@pytest.mark.parametrize(
    "name",
    [
        "COMMAND",
        "CONTAINER",
        "STRICT_PAIR_JOB_WRAPPER",
        "EXPECTED_STRICT_PAIR_CONTAINER_PYTHON_SHA256",
        "EXPECTED_NEMO_HEAD",
    ],
)
def test_static_loader_rejects_any_noncontract_export_value(tmp_path: Path, name: str) -> None:
    source = load_authenticated_off_source_capture(**_authenticated_source_fixture(tmp_path))
    _seal_replay_export(tmp_path, pair=source.pair_manifest, attempt_id="replay-1")
    export_path = tmp_path / "results" / "captured_replay" / "slurm_exports" / "pair-abc" / "replay-1.env"
    payload = export_path.read_bytes().replace(f"{name}=\0".encode("ascii"), f"{name}=forged\0".encode("ascii"))
    export_path.chmod(0o600)
    export_path.write_bytes(payload)
    export_path.chmod(0o400)
    with pytest.raises(ValueError, match=f"{name} value differs"):
        build_replay_submission_contract(
            authenticated_source=source,
            attempt_id="replay-1",
            submission_nonce=_digest("export-forgery"),
        )


def test_submission_contract_rejects_cross_attempt_publication(tmp_path: Path) -> None:
    source = load_authenticated_off_source_capture(**_authenticated_source_fixture(tmp_path))
    _seal_replay_export(tmp_path, pair=source.pair_manifest, attempt_id="replay-1")
    contract = build_replay_submission_contract(
        authenticated_source=source,
        attempt_id="replay-1",
        submission_nonce=_digest("cross-attempt"),
    )
    contract["attempt_id"] = "replay-2"
    parent = tmp_path / "results" / "captured_replay" / "replay_submission_state" / "pair-abc"
    parent.mkdir(mode=0o700, parents=True)
    with pytest.raises(ValueError, match="attempt differs"):
        publish_replay_submission_contract(
            authenticated_source=source,
            attempt_id="replay-1",
            document=contract,
        )


@pytest.mark.parametrize(
    ("snapshot_forgery", "message"),
    [
        ("escaping_symlink", "symlink escapes root"),
        ("symlink_cycle", "symlink cycle"),
        ("embedded_symlink_dotdot_escape", "symlink escapes root"),
        ("symlink_file_parent", "symlink traverses non-directory"),
        ("empty_directory", "directory inventory differs"),
        ("config_digest", "config digest does not close"),
        ("gym_verifier_digest", "selected Gym verifier_source digest"),
    ],
)
def test_static_loader_rejects_unauthenticated_pair_on_tree(
    tmp_path: Path, snapshot_forgery: str, message: str
) -> None:
    source = load_authenticated_off_source_capture(
        **_authenticated_source_fixture(tmp_path, snapshot_forgery=snapshot_forgery)
    )
    _seal_replay_export(tmp_path, pair=source.pair_manifest, attempt_id="replay-1")
    with pytest.raises((ValueError, RuntimeError), match=message):
        build_replay_submission_contract(
            authenticated_source=source,
            attempt_id="replay-1",
            submission_nonce=_digest("snapshot-forgery"),
        )


def test_static_loader_accepts_in_root_componentwise_symlinks(tmp_path: Path) -> None:
    source = load_authenticated_off_source_capture(
        **_authenticated_source_fixture(tmp_path, snapshot_forgery="legitimate_symlinks")
    )
    _seal_replay_export(tmp_path, pair=source.pair_manifest, attempt_id="replay-1")
    contract = build_replay_submission_contract(
        authenticated_source=source,
        attempt_id="replay-1",
        submission_nonce=_digest("legitimate-symlinks"),
    )
    assert contract["attempt_id"] == "replay-1"


def test_manifest_rejects_double_slash_absolute_paths(tmp_path: Path) -> None:
    source = load_authenticated_off_source_capture(**_authenticated_source_fixture(tmp_path))
    source.pair_manifest["paths"]["results_root"] = f"//{str(tmp_path).lstrip('/')}/results"
    with pytest.raises(ValueError, match="canonical absolute path"):
        build_replay_execution_manifest(
            authenticated_source=source,
            attempt_id="replay-1",
        )


def _seal_document(path: Path, document: dict[str, object], *, lf: bool) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    _, digest = publish_evidence_document(output=path, document=document, trailing_lf=lf)
    return digest


def _rewrite_sealed_document(path: Path, document: dict[str, object], *, lf: bool) -> str:
    payload = canonical_ascii_json(document) + (b"\n" if lf else b"")
    path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(0o400)
    return hashlib.sha256(payload).hexdigest()


def _seal_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o400)
    return hashlib.sha256(payload).hexdigest()


def _seal_replay_export(root: Path, *, pair: dict[str, object], attempt_id: str) -> str:
    values = {name: b"" for name in SLURM_EXPORT_ALLOWED_NAMES}
    values.update(
        {
            "EXPECTED_GYM_GITLINK_COMMIT": pair["source"]["gym"]["gitlink_commit"].encode("ascii"),
            "EXPECTED_GYM_TREE": pair["source"]["gym"]["tree"].encode("ascii"),
            "PAIR_ID": pair["pair_id"].encode("ascii"),
            "RESULTS_DIR": (f"{pair['paths']['results_root']}/captured_replay/{attempt_id}").encode("ascii"),
            "STRICT_PAIR_ENVIRONMENT": pair["selection"]["environment"].encode("ascii"),
            "STRICT_PREBUILT_SNAPSHOT_DIR": pair["source"]["snapshots"]["on"]["path"].encode("ascii"),
        }
    )
    payload = b"".join(name.encode("ascii") + b"=" + values[name] + b"\0" for name in SLURM_EXPORT_ALLOWED_NAMES)
    path = root / "results" / "captured_replay" / "slurm_exports" / pair["pair_id"] / f"{attempt_id}.env"
    return _seal_bytes(path, payload)


def _seal_pair_on_snapshot(
    root: Path,
    pair: dict[str, object],
    *,
    snapshot_forgery: str | None = None,
) -> None:
    snapshot = root / "snapshots" / "on-pair-abc"
    snapshot.mkdir(mode=0o700, parents=True)
    program_payloads = {
        name: f"# authenticated test replay member: {name}\n".encode("ascii")
        for name in (
            "entrypoint",
            "evidence_utility",
            "gym_child_bootstrap",
            "gym_child_runtime",
            "job_wrapper",
            "manifest_utility",
            "raw_transport_owner",
            "result_sealer",
            "runtime",
            "submission_launcher",
        )
    }
    program_paths = {
        "entrypoint": "examples/nemo_gym/run_strict_captured_replay.py",
        "evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence.py",
        "gym_child_bootstrap": ("nemo_rl/environments/_strict_gym_child_bootstrap/sitecustomize.py"),
        "gym_child_runtime": "nemo_rl/environments/strict_gym_child_runtime.py",
        "job_wrapper": "strict_pair_replay_job_wrapper.sh",
        "manifest_utility": "nemo_rl/utils/strict_captured_replay_manifest.py",
        "raw_transport_owner": "nemo_rl/utils/strict_model_transport_replay.py",
        "result_sealer": "nemo_rl/utils/strict_captured_replay_seal.py",
        "runtime": "nemo_rl/algorithms/strict_captured_replay_runtime.py",
        "submission_launcher": "strict_pair_replay_launch.sh",
    }
    authenticated_payloads = {
        pair["selection"]["config"]["path"]: b"config",
        "examples/run_grpo_single_controller.py": b"main-entrypoint",
        ("3rdparty/Gym-workspace/Gym/" + pair["selection"]["gym_resources"]["config"]["path"]): b"gym-config",
        (
            "3rdparty/Gym-workspace/Gym/" + pair["selection"]["gym_resources"]["requirements"]["path"]
        ): b"gym-requirements",
        ("3rdparty/Gym-workspace/Gym/" + pair["selection"]["gym_resources"]["verifier_source"]["path"]): b"verifier",
        ("3rdparty/Gym-workspace/Gym/resources_servers/reasoning_gym/" "configs/resources_only.yaml"): (
            b"reasoning_gym:\n"
            b"  resources_servers:\n"
            b"    reasoning_gym:\n"
            b"      entrypoint: app.py\n"
            b"      domain: knowledge\n"
            b"      verified: false\n"
        ),
    }
    if snapshot_forgery in {
        "embedded_symlink_dotdot_escape",
        "legitimate_symlinks",
        "symlink_file_parent",
    }:
        authenticated_payloads["safe/member"] = b"authenticated-safe-member"
    executable_paths = {
        program_paths["job_wrapper"],
        program_paths["submission_launcher"],
    }
    for name, relative in program_paths.items():
        path = snapshot / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(program_payloads[name])
    for relative, payload in authenticated_payloads.items():
        path = snapshot / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
    symlinks: dict[str, str] = {}
    if snapshot_forgery == "escaping_symlink":
        (snapshot / "escape").symlink_to("../../outside")
        symlinks["escape"] = "../../outside"
    elif snapshot_forgery == "symlink_cycle":
        (snapshot / "cycle-a").symlink_to("cycle-b")
        (snapshot / "cycle-b").symlink_to("cycle-a")
        symlinks.update({"cycle-a": "cycle-b", "cycle-b": "cycle-a"})
    elif snapshot_forgery == "embedded_symlink_dotdot_escape":
        (snapshot / "d").mkdir(mode=0o700)
        (snapshot / "d" / "b").symlink_to("../safe")
        (snapshot / "a").symlink_to("d/b/../..")
        symlinks.update({"a": "d/b/../..", "d/b": "../safe"})
    elif snapshot_forgery == "legitimate_symlinks":
        (snapshot / "nested").mkdir(mode=0o700)
        (snapshot / "alias").symlink_to("safe")
        (snapshot / "repeat").symlink_to("alias/../alias")
        (snapshot / "nested" / "up").symlink_to("../safe/member")
        symlinks.update(
            {
                "alias": "safe",
                "nested/up": "../safe/member",
                "repeat": "alias/../alias",
            }
        )
    elif snapshot_forgery == "symlink_file_parent":
        (snapshot / "file-parent").symlink_to("safe/member/..")
        symlinks["file-parent"] = "safe/member/.."
    elif snapshot_forgery == "empty_directory":
        (snapshot / "unmanifested-empty-directory").mkdir(mode=0o700)
    elif snapshot_forgery == "config_digest":
        (snapshot / pair["selection"]["config"]["path"]).write_bytes(b"forged")
    elif snapshot_forgery == "gym_verifier_digest":
        gym_verifier = (
            snapshot
            / "3rdparty"
            / "Gym-workspace"
            / "Gym"
            / pair["selection"]["gym_resources"]["verifier_source"]["path"]
        )
        gym_verifier.write_bytes(b"forged")
    symlink_path = snapshot / "strict-pair-snapshot-symlinks.json"
    symlink_path.write_bytes(
        canonical_ascii_json({"schema": "nemo-rl-strict-snapshot-symlinks-v1", "symlinks": symlinks}) + b"\n"
    )
    mode_path = snapshot / "strict-pair-snapshot-modes.json"
    mode_values = {
        **{relative: relative in executable_paths for relative in program_paths.values()},
        **{relative: False for relative in authenticated_payloads},
        "strict-pair-snapshot-symlinks.json": False,
        "strict-pair-snapshot-modes.json": False,
    }
    mode_path.write_bytes(
        canonical_ascii_json(
            {
                "regular_file_executable": mode_values,
                "schema": "nemo-rl-strict-snapshot-modes-v1",
            }
        )
        + b"\n"
    )
    regular_paths = [
        *program_paths.values(),
        *authenticated_payloads,
        symlink_path.name,
        mode_path.name,
    ]
    manifest_path = snapshot / "strict-pair-snapshot-manifest.sha256"
    manifest_path.write_bytes(
        b"".join(
            hashlib.sha256((snapshot / relative).read_bytes()).hexdigest().encode("ascii")
            + b"  "
            + relative.encode("ascii")
            + b"\n"
            for relative in regular_paths
        )
    )
    for relative in regular_paths:
        (snapshot / relative).chmod(0o500 if relative in executable_paths else 0o400)
    manifest_path.chmod(0o400)
    directories = sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o500)
    snapshot.chmod(0o500)
    pair["source"]["snapshots"]["on"] = {
        "config_sha256": pair["selection"]["config"]["sha256"],
        "entrypoint_sha256": pair["source"]["entrypoint_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "path": str(snapshot),
    }


def _install_off_transport_policy_sources(root: Path, pair: dict[str, object]) -> dict[str, object]:
    snapshot = root / "snapshots" / "off"
    payloads = {
        "nemo_rl/utils/strict_model_transport.py": b"fixture-transport-collector",
        "nemo_rl/experience/rollout_manager.py": b"fixture-rollout-finalizer",
        "nemo_rl/models/generation/vllm/vllm_worker_async.py": (b"fixture-vllm-route"),
    }
    digests: dict[str, str] = {}
    source_names = {
        "nemo_rl/utils/strict_model_transport.py": "collector",
        "nemo_rl/experience/rollout_manager.py": "rollout_finalizer",
        "nemo_rl/models/generation/vllm/vllm_worker_async.py": "vllm_route",
    }
    for relative, payload in payloads.items():
        path = snapshot / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o400)
        digests[source_names[relative]] = hashlib.sha256(payload).hexdigest()
    for directory in sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    snapshot.chmod(0o500)
    pair["source"]["snapshots"]["off"]["path"] = str(snapshot)
    return build_model_transport_policy(
        collector_sha256=digests["collector"],
        rollout_finalizer_sha256=digests["rollout_finalizer"],
        vllm_route_sha256=digests["vllm_route"],
    )


def _fixture_shared_prefix_group_id(task_index: int) -> str:
    high = 0x305FE72A79E34B88
    low = ((high & ((1 << 63) - 1)) ^ task_index) | (1 << 63)
    return str(uuid.UUID(int=(high << 64) | low))


def _build_fixture_source_ledger(
    *,
    pair: dict[str, object],
    transcript: dict[str, object],
    transcript_ref: dict[str, str],
) -> dict[str, object]:
    task_index = transcript["entries"][0]["agent_run_request"]["_ng_task_index"]
    group_id = _fixture_shared_prefix_group_id(task_index)
    advantages = (
        -1.1546986103057861,
        1.1546984910964966,
        -1.1546986103057861,
        1.1546984910964966,
    )
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(transcript["entries"]):
        response_item = entry["model_response"]["output"][0]
        prompt = response_item["prompt_token_ids"]
        completion = response_item["generation_token_ids"]
        token_ids = [*prompt, *completion]
        reward = entry["raw_environment_reward"]
        rows.append(
            {
                "sample_index": index,
                "sample_id": f"{group_id}_g{index}",
                "shared_prefix_group_id": group_id,
                "fixture_row_index": 0,
                "rollout_index": index,
                "generation_seed": entry["generation_seed"],
                "request_sha256": entry["generation_request_sha256"],
                "response_sha256": entry["model_response_sha256"],
                "agent_run_request_sha256": entry["agent_run_request_sha256"],
                "derived_verifier_request_sha256": entry["derived_verifier_request_sha256"],
                "verifier_response_sha256": entry["verifier_response_sha256"],
                "token_ids": token_ids,
                "input_length": len(token_ids),
                "prompt_token_ids": prompt,
                "completion_token_ids": completion,
                "token_loss_mask": [0.0] * len(prompt) + [1.0] * len(completion),
                "raw_environment_reward": reward,
                "pre_penalty_environment_reward": reward,
                "penalty_flags": {
                    "reasoning_equal_to_final_answer": False,
                    "empty_final_answer": False,
                    "unwanted_token": False,
                    "malformed_think_tag": False,
                    "invalid_tool_call": False,
                    "malformed_thinking": False,
                    "raw_invalid_tool_call": False,
                    "raw_malformed_thinking": False,
                    "invalid_and_malformed": False,
                },
                "verifier_reward": reward,
                "processed_reward": reward,
                "sample_mask": 1.0,
                "advantages": [advantages[index]] * len(token_ids),
                "valid_loss_tokens": len(completion),
                "total_tokens": len(token_ids),
            }
        )
    return build_main_step1_ledger(
        pair_id=pair["pair_id"],
        environment=pair["selection"]["environment"],
        arm="off",
        mode="observe",
        generation=transcript["generation"],
        bindings={
            **transcript["bindings"],
            "restart_count": 0,
            "pair_campaign_sha256": pair["pair_campaign_sha256"],
            "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        },
        transcript_bundle=transcript_ref,
        row_inputs=rows,
        update_successful=True,
    )


def _authenticated_source_fixture(
    tmp_path: Path,
    *,
    forgery: str | None = None,
    snapshot_forgery: str | None = None,
) -> dict[str, object]:
    root = str(tmp_path)
    pair = _pair(root)
    containers = tmp_path / "containers"
    containers.mkdir(mode=0o700)
    for name, payload in (
        ("train.sqsh", b"container"),
        ("sandbox.sqsh", b"sandbox"),
    ):
        container = containers / name
        container.write_bytes(payload)
        container.chmod(0o400)
    pair["model_transport"] = _install_off_transport_policy_sources(tmp_path, pair)
    pair["source"]["config_sha256"] = pair["selection"]["config"]["sha256"]
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    pair["campaign"] = {
        "generation_seed_base": 42,
        "generation": {
            "max_new_tokens": 768,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "vllm_gpu_memory_utilization": 0.1,
        },
        "nodes": 1,
        "slurm": {
            "account": "nemotron_sw_post",
            "partition": "batch",
            "qos": "normal",
        },
        "reward_and_advantage": {"strict": True},
    }
    pair["execution_environment"]["arms"] = {
        arm: {
            "scheduler": {"batch_working_directory": f"{root}/snapshots/{arm}-pair-abc"},
            "setup_command": {"byte_count": 12, "sha256": _digest("setup")},
        }
        for arm in ("off", "on")
    }
    pair["wandb"]["arms"] = {
        arm: {
            "name": f"{arm}-reasoning_gym-pair-abc",
            "name_template": f"{arm}-{{environment}}-{{pair_id}}",
            "run_id": _digest(f"wandb-{arm}"),
        }
        for arm in ("off", "on")
    }
    trusted_tools = tmp_path / "trusted-tools"
    trusted_tools.mkdir(mode=0o700)
    host_tools = {}
    for name in ("env", "python", "sbatch", "scancel", "scontrol", "nvidia-smi"):
        tool_path = trusted_tools / name
        tool_path.write_bytes(f"tool-{name}".encode("ascii"))
        tool_path.chmod(0o500)
        manifest_name = "nvidia_smi" if name == "nvidia-smi" else name
        host_tools[manifest_name] = _file(str(tool_path), f"tool-{name}")
    trusted_tools.chmod(0o500)
    pair["runtime_tools"]["document"] = {
        "schema": "nemo-rl-strict-runtime-tools-v2",
        "host": host_tools,
        "container": {
            "python": _file("/usr/bin/python3", "container-python"),
            "uv": _file("/usr/bin/uv", "container-uv"),
            "uv_shim": _file(f"{root}/strict_pair_uv.sh", "uv-shim"),
        },
    }
    accepted_paths = {
        arm: results / "strict_pair_submission_state" / "pair-abc" / f"{arm}.job-id" for arm in ("off", "on")
    }
    job_ids = {"off": "6787903", "on": "6787904"}
    for arm in ("off", "on"):
        _seal_bytes(accepted_paths[arm], f"{job_ids[arm]}\n".encode("ascii"))
    pair["scheduler_submission"] = {
        "nonce": _digest("pair-submission-nonce"),
        "identity": {"submitter_euid": os.geteuid()},
        "contract": _file(f"{root}/results/STRICT_PAIR_SUBMISSION_CONTRACT.json", "pair-contract"),
        "accepted_id_records": {arm: {"path": str(accepted_paths[arm])} for arm in ("off", "on")},
    }
    pair["slurm_export_boundary"] = {
        "allowed_names": list(SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "arms": {
            arm: _file(
                f"{root}/results/strict_pair_slurm_exports/pair-abc/{arm}.env",
                f"export-{arm}",
            )
            for arm in ("off", "on")
        },
        "format": "nul-separated-name-value",
        "get_user_env": False,
        "job_argv": [
            "--pair-manifest",
            "{pair_manifest_path}",
            "--pair-manifest-sha256",
            "{pair_manifest_sha256}",
            "--arm",
            "{arm}",
        ],
        "schema": "nemo-rl-strict-slurm-export-file-v2",
    }
    pair["runtime_attestation"] = {
        "expected_count_per_fresh_process_group": 4,
        "lines": {
            "off": {"sha256_ascii_no_newline": _digest("attestation-off")},
            "on": {"sha256_ascii_no_newline": _digest("attestation-on")},
        },
        "schema": "nemo-rl-shared-prefix-determinism-attestation-v1",
    }
    _seal_pair_on_snapshot(tmp_path, pair, snapshot_forgery=snapshot_forgery)
    pair_manifest_path = results / "PAIR_MANIFEST.json"
    pair_sha = _seal_document(pair_manifest_path, pair, lf=True)

    nonce = pair["scheduler_submission"]["nonce"]
    identities = {
        arm: {
            "comment": f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair_sha}",
            "job_id": job_ids[arm],
            "job_name": f"{arm}-reasoning_gym-pair-abc",
            "user_id": str(os.geteuid()),
        }
        for arm in ("off", "on")
    }
    if forgery == "off_comment":
        identities["off"]["comment"] = "nemo-rl-strict-pair-v1:off:forged"
    scheduler_tools = {
        "client_environment": {
            "ambient_merge": False,
            "env": host_tools["env"],
            "variables": {
                "LC_ALL": "C",
                "SLURM_CONF": _file(f"{root}/slurm.conf", "slurm-conf"),
            },
        },
        "sbatch": host_tools["sbatch"],
        "scancel": host_tools["scancel"],
        "scontrol": host_tools["scontrol"],
    }
    candidate_ids = {
        "off": [job_ids["off"]],
        "on": [job_ids["on"]],
        "unattributed": [],
    }

    def query(phase: str) -> dict[str, object]:
        records = {}
        for arm in ("off", "on"):
            held = phase == "pre"
            records[arm] = [
                {
                    **identities[arm],
                    "held": held,
                    "job_state": "PENDING" if held else "RUNNING",
                    "reason": "JobHeldUser" if held else "None",
                    "work_dir": pair["execution_environment"]["arms"][arm]["scheduler"]["batch_working_directory"],
                }
            ]
        ordered = f"{job_ids['off']},{job_ids['on']}"
        return {
            "argv": [
                host_tools["scontrol"]["path"],
                "show",
                "job",
                "--json",
                ordered,
            ],
            "authenticated_job_ids": [job_ids["off"], job_ids["on"]],
            "byte_count": 200,
            "candidate_job_ids": candidate_ids,
            "complete": True,
            "line_count": 2,
            "normalization_status": 0,
            "output_sha256_raw": _digest(f"query-{phase}"),
            "phase": phase,
            "records": records,
            "securely_unlinked": True,
            "status": 0,
            "unterminated_final_line": False,
            "unresolved_job_ids": [],
        }

    held_submissions = {}
    for arm in ("off", "on"):
        accepted_payload = f"{job_ids[arm]}\n".encode("ascii")
        held_submissions[arm] = {
            "accepted_id_record": {
                "parsed_job_id": job_ids[arm],
                "path": str(accepted_paths[arm]),
                "sha256": hashlib.sha256(accepted_payload).hexdigest(),
            },
            "candidate_job_id": job_ids[arm],
            "candidate_job_id_sha256_ascii_no_newline": hashlib.sha256(job_ids[arm].encode("ascii")).hexdigest(),
            "candidate_job_id_source": "accepted-id-record",
            "submission_rpc": {
                "drained_unix_ns": 2,
                "relay_status": 0,
                "sbatch_status": 0,
                "started_unix_ns": 1,
                "writer_drained": True,
            },
            "wrapper_status": 0,
        }
    submission_path = results / "PAIR_SUBMISSION_RECEIPT.json"
    submission = {
        "acceptance": pair["acceptance"],
        "authenticated_jobs": {arm: [identities[arm]] for arm in ("off", "on")},
        "cancellations": [],
        "execution_environment": pair["execution_environment"],
        "held_submissions": held_submissions,
        "model_transport": pair["model_transport"],
        "outcome": "released",
        "pair": {
            "id": "pair-abc",
            "manifest": {"path": str(pair_manifest_path), "sha256": pair_sha},
        },
        "post_cancel_queries": [],
        "post_release_query": query("post"),
        "pre_cancel_queries": [],
        "pre_release_query": query("pre"),
        "receipt": {
            "path": str(submission_path),
            "schema": "nemo-rl-strict-pair-submission-receipt-v2",
        },
        "recovery_query": None,
        "release": {
            "argv": [
                host_tools["scontrol"]["path"],
                "release",
                f"{job_ids['off']},{job_ids['on']}",
            ],
            "output_sha256_ascii_no_newline": _digest("release"),
            "status": 0,
        },
        "rollback_candidates": candidate_ids,
        "rollback_confirmed": None,
        "runtime_tools": {
            "manifest": pair["runtime_tools"]["manifest"],
            "schema": "nemo-rl-strict-runtime-tools-v2",
        },
        "scheduler_tools": scheduler_tools,
        "schema": "nemo-rl-strict-pair-submission-receipt-v2",
        "selection": pair["selection"],
        "source": {
            "bridge": pair["source"]["bridge"],
            "mcore": pair["source"]["mcore"],
        },
        "stage": "complete",
        "submission_contract": pair["scheduler_submission"]["contract"],
        "submission_nonce": nonce,
        "wandb": pair["wandb"],
    }
    if forgery == "legacy_release_oneliner":
        for name in ("pre_release_query", "post_release_query"):
            submission[name]["argv"][3] = "--oneliner"
    normalized_record_poison = {
        "numeric": 123,
        "bool": True,
        "list": [],
        "dict": {},
        "c0": "None\x00",
        "del": "None\x7f",
    }
    for field in ("job_state", "reason"):
        prefix = f"post_{field}_"
        if forgery is not None and forgery.startswith(prefix):
            poison = forgery.removeprefix(prefix)
            submission["post_release_query"]["records"]["off"][0][field] = normalized_record_poison[poison]
    submission_sha = _seal_document(submission_path, submission, lf=True)

    from tests.unit.utils.test_strict_model_transport import (
        _fixture_row,
        _request,
        _response,
        _server,
        _transcript_entry,
    )

    generation = {
        "seed_base": pair["campaign"]["generation_seed_base"],
        **{name: pair["campaign"]["generation"][name] for name in ("max_new_tokens", "temperature", "top_k", "top_p")},
    }
    transport_dir = results / "off" / "strict_model_transport"
    transport_dir.mkdir(mode=0o700, parents=True)
    capture_server = _server()
    arrivals = (2, 0, 3, 1)
    model_path = pair["artifacts"]["model"]["path"]
    requests = [_request(index) for index in range(4)]
    responses = [_response(index) for index in range(4)]
    for request, response in zip(requests, responses, strict=True):
        request["model"] = model_path
        response["model"] = model_path
    entries = [
        build_model_transport_call(
            pair_id="pair-abc",
            environment="reasoning_gym",
            arm="off",
            capture_server=capture_server,
            rollout_index=index,
            generation_seed=requests[index]["seed"],
            arrival_index=arrivals[index],
            model_path=model_path,
            request_body=canonical_ascii_json(requests[index]),
            response_body=canonical_ascii_json(responses[index]),
            expected_request_payload=requests[index],
            expected_response_payload=responses[index],
        )
        for index in range(4)
    ]
    transport_bundle = build_model_transport_bundle(
        pair_id="pair-abc",
        environment="reasoning_gym",
        arm="off",
        model_transport_policy=pair["model_transport"],
        capture_server=capture_server,
        entries=entries,
        model_path=model_path,
    )
    ordered_sha = transport_bundle["ordered_entries_sha256"]
    transport_bundle_path = transport_dir / "model-transport-bundle.json"
    transport_bundle_sha = _seal_document(transport_bundle_path, transport_bundle, lf=False)
    raw_path = transport_dir / "model-transport.jsonl"
    raw_sha = _seal_bytes(raw_path, b"".join(canonical_ascii_json(entry) + b"\n" for entry in entries))
    transport_bundle_ref = {
        "path": str(transport_bundle_path),
        "schema": "nemo-rl-strict-model-transport-bundle-v1",
        "sha256": transport_bundle_sha,
    }
    transcript_bindings = {
        "pair_manifest_sha256": pair_sha,
        "submission_receipt_sha256": submission_sha,
        "job_id": job_ids["off"],
        "run_id": pair["wandb"]["arms"]["off"]["run_id"],
        "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
        "verifier_source_sha256": pair["selection"]["gym_resources"]["verifier_source"]["sha256"],
        "config_sha256": pair["selection"]["config"]["sha256"],
        "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
    }

    def transcript_entry(index: int) -> dict[str, object]:
        entry = _transcript_entry(index, transport_bundle["entries"][index])
        entry["model_response"]["model"] = model_path
        entry["derived_verifier_request"]["response"]["model"] = model_path
        entry["verifier_response"]["response"]["model"] = model_path
        return entry

    transcript = build_transcript_bundle(
        pair_id="pair-abc",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=generation,
        bindings=transcript_bindings,
        fixture_row=_fixture_row(),
        model_transport_bundle=transport_bundle_ref,
        verifier_request_derivation=build_verifier_request_derivation(
            gym_gitlink_commit=pair["source"]["gym"]["gitlink_commit"],
            gym_tree=pair["source"]["gym"]["tree"],
            openai_version="2.6.1",
            pydantic_version="2.13.4",
        ),
        entry_inputs=[transcript_entry(index) for index in range(4)],
    )
    step1_dir = results / "off" / "strict_pair_step1_evidence"
    transcript_path = step1_dir / "transcript-bundle.json"
    transcript_sha = hashlib.sha256(canonical_ascii_json(transcript)).hexdigest()
    transcript_ref = {
        "path": str(transcript_path),
        "schema": "nemo-rl-strict-step1-transcript-bundle-v4",
        "sha256": transcript_sha,
    }
    ledger = _build_fixture_source_ledger(pair=pair, transcript=transcript, transcript_ref=transcript_ref)
    if forgery == "transcript_generation_extra_vllm_memory":
        transcript["generation"]["vllm_gpu_memory_utilization"] = 0.1
    elif forgery == "transcript_generation_missing_top_p":
        del transcript["generation"]["top_p"]
    if forgery in {
        "transcript_generation_extra_vllm_memory",
        "transcript_generation_missing_top_p",
    }:
        transcript_sha = hashlib.sha256(canonical_ascii_json(transcript)).hexdigest()
        transcript_ref["sha256"] = transcript_sha
        ledger["transcript_bundle"]["sha256"] = transcript_sha
    assert _seal_document(transcript_path, transcript, lf=False) == transcript_sha
    ledger_path = step1_dir / "main-ledger.json"
    ledger_sha = _seal_document(ledger_path, ledger, lf=False)
    ledger_ref = {
        "path": str(ledger_path),
        "schema": "nemo-rl-strict-main-step1-ledger-v5",
        "sha256": ledger_sha,
    }
    raw_ref = {
        "path": str(raw_path),
        "record_schema": "nemo-rl-strict-model-transport-call-v1",
        "record_count": 4,
        "sha256": raw_sha,
    }
    transport_manifest = build_model_transport_manifest(
        pair_id="pair-abc",
        environment="reasoning_gym",
        arm="off",
        pair_manifest_sha256=pair_sha,
        authenticated_job_id=job_ids["off"],
        submission_receipt_sha256=submission_sha,
        capture_server=capture_server,
        main_transcript_bundle=transcript_ref,
        main_ledger=ledger_ref,
        transport_bundle=transport_bundle_ref,
        transport_capture=raw_ref,
        model_transport_policy_sha256=pair["model_transport"]["policy_sha256"],
        entry_count=4,
        ordered_entries_sha256=ordered_sha,
    )
    if forgery == "transport_manifest_job":
        transport_manifest["authenticated_job_id"] = "9999999"
    transport_manifest_path = transport_dir / "model-transport-manifest.json"
    transport_manifest_sha = _seal_document(transport_manifest_path, transport_manifest, lf=False)
    evidence = {
        "schema": "nemo-rl-strict-step1-evidence-index-v4",
        "main_ledger": ledger_ref,
        "transcript_bundle": transcript_ref,
        "model_transport": {
            "schema": "nemo-rl-strict-model-transport-evidence-index-v1",
            "bundle": transport_bundle_ref,
            "manifest": {
                "path": str(transport_manifest_path),
                "schema": "nemo-rl-strict-model-transport-manifest-v1",
                "sha256": transport_manifest_sha,
            },
            "raw_log": raw_ref,
            "ordered_entries_sha256": ordered_sha,
        },
    }

    boundary = {key: value for key, value in pair["slurm_export_boundary"].items() if key != "arms"}
    boundary.update(
        {
            "arm": "off",
            "path": pair["slurm_export_boundary"]["arms"]["off"]["path"],
            "sha256": pair["slurm_export_boundary"]["arms"]["off"]["sha256"],
            "job_argv": [
                "--pair-manifest",
                str(pair_manifest_path),
                "--pair-manifest-sha256",
                pair_sha,
                "--arm",
                "off",
            ],
        }
    )
    selected_wandb = {
        "entity": pair["wandb"]["entity"],
        "group": pair["wandb"]["group"]["value"],
        "name": pair["wandb"]["arms"]["off"]["name"],
        "name_template": pair["wandb"]["arms"]["off"]["name_template"],
        "project": pair["wandb"]["project"],
        "resume": pair["wandb"]["resume"],
        "run_id": pair["wandb"]["arms"]["off"]["run_id"],
        "run_id_derivation": pair["wandb"]["run_id_derivation"],
    }
    runtime_tools = pair["runtime_tools"]["document"]
    pre = {key: _digest(f"pre-{key}") for key in PAIR_PRE_RECEIPT_KEYS}
    pre.update(
        {
            "arm": "off",
            "bridge_runnable_manifest_sha256": pair["deployment"]["bridge_runnable_manifest_sha256"],
            "config_sha256": pair["selection"]["config"]["sha256"],
            "container_entry_boundary": pair["container_entry_boundary"],
            "container_entry_boundary_sha256": hashlib.sha256(
                canonical_ascii_json(pair["container_entry_boundary"])
            ).hexdigest(),
            "container_sha256": pair["artifacts"]["container"]["sha256"],
            "deployment_ready": pair["deployment"]["ready"],
            "deployment_ready_file_sha256": pair["deployment"]["ready_file_sha256"],
            "deployment_ready_sha256": pair["deployment"]["ready"],
            "deterministic_controls": {"required": True},
            "entrypoint_sha256": pair["source"]["entrypoint_sha256"],
            "environment": "reasoning_gym",
            "execution_environment": pair["execution_environment"],
            "fixture_rows": 5,
            "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
            "gpus_per_node": 4,
            "gym_gitlink_commit": pair["source"]["gym"]["gitlink_commit"],
            "gym_tree": pair["source"]["gym"]["tree"],
            "job_account": "nemotron_sw_post",
            "job_id": job_ids["off"],
            "job_name": identities["off"]["job_name"],
            "job_num_nodes": 1,
            "job_partition": "batch",
            "job_qos": "normal",
            "mcore_runnable_manifest_sha256": pair["deployment"]["mcore_runnable_manifest_sha256"],
            "model_tree_sha256_v1": pair["artifacts"]["model"]["tree_sha256_v1"],
            "nemo_runnable_manifest_sha256": pair["deployment"]["nemo_runnable_manifest_sha256"],
            "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
            "pair_campaign_sha256": pair["pair_campaign_sha256"],
            "pair_id": "pair-abc",
            "pair_manifest_sha256": pair_sha,
            "phase": "PRE",
            "post_verified": False,
            "restart_count": 0,
            "reward_semantics_config_sha256": pair["selection"]["config"]["sha256"],
            "reward_semantics_contract_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
            "runtime_attestation_expected_count": 4,
            "runtime_attestation_marker_sha256": pair["runtime_attestation"]["lines"]["off"]["sha256_ascii_no_newline"],
            "runtime_attestation_receipt_dir": (
                f"{root}/results/off/shared_prefix_determinism_receipts/" f"{job_ids['off']}-0"
            ),
            "runtime_attestation_receipt_dir_device": 11,
            "runtime_attestation_receipt_dir_inode": 12,
            "runtime_tool_container_python_path": runtime_tools["container"]["python"]["path"],
            "runtime_tool_container_python_sha256": runtime_tools["container"]["python"]["sha256"],
            "runtime_tool_container_uv_path": runtime_tools["container"]["uv"]["path"],
            "runtime_tool_container_uv_sha256": runtime_tools["container"]["uv"]["sha256"],
            "runtime_tool_host_python_path": runtime_tools["host"]["python"]["path"],
            "runtime_tool_host_python_sha256": runtime_tools["host"]["python"]["sha256"],
            "runtime_tool_manifest_path": pair["runtime_tools"]["manifest"]["path"],
            "runtime_tool_manifest_sha256": pair["runtime_tools"]["manifest"]["sha256"],
            "runtime_tool_uv_shim_path": runtime_tools["container"]["uv_shim"]["path"],
            "runtime_tool_uv_shim_sha256": runtime_tools["container"]["uv_shim"]["sha256"],
            "sandbox_container_sha256": pair["artifacts"]["sandbox_container"]["sha256"],
            "scheduler_client_environment": {
                "ambient_merge": False,
                "SLURM_CONF": scheduler_tools["client_environment"]["variables"]["SLURM_CONF"],
                "propagated_to_inner_ray": True,
            },
            "schema": "nemo-rl-strict-pair-job-receipt-v2",
            "selected_config_sha256": pair["selection"]["config"]["sha256"],
            "selection": pair["selection"],
            "slurm_export_boundary": boundary,
            "slurm_export_boundary_sha256": hashlib.sha256(canonical_ascii_json(boundary)).hexdigest(),
            "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
            "source": pair["source"],
            "source_head": pair["source"]["head"],
            "source_tree": pair["source"]["tree"],
            "strict_pair_arm_wrapper_sha256": pair["source"]["arm_wrapper_sha256"],
            "strict_pair_contract_sha256": pair["source"]["contract_sha256"],
            "strict_pair_parent_wrapper_sha256": pair["source"]["parent_wrapper_sha256"],
            "submission_contract_path": pair["scheduler_submission"]["contract"]["path"],
            "submission_contract_sha256": pair["scheduler_submission"]["contract"]["sha256"],
            "submission_nonce": nonce,
            "submission_receipt_path": str(submission_path),
            "submission_receipt_sha256": submission_sha,
            "wandb": selected_wandb,
            "wrapper_sha256": pair["source"]["job_wrapper"]["sha256"],
        }
    )
    receipt_dir = results / "off" / "strict_pair_job_state" / f"{job_ids['off']}-0" / "receipts"
    pre_sha = _seal_document(receipt_dir / "PRE.json", pre, lf=True)
    exit_receipt = {
        **pre,
        "driver_exit_code": 0,
        "hardware": {"gpu_count": 4},
        "phase": "EXIT",
        "post_verified": True,
        "pre_receipt_sha256": pre_sha,
        "runtime_attestation_actual_count": 4,
        "runtime_attestation_aggregate_sha256": _digest("attestation-aggregate"),
        "runtime_attestation_receipts_sha256": {
            f"rank-{index}.receipt": _digest(f"attestation-{index}") for index in range(4)
        },
        "scheduler_device_environment": {"cuda_visible_devices": "0,1,2,3"},
        "step1_evidence": evidence,
    }
    if forgery == "exit_config":
        exit_receipt["config_sha256"] = _digest("forged-exit-config")
    exit_path = receipt_dir / "EXIT.json"
    exit_sha = _seal_document(exit_path, exit_receipt, lf=True)
    return {
        "pair_manifest": pair,
        "pair_manifest_path": str(pair_manifest_path),
        "pair_manifest_sha256": pair_sha,
        "pair_submission_receipt_path": str(submission_path),
        "pair_submission_receipt_sha256": submission_sha,
        "trusted_off_exit_receipt_path": str(exit_path),
        "trusted_off_exit_receipt_sha256": exit_sha,
    }


def test_load_authenticated_off_source_capture_derives_all_authority(
    tmp_path: Path,
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path)
    source = load_authenticated_off_source_capture(**kwargs)
    assert isinstance(source, AuthenticatedOffSourceCapture)
    assert source.document["authenticated_job"]["job_id"] == "6787903"
    assert source.document["arm"] == "off"
    assert source.document["restart_count"] == 0
    assert source.document["job_receipts"]["exit"] == {
        "path": kwargs["trusted_off_exit_receipt_path"],
        "schema": "nemo-rl-strict-pair-job-receipt-v2",
        "sha256": kwargs["trusted_off_exit_receipt_sha256"],
    }
    assert source.trusted_off_exit_receipt_path == kwargs["trusted_off_exit_receipt_path"]
    assert source.trusted_off_exit_receipt_sha256 == kwargs["trusted_off_exit_receipt_sha256"]
    for name in ("pre_release_query", "post_release_query"):
        assert source.pair_submission_receipt[name]["argv"][3] == "--json"
    assert len(source.transport_records) == 4
    detached = source.document
    detached["authenticated_job"]["job_id"] = "999"
    assert source.source_capture["authenticated_job"]["job_id"] == "6787903"


def test_trusted_off_exit_is_loaded_before_supporting_pair_authority(
    tmp_path: Path,
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path)
    kwargs["pair_manifest_path"] = str(tmp_path / "absent-pair.json")
    kwargs["trusted_off_exit_receipt_sha256"] = _digest("wrong-oob-exit-anchor")
    with pytest.raises(ValueError, match="trusted source OFF EXIT receipt bytes differ"):
        load_authenticated_off_source_capture(**kwargs)


def test_trusted_off_exit_rejects_coherent_rewrite_and_self_minted_ref(
    tmp_path: Path,
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path)
    source = load_authenticated_off_source_capture(**kwargs)
    exit_path = Path(str(kwargs["trusted_off_exit_receipt_path"]))
    forged_exit = json.loads(exit_path.read_text(encoding="ascii"))
    forged_exit["hardware"] = {
        **forged_exit["hardware"],
        "gpu_count": 8,
    }
    forged_sha = _rewrite_sealed_document(exit_path, forged_exit, lf=True)
    assert forged_sha != kwargs["trusted_off_exit_receipt_sha256"]

    # Model the old fail-open: every mutable, derived view self-mints the new
    # digest and remains internally coherent.  Reload must still use the frozen
    # OOB primitive rather than any detached document/reference.
    source.source_capture["job_receipts"]["exit"]["sha256"] = forged_sha
    source.exit_receipt.clear()
    source.exit_receipt.update(forged_exit)
    with pytest.raises(ValueError, match="trusted source OFF EXIT receipt bytes differ"):
        build_replay_execution_manifest(
            authenticated_source=source,
            attempt_id="replay-1",
        )


def test_trusted_off_exit_path_must_match_released_off_identity(
    tmp_path: Path,
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path)
    original = Path(str(kwargs["trusted_off_exit_receipt_path"]))
    alternate_parent = tmp_path / "alternate"
    alternate_parent.mkdir(mode=0o700)
    alternate = alternate_parent / "EXIT.json"
    alternate.write_bytes(original.read_bytes())
    alternate.chmod(0o400)
    kwargs["trusted_off_exit_receipt_path"] = str(alternate)
    with pytest.raises(ValueError, match="trusted source OFF EXIT path differs"):
        load_authenticated_off_source_capture(**kwargs)


@pytest.mark.parametrize(
    "forgery",
    [
        "transcript_generation_extra_vllm_memory",
        "transcript_generation_missing_top_p",
    ],
)
def test_load_authenticated_off_source_capture_rejects_transcript_generation_shape(
    tmp_path: Path, forgery: str
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path, forgery=forgery)
    with pytest.raises(ValueError, match="source step1 generation contract differs from Pair"):
        load_authenticated_off_source_capture(**kwargs)


@pytest.mark.parametrize(
    ("forgery", "message"),
    [
        ("off_comment", "authenticate exactly one off job"),
        ("legacy_release_oneliner", "query lifecycle evidence differs"),
        ("exit_config", "PRE/EXIT common field differs"),
        ("transport_manifest_job", "transport manifest differs"),
    ],
)
def test_load_authenticated_off_source_capture_rejects_coherent_forgery(
    tmp_path: Path, forgery: str, message: str
) -> None:
    kwargs = _authenticated_source_fixture(tmp_path, forgery=forgery)
    with pytest.raises(ValueError, match=message):
        load_authenticated_off_source_capture(**kwargs)


@pytest.mark.parametrize("field", ("job_state", "reason"))
@pytest.mark.parametrize("poison", ("numeric", "bool", "list", "dict", "c0", "del"))
def test_load_authenticated_off_source_capture_rejects_unclean_release_record_strings(
    tmp_path: Path, field: str, poison: str
) -> None:
    kwargs = _authenticated_source_fixture(
        tmp_path,
        forgery=f"post_{field}_{poison}",
    )
    with pytest.raises(ValueError, match="scheduler state/reason differs"):
        load_authenticated_off_source_capture(**kwargs)
