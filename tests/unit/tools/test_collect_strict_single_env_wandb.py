# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_PATH = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/collect_strict_single_env_wandb.py"
EVALUATOR_PATH = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/evaluate_strict_single_env_live.py"
PAIR_TEST_PATH = Path(__file__).with_name("test_nano35_single_env_pair.py")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECTOR = _load_module("strict_pair_wandb_collector", COLLECTOR_PATH)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _config(pair_id: str, arm: str, environment: str) -> dict[str, Any]:
    return {
        "max_num_steps": 100,
        "epochs": 20,
        "steps_per_epoch": 5,
        "fixture_rows": 5,
        "num_prompts_per_step": 1,
        "num_generations_per_prompt": 4,
        "seed": 42,
        "generation_seed_base": 42,
        "data_shuffle": False,
        "reward_scaling_enabled": False,
        "reward_shaping_enabled": False,
        "shared_prefix_mode": "observe" if arm == "off" else "train",
        "wandb_enabled": True,
        "tensorboard_enabled": False,
        "wandb_entity": "nvidia",
        "wandb_project": "nano35-rlvr-convergence",
        "wandb_group": f"{environment}-{pair_id}",
        "wandb_run_name": f"{arm}-{environment}-{pair_id}",
        "tensor_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "mtp_num_layers": 5,
        "mtp_use_repeated_layer": True,
        "mtp_detach_heads": True,
        "mtp_loss_scaling_factor": 0.3,
        "slurm_partition": "batch",
        "slurm_account": "nemotron_sw_post",
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }


def _run_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster": {"gpus_per_node": 4, "num_nodes": 1},
        "grpo": {
            "max_num_steps": config["max_num_steps"],
            "max_num_epochs": config["epochs"],
            "num_prompts_per_step": config["num_prompts_per_step"],
            "num_generations_per_prompt": config["num_generations_per_prompt"],
            "seed": config["seed"],
            "reward_scaling": {"enabled": config["reward_scaling_enabled"]},
            "reward_shaping": {"enabled": config["reward_shaping_enabled"]},
        },
        "data": {"shuffle": config["data_shuffle"]},
        "logger": {
            "wandb_enabled": config["wandb_enabled"],
            "tensorboard_enabled": config["tensorboard_enabled"],
            "wandb": {
                "entity": config["wandb_entity"],
                "project": config["wandb_project"],
                "name": config["wandb_run_name"],
            },
        },
        "policy": {
            "shared_prefix_training": {
                "mode": config["shared_prefix_mode"],
                "require_deterministic_execution": True,
            },
            "megatron_cfg": {
                "tensor_model_parallel_size": config["tensor_parallel_size"],
                "context_parallel_size": config["context_parallel_size"],
                "sequence_parallel": config["sequence_parallel"],
                "pipeline_model_parallel_size": config["pipeline_parallel_size"],
                "expert_model_parallel_size": config["expert_parallel_size"],
                "expert_tensor_parallel_size": config["expert_tensor_parallel_size"],
                "mtp_num_layers": config["mtp_num_layers"],
                "mtp_use_repeated_layer": config["mtp_use_repeated_layer"],
                "mtp_detach_heads": config["mtp_detach_heads"],
                "mtp_loss_scaling_factor": config["mtp_loss_scaling_factor"],
            },
            "generation": {
                "nemo_gym_add_seed_per_rollout": True,
                "nemo_gym_per_rollout_seed_base": config["generation_seed_base"],
                "max_new_tokens": config["max_new_tokens"],
                "temperature": config["temperature"],
                "top_k": config["top_k"],
                "top_p": config["top_p"],
            },
        },
    }


def _wandb_identity(pair_id: str, environment: str) -> dict[str, Any]:
    return {
        "entity": "nvidia",
        "project": "nano35-rlvr-convergence",
        "group": {
            "template": "{environment}-{pair_id}",
            "value": f"{environment}-{pair_id}",
        },
        "resume": "never",
        "arms": {
            arm: {
                "name_template": f"{arm}-{{environment}}-{{pair_id}}",
                "name": f"{arm}-{environment}-{pair_id}",
                "run_id": COLLECTOR.derive_wandb_run_id(environment, pair_id, arm),
            }
            for arm in ("off", "on")
        },
        "run_id_derivation": COLLECTOR.WANDB_RUN_ID_DERIVATION,
    }


@dataclass
class FakeRun:
    entity: str
    project: str
    id: str
    name: str
    group: str
    state: str
    path: list[str]
    config: dict[str, Any]
    rows: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.scan_calls: list[dict[str, Any]] = []

    def scan_history(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.scan_calls.append(kwargs)
        # Match W&B's all-keys intersection so a future regression that passes
        # the full sparse inventory cannot remain hidden by this fake.
        if "keys" in kwargs:
            keys = kwargs["keys"]
            return [row for row in self.rows if all(key in row for key in keys)]
        return copy.deepcopy(self.rows)


@dataclass
class FakeApi:
    selected_run: FakeRun

    def __post_init__(self) -> None:
        self.paths: list[str] = []
        self.viewer = type("AuthenticatedViewer", (), {"entity": "nvidia"})()

    def run(self, path: str) -> FakeRun:
        self.paths.append(path)
        return self.selected_run


@dataclass
class FakePairApi:
    runs: dict[str, FakeRun]

    def __post_init__(self) -> None:
        self.paths: list[str] = []
        self.viewer = type("AuthenticatedViewer", (), {"entity": "nvidia"})()

    def run(self, path: str) -> FakeRun:
        self.paths.append(path)
        run_id = path.rsplit("/", 1)[-1]
        return self.runs[run_id]


@dataclass
class Fixture:
    pair: dict[str, Any]
    pair_sha256: str
    submission: dict[str, Any]
    submission_sha256: str
    contract: dict[str, Any]
    collector_sha256: str
    run: FakeRun
    api: FakeApi

    def collect(self, arm: str = "off") -> dict[str, Any]:
        return COLLECTOR.collect_run_export(
            pair_manifest=self.pair,
            pair_manifest_sha256=self.pair_sha256,
            submission_receipt=self.submission,
            submission_receipt_sha256=self.submission_sha256,
            acceptance_contract=self.contract,
            arm=arm,
            api=self.api,
            fetched_at_unix_ns=1_725_000_000_000_000_000,
            collector_sha256=self.collector_sha256,
        )


def _fixture(arm: str = "off") -> Fixture:
    pair_id = "strict-spfx-ab"
    environment = "reasoning_gym"
    wandb = _wandb_identity(pair_id, environment)
    pair = {
        "schema": COLLECTOR.PAIR_MANIFEST_SCHEMA,
        "pair_id": pair_id,
        "selection": {"environment": environment},
        "arms": {"off": "observe", "on": "train"},
        "wandb": wandb,
    }
    pair_raw = COLLECTOR.canonical_json_bytes(pair, "test Pair") + b"\n"
    pair_sha256 = hashlib.sha256(pair_raw).hexdigest()
    job_ids = {"off": "41001", "on": "41002"}
    submission = {
        "schema": COLLECTOR.SUBMISSION_RECEIPT_SCHEMA,
        "outcome": "released",
        "stage": "complete",
        "pair": {"id": pair_id, "manifest": {"sha256": pair_sha256}},
        "wandb": copy.deepcopy(wandb),
        "held_submissions": {
            selected_arm: {"candidate_job_id": job_ids[selected_arm]} for selected_arm in ("off", "on")
        },
        "authenticated_jobs": {
            selected_arm: [
                {
                    "job_id": job_ids[selected_arm],
                    "job_name": wandb["arms"][selected_arm]["name"],
                }
            ]
            for selected_arm in ("off", "on")
        },
    }
    submission_raw = COLLECTOR.canonical_json_bytes(submission, "test submission") + b"\n"
    submission_sha256 = hashlib.sha256(submission_raw).hexdigest()
    collector_sha256 = hashlib.sha256(COLLECTOR_PATH.read_bytes()).hexdigest()
    configs = {selected_arm: _config(pair_id, selected_arm, environment) for selected_arm in ("off", "on")}
    contract = {
        "schema": COLLECTOR.ACCEPTANCE_CONTRACT_SCHEMA,
        "pair": {
            "pair_id": pair_id,
            "environment": environment,
            "entity": wandb["entity"],
            "project": wandb["project"],
            "group": wandb["group"]["value"],
            "run_ids": {selected_arm: wandb["arms"][selected_arm]["run_id"] for selected_arm in ("off", "on")},
        },
        "configs": configs,
        "provenance": {
            "common": {key: _digest(f"common:{key}") for key in COLLECTOR.COMMON_PROVENANCE_KEYS},
            "source_commits": {
                key: hashlib.sha1(f"commit:{key}".encode("ascii")).hexdigest() for key in COLLECTOR.SOURCE_KEYS
            },
            "source_git_trees": {
                key: hashlib.sha1(f"tree:{key}".encode("ascii")).hexdigest() for key in COLLECTOR.SOURCE_KEYS
            },
            "trusted_oob_declarations": {
                "schema": COLLECTOR.TRUSTED_OOB_DECLARATIONS_SCHEMA,
                "assurance": COLLECTOR.TRUSTED_OOB_DECLARATIONS_ASSURANCE,
                "common": {key: _digest(f"declared-common:{key}") for key in COLLECTOR.TRUSTED_DECLARED_COMMON_KEYS},
                "arms": {
                    selected_arm: {
                        key: _digest(f"declared-{selected_arm}:{key}") for key in COLLECTOR.TRUSTED_DECLARED_ARM_KEYS
                    }
                    for selected_arm in ("off", "on")
                },
                "sources": {
                    name: {
                        "commit": hashlib.sha1(f"declared-commit:{name}".encode("ascii")).hexdigest(),
                        "git_tree": hashlib.sha1(f"declared-tree:{name}".encode("ascii")).hexdigest(),
                        "source_tree_sha256": _digest(f"declared-source-tree:{name}"),
                    }
                    for name in COLLECTOR.TRUSTED_DECLARED_SOURCE_KEYS
                },
            },
            "topology": {"allocated_nodes": 1, "gpus_per_node": 4},
            "arms": {
                selected_arm: {key: _digest(f"{selected_arm}:{key}") for key in COLLECTOR.ARM_PROVENANCE_KEYS}
                for selected_arm in ("off", "on")
            },
        },
    }
    contract["provenance"]["common"]["pair_manifest_sha256"] = pair_sha256
    contract["provenance"]["common"]["wandb_exporter_sha256"] = collector_sha256
    contract["provenance"]["common"]["acceptance_contract_sha256"] = COLLECTOR.acceptance_contract_payload_sha256(
        contract
    )
    identity = wandb["arms"][arm]
    rows = [
        {
            "_step": 2,
            "train/reward": -0.0,
            "train/total_num_tokens": 400.0,
            "unrequested": float("inf"),
        },
        {
            "_step": 1,
            "train/reward": 0.5,
            "train/rollout/samples": 4.0,
            "_runtime": 123.0,
        },
        {"_step": 1, "timing/train/policy_training": 10},
        {"_step": 3, "unrequested": "ignored"},
    ]
    run = FakeRun(
        entity=wandb["entity"],
        project=wandb["project"],
        id=identity["run_id"],
        name=identity["name"],
        group=wandb["group"]["value"],
        state="finished",
        path=[wandb["entity"], wandb["project"], identity["run_id"]],
        config=_run_config(configs[arm]),
        rows=rows,
    )
    return Fixture(
        pair=pair,
        pair_sha256=pair_sha256,
        submission=submission,
        submission_sha256=submission_sha256,
        contract=contract,
        collector_sha256=collector_sha256,
        run=run,
        api=FakeApi(run),
    )


def _write_document(path: Path, value: dict[str, Any], *, trailing_lf: bool) -> str:
    raw = COLLECTOR.canonical_json_bytes(value, f"test {path.name}")
    if trailing_lf:
        raw += b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _canary_local_summary(arm: str) -> dict[str, Any]:
    return {
        "train/total_num_tokens": 3572,
        "train/shared_prefix/valid_loss_tokens": 3072,
        "train/shared_prefix/fallback_sequences": 0,
        "train/shared_prefix/complete_groups": 1,
        "train/shared_prefix/eligible_sequences": 4,
        "train/shared_prefix/total_tokens": 3572,
        "train/raw_environment_reward/min": 0,
        "train/raw_environment_reward/max": 0,
        "train/shared_prefix/ideal_shared_token_work": 3197,
        "train/shared_prefix/ideal_token_reduction": 0.10498320268756998,
        "_step": 2,
        "train/shared_prefix/ideal_token_work_speedup": 1.1172974663747264,
        "train/shared_prefix/shareable_prompt_tokens": 375,
        "train/shared_prefix/total_sequences": 4,
        "train/rollout/samples": 4,
        "train/raw_environment_reward": 0,
        # The observed OFF canary reported 1 here while its authoritative
        # W&B _step was 2.  The diagnostic must never use this as progress.
        "rollout/train_steps": 1 if arm == "off" else 2,
        "timing/train/policy_training": (3.7598189060081495 if arm == "off" else 5.811881213972811),
        "train/global_valid_seqs": 4,
        "train/global_valid_toks": 3072,
        "train/num_valid_samples": 4,
        "unselected/canary_field": 17,
    }


def _summary_document(value: dict[str, Any]) -> Any:
    raw = json.dumps(value, separators=(", ", ": "), ensure_ascii=True).encode("ascii")
    return COLLECTOR.Document(
        value=copy.deepcopy(value),
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _diagnostic_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_ids = {"off": "t4o5v4hn", "on": "8mfmyvrf"}
    pair_label = "exploratory-rgy2-flashinfer-non-acceptance"
    pair_root = tmp_path / pair_label
    pair_root.mkdir()
    paths: dict[str, Path] = {}
    for arm, stamp in (("off", "20260902_180653"), ("on", "20260902_180656")):
        path = pair_root / arm / "wandb" / f"run-{stamp}-{run_ids[arm]}" / "files" / "wandb-summary.json"
        path.parent.mkdir(parents=True)
        summary = _canary_local_summary(arm)
        raw = json.dumps(summary, separators=(", ", ": "), ensure_ascii=True).encode("ascii")
        path.write_bytes(raw)
        paths[arm] = path
    documents = {arm: COLLECTOR.load_document(path, f"test {arm} local W&B summary") for arm, path in paths.items()}
    kwargs = {
        "pair_label": pair_label,
        "pair_root": pair_root,
        "environment": "reasoning_gym",
        "run_ids": run_ids,
        "summary_paths": paths,
        "expected_summary_sha256": {arm: documents[arm].sha256 for arm in ("off", "on")},
        "collected_at_unix_ns": 1_725_000_000_000_000_000,
        "collector_sha256": _digest("diagnostic collector"),
    }
    return COLLECTOR.collect_diagnostic_local_summary_pair(**kwargs), kwargs


def _canary_history_metrics(arm: str, step: int) -> dict[str, Any]:
    summary = _canary_local_summary(arm)
    metrics = {metric: summary[metric] for metric in COLLECTOR.DIAGNOSTIC_LOCAL_SUMMARY_METRICS}
    if step == 1:
        reward = 0.25 if arm == "off" else 1.0
        metrics.update(
            {
                "train/raw_environment_reward": reward,
                "train/raw_environment_reward/min": 0 if arm == "off" else 1,
                "train/raw_environment_reward/max": 1,
                "timing/train/policy_training": (43.6836 if arm == "off" else 55.5487),
                "train/global_valid_toks": 3035 if arm == "off" else 2622,
                "train/total_num_tokens": 3515 if arm == "off" else 3102,
                "train/shared_prefix/total_tokens": 3515 if arm == "off" else 3102,
                "train/shared_prefix/valid_loss_tokens": (3035 if arm == "off" else 2622),
                "train/shared_prefix/ideal_shared_token_work": (3200 if arm == "off" else 2800),
                "train/shared_prefix/ideal_token_reduction": (0.09 if arm == "off" else 0.1),
                "train/shared_prefix/ideal_token_work_speedup": (1.09 if arm == "off" else 1.1),
            }
        )
    return metrics


def _diagnostic_history_fixture() -> tuple[dict[str, Any], dict[str, Any], FakePairApi]:
    pair_label = "exploratory-rgy2-flashinfer-non-acceptance"
    run_ids = {"off": "t4o5v4hn", "on": "8mfmyvrf"}
    run_names = {
        "off": f"{pair_label}-off",
        "on": f"{pair_label}-on",
    }
    runs = {}
    for arm in ("off", "on"):
        step_one = _canary_history_metrics(arm, 1)
        split = len(step_one) // 2
        items = list(step_one.items())
        rows = [
            {"_step": 0, "rollout/train_steps": 0, "_runtime": 1.0},
            {"_step": 1, **dict(items[:split])},
            {"_step": 1, **dict(items[split:])},
            {"_step": 2, **_canary_history_metrics(arm, 2)},
        ]
        runs[run_ids[arm]] = FakeRun(
            entity="nvidia",
            project="nano35-rlvr-convergence",
            id=run_ids[arm],
            name=run_names[arm],
            group=pair_label,
            state="finished",
            path=["nvidia", "nano35-rlvr-convergence", run_ids[arm]],
            config={},
            rows=rows,
        )
    api = FakePairApi(runs)
    kwargs = {
        "pair_label": pair_label,
        "environment": "reasoning_gym",
        "entity": "nvidia",
        "project": "nano35-rlvr-convergence",
        "group": pair_label,
        "run_ids": run_ids,
        "run_names": run_names,
        "api": api,
        "fetched_at_unix_ns": 1_725_000_000_000_000_000,
        "collector_sha256": _digest("diagnostic history collector"),
    }
    return COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs), kwargs, api


def test_requested_metric_inventory_matches_offline_evaluator() -> None:
    evaluator = _load_module("strict_pair_evaluator_for_collector", EVALUATOR_PATH)
    for (
        environment,
        verifier_metric,
    ) in COLLECTOR.VERIFIER_METRIC_BY_ENVIRONMENT.items():
        assert COLLECTOR.requested_metrics(environment) == evaluator._requested_history_metrics(verifier_metric)
    assert COLLECTOR.INTEGER_METRICS == evaluator.INTEGER_METRICS
    assert COLLECTOR.RUN_EXPORT_SCHEMA == evaluator.RUN_EXPORT_SCHEMA
    assert COLLECTOR.PAIR_MANIFEST_SCHEMA == evaluator.PAIR_MANIFEST_SCHEMA
    assert COLLECTOR.SUBMISSION_RECEIPT_SCHEMA == evaluator.SUBMISSION_RECEIPT_SCHEMA
    assert COLLECTOR.WANDB_API_BASE_URL == evaluator.WANDB_API_BASE_URL
    assert COLLECTOR.WANDB_HISTORY_METHOD == evaluator.WANDB_HISTORY_METHOD
    assert COLLECTOR.WANDB_EXPORT_CANONICALIZATION == evaluator.WANDB_EXPORT_CANONICALIZATION
    assert COLLECTOR.WANDB_RUN_ID_DERIVATION == evaluator.WANDB_RUN_ID_DERIVATION
    assert COLLECTOR.WANDB_SDK_VERSION == evaluator.WANDB_SDK_VERSION
    assert COLLECTOR.MAX_HISTORY_ROWS == evaluator.MAX_WANDB_HISTORY_ROWS
    assert COLLECTOR.COMMON_PROVENANCE_KEYS == evaluator.COMMON_PROVENANCE_KEYS
    assert "terminal_scheduler_collector_sha256" in COLLECTOR.COMMON_PROVENANCE_KEYS


def test_production_exports_retain_pairable_reward_speed_and_work_history() -> None:
    exports = {}
    for arm, rewards, seconds in (
        ("off", (0, 0.5), (6, 5.0)),
        ("on", (0.0, 0.75), (4.0, 3)),
    ):
        fixture = _fixture(arm)
        fixture.run.rows = [
            {
                "_step": step,
                "train/raw_environment_reward": rewards[step - 1],
                "timing/train/policy_training": seconds[step - 1],
                "train/total_num_tokens": 3572.0,
                "train/global_valid_toks": 3072,
            }
            for step in (1, 2)
        ]
        exports[arm] = fixture.collect(arm)

    for arm in ("off", "on"):
        assert exports[arm]["capture"]["summary_fallback_used"] is False
        assert exports[arm]["history"] == sorted(exports[arm]["history"], key=lambda row: row["_step"])
        assert [row["_step"] for row in exports[arm]["history"]] == [1, 2]
        for row in exports[arm]["history"]:
            assert type(row["train/raw_environment_reward"]) is float
            assert type(row["timing/train/policy_training"]) is float
            assert type(row["train/total_num_tokens"]) is int
            assert type(row["train/global_valid_toks"]) is int
    assert [
        exports["on"]["history"][index]["train/raw_environment_reward"]
        - exports["off"]["history"][index]["train/raw_environment_reward"]
        for index in range(2)
    ] == [0.0, 0.25]


def test_canary_local_summaries_make_only_an_exact_non_acceptance_latest_point(
    tmp_path: Path,
) -> None:
    diagnostic, kwargs = _diagnostic_fixture(tmp_path)

    assert set(diagnostic) == {
        "schema",
        "assurance",
        "pair",
        "capture",
        "arms",
        "latest_comparison",
        "acceptance_boundary",
        "diagnostic_receipt",
    }
    assert diagnostic["schema"] == COLLECTOR.DIAGNOSTIC_PAIR_SUMMARY_SCHEMA
    assert diagnostic["assurance"] == COLLECTOR.DIAGNOSTIC_PAIR_SUMMARY_ASSURANCE
    assert diagnostic["pair"] == {
        "label": kwargs["pair_label"],
        "root": kwargs["pair_root"].as_posix(),
        "environment": "reasoning_gym",
        "run_ids": {"off": "t4o5v4hn", "on": "8mfmyvrf"},
    }
    capture = diagnostic["capture"]
    assert set(capture) == {
        "acceptance_eligible",
        "summary_progress_field",
        "collected_at_unix_ns",
        "collector_sha256",
        "history_complete",
        "ignored_progress_fields",
        "limitations",
        "method",
        "selected_metrics",
    }
    assert capture["acceptance_eligible"] is False
    assert capture["history_complete"] is False
    assert capture["summary_progress_field"] == "_step"
    assert capture["ignored_progress_fields"] == ["rollout/train_steps"]
    assert capture["selected_metrics"] == [
        "_step",
        *COLLECTOR.DIAGNOSTIC_LOCAL_SUMMARY_METRICS,
    ]
    assert "train/raw_environment_reward" in COLLECTOR.requested_metrics("reasoning_gym")
    assert "timing/train/policy_training" in COLLECTOR.requested_metrics("reasoning_gym")

    for arm in ("off", "on"):
        latest = diagnostic["arms"][arm]["latest"]
        assert latest["_step"] == 2
        assert type(latest["metrics"]["train/raw_environment_reward"]) is float
        assert type(latest["metrics"]["train/total_num_tokens"]) is int
        assert "rollout/train_steps" not in latest["metrics"]
        assert diagnostic["arms"][arm]["source"]["sha256"] == (kwargs["expected_summary_sha256"][arm])
        source = diagnostic["arms"][arm]["source"]
        assert source["source_keys"] == sorted(_canary_local_summary(arm))
        assert source["source_key_count"] == len(source["source_keys"])
        assert source["source_keyset_sha256"] == COLLECTOR.canonical_json_sha256(
            source["source_keys"], f"{arm} source keys"
        )
    comparison = diagnostic["latest_comparison"]
    assert comparison["step"] == 2
    assert comparison["raw_environment_reward"] == {
        "off": 0.0,
        "on": 0.0,
        "on_minus_off": 0.0,
    }
    assert comparison["policy_training_seconds"]["off"] == 3.7598189060081495
    assert comparison["policy_training_seconds"]["on"] == 5.811881213972811
    assert comparison["policy_training_seconds"]["same_work"] is True
    assert comparison["policy_training_seconds"]["off_over_on"] == pytest.approx(3.7598189060081495 / 5.811881213972811)
    assert "curves" not in diagnostic
    assert diagnostic["acceptance_boundary"] == {
        "eligible": False,
        "equal_summary_cursors": [2],
        "reason": "diagnostic_local_summary_is_not_complete_authenticated_history",
        "required_complete_steps": {"first": 1, "last": 100, "count": 100},
        "reward_claim_scope": "none_diagnostic_only",
    }

    raw = COLLECTOR.serialize_diagnostic_pair_summary(diagnostic)
    assert raw == COLLECTOR.canonical_json_bytes(diagnostic, "diagnostic") + b"\n"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert all(byte < 128 for byte in raw)
    payload = {key: item for key, item in diagnostic.items() if key != "diagnostic_receipt"}
    assert diagnostic["diagnostic_receipt"] == {
        "canonicalization": COLLECTOR.WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": COLLECTOR.canonical_json_sha256(payload, "payload"),
        "off_source_sha256": kwargs["expected_summary_sha256"]["off"],
        "on_source_sha256": kwargs["expected_summary_sha256"]["on"],
    }


def test_diagnostic_ratio_is_suppressed_when_work_is_not_matched(
    tmp_path: Path,
) -> None:
    _, kwargs = _diagnostic_fixture(tmp_path)
    on_path = kwargs["summary_paths"]["on"]
    on_summary = _canary_local_summary("on")
    on_summary["train/global_valid_toks"] += 1
    raw = json.dumps(on_summary, separators=(", ", ": ")).encode("ascii")
    on_path.write_bytes(raw)
    kwargs["expected_summary_sha256"]["on"] = hashlib.sha256(raw).hexdigest()

    diagnostic = COLLECTOR.collect_diagnostic_local_summary_pair(**kwargs)

    timing = diagnostic["latest_comparison"]["policy_training_seconds"]
    assert timing == {
        "off": 3.7598189060081495,
        "on": 5.811881213972811,
        "same_work": False,
        "off_over_on": None,
    }


def _reseal_diagnostic(diagnostic: dict[str, Any]) -> None:
    diagnostic["latest_comparison"] = COLLECTOR._diagnostic_latest_comparison(diagnostic["arms"])
    payload = {key: item for key, item in diagnostic.items() if key != "diagnostic_receipt"}
    diagnostic["diagnostic_receipt"] = COLLECTOR._diagnostic_pair_summary_receipt(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_root", "fields differ"),
        ("acceptance_alias", "wrong JSON type"),
        ("integer_as_float", "exact non-negative JSON integer"),
        ("reward_as_int", "finite JSON float"),
        ("negative_count", "exact non-negative JSON integer"),
        ("oversized_count", "exact non-negative JSON integer"),
        ("zero_total_tokens", "must be positive"),
        ("reward_outside_range", "outside its min/max"),
        ("tampered_comparison", "exact expected value"),
        ("tampered_receipt", "exact expected value"),
    ],
)
def test_diagnostic_serializer_has_closed_types_semantics_and_receipt(
    tmp_path: Path, mutation: str, message: str
) -> None:
    diagnostic, _ = _diagnostic_fixture(tmp_path)
    if mutation == "extra_root":
        diagnostic["unexpected"] = None
    elif mutation == "acceptance_alias":
        diagnostic["capture"]["acceptance_eligible"] = 0
    elif mutation == "integer_as_float":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/total_num_tokens"] = 3572.0
    elif mutation == "reward_as_int":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/raw_environment_reward"] = 0
    elif mutation == "negative_count":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/global_valid_toks"] = -1
        _reseal_diagnostic(diagnostic)
    elif mutation == "oversized_count":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/global_valid_toks"] = 2**54
        _reseal_diagnostic(diagnostic)
    elif mutation == "zero_total_tokens":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/total_num_tokens"] = 0
        _reseal_diagnostic(diagnostic)
    elif mutation == "reward_outside_range":
        diagnostic["arms"]["off"]["latest"]["metrics"]["train/raw_environment_reward"] = 2.0
        _reseal_diagnostic(diagnostic)
    elif mutation == "tampered_comparison":
        diagnostic["latest_comparison"]["policy_training_seconds"]["off_over_on"] = 123.0
    else:
        diagnostic["diagnostic_receipt"]["canonical_sha256"] = _digest("wrong")

    with pytest.raises(COLLECTOR.CollectionError, match=message):
        COLLECTOR.serialize_diagnostic_pair_summary(diagnostic)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_digest", "trusted SHA-256"),
        ("same_run_id", "run IDs must differ"),
        ("wrong_arm_root", "outside the declared arm root"),
        ("mismatched_step", "do not share one latest _step"),
        ("boolean_reward", "must be numeric"),
        ("fractional_integer", "exact integer"),
        ("missing_metric", "lacks diagnostic metric"),
    ],
)
def test_diagnostic_collection_rejects_unbound_or_invalid_sources(tmp_path: Path, mutation: str, message: str) -> None:
    _, kwargs = _diagnostic_fixture(tmp_path)
    if mutation == "wrong_digest":
        kwargs["expected_summary_sha256"]["off"] = _digest("wrong")
    elif mutation == "same_run_id":
        kwargs["run_ids"]["on"] = kwargs["run_ids"]["off"]
    elif mutation == "wrong_arm_root":
        kwargs["summary_paths"]["off"] = kwargs["summary_paths"]["on"]
    else:
        on_path = kwargs["summary_paths"]["on"]
        value = _canary_local_summary("on")
        if mutation == "mismatched_step":
            value["_step"] = 1
        elif mutation == "boolean_reward":
            value["train/raw_environment_reward"] = False
        elif mutation == "fractional_integer":
            value["train/global_valid_toks"] = 3.5
        else:
            del value["timing/train/policy_training"]
        raw = json.dumps(value, separators=(", ", ": ")).encode("ascii")
        on_path.write_bytes(raw)
        kwargs["expected_summary_sha256"]["on"] = hashlib.sha256(raw).hexdigest()

    with pytest.raises(COLLECTOR.CollectionError, match=message):
        COLLECTOR.collect_diagnostic_local_summary_pair(**kwargs)


def test_diagnostic_authentication_reparses_raw_and_rehashes_document() -> None:
    value = _canary_local_summary("off")
    document = _summary_document(value)
    forged_value = copy.deepcopy(value)
    forged_value["_step"] = 1
    forged = COLLECTOR.Document(
        value=forged_value,
        raw=document.raw,
        sha256=document.sha256,
    )
    with pytest.raises(COLLECTOR.CollectionError, match="parsed value"):
        COLLECTOR.authenticate_local_summary(forged, document.sha256, label="forged summary")

    forged_digest = COLLECTOR.Document(
        value=value,
        raw=document.raw,
        sha256=_digest("forged-document-digest"),
    )
    with pytest.raises(COLLECTOR.CollectionError, match="exact bytes"):
        COLLECTOR.authenticate_local_summary(forged_digest, document.sha256, label="forged summary")


def test_diagnostic_collection_rejects_symlinked_summary_path(tmp_path: Path) -> None:
    _, kwargs = _diagnostic_fixture(tmp_path)
    off_path = kwargs["summary_paths"]["off"]
    raw = off_path.read_bytes()
    target = kwargs["pair_root"] / "off" / "actual-summary.json"
    target.write_bytes(raw)
    off_path.unlink()
    off_path.symlink_to(target)

    with pytest.raises(COLLECTOR.CollectionError, match="cannot contain symlinks"):
        COLLECTOR.collect_diagnostic_local_summary_pair(**kwargs)


def test_diagnostic_source_key_inventory_and_paths_are_closed(tmp_path: Path) -> None:
    diagnostic, _ = _diagnostic_fixture(tmp_path)
    cases = []

    wrong_count = copy.deepcopy(diagnostic)
    wrong_count["arms"]["off"]["source"]["source_key_count"] += 1
    cases.append((wrong_count, "key count"))

    duplicate_key = copy.deepcopy(diagnostic)
    duplicate_key["arms"]["off"]["source"]["source_keys"].append(
        duplicate_key["arms"]["off"]["source"]["source_keys"][-1]
    )
    cases.append((duplicate_key, "unique sorted strings"))

    wrong_keyset_digest = copy.deepcopy(diagnostic)
    wrong_keyset_digest["arms"]["off"]["source"]["source_keyset_sha256"] = _digest("wrong-keyset")
    cases.append((wrong_keyset_digest, "source-keyset SHA-256"))

    missing_selected_key = copy.deepcopy(diagnostic)
    source = missing_selected_key["arms"]["off"]["source"]
    source["source_keys"].remove("train/raw_environment_reward")
    source["source_key_count"] = len(source["source_keys"])
    source["source_keyset_sha256"] = COLLECTOR.canonical_json_sha256(
        source["source_keys"], "missing selected source key"
    )
    cases.append((missing_selected_key, "lacks selected metrics"))

    noncanonical_path = copy.deepcopy(diagnostic)
    pair_root = Path(noncanonical_path["pair"]["root"])
    source_path = Path(noncanonical_path["arms"]["off"]["source"]["path"])
    relative = source_path.relative_to(pair_root / "off")
    noncanonical_path["arms"]["off"]["source"]["path"] = (pair_root / "off" / ".." / "off" / relative).as_posix()
    cases.append((noncanonical_path, "canonical absolute path"))

    for value, message in cases:
        with pytest.raises(COLLECTOR.CollectionError, match=message):
            COLLECTOR.serialize_diagnostic_pair_summary(value)


def test_diagnostic_environment_type_is_fail_closed(tmp_path: Path) -> None:
    _, kwargs = _diagnostic_fixture(tmp_path)
    kwargs["environment"] = []
    with pytest.raises(COLLECTOR.CollectionError, match="safe identifier"):
        COLLECTOR.collect_diagnostic_local_summary_pair(**kwargs)


def test_diagnostic_cli_never_imports_wandb_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, kwargs = _diagnostic_fixture(tmp_path)
    output = tmp_path / "diagnostic-pair-summary.json"

    class ForbiddenWandb:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"diagnostic mode unexpectedly touched W&B: {name}")

    monkeypatch.setitem(sys.modules, "wandb", ForbiddenWandb())
    monkeypatch.setattr(COLLECTOR.time, "time_ns", lambda: 123456789)
    argv = [
        "diagnostic-local-summary-pair",
        "--pair-label",
        kwargs["pair_label"],
        "--pair-root",
        str(kwargs["pair_root"]),
        "--environment",
        "reasoning_gym",
        "--off-run-id",
        kwargs["run_ids"]["off"],
        "--on-run-id",
        kwargs["run_ids"]["on"],
        "--off-summary",
        str(kwargs["summary_paths"]["off"]),
        "--on-summary",
        str(kwargs["summary_paths"]["on"]),
        "--expected-off-summary-sha256",
        kwargs["expected_summary_sha256"]["off"],
        "--expected-on-summary-sha256",
        kwargs["expected_summary_sha256"]["on"],
        "--output",
        str(output),
    ]
    assert COLLECTOR.main(argv) == 0
    raw = output.read_bytes()
    value = json.loads(raw)
    assert value["capture"]["acceptance_eligible"] is False
    assert value["capture"]["history_complete"] is False
    assert value["capture"]["collected_at_unix_ns"] == 123456789
    assert "acceptance_eligible=false" in capsys.readouterr().out

    assert COLLECTOR.main(argv) == 2
    assert output.read_bytes() == raw
    assert "cannot publish diagnostic" in capsys.readouterr().err


def test_diagnostic_wandb_history_emits_two_realistic_non_acceptance_steps() -> None:
    diagnostic, kwargs, api = _diagnostic_history_fixture()

    assert set(diagnostic) == {
        "schema",
        "assurance",
        "pair",
        "capture",
        "arms",
        "coverage",
        "paired_history",
        "acceptance_boundary",
        "diagnostic_receipt",
    }
    assert diagnostic["schema"] == COLLECTOR.DIAGNOSTIC_PAIR_HISTORY_SCHEMA
    assert diagnostic["assurance"] == COLLECTOR.DIAGNOSTIC_PAIR_HISTORY_ASSURANCE
    assert diagnostic["capture"]["acceptance_eligible"] is False
    assert diagnostic["capture"]["wandb_api_viewer_resolved"] is True
    assert diagnostic["capture"]["scan_iteration_completed"] is True
    assert diagnostic["capture"]["history_completeness_status"] == "not_proven"
    assert diagnostic["capture"]["authoritative_progress_field"] == "_step"
    assert diagnostic["capture"]["ignored_progress_fields"] == ["rollout/train_steps"]
    assert diagnostic["capture"]["step_zero_policy"] == ("scanned_excluded_non_training")
    assert diagnostic["capture"]["identity_binding"] == ("caller_supplied_wandb_path")
    assert diagnostic["capture"]["scan_query"] == {
        "keys": None,
        "min_step_inclusive": 0,
        "max_step_exclusive": 101,
        "page_size": 1000,
        "use_cache": False,
    }
    assert diagnostic["capture"]["summary_fallback_used"] is False
    assert diagnostic["capture"]["selected_metrics"] == [
        "_step",
        *COLLECTOR.DIAGNOSTIC_LOCAL_SUMMARY_METRICS,
    ]
    assert api.paths == [
        "nvidia/nano35-rlvr-convergence/t4o5v4hn",
        "nvidia/nano35-rlvr-convergence/8mfmyvrf",
    ]
    for arm in ("off", "on"):
        run = api.runs[kwargs["run_ids"][arm]]
        assert run.scan_calls[-1] == {
            "page_size": 1000,
            "min_step": 0,
            "max_step": 101,
            "use_cache": False,
        }
        history = diagnostic["arms"][arm]["history"]
        assert [row["_step"] for row in history] == [1, 2]
        assert all(type(row["metrics"]["train/raw_environment_reward"]) is float for row in history)
        receipt = diagnostic["arms"][arm]["history_receipt"]
        assert receipt["scan_row_count"] == 4
        assert receipt["scanned_steps"] == [0, 1, 2]
        assert receipt["training_steps"] == [1, 2]
        assert receipt["selected_step_count"] == 2
        assert receipt["scan_iteration_completed"] is True
        assert receipt["normalized_selected_history_sha256"] == COLLECTOR.canonical_json_sha256(
            history, f"{arm} selected history"
        )

    paired = diagnostic["paired_history"]
    assert diagnostic["coverage"] == {
        "off_training_steps": [1, 2],
        "on_training_steps": [1, 2],
        "paired_steps": [1, 2],
        "off_only_steps": [],
        "on_only_steps": [],
    }
    assert [row["step"] for row in paired] == [1, 2]
    assert paired[0]["raw_environment_reward"] == {
        "off": 0.25,
        "on": 1.0,
        "on_minus_off": 0.75,
    }
    assert paired[0]["policy_training_seconds"] == {
        "off": 43.6836,
        "on": 55.5487,
        "same_work": False,
        "off_over_on": None,
    }
    assert paired[0]["token_counts"]["train/global_valid_toks"] == {
        "off": 3035,
        "on": 2622,
    }
    assert paired[1]["raw_environment_reward"] == {
        "off": 0.0,
        "on": 0.0,
        "on_minus_off": 0.0,
    }
    assert paired[1]["policy_training_seconds"]["same_work"] is True
    assert paired[1]["policy_training_seconds"]["off_over_on"] == pytest.approx(3.7598189060081495 / 5.811881213972811)
    assert diagnostic["acceptance_boundary"] == {
        "eligible": False,
        "observed_paired_steps": [1, 2],
        "reason": "exploratory_identity_lacks_authenticated_pair_scheduler_provenance",
        "required_complete_steps": {"first": 1, "last": 100, "count": 100},
        "reward_claim_scope": "none_diagnostic_live_rollouts_not_parity",
        "training_step_coverage_complete": False,
    }
    raw = COLLECTOR.serialize_diagnostic_history_pair(diagnostic)
    assert raw == COLLECTOR.canonical_json_bytes(diagnostic, "history pair") + b"\n"
    payload = {key: item for key, item in diagnostic.items() if key != "diagnostic_receipt"}
    assert diagnostic["diagnostic_receipt"] == {
        "canonicalization": COLLECTOR.WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": COLLECTOR.canonical_json_sha256(payload, "payload"),
        "off_normalized_selected_history_sha256": diagnostic["arms"]["off"]["history_receipt"][
            "normalized_selected_history_sha256"
        ],
        "on_normalized_selected_history_sha256": diagnostic["arms"]["on"]["history_receipt"][
            "normalized_selected_history_sha256"
        ],
    }


def test_diagnostic_wandb_history_pairs_exact_intersection_and_reports_gaps() -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    on_run = api.runs[kwargs["run_ids"]["on"]]
    on_run.rows = [row for row in on_run.rows if row.get("_step") != 2]

    diagnostic = COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)

    assert diagnostic["coverage"] == {
        "off_training_steps": [1, 2],
        "on_training_steps": [1],
        "paired_steps": [1],
        "off_only_steps": [2],
        "on_only_steps": [],
    }
    assert [row["step"] for row in diagnostic["paired_history"]] == [1]
    assert diagnostic["acceptance_boundary"]["eligible"] is False
    assert diagnostic["acceptance_boundary"]["training_step_coverage_complete"] is False
    COLLECTOR.validate_diagnostic_history_pair(diagnostic)


def test_diagnostic_wandb_history_excludes_selected_metric_at_step_zero() -> None:
    baseline, kwargs, api = _diagnostic_history_fixture()
    off_run = api.runs[kwargs["run_ids"]["off"]]
    off_run.rows[0]["train/raw_environment_reward"] = 999.0

    diagnostic = COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)

    assert diagnostic["arms"]["off"]["history"] == baseline["arms"]["off"]["history"]
    assert diagnostic["arms"]["off"]["history_receipt"] == baseline["arms"]["off"]["history_receipt"]


def test_diagnostic_wandb_history_requires_at_least_one_exact_paired_step() -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    off_run = api.runs[kwargs["run_ids"]["off"]]
    on_run = api.runs[kwargs["run_ids"]["on"]]
    off_run.rows = [row for row in off_run.rows if row.get("_step") != 2]
    on_run.rows = [row for row in on_run.rows if row.get("_step") != 1]

    with pytest.raises(COLLECTOR.CollectionError, match="no paired training steps"):
        COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)


def test_diagnostic_wandb_history_fails_if_iteration_breaks_after_start() -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    run = api.runs[kwargs["run_ids"]["on"]]

    def broken_scan(**call: Any) -> Any:
        run.scan_calls.append(call)

        def rows() -> Any:
            yield copy.deepcopy(run.rows[0])
            raise RuntimeError("remote scan broke")

        return rows()

    run.scan_history = broken_scan  # type: ignore[method-assign]
    with pytest.raises(COLLECTOR.CollectionError, match="history scan failed"):
        COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_metric", "metric fields differ"),
        ("boolean_reward", "must be numeric"),
        ("nan_timing", "must be finite"),
        ("fractional_integer", "exact integer"),
        ("float_step", "invalid _step"),
        ("unselected_bad_step", "invalid _step"),
        ("conflict", "conflicting train/raw_environment_reward"),
        ("no_training", "no training steps"),
    ],
)
def test_diagnostic_wandb_history_fails_closed_on_incomplete_or_poisoned_rows(mutation: str, message: str) -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    on_run = api.runs[kwargs["run_ids"]["on"]]
    if mutation == "missing_metric":
        for row in on_run.rows:
            row.pop("train/global_valid_toks", None)
    elif mutation == "boolean_reward":
        on_run.rows[-1]["train/raw_environment_reward"] = True
    elif mutation == "nan_timing":
        on_run.rows[-1]["timing/train/policy_training"] = float("nan")
    elif mutation == "fractional_integer":
        on_run.rows[-1]["train/global_valid_toks"] = 3.5
    elif mutation == "float_step":
        on_run.rows[-1]["_step"] = 2.0
    elif mutation == "unselected_bad_step":
        on_run.rows.append({"_step": True, "_runtime": 99.0})
    elif mutation == "conflict":
        on_run.rows.append({"_step": 1, "train/raw_environment_reward": 0.0})
    else:
        on_run.rows = [{"_step": 0, "_runtime": 1.0}]

    with pytest.raises(COLLECTOR.CollectionError, match=message):
        COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)


@pytest.mark.parametrize("field", ["state", "group", "name", "path"])
def test_diagnostic_wandb_history_requires_exact_finished_run_identity(
    field: str,
) -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    run = api.runs[kwargs["run_ids"]["off"]]
    if field == "path":
        run.path = ["nvidia", "nano35-rlvr-convergence", "wrong"]
    else:
        setattr(run, field, "running" if field == "state" else "wrong")
    with pytest.raises(COLLECTOR.CollectionError, match="diagnostic off W&B"):
        COLLECTOR.collect_diagnostic_wandb_history_pair(**kwargs)


def test_diagnostic_wandb_history_serializer_rejects_tampering() -> None:
    diagnostic, _, _ = _diagnostic_history_fixture()
    cases = []

    acceptance_alias = copy.deepcopy(diagnostic)
    acceptance_alias["capture"]["acceptance_eligible"] = 0
    cases.append((acceptance_alias, "wrong JSON type"))

    reward_as_integer = copy.deepcopy(diagnostic)
    reward_as_integer["arms"]["off"]["history"][0]["metrics"]["train/raw_environment_reward"] = 0
    cases.append((reward_as_integer, "finite JSON float"))

    missing_scanned_step = copy.deepcopy(diagnostic)
    missing_scanned_step["arms"]["off"]["history_receipt"]["scanned_steps"] = [
        0,
        2,
    ]
    cases.append((missing_scanned_step, "omit selected training steps"))

    oversized_scan_count = copy.deepcopy(diagnostic)
    oversized_scan_count["arms"]["off"]["history_receipt"]["scan_row_count"] = COLLECTOR.MAX_HISTORY_ROWS + 1
    cases.append((oversized_scan_count, "strict row-count limit"))

    paired_tamper = copy.deepcopy(diagnostic)
    paired_tamper["paired_history"][0]["raw_environment_reward"]["on_minus_off"] = 999.0
    cases.append((paired_tamper, "exact expected value"))

    receipt_tamper = copy.deepcopy(diagnostic)
    receipt_tamper["diagnostic_receipt"]["canonical_sha256"] = _digest("wrong")
    cases.append((receipt_tamper, "exact expected value"))

    for value, message in cases:
        with pytest.raises(COLLECTOR.CollectionError, match=message):
            COLLECTOR.serialize_diagnostic_history_pair(value)


def test_diagnostic_wandb_history_cli_uses_authenticated_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, kwargs, api = _diagnostic_history_fixture()
    output = tmp_path / "diagnostic-history-pair.json"
    api_calls = []

    def api_factory(**call: Any) -> FakePairApi:
        api_calls.append(call)
        return api

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=api_factory, __version__=COLLECTOR.WANDB_SDK_VERSION),
    )
    monkeypatch.setenv("WANDB_API_KEY", "a" * 40)
    monkeypatch.setattr(COLLECTOR.time, "time_ns", lambda: 123456789)
    argv = [
        "diagnostic-wandb-history-pair",
        "--pair-label",
        kwargs["pair_label"],
        "--environment",
        kwargs["environment"],
        "--entity",
        kwargs["entity"],
        "--project",
        kwargs["project"],
        "--group",
        kwargs["group"],
        "--off-run-id",
        kwargs["run_ids"]["off"],
        "--on-run-id",
        kwargs["run_ids"]["on"],
        "--off-run-name",
        kwargs["run_names"]["off"],
        "--on-run-name",
        kwargs["run_names"]["on"],
        "--output",
        str(output),
    ]
    assert COLLECTOR.main(argv) == 0
    value = json.loads(output.read_bytes())
    assert value["capture"]["fetched_at_unix_ns"] == 123456789
    assert value["capture"]["acceptance_eligible"] is False
    assert value["acceptance_boundary"]["observed_paired_steps"] == [1, 2]
    assert "acceptance_eligible=false" in capsys.readouterr().out
    assert api_calls == [
        {
            "api_key": "a" * 40,
            "overrides": {"base_url": COLLECTOR.WANDB_API_BASE_URL},
        }
    ]
    original = output.read_bytes()
    assert COLLECTOR.main(argv) == 2
    assert output.read_bytes() == original
    assert "cannot publish diagnostic" in capsys.readouterr().err


def test_diagnostic_history_schema_cannot_be_consumed_as_a_v2_run_export() -> None:
    evaluator = _load_module("strict_pair_evaluator_diagnostic_reject", EVALUATOR_PATH)
    diagnostic, _, _ = _diagnostic_history_fixture()
    raw = COLLECTOR.serialize_diagnostic_history_pair(diagnostic)
    digest = hashlib.sha256(raw).hexdigest()
    fixture = _fixture()
    evaluator_contract = copy.deepcopy(fixture.contract)
    evaluator_contract["verifier_metric"] = COLLECTOR.VERIFIER_METRIC_BY_ENVIRONMENT["reasoning_gym"]

    with pytest.raises(evaluator.EvidenceError, match="run export fields differ"):
        evaluator._validate_run(
            evaluator.Document(value=diagnostic, sha256=digest, raw=raw),
            digest,
            evaluator_contract,
            "off",
            pair_manifest_sha256=fixture.pair_sha256,
            submission_receipt_sha256=fixture.submission_sha256,
            scheduler_job_id="41001",
        )


def test_collector_export_is_accepted_by_offline_evaluator() -> None:
    evaluator = _load_module("strict_pair_evaluator_export_compat", EVALUATOR_PATH)
    fixture = _fixture()
    exported = fixture.collect()
    raw = COLLECTOR.canonical_json_bytes(exported, "collector export") + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    evaluator_contract = copy.deepcopy(fixture.contract)
    evaluator_contract["verifier_metric"] = COLLECTOR.VERIFIER_METRIC_BY_ENVIRONMENT["reasoning_gym"]

    run = evaluator._validate_run(
        evaluator.Document(value=exported, sha256=digest, raw=raw),
        digest,
        evaluator_contract,
        "off",
        pair_manifest_sha256=fixture.pair_sha256,
        submission_receipt_sha256=fixture.submission_sha256,
        scheduler_job_id="41001",
    )

    assert run.arm == "off"
    assert run.identity["run_id"] == fixture.run.id
    assert run.history.number(1, "train/reward") == 0.5


def test_collects_exact_v2_export_and_receipts_every_payload() -> None:
    fixture = _fixture()
    exported = fixture.collect()

    assert set(exported) == {
        "schema",
        "identity",
        "scheduler",
        "capture",
        "provenance",
        "config",
        "history",
        "export_receipt",
    }
    assert exported["schema"] == COLLECTOR.RUN_EXPORT_SCHEMA
    assert exported["identity"] == {
        "pair_id": "strict-spfx-ab",
        "environment": "reasoning_gym",
        "arm": "off",
        "shared_prefix_mode": "observe",
        "entity": "nvidia",
        "project": "nano35-rlvr-convergence",
        "group": "reasoning_gym-strict-spfx-ab",
        "run_id": COLLECTOR.derive_wandb_run_id("reasoning_gym", "strict-spfx-ab", "off"),
        "run_name": "off-reasoning_gym-strict-spfx-ab",
        "state": "finished",
    }
    assert exported["scheduler"] == {
        "job_id": "41001",
        "pair_manifest_sha256": fixture.pair_sha256,
        "submission_receipt_sha256": fixture.submission_sha256,
    }
    assert fixture.api.paths == [f"nvidia/nano35-rlvr-convergence/{exported['identity']['run_id']}"]
    assert fixture.run.scan_calls == [{"page_size": 1000, "min_step": 1, "max_step": 101, "use_cache": False}]
    assert exported["history"] == [
        {"_step": 1, "timing/train/policy_training": 10.0},
        {"_step": 1, "train/reward": 0.5, "train/rollout/samples": 4},
        {"_step": 2, "train/reward": 0.0, "train/total_num_tokens": 400},
    ]
    zero = exported["history"][2]["train/reward"]
    assert math.copysign(1.0, zero) == 1.0
    capture = exported["capture"]
    assert set(capture) == {
        "api_base_url",
        "authenticated",
        "collector_sha256",
        "complete",
        "fetched_at_unix_ns",
        "history_method",
        "requested_metrics",
        "summary_fallback_used",
        "wandb_sdk_version",
    }
    assert capture["requested_metrics"] == sorted(capture["requested_metrics"])
    assert capture["summary_fallback_used"] is False
    assert capture["history_method"] == "scan_history"
    assert capture["wandb_sdk_version"] == COLLECTOR.WANDB_SDK_VERSION

    payload = {key: exported[key] for key in exported if key != "export_receipt"}
    receipt = exported["export_receipt"]
    assert receipt == {
        "canonicalization": COLLECTOR.WANDB_EXPORT_CANONICALIZATION,
        "canonical_sha256": COLLECTOR.canonical_json_sha256(payload, "payload"),
        "config_sha256": COLLECTOR.canonical_json_sha256(exported["config"], "config"),
        "history_row_count": 3,
        "history_sha256": COLLECTOR.canonical_json_sha256(exported["history"], "history"),
        "provenance_sha256": COLLECTOR.canonical_json_sha256(exported["provenance"], "provenance"),
    }
    raw = COLLECTOR.serialize_export(exported)
    assert raw == COLLECTOR.canonical_json_bytes(exported, "export") + b"\n"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert all(byte < 128 for byte in raw)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state", "running"),
        ("name", "wrong-name"),
        ("group", "wrong-group"),
        ("entity", "wrong-entity"),
        ("project", "wrong-project"),
        ("id", _digest("wrong-run-id")),
    ],
)
def test_fetched_run_identity_must_match_pair(field: str, replacement: str) -> None:
    fixture = _fixture()
    setattr(fixture.run, field, replacement)
    with pytest.raises(COLLECTOR.CollectionError, match="fetched W&B"):
        fixture.collect()


def test_fetched_run_path_must_match_pair() -> None:
    fixture = _fixture()
    fixture.run.path = ["nvidia", "nano35-rlvr-convergence", "wrong"]
    with pytest.raises(COLLECTOR.CollectionError, match="run path"):
        fixture.collect()


@pytest.mark.parametrize(
    "mutation",
    [
        "run_id",
        "run_id_derivation",
        "group_template",
        "name_template",
        "receipt_wandb",
    ],
)
def test_pair_and_submission_wandb_identity_is_fail_closed(mutation: str) -> None:
    fixture = _fixture()
    if mutation == "run_id":
        fixture.pair["wandb"]["arms"]["off"]["run_id"] = _digest("wrong")
    elif mutation == "run_id_derivation":
        fixture.pair["wandb"]["run_id_derivation"] = "self-declared"
    elif mutation == "group_template":
        fixture.pair["wandb"]["group"]["template"] = "{pair_id}"
    elif mutation == "name_template":
        fixture.pair["wandb"]["arms"]["off"]["name_template"] = "off-{pair_id}"
    else:
        fixture.submission["wandb"]["arms"]["off"]["name"] = "wrong"
    with pytest.raises(COLLECTOR.CollectionError):
        fixture.collect()


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_sha",
        "duplicate_job_ids",
        "candidate_job_id",
        "authenticated_job_id",
        "authenticated_job_name",
        "not_released",
    ],
)
def test_scheduler_binding_is_fail_closed(mutation: str) -> None:
    fixture = _fixture()
    if mutation == "manifest_sha":
        fixture.submission["pair"]["manifest"]["sha256"] = _digest("wrong")
    elif mutation == "duplicate_job_ids":
        fixture.submission["held_submissions"]["on"]["candidate_job_id"] = "41001"
    elif mutation == "candidate_job_id":
        fixture.submission["held_submissions"]["off"]["candidate_job_id"] = True
    elif mutation == "authenticated_job_id":
        fixture.submission["authenticated_jobs"]["off"][0]["job_id"] = "99999"
    elif mutation == "authenticated_job_name":
        fixture.submission["authenticated_jobs"]["off"][0]["job_name"] = "wrong"
    else:
        fixture.submission["outcome"] = "rolled_back"
    with pytest.raises(COLLECTOR.CollectionError):
        fixture.collect()


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("train/reward", float("nan"), "finite"),
        ("train/reward", float("inf"), "finite"),
        ("train/reward", True, "numeric"),
        ("train/rollout/samples", 3.5, "exact integer"),
        ("train/rollout/samples", 2**54, "exact integer"),
    ],
)
def test_history_rejects_invalid_requested_values(metric: str, value: Any, message: str) -> None:
    fixture = _fixture()
    fixture.run.rows = [{"_step": 1, metric: value}]
    with pytest.raises(COLLECTOR.CollectionError, match=message):
        fixture.collect()


@pytest.mark.parametrize("step", [True, 1.0, 0, 101, -1])
def test_history_rejects_non_exact_or_out_of_range_steps(step: Any) -> None:
    fixture = _fixture()
    fixture.run.rows = [{"_step": step, "train/reward": 0.5}]
    with pytest.raises(COLLECTOR.CollectionError, match="invalid strict _step"):
        fixture.collect()


def test_history_requires_at_least_one_requested_observation() -> None:
    fixture = _fixture()
    fixture.run.rows = [{"_step": 1, "loss": 1.0}]
    with pytest.raises(COLLECTOR.CollectionError, match="no strict acceptance metrics"):
        fixture.collect()


def test_wandb_logged_config_must_match_acceptance_config() -> None:
    fixture = _fixture()
    fixture.run.config["policy"]["megatron_cfg"]["context_parallel_size"] = 1
    with pytest.raises(COLLECTOR.CollectionError, match="context_parallel_size"):
        fixture.collect()


def test_epoch_geometry_must_close_against_observed_wandb_config() -> None:
    fixture = _fixture()
    fixture.contract["configs"]["off"]["steps_per_epoch"] = 4
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
        COLLECTOR.acceptance_contract_payload_sha256(fixture.contract)
    )
    with pytest.raises(COLLECTOR.CollectionError, match="steps_per_epoch"):
        fixture.collect()


def test_acceptance_config_has_an_exact_closed_keyset() -> None:
    fixture = _fixture()
    fixture.contract["configs"]["off"]["undeclared"] = True
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
        COLLECTOR.acceptance_contract_payload_sha256(fixture.contract)
    )
    with pytest.raises(COLLECTOR.CollectionError, match="acceptance config fields"):
        fixture.collect()


def test_acceptance_contract_must_bind_pair_and_collector() -> None:
    fixture = _fixture()
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = _digest(
        "wrong-acceptance-contract-payload"
    )
    with pytest.raises(COLLECTOR.CollectionError, match="canonical contract payload"):
        fixture.collect()

    for field in ("pair_manifest_sha256", "wandb_exporter_sha256"):
        fixture = _fixture()
        fixture.contract["provenance"]["common"][field] = _digest(f"wrong-{field}")
        fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
            COLLECTOR.acceptance_contract_payload_sha256(fixture.contract)
        )
        with pytest.raises(COLLECTOR.CollectionError, match="does not bind"):
            fixture.collect()


def test_api_authentication_is_verified_and_timestamp_is_not_a_truthy_alias() -> None:
    fixture = _fixture()
    kwargs = {
        "pair_manifest": fixture.pair,
        "pair_manifest_sha256": fixture.pair_sha256,
        "submission_receipt": fixture.submission,
        "submission_receipt_sha256": fixture.submission_sha256,
        "acceptance_contract": fixture.contract,
        "arm": "off",
        "api": fixture.api,
        "collector_sha256": fixture.collector_sha256,
    }
    with pytest.raises(COLLECTOR.CollectionError, match="fetched_at_unix_ns"):
        COLLECTOR.collect_run_export(
            **kwargs,
            fetched_at_unix_ns=True,
        )

    fixture = _fixture()
    fixture.api.viewer = None
    with pytest.raises(COLLECTOR.CollectionError, match="authenticate a viewer"):
        fixture.collect()

    for invalid_viewer in (False, "nvidia"):
        fixture = _fixture()
        fixture.api.viewer = invalid_viewer
        with pytest.raises(COLLECTOR.CollectionError, match="authenticate a viewer"):
            fixture.collect()

    fixture = _fixture()
    fixture.api.viewer = type(
        "AuthenticatedViewerWithPersonalDefault",
        (),
        {"entity": "personal-default"},
    )()
    assert fixture.collect()["identity"]["entity"] == "nvidia"

    class BrokenViewerApi:
        @property
        def viewer(self) -> object:
            raise RuntimeError("invalid API credential")

    with pytest.raises(COLLECTOR.CollectionError, match="authenticate a viewer"):
        COLLECTOR.validate_wandb_api_authentication(BrokenViewerApi())


def test_document_loader_rejects_duplicate_nonfinite_and_negative_zero(
    tmp_path: Path,
) -> None:
    invalid = {
        "duplicate": b'{"a":1,"a":2}\n',
        "nan": b'{"a":NaN}\n',
        "overflow": b'{"a":1e999}\n',
        "negative-zero": b'{"a":-0.0}\n',
    }
    for name, raw in invalid.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        with pytest.raises(COLLECTOR.CollectionError):
            COLLECTOR.load_document(path, name)


def test_authentication_requires_oob_hash_and_exact_canonical_lf(
    tmp_path: Path,
) -> None:
    value = {"b": 2, "a": 1}
    canonical = COLLECTOR.canonical_json_bytes(value, "value") + b"\n"
    cases = [
        (b'{"b":2,"a":1}\n', hashlib.sha256(b'{"b":2,"a":1}\n').hexdigest()),
        (canonical[:-1], hashlib.sha256(canonical[:-1]).hexdigest()),
        (canonical + b"\n", hashlib.sha256(canonical + b"\n").hexdigest()),
    ]
    for index, (raw, digest) in enumerate(cases):
        path = tmp_path / f"noncanonical-{index}.json"
        path.write_bytes(raw)
        document = COLLECTOR.load_document(path, "document")
        with pytest.raises(COLLECTOR.CollectionError, match="canonical ASCII"):
            COLLECTOR.authenticate_document(
                document,
                digest,
                label="document",
                canonical_lf=True,
            )

    path = tmp_path / "canonical.json"
    path.write_bytes(canonical)
    document = COLLECTOR.load_document(path, "document")
    with pytest.raises(COLLECTOR.CollectionError, match="trusted document SHA-256"):
        COLLECTOR.authenticate_document(
            document,
            "not-a-digest",
            label="document",
            canonical_lf=True,
        )
    with pytest.raises(COLLECTOR.CollectionError, match="differs"):
        COLLECTOR.authenticate_document(
            document,
            _digest("wrong"),
            label="document",
            canonical_lf=True,
        )

    no_lf_path = tmp_path / "canonical-no-lf.json"
    no_lf_path.write_bytes(canonical[:-1])
    no_lf_document = COLLECTOR.load_document(no_lf_path, "no-LF document")
    assert (
        COLLECTOR.authenticate_document(
            no_lf_document,
            no_lf_document.sha256,
            label="no-LF document",
            canonical_lf=False,
        )
        == value
    )
    with pytest.raises(COLLECTOR.CollectionError, match="without LF"):
        COLLECTOR.authenticate_document(
            document,
            document.sha256,
            label="LF-framed document",
            canonical_lf=False,
        )


def test_create_wandb_api_is_lazy_and_uses_fixed_authenticated_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def api_factory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=api_factory, __version__=COLLECTOR.WANDB_SDK_VERSION),
    )
    api = COLLECTOR.create_wandb_api("a" * 40)
    assert api is not None
    assert calls == [
        {
            "api_key": "a" * 40,
            "overrides": {"base_url": COLLECTOR.WANDB_API_BASE_URL},
        }
    ]
    with pytest.raises(COLLECTOR.CollectionError, match="WANDB_API_KEY"):
        COLLECTOR.create_wandb_api("too short")


def test_cli_authenticates_local_evidence_before_importing_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture()
    pair_path = tmp_path / "PAIR_MANIFEST.json"
    receipt_path = tmp_path / "PAIR_SUBMISSION_RECEIPT.json"
    contract_path = tmp_path / "acceptance.json"
    pair_sha = _write_document(pair_path, fixture.pair, trailing_lf=True)
    receipt_sha = _write_document(receipt_path, fixture.submission, trailing_lf=True)
    contract_sha = _write_document(contract_path, fixture.contract, trailing_lf=False)

    class ForbiddenWandb:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"W&B touched before local authentication: {name}")

    monkeypatch.setitem(sys.modules, "wandb", ForbiddenWandb())
    status = COLLECTOR.main(
        [
            "--pair-manifest",
            str(pair_path),
            "--expected-pair-manifest-sha256",
            _digest("wrong-pair"),
            "--submission-receipt",
            str(receipt_path),
            "--expected-submission-receipt-sha256",
            receipt_sha,
            "--acceptance-contract",
            str(contract_path),
            "--expected-acceptance-contract-sha256",
            contract_sha,
            "--arm",
            "off",
            "--output",
            str(tmp_path / "off.json"),
        ]
    )
    assert pair_sha != _digest("wrong-pair")
    assert status == 2
    assert "trusted SHA-256" in capsys.readouterr().err

    fixture.pair["wandb"]["arms"]["off"]["run_id"] = _digest("semantic-drift")
    pair_sha = _write_document(pair_path, fixture.pair, trailing_lf=True)
    fixture.submission["pair"]["manifest"]["sha256"] = pair_sha
    fixture.contract["provenance"]["common"]["pair_manifest_sha256"] = pair_sha
    receipt_sha = _write_document(receipt_path, fixture.submission, trailing_lf=True)
    contract_sha = _write_document(contract_path, fixture.contract, trailing_lf=False)
    status = COLLECTOR.main(
        [
            "--pair-manifest",
            str(pair_path),
            "--expected-pair-manifest-sha256",
            pair_sha,
            "--submission-receipt",
            str(receipt_path),
            "--expected-submission-receipt-sha256",
            receipt_sha,
            "--acceptance-contract",
            str(contract_path),
            "--expected-acceptance-contract-sha256",
            contract_sha,
            "--arm",
            "off",
            "--output",
            str(tmp_path / "semantic-off.json"),
        ]
    )
    assert status == 2
    assert "run_id differs" in capsys.readouterr().err


def test_cli_fetches_with_fake_api_and_never_overwrites_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture()
    pair_path = tmp_path / "PAIR_MANIFEST.json"
    receipt_path = tmp_path / "PAIR_SUBMISSION_RECEIPT.json"
    contract_path = tmp_path / "acceptance.json"
    output_path = tmp_path / "off-run-export.json"
    pair_sha = _write_document(pair_path, fixture.pair, trailing_lf=True)
    receipt_sha = _write_document(receipt_path, fixture.submission, trailing_lf=True)
    contract_sha = _write_document(contract_path, fixture.contract, trailing_lf=False)
    api_calls: list[dict[str, Any]] = []

    def api_factory(**kwargs: Any) -> FakeApi:
        api_calls.append(kwargs)
        return fixture.api

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=api_factory, __version__=COLLECTOR.WANDB_SDK_VERSION),
    )
    monkeypatch.setenv("WANDB_API_KEY", "a" * 40)
    monkeypatch.setattr(COLLECTOR.time, "time_ns", lambda: 123456789)
    argv = [
        "--pair-manifest",
        str(pair_path),
        "--expected-pair-manifest-sha256",
        pair_sha,
        "--submission-receipt",
        str(receipt_path),
        "--expected-submission-receipt-sha256",
        receipt_sha,
        "--acceptance-contract",
        str(contract_path),
        "--expected-acceptance-contract-sha256",
        contract_sha,
        "--arm",
        "off",
        "--output",
        str(output_path),
    ]
    assert COLLECTOR.main(argv) == 0
    raw = output_path.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw)["capture"]["fetched_at_unix_ns"] == 123456789
    digest = hashlib.sha256(raw).hexdigest()
    assert f"sha256={digest}" in capsys.readouterr().out
    assert api_calls == [
        {
            "api_key": "a" * 40,
            "overrides": {"base_url": COLLECTOR.WANDB_API_BASE_URL},
        }
    ]

    assert COLLECTOR.main(argv) == 2
    assert output_path.read_bytes() == raw
    assert "cannot publish" in capsys.readouterr().err


def test_current_pair79_submission_collects_with_unchanged_collector() -> None:
    pair_test = _load_module("current_pair79_for_collector", PAIR_TEST_PATH)
    run = pair_test._run_pair("--submit", PAIR_ID="strict-spfx-ab")
    assert run.process.returncode == 0, run.process.stderr
    assert run.manifest is not None
    assert run.submission_receipt is not None
    pair = json.loads(run.manifest)
    submission = json.loads(run.submission_receipt)
    pair_sha256 = hashlib.sha256(run.manifest).hexdigest()
    submission_sha256 = hashlib.sha256(run.submission_receipt).hexdigest()

    boundary = pair["slurm_export_boundary"]
    assert boundary["schema"] == "nemo-rl-strict-slurm-export-file-v3"
    assert len(boundary["allowed_names"]) == 79
    assert boundary["allowed_names"] == sorted(boundary["allowed_names"])

    fixture = _fixture("off")
    fixture.pair = pair
    fixture.pair_sha256 = pair_sha256
    fixture.submission = submission
    fixture.submission_sha256 = submission_sha256
    fixture.contract["provenance"]["common"]["pair_manifest_sha256"] = pair_sha256
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
        COLLECTOR.acceptance_contract_payload_sha256(fixture.contract)
    )

    exported = fixture.collect("off")

    assert exported["identity"] == COLLECTOR.validate_pair_wandb_identity(pair, "off")
    assert exported["scheduler"] == {
        "job_id": submission["held_submissions"]["off"]["candidate_job_id"],
        "pair_manifest_sha256": pair_sha256,
        "submission_receipt_sha256": submission_sha256,
    }
    assert exported["capture"]["collector_sha256"] == hashlib.sha256(COLLECTOR_PATH.read_bytes()).hexdigest()
