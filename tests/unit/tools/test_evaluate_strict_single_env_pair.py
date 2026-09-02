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
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = (
    REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/evaluate_strict_single_env_pair.py"
)


def _load_evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "strict_pair_evaluator", EVALUATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _config(pair_id: str, arm: str) -> dict[str, Any]:
    mode = "observe" if arm == "off" else "train"
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
        "shared_prefix_mode": mode,
        "wandb_enabled": True,
        "tensorboard_enabled": False,
        "wandb_entity": "nvidia",
        "wandb_project": "nano35-rlvr-convergence",
        "wandb_group": pair_id,
        "wandb_run_name": f"{arm}-{pair_id}",
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
        "resolved_config_sha256": _digest(f"{arm}:resolved-config"),
        # The exact normalized export may carry additional frozen values.  The
        # evaluator compares the complete object, not only its mandatory core.
        "max_new_tokens": 768,
        "temperature": 1.0,
    }


def _reward_mean(step: int) -> float:
    return (0.25, 0.5, 0.75)[(step - 1) % 3]


def _row(step: int, arm: str) -> dict[str, int | float]:
    reward = _reward_mean(step)
    total_tokens = 400
    prompt_tokens = 200
    valid_tokens = 150
    suffix_tokens = 50
    shareable_tokens = 150
    ideal_work = total_tokens - shareable_tokens
    row: dict[str, int | float] = {
        "_step": step,
        "train/raw_environment_reward": reward,
        "train/raw_environment_reward/min": 0.0,
        "train/raw_environment_reward/max": 1.0,
        "train/pre_penalty_environment_reward": reward,
        "train/pre_penalty_environment_reward/min": 0.0,
        "train/pre_penalty_environment_reward/max": 1.0,
        "train/verifier_reward": reward,
        "train/total_reward/mean": reward,
        "train/total_reward/min": 0.0,
        "train/total_reward/max": 1.0,
        "train/reward": reward,
        "train/reward_processing_delta": 0.0,
        "train/effort_low_sample_count": 0,
        "train/effort_low_sample_rate": 0.0,
        "train/effort_reward_delta": 0.0,
        "train/num_mask_sample_filtered": 0,
        "train/mask_sample_rate": 0.0,
        "train/reasoning_equal_to_final_answer_rate": 0.0,
        "train/empty_final_answer_rate": 0.0,
        "train/unwanted_token_rate": 0.0,
        "train/malformed_think_tag_rate": 0.0,
        "train/invalid_tool_call_rate": 0.0,
        "train/malformed_thinking_rate": 0.0,
        "train/raw_invalid_tool_call_rate": 0.0,
        "train/raw_malformed_thinking_rate": 0.0,
        "train/invalid_and_malformed_rate": 0.0,
        "train/rollout/samples": 4,
        "train/reasoning_gym_simple_agent/score/mean": reward,
        "train/num_valid_samples": 4,
        "train/global_valid_seqs": 4,
        "train/global_valid_toks": valid_tokens,
        "train/total_num_tokens": total_tokens,
        "train/shared_prefix/total_sequences": 4,
        "train/shared_prefix/eligible_sequences": 4,
        "train/shared_prefix/complete_groups": 1,
        "train/shared_prefix/fallback_sequences": 0,
        "train/shared_prefix/runtime_fallback_sequences": 0,
        "train/shared_prefix/total_tokens": total_tokens,
        "train/shared_prefix/prompt_tokens": prompt_tokens,
        "train/shared_prefix/valid_loss_tokens": valid_tokens,
        "train/shared_prefix/non_loss_suffix_tokens": suffix_tokens,
        "train/shared_prefix/shareable_prompt_tokens": shareable_tokens,
        "train/shared_prefix/ideal_shared_token_work": ideal_work,
        "train/shared_prefix/ideal_token_reduction": shareable_tokens / total_tokens,
        "train/shared_prefix/ideal_token_work_speedup": total_tokens / ideal_work,
        "train/advantages/min": -1.0,
        "train/advantages/max": 1.0,
        "train/grad_norm": 1.0,
        "train/mtp/grad_norm": 0.5,
        "train/num_masked_seqs_by_logprob_error": 0,
        "train/dropped_prompt_groups": 0,
        "train/replaced_prompt_groups": 0,
        "train/promoted_prompt_groups": 0,
        "train/evicted_stale_prompt_groups": 0,
        "train/aborted_stale_inflight_groups": 0,
        "rollout/skipped_total": 0,
        "rollout/redispatch_total": 0,
        "rollout/data_retry_total": 0,
        "rollout/data_failures_total": 0,
        "rollout/gym_row_redispatch_total": 0,
        "rollout/infra_drops_total": 0,
        "rollout/max_consecutive_infra_drops": 0,
        "timing/train/policy_training": 10.0 if arm == "off" else 8.0,
    }
    return row


def _sparse_rows(arm: str) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for step in EVALUATOR.STEPS:
        full = _row(step, arm)
        metrics = [(key, value) for key, value in full.items() if key != "_step"]
        one_third = len(metrics) // 3
        for part in (
            metrics[:one_third],
            metrics[one_third : 2 * one_third],
            metrics[2 * one_third :],
        ):
            rows.append({"_step": step, **dict(part)})
        # Repeated identical sparse values are normal in a scan history export.
        rows.append({"_step": step, "train/reward": full["train/reward"]})
    return rows


def _job_exit(
    pair_id: str,
    arm: str,
    common: dict[str, Any],
    source_commits: dict[str, Any],
    source_git_trees: dict[str, Any],
    arm_provenance: dict[str, Any],
) -> dict[str, Any]:
    mode = "observe" if arm == "off" else "train"
    return {
        "schema": EVALUATOR.JOB_RECEIPT_SCHEMA,
        "phase": "EXIT",
        "post_verified": True,
        "driver_exit_code": 0,
        "pair_id": pair_id,
        "arm": arm,
        "runtime_attestation_expected_count": 4,
        "runtime_attestation_actual_count": 4,
        "runtime_attestation_receipts_sha256": {
            f"shared_prefix_determinism.{mode}.rank-{rank}.receipt": _digest(
                f"{arm}-rank-{rank}"
            )
            for rank in range(4)
        },
        "runtime_attestation_aggregate_sha256": _digest(f"{arm}-aggregate"),
        "pre_receipt_sha256": _digest(f"{arm}-PRE"),
        "pair_manifest_sha256": common["pair_manifest_sha256"],
        "fixture_sha256": common["fixture_sha256"],
        "fixture_rows": 5,
        "model_tree_sha256_v1": common["model_tree_sha256"],
        "container_sha256": common["training_container_sha256"],
        "sandbox_container_sha256": common["sandbox_container_sha256"],
        "source_head": source_commits["nemo_rl"],
        "source_tree": source_git_trees["nemo_rl"],
        "config_sha256": common["base_recipe_sha256"],
        "reward_semantics_config_sha256": common["base_recipe_sha256"],
        "reward_semantics_contract_sha256": common["reward_semantics_contract_sha256"],
        "nemo_runnable_manifest_sha256": common["nemo_runnable_manifest_sha256"],
        "bridge_runnable_manifest_sha256": common["bridge_runnable_manifest_sha256"],
        "mcore_runnable_manifest_sha256": common["mcore_runnable_manifest_sha256"],
        "deployment_ready_file_sha256": common["deployment_ready_file_sha256"],
        "snapshot_manifest_sha256": arm_provenance["snapshot_manifest_sha256"],
        "entrypoint_sha256": arm_provenance["entrypoint_sha256"],
        "wrapper_sha256": arm_provenance["wrapper_sha256"],
        "inner_ray_sha256": arm_provenance["inner_ray_sha256"],
        "command_sha256": arm_provenance["command_sha256"],
        "mounts_sha256": arm_provenance["mounts_sha256"],
    }


@dataclass
class Fixture:
    contract: dict[str, Any]
    off: dict[str, Any]
    on: dict[str, Any]
    artifacts: dict[str, Any]

    def evaluate(self) -> dict[str, Any]:
        return EVALUATOR.evaluate_pair(self.contract, self.off, self.on, self.artifacts)


def _pin(document: Any, semantic_pins: dict[str, Any]) -> dict[str, Any]:
    return {"sha256": document.sha256, "semantic_pins": semantic_pins}


def _fixture() -> Fixture:
    pair_id = "reasoning-gym-strict-spfx-ab"
    environment = "reasoning_gym"
    run_ids = {"off": "wandb-off-a1b2c3", "on": "wandb-on-d4e5f6"}
    fixture_sha = _digest("fixture")
    common = {key: _digest(f"common:{key}") for key in EVALUATOR.COMMON_PROVENANCE_KEYS}
    common["fixture_sha256"] = fixture_sha
    source_commits = {key: _commit(key) for key in EVALUATOR.SOURCE_KEYS}
    source_git_trees = {
        key: _commit(f"git-tree:{key}") for key in EVALUATOR.SOURCE_KEYS
    }
    source_trees = {key: _digest(f"tree:{key}") for key in EVALUATOR.SOURCE_KEYS}
    arms = {
        arm: {key: _digest(f"{arm}:{key}") for key in EVALUATOR.ARM_PROVENANCE_KEYS}
        for arm in ("off", "on")
    }
    for arm in ("off", "on"):
        arms[arm]["resolved_config_sha256"] = _digest(f"{arm}:resolved-config")
    topology = {
        "cluster_name": "HSG",
        "slurm_partition": "batch",
        "gpu_model": "H100-SXM",
        "nvidia_driver_version": "580.82.07",
        "allocated_nodes": 1,
        "gpus_per_node": 4,
        "trainer_nodes": 1,
        "trainer_gpus_per_node": 4,
        "generation_nodes": 1,
        "generation_gpus_per_node": 4,
        "tensor_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "mtp_num_layers": 5,
    }
    same_arm = EVALUATOR.document_from_value(
        {
            "scope": "same_arm_fresh_process_reproducibility",
            "status": "PASS",
            "pair_id": pair_id,
            "environment": environment,
            "step": 1,
            "sample_count": 4,
            "fresh_process": True,
            "reproducible": True,
        }
    )
    cross_arm = EVALUATOR.document_from_value(
        {
            "scope": "off_on_step1_row_parity",
            "status": "PASS",
            "pair_id": pair_id,
            "environment": environment,
            "step": 1,
            "sample_count": 4,
            "off_run_id": run_ids["off"],
            "on_run_id": run_ids["on"],
            "ordered_rows_equal": True,
            "ordered_rows_sha256": _digest("ordered-step1-rows"),
            "compared_fields": list(EVALUATOR.CROSS_ARM_PARITY_FIELDS),
        }
    )
    common["same_arm_reproducibility_receipt_sha256"] = same_arm.sha256
    common["cross_arm_step1_parity_receipt_sha256"] = cross_arm.sha256
    common["paired_step1_ledger_sha256"] = _digest("ordered-step1-rows")
    execution = {
        arm: EVALUATOR.document_from_value(
            {
                "schema": EVALUATOR.EXECUTION_MARKER_RECEIPT_SCHEMA,
                "scope": "shared_prefix_physical_execution",
                "marker_semantics": EVALUATOR.EXECUTION_MARKER_SEMANTICS,
                "status": "PASS",
                "pair_id": pair_id,
                "environment": environment,
                "arm": arm,
                "shared_prefix_mode": "observe" if arm == "off" else "train",
                "shared_prefix_runtime_trace_sha256": arms[arm][
                    "shared_prefix_runtime_trace_sha256"
                ],
                "shared_prefix_execution_marker_count": 0 if arm == "off" else 4,
            }
        )
        for arm in ("off", "on")
    }
    for arm in ("off", "on"):
        arms[arm]["runtime_direction_receipt_sha256"] = execution[arm].sha256
    job_exit = {
        arm: EVALUATOR.document_from_value(
            _job_exit(
                pair_id,
                arm,
                common,
                source_commits,
                source_git_trees,
                arms[arm],
            )
        )
        for arm in ("off", "on")
    }
    holdout = EVALUATOR.document_from_value(
        {
            "schema": "nemorl-single-env-reward-liveness-holdout-v1",
            "contract_sha256": common["reward_liveness_contract_sha256"],
            "selection_receipt_sha256": _digest("selection-receipt"),
            "holdout_observation_sha256": _digest("holdout-observations"),
            "environment": environment,
            "selected_fixture_sha256": fixture_sha,
            "frozen_reward_primary_mean_min": 0.1,
            "frozen_reward_tail_mean_min": 0.1,
            "eligible": True,
        }
    )
    receipts = {
        "external_step1_same_arm_reproducibility_receipt": _pin(
            same_arm,
            {
                "scope": "same_arm_fresh_process_reproducibility",
                "status": "PASS",
            },
        ),
        "external_step1_off_on_parity_receipt": _pin(
            cross_arm,
            {"scope": "off_on_step1_row_parity", "status": "PASS"},
        ),
        "shared_prefix_execution_marker_receipts": {
            arm: _pin(
                execution[arm],
                {
                    "schema": EVALUATOR.EXECUTION_MARKER_RECEIPT_SCHEMA,
                    "scope": "shared_prefix_physical_execution",
                    "marker_semantics": EVALUATOR.EXECUTION_MARKER_SEMANTICS,
                    "status": "PASS",
                    "arm": arm,
                },
            )
            for arm in ("off", "on")
        },
        "strict_job_exit_receipts": {
            arm: _pin(
                job_exit[arm],
                {
                    "schema": EVALUATOR.JOB_RECEIPT_SCHEMA,
                    "phase": "EXIT",
                    "arm": arm,
                },
            )
            for arm in ("off", "on")
        },
    }
    contract = {
        "schema": EVALUATOR.CONTRACT_SCHEMA,
        "pair": {
            "pair_id": pair_id,
            "environment": environment,
            "entity": "nvidia",
            "project": "nano35-rlvr-convergence",
            "group": pair_id,
            "run_ids": run_ids,
        },
        "acceptance": copy.deepcopy(EVALUATOR.ACCEPTANCE),
        "provenance": {
            "common": common,
            "source_commits": source_commits,
            "source_git_trees": source_git_trees,
            "source_trees_sha256": source_trees,
            "topology": topology,
            "arms": arms,
        },
        "configs": {
            "off": _config(pair_id, "off"),
            "on": _config(pair_id, "on"),
        },
        "holdout": {
            "receipt_sha256": holdout.sha256,
            "primary_reward_mean_min": 0.1,
            "tail_reward_mean_min": 0.1,
        },
        "receipts": receipts,
        "verifier_metric": "train/reasoning_gym_simple_agent/score/mean",
    }

    def run_export(arm: str) -> dict[str, Any]:
        return {
            "schema": EVALUATOR.RUN_EXPORT_SCHEMA,
            "identity": {
                "pair_id": pair_id,
                "environment": environment,
                "arm": arm,
                "shared_prefix_mode": "observe" if arm == "off" else "train",
                "entity": "nvidia",
                "project": "nano35-rlvr-convergence",
                "group": pair_id,
                "run_id": run_ids[arm],
                "run_name": f"{arm}-{pair_id}",
                "state": "finished",
            },
            "provenance": {
                "common": copy.deepcopy(common),
                "source_commits": copy.deepcopy(source_commits),
                "source_git_trees": copy.deepcopy(source_git_trees),
                "source_trees_sha256": copy.deepcopy(source_trees),
                "topology": copy.deepcopy(topology),
                "arm": copy.deepcopy(arms[arm]),
            },
            "config": copy.deepcopy(contract["configs"][arm]),
            "history": _sparse_rows(arm),
        }

    return Fixture(
        contract=contract,
        off=run_export("off"),
        on=run_export("on"),
        artifacts={
            "holdout": holdout,
            "same_arm": same_arm,
            "cross_arm": cross_arm,
            "off_execution": execution["off"],
            "on_execution": execution["on"],
            "off_job_exit": job_exit["off"],
            "on_job_exit": job_exit["on"],
        },
    )


def _history_metric_rows(run: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    return [row for row in run["history"] if metric in row]


def _set_metric(run: dict[str, Any], metric: str, value: int | float) -> None:
    rows = _history_metric_rows(run, metric)
    assert rows
    for row in rows:
        row[metric] = value


def _set_step_metric(
    run: dict[str, Any], step: int, metric: str, value: int | float
) -> None:
    rows = [row for row in run["history"] if row.get("_step") == step and metric in row]
    assert rows
    for row in rows:
        row[metric] = value


def _replace_common_pin(fixture: Fixture, key: str, value: str) -> None:
    fixture.contract["provenance"]["common"][key] = value
    fixture.off["provenance"]["common"][key] = value
    fixture.on["provenance"]["common"][key] = value


def _replace_arm_pin(fixture: Fixture, arm: str, key: str, value: str) -> None:
    fixture.contract["provenance"]["arms"][arm][key] = value
    run = fixture.off if arm == "off" else fixture.on
    run["provenance"]["arm"][key] = value


def test_sparse_known_good_pair_is_green() -> None:
    report = _fixture().evaluate()

    assert report["overall"]["status"] == "GREEN"
    assert report["overall"]["independent_statuses"] == {
        "reward_correctness": "PASS",
        "learning_behavior": "PASS",
        "speed_evidence": "PASS",
    }
    speed = report["speed_evidence"]["evidence"]
    assert speed["matched_epoch_numbers"] == list(range(3, 21))
    assert speed["bootstrap_95_ci"]["resamples"] == 10_000
    assert speed["bootstrap_95_ci"]["seed"] == EVALUATOR.BOOTSTRAP_SEED
    assert math.isclose(speed["bootstrap_95_ci"]["low"], 1.25)


def test_missing_reward_only_metric_leaves_speed_independently_passed() -> None:
    fixture = _fixture()
    for row in fixture.on["history"]:
        row.pop("train/reward_processing_delta", None)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "UNVERIFIABLE"


def test_conflicting_sparse_reward_metric_is_not_last_value_wins() -> None:
    fixture = _fixture()
    fixture.on["history"].append({"_step": 9, "train/reward_processing_delta": 0.125})

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert "conflicting" in " ".join(report["reward_correctness"]["unavailable"])
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"


def test_observed_reward_stage_violation_is_fail_not_unverifiable() -> None:
    fixture = _fixture()
    _set_step_metric(fixture.on, 8, "train/reward_processing_delta", 0.25)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "FAIL"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "RED"


def test_constant_live_rewards_fail_liveness_but_not_accounting() -> None:
    fixture = _fixture()
    stage_metrics = (
        "train/raw_environment_reward",
        "train/pre_penalty_environment_reward",
        "train/verifier_reward",
        "train/total_reward/mean",
        "train/reward",
        "train/reasoning_gym_simple_agent/score/mean",
    )
    for run in (fixture.off, fixture.on):
        for metric in stage_metrics:
            _set_metric(run, metric, 0.5)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "FAIL"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "RED"


def test_speed_ci_must_be_strictly_above_one() -> None:
    fixture = _fixture()
    _set_metric(fixture.on, "timing/train/policy_training", 10.0)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "FAIL"
    assert report["speed_evidence"]["evidence"]["bootstrap_95_ci"]["low"] == 1.0
    assert report["overall"]["status"] == "RED"


def test_speed_window_does_not_require_warmup_timing() -> None:
    fixture = _fixture()
    for run in (fixture.off, fixture.on):
        for row in run["history"]:
            if row.get("_step") == 1:
                row.pop("timing/train/policy_training", None)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"


def test_same_arm_receipt_cannot_substitute_cross_arm_scope() -> None:
    fixture = _fixture()
    replacement = copy.deepcopy(fixture.artifacts["same_arm"].value)
    replacement["scope"] = "off_on_step1_row_parity"
    document = EVALUATOR.document_from_value(replacement)
    fixture.artifacts["same_arm"] = document
    fixture.contract["receipts"]["external_step1_same_arm_reproducibility_receipt"] = (
        _pin(document, {"scope": "off_on_step1_row_parity", "status": "PASS"})
    )
    _replace_common_pin(
        fixture, "same_arm_reproducibility_receipt_sha256", document.sha256
    )

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert report["speed_evidence"]["status"] == "PASS"


def test_absent_cross_arm_receipt_does_not_erase_other_gate_results() -> None:
    fixture = _fixture()
    del fixture.artifacts["cross_arm"]

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "UNVERIFIABLE"


def test_cli_absent_cross_arm_receipt_preserves_independent_results(
    tmp_path: Path, capsys: Any
) -> None:
    fixture = _fixture()

    def write(name: str, value: dict[str, Any]) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return path

    arguments = [
        "--contract",
        str(write("contract", fixture.contract)),
        "--off-export",
        str(write("off", fixture.off)),
        "--on-export",
        str(write("on", fixture.on)),
        "--holdout-receipt",
        str(write("holdout", fixture.artifacts["holdout"].value)),
        "--same-arm-repro-receipt",
        str(write("same", fixture.artifacts["same_arm"].value)),
        "--off-execution-receipt",
        str(write("off-execution", fixture.artifacts["off_execution"].value)),
        "--on-execution-receipt",
        str(write("on-execution", fixture.artifacts["on_execution"].value)),
        "--off-job-exit-receipt",
        str(write("off-exit", fixture.artifacts["off_job_exit"].value)),
        "--on-job-exit-receipt",
        str(write("on-exit", fixture.artifacts["on_job_exit"].value)),
        "--format",
        "json",
    ]

    assert EVALUATOR.main(arguments) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"


def test_total_valid_toks_alias_cannot_replace_global_valid_toks() -> None:
    fixture = _fixture()
    for row in fixture.on["history"]:
        if "train/global_valid_toks" in row:
            row["train/total_valid_toks"] = row.pop("train/global_valid_toks")

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "UNVERIFIABLE"


def test_on_execution_marker_is_mandatory_for_speed_claim() -> None:
    fixture = _fixture()
    replacement = copy.deepcopy(fixture.artifacts["on_execution"].value)
    replacement["shared_prefix_execution_marker_count"] = 0
    document = EVALUATOR.document_from_value(replacement)
    fixture.artifacts["on_execution"] = document
    fixture.contract["receipts"]["shared_prefix_execution_marker_receipts"]["on"] = (
        _pin(
            document,
            {
                "schema": EVALUATOR.EXECUTION_MARKER_RECEIPT_SCHEMA,
                "scope": "shared_prefix_physical_execution",
                "marker_semantics": EVALUATOR.EXECUTION_MARKER_SEMANTICS,
                "status": "PASS",
                "arm": "on",
            },
        )
    )
    _replace_arm_pin(fixture, "on", "runtime_direction_receipt_sha256", document.sha256)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "UNVERIFIABLE"


def test_execution_marker_must_name_the_production_packed_fused_path() -> None:
    fixture = _fixture()
    replacement = copy.deepcopy(fixture.artifacts["on_execution"].value)
    replacement["marker_semantics"] = "generic_attention_forward"
    document = EVALUATOR.document_from_value(replacement)
    fixture.artifacts["on_execution"] = document
    fixture.contract["receipts"]["shared_prefix_execution_marker_receipts"]["on"] = (
        _pin(
            document,
            {
                "schema": EVALUATOR.EXECUTION_MARKER_RECEIPT_SCHEMA,
                "scope": "shared_prefix_physical_execution",
                "marker_semantics": "generic_attention_forward",
                "status": "PASS",
                "arm": "on",
            },
        )
    )
    _replace_arm_pin(fixture, "on", "runtime_direction_receipt_sha256", document.sha256)

    report = fixture.evaluate()

    assert report["reward_correctness"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "UNVERIFIABLE"
    assert "production_packed_fused_training_path" in " ".join(
        report["speed_evidence"]["unavailable"]
    )


def test_job_exit_requires_pre_hash_and_exact_provenance_bindings() -> None:
    for mutation in ("missing-pre", "wrong-config"):
        fixture = _fixture()
        replacement = copy.deepcopy(fixture.artifacts["on_job_exit"].value)
        if mutation == "missing-pre":
            del replacement["pre_receipt_sha256"]
        else:
            replacement["config_sha256"] = _digest("unbound-config")
            replacement["reward_semantics_config_sha256"] = replacement["config_sha256"]
        document = EVALUATOR.document_from_value(replacement)
        fixture.artifacts["on_job_exit"] = document
        fixture.contract["receipts"]["strict_job_exit_receipts"]["on"] = _pin(
            document,
            {
                "schema": EVALUATOR.JOB_RECEIPT_SCHEMA,
                "phase": "EXIT",
                "arm": "on",
            },
        )

        report = fixture.evaluate()

        assert report["reward_correctness"]["status"] == "PASS"
        assert report["learning_behavior"]["status"] == "PASS"
        assert report["speed_evidence"]["status"] == "UNVERIFIABLE"


def test_missing_full_audit_provenance_pin_invalidates_all_claims() -> None:
    fixture = _fixture()
    del fixture.contract["provenance"]["common"]["prompt_schedule_sha256"]

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert set(report["overall"]["independent_statuses"].values()) == {"UNVERIFIABLE"}


def test_exact_config_or_provenance_drift_invalidates_every_claim() -> None:
    for kind in ("config", "provenance"):
        fixture = _fixture()
        if kind == "config":
            fixture.on["config"]["seed"] = 43
        else:
            fixture.on["provenance"]["common"]["fixture_sha256"] = _digest(
                "different-fixture"
            )

        report = fixture.evaluate()

        assert report["overall"]["status"] == "UNVERIFIABLE"
    assert set(report["overall"]["independent_statuses"].values()) == {"UNVERIFIABLE"}


def test_hidden_off_on_config_asymmetry_invalidates_all_claims() -> None:
    fixture = _fixture()
    fixture.contract["configs"]["on"]["temperature"] = 0.8
    fixture.on["config"]["temperature"] = 0.8

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "differ outside" in " ".join(report["reward_correctness"]["unavailable"])


def test_noninteger_step_and_out_of_range_step_fail_closed() -> None:
    for bad_step in (1.0, True, 101):
        fixture = _fixture()
        fixture.on["history"].append(
            {"_step": bad_step, "train/reward_processing_delta": 0.0}
        )

        report = fixture.evaluate()

        assert report["reward_correctness"]["status"] == "UNVERIFIABLE"
        assert report["speed_evidence"]["status"] == "UNVERIFIABLE"
