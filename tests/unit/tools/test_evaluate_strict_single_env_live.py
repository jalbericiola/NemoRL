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

import base64
import copy
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/evaluate_strict_single_env_live.py"
TERMINAL_COLLECTOR_PATH = EVALUATOR_PATH.with_name("collect_strict_single_env_terminal_jobs.py")


def _load_evaluator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("strict_pair_evaluator", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def _load_terminal_collector() -> ModuleType:
    spec = importlib.util.spec_from_file_location("strict_pair_terminal_collector", TERMINAL_COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TERMINAL_COLLECTOR = _load_terminal_collector()
AUTO_EXPORT_SHA256 = "<fixture-auto-export-sha256>"
FIXTURE_PATH_BY_ENVIRONMENT = {
    "reasoning_gym": Path(__file__).with_name("data") / "reasoning_gym_example.jsonl",
    "citation": Path(__file__).with_name("data") / "citation_example.jsonl",
    "freeform": Path(__file__).with_name("data") / "freeform_example.jsonl",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _config(pair_id: str, arm: str, environment: str = "reasoning_gym") -> dict[str, Any]:
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


def _topology() -> dict[str, Any]:
    return {
        "cluster_name": "HSG",
        "slurm_partition": "batch",
        "gpu_model": "NVIDIA GB200",
        "nvidia_driver_version": "580.126.20",
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


def _execution_environment(pair_id: str, fixture_path: str) -> dict[str, Any]:
    results_root = f"/results/{pair_id}"
    snapshot_parent = f"{results_root}/code_snapshots_strict_pairs/{pair_id}"
    arms: dict[str, Any] = {}
    for arm in ("off", "on"):
        results = f"{results_root}/{arm}"
        persistent_cache = f"/cache/strict-pair/{arm}"
        hf_home = f"/cache/huggingface/{arm}"
        snapshot = f"{snapshot_parent}/{arm}-{pair_id}"
        setup_command = EVALUATOR._expected_setup_command_bytes()
        arms[arm] = {
            "base_log_dir": f"{results}/ray_logs",
            "cache_read": {
                "entry_count": 0,
                "mode": "0700",
                "path": f"{persistent_cache}/cache_read",
                "policy": "empty-at-publication-and-job-entry-no-read",
            },
            "hf_datasets_cache": f"{hf_home}/hub",
            "hf_home": hf_home,
            "hf_hub_cache": f"{hf_home}/hub",
            "persistent_cache": persistent_cache,
            "results_dir": results,
            "scheduler": {
                "batch_working_directory": snapshot,
                "sbatch_chdir_argument": f"--chdir={snapshot}",
                "sbatch_client_cwd": snapshot,
                "slurm_submit_dir": snapshot,
            },
            "setup_command": {
                "byte_count": len(setup_command),
                "sha256": hashlib.sha256(setup_command).hexdigest(),
            },
        }
    return {
        "arm_launcher": {
            "ambient_merge": False,
            "argv_prefix": ["-i"],
            "forbidden_caller_names": [
                "BASE_LOG_DIR",
                "CPUS_PER_WORKER",
                "NEMO_SKILLS_SANDBOX_PORT",
                "RAY_LOG_SYNC_FREQUENCY",
                "SANDBOX_COMMAND",
                "SETUP_COMMAND",
                "SLURM_SUBMIT_DIR",
            ],
        },
        "arms": arms,
        "fixed": {
            "cpus_per_worker": "144",
            "nemo_skills_sandbox_port": "6000",
            "ray_log_sync_frequency": "60",
            "sandbox_command": "/start-with-nginx.sh",
            "train_path": fixture_path,
            "val_path": fixture_path,
        },
        "schema": EVALUATOR.EXECUTION_ENVIRONMENT_SCHEMA,
    }


def _reward_mean(step: int) -> float:
    return (0.25, 0.5, 0.75)[(step - 1) % 3]


def _row(step: int, arm: str, environment: str = "reasoning_gym") -> dict[str, int | float]:
    reward = _reward_mean(step)
    total_tokens = 400
    prompt_tokens = 200
    valid_tokens = 150
    suffix_tokens = 50
    shareable_tokens = 150
    ideal_work = total_tokens - shareable_tokens
    advantage_min, advantage_mean, advantage_max = -1.0, 0.0, 1.0
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
        EVALUATOR.VERIFIER_METRIC_BY_ENVIRONMENT[environment]: reward,
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
        "train/advantages/mean": advantage_mean,
        "train/advantages/min": advantage_min,
        "train/advantages/max": advantage_max,
        "train/grad_norm": 1.0,
        "train/mtp/grad_norm": 0.5,
        "train/update_successful": True,
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


def _sparse_rows(arm: str, environment: str = "reasoning_gym") -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for step in EVALUATOR.STEPS:
        full = _row(step, arm, environment)
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
    return sorted(
        rows,
        key=lambda row: (
            row["_step"],
            EVALUATOR._canonical_json_bytes(row, "fixture history row"),
        ),
    )


def _step1_generation() -> dict[str, Any]:
    return {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }


STEP1_MAIN_GROUP_IDS = {
    "off": "11111111-1111-4111-8111-111111111111",
    "on": "22222222-2222-4222-8222-222222222222",
}


def _uuid4_hex(value: int) -> str:
    characters = list(f"{value:032x}")
    characters[12] = "4"
    characters[16] = "8"
    return "".join(characters)


def _step1_transcript_entry_inputs(environment: str, *, group_id: str) -> list[dict[str, Any]]:
    generation = _step1_generation()
    fixture_row = copy.deepcopy(EVALUATOR.load_fixture_document(FIXTURE_PATH_BY_ENVIRONMENT[environment]).rows[0])
    prompt_ids = list(range(1_000, 1_050))
    rewards = (1.0, 0.3 + (0.7 * 1 / 2), 0.0, 0.0) if environment == "reasoning_gym" else (1.0, 0.0, 0.0, 0.0)
    entries: list[dict[str, Any]] = []
    for sample_index in range(4):
        completion_ids = list(range(2_000 + 100 * sample_index, 2_050 + 100 * sample_index))
        raw_reward = rewards[sample_index]
        generation_seed = EVALUATOR._nemo_gym_request_seed(generation["seed_base"], 0, sample_index)
        agent_run_request = EVALUATOR._expected_transformed_fixture_request(
            fixture_row,
            rollout_index=sample_index,
            generation_seed=generation_seed,
            task_index=EVALUATOR._nemo_gym_task_index_from_group_id(group_id),
        )
        generation_request = copy.deepcopy(agent_run_request["responses_create_params"])
        generation_log_probs = [-0.5] * len(completion_ids)
        if environment == "citation":
            response_text = "The original name was Roosterville [ref:2]" if raw_reward else "Roosterville"
        elif environment == "freeform":
            response_text = (
                '1. "John Backup": person\n2. "Massachusetts": place\n' '3. "Boston": place'
                if raw_reward
                else "John Backup"
            )
        else:
            response_text = (
                "<answer>Zoey is a fool, and Riley is a sage.</answer>"
                if sample_index == 0
                else (
                    "<answer>Zoey fool Zoey sage</answer>"
                    if sample_index == 1
                    else "<answer>Zoey is a sage, and Riley is a fool.</answer>"
                )
            )
        model_response = {
            "background": None,
            "conversation": None,
            "created_at": float(1_788_000_000 + sample_index),
            "error": None,
            "id": f"resp_{_uuid4_hex(100 + sample_index)}",
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": 768,
            "max_tool_calls": None,
            "metadata": copy.deepcopy(generation_request["metadata"]),
            "model": "/models/nemotron",
            "object": "response",
            "output": [
                {
                    "content": [
                        {
                            "annotations": [],
                            "logprobs": None,
                            "text": response_text,
                            "type": "output_text",
                        }
                    ],
                    "generation_log_probs": generation_log_probs,
                    "generation_token_ids": completion_ids,
                    "id": f"msg_{_uuid4_hex(200 + sample_index)}",
                    "prompt_token_ids": prompt_ids,
                    "role": "assistant",
                    "routed_experts": None,
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": "completed",
            "temperature": 1.0,
            "text": None,
            "tool_choice": "auto",
            "tools": [],
            "top_logprobs": None,
            "top_p": 1.0,
            "truncation": None,
            "usage": {
                "input_tokens": len(prompt_ids),
                "input_tokens_details": {"cached_tokens": None},
                "output_tokens": len(completion_ids),
                "output_tokens_details": {"reasoning_tokens": None},
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
            "user": None,
        }
        verifier_response = {
            "responses_create_params": EVALUATOR._expanded_nemo_gym_generation_request(
                generation_request, "fixture compact generation request"
            ),
            "response": copy.deepcopy(model_response),
            "reward": raw_reward,
        }
        if environment == "reasoning_gym":
            verifier_response.update(
                {
                    "task_name": agent_run_request["metadata"]["source_dataset"],
                    "score": raw_reward,
                    "extracted_answer": EVALUATOR._reasoning_gym_extracted_answer(response_text),
                }
            )
        elif environment == "citation":
            verifier = copy.deepcopy(agent_run_request["verifier"])
            expected = verifier["expected_markers"]
            missing = [marker for marker in expected if marker not in response_text]
            expected_set = set(expected)
            spurious = [
                match.group(0)
                for pattern in verifier["patterns"]
                for match in re.finditer(pattern, response_text)
                if match.group(0) not in expected_set
            ]
            verifier_response.update(
                {
                    "verifier": verifier,
                    "match_details": {
                        "expected": expected,
                        "missing": missing,
                        "spurious": spurious,
                        "passed": not missing and not spurious,
                    },
                }
            )
        else:
            verifier = copy.deepcopy(agent_run_request["verifier"])
            patterns = [re.compile(pattern) for pattern in verifier["verify_regex"]]
            matching_lines = sum(
                1 for line in response_text.split("\n") if any(pattern.search(line) for pattern in patterns)
            )
            verifier_response.update(
                {
                    "verifier": verifier,
                    "match_details": {
                        "matching_lines": matching_lines,
                        "min_matches": verifier["verify_min_matches"],
                        "passed": matching_lines >= verifier["verify_min_matches"],
                    },
                }
            )
        derived_verifier_request = copy.deepcopy(agent_run_request)
        derived_verifier_request["responses_create_params"] = copy.deepcopy(
            verifier_response["responses_create_params"]
        )
        derived_verifier_request["response"] = copy.deepcopy(model_response)
        entries.append(
            {
                "sample_index": sample_index,
                "fixture_row_index": 0,
                "rollout_index": sample_index,
                "generation_seed": generation_seed,
                "generation_request": generation_request,
                "model_response": model_response,
                "agent_run_request": agent_run_request,
                "derived_verifier_request": derived_verifier_request,
                "verifier_response": verifier_response,
                "raw_environment_reward": raw_reward,
            }
        )
    return entries


def _capture_server(arm: str) -> dict[str, Any]:
    server = {
        "boot_id_sha256": _digest(f"{arm}:transport-boot"),
        "pid": 6101 if arm == "off" else 6102,
        "start_time_ticks": 91_001 if arm == "off" else 91_002,
        "hostname": f"gb200-{arm}",
    }
    server["server_instance_id"] = EVALUATOR._step1_projection_sha256("model-transport-server-instance", server)
    return server


def _model_transport_evidence(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    entry_inputs: list[dict[str, Any]],
) -> tuple[Any, Any]:
    endpoint = {
        "method": "POST",
        "path": "/v1/chat/completions",
        "request_media_type": "application/json",
        "response_media_type": "application/json",
        "status_code": 200,
        "streaming": False,
    }
    server = _capture_server(arm)
    entries: list[dict[str, Any]] = []
    arrival_order = (2, 0, 3, 1)
    for index, source in enumerate(entry_inputs):
        seed = source["generation_seed"]
        generation_request = source["generation_request"]
        model_response = source["model_response"]
        token_output = model_response["output"][0]
        prompt_ids = token_output["prompt_token_ids"]
        generation_ids = token_output["generation_token_ids"]
        generation_log_probs = token_output["generation_log_probs"]
        request_payload = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "truncate_history_thinking": False,
            },
            "logprobs": True,
            "max_tokens": 768,
            "messages": copy.deepcopy(generation_request["input"]),
            "metadata": {"extra_body": json.dumps({"seed": seed}, sort_keys=True, separators=(",", ":"))},
            "model": "/models/nemotron",
            "return_tokens_as_token_ids": True,
            "seed": seed,
            "temperature": 1.0,
            "top_logprobs": 0,
            "top_p": 1.0,
        }
        response_payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": {
                        "content": [
                            {
                                "bytes": [65],
                                "logprob": logprob,
                                "token": f"token_id:{token_id}",
                                "top_logprobs": [],
                            }
                            for token_id, logprob in zip(generation_ids, generation_log_probs, strict=True)
                        ]
                    },
                    "message": {
                        "annotations": None,
                        "audio": None,
                        "content": EVALUATOR._nemo_gym_reward_facing_text(model_response, "fixture model response"),
                        "function_call": None,
                        "generation_log_probs": copy.deepcopy(generation_log_probs),
                        "generation_token_ids": copy.deepcopy(generation_ids),
                        "prompt_token_ids": copy.deepcopy(prompt_ids),
                        "reasoning": None,
                        "refusal": None,
                        "role": "assistant",
                    },
                    "routed_experts": None,
                    "stop_reason": None,
                    "token_ids": None,
                }
            ],
            "created": 1_788_000_000 + index,
            "id": f"chatcmpl-{index + 1:016x}",
            "kv_transfer_params": None,
            "metrics": None,
            "model": "/models/nemotron",
            "object": "chat.completion",
            "prompt_logprobs": None,
            "prompt_text": None,
            "prompt_token_ids": None,
            "service_tier": None,
            "system_fingerprint": None,
            "usage": {
                "completion_tokens": len(generation_ids),
                "prompt_tokens": len(prompt_ids),
                "prompt_tokens_details": None,
                "total_tokens": len(prompt_ids) + len(generation_ids),
            },
        }
        request_body = json.dumps(request_payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        response_body = json.dumps(response_payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        entry = {
            "schema": EVALUATOR.MODEL_TRANSPORT_CALL_SCHEMA,
            "rollout_index": index,
            "generation_seed": seed,
            "arrival_index": arrival_order[index],
            "request_body_base64": base64.b64encode(request_body).decode("ascii"),
            "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
            "request_payload": request_payload,
            "request_payload_sha256": EVALUATOR._step1_projection_sha256(
                "model-transport-request-payload", request_payload
            ),
            "response_body_base64": base64.b64encode(response_body).decode("ascii"),
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "response_payload": response_payload,
            "response_payload_sha256": EVALUATOR._step1_projection_sha256(
                "model-transport-response-payload", response_payload
            ),
        }
        entry["entry_sha256"] = EVALUATOR._step1_projection_sha256(
            "model-transport-entry",
            {
                "pair_id": pair_id,
                "environment": environment,
                "arm": arm,
                "endpoint": endpoint,
                "capture_server": server,
                "entry": copy.deepcopy(entry),
            },
        )
        entries.append(entry)
    bundle = {
        "schema": EVALUATOR.MODEL_TRANSPORT_BUNDLE_SCHEMA,
        "hash_domain": EVALUATOR.STEP1_HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "endpoint": endpoint,
        "capture_window": copy.deepcopy(_model_transport_policy()["capture_window"]),
        "capture_server": server,
        "entry_count": 4,
        "entries": entries,
        "ordered_entries_sha256": EVALUATOR._step1_projection_sha256("model-transport-ordered-entries", entries),
    }
    return (
        EVALUATOR.document_from_value(bundle),
        EVALUATOR.jsonl_document_from_values(entries),
    )


def _model_transport_manifest(
    *,
    arm: str,
    pair_manifest: Any,
    submission_receipt_sha256: str,
    ledger: Any,
    transcript: Any,
    bundle: Any,
    raw_log: Any,
) -> Any:
    manifest = pair_manifest.value
    results_dir = manifest["execution_environment"]["arms"][arm]["results_dir"]
    return EVALUATOR.document_from_value(
        {
            "schema": EVALUATOR.MODEL_TRANSPORT_MANIFEST_SCHEMA,
            "hash_domain": EVALUATOR.STEP1_HASH_DOMAIN,
            "pair_id": manifest["pair_id"],
            "environment": manifest["selection"]["environment"],
            "arm": arm,
            "pair_manifest_sha256": pair_manifest.sha256,
            "authenticated_job_id": "41001" if arm == "off" else "41002",
            "submission_receipt_sha256": submission_receipt_sha256,
            "capture_server": copy.deepcopy(bundle.value["capture_server"]),
            "main_transcript_bundle": {
                "path": f"{results_dir}/strict_pair_step1_evidence/transcript-bundle.json",
                "schema": EVALUATOR.STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
                "sha256": transcript.sha256,
            },
            "main_ledger": {
                "path": f"{results_dir}/strict_pair_step1_evidence/main-ledger.json",
                "schema": EVALUATOR.MAIN_STEP1_LEDGER_SCHEMA,
                "sha256": ledger.sha256,
            },
            "transport_bundle": {
                "path": f"{results_dir}/strict_model_transport/model-transport-bundle.json",
                "schema": EVALUATOR.MODEL_TRANSPORT_BUNDLE_SCHEMA,
                "sha256": bundle.sha256,
            },
            "transport_capture": {
                "path": f"{results_dir}/strict_model_transport/model-transport.jsonl",
                "record_schema": EVALUATOR.MODEL_TRANSPORT_CALL_SCHEMA,
                "record_count": 4,
                "sha256": raw_log.sha256,
            },
            "model_transport_policy_sha256": manifest["model_transport"]["policy_sha256"],
            "entry_count": 4,
            "ordered_entries_sha256": bundle.value["ordered_entries_sha256"],
        }
    )


def _verifier_request_derivation(pair_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": EVALUATOR.DERIVED_VERIFIER_REQUEST_SCHEMA,
        "assurance": "deterministic-reconstruction-not-wire-capture",
        "algorithm": "pinned-simple-agent-model-dump-v1",
        "gym_gitlink_commit": pair_manifest["source"]["gym"]["gitlink_commit"],
        "gym_tree": pair_manifest["source"]["gym"]["tree"],
        "runtime": {"openai_version": "2.6.1", "pydantic_version": "2.13.4"},
        "sources": copy.deepcopy(EVALUATOR.DERIVED_VERIFIER_REQUEST_SOURCES),
    }


def _step1_transcript_bundle(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    mode: str,
    attempt_id: str | None,
    bindings: dict[str, Any],
    model_transport_bundle: Any,
    model_transport_path: str,
    verifier_request_derivation: dict[str, Any],
    entry_inputs: list[dict[str, Any]] | None = None,
) -> Any:
    fixture_row = copy.deepcopy(EVALUATOR.load_fixture_document(FIXTURE_PATH_BY_ENVIRONMENT[environment]).rows[0])
    group_id = STEP1_MAIN_GROUP_IDS["off" if mode == "captured_replay" else arm]
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(entry_inputs or _step1_transcript_entry_inputs(environment, group_id=group_id)):
        entry = copy.deepcopy(raw)
        transport_entry = model_transport_bundle.value["entries"][index]
        entry.update(
            {
                "model_transport_entry_sha256": transport_entry["entry_sha256"],
                "model_transport_request_body_sha256": transport_entry["request_body_sha256"],
                "model_transport_response_body_sha256": transport_entry["response_body_sha256"],
            }
        )
        for field, domain in (
            ("generation_request", "step1-generation-request"),
            ("model_response", "step1-model-response"),
            ("agent_run_request", "step1-agent-run-request"),
            (
                "derived_verifier_request",
                "step1-derived-verifier-request",
            ),
            ("verifier_response", "step1-verifier-response"),
        ):
            entry[f"{field}_sha256"] = EVALUATOR._step1_projection_sha256(domain, entry[field])
        entry["entry_sha256"] = EVALUATOR._step1_projection_sha256("step1-transcript-entry", entry)
        entries.append(entry)
    value = {
        "schema": EVALUATOR.STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
        "hash_domain": EVALUATOR.STEP1_HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "mode": mode,
        "attempt_id": attempt_id,
        "step": 1,
        "sample_count": 4,
        "generation": _step1_generation(),
        "fixture_row": {"index": 0, "value": fixture_row},
        "verifier_request_derivation": copy.deepcopy(verifier_request_derivation),
        "model_transport_bundle": {
            "path": model_transport_path,
            "schema": EVALUATOR.MODEL_TRANSPORT_BUNDLE_SCHEMA,
            "sha256": model_transport_bundle.sha256,
        },
        "bindings": copy.deepcopy(bindings),
        "entries": entries,
        "entries_sha256": EVALUATOR._step1_projection_sha256("step1-transcript-entries", entries),
    }
    value["fixture_row"]["sha256"] = EVALUATOR._step1_projection_sha256("step1-fixture-row", value["fixture_row"])
    return EVALUATOR.document_from_value(value)


def _step1_rows(group_id: str, transcript: Any) -> list[dict[str, Any]]:
    generation = _step1_generation()
    valid_counts = (37, 38, 37, 38)
    training_rewards = [EVALUATOR._float32(entry["raw_environment_reward"]) for entry in transcript.value["entries"]]
    advantage_values = EVALUATOR._expected_step1_grpo_advantages(training_rewards)
    rows: list[dict[str, Any]] = []
    for sample_index, entry in enumerate(transcript.value["entries"]):
        prompt_ids, token_ids, completion_ids = EVALUATOR._model_response_token_geometry(
            entry["model_response"], f"fixture transcript {sample_index}"
        )
        token_loss_mask = (
            [0.0] * len(prompt_ids)
            + [1.0] * valid_counts[sample_index]
            + [0.0] * (len(completion_ids) - valid_counts[sample_index])
        )
        raw_reward = training_rewards[sample_index]
        row = {
            "sample_index": sample_index,
            "sample_id": f"{group_id}_g{sample_index}",
            "shared_prefix_group_id": group_id,
            "fixture_row_index": 0,
            "rollout_index": sample_index,
            "prompt_sha256": EVALUATOR._step1_projection_sha256("step1-prompt", prompt_ids),
            "request_sha256": entry["generation_request_sha256"],
            "response_sha256": entry["model_response_sha256"],
            "agent_run_request_sha256": entry["agent_run_request_sha256"],
            "derived_verifier_request_sha256": entry["derived_verifier_request_sha256"],
            "verifier_response_sha256": entry["verifier_response_sha256"],
            "generation_seed": entry["generation_seed"],
            "token_ids": token_ids,
            "input_length": len(token_ids),
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "token_loss_mask": token_loss_mask,
            "raw_environment_reward": raw_reward,
            "pre_penalty_environment_reward": raw_reward,
            "penalty_flags": {key: False for key in EVALUATOR.STEP1_PENALTY_FLAG_KEYS},
            "verifier_reward": raw_reward,
            "processed_reward": raw_reward,
            "sample_mask": 1.0,
            "advantages": [advantage_values[sample_index]] * len(token_ids),
            "valid_loss_tokens": valid_counts[sample_index],
            "total_tokens": len(token_ids),
        }
        row["row_sha256"] = EVALUATOR._step1_projection_sha256("step1-row", row)
        rows.append(row)
    return rows


def _step1_derived(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort_fields = (
        "sample_index",
        "fixture_row_index",
        "rollout_index",
        "prompt_sha256",
        "request_sha256",
        "generation_seed",
        "prompt_token_ids",
    )
    output_fields = (
        *cohort_fields,
        "response_sha256",
        "agent_run_request_sha256",
        "derived_verifier_request_sha256",
        "verifier_response_sha256",
        "token_ids",
        "input_length",
        "completion_token_ids",
        "token_loss_mask",
        "valid_loss_tokens",
        "total_tokens",
    )
    reward_fields = (
        "sample_index",
        "raw_environment_reward",
        "pre_penalty_environment_reward",
        "penalty_flags",
        "verifier_reward",
        "processed_reward",
        "sample_mask",
        "advantages",
    )
    project = lambda fields: [{field: row[field] for field in fields} for row in rows]
    totals = {
        "raw_environment_reward_sum": float(sum(row["raw_environment_reward"] for row in rows)),
        "pre_penalty_environment_reward_sum": float(sum(row["pre_penalty_environment_reward"] for row in rows)),
        "verifier_reward_sum": float(sum(row["verifier_reward"] for row in rows)),
        "processed_reward_sum": float(sum(row["processed_reward"] for row in rows)),
        "sample_mask_sum": int(sum(row["sample_mask"] for row in rows)),
        "global_valid_toks": sum(row["valid_loss_tokens"] for row in rows),
        "total_num_tokens": sum(row["total_tokens"] for row in rows),
    }
    return {
        "step_totals": totals,
        "cohort_sha256": EVALUATOR._step1_projection_sha256("step1-cohort", project(cohort_fields)),
        "outputs_sha256": EVALUATOR._step1_projection_sha256("step1-outputs", project(output_fields)),
        "rewards_sha256": EVALUATOR._step1_projection_sha256("step1-rewards", project(reward_fields)),
        "ordered_rows_sha256": EVALUATOR._step1_projection_sha256("step1-ordered-rows", rows),
    }


def _main_step1_ledger(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    run_id: str,
    common: dict[str, Any],
    arm_provenance: dict[str, Any],
    submission_receipt_sha256: str,
    results_dir: str,
    pair_manifest: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    transcript_bindings = {
        "pair_manifest_sha256": common["pair_manifest_sha256"],
        "submission_receipt_sha256": submission_receipt_sha256,
        "job_id": "41001" if arm == "off" else "41002",
        "run_id": run_id,
        "fixture_sha256": common["fixture_sha256"],
        "verifier_source_sha256": common["verifier_source_sha256"],
        "snapshot_manifest_sha256": arm_provenance["snapshot_manifest_sha256"],
        "config_sha256": common["environment_recipe_sha256"],
    }
    group_id = STEP1_MAIN_GROUP_IDS[arm]
    entry_inputs = _step1_transcript_entry_inputs(environment, group_id=group_id)
    transport_bundle, transport_log = _model_transport_evidence(
        pair_id=pair_id,
        environment=environment,
        arm=arm,
        entry_inputs=entry_inputs,
    )
    transcript = _step1_transcript_bundle(
        pair_id=pair_id,
        environment=environment,
        arm=arm,
        mode="observe" if arm == "off" else "train",
        attempt_id=None,
        bindings=transcript_bindings,
        model_transport_bundle=transport_bundle,
        model_transport_path=(f"{results_dir}/strict_model_transport/model-transport-bundle.json"),
        verifier_request_derivation=_verifier_request_derivation(pair_manifest.value),
        entry_inputs=entry_inputs,
    )
    rows = _step1_rows(STEP1_MAIN_GROUP_IDS[arm], transcript)
    value = {
        "schema": EVALUATOR.MAIN_STEP1_LEDGER_SCHEMA,
        "hash_domain": EVALUATOR.STEP1_HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "mode": "observe" if arm == "off" else "train",
        "step": 1,
        "sample_count": 4,
        "update_successful": True,
        "compared_fields": list(EVALUATOR.CROSS_ARM_PARITY_FIELDS),
        "generation": _step1_generation(),
        "bindings": {
            "pair_manifest_sha256": common["pair_manifest_sha256"],
            "submission_receipt_sha256": submission_receipt_sha256,
            "job_id": "41001" if arm == "off" else "41002",
            "run_id": run_id,
            "restart_count": 0,
            "fixture_sha256": common["fixture_sha256"],
            "verifier_source_sha256": common["verifier_source_sha256"],
            "snapshot_manifest_sha256": arm_provenance["snapshot_manifest_sha256"],
            "config_sha256": common["environment_recipe_sha256"],
            "pair_campaign_sha256": common["pair_campaign_sha256"],
            "pair_campaign_reward_and_advantage_sha256": common["pair_campaign_reward_and_advantage_sha256"],
        },
        "transcript_bundle": {
            "path": f"{results_dir}/strict_pair_step1_evidence/transcript-bundle.json",
            "schema": EVALUATOR.STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
            "sha256": transcript.sha256,
        },
        "rows": rows,
        **_step1_derived(rows),
    }
    ledger = EVALUATOR.document_from_value(value)
    transport_manifest = _model_transport_manifest(
        arm=arm,
        pair_manifest=pair_manifest,
        submission_receipt_sha256=submission_receipt_sha256,
        ledger=ledger,
        transcript=transcript,
        bundle=transport_bundle,
        raw_log=transport_log,
    )
    return ledger, transcript, transport_bundle, transport_log, transport_manifest


def _job_exit(
    pair_id: str,
    arm: str,
    common: dict[str, Any],
    source_commits: dict[str, Any],
    source_git_trees: dict[str, Any],
    arm_provenance: dict[str, Any],
    pair_manifest: dict[str, Any],
    submission_receipt: Any,
    topology: dict[str, Any],
    step1_ledger_sha256: str,
    step1_transcript_sha256: str,
    model_transport_bundle: Any,
    model_transport_log: Any,
    model_transport_manifest: Any,
) -> dict[str, Any]:
    mode = "observe" if arm == "off" else "train"
    pair_boundary = pair_manifest["slurm_export_boundary"]
    arm_boundary = pair_boundary["arms"][arm]
    resolved_boundary = {
        "schema": EVALUATOR.SLURM_EXPORT_BOUNDARY_SCHEMA,
        "format": "nul-separated-name-value",
        "allowed_names": list(EVALUATOR.SLURM_EXPORT_ALLOWED_NAMES),
        "ambient_merge": False,
        "get_user_env": False,
        "arm": arm,
        "path": arm_boundary["path"],
        "sha256": arm_boundary["sha256"],
        "job_argv": [
            "--pair-manifest",
            f"{pair_manifest['paths']['results_root']}/PAIR_MANIFEST.json",
            "--pair-manifest-sha256",
            common["pair_manifest_sha256"],
            "--arm",
            arm,
        ],
    }
    resolved_boundary_sha256 = hashlib.sha256(
        json.dumps(resolved_boundary, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    runtime_tools = pair_manifest["runtime_tools"]
    runtime_document = runtime_tools["document"]
    host_tools = runtime_document["host"]
    container_tools = runtime_document["container"]
    container_entry_boundary = pair_manifest["container_entry_boundary"]
    attestation_line = pair_manifest["runtime_attestation"]["lines"][arm]
    attestation_names = [f"shared_prefix_determinism.{mode}.rank-{rank}.receipt" for rank in range(4)]
    job_id = "41001" if arm == "off" else "41002"
    restart_count = 0
    return {
        "schema": EVALUATOR.JOB_RECEIPT_SCHEMA,
        "phase": "EXIT",
        "post_verified": True,
        "driver_exit_code": 0,
        "job_id": job_id,
        "job_account": "nemotron_sw_post",
        "job_name": pair_manifest["scheduler_submission"]["identity"]["job_names"][arm],
        "job_num_nodes": 1,
        "job_partition": "batch",
        "job_qos": "normal",
        "gpus_per_node": 4,
        "restart_count": restart_count,
        "pair_id": pair_id,
        "environment": pair_manifest["selection"]["environment"],
        "execution_environment": copy.deepcopy(pair_manifest["execution_environment"]),
        "deterministic_controls": copy.deepcopy(EVALUATOR.DETERMINISTIC_CONTROLS),
        "arm": arm,
        "runtime_attestation_expected_count": 4,
        "runtime_attestation_actual_count": 4,
        "runtime_attestation_receipts_sha256": {
            name: attestation_line["sha256_ascii_no_newline"] for name in attestation_names
        },
        "runtime_attestation_aggregate_sha256": (
            EVALUATOR._runtime_attestation_aggregate_sha256(attestation_names, attestation_line["text"])
        ),
        "runtime_attestation_marker_sha256": attestation_line["sha256_ascii_no_newline"],
        "runtime_attestation_receipt_dir": (
            f"{pair_manifest['paths']['results_root']}/{arm}/"
            f"{pair_manifest['determinism_receipt_dir']}/{job_id}-{restart_count}"
        ),
        "runtime_attestation_receipt_dir_device": 101,
        "runtime_attestation_receipt_dir_inode": 1001 if arm == "off" else 1002,
        "pre_receipt_sha256": _digest(f"{arm}-PRE"),
        "pair_manifest_sha256": common["pair_manifest_sha256"],
        "fixture_sha256": common["fixture_sha256"],
        "fixture_rows": 5,
        "model_tree_sha256_v1": common["model_tree_sha256"],
        "container_sha256": common["training_container_sha256"],
        "sandbox_container_sha256": common["sandbox_container_sha256"],
        "source_head": source_commits["nemo_rl"],
        "source_tree": source_git_trees["nemo_rl"],
        "config_sha256": common["environment_recipe_sha256"],
        "reward_semantics_config_sha256": common["environment_recipe_sha256"],
        "reward_semantics_contract_sha256": common["reward_semantics_contract_sha256"],
        "selected_config_sha256": common["environment_recipe_sha256"],
        "nemo_runnable_manifest_sha256": common["nemo_runnable_manifest_sha256"],
        "bridge_runnable_manifest_sha256": common["bridge_runnable_manifest_sha256"],
        "mcore_runnable_manifest_sha256": common["mcore_runnable_manifest_sha256"],
        "deployment_ready": common["deployment_ready_sha256"],
        "deployment_ready_sha256": common["deployment_ready_sha256"],
        "deployment_ready_file_sha256": common["deployment_ready_file_sha256"],
        "pair_campaign_sha256": common["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": common["pair_campaign_reward_and_advantage_sha256"],
        "submission_contract_sha256": common["submission_contract_sha256"],
        "submission_contract_path": pair_manifest["scheduler_submission"]["contract"]["path"],
        "submission_nonce": pair_manifest["scheduler_submission"]["nonce"],
        "submission_receipt_path": pair_manifest["scheduler_submission"]["receipt"]["path"],
        "submission_receipt_sha256": submission_receipt.sha256,
        "strict_pair_arm_wrapper_sha256": common["strict_pair_arm_wrapper_sha256"],
        "strict_pair_contract_sha256": common["strict_pair_contract_sha256"],
        "strict_pair_parent_wrapper_sha256": common["strict_pair_parent_wrapper_sha256"],
        "snapshot_manifest_sha256": arm_provenance["snapshot_manifest_sha256"],
        "entrypoint_sha256": arm_provenance["entrypoint_sha256"],
        "wrapper_sha256": arm_provenance["wrapper_sha256"],
        "inner_ray_sha256": arm_provenance["inner_ray_sha256"],
        "command_sha256": arm_provenance["command_sha256"],
        "mounts_sha256": arm_provenance["mounts_sha256"],
        "container_entry_boundary": copy.deepcopy(container_entry_boundary),
        "container_entry_boundary_sha256": hashlib.sha256(
            json.dumps(container_entry_boundary, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "gym_gitlink_commit": pair_manifest["source"]["gym"]["gitlink_commit"],
        "gym_tree": pair_manifest["source"]["gym"]["tree"],
        "hardware": {
            "schema": EVALUATOR.HARDWARE_OBSERVATION_SCHEMA,
            "gpu_model": topology["gpu_model"],
            "driver_version": topology["nvidia_driver_version"],
            "nvidia_smi": copy.deepcopy(host_tools["nvidia_smi"]),
        },
        "selection": copy.deepcopy(pair_manifest["selection"]),
        "source": copy.deepcopy(pair_manifest["source"]),
        "wandb": {
            "entity": pair_manifest["wandb"]["entity"],
            "project": pair_manifest["wandb"]["project"],
            "group": pair_manifest["wandb"]["group"]["value"],
            "name": pair_manifest["wandb"]["arms"][arm]["name"],
            "name_template": pair_manifest["wandb"]["arms"][arm]["name_template"],
            "run_id": pair_manifest["wandb"]["arms"][arm]["run_id"],
            "run_id_derivation": pair_manifest["wandb"]["run_id_derivation"],
            "resume": pair_manifest["wandb"]["resume"],
        },
        "scheduler_client_environment": {
            "ambient_merge": False,
            "SLURM_CONF": copy.deepcopy(EVALUATOR.HSG_SLURM_CONF),
            "propagated_to_inner_ray": True,
        },
        "scheduler_device_environment": {
            "schema": EVALUATOR.SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA,
            "cuda_visible_devices": "0,1,2,3",
            "gpu_device_ordinal": "0,1,2,3",
            "nvidia_visible_devices": "0,1,2,3",
            "rocr_visible_devices": None,
            "ze_affinity_mask": None,
        },
        "step1_evidence": {
            "schema": EVALUATOR.STEP1_EVIDENCE_INDEX_SCHEMA,
            "main_ledger": {
                "path": (
                    f"{pair_manifest['execution_environment']['arms'][arm]['results_dir']}"
                    "/strict_pair_step1_evidence/main-ledger.json"
                ),
                "schema": EVALUATOR.MAIN_STEP1_LEDGER_SCHEMA,
                "sha256": step1_ledger_sha256,
            },
            "transcript_bundle": {
                "path": (
                    f"{pair_manifest['execution_environment']['arms'][arm]['results_dir']}"
                    "/strict_pair_step1_evidence/transcript-bundle.json"
                ),
                "schema": EVALUATOR.STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
                "sha256": step1_transcript_sha256,
            },
            "model_transport": {
                "schema": EVALUATOR.MODEL_TRANSPORT_EVIDENCE_INDEX_SCHEMA,
                "bundle": {
                    "path": (
                        f"{pair_manifest['execution_environment']['arms'][arm]['results_dir']}"
                        "/strict_model_transport/model-transport-bundle.json"
                    ),
                    "schema": EVALUATOR.MODEL_TRANSPORT_BUNDLE_SCHEMA,
                    "sha256": model_transport_bundle.sha256,
                },
                "manifest": {
                    "path": (
                        f"{pair_manifest['execution_environment']['arms'][arm]['results_dir']}"
                        "/strict_model_transport/model-transport-manifest.json"
                    ),
                    "schema": EVALUATOR.MODEL_TRANSPORT_MANIFEST_SCHEMA,
                    "sha256": model_transport_manifest.sha256,
                },
                "raw_log": {
                    "path": (
                        f"{pair_manifest['execution_environment']['arms'][arm]['results_dir']}"
                        "/strict_model_transport/model-transport.jsonl"
                    ),
                    "record_schema": EVALUATOR.MODEL_TRANSPORT_CALL_SCHEMA,
                    "record_count": 4,
                    "sha256": model_transport_log.sha256,
                },
                "ordered_entries_sha256": model_transport_bundle.value["ordered_entries_sha256"],
            },
        },
        "runtime_tool_manifest_path": runtime_tools["manifest"]["path"],
        "runtime_tool_manifest_sha256": runtime_tools["manifest"]["sha256"],
        "runtime_tool_host_python_path": host_tools["python"]["path"],
        "runtime_tool_host_python_sha256": host_tools["python"]["sha256"],
        "runtime_tool_container_python_path": container_tools["python"]["path"],
        "runtime_tool_container_python_sha256": container_tools["python"]["sha256"],
        "runtime_tool_container_uv_path": container_tools["uv"]["path"],
        "runtime_tool_container_uv_sha256": container_tools["uv"]["sha256"],
        "runtime_tool_uv_shim_path": container_tools["uv_shim"]["path"],
        "runtime_tool_uv_shim_sha256": container_tools["uv_shim"]["sha256"],
        "slurm_export_boundary": resolved_boundary,
        "slurm_export_boundary_sha256": resolved_boundary_sha256,
    }


def _submission_receipt(pair_manifest: Any) -> Any:
    manifest = pair_manifest.value
    pair_id = manifest["pair_id"]
    pair_sha256 = pair_manifest.sha256
    scheduler = manifest["scheduler_submission"]
    nonce = scheduler["nonce"]
    host_tools = manifest["runtime_tools"]["document"]["host"]
    candidate_ids = {"off": "41001", "on": "41002"}
    labels = {
        arm: {
            "comment": f"nemo-rl-strict-pair-v1:{arm}:{nonce}:{pair_sha256}",
            "job_id": candidate_ids[arm],
            "job_name": scheduler["identity"]["job_names"][arm],
            "user_id": str(scheduler["identity"]["submitter_euid"]),
        }
        for arm in ("off", "on")
    }

    def state_record(arm: str, phase: str) -> dict[str, Any]:
        if phase in {"pre", "recovery"}:
            job_state = "PENDING"
            reason = "JobHeldUser"
        else:
            job_state = "RUNNING"
            reason = "None"
        return {
            **labels[arm],
            "held": phase in {"pre", "recovery"},
            "job_state": job_state,
            "reason": reason,
            "work_dir": manifest["source"]["snapshots"][arm]["path"],
        }

    def release_query(phase: str) -> dict[str, Any]:
        ordered = [candidate_ids["off"], candidate_ids["on"]]
        return {
            "argv": [
                host_tools["scontrol"]["path"],
                "show",
                "job",
                "--json",
                ",".join(ordered),
            ],
            "authenticated_job_ids": ordered,
            "byte_count": 512,
            "candidate_job_ids": {
                "off": [candidate_ids["off"]],
                "on": [candidate_ids["on"]],
                "unattributed": [],
            },
            "complete": True,
            "line_count": 2,
            "normalization_status": 0,
            "output_sha256_raw": _digest(f"scheduler-{phase}-query"),
            "phase": phase,
            "records": {arm: [state_record(arm, phase)] for arm in ("off", "on")},
            "securely_unlinked": True,
            "status": 0,
            "unterminated_final_line": False,
            "unresolved_job_ids": [],
        }

    held = {}
    for offset, arm in enumerate(("off", "on")):
        job_id = candidate_ids[arm]
        held[arm] = {
            "accepted_id_record": {
                "parsed_job_id": job_id,
                "path": scheduler["accepted_id_records"][arm]["path"],
                "sha256": hashlib.sha256(f"{job_id}\n".encode("ascii")).hexdigest(),
            },
            "candidate_job_id": job_id,
            "candidate_job_id_sha256_ascii_no_newline": hashlib.sha256(job_id.encode("ascii")).hexdigest(),
            "candidate_job_id_source": "accepted-id-record",
            "submission_rpc": {
                "drained_unix_ns": 2_000_000 + offset,
                "relay_status": 0,
                "sbatch_status": 0,
                "started_unix_ns": 1_000_000 + offset,
                "writer_drained": True,
            },
            "wrapper_status": 0,
        }
    value = {
        "acceptance": copy.deepcopy(manifest["acceptance"]),
        "authenticated_jobs": {arm: [labels[arm]] for arm in ("off", "on")},
        "cancellations": [],
        "execution_environment": copy.deepcopy(manifest["execution_environment"]),
        "model_transport": copy.deepcopy(manifest["model_transport"]),
        "held_submissions": held,
        "outcome": "released",
        "pair": {
            "id": pair_id,
            "manifest": {
                "path": f"{manifest['paths']['results_root']}/PAIR_MANIFEST.json",
                "sha256": pair_sha256,
            },
        },
        "post_cancel_queries": [],
        "post_release_query": release_query("post"),
        "pre_cancel_queries": [],
        "pre_release_query": release_query("pre"),
        "receipt": copy.deepcopy(scheduler["receipt"]),
        "recovery_query": None,
        "release": {
            "argv": [
                host_tools["scontrol"]["path"],
                "release",
                f"{candidate_ids['off']},{candidate_ids['on']}",
            ],
            "output_sha256_ascii_no_newline": _digest("scheduler-release-output"),
            "status": 0,
        },
        "rollback_candidates": {
            "off": [candidate_ids["off"]],
            "on": [candidate_ids["on"]],
            "unattributed": [],
        },
        "rollback_confirmed": None,
        "runtime_tools": {
            "manifest": copy.deepcopy(manifest["runtime_tools"]["manifest"]),
            "schema": EVALUATOR.RUNTIME_TOOL_MANIFEST_SCHEMA,
        },
        "scheduler_tools": {
            "client_environment": {
                "ambient_merge": False,
                "env": copy.deepcopy(host_tools["env"]),
                "variables": {
                    "LC_ALL": "C",
                    "SLURM_CONF": copy.deepcopy(EVALUATOR.HSG_SLURM_CONF),
                },
            },
            "sbatch": copy.deepcopy(host_tools["sbatch"]),
            "scancel": copy.deepcopy(host_tools["scancel"]),
            "scontrol": copy.deepcopy(host_tools["scontrol"]),
        },
        "schema": EVALUATOR.SUBMISSION_RECEIPT_SCHEMA,
        "selection": copy.deepcopy(manifest["selection"]),
        "source": {
            "bridge": copy.deepcopy(manifest["source"]["bridge"]),
            "mcore": copy.deepcopy(manifest["source"]["mcore"]),
        },
        "stage": "complete",
        "submission_contract": copy.deepcopy(scheduler["contract"]),
        "submission_nonce": nonce,
        "wandb": copy.deepcopy(manifest["wandb"]),
    }
    return EVALUATOR.document_from_value(value, trailing_lf=True)


def _terminal_pair_receipt(
    pair_document: Any,
    submission_document: Any,
    exit_documents: dict[str, Any],
    collector_sha256: str,
) -> Any:
    terminal_pair_document = TERMINAL_COLLECTOR.Document(
        value=pair_document.value,
        raw=pair_document.raw,
        sha256=pair_document.sha256,
    )
    terminal_submission_document = TERMINAL_COLLECTOR.Document(
        value=submission_document.value,
        raw=submission_document.raw,
        sha256=submission_document.sha256,
    )
    contexts = {
        arm: TERMINAL_COLLECTOR._submission_context(
            terminal_pair_document,
            terminal_submission_document,
            expected_pair_sha256=pair_document.sha256,
            expected_submission_sha256=submission_document.sha256,
            arm=arm,
        )
        for arm in ("off", "on")
    }
    capture_documents = {}
    captures = {}
    for offset, arm in enumerate(("off", "on")):
        context = contexts[arm]
        job_id = int(context["job_id"])
        scheduler_output = {
            "errors": [],
            "jobs": [
                {
                    "comment": context["comment"],
                    "current_working_directory": context["work_dir"],
                    "end_time": {"set": True, "infinite": False, "number": 1_788_000_200 + offset},
                    "exit_code": {
                        "status": ["SUCCESS"],
                        "return_code": {"set": True, "infinite": False, "number": 0},
                        "signal": {
                            "id": {"set": False, "infinite": False, "number": 0},
                            "name": "",
                        },
                    },
                    "derived_exit_code": {
                        "status": ["SUCCESS"],
                        "return_code": {"set": True, "infinite": False, "number": 0},
                        "signal": {
                            "id": {"set": False, "infinite": False, "number": 0},
                            "name": "",
                        },
                    },
                    "hold": False,
                    "job_id": job_id,
                    "job_state": ["COMPLETED"],
                    "name": context["job_name"],
                    "restart_cnt": 0,
                    "start_time": {"set": True, "infinite": False, "number": 1_788_000_000 + offset},
                    "state_reason": "None",
                    "user_id": int(context["user_id"]),
                }
            ],
            "last_backfill": {"set": True, "infinite": False, "number": 1},
            "last_update": {"set": True, "infinite": False, "number": 1_788_000_200 + offset},
            "meta": {
                "plugin": {
                    "type": "",
                    "name": "",
                    "data_parser": "data_parser/v0.0.44",
                    "accounting_storage": "accounting_storage/slurmdbd",
                },
                "client": {"source": "", "user": "jalbericiola", "group": "users"},
                "command": ["show", "job"],
                "slurm": {
                    "version": {"major": "25", "minor": "11", "micro": "6"},
                    "release": "25.11.6",
                    "cluster": "oci-hsg-cs-001",
                },
            },
            "warnings": [],
        }
        raw_stdout = (
            TERMINAL_COLLECTOR.canonical_json_bytes(
                scheduler_output,
                f"{arm} fixture scontrol output",
            )
            + b"\n"
        )
        terminal_record = TERMINAL_COLLECTOR.normalize_scontrol_terminal(raw_stdout, context)
        assert terminal_record is not None
        capture = {
            "schema": TERMINAL_COLLECTOR.ARM_CAPTURE_SCHEMA,
            "capture_method": TERMINAL_COLLECTOR.CAPTURE_METHOD,
            "collector_sha256": collector_sha256,
            "pair_id": context["pair_id"],
            "environment": context["environment"],
            "arm": arm,
            "pair_manifest_sha256": pair_document.sha256,
            "submission_receipt_sha256": submission_document.sha256,
            "submission_contract_sha256": context["submission_contract"]["sha256"],
            "runtime_tool_manifest_sha256": context["runtime_tool_manifest_sha256"],
            "scheduler_tool": copy.deepcopy(context["scontrol"]),
            "slurm_conf": copy.deepcopy(TERMINAL_COLLECTOR.HSG_SLURM_CONF),
            "query": {
                "argv": [context["scontrol"]["path"], "show", "job", "--json", context["job_id"]],
                "return_code": 0,
                "started_at_unix_ns": str(1_788_000_000_000_000_000 + offset),
                "finished_at_unix_ns": str(1_788_000_001_000_000_000 + offset),
                "raw_stdout_base64": base64.b64encode(raw_stdout).decode("ascii"),
                "raw_stdout_sha256": hashlib.sha256(raw_stdout).hexdigest(),
                "raw_stdout_byte_count": len(raw_stdout),
                "raw_stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "raw_stderr_byte_count": 0,
            },
            "terminal_record": terminal_record,
        }
        capture_raw = TERMINAL_COLLECTOR.canonical_json_bytes(capture, f"{arm} terminal capture") + b"\n"
        capture_documents[arm] = TERMINAL_COLLECTOR.Document(
            value=capture,
            raw=capture_raw,
            sha256=hashlib.sha256(capture_raw).hexdigest(),
        )
        captures[arm] = capture
    exit_sha256s = {arm: exit_documents[arm].sha256 for arm in ("off", "on")}
    capture_sha256s = {arm: capture_documents[arm].sha256 for arm in ("off", "on")}
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "job_exit_receipt_sha256s": exit_sha256s,
        "capture_sha256s": capture_sha256s,
    }
    receipt = {
        "schema": TERMINAL_COLLECTOR.PAIR_RECEIPT_SCHEMA,
        "capture_method": TERMINAL_COLLECTOR.CAPTURE_METHOD,
        "collector_sha256": collector_sha256,
        "pair_id": contexts["off"]["pair_id"],
        "environment": contexts["off"]["environment"],
        "pair_manifest_sha256": pair_document.sha256,
        "submission_receipt_sha256": submission_document.sha256,
        "job_exit_receipt_sha256s": exit_sha256s,
        "submission_contract_sha256": contexts["off"]["submission_contract"]["sha256"],
        "runtime_tool_manifest_sha256": contexts["off"]["runtime_tool_manifest_sha256"],
        "capture_sha256s": capture_sha256s,
        "composition_sha256": hashlib.sha256(
            TERMINAL_COLLECTOR.canonical_json_bytes(composition, "fixture terminal Pair composition")
        ).hexdigest(),
        "captures": captures,
    }
    return EVALUATOR.document_from_value(receipt, trailing_lf=True)


@dataclass
class Fixture:
    contract: dict[str, Any]
    off: dict[str, Any]
    on: dict[str, Any]
    artifacts: dict[str, Any]
    expected_submission_receipt_sha256: str | None
    expected_step1_hashes: dict[str, str]
    expected_off_export_sha256: str | None = AUTO_EXPORT_SHA256
    expected_on_export_sha256: str | None = AUTO_EXPORT_SHA256

    def evaluate(self) -> dict[str, Any]:
        contract_payload_sha256 = EVALUATOR._acceptance_contract_payload_sha256(self.contract)
        self.contract["provenance"]["common"]["acceptance_contract_sha256"] = contract_payload_sha256
        for run in (self.off, self.on):
            run["provenance"]["common"]["acceptance_contract_sha256"] = contract_payload_sha256
        off_document = EVALUATOR.wandb_export_document_from_payload(self.off)
        on_document = EVALUATOR.wandb_export_document_from_payload(self.on)
        return EVALUATOR.evaluate_pair(
            self.contract,
            off_document,
            on_document,
            self.artifacts,
            expected_submission_receipt_sha256=(self.expected_submission_receipt_sha256),
            expected_off_export_sha256=(
                off_document.sha256
                if self.expected_off_export_sha256 == AUTO_EXPORT_SHA256
                else self.expected_off_export_sha256
            ),
            expected_on_export_sha256=(
                on_document.sha256
                if self.expected_on_export_sha256 == AUTO_EXPORT_SHA256
                else self.expected_on_export_sha256
            ),
        )


def _pin(document: Any, semantic_pins: dict[str, Any]) -> dict[str, Any]:
    return {"sha256": document.sha256, "semantic_pins": semantic_pins}


def _reseal_terminal_scheduler_fixture(fixture: Fixture, value: dict[str, Any]) -> None:
    capture_sha256s = {}
    for arm in ("off", "on"):
        capture_raw = (
            TERMINAL_COLLECTOR.canonical_json_bytes(
                value["captures"][arm],
                f"{arm} mutated terminal capture",
            )
            + b"\n"
        )
        capture_sha256s[arm] = hashlib.sha256(capture_raw).hexdigest()
    value["capture_sha256s"] = capture_sha256s
    composition = {
        "domain": "nemo-rl-strict-terminal-pair-composition-v1",
        "pair_manifest_sha256": fixture.artifacts["pair_manifest"].sha256,
        "submission_receipt_sha256": fixture.artifacts["submission_receipt"].sha256,
        "job_exit_receipt_sha256s": {arm: fixture.artifacts[f"{arm}_job_exit"].sha256 for arm in ("off", "on")},
        "capture_sha256s": capture_sha256s,
    }
    value["composition_sha256"] = hashlib.sha256(
        TERMINAL_COLLECTOR.canonical_json_bytes(
            composition,
            "mutated terminal Pair composition",
        )
    ).hexdigest()
    document = EVALUATOR.document_from_value(value, trailing_lf=True)
    fixture.artifacts["terminal_scheduler"] = document
    fixture.contract["receipts"]["terminal_scheduler_pair_receipt"] = _pin(
        document,
        {key: copy.deepcopy(item) for key, item in value.items() if key != "captures"},
    )


def _evaluate_export_documents(
    fixture: Fixture,
    off_document: Any,
    on_document: Any,
    *,
    off_sha256: str | None = None,
    on_sha256: str | None = None,
) -> dict[str, Any]:
    return EVALUATOR.evaluate_pair(
        fixture.contract,
        off_document,
        on_document,
        fixture.artifacts,
        expected_submission_receipt_sha256=(fixture.expected_submission_receipt_sha256),
        expected_off_export_sha256=(off_document.sha256 if off_sha256 is None else off_sha256),
        expected_on_export_sha256=(on_document.sha256 if on_sha256 is None else on_sha256),
    )


def _model_transport_policy() -> dict[str, Any]:
    policy = {
        "schema": EVALUATOR.MODEL_TRANSPORT_POLICY_SCHEMA,
        "hash_domain": EVALUATOR.STEP1_HASH_DOMAIN,
        "enabled": True,
        "arms": ["off", "on"],
        "sources": {
            "collector": {
                "path": "nemo_rl/utils/strict_model_transport.py",
                "sha256": _digest("model-transport:collector"),
            },
            "vllm_route": {
                "path": "nemo_rl/models/generation/vllm/vllm_worker_async.py",
                "sha256": _digest("model-transport:vllm-route"),
            },
            "rollout_finalizer": {
                "path": "nemo_rl/experience/rollout_manager.py",
                "sha256": _digest("model-transport:rollout-finalizer"),
            },
        },
        "activation": {
            "config_key": "policy.generation.vllm_cfg.strict_model_transport",
            "main_mode": "capture",
            "replay_mode": "replay",
            "pair_id_environment": "PAIR_ID",
            "environment_environment": "STRICT_PAIR_ENVIRONMENT",
            "arm_environment": {
                "name": "STRICT_PAIR_ARM",
                "off": "off",
                "on": "on",
            },
            "results_dir_environment": "RESULTS_DIR",
        },
        "capture_window": {
            "step": 1,
            "fixture_row_index": 0,
            "sample_count": 4,
            "logical_rollout_indices": [0, 1, 2, 3],
            "seed_base": 42,
            "seed_derivation": "sha256-canonical-ascii-json-int63-v1",
            "concurrency": "arrival-independent",
            "duplicate_or_retry": "reject",
            "seal": "atomic-after-four-successes",
            "main_after_seal": ("reject-until-rollout-finalizer-attests-step1-complete-then-pass-through"),
            "replay_after_seal": "reject-terminal",
        },
        "http": {
            "body_boundary": "http-body-bytes-only",
            "headers": "excluded",
            "cookies": "excluded",
            "authorization": "excluded",
            "query": "forbidden",
            "encoding": "utf-8",
            "request_media_type": "application/json",
            "response_media_type": "application/json",
            "response_status_code": 200,
            "streaming": False,
            "max_request_body_bytes": 16_777_216,
            "max_response_body_bytes": 16_777_216,
            "max_bundle_bytes": 201_326_592,
            "endpoint_allowlist": [{"method": "POST", "path": "/v1/chat/completions", "logical_count": 4}],
            "probe_allowlist": [],
            "tokenize_count": 0,
            "unlisted_during_window": "reject",
            "unlisted_during_replay": "reject",
            "direct_python_generation_during_replay": "reject",
        },
        "artifacts": {
            "directory": {
                "relative_path": "strict_model_transport",
                "mode": "0700",
                "inventory": [
                    "model-transport.jsonl",
                    "model-transport-bundle.json",
                    "model-transport-manifest.json",
                ],
                "precondition": "absent-at-pre-runtime-creates-exclusively",
            },
            "log": {
                "relative_path": "strict_model_transport/model-transport.jsonl",
                "schema": EVALUATOR.MODEL_TRANSPORT_CALL_SCHEMA,
                "framing": "canonical-ascii-json-line-lf",
                "mode": "0400",
                "lines": 4,
            },
            "bundle": {
                "relative_path": "strict_model_transport/model-transport-bundle.json",
                "schema": EVALUATOR.MODEL_TRANSPORT_BUNDLE_SCHEMA,
                "framing": "canonical-ascii-json-no-lf",
                "mode": "0400",
            },
            "manifest": {
                "relative_path": "strict_model_transport/model-transport-manifest.json",
                "schema": EVALUATOR.MODEL_TRANSPORT_MANIFEST_SCHEMA,
                "framing": "canonical-ascii-json-no-lf",
                "mode": "0400",
            },
            "owner": "effective-uid",
            "nlink": 1,
            "publication": "o-excl-fsync-atomic-seal",
        },
    }
    policy["policy_sha256"] = EVALUATOR._step1_projection_sha256("model-transport-policy", policy)
    return policy


def _pair_manifest(
    pair_id: str,
    environment: str,
    source_commits: dict[str, Any],
    source_git_trees: dict[str, Any],
) -> Any:
    host_tools = {
        name: {
            "path": (
                "/bin/bash"
                if name == "bash"
                else (
                    "/usr/bin/nvidia-smi"
                    if name == "nvidia_smi"
                    else (
                        f"/cm/local/apps/slurm/25.11/bin/{name}"
                        if name in {"sbatch", "scancel", "scontrol"}
                        else f"/usr/bin/{name}"
                    )
                )
            ),
            "sha256": (
                EVALUATOR.EXPECTED_HOST_SCHEDULER_TOOLS[name]["sha256"]
                if name in EVALUATOR.EXPECTED_HOST_SCHEDULER_TOOLS
                else _digest(f"runtime-tool:host:{name}")
            ),
        }
        for name in EVALUATOR.HOST_RUNTIME_TOOL_NAMES
    }
    container_tools = {
        "python": {
            "path": ("/root/.local/share/uv/python/" "cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"),
            "sha256": _digest("runtime-tool:container:python"),
        },
        "uv": {
            "path": "/root/.local/bin/uv",
            "sha256": _digest("runtime-tool:container:uv"),
        },
        "uv_shim": {
            "path": "/deployment/runtime/uv",
            "sha256": _digest("runtime-tool:container:uv-shim"),
        },
    }
    runtime_tool_document = {
        "schema": EVALUATOR.RUNTIME_TOOL_MANIFEST_SCHEMA,
        "host": host_tools,
        "container": container_tools,
    }
    runtime_tool_manifest_sha256 = hashlib.sha256(
        EVALUATOR._canonical_json_bytes(runtime_tool_document, "fixture runtime-tool document") + b"\n"
    ).hexdigest()
    campaign = EVALUATOR._expected_pair_campaign(
        {
            "pair": {
                "entity": "nvidia",
                "environment": environment,
                "group": f"{environment}-{pair_id}",
                "project": "nano35-rlvr-convergence",
            },
            "configs": {
                "off": _config(pair_id, "off", environment),
                "on": _config(pair_id, "on", environment),
            },
            "provenance": {"topology": _topology()},
        }
    )
    results_root = f"/results/{pair_id}"
    selection_contract = EVALUATOR.ENVIRONMENT_SELECTIONS[environment]
    fixture_record = {
        "path": f"/fixtures/{environment}.jsonl",
        "rows": 5,
        "sha256": selection_contract["fixture_sha256"],
    }
    selection = {
        "config": copy.deepcopy(selection_contract["config"]),
        "environment": environment,
        "fixture": copy.deepcopy(fixture_record),
        "gym_resources": copy.deepcopy(selection_contract["gym_resources"]),
    }
    entrypoint_sha256 = _digest("common:entrypoint_sha256")
    job_wrapper_sha256 = _digest("common:job_wrapper_sha256")
    submission_nonce = _digest("submission-nonce")
    return EVALUATOR.document_from_value(
        {
            "schema": EVALUATOR.PAIR_MANIFEST_SCHEMA,
            "pair_id": pair_id,
            "arms": {"off": "observe", "on": "train"},
            "acceptance": copy.deepcopy(EVALUATOR.LIVE_LEARNING_ACCEPTANCE_POLICY),
            "campaign": copy.deepcopy(campaign),
            "pair_campaign_sha256": EVALUATOR._canonical_json_sha256(campaign, "fixture Pair campaign"),
            "pair_campaign_reward_and_advantage_sha256": (
                EVALUATOR._canonical_json_sha256(
                    campaign["reward_and_advantage"],
                    "fixture Pair reward-and-advantage policy",
                )
            ),
            "artifacts": {
                "container": {
                    "path": "/containers/training.sqsh",
                    "sha256": _digest("common:training_container_sha256"),
                },
                "fixture": copy.deepcopy(fixture_record),
                "model": {
                    "path": "/models/nemotron",
                    "tree_sha256_v1": _digest("common:model_tree_sha256"),
                },
                "sandbox_container": {
                    "path": "/containers/sandbox.sqsh",
                    "sha256": _digest("common:sandbox_container_sha256"),
                },
            },
            "paths": {
                "cache_root": "/cache/strict-pair",
                "hf_home": "/cache/huggingface",
                "results_root": results_root,
                "snapshot_parent": (f"{results_root}/code_snapshots_strict_pairs/{pair_id}"),
            },
            "deployment": {
                "bridge_runnable_manifest_sha256": _digest("common:bridge_runnable_manifest_sha256"),
                "mcore_runnable_manifest_sha256": _digest("common:mcore_runnable_manifest_sha256"),
                "nemo_runnable_manifest_sha256": _digest("common:nemo_runnable_manifest_sha256"),
                "ready": _digest("common:deployment_ready_sha256"),
                "ready_file_sha256": _digest("common:deployment_ready_file_sha256"),
                "root": "/deployment",
            },
            "container_entry_boundary": copy.deepcopy(EVALUATOR.CONTAINER_ENTRY_BOUNDARY),
            "runtime_tools": {
                "bootstrap_sha256sum": copy.deepcopy(host_tools["sha256sum"]),
                "document": runtime_tool_document,
                "manifest": {
                    "path": "/deployment/strict_pair_runtime_tools.json",
                    "sha256": runtime_tool_manifest_sha256,
                },
            },
            "determinism_receipt_dir": "shared_prefix_determinism_receipts",
            "execution_environment": _execution_environment(pair_id, fixture_record["path"]),
            "model_transport": _model_transport_policy(),
            "runtime_attestation": {
                "expected_count_per_fresh_process_group": 4,
                "lines": {
                    arm: {
                        "mode": mode,
                        "sha256_ascii_no_newline": hashlib.sha256(
                            EVALUATOR._determinism_marker(mode).encode("ascii")
                        ).hexdigest(),
                        "text": EVALUATOR._determinism_marker(mode),
                    }
                    for arm, mode in (("off", "observe"), ("on", "train"))
                },
                "receipt_requires_line_count_and_hash": True,
                "schema": "nemo-rl-shared-prefix-determinism-attestation-v1",
            },
            "scheduler_submission": {
                "accepted_id_records": {
                    arm: {
                        "accepted_format": "ascii-positive-decimal-lf",
                        "capture_format": "opaque-sbatch-stdout",
                        "initial_mode": "0600",
                        "path": (f"{results_root}/strict_pair_submission_state/" f"{pair_id}/{arm}.job-id"),
                        "sealed_mode": "0400",
                    }
                    for arm in ("off", "on")
                },
                "contract": {
                    "path": f"{results_root}/STRICT_PAIR_SUBMISSION_CONTRACT.json",
                    "sha256": _digest("common:submission_contract_sha256"),
                },
                "identity": {
                    "comment_template": ("nemo-rl-strict-pair-v1:{arm}:" "{submission_nonce}:{pair_manifest_sha256}"),
                    "job_names": {
                        "off": f"off-{environment}-{pair_id}",
                        "on": f"on-{environment}-{pair_id}",
                    },
                    "submitter_euid": 1000,
                },
                "nonce": submission_nonce,
                "receipt": {
                    "path": f"{results_root}/PAIR_SUBMISSION_RECEIPT.json",
                    "schema": EVALUATOR.SUBMISSION_RECEIPT_SCHEMA,
                },
                "schema": "nemo-rl-strict-scheduler-submission-v1",
            },
            "selection": selection,
            "wandb": EVALUATOR._expected_pair_wandb({"pair": {"environment": environment, "pair_id": pair_id}}),
            "source": {
                "arm_wrapper_sha256": _digest("common:strict_pair_arm_wrapper_sha256"),
                "bridge": {
                    "head": source_commits["megatron_bridge"],
                    "root": "/deployment/runnable/Megatron-Bridge",
                    "tree": source_git_trees["megatron_bridge"],
                },
                "config_sha256": selection_contract["config"]["sha256"],
                "contract_sha256": _digest("common:strict_pair_contract_sha256"),
                "entrypoint_sha256": entrypoint_sha256,
                "gym": {
                    "gitlink_commit": source_commits["nemo_gym"],
                    "path": ("/deployment/runnable/NemoRL/3rdparty/Gym-workspace/Gym"),
                    "tree": source_git_trees["nemo_gym"],
                },
                "head": source_commits["nemo_rl"],
                "job_wrapper": {
                    "path": "/deployment/strict_pair_job_wrapper.sh",
                    "sha256": job_wrapper_sha256,
                },
                "launcher_sha256": _digest("common:launcher_sha256"),
                "mcore": {
                    "head": source_commits["megatron_lm"],
                    "root": "/deployment/runnable/Megatron-LM",
                    "tree": source_git_trees["megatron_lm"],
                },
                "parent_wrapper_sha256": _digest("common:strict_pair_parent_wrapper_sha256"),
                "root": "/deployment/runnable/NemoRL",
                "snapshots": {
                    arm: {
                        "config_sha256": selection_contract["config"]["sha256"],
                        "entrypoint_sha256": entrypoint_sha256,
                        "manifest_sha256": _digest("shared:snapshot_manifest_sha256"),
                        "path": (f"{results_root}/code_snapshots_strict_pairs/" f"{pair_id}/{arm}-{pair_id}"),
                    }
                    for arm in ("off", "on")
                },
                "tree": source_git_trees["nemo_rl"],
            },
            "slurm_export_boundary": {
                "schema": EVALUATOR.SLURM_EXPORT_BOUNDARY_SCHEMA,
                "format": "nul-separated-name-value",
                "allowed_names": list(EVALUATOR.SLURM_EXPORT_ALLOWED_NAMES),
                "ambient_merge": False,
                "get_user_env": False,
                "arms": {
                    "off": {
                        "path": (f"/results/{pair_id}/strict_pair_slurm_exports/" f"{pair_id}/off.env"),
                        "sha256": _digest("off-slurm-export"),
                    },
                    "on": {
                        "path": (f"/results/{pair_id}/strict_pair_slurm_exports/" f"{pair_id}/on.env"),
                        "sha256": _digest("on-slurm-export"),
                    },
                },
                "job_argv": list(EVALUATOR.SLURM_EXPORT_JOB_ARGV),
            },
        },
        trailing_lf=True,
    )


def _fixture(environment: str = "reasoning_gym") -> Fixture:
    pair_id = "strict-spfx-ab"
    run_ids = {
        arm: hashlib.sha256(f"nemo-rl-strict-wandb-v1:{environment}:{pair_id}:{arm}".encode("ascii")).hexdigest()
        for arm in ("off", "on")
    }
    selection_contract = EVALUATOR.ENVIRONMENT_SELECTIONS[environment]
    fixture_sha = selection_contract["fixture_sha256"]
    source_commits = {key: _commit(key) for key in EVALUATOR.SOURCE_KEYS}
    source_git_trees = {key: _commit(f"git-tree:{key}") for key in EVALUATOR.SOURCE_KEYS}
    pair_manifest = _pair_manifest(pair_id, environment, source_commits, source_git_trees)
    common = {key: _digest(f"common:{key}") for key in EVALUATOR.COMMON_PROVENANCE_KEYS}
    common["pair_manifest_sha256"] = pair_manifest.sha256
    common["fixture_sha256"] = fixture_sha
    common["environment_recipe_sha256"] = selection_contract["config"]["sha256"]
    common["gym_config_sha256"] = selection_contract["gym_resources"]["config"]["sha256"]
    common["verifier_source_sha256"] = selection_contract["gym_resources"]["verifier_source"]["sha256"]
    common["runtime_tool_manifest_sha256"] = pair_manifest.value["runtime_tools"]["manifest"]["sha256"]
    common["terminal_scheduler_collector_sha256"] = hashlib.sha256(TERMINAL_COLLECTOR_PATH.read_bytes()).hexdigest()
    campaign = pair_manifest.value["campaign"]
    common["pair_campaign_sha256"] = EVALUATOR._canonical_json_sha256(campaign, "fixture campaign")
    common["pair_campaign_reward_and_advantage_sha256"] = EVALUATOR._canonical_json_sha256(
        campaign["reward_and_advantage"],
        "fixture reward-and-advantage policy",
    )
    common["reward_semantics_contract_sha256"] = common["pair_campaign_reward_and_advantage_sha256"]
    unverified_lineage = {
        "schema": EVALUATOR.LEGACY_UNVERIFIED_LINEAGE_SCHEMA,
        "assurance": EVALUATOR.UNVERIFIED_LINEAGE_ASSURANCE,
        "common": {key: _digest(f"declared-common:{key}") for key in EVALUATOR.UNVERIFIED_LINEAGE_COMMON_KEYS},
        "arms": {
            arm: {key: _digest(f"declared-{arm}:{key}") for key in EVALUATOR.UNVERIFIED_LINEAGE_ARM_KEYS}
            for arm in ("off", "on")
        },
        "sources": {
            name: {
                "commit": _commit(f"declared:{name}"),
                "git_tree": _commit(f"declared-tree:{name}"),
                "source_tree_sha256": _digest(f"declared-source-tree:{name}"),
            }
            for name in EVALUATOR.UNVERIFIED_LINEAGE_SOURCE_KEYS
        },
    }
    arms = {arm: {key: _digest(f"{arm}:{key}") for key in EVALUATOR.ARM_PROVENANCE_KEYS} for arm in ("off", "on")}
    for arm in ("off", "on"):
        arms[arm]["entrypoint_sha256"] = pair_manifest.value["source"]["entrypoint_sha256"]
        arms[arm]["runtime_environment_sha256"] = pair_manifest.value["slurm_export_boundary"]["arms"][arm]["sha256"]
        arms[arm]["wrapper_sha256"] = pair_manifest.value["source"]["job_wrapper"]["sha256"]
        arms[arm]["inner_ray_sha256"] = _digest("shared:inner-ray")
        arms[arm]["snapshot_manifest_sha256"] = _digest("shared:snapshot_manifest_sha256")
    topology = _topology()
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
                "shared_prefix_runtime_trace_sha256": arms[arm]["shared_prefix_runtime_trace_sha256"],
                "shared_prefix_execution_marker_count": 0 if arm == "off" else 4,
            }
        )
        for arm in ("off", "on")
    }
    for arm in ("off", "on"):
        arms[arm]["runtime_direction_receipt_sha256"] = execution[arm].sha256
    submission_receipt = _submission_receipt(pair_manifest)
    main_ledgers = {arm: EVALUATOR.document_from_value({"arm": arm, "kind": "main-ledger"}) for arm in ("off", "on")}
    main_transcripts = {
        arm: EVALUATOR.document_from_value({"arm": arm, "kind": "transcript-bundle"}) for arm in ("off", "on")
    }
    transport_bundles = {
        arm: EVALUATOR.document_from_value(
            {
                "arm": arm,
                "ordered_entries_sha256": _digest(f"{arm}:model-transport-ordered-entries"),
            }
        )
        for arm in ("off", "on")
    }
    transport_logs = {
        arm: EVALUATOR.document_from_value({"arm": arm, "kind": "model-transport-log"}) for arm in ("off", "on")
    }
    transport_manifests = {
        arm: EVALUATOR.document_from_value({"arm": arm, "kind": "model-transport-manifest"}) for arm in ("off", "on")
    }
    job_exit = {
        arm: EVALUATOR.document_from_value(
            _job_exit(
                pair_id,
                arm,
                common,
                source_commits,
                source_git_trees,
                arms[arm],
                pair_manifest.value,
                submission_receipt,
                topology,
                main_ledgers[arm].sha256,
                main_transcripts[arm].sha256,
                transport_bundles[arm],
                transport_logs[arm],
                transport_manifests[arm],
            ),
            trailing_lf=True,
        )
        for arm in ("off", "on")
    }
    terminal_scheduler = _terminal_pair_receipt(
        pair_manifest,
        submission_receipt,
        job_exit,
        common["terminal_scheduler_collector_sha256"],
    )
    holdout_candidate_sha256s = sorted([_digest(f"holdout-candidate-{index}") for index in range(5)])
    holdout = EVALUATOR.document_from_value(
        {
            "schema": "nemorl-single-env-reward-liveness-holdout-v1",
            "contract_sha256": common["reward_liveness_contract_sha256"],
            "selection_receipt_sha256": _digest("selection-receipt"),
            "holdout_observation_sha256": _digest("holdout-observations"),
            "environment": environment,
            "selected_fixture_sha256": fixture_sha,
            "per_candidate": [
                {
                    "candidate_sha256": candidate_sha256,
                    "environment_successes": 9,
                    "effective_successes": 8,
                    "trials": 16,
                    "mixed_blocks": ["100,101,102,103"],
                }
                for candidate_sha256 in holdout_candidate_sha256s
            ],
            "effective_successes": 40,
            "trials": 80,
            "effective_success_rate": 0.5,
            "effective_reward_wilson95_lower": 0.3929741508611972,
            "frozen_reward_primary_mean_min": 0.19,
            "frozen_reward_tail_mean_min": 0.19,
            "eligible": True,
            "execution_authorized": False,
        },
        trailing_lf=True,
    )
    receipts = {
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
        "terminal_scheduler_pair_receipt": _pin(
            terminal_scheduler,
            {key: copy.deepcopy(value) for key, value in terminal_scheduler.value.items() if key != "captures"},
        ),
    }
    contract = {
        "schema": EVALUATOR.CONTRACT_SCHEMA,
        "pair": {
            "pair_id": pair_id,
            "environment": environment,
            "entity": "nvidia",
            "project": "nano35-rlvr-convergence",
            "group": f"{environment}-{pair_id}",
            "run_ids": run_ids,
        },
        "campaign": copy.deepcopy(campaign),
        "acceptance": copy.deepcopy(EVALUATOR.ACCEPTANCE),
        "provenance": {
            "common": common,
            "source_commits": source_commits,
            "source_git_trees": source_git_trees,
            "trusted_oob_declarations": unverified_lineage,
            "topology": topology,
            "arms": arms,
        },
        "configs": {
            "off": _config(pair_id, "off", environment),
            "on": _config(pair_id, "on", environment),
        },
        "holdout": {
            "receipt_sha256": holdout.sha256,
            "primary_reward_mean_min": 0.19,
            "tail_reward_mean_min": 0.19,
        },
        "receipts": receipts,
        "verifier_metric": EVALUATOR.VERIFIER_METRIC_BY_ENVIRONMENT[environment],
    }
    common["acceptance_contract_sha256"] = EVALUATOR._acceptance_contract_payload_sha256(contract)

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
                "group": f"{environment}-{pair_id}",
                "run_id": run_ids[arm],
                "run_name": f"{arm}-{environment}-{pair_id}",
                "state": "finished",
            },
            "scheduler": {
                "job_id": submission_receipt.value["held_submissions"][arm]["candidate_job_id"],
                "pair_manifest_sha256": pair_manifest.sha256,
                "submission_receipt_sha256": submission_receipt.sha256,
            },
            "capture": {
                "api_base_url": EVALUATOR.WANDB_API_BASE_URL,
                "authenticated": True,
                "collector_sha256": common["wandb_exporter_sha256"],
                "complete": True,
                "fetched_at_unix_ns": 1_788_000_000_000_000_000,
                "history_method": EVALUATOR.WANDB_HISTORY_METHOD,
                "requested_metrics": EVALUATOR._requested_history_metrics(contract["verifier_metric"]),
                "summary_fallback_used": False,
                "wandb_sdk_version": EVALUATOR.WANDB_SDK_VERSION,
            },
            "provenance": {
                "common": copy.deepcopy(common),
                "source_commits": copy.deepcopy(source_commits),
                "source_git_trees": copy.deepcopy(source_git_trees),
                "trusted_oob_declarations": copy.deepcopy(unverified_lineage),
                "topology": copy.deepcopy(topology),
                "arm": copy.deepcopy(arms[arm]),
            },
            "config": copy.deepcopy(contract["configs"][arm]),
            "history": _sparse_rows(arm, environment),
        }

    return Fixture(
        contract=contract,
        off=run_export("off"),
        on=run_export("on"),
        expected_submission_receipt_sha256=submission_receipt.sha256,
        expected_step1_hashes={
            "off_main_ledger": main_ledgers["off"].sha256,
            "off_transcript_bundle": main_transcripts["off"].sha256,
            "on_main_ledger": main_ledgers["on"].sha256,
            "on_transcript_bundle": main_transcripts["on"].sha256,
            "off_model_transport_bundle": transport_bundles["off"].sha256,
            "off_model_transport_log": transport_logs["off"].sha256,
            "off_model_transport_manifest": transport_manifests["off"].sha256,
            "on_model_transport_bundle": transport_bundles["on"].sha256,
            "on_model_transport_log": transport_logs["on"].sha256,
            "on_model_transport_manifest": transport_manifests["on"].sha256,
        },
        artifacts={
            "pair_manifest": pair_manifest,
            "submission_receipt": submission_receipt,
            "holdout": holdout,
            "off_main_ledger": main_ledgers["off"],
            "off_transcript_bundle": main_transcripts["off"],
            "on_main_ledger": main_ledgers["on"],
            "on_transcript_bundle": main_transcripts["on"],
            "off_model_transport_bundle": transport_bundles["off"],
            "off_model_transport_log": transport_logs["off"],
            "off_model_transport_manifest": transport_manifests["off"],
            "on_model_transport_bundle": transport_bundles["on"],
            "on_model_transport_log": transport_logs["on"],
            "on_model_transport_manifest": transport_manifests["on"],
            "off_execution": execution["off"],
            "on_execution": execution["on"],
            "off_job_exit": job_exit["off"],
            "on_job_exit": job_exit["on"],
            "terminal_scheduler": terminal_scheduler,
        },
    )


@pytest.mark.parametrize("environment", ["reasoning_gym", "citation", "freeform"])
def test_live_pair_is_green_for_every_single_environment(environment: str) -> None:
    report = _fixture(environment).evaluate()

    assert report["overall"]["status"] == "GREEN"
    assert report["overall"]["independent_statuses"] == {
        "live_reward_consistency": "PASS",
        "learning_behavior": "PASS",
        "speed_evidence": "PASS",
    }
    assert report["acceptance_scope"] == EVALUATOR.ACCEPTANCE_SCOPE
    assert "not_captured_output_or_trajectory_equivalence" in report["acceptance_scope"]
    assert (
        "not_captured_output_parity_or_trajectory_equivalence"
        in report["live_reward_consistency"]["evidence"]["claim_scope"]
    )
    assert (
        report["learning_behavior"]["evidence"]["live_reward_noninferiority"]["claim_scope"]
        == "reward-noninferiority-only-not-trajectory-equivalence"
    )


def test_missing_terminal_scheduler_receipt_is_unverifiable() -> None:
    fixture = _fixture()
    del fixture.artifacts["terminal_scheduler"]

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "missing terminal scheduler Pair receipt" in report["live_reward_consistency"]["unavailable"][0]


def test_terminal_scheduler_receipt_raw_query_is_replayed() -> None:
    fixture = _fixture()
    value = copy.deepcopy(fixture.artifacts["terminal_scheduler"].value)
    query = value["captures"]["off"]["query"]
    raw = base64.b64decode(query["raw_stdout_base64"], validate=True)
    scheduler = json.loads(raw)
    scheduler["jobs"][0]["comment"] = "hostile-login-authored-comment"
    mutated_raw = TERMINAL_COLLECTOR.canonical_json_bytes(scheduler, "mutated scontrol output") + b"\n"
    query["raw_stdout_base64"] = base64.b64encode(mutated_raw).decode("ascii")
    query["raw_stdout_sha256"] = hashlib.sha256(mutated_raw).hexdigest()
    query["raw_stdout_byte_count"] = len(mutated_raw)
    _reseal_terminal_scheduler_fixture(fixture, value)

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "name/comment differs from submission" in report["live_reward_consistency"]["unavailable"][0]


def test_terminal_scheduler_normalized_job_id_must_match_raw_and_receipts() -> None:
    fixture = _fixture()
    value = copy.deepcopy(fixture.artifacts["terminal_scheduler"].value)
    value["captures"]["off"]["terminal_record"]["job_id"] = "99999"
    _reseal_terminal_scheduler_fixture(fixture, value)

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "normalized record differs from raw scheduler output" in report["live_reward_consistency"]["unavailable"][0]


def test_terminal_scheduler_non_success_state_is_unverifiable() -> None:
    fixture = _fixture()
    value = copy.deepcopy(fixture.artifacts["terminal_scheduler"].value)
    query = value["captures"]["on"]["query"]
    scheduler = json.loads(base64.b64decode(query["raw_stdout_base64"], validate=True))
    scheduler["jobs"][0]["job_state"] = ["FAILED"]
    mutated_raw = TERMINAL_COLLECTOR.canonical_json_bytes(scheduler, "failed scontrol output") + b"\n"
    query["raw_stdout_base64"] = base64.b64encode(mutated_raw).decode("ascii")
    query["raw_stdout_sha256"] = hashlib.sha256(mutated_raw).hexdigest()
    query["raw_stdout_byte_count"] = len(mutated_raw)
    _reseal_terminal_scheduler_fixture(fixture, value)

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "non-success terminal state 'FAILED'" in report["live_reward_consistency"]["unavailable"][0]


def test_terminal_collector_interface_regression_is_unverifiable(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()

    class MalformedCollector:
        Document = TERMINAL_COLLECTOR.Document

        @staticmethod
        def validate_pair_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(EVALUATOR, "_load_terminal_collector", lambda expected_sha256: MalformedCollector)

    report = fixture.evaluate()

    assert report["overall"]["status"] == "UNVERIFIABLE"
    assert "terminal scheduler receipt rejected: KeyError: 'captures'" in (
        report["live_reward_consistency"]["unavailable"][0]
    )


def test_evaluator_cli_authenticates_terminal_receipt_and_reports_green(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture("citation")
    contract_payload_sha256 = EVALUATOR._acceptance_contract_payload_sha256(fixture.contract)
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = contract_payload_sha256
    for run in (fixture.off, fixture.on):
        run["provenance"]["common"]["acceptance_contract_sha256"] = contract_payload_sha256
    contract = EVALUATOR.document_from_value(fixture.contract)
    off_export = EVALUATOR.wandb_export_document_from_payload(fixture.off)
    on_export = EVALUATOR.wandb_export_document_from_payload(fixture.on)
    documents = {
        "contract": contract,
        "pair": fixture.artifacts["pair_manifest"],
        "submission": fixture.artifacts["submission_receipt"],
        "off-export": off_export,
        "on-export": on_export,
        "holdout": fixture.artifacts["holdout"],
        "off-execution": fixture.artifacts["off_execution"],
        "on-execution": fixture.artifacts["on_execution"],
        "off-exit": fixture.artifacts["off_job_exit"],
        "on-exit": fixture.artifacts["on_job_exit"],
        "terminal": fixture.artifacts["terminal_scheduler"],
    }
    paths = {name: tmp_path / f"{name}.json" for name in documents}
    for name, document in documents.items():
        paths[name].write_bytes(document.raw)

    status = EVALUATOR.main(
        [
            "--contract",
            str(paths["contract"]),
            "--expected-evaluator-sha256",
            hashlib.sha256(EVALUATOR_PATH.read_bytes()).hexdigest(),
            "--expected-acceptance-contract-sha256",
            contract.sha256,
            "--pair-manifest",
            str(paths["pair"]),
            "--submission-receipt",
            str(paths["submission"]),
            "--expected-submission-receipt-sha256",
            fixture.expected_submission_receipt_sha256,
            "--off-export",
            str(paths["off-export"]),
            "--on-export",
            str(paths["on-export"]),
            "--expected-off-export-sha256",
            off_export.sha256,
            "--expected-on-export-sha256",
            on_export.sha256,
            "--holdout-receipt",
            str(paths["holdout"]),
            "--off-execution-receipt",
            str(paths["off-execution"]),
            "--on-execution-receipt",
            str(paths["on-execution"]),
            "--off-job-exit-receipt",
            str(paths["off-exit"]),
            "--on-job-exit-receipt",
            str(paths["on-exit"]),
            "--terminal-scheduler-receipt",
            str(paths["terminal"]),
            "--format",
            "json",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["overall"]["status"] == "GREEN"


def test_live_evaluator_closes_current_pair_v3_sorted_79_boundary() -> None:
    fixture = _fixture()
    boundary = fixture.artifacts["pair_manifest"].value["slurm_export_boundary"]

    assert boundary["schema"] == "nemo-rl-strict-slurm-export-file-v3"
    assert len(boundary["allowed_names"]) == 79
    assert boundary["allowed_names"] == sorted(boundary["allowed_names"])
    assert boundary["allowed_names"] == list(EVALUATOR.SLURM_EXPORT_ALLOWED_NAMES)
    assert {
        "EXPECTED_STRICT_PAIR_BOOTSTRAP_SHA256SUM_SHA256",
        "EXPECTED_STRICT_PAIR_HOST_PYTHON_SHA256",
        "EXPECTED_STRICT_PAIR_RUNTIME_TOOL_MANIFEST_SHA256",
        "STRICT_PAIR_HOST_PYTHON",
        "STRICT_PAIR_RUNTIME_TOOL_MANIFEST",
    }.issubset(boundary["allowed_names"])
    assert (
        EVALUATOR.ENVIRONMENT_SELECTIONS["reasoning_gym"]["config"]["sha256"]
        == "f5517d8edabed2b4d77b493fa6a8a5f55fa8eb4b3da33d66f5f40b1afbf5d8c8"
    )
    assert fixture.evaluate()["overall"]["status"] == "GREEN"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("v2", "boundary schema"),
        ("drop_name", "allowed_names differ"),
        ("extra_name", "allowed_names differ"),
    ],
)
def test_live_evaluator_rejects_non_pair79_boundaries(mutation: str, message: str) -> None:
    fixture = _fixture()
    value = copy.deepcopy(fixture.artifacts["pair_manifest"].value)
    boundary = value["slurm_export_boundary"]
    if mutation == "v2":
        boundary["schema"] = "nemo-rl-strict-slurm-export-file-v2"
    elif mutation == "drop_name":
        boundary["allowed_names"].pop()
    else:
        boundary["allowed_names"].append("ZZZ_HOSTILE_EXTRA")
    document = EVALUATOR.document_from_value(value, trailing_lf=True)
    fixture.contract["provenance"]["common"]["pair_manifest_sha256"] = document.sha256
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
        EVALUATOR._acceptance_contract_payload_sha256(fixture.contract)
    )

    with pytest.raises(EVALUATOR.EvidenceError, match=message):
        EVALUATOR._validate_pair_manifest(document, fixture.contract)


def _set_history_metric(run: dict[str, Any], step: int, metric: str, value: Any) -> None:
    matches = [row for row in run["history"] if row["_step"] == step and metric in row]
    assert matches
    for row in matches:
        row[metric] = value


def _set_reward_mean(run: dict[str, Any], step: int, value: float) -> None:
    for metric in (
        "train/raw_environment_reward",
        "train/pre_penalty_environment_reward",
        "train/verifier_reward",
        "train/total_reward/mean",
        "train/reward",
        EVALUATOR.VERIFIER_METRIC_BY_ENVIRONMENT["reasoning_gym"],
    ):
        _set_history_metric(run, step, metric, value)


def test_live_reward_equation_failure_is_red_without_a_replay_adapter() -> None:
    fixture = _fixture()
    _set_history_metric(fixture.on, 17, "train/reward_processing_delta", 0.25)

    report = fixture.evaluate()

    assert report["live_reward_consistency"]["status"] == "FAIL"
    assert any("effective to processed delta" in message for message in report["live_reward_consistency"]["failures"])
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "RED"


def test_missing_verifier_alias_is_local_unverifiable_live_reward_evidence() -> None:
    fixture = _fixture()
    metric = fixture.contract["verifier_metric"]
    for row in fixture.off["history"]:
        row.pop(metric, None)

    report = fixture.evaluate()

    assert report["live_reward_consistency"]["status"] == "UNVERIFIABLE"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "PASS"
    assert report["overall"]["status"] == "UNVERIFIABLE"


def test_live_reward_noninferiority_is_not_trajectory_equivalence() -> None:
    fixture = _fixture("reasoning_gym")
    for step in EVALUATOR.PRIMARY_STEPS:
        observed = next(
            row["train/raw_environment_reward"]
            for row in fixture.on["history"]
            if row["_step"] == step and "train/raw_environment_reward" in row
        )
        _set_reward_mean(fixture.on, step, float(observed) - 0.2)

    report = fixture.evaluate()

    assert report["live_reward_consistency"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "FAIL"
    assert any(
        "paired live-reward bootstrap lower confidence bound" in message
        for message in report["learning_behavior"]["failures"]
    )
    evidence = report["learning_behavior"]["evidence"]["live_reward_noninferiority"]
    assert evidence["claim_scope"] == ("reward-noninferiority-only-not-trajectory-equivalence")


def test_optimizer_update_witness_requires_exact_true() -> None:
    fixture = _fixture()
    _set_history_metric(fixture.on, 42, "train/update_successful", False)

    report = fixture.evaluate()

    assert report["learning_behavior"]["status"] == "FAIL"
    assert any("optimizer update was not successful" in message for message in report["learning_behavior"]["failures"])


def test_paired_speed_gate_requires_a_strict_lower_bound_above_one() -> None:
    fixture = _fixture()
    for step in EVALUATOR.PRIMARY_STEPS:
        _set_history_metric(fixture.on, step, "timing/train/policy_training", 10.0)

    report = fixture.evaluate()

    assert report["live_reward_consistency"]["status"] == "PASS"
    assert report["learning_behavior"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "FAIL"
    assert any("not strictly >1.0" in message for message in report["speed_evidence"]["failures"])


def test_speed_excludes_epochs_with_different_prompt_suffix_geometry() -> None:
    fixture = _fixture()
    for step in EVALUATOR.PRIMARY_STEPS:
        _set_history_metric(
            fixture.on,
            step,
            "train/shared_prefix/prompt_tokens",
            250,
        )
        _set_history_metric(
            fixture.on,
            step,
            "train/shared_prefix/non_loss_suffix_tokens",
            0,
        )

    report = fixture.evaluate()

    assert report["live_reward_consistency"]["status"] == "PASS"
    assert report["speed_evidence"]["status"] == "UNVERIFIABLE"
    assert any("complete matched primary epochs=0" in message for message in report["speed_evidence"]["failures"])


def test_identical_code_snapshot_is_required_but_runtime_directions_are_distinct() -> None:
    fixture = _fixture()
    arms = fixture.contract["provenance"]["arms"]

    assert arms["off"]["snapshot_manifest_sha256"] == arms["on"]["snapshot_manifest_sha256"]
    assert arms["off"]["runtime_environment_sha256"] != arms["on"]["runtime_environment_sha256"]
    assert fixture.evaluate()["overall"]["status"] == "GREEN"

    arms["on"]["runtime_environment_sha256"] = arms["off"]["runtime_environment_sha256"]
    fixture.contract["provenance"]["common"]["acceptance_contract_sha256"] = (
        EVALUATOR._acceptance_contract_payload_sha256(fixture.contract)
    )
    with pytest.raises(EVALUATOR.EvidenceError, match="runtime_environment_sha256"):
        EVALUATOR.validate_contract(fixture.contract)


def test_live_evaluator_has_no_captured_replay_cli_or_argument_surface() -> None:
    import inspect

    assert "captured_replay_inputs" not in inspect.signature(EVALUATOR.evaluate_pair).parameters
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert "--captured-replay" not in source
    assert "_run_isolated" not in source
    assert "replay-1_manifest" not in source
