# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_COMPARED_FIELDS,
    MAIN_STEP1_HASH_DOMAIN,
    MAIN_STEP1_LEDGER_SCHEMA,
    build_main_step1_ledger,
    canonical_ascii_json,
    derive_nemo_gym_request_seed,
    domain_sha256,
    main_step1_runtime_contract,
    publish_main_step1_ledger,
    strict_main_step1_enabled,
    validate_main_step1_ledger,
)


def _penalty_flags(**overrides: bool) -> dict[str, bool]:
    flags = {
        "reasoning_equal_to_final_answer": False,
        "empty_final_answer": False,
        "unwanted_token": False,
        "malformed_think_tag": False,
        "invalid_tool_call": False,
        "malformed_thinking": False,
        "raw_invalid_tool_call": False,
        "raw_malformed_thinking": False,
        "invalid_and_malformed": False,
    }
    flags.update(overrides)
    return flags


def _generation() -> dict[str, object]:
    return {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }


def _bindings() -> dict[str, object]:
    names = (
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "run_id",
        "fixture_sha256",
        "verifier_source_sha256",
        "snapshot_manifest_sha256",
        "config_sha256",
        "pair_campaign_sha256",
        "pair_campaign_reward_and_advantage_sha256",
    )
    values: dict[str, object] = {
        name: hashlib.sha256(name.encode("ascii")).hexdigest() for name in names
    }
    values.update({"job_id": "6830125", "restart_count": 0})
    return values


def _transcript_ref() -> dict[str, str]:
    return {
        "path": "/strict/results/strict_pair_step1_evidence/transcript-bundle.json",
        "schema": "nemo-rl-strict-step1-transcript-bundle-v4",
        "sha256": hashlib.sha256(b"transcript bundle").hexdigest(),
    }


def _row(sample_index: int) -> dict[str, object]:
    fixture_row_index = 0
    group_id = "305fe72a-79e3-4b88-9eb2-000000000000"
    prompt = [101, 102]
    completion = [200 + sample_index, 300 + sample_index]
    token_ids = [*prompt, *completion]
    token_mask = [0.0, 0.0, 1.0, 1.0]
    rloo_advantage = (
        -1.1546986103057861 if sample_index % 2 == 0 else 1.1546984910964966
    )
    return {
        "sample_index": sample_index,
        "sample_id": f"{group_id}_g{sample_index}",
        "shared_prefix_group_id": group_id,
        "fixture_row_index": fixture_row_index,
        "rollout_index": sample_index,
        "generation_seed": derive_nemo_gym_request_seed(
            seed_base=42,
            fixture_row_index=fixture_row_index,
            rollout_index=sample_index,
        ),
        "request_sha256": domain_sha256(
            "step1-generation-request", {"row": sample_index}
        ),
        "response_sha256": domain_sha256("step1-model-response", {"row": sample_index}),
        "agent_run_request_sha256": domain_sha256(
            "step1-agent-run-request", {"row": sample_index}
        ),
        "derived_verifier_request_sha256": domain_sha256(
            "step1-derived-verifier-request", {"row": sample_index}
        ),
        "verifier_response_sha256": domain_sha256(
            "step1-verifier-response", {"row": sample_index}
        ),
        "token_ids": token_ids,
        "input_length": len(token_ids),
        "prompt_token_ids": prompt,
        "completion_token_ids": completion,
        "token_loss_mask": token_mask,
        "raw_environment_reward": float(sample_index % 2),
        "pre_penalty_environment_reward": float(sample_index % 2),
        "penalty_flags": _penalty_flags(),
        "verifier_reward": float(sample_index % 2),
        "processed_reward": float(sample_index % 2),
        "sample_mask": 1.0,
        "advantages": [rloo_advantage] * len(token_ids),
        "valid_loss_tokens": 2,
        "total_tokens": len(token_ids),
    }


def _ledger() -> dict[str, object]:
    return build_main_step1_ledger(
        pair_id="rg-step1-pair",
        environment="reasoning_gym",
        arm="on",
        mode="train",
        generation=_generation(),
        bindings=_bindings(),
        transcript_bundle=_transcript_ref(),
        row_inputs=[_row(index) for index in range(4)],
        update_successful=True,
    )


def test_builds_exact_canonical_content_bound_main_ledger() -> None:
    ledger = _ledger()

    assert ledger["schema"] == MAIN_STEP1_LEDGER_SCHEMA
    assert ledger["hash_domain"] == MAIN_STEP1_HASH_DOMAIN
    assert ledger["step"] == 1
    assert ledger["update_successful"] is True
    assert ledger["sample_count"] == 4
    assert ledger["compared_fields"] == MAIN_STEP1_COMPARED_FIELDS
    # Full response documents remain authenticated arm-local evidence, but Gym
    # deliberately generates fresh UUID4 response/item IDs on every execution.
    # They therefore close row/output hashes without becoming parity fields.
    assert "response_sha256" not in ledger["compared_fields"]
    assert "derived_verifier_request_sha256" not in ledger["compared_fields"]
    assert "verifier_response_sha256" not in ledger["compared_fields"]
    assert ledger["step_totals"] == {
        "raw_environment_reward_sum": 2.0,
        "pre_penalty_environment_reward_sum": 2.0,
        "verifier_reward_sum": 2.0,
        "processed_reward_sum": 2.0,
        "sample_mask_sum": 4,
        "global_valid_toks": 8,
        "total_num_tokens": 16,
    }

    row = ledger["rows"][0]
    assert row["prompt_sha256"] == domain_sha256("step1-prompt", [101, 102])
    assert row["request_sha256"] == domain_sha256(
        "step1-generation-request", {"row": 0}
    )
    assert row["response_sha256"] == domain_sha256("step1-model-response", {"row": 0})
    assert "verifier_response_sha256" in row
    assert ledger["outputs_sha256"] == domain_sha256(
        "step1-outputs",
        [
            {
                "sample_index": item["sample_index"],
                "fixture_row_index": item["fixture_row_index"],
                "rollout_index": item["rollout_index"],
                "prompt_sha256": item["prompt_sha256"],
                "request_sha256": item["request_sha256"],
                "generation_seed": item["generation_seed"],
                "prompt_token_ids": item["prompt_token_ids"],
                "response_sha256": item["response_sha256"],
                "agent_run_request_sha256": item["agent_run_request_sha256"],
                "derived_verifier_request_sha256": item[
                    "derived_verifier_request_sha256"
                ],
                "verifier_response_sha256": item["verifier_response_sha256"],
                "token_ids": item["token_ids"],
                "input_length": item["input_length"],
                "completion_token_ids": item["completion_token_ids"],
                "token_loss_mask": item["token_loss_mask"],
                "valid_loss_tokens": item["valid_loss_tokens"],
                "total_tokens": item["total_tokens"],
            }
            for item in ledger["rows"]
        ],
    )
    assert row["row_sha256"] == domain_sha256(
        "step1-row", {key: value for key, value in row.items() if key != "row_sha256"}
    )

    payload = canonical_ascii_json(ledger)
    assert not payload.endswith(b"\n")
    assert payload == json.dumps(
        ledger,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    validate_main_step1_ledger(ledger)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["rows"][0].update(sample_id="other_g0"),
            "must equal",
        ),
        (
            lambda value: value["rows"][0]["token_ids"].__setitem__(3, 999),
            "completion_token_ids",
        ),
        (
            lambda value: value["rows"][0].update(generation_seed=7),
            "request seed",
        ),
        (
            lambda value: value["rows"][0]["penalty_flags"].update(extra=False),
            "keyset mismatch",
        ),
        (
            lambda value: value["rows"][0].update(processed_reward=-0.0),
            "negative zero",
        ),
        (
            lambda value: value.update(step=1.0),
            "step must be integer 1",
        ),
        (
            lambda value: value.update(sample_count=4.0),
            "sample_count must be integer 4",
        ),
        (
            lambda value: value.update(update_successful=1),
            "update_successful must be exact true",
        ),
        (
            lambda value: value["rows"][0].update(sample_mask=0.0),
            "sample_mask=1.0",
        ),
        (
            lambda value: value["rows"][0]["penalty_flags"].update(unwanted_token=True),
            "all penalties false",
        ),
        (
            lambda value: value["rows"][0].update(verifier_reward=1.0),
            "rewards to match rowwise",
        ),
    ],
)
def test_rejects_changed_or_malformed_ledger(mutation, match: str) -> None:
    ledger = copy.deepcopy(_ledger())
    mutation(ledger)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_main_step1_ledger(ledger)


def test_rejects_nonexact_json_scalar_types() -> None:
    row = _row(0)
    row["sample_mask"] = 1
    rows = [row, *[_row(index) for index in range(1, 4)]]
    with pytest.raises(TypeError, match="sample_mask must be an exact finite float"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )

    with pytest.raises(ValueError, match="update_successful=true"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=[_row(index) for index in range(4)],
            update_successful=False,
        )


def test_rejects_pre_r334_schema_and_verifier_request_field() -> None:
    ledger = copy.deepcopy(_ledger())
    ledger["schema"] = "nemo-rl-strict-main-step1-ledger-v4"
    with pytest.raises(ValueError, match="main-step ledger schema mismatch"):
        validate_main_step1_ledger(ledger)

    ledger = copy.deepcopy(_ledger())
    row = ledger["rows"][0]
    row["verifier_request_sha256"] = row.pop("agent_run_request_sha256")
    with pytest.raises(ValueError, match="keyset mismatch"):
        validate_main_step1_ledger(ledger)


def test_reasoning_gym_accepts_fractional_reward_but_format_envs_do_not() -> None:
    partial_reward = 0.6499999761581421
    rows = [_row(index) for index in range(4)]
    for row in rows:
        row.update(
            raw_environment_reward=partial_reward,
            pre_penalty_environment_reward=partial_reward,
            verifier_reward=partial_reward,
            processed_reward=partial_reward,
            advantages=[0.0] * row["input_length"],
        )
    ledger = build_main_step1_ledger(
        pair_id="rg-step1-pair",
        environment="reasoning_gym",
        arm="on",
        mode="train",
        generation=_generation(),
        bindings=_bindings(),
        transcript_bundle=_transcript_ref(),
        row_inputs=rows,
        update_successful=True,
    )
    assert ledger["step_totals"]["raw_environment_reward_sum"] == (4 * partial_reward)

    with pytest.raises(ValueError, match="citation/freeform"):
        build_main_step1_ledger(
            pair_id="citation-step1-pair",
            environment="citation",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )

    rows = [_row(index) for index in range(4)]
    for row in rows:
        row.update(
            raw_environment_reward=0.6499999999999999,
            pre_penalty_environment_reward=0.6499999999999999,
            verifier_reward=0.6499999999999999,
            processed_reward=0.6499999999999999,
            advantages=[0.0] * row["input_length"],
        )
    with pytest.raises(ValueError, match="IEEE float32 round-trip"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )


def test_rloo_advantage_tolerance_is_absolute_and_covers_full_row() -> None:
    rows = [_row(index) for index in range(4)]
    rows[0]["advantages"][0] += 1.9e-6
    build_main_step1_ledger(
        pair_id="rg-step1-pair",
        environment="reasoning_gym",
        arm="on",
        mode="train",
        generation=_generation(),
        bindings=_bindings(),
        transcript_bundle=_transcript_ref(),
        row_inputs=rows,
        update_successful=True,
    )

    rows = [_row(index) for index in range(4)]
    rows[0]["advantages"][0] += 2.1e-6
    with pytest.raises(ValueError, match="float32 K4 RLOO"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )


def test_rejects_generation_drift_and_unbounded_sequence_geometry() -> None:
    generation = _generation()
    generation["seed_base"] = 43
    with pytest.raises(ValueError, match="generation policy differs"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=generation,
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=[_row(index) for index in range(4)],
            update_successful=True,
        )

    rows = [_row(index) for index in range(4)]
    rows[0].update(input_length=131_073, total_tokens=131_073)
    with pytest.raises(ValueError, match="must be <= 131072"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )


def test_rejects_noncanonical_identity_and_zero_digest() -> None:
    rows = [_row(index) for index in range(4)]
    rows[0]["shared_prefix_group_id"] = "305fe72a-79e3-5b88-9eb2-000000000000"
    rows[0]["sample_id"] = f"{rows[0]['shared_prefix_group_id']}_g0"
    with pytest.raises(ValueError, match="canonical UUID4"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )

    boundary = build_main_step1_ledger(
        pair_id="x" * 64,
        environment="reasoning_gym",
        arm="on",
        mode="train",
        generation=_generation(),
        bindings=_bindings(),
        transcript_bundle=_transcript_ref(),
        row_inputs=[_row(index) for index in range(4)],
        update_successful=True,
    )
    assert boundary["pair_id"] == "x" * 64

    with pytest.raises(ValueError, match=r"\{0,63\}"):
        build_main_step1_ledger(
            pair_id="x" * 65,
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=[_row(index) for index in range(4)],
            update_successful=True,
        )

    rows = [_row(index) for index in range(4)]
    rows[0]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nonzero lowercase SHA-256"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )


def test_rejects_nonzero_fixture_or_different_prompt_within_k4() -> None:
    rows = [_row(index) for index in range(4)]
    rows[0]["fixture_row_index"] = 1
    rows[0]["generation_seed"] = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=1, rollout_index=0
    )
    with pytest.raises(ValueError, match="fixture_row_index exactly 0"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )

    rows = [_row(index) for index in range(4)]
    rows[1].update(
        prompt_token_ids=[101, 999],
        token_ids=[101, 999, 201, 301],
    )
    with pytest.raises(ValueError, match="identical prompt_token_ids"):
        build_main_step1_ledger(
            pair_id="rg-step1-pair",
            environment="reasoning_gym",
            arm="on",
            mode="train",
            generation=_generation(),
            bindings=_bindings(),
            transcript_bundle=_transcript_ref(),
            row_inputs=rows,
            update_successful=True,
        )


def test_loads_only_complete_wrapper_bound_runtime_contract(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    sha = hashlib.sha256(b"binding").hexdigest()
    environ = {
        "STRICT_PAIR_LAUNCH_MODE": "submit",
        "STRICT_PAIR_ARM": "off",
        "STRICT_PAIR_ENVIRONMENT": "citation",
        "PAIR_ID": "strict-citation-pair",
        "RESULTS_DIR": str(tmp_path),
        "TRAIN_PATH": str(tmp_path / "fixture.jsonl"),
        "EXPECTED_PAIR_MANIFEST_SHA256": sha,
        "EXPECTED_STRICT_PAIR_SUBMISSION_RECEIPT_SHA256": sha,
        "STRICT_PAIR_BOUND_JOB_ID": "6830125",
        "WANDB_RUN_ID": sha,
        "STRICT_PAIR_BOUND_RESTART_COUNT": "0",
        "EXPECTED_STRICT_PAIR_FIXTURE_SHA256": sha,
        "EXPECTED_STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256": sha,
        "EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256": sha,
        "EXPECTED_STRICT_PAIR_CONFIG_SHA256": sha,
        "EXPECTED_STRICT_PAIR_CAMPAIGN_SHA256": sha,
        "EXPECTED_STRICT_PAIR_REWARD_AND_ADVANTAGE_SHA256": sha,
        "EXPECTED_STRICT_PAIR_MODEL_TRANSPORT_POLICY_SHA256": sha,
        "EXPECTED_GYM_GITLINK_COMMIT": "1" * 40,
        "EXPECTED_GYM_TREE": "2" * 40,
    }

    contract = main_step1_runtime_contract(environ)

    assert contract == {
        "pair_id": "strict-citation-pair",
        "environment": "citation",
        "arm": "off",
        "mode": "observe",
        "results_dir": str(tmp_path),
        "fixture_path": str(tmp_path / "fixture.jsonl"),
        "model_transport_policy_sha256": sha,
        "gym_gitlink_commit": "1" * 40,
        "gym_tree": "2" * 40,
        "bindings": {
            "pair_manifest_sha256": sha,
            "submission_receipt_sha256": sha,
            "job_id": "6830125",
            "run_id": sha,
            "restart_count": 0,
            "fixture_sha256": sha,
            "verifier_source_sha256": sha,
            "snapshot_manifest_sha256": sha,
            "config_sha256": sha,
            "pair_campaign_sha256": sha,
            "pair_campaign_reward_and_advantage_sha256": sha,
        },
    }

    environ["RESULTS_DIR"] = "//tmp/strict-pair-results"
    with pytest.raises(ValueError, match="canonical absolute path"):
        main_step1_runtime_contract(environ)
    environ["RESULTS_DIR"] = str(tmp_path)

    del environ["EXPECTED_STRICT_PAIR_CAMPAIGN_SHA256"]
    with pytest.raises(RuntimeError, match="EXPECTED_STRICT_PAIR_CAMPAIGN_SHA256"):
        main_step1_runtime_contract(environ)

    environ["EXPECTED_STRICT_PAIR_CAMPAIGN_SHA256"] = sha
    environ["EXPECTED_GYM_TREE"] = "0" * 40
    with pytest.raises(ValueError, match="nonzero lowercase 40-hex"):
        main_step1_runtime_contract(environ)


def test_publish_is_exclusive_mode_0400_single_link_and_no_lf(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    ledger = _ledger()

    path, digest = publish_main_step1_ledger(results_dir=str(tmp_path), document=ledger)

    assert path == tmp_path / "strict_pair_step1_evidence" / "main-ledger.json"
    payload = path.read_bytes()
    metadata = os.lstat(path)
    assert payload == canonical_ascii_json(ledger)
    assert not payload.endswith(b"\n")
    assert hashlib.sha256(payload).hexdigest() == digest
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_nlink == 1
    assert stat.S_IMODE(os.lstat(path.parent).st_mode) == 0o700

    with pytest.raises(FileExistsError, match="already exists"):
        publish_main_step1_ledger(results_dir=str(tmp_path), document=ledger)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({"STRICT_PAIR_LAUNCH_MODE": "dry-run"}, False),
        (
            {"STRICT_PAIR_LAUNCH_MODE": "submit", "STRICT_PAIR_ARM": "off"},
            True,
        ),
        (
            {"STRICT_PAIR_LAUNCH_MODE": "submit", "STRICT_PAIR_ARM": "on"},
            True,
        ),
    ],
)
def test_activation_requires_submit_and_bound_arm(
    environ: dict[str, str], expected: bool
) -> None:
    assert strict_main_step1_enabled(environ) is expected


@pytest.mark.parametrize(
    "environ",
    [
        {"STRICT_PAIR_ARM": "off"},
        {"STRICT_PAIR_LAUNCH_MODE": "submit"},
        {"STRICT_PAIR_LAUNCH_MODE": "submit", "STRICT_PAIR_ARM": "bad"},
    ],
)
def test_rejects_ambiguous_partial_activation(environ: dict[str, str]) -> None:
    with pytest.raises(RuntimeError):
        strict_main_step1_enabled(environ)
