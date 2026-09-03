from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from nemo_rl.utils.strict_model_transport import (
    HASH_DOMAIN,
    K4_SAMPLES,
    MAIN_STEP1_LEDGER_SCHEMA,
    MODEL_TRANSPORT_BUNDLE,
    MODEL_TRANSPORT_BUNDLE_SCHEMA,
    MODEL_TRANSPORT_CALL_SCHEMA,
    MODEL_TRANSPORT_LOG,
    MODEL_TRANSPORT_MANIFEST,
    MODEL_TRANSPORT_MANIFEST_SCHEMA,
    MODEL_TRANSPORT_POLICY_SCHEMA,
    STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
    StrictModelTransportCapture,
    build_model_transport_bundle,
    build_model_transport_call,
    build_model_transport_manifest,
    build_model_transport_policy,
    initialize_model_transport_directory,
    observe_capture_server_identity,
    publish_model_transport_capture,
    publish_model_transport_manifest,
    validate_model_transport_bundle,
    validate_model_transport_call,
    validate_model_transport_manifest,
    validate_model_transport_model_response_join,
    validate_model_transport_policy,
)
from nemo_rl.utils.strict_captured_replay_evidence import (
    build_transcript_bundle,
    build_verifier_request_derivation,
    canonical_ascii_json,
    derive_nemo_gym_request_seed,
    document_sha256,
    domain_sha256,
    validate_transcript_model_transport_join,
)


def _digest(byte: str) -> str:
    return byte * 64


def _policy() -> dict[str, object]:
    return build_model_transport_policy(
        collector_sha256=_digest("1"),
        vllm_route_sha256=_digest("2"),
        rollout_finalizer_sha256=_digest("3"),
    )


def _server() -> dict[str, object]:
    base = {
        "boot_id_sha256": _digest("4"),
        "hostname": "hsg-node-0001",
        "pid": 4321,
        "start_time_ticks": 987654,
    }
    return base | {
        "server_instance_id": domain_sha256("model-transport-server-instance", base)
    }


def _request(index: int) -> dict[str, object]:
    seed = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=index
    )
    return {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "truncate_history_thinking": False,
        },
        "logprobs": True,
        "max_tokens": 768,
        "messages": [{"content": "Zoey and Riley?", "role": "user"}],
        "metadata": {"extra_body": json.dumps({"seed": seed}, separators=(",", ":"))},
        "model": "/immutable/model",
        "return_tokens_as_token_ids": True,
        "seed": seed,
        "temperature": 1.0,
        "top_logprobs": 0,
        "top_p": 1.0,
    }


def _response(index: int) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "content": [
                        {
                            "bytes": [65],
                            "logprob": -0.1,
                            "token": f"token_id:{200 + index}",
                            "top_logprobs": [],
                        }
                    ]
                },
                "message": {
                    "annotations": None,
                    "audio": None,
                    "content": None,
                    "function_call": None,
                    "generation_log_probs": [-0.1],
                    "generation_token_ids": [200 + index],
                    "prompt_token_ids": [101, 102],
                    "reasoning": f"answer {index}",
                    "refusal": None,
                    "role": "assistant",
                },
                "routed_experts": None,
                "stop_reason": None,
                "token_ids": None,
            }
        ],
        "created": 1,
        "id": f"chatcmpl-{index:016x}",
        "kv_transfer_params": None,
        "metrics": None,
        "model": "/immutable/model",
        "object": "chat.completion",
        "prompt_logprobs": None,
        "prompt_text": None,
        "prompt_token_ids": None,
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 2,
            "prompt_tokens_details": None,
            "total_tokens": 3,
        },
    }


def _generation_request(index: int) -> dict[str, object]:
    raw = _request(index)
    return {
        "input": copy.deepcopy(raw["messages"]),
        "max_output_tokens": raw["max_tokens"],
        "metadata": copy.deepcopy(raw["metadata"]),
        "temperature": raw["temperature"],
        "top_p": raw["top_p"],
    }


def _expanded_generation_request(index: int) -> dict[str, object]:
    compact = _generation_request(index)
    message = copy.deepcopy(compact["input"][0])
    message["type"] = "message"
    return {
        "background": None,
        "include": None,
        "input": [message],
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": copy.deepcopy(compact["metadata"]),
        "model": None,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": None,
        "service_tier": None,
        "store": None,
        "stream": None,
        "temperature": 1.0,
        "text": None,
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": None,
        "top_p": 1.0,
        "truncation": None,
        "user": None,
    }


def _fixture_row() -> dict[str, object]:
    return {
        "agent_ref": {
            "name": "reasoning_gym_simple_agent",
            "type": "responses_api_agents",
        },
        "answer": "Zoey",
        "metadata": {"source_dataset": "knights_knaves"},
        "question": "Zoey and Riley?",
        "responses_create_params": {
            "input": [{"content": "Zoey and Riley?", "role": "user"}]
        },
    }


def _model_response(index: int) -> dict[str, object]:
    generation_request = _generation_request(index)
    return {
        "background": None,
        "conversation": None,
        "created_at": 1.0,
        "error": None,
        "id": f"resp_{index:012x}40008000{index:012x}",
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 768,
        "max_tool_calls": None,
        "metadata": copy.deepcopy(generation_request["metadata"]),
        "model": "/immutable/model",
        "object": "response",
        "output": [
            {
                "content": None,
                "encrypted_content": None,
                "generation_log_probs": [-0.1],
                "generation_token_ids": [200 + index],
                "id": f"rs_{index:012x}40008000{index:012x}",
                "prompt_token_ids": [101, 102],
                "routed_experts": None,
                "summary": [{"text": f"answer {index}", "type": "summary_text"}],
                "type": "reasoning",
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
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": None},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 3,
        },
        "user": None,
    }


def _transcript_entry(index: int, transport: dict[str, object]) -> dict[str, object]:
    generation_request = _generation_request(index)
    model_response = _model_response(index)
    agent_run_request = copy.deepcopy(_fixture_row())
    agent_run_request.update(
        {
            "_ng_rollout_index": index,
            "_ng_task_index": 123456,
            "_rowidx": index,
            "responses_create_params": copy.deepcopy(generation_request),
        }
    )
    verifier_response = {
        "response": copy.deepcopy(model_response),
        "responses_create_params": _expanded_generation_request(index),
        "reward": float(index % 2),
        "task_name": "knights_knaves",
        "score": float(index % 2),
        "extracted_answer": "",
    }
    derived_verifier_request = copy.deepcopy(agent_run_request)
    derived_verifier_request["responses_create_params"] = _expanded_generation_request(
        index
    )
    derived_verifier_request["response"] = copy.deepcopy(model_response)
    return {
        "fixture_row_index": 0,
        "generation_request": generation_request,
        "generation_seed": _request(index)["seed"],
        "model_response": model_response,
        "model_transport_entry_sha256": transport["entry_sha256"],
        "model_transport_request_body_sha256": transport["request_body_sha256"],
        "model_transport_response_body_sha256": transport["response_body_sha256"],
        "raw_environment_reward": float(index % 2),
        "rollout_index": index,
        "sample_index": index,
        "agent_run_request": agent_run_request,
        "derived_verifier_request": derived_verifier_request,
        "verifier_response": verifier_response,
    }


def _transcript(bundle: dict[str, object]) -> dict[str, object]:
    return build_transcript_bundle(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation={
            "max_new_tokens": 768,
            "seed_base": 42,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
        },
        bindings={
            name: _digest(value)
            for name, value in {
                "config_sha256": "a",
                "fixture_sha256": "b",
                "pair_manifest_sha256": "c",
                "snapshot_manifest_sha256": "d",
                "submission_receipt_sha256": "e",
                "verifier_source_sha256": "f",
            }.items()
        }
        | {"job_id": "12345", "run_id": _digest("9")},
        fixture_row=_fixture_row(),
        model_transport_bundle={
            "path": "/results/strict_model_transport/model-transport-bundle.json",
            "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
            "sha256": document_sha256(bundle, trailing_lf=False),
        },
        verifier_request_derivation=build_verifier_request_derivation(
            gym_gitlink_commit="354babf7e3554fcd006807c86e80ef476aec9408",
            gym_tree="1234567890abcdef1234567890abcdef12345678",
            openai_version="2.6.1",
            pydantic_version="2.13.4",
        ),
        entry_inputs=[
            _transcript_entry(index, bundle["entries"][index])
            for index in range(K4_SAMPLES)
        ],
    )


def _call(index: int, *, arrival_index: int | None = None) -> dict[str, object]:
    request = _request(index)
    response = _response(index)
    return build_model_transport_call(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        capture_server=_server(),
        rollout_index=index,
        generation_seed=request["seed"],
        arrival_index=index if arrival_index is None else arrival_index,
        model_path="/immutable/model",
        request_body=canonical_ascii_json(request),
        response_body=canonical_ascii_json(response),
        expected_request_payload=request,
        expected_response_payload=response,
    )


def _bundle() -> dict[str, object]:
    arrivals = [2, 0, 3, 1]
    return build_model_transport_bundle(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        model_transport_policy=_policy(),
        capture_server=_server(),
        entries=[_call(index, arrival_index=arrivals[index]) for index in range(4)],
        model_path="/immutable/model",
    )


def test_policy_r2_is_exact_and_digest_omits_only_digest_field() -> None:
    policy = _policy()
    assert policy["schema"] == MODEL_TRANSPORT_POLICY_SCHEMA
    assert policy["hash_domain"] == HASH_DOMAIN
    assert policy["activation"]["arm_environment"] == {
        "name": "STRICT_PAIR_ARM",
        "off": "off",
        "on": "on",
    }
    assert policy["artifacts"]["directory"]["precondition"] == (
        "absent-at-pre-runtime-creates-exclusively"
    )
    assert policy["http"]["max_request_body_bytes"] == 16 * 1024 * 1024
    assert policy["http"]["max_response_body_bytes"] == 16 * 1024 * 1024
    projection = {key: value for key, value in policy.items() if key != "policy_sha256"}
    assert policy["policy_sha256"] == domain_sha256(
        "model-transport-policy", projection
    )
    validate_model_transport_policy(policy)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["capture_window"].update(step=True),
        lambda value: value["http"].update(max_response_body_bytes=16_777_216.0),
        lambda value: value["activation"]["arm_environment"].update(name="BAD"),
        lambda value: value["artifacts"]["directory"].pop("precondition"),
        lambda value: value["sources"]["collector"].update(sha256="0" * 64),
    ],
)
def test_policy_rejects_exact_type_value_key_and_digest_drift(mutate) -> None:
    policy = _policy()
    mutate(policy)
    with pytest.raises((TypeError, ValueError)):
        validate_model_transport_policy(policy)


def test_raw_call_binds_exact_bytes_parsed_payload_and_identity() -> None:
    call = _call(0)
    assert call["schema"] == MODEL_TRANSPORT_CALL_SCHEMA
    assert (
        call["request_body_sha256"]
        == hashlib.sha256(canonical_ascii_json(_request(0))).hexdigest()
    )
    assert call["request_payload_sha256"] == domain_sha256(
        "model-transport-request-payload", _request(0)
    )
    assert call["entry_sha256"] == domain_sha256(
        "model-transport-entry",
        {
            "arm": "off",
            "capture_server": _server(),
            "endpoint": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "request_media_type": "application/json",
                "response_media_type": "application/json",
                "status_code": 200,
                "streaming": False,
            },
            "entry": {
                key: value for key, value in call.items() if key != "entry_sha256"
            },
            "environment": "reasoning_gym",
            "pair_id": "pair-001",
        },
    )


def test_call_rejects_wrong_seed_noncanonical_base64_and_nested_alias() -> None:
    request = _request(0)
    response = _response(0)
    with pytest.raises(ValueError, match="generation_seed"):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            rollout_index=0,
            generation_seed=request["seed"] + 1,
            arrival_index=0,
            model_path="/immutable/model",
            request_body=canonical_ascii_json(request),
            response_body=canonical_ascii_json(response),
        )

    call = _call(0)
    call["request_body_base64"] += "="
    with pytest.raises(ValueError, match="base64"):
        validate_model_transport_call(
            call,
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            model_path="/immutable/model",
        )

    request_alias = _request(0)
    request_alias["max_tokens"] = 768.0
    with pytest.raises(ValueError, match="exact integer"):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            rollout_index=0,
            generation_seed=request["seed"],
            arrival_index=0,
            model_path="/immutable/model",
            request_body=canonical_ascii_json(request_alias),
            response_body=canonical_ascii_json(response),
            expected_request_payload=request,
            expected_response_payload=response,
        )

    response_alias = _response(0)
    response_alias["choices"][0]["message"]["generation_log_probs"][0] = 0
    response_alias["choices"][0]["logprobs"]["content"][0]["logprob"] = 0
    with pytest.raises(TypeError, match="exact finite JSON float"):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            rollout_index=0,
            generation_seed=request["seed"],
            arrival_index=0,
            model_path="/immutable/model",
            request_body=canonical_ascii_json(request),
            response_body=canonical_ascii_json(response_alias),
        )

    uuid4_shaped_response = _response(0)
    uuid4_shaped_response["id"] = "chatcmpl-00000000000040008000000000000000"
    with pytest.raises(ValueError, match="frozen ChatCompletion form"):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            rollout_index=0,
            generation_seed=request["seed"],
            arrival_index=0,
            model_path="/immutable/model",
            request_body=canonical_ascii_json(request),
            response_body=canonical_ascii_json(uuid4_shaped_response),
        )

    forged_server = _server()
    forged_server["server_instance_id"] = _digest("9")
    with pytest.raises(ValueError, match="does not close over process identity"):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=forged_server,
            rollout_index=0,
            generation_seed=request["seed"],
            arrival_index=0,
            model_path="/immutable/model",
            request_body=canonical_ascii_json(request),
            response_body=canonical_ascii_json(response),
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"duplicate":1,"duplicate":2}',
        b'{"bad":NaN}',
        b'{"negative_zero":-0.0}',
        b"[]",
        b"\xff",
    ],
)
def test_call_rejects_non_strict_request_json(raw: bytes) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_model_transport_call(
            pair_id="pair-001",
            environment="reasoning_gym",
            arm="off",
            capture_server=_server(),
            rollout_index=0,
            generation_seed=_request(0)["seed"],
            arrival_index=0,
            model_path="/immutable/model",
            request_body=raw,
            response_body=canonical_ascii_json(_response(0)),
        )


def test_bundle_is_logical_order_with_arrival_permutation() -> None:
    bundle = _bundle()
    assert bundle["schema"] == MODEL_TRANSPORT_BUNDLE_SCHEMA
    assert [entry["rollout_index"] for entry in bundle["entries"]] == [0, 1, 2, 3]
    assert [entry["arrival_index"] for entry in bundle["entries"]] == [2, 0, 3, 1]
    validate_model_transport_bundle(
        bundle, model_transport_policy=_policy(), model_path="/immutable/model"
    )

    bad = copy.deepcopy(bundle)
    bad["entries"][1]["arrival_index"] = 2
    bad["entries"][1]["entry_sha256"] = _call(1, arrival_index=2)["entry_sha256"]
    bad["ordered_entries_sha256"] = domain_sha256(
        "model-transport-ordered-entries", bad["entries"]
    )
    with pytest.raises(ValueError, match="permutation"):
        validate_model_transport_bundle(
            bad, model_transport_policy=_policy(), model_path="/immutable/model"
        )


def test_transcript_flat_digests_join_to_validated_raw_transport_bundle() -> None:
    bundle = _bundle()
    transcript = _transcript(bundle)
    validate_transcript_model_transport_join(
        transcript_bundle=transcript,
        model_transport_bundle=bundle,
        model_transport_policy=_policy(),
        model_path="/immutable/model",
    )

    bad_entries = [
        _transcript_entry(index, bundle["entries"][index])
        for index in range(K4_SAMPLES)
    ]
    bad_entries[0]["model_transport_response_body_sha256"] = _digest("8")
    bad = build_transcript_bundle(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=transcript["generation"],
        bindings=transcript["bindings"],
        fixture_row=_fixture_row(),
        model_transport_bundle=transcript["model_transport_bundle"],
        verifier_request_derivation=transcript["verifier_request_derivation"],
        entry_inputs=bad_entries,
    )
    with pytest.raises(ValueError, match="response_body_sha256 differs"):
        validate_transcript_model_transport_join(
            transcript_bundle=bad,
            model_transport_bundle=bundle,
            model_transport_policy=_policy(),
            model_path="/immutable/model",
        )

    semantic_entries = [
        _transcript_entry(index, bundle["entries"][index])
        for index in range(K4_SAMPLES)
    ]
    semantic_entries[0]["model_response"]["output"][0]["summary"][0]["text"] = (
        "changed answer"
    )
    semantic_entries[0]["verifier_response"]["response"] = copy.deepcopy(
        semantic_entries[0]["model_response"]
    )
    semantic_entries[0]["derived_verifier_request"]["response"] = copy.deepcopy(
        semantic_entries[0]["model_response"]
    )
    semantic_drift = build_transcript_bundle(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=transcript["generation"],
        bindings=transcript["bindings"],
        fixture_row=_fixture_row(),
        model_transport_bundle=transcript["model_transport_bundle"],
        verifier_request_derivation=transcript["verifier_request_derivation"],
        entry_inputs=semantic_entries,
    )
    with pytest.raises(ValueError, match="raw ChatCompletion/Gym model response"):
        validate_transcript_model_transport_join(
            transcript_bundle=semantic_drift,
            model_transport_bundle=bundle,
            model_transport_policy=_policy(),
            model_path="/immutable/model",
        )


def test_r31_truncated_think_content_projects_to_message_only_response() -> None:
    raw_responses = [_response(index) for index in range(K4_SAMPLES)]
    raw_responses[0]["choices"][0]["finish_reason"] = "length"
    raw_responses[0]["choices"][0]["stop_reason"] = 17
    raw_responses[0]["choices"][0]["message"]["reasoning"] = None
    raw_responses[0]["choices"][0]["message"]["content"] = "<think>"
    calls = []
    for index, response in enumerate(raw_responses):
        request = _request(index)
        calls.append(
            build_model_transport_call(
                pair_id="pair-001",
                environment="reasoning_gym",
                arm="off",
                capture_server=_server(),
                rollout_index=index,
                generation_seed=request["seed"],
                arrival_index=index,
                model_path="/immutable/model",
                request_body=canonical_ascii_json(request),
                response_body=canonical_ascii_json(response),
            )
        )
    bundle = build_model_transport_bundle(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        model_transport_policy=_policy(),
        capture_server=_server(),
        entries=calls,
        model_path="/immutable/model",
    )
    model_responses = [_model_response(index) for index in range(K4_SAMPLES)]
    model_responses[0]["incomplete_details"] = {"reason": "max_output_tokens"}
    model_responses[0]["status"] = "incomplete"
    model_responses[0]["output"] = [
        {
            "content": [
                {
                    "annotations": [],
                    "logprobs": None,
                    "text": "<think>",
                    "type": "output_text",
                }
            ],
            "generation_log_probs": [-0.1],
            "generation_token_ids": [200],
            "id": "msg_00000000000040008000000000000000",
            "prompt_token_ids": [101, 102],
            "role": "assistant",
            "routed_experts": None,
            "status": "completed",
            "type": "message",
        }
    ]
    validate_model_transport_model_response_join(
        bundle,
        generation_requests=[_generation_request(index) for index in range(K4_SAMPLES)],
        model_responses=model_responses,
    )


def test_publish_capture_and_manifest_leave_exact_three_file_inventory(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    policy = _policy()
    directory = initialize_model_transport_directory(
        results_dir=results, model_transport_policy=policy
    )
    with pytest.raises(FileExistsError):
        initialize_model_transport_directory(
            results_dir=results, model_transport_policy=policy
        )

    bundle = _bundle()
    capture_ref, bundle_ref = publish_model_transport_capture(
        transport_directory=directory,
        bundle=bundle,
        model_transport_policy=policy,
        model_path="/immutable/model",
    )
    assert capture_ref["record_count"] == K4_SAMPLES
    assert capture_ref["record_schema"] == MODEL_TRANSPORT_CALL_SCHEMA
    assert bundle_ref["schema"] == MODEL_TRANSPORT_BUNDLE_SCHEMA

    transcript_ref = {
        "path": str(tmp_path / "transcript-bundle.json"),
        "schema": STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": _digest("5"),
    }
    ledger_ref = {
        "path": str(tmp_path / "main-ledger.json"),
        "schema": MAIN_STEP1_LEDGER_SCHEMA,
        "sha256": _digest("6"),
    }
    manifest = build_model_transport_manifest(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        pair_manifest_sha256=_digest("7"),
        authenticated_job_id="12345",
        submission_receipt_sha256=_digest("8"),
        capture_server=_server(),
        main_transcript_bundle=transcript_ref,
        main_ledger=ledger_ref,
        transport_bundle=bundle_ref,
        transport_capture=capture_ref,
        model_transport_policy_sha256=policy["policy_sha256"],
        entry_count=bundle["entry_count"],
        ordered_entries_sha256=bundle["ordered_entries_sha256"],
    )
    assert manifest["schema"] == MODEL_TRANSPORT_MANIFEST_SCHEMA
    validate_model_transport_manifest(manifest)

    old_transcript_manifest = copy.deepcopy(manifest)
    old_transcript_manifest["main_transcript_bundle"]["schema"] = (
        "nemo-rl-strict-step1-transcript-bundle-v3"
    )
    with pytest.raises(ValueError, match="main_transcript_bundle.schema"):
        validate_model_transport_manifest(old_transcript_manifest)

    old_ledger_manifest = copy.deepcopy(manifest)
    old_ledger_manifest["main_ledger"]["schema"] = "nemo-rl-strict-main-step1-ledger-v4"
    with pytest.raises(ValueError, match="main_ledger.schema"):
        validate_model_transport_manifest(old_ledger_manifest)

    double_slash_manifest = copy.deepcopy(manifest)
    double_slash_manifest["main_transcript_bundle"]["path"] = (
        "//tmp/transcript-bundle.json"
    )
    with pytest.raises(ValueError, match="absolute path"):
        validate_model_transport_manifest(double_slash_manifest)

    manifest_path, _ = publish_model_transport_manifest(
        transport_directory=directory, manifest=manifest
    )
    assert manifest_path == directory / MODEL_TRANSPORT_MANIFEST

    assert sorted(item.name for item in directory.iterdir()) == sorted(
        [MODEL_TRANSPORT_LOG, MODEL_TRANSPORT_BUNDLE, MODEL_TRANSPORT_MANIFEST]
    )
    assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700
    for name in (MODEL_TRANSPORT_LOG, MODEL_TRANSPORT_BUNDLE, MODEL_TRANSPORT_MANIFEST):
        metadata = os.lstat(directory / name)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 1
    log = (directory / MODEL_TRANSPORT_LOG).read_bytes()
    assert log.endswith(b"\n")
    assert len(log.splitlines()) == K4_SAMPLES
    assert not (directory / MODEL_TRANSPORT_BUNDLE).read_bytes().endswith(b"\n")
    assert not (directory / MODEL_TRANSPORT_MANIFEST).read_bytes().endswith(b"\n")


def test_capture_state_machine_publishes_only_after_explicit_attestation(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    capture = StrictModelTransportCapture(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        results_dir=results,
        model_path="/immutable/model",
        model_transport_policy=_policy(),
        capture_server=_server(),
    )
    directory = results / "strict_model_transport"
    assert capture.phase == "open"
    assert sorted(item.name for item in directory.iterdir()) == [
        "capture-open.json",
        "spool",
    ]

    for index in (2, 0, 3, 1):
        request = _request(index)
        ticket = capture.begin_chat_call(
            request_body=canonical_ascii_json(request),
            typed_seed=request["seed"],
            method="POST",
            path="/v1/chat/completions",
            query="",
            media_type="application/json",
            expected_request_payload=request,
            expected_generation_input=request["messages"],
        )
        assert ticket is not None
        capture.record_success(
            ticket,
            response_body=canonical_ascii_json(_response(index)),
            status_code=200,
            media_type="application/json",
            streaming=False,
            expected_response_payload=_response(index),
            expected_generation_input=request["messages"],
        )

    assert capture.phase == "full-set-awaiting-finalizer"
    assert not (directory / MODEL_TRANSPORT_LOG).exists()
    assert not (directory / MODEL_TRANSPORT_BUNDLE).exists()
    expected_requests = [_generation_request(index) for index in range(K4_SAMPLES)]
    request_digests = [
        hashlib.sha256(canonical_ascii_json(_request(index))).hexdigest()
        for index in range(K4_SAMPLES)
    ]
    capture_ref, bundle_ref, bundle = capture.attest_step1_complete(
        expected_generation_requests=expected_requests,
        expected_model_responses=[
            _model_response(index) for index in range(K4_SAMPLES)
        ],
        expected_request_body_sha256s=request_digests,
    )
    assert capture.phase == "pass-through"
    assert (
        capture_ref["sha256"]
        == hashlib.sha256((directory / MODEL_TRANSPORT_LOG).read_bytes()).hexdigest()
    )
    assert (
        bundle_ref["sha256"]
        == hashlib.sha256((directory / MODEL_TRANSPORT_BUNDLE).read_bytes()).hexdigest()
    )
    assert [entry["rollout_index"] for entry in bundle["entries"]] == [0, 1, 2, 3]
    assert [entry["arrival_index"] for entry in bundle["entries"]] == [1, 3, 0, 2]
    assert sorted(item.name for item in directory.iterdir()) == [
        MODEL_TRANSPORT_BUNDLE,
        MODEL_TRANSPORT_LOG,
    ]
    capture.record_unmatched_failure(reason="later unobserved traffic")
    capture.guard_unlisted(method="GET", path="/health")
    assert (
        capture.begin_chat_call(
            request_body=b"later live traffic is not inspected",
            typed_seed=0,
            method="POST",
            path="/v1/chat/completions",
            query="",
            media_type="application/json",
        )
        is None
    )


def test_capture_poison_is_terminal_on_pre_attestation_drift(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    capture = StrictModelTransportCapture(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        results_dir=results,
        model_path="/immutable/model",
        model_transport_policy=_policy(),
        capture_server=_server(),
    )
    for index in range(K4_SAMPLES):
        request = _request(index)
        ticket = capture.begin_chat_call(
            request_body=canonical_ascii_json(request),
            typed_seed=request["seed"],
            method="POST",
            path="/v1/chat/completions",
            query="",
            media_type="application/json",
        )
        assert ticket is not None
        capture.record_success(
            ticket,
            response_body=canonical_ascii_json(_response(index)),
            status_code=200,
            media_type="application/json",
            streaming=False,
        )

    wrong = [_generation_request(index) for index in range(K4_SAMPLES)]
    wrong[0]["input"][0]["content"] = "wrong"
    with pytest.raises(ValueError, match="generation input"):
        capture.attest_step1_complete(
            expected_generation_requests=wrong,
            expected_model_responses=[
                _model_response(index) for index in range(K4_SAMPLES)
            ],
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        capture.attest_step1_complete(
            expected_generation_requests=[
                _generation_request(index) for index in range(K4_SAMPLES)
            ],
            expected_model_responses=[
                _model_response(index) for index in range(K4_SAMPLES)
            ],
        )
    assert not (results / "strict_model_transport" / MODEL_TRANSPORT_LOG).exists()


def test_capture_unmatched_failure_poison_is_terminal(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    capture = StrictModelTransportCapture(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        results_dir=results,
        model_path="/immutable/model",
        model_transport_policy=_policy(),
        capture_server=_server(),
    )
    capture.record_unmatched_failure(
        reason="request validation failed before capture ticket"
    )
    request = _request(0)
    with pytest.raises(RuntimeError, match="poisoned"):
        capture.begin_chat_call(
            request_body=canonical_ascii_json(request),
            typed_seed=request["seed"],
            method="POST",
            path="/v1/chat/completions",
            query="",
            media_type="application/json",
        )


def test_capture_server_hashes_raw_boot_id_including_lf(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "self").mkdir()
    boot_raw = b"12345678-1234-1234-1234-123456789abc\n"
    (proc / "sys/kernel/random/boot_id").write_bytes(boot_raw)
    fields_after_comm = ["S", *[str(index) for index in range(1, 19)], "987654"]
    (proc / "self/stat").write_text(
        f"4321 (capture worker) {' '.join(fields_after_comm)}\n", encoding="ascii"
    )

    observed = observe_capture_server_identity(proc_root=proc, hostname="node-1")
    assert observed["boot_id_sha256"] == hashlib.sha256(boot_raw).hexdigest()
    assert observed["start_time_ticks"] == 987654
    assert observed["hostname"] == "node-1"
