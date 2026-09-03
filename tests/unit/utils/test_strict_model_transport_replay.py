from __future__ import annotations

import copy
import hashlib
import os
import stat
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nemo_rl.environments.strict_gym_child_runtime as child_runtime
from nemo_rl.utils.strict_captured_replay_evidence import (
    TRANSCRIPT_BUNDLE_SCHEMA,
    canonical_ascii_json,
    publish_evidence_document,
    validate_transcript_bundle,
)
from nemo_rl.utils.strict_captured_replay_manifest import (
    AuthenticatedOffSourceCapture,
)
from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_LEDGER_SCHEMA,
    build_main_step1_ledger,
)
from nemo_rl.utils.strict_model_transport import (
    MODEL_TRANSPORT_MANIFEST_SCHEMA,
    build_model_transport_policy,
    build_model_transport_manifest,
    initialize_model_transport_directory,
    publish_model_transport_capture,
    publish_model_transport_manifest,
)
import nemo_rl.utils.strict_model_transport_replay as replay_module
from nemo_rl.utils.strict_model_transport_replay import (
    MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA,
    REASONING_SCORE_CALL_INDEX_SCHEMA,
    REASONING_SCORE_CLOSED_SCHEMA,
    ReplayVerifierMaterial,
    StrictModelTransportReplayError,
    load_strict_model_transport_replay_source,
    publish_strict_model_transport_replay_consumption,
    validate_strict_model_transport_replay_consumption,
)
from tests.unit.utils.test_strict_model_transport import (
    _bundle,
    _transcript,
)
from tests.unit.environments.test_strict_gym_child_runtime import (
    _score_finalizer_fixture,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _device_environment() -> dict[str, Any]:
    return {
        "schema": "nemo-rl-strict-scheduler-device-environment-v1",
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "all",
        "rocr_visible_devices": None,
        "ze_affinity_mask": None,
    }


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _policy() -> dict[str, Any]:
    root = _source_root()
    return build_model_transport_policy(
        collector_sha256=hashlib.sha256(
            (root / "nemo_rl/utils/strict_model_transport.py").read_bytes()
        ).hexdigest(),
        vllm_route_sha256=hashlib.sha256(
            (root / "nemo_rl/models/generation/vllm/vllm_worker_async.py").read_bytes()
        ).hexdigest(),
        rollout_finalizer_sha256=hashlib.sha256(
            (root / "nemo_rl/experience/rollout_manager.py").read_bytes()
        ).hexdigest(),
    )


@pytest.fixture(autouse=True)
def _stub_full_replay_manifest_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest utility's own suite covers its large Pair-derived schema."""
    monkeypatch.setattr(
        replay_module,
        "validate_replay_execution_manifest",
        lambda document, *, authenticated_source: None,
    )


def _group_id_for_task_index(task_index: int) -> str:
    high = 0x305FE72A79E34B88
    low = ((high & ((1 << 63) - 1)) ^ task_index) | (1 << 63)
    value = uuid.UUID(int=(high << 64) | low)
    assert value.version == 4
    assert (value.int ^ (value.int >> 64)) & ((1 << 63) - 1) == task_index
    return str(value)


def _main_ledger(
    transcript: dict[str, Any], transcript_ref: dict[str, Any]
) -> dict[str, Any]:
    task_index = transcript["entries"][0]["agent_run_request"]["_ng_task_index"]
    group_id = _group_id_for_task_index(task_index)
    advantages = (
        -1.1546986103057861,
        1.1546984910964966,
        -1.1546986103057861,
        1.1546984910964966,
    )
    rows: list[dict[str, Any]] = []
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
                "derived_verifier_request_sha256": entry[
                    "derived_verifier_request_sha256"
                ],
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
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        mode="observe",
        generation=transcript["generation"],
        bindings={
            **transcript["bindings"],
            "restart_count": 0,
            "pair_campaign_sha256": _digest("pair-campaign"),
            "pair_campaign_reward_and_advantage_sha256": _digest("reward-policy"),
        },
        transcript_bundle=transcript_ref,
        row_inputs=rows,
        update_successful=True,
    )


def _source_artifacts(
    tmp_path: Path, *, corrupt_raw_log: bool = False
) -> dict[str, Any]:
    results = tmp_path / "off-results"
    results.mkdir(mode=0o700, parents=True)
    policy = _policy()
    transport_directory = initialize_model_transport_directory(
        results_dir=results, model_transport_policy=policy
    )
    bundle = _bundle()
    raw_log_ref, bundle_ref = publish_model_transport_capture(
        transport_directory=transport_directory,
        bundle=bundle,
        model_transport_policy=policy,
        model_path="/immutable/model",
    )
    if corrupt_raw_log:
        raw_path = Path(raw_log_ref["path"])
        os.chmod(raw_path, 0o600)
        raw_path.write_bytes(raw_path.read_bytes() + b"\n")
        os.chmod(raw_path, 0o400)
        raw_log_ref["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    evidence_directory = results / "strict_pair_step1_evidence"
    evidence_directory.mkdir(mode=0o700)
    transcript = _transcript(bundle)
    transcript["model_transport_bundle"] = copy.deepcopy(bundle_ref)
    validate_transcript_bundle(transcript)
    transcript_path = evidence_directory / "transcript-bundle.json"
    _, transcript_sha256 = publish_evidence_document(
        output=transcript_path, document=transcript, trailing_lf=False
    )
    transcript_ref = {
        "path": str(transcript_path),
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": transcript_sha256,
    }
    ledger = _main_ledger(transcript, transcript_ref)
    ledger_path = evidence_directory / "main-ledger.json"
    _, ledger_sha256 = publish_evidence_document(
        output=ledger_path, document=ledger, trailing_lf=False
    )
    ledger_ref = {
        "path": str(ledger_path),
        "schema": MAIN_STEP1_LEDGER_SCHEMA,
        "sha256": ledger_sha256,
    }
    manifest = build_model_transport_manifest(
        pair_id="pair-001",
        environment="reasoning_gym",
        arm="off",
        pair_manifest_sha256="c" * 64,
        authenticated_job_id="12345",
        submission_receipt_sha256="e" * 64,
        capture_server=bundle["capture_server"],
        main_transcript_bundle=transcript_ref,
        main_ledger=ledger_ref,
        transport_bundle=bundle_ref,
        transport_capture=raw_log_ref,
        model_transport_policy_sha256=policy["policy_sha256"],
        entry_count=4,
        ordered_entries_sha256=bundle["ordered_entries_sha256"],
    )
    manifest_path, manifest_sha256 = publish_model_transport_manifest(
        transport_directory=transport_directory, manifest=manifest
    )
    manifest_ref = {
        "path": str(manifest_path),
        "schema": MODEL_TRANSPORT_MANIFEST_SCHEMA,
        "sha256": manifest_sha256,
    }
    source_capture = {
        "arm": "off",
        "restart_count": 0,
        "authenticated_job": {
            "comment": "source-comment",
            "job_id": "12345",
            "job_name": "off-reasoning_gym-pair-001",
            "user_id": str(os.geteuid()),
        },
        "job_receipts": {
            "pre": {
                "path": str(results / "PRE.json"),
                "schema": "nemo-rl-strict-pair-job-receipt-v2",
                "sha256": _digest("pre"),
            },
            "exit": {
                "path": str(results / "EXIT.json"),
                "schema": "nemo-rl-strict-pair-job-receipt-v2",
                "sha256": _digest("exit"),
            },
        },
        "step1_evidence": {
            "schema": "nemo-rl-strict-step1-evidence-index-v4",
            "main_ledger": ledger_ref,
            "transcript_bundle": transcript_ref,
            "model_transport": {
                "schema": "nemo-rl-strict-model-transport-evidence-index-v1",
                "bundle": bundle_ref,
                "manifest": manifest_ref,
                "raw_log": raw_log_ref,
                "ordered_entries_sha256": bundle["ordered_entries_sha256"],
            },
        },
    }
    authenticated_source = AuthenticatedOffSourceCapture(
        source_capture=copy.deepcopy(source_capture),
        pair_manifest={
            "model_transport": copy.deepcopy(policy),
            "source": {"snapshots": {"off": {"path": str(_source_root())}}},
        },
        pair_manifest_sha256="c" * 64,
        pair_submission_receipt={"test": "authenticated Pair receipt"},
        pair_submission_receipt_sha256="e" * 64,
        trusted_off_exit_receipt_path=str(results / "EXIT.json"),
        trusted_off_exit_receipt_sha256=_digest("exit"),
        pre_receipt={"test": "authenticated PRE"},
        pre_receipt_sha256=_digest("pre"),
        exit_receipt={"test": "authenticated EXIT"},
        exit_receipt_sha256=_digest("exit"),
        main_ledger=copy.deepcopy(ledger),
        transcript_bundle=copy.deepcopy(transcript),
        transport_bundle=copy.deepcopy(bundle),
        transport_manifest=copy.deepcopy(manifest),
        transport_records=tuple(copy.deepcopy(bundle["entries"])),
    )
    replay_manifest = {
        "pair_id": "pair-001",
        "environment": "reasoning_gym",
        "attempt_id": "replay-1",
        "pair": {
            "manifest": {"sha256": "c" * 64},
            "submission_receipt": {"sha256": "e" * 64},
            "model_transport_policy_sha256": policy["policy_sha256"],
        },
        "source_capture": copy.deepcopy(source_capture),
        "artifacts": {
            "outputs": {"directory": {"path": str(tmp_path / "replay-attempt")}}
        },
    }
    return {
        "attempt_root": tmp_path / "replay-attempt",
        "bundle_ref": bundle_ref,
        "ledger": ledger,
        "manifest_ref": manifest_ref,
        "policy": policy,
        "raw_log_ref": raw_log_ref,
        "replay_manifest": replay_manifest,
        "source": authenticated_source,
        "transcript": transcript,
        "transcript_ref": transcript_ref,
    }


def _load_source(artifacts: dict[str, Any]):
    return load_strict_model_transport_replay_source(
        source=artifacts["source"],
        replay_execution_manifest=artifacts["replay_manifest"],
    )


def _replace_immutable_document(path: Path, document: dict[str, Any]) -> str:
    os.chmod(path, 0o600)
    path.write_bytes(canonical_ascii_json(document))
    os.chmod(path, 0o400)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_score_evidence(
    root: Path,
    transcript: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_id: str = "67890",
) -> dict[str, Any]:
    receipt_root = root / "strict_gym_child_runtime"
    receipt_root.mkdir(mode=0o700, parents=True)

    original_build_spec = child_runtime._build_spec

    def build_spec(**kwargs: Any) -> dict[str, Any]:
        kwargs["pair_id"] = "pair-001"
        kwargs["job_id"] = job_id
        return original_build_spec(**kwargs)

    monkeypatch.setattr(child_runtime, "_build_spec", build_spec)
    session, _, call_documents, run_helper = _score_finalizer_fixture(
        monkeypatch, receipt_root
    )
    expected_calls: list[dict[str, Any]] = []
    closed_path = receipt_root / "reasoning-score-closed.json"
    closed = replay_module._load_artifact_document(
        {
            "path": str(closed_path),
            "sha256": hashlib.sha256(closed_path.read_bytes()).hexdigest(),
            "schema": REASONING_SCORE_CLOSED_SCHEMA,
        }
    )
    for index, (entry, call_document) in enumerate(
        zip(transcript["entries"], call_documents, strict=True)
    ):
        request = entry["derived_verifier_request"]
        response = entry["verifier_response"]
        scorer_entry = {
            "question": copy.deepcopy(request["question"]),
            "answer": copy.deepcopy(request["answer"]),
            "metadata": copy.deepcopy(request["metadata"]),
        }
        expectation = child_runtime.reasoning_score_call_expectation(
            task_name="knights_knaves",
            answer=response["extracted_answer"],
            entry=scorer_entry,
            float_result=response["reward"],
        )
        expected_calls.append(expectation)
        call_document["input"] = {
            "answer_sha256": expectation["answer_sha256"],
            "entry_sha256": expectation["entry_sha256"],
        }
        call_document["outcome"] = {
            "kind": "returned",
            "float_result": expectation["float_result"],
        }
        call_sha256 = _replace_immutable_document(
            receipt_root / f"reasoning-score-call-{index + 1:08d}.json",
            call_document,
        )
        closed["calls"][index]["sha256"] = call_sha256
    _replace_immutable_document(closed_path, closed)
    terminal, digest = session.finalize_score_calls(
        expected_calls, run_helper=run_helper
    )
    assert terminal["quiescence"]["child_termination_signal"] == "SIGINT"
    return {
        "path": str(receipt_root / "reasoning-score-call-index.json"),
        "sha256": digest,
        "schema": REASONING_SCORE_CALL_INDEX_SCHEMA,
    }


def _complete(
    source,
    transcript: dict[str, Any],
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    _consume_and_record(source, transcript)
    return source.finalize(
        replay_execution_manifest_sha256=_digest("replay-manifest"),
        authenticated_job_id="67890",
        process={
            "boot_id_sha256": _digest("boot"),
            "pid": 711,
            "start_time_ticks": 98123,
        },
        scheduler_device_environment=_device_environment(),
        reasoning_score_call_index_ref=_publish_score_evidence(
            output_root, transcript, monkeypatch
        ),
    )


def _consume_and_record(source, transcript: dict[str, Any]) -> None:
    for index, entry in enumerate(transcript["entries"]):
        material = source.consume(
            rollout_index=index, generation_seed=entry["generation_seed"]
        )
        assert isinstance(material, ReplayVerifierMaterial)
        assert set(material) == {
            "rollout_index",
            "generation_seed",
            "model_response",
            "agent_run_request",
            "derived_verifier_request",
            "source_entry_sha256",
            "request_body_sha256",
            "response_body_sha256",
        }
        source.record_fresh_verifier_result(
            rollout_index=index,
            verifier_response=entry["verifier_response"],
        )


def _finalize_with_score_ref(source, score_ref: dict[str, Any]) -> dict[str, Any]:
    return source.finalize(
        replay_execution_manifest_sha256=_digest("replay-manifest"),
        authenticated_job_id="67890",
        process={
            "boot_id_sha256": _digest("boot"),
            "pid": 711,
            "start_time_ticks": 98123,
        },
        scheduler_device_environment=_device_environment(),
        reasoning_score_call_index_ref=score_ref,
    )


def test_loader_consumes_k4_and_publishes_terminal_no_lf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _source_artifacts(tmp_path)
    source = _load_source(artifacts)
    assert source.source_transcript_document == artifacts["transcript"]
    assert source.source_main_ledger_document == artifacts["ledger"]

    document = _complete(
        source, artifacts["transcript"], artifacts["attempt_root"], monkeypatch
    )
    assert source.phase == "complete-terminal"
    assert document["schema"] == MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA
    assert document["schema"] == (
        "nemo-rl-strict-model-transport-replay-consumption-v2"
    )
    assert document["status"] == "complete-terminal"
    assert document["replay"]["scheduler_device_environment"] == (_device_environment())
    assert set(document["replay"]) == {
        "attempt_id",
        "replay_execution_manifest_sha256",
        "authenticated_job_id",
        "process",
        "scheduler_device_environment",
        "scorer_evidence",
    }
    assert [entry["rollout_index"] for entry in document["entries"]] == [0, 1, 2, 3]
    validate_strict_model_transport_replay_consumption(document)
    with pytest.raises(StrictModelTransportReplayError, match="complete-terminal"):
        source.consume(
            rollout_index=0,
            generation_seed=artifacts["transcript"]["entries"][0]["generation_seed"],
        )

    replay_directory = tmp_path / "replay-results"
    replay_directory.mkdir(mode=0o700)
    output, digest = publish_strict_model_transport_replay_consumption(
        output=replay_directory / "model-transport-replay-consumption.json",
        document=document,
    )
    raw = output.read_bytes()
    assert not raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == digest
    metadata = os.lstat(output)
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_nlink == 1


def test_material_payload_mutation_does_not_change_retained_source(
    tmp_path: Path,
) -> None:
    artifacts = _source_artifacts(tmp_path)
    source = _load_source(artifacts)
    entry = artifacts["transcript"]["entries"][0]
    material = source.consume(rollout_index=0, generation_seed=entry["generation_seed"])
    payload = material["model_response"]
    payload["status"] = "changed"
    assert (
        source.source_transcript_document["entries"][0]["model_response"]["status"]
        == "completed"
    )


def test_wrong_seed_and_duplicate_consumption_poison_source(tmp_path: Path) -> None:
    artifacts = _source_artifacts(tmp_path)
    source = _load_source(artifacts)
    seed = artifacts["transcript"]["entries"][0]["generation_seed"]
    with pytest.raises(StrictModelTransportReplayError, match="generation_seed"):
        source.consume(rollout_index=0, generation_seed=seed + 1)
    assert source.phase == "poisoned"
    with pytest.raises(StrictModelTransportReplayError, match="poisoned"):
        source.consume(rollout_index=0, generation_seed=seed)

    source = _load_source(artifacts)
    source.consume(rollout_index=0, generation_seed=seed)
    with pytest.raises(StrictModelTransportReplayError, match="already consumed"):
        source.consume(rollout_index=0, generation_seed=seed)
    assert source.phase == "poisoned"


def test_invalid_fresh_result_and_premature_finalize_poison_source(
    tmp_path: Path,
) -> None:
    artifacts = _source_artifacts(tmp_path)
    entry = artifacts["transcript"]["entries"][0]
    source = _load_source(artifacts)
    source.consume(rollout_index=0, generation_seed=entry["generation_seed"])
    changed = copy.deepcopy(entry["verifier_response"])
    changed["score"] = 0.25
    with pytest.raises(StrictModelTransportReplayError, match="score differs"):
        source.record_fresh_verifier_result(rollout_index=0, verifier_response=changed)
    assert source.phase == "poisoned"

    source = _load_source(artifacts)
    with pytest.raises(StrictModelTransportReplayError, match="not all K=4"):
        source.finalize(
            replay_execution_manifest_sha256=_digest("replay-manifest"),
            authenticated_job_id="67890",
            process={
                "boot_id_sha256": _digest("boot"),
                "pid": 711,
                "start_time_ticks": 98123,
            },
            scheduler_device_environment=_device_environment(),
            reasoning_score_call_index_ref={},
        )
    assert source.phase == "poisoned"


def test_loader_rejects_identity_and_raw_log_drift(tmp_path: Path) -> None:
    artifacts = _source_artifacts(tmp_path / "identity")
    bad_capture = copy.deepcopy(artifacts["source"].source_capture)
    bad_capture["authenticated_job"]["job_id"] = "12346"
    bad_source = replace(artifacts["source"], source_capture=bad_capture)
    bad_replay = copy.deepcopy(artifacts["replay_manifest"])
    bad_replay["source_capture"] = copy.deepcopy(bad_capture)
    with pytest.raises(
        ValueError, match="transport manifest source authenticated_job_id differs"
    ):
        load_strict_model_transport_replay_source(
            source=bad_source,
            replay_execution_manifest=bad_replay,
        )

    corrupt = _source_artifacts(tmp_path / "raw", corrupt_raw_log=True)
    with pytest.raises(ValueError, match="raw transport log differs"):
        _load_source(corrupt)


def test_loader_rejects_reordered_authenticated_records_and_extra_ref_key(
    tmp_path: Path,
) -> None:
    artifacts = _source_artifacts(tmp_path / "reorder")
    reordered = replace(
        artifacts["source"],
        transport_records=tuple(reversed(artifacts["source"].transport_records)),
    )
    with pytest.raises(ValueError, match="transport records differ"):
        load_strict_model_transport_replay_source(
            source=reordered,
            replay_execution_manifest=artifacts["replay_manifest"],
        )

    artifacts = _source_artifacts(tmp_path / "extra")
    bad_capture = copy.deepcopy(artifacts["source"].source_capture)
    bad_capture["step1_evidence"]["model_transport"]["raw_log"]["extra"] = True
    bad_source = replace(artifacts["source"], source_capture=bad_capture)
    bad_replay = copy.deepcopy(artifacts["replay_manifest"])
    bad_replay["source_capture"] = copy.deepcopy(bad_capture)
    with pytest.raises(ValueError, match="raw_log_ref keys differ"):
        load_strict_model_transport_replay_source(
            source=bad_source,
            replay_execution_manifest=bad_replay,
        )


def test_loader_rejects_symlinked_raw_named_path(tmp_path: Path) -> None:
    artifacts = _source_artifacts(tmp_path)
    raw_path = Path(artifacts["raw_log_ref"]["path"])
    saved_path = raw_path.with_name("saved-model-transport.jsonl")
    raw_path.rename(saved_path)
    raw_path.symlink_to(saved_path)
    with pytest.raises(ValueError, match="secure path contains a symlink"):
        _load_source(artifacts)


def test_raw_stable_reader_rejects_named_path_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _source_artifacts(tmp_path)
    raw_path = Path(artifacts["raw_log_ref"]["path"])
    real_lstat = replay_module.os.lstat
    target_hits = 0

    def drifting_lstat(path):
        nonlocal target_hits
        metadata = real_lstat(path)
        if Path(path) != raw_path:
            return metadata
        target_hits += 1
        if target_hits < 3:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_uid=metadata.st_uid,
            st_nlink=metadata.st_nlink,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns + 1,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    monkeypatch.setattr(replay_module.os, "lstat", drifting_lstat)
    with pytest.raises(RuntimeError, match="changed during stable read"):
        replay_module._load_immutable_bytes(
            raw_path,
            expected_sha256=artifacts["raw_log_ref"]["sha256"],
            maximum=replay_module.MAX_BUNDLE_BYTES,
        )


def test_finalize_rejects_tampered_and_reordered_scorer_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _source_artifacts(tmp_path / "tamper")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    first_call = (
        artifacts["attempt_root"]
        / "strict_gym_child_runtime/reasoning-score-call-00000001.json"
    )
    os.chmod(first_call, 0o600)
    first_call.write_bytes(first_call.read_bytes() + b" ")
    os.chmod(first_call, 0o400)
    with pytest.raises(StrictModelTransportReplayError, match="canonical JSON"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"

    artifacts = _source_artifacts(tmp_path / "reorder-score")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    terminal_path = Path(score_ref["path"])
    terminal = replay_module._load_artifact_document(score_ref)
    terminal["calls"] = list(reversed(terminal["calls"]))
    score_ref["sha256"] = _replace_immutable_document(terminal_path, terminal)
    with pytest.raises(StrictModelTransportReplayError, match="record 1 differs"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"

    artifacts = _source_artifacts(tmp_path / "extra-terminal-call-ref")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    terminal_path = Path(score_ref["path"])
    terminal = replay_module._load_artifact_document(score_ref)
    terminal["calls"][0]["receipt"]["unexpected"] = True
    score_ref["sha256"] = _replace_immutable_document(terminal_path, terminal)
    with pytest.raises(StrictModelTransportReplayError, match="wrong keyset"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"


def test_finalize_rejects_tampered_closure_and_unreaped_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _source_artifacts(tmp_path / "closure")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    terminal_path = Path(score_ref["path"])
    terminal = replay_module._load_artifact_document(score_ref)
    closed_ref = terminal["score_closed"]
    closed_path = Path(closed_ref["path"])
    closed = replay_module._load_artifact_document(closed_ref)
    closed["calls"] = list(reversed(closed["calls"]))
    terminal["score_closed"]["sha256"] = _replace_immutable_document(
        closed_path, closed
    )
    score_ref["sha256"] = _replace_immutable_document(terminal_path, terminal)
    with pytest.raises(StrictModelTransportReplayError, match="closed ref 1 differs"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"

    artifacts = _source_artifacts(tmp_path / "unreaped")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    terminal_path = Path(score_ref["path"])
    terminal = replay_module._load_artifact_document(score_ref)
    terminal["quiescence"]["original_process_reaped"] = False
    score_ref["sha256"] = _replace_immutable_document(terminal_path, terminal)
    with pytest.raises(StrictModelTransportReplayError, match="quiescence differs"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"

    artifacts = _source_artifacts(tmp_path / "bad-returncode")
    source = _load_source(artifacts)
    _consume_and_record(source, artifacts["transcript"])
    score_ref = _publish_score_evidence(
        artifacts["attempt_root"], artifacts["transcript"], monkeypatch
    )
    terminal_path = Path(score_ref["path"])
    terminal = replay_module._load_artifact_document(score_ref)
    terminal["quiescence"]["wrapper_returncode"] = False
    score_ref["sha256"] = _replace_immutable_document(terminal_path, terminal)
    with pytest.raises(StrictModelTransportReplayError, match="quiescence differs"):
        _finalize_with_score_ref(source, score_ref)
    assert source.phase == "poisoned"


def test_consumption_validator_rejects_changed_entry_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _source_artifacts(tmp_path)
    document = _complete(
        _load_source(artifacts),
        artifacts["transcript"],
        artifacts["attempt_root"],
        monkeypatch,
    )
    wrong_environment = copy.deepcopy(document)
    wrong_environment["environment"] = "citation"
    with pytest.raises(ValueError, match="only reasoning_gym"):
        validate_strict_model_transport_replay_consumption(wrong_environment)

    extra_scorer_leaf = copy.deepcopy(document)
    extra_scorer_leaf["replay"]["scorer_evidence"]["call_receipts"] = []
    with pytest.raises(ValueError, match="scorer evidence keys differ"):
        validate_strict_model_transport_replay_consumption(extra_scorer_leaf)

    missing_device_environment = copy.deepcopy(document)
    missing_device_environment["replay"].pop("scheduler_device_environment")
    with pytest.raises(ValueError, match="replay consumption replay keys differ"):
        validate_strict_model_transport_replay_consumption(missing_device_environment)

    invalid_device_environment = copy.deepcopy(document)
    invalid_device_environment["replay"]["scheduler_device_environment"][
        "cuda_visible_devices"
    ] = "0,1,2"
    with pytest.raises(ValueError, match="four canonical"):
        validate_strict_model_transport_replay_consumption(invalid_device_environment)

    changed = copy.deepcopy(document)
    changed["entries"][0]["generation_seed"] += 1
    with pytest.raises(ValueError, match="generation seed does not close"):
        validate_strict_model_transport_replay_consumption(changed)
