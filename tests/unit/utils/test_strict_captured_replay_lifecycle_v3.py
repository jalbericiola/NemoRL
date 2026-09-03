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
from types import SimpleNamespace
from typing import Any

import pytest

import nemo_rl.environments.strict_gym_child_runtime as child_runtime
import nemo_rl.utils.strict_captured_replay_evidence as evidence
import nemo_rl.utils.strict_captured_replay_manifest as replay_manifest_module
from nemo_rl.algorithms.strict_captured_replay_runtime import (
    execute_captured_replay_cohort,
    reasoning_gym_score_call_material,
)
from nemo_rl.utils.strict_captured_replay_evidence import (
    REPLAY_EXIT_ROOT_KEYS,
    REPLAY_POST_INDEX_ROOT_KEYS,
    REPLAY_PRE_ROOT_KEYS,
    REPLAY_SCHEDULER_QUERY_ROOT_KEYS,
    REPLAY_SUBMISSION_ROOT_KEYS,
    build_captured_replay_evidence_index,
    build_captured_replay_exit_receipt,
    build_captured_replay_pre_receipt,
    build_captured_replay_scheduler_query,
    build_captured_replay_submission_receipt,
    load_captured_replay_evidence_index,
    load_captured_replay_exit_receipt,
    load_captured_replay_pre_receipt,
    load_captured_replay_scheduler_query,
    load_captured_replay_submission_receipt,
    load_evidence_document,
    publish_captured_replay_evidence_index,
    publish_captured_replay_exit_receipt,
    publish_captured_replay_pre_receipt,
    publish_captured_replay_scheduler_query,
    publish_captured_replay_submission_receipt,
    publish_evidence_document,
    replay_run_id,
    validate_captured_replay_evidence_index,
    validate_captured_replay_exit_receipt,
    validate_captured_replay_pre_receipt,
    validate_captured_replay_submission_receipt,
)
from nemo_rl.utils.strict_captured_replay_manifest import (
    AuthenticatedOffSourceCapture,
    publish_replay_execution_manifest,
)
from nemo_rl.utils.strict_model_transport_replay import (
    load_strict_model_transport_replay_source,
    publish_strict_model_transport_replay_consumption,
)
from tests.unit.environments.test_strict_gym_child_runtime import (
    _bootstrap_module,
    _score_finalizer_fixture,
)
from tests.unit.utils.test_strict_captured_replay_manifest import _manifest, _seal_bytes


@pytest.fixture(autouse=True)
def _trusted_foreign_container_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project the cluster's pinned foreign container owner into local fixtures."""

    def identity(pair: dict[str, Any]) -> dict[str, Any]:
        return {
            **pair["artifacts"]["container"],
            "owner_uid": 153493,
            "owner_gid": 30,
        }

    monkeypatch.setattr(
        replay_manifest_module,
        "_stable_container_asset_identity",
        identity,
    )


_JOB_ID = "82001"
_PAIR_JOB_IDS = ("6787903", "6787904")


@pytest.fixture(autouse=True)
def _restore_test_tree_permissions(tmp_path: Path):
    """Return imported sealed-snapshot fixtures to removable modes."""
    yield
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


def _device_environment() -> dict[str, Any]:
    return {
        "schema": evidence.SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA,
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "all",
        "rocr_visible_devices": None,
        "ze_affinity_mask": None,
    }


def _hardware_observation(nvidia_smi: dict[str, Any]) -> dict[str, Any]:
    raw = "NVIDIA GB200, 580.126.20"
    rows = [
        {
            "index": index,
            "raw": raw,
            "gpu_model": "NVIDIA GB200",
            "driver_version": "580.126.20",
        }
        for index in range(4)
    ]
    return {
        "schema": evidence.HARDWARE_OBSERVATION_SCHEMA,
        "gpu_model": "NVIDIA GB200",
        "driver_version": "580.126.20",
        "gpu_row_count": 4,
        "ordered_rows": rows,
        "raw_output_sha256": hashlib.sha256(((raw + "\n") * 4).encode("ascii")).hexdigest(),
        "ordered_rows_sha256": evidence.domain_sha256(evidence.HARDWARE_ORDERED_ROWS_HASH_LABEL, rows),
        "nvidia_smi": copy.deepcopy(nvidia_smi),
    }


def _query_raw(record: dict[str, Any], *, duplicate_job_id: bool = False) -> bytes:
    document = {
        "errors": [],
        "jobs": [
            {
                "comment": record["comment"],
                "current_working_directory": record["work_dir"],
                "hold": record["held"],
                "job_id": int(record["job_id"]),
                "job_state": [record["job_state"]],
                "name": record["job_name"],
                "restart_cnt": record["restart_count"],
                "state_reason": record["reason"],
                "user_id": int(record["user_id"]),
            }
        ],
        "last_backfill": {},
        "last_update": {},
        "meta": {},
        "warnings": [],
    }
    raw = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if duplicate_job_id:
        token = f'"job_id":{int(record["job_id"])}'.encode("ascii")
        raw = raw.replace(token, token + b"," + token, 1)
    return raw + b"\n"


def _record(
    manifest: dict[str, Any],
    *,
    comment: str,
    phase: str,
    job_id: str = _JOB_ID,
) -> dict[str, Any]:
    held = phase == "PRE_RELEASE"
    return {
        "job_id": job_id,
        "job_name": manifest["scheduler_submission"]["identity"]["job_name"],
        "comment": comment,
        "user_id": str(manifest["scheduler_submission"]["identity"]["submitter_euid"]),
        "work_dir": manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        "job_state": "PENDING" if held else "RUNNING",
        "reason": "JobHeldUser" if held else "None",
        "held": held,
        "restart_count": 0,
    }


def _seal_query(
    manifest: dict[str, Any],
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    phase: str,
    raw_path: Path,
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    raw_sha = _seal_bytes(raw_path, _query_raw(record))
    query = build_captured_replay_scheduler_query(
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
        phase=phase,
        raw_output_path=str(raw_path),
        raw_output_sha256=raw_sha,
        record=record,
    )
    query_path = raw_path.with_name(raw_path.name.removesuffix(".scontrol.raw") + ".scontrol-query.json")
    _, query_sha = publish_captured_replay_scheduler_query(
        output=query_path,
        document=query,
        replay_execution_manifest=manifest,
        authenticated_source=authenticated_source,
    )
    return query, {
        "path": str(query_path),
        "schema": evidence.REPLAY_SCHEDULER_QUERY_SCHEMA,
        "sha256": query_sha,
    }


def _sbatch_argv(manifest: dict[str, Any], *, manifest_path: Path, manifest_sha: str, comment: str) -> list[str]:
    pair_path = Path(manifest["pair"]["manifest"]["path"])
    pair, _ = load_evidence_document(
        path=pair_path,
        expected_sha256=manifest["pair"]["manifest"]["sha256"],
        trailing_lf=True,
    )
    campaign = pair["campaign"]
    slurm = campaign["slurm"]
    snapshot_root = Path(manifest["replay_contract"]["source_snapshot"]["ref"]["path"])
    wrapper = snapshot_root / manifest["replay_contract"]["program"]["job_wrapper"]["path"]
    slurm_root = Path(manifest["execution_environment"]["attempt"]["operational"]["slurm"])
    off_exit = manifest["source_capture"]["job_receipts"]["exit"]
    return [
        "--parsable",
        "--hold",
        f"--chdir={snapshot_root}",
        f"--nodes={campaign['nodes']}",
        f"--account={slurm['account']}",
        f"--job-name={manifest['scheduler_submission']['identity']['job_name']}",
        f"--partition={slurm['partition']}",
        "--time=04:00:00",
        "--gres=gpu:4",
        "--exclusive",
        "--mem=0",
        "--dependency=singleton",
        "--segment=1",
        f"--output={slurm_root}/slurm-%j.out",
        f"--error={slurm_root}/slurm-%j.err",
        f"--qos={slurm['qos']}",
        f"--comment={comment}",
        f"--export-file={manifest['slurm_export_boundary']['path']}",
        str(wrapper),
        "--pair-manifest",
        manifest["pair"]["manifest"]["path"],
        "--pair-manifest-sha256",
        manifest["pair"]["manifest"]["sha256"],
        "--pair-submission-receipt",
        manifest["pair"]["submission_receipt"]["path"],
        "--pair-submission-receipt-sha256",
        manifest["pair"]["submission_receipt"]["sha256"],
        "--off-exit-receipt",
        off_exit["path"],
        "--off-exit-receipt-sha256",
        off_exit["sha256"],
        "--replay-manifest",
        str(manifest_path),
        "--replay-manifest-sha256",
        manifest_sha,
    ]


def _submission_and_pre(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay_job_id: str = _JOB_ID,
) -> tuple[
    dict[str, Any],
    AuthenticatedOffSourceCapture,
    Path,
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    manifest, pair, source = _manifest(str(tmp_path))
    # The manifest fixture uses platform-placeholder runtime tool paths. Tool
    # byte stability is separately covered for those placeholders.  The
    # deployment-local nvidia-smi fixture is real, so retain its stable byte
    # authentication through the complete EXIT/index graph.
    real_tool_loader = evidence._load_bound_runtime_tool_bytes
    nvidia_smi_path = Path(manifest["runtime_tools"]["document"]["host"]["nvidia_smi"]["path"])

    def load_fixture_tool(path: Path, *, expected_sha256: str) -> bytes:
        if path == nvidia_smi_path:
            return real_tool_loader(path, expected_sha256=expected_sha256)
        return b"tool"

    monkeypatch.setattr(evidence, "_load_bound_runtime_tool_bytes", load_fixture_tool)
    manifest_parent = tmp_path / "control"
    manifest_parent.mkdir(mode=0o700)
    manifest_path, manifest_sha = publish_replay_execution_manifest(
        output=manifest_parent / "REPLAY_EXECUTION_MANIFEST.json",
        document=manifest,
        authenticated_source=source,
    )
    Path(manifest["execution_environment"]["attempt"]["operational"]["slurm"]).mkdir(mode=0o700, parents=True)
    accepted_path = Path(manifest["scheduler_submission"]["accepted_id_record"]["path"])
    accepted_sha = _seal_bytes(accepted_path, f"{replay_job_id}\n".encode("ascii"))
    comment = (
        f"nemo-rl-strict-captured-replay-v1:{manifest['attempt_id']}:"
        f"{manifest['scheduler_submission']['nonce']}:{manifest_sha}"
    )
    pre_release_record = _record(manifest, comment=comment, phase="PRE_RELEASE", job_id=replay_job_id)
    pre_release_raw = accepted_path.parent / "PRE_RELEASE.scontrol.raw"
    _, pre_release_ref = _seal_query(
        manifest,
        authenticated_source=source,
        phase="PRE_RELEASE",
        raw_path=pre_release_raw,
        record=pre_release_record,
    )
    pair_submission, _ = load_evidence_document(
        path=manifest["pair"]["submission_receipt"]["path"],
        expected_sha256=manifest["pair"]["submission_receipt"]["sha256"],
        trailing_lf=True,
    )
    authenticated_scheduler = pair_submission["scheduler_tools"]
    scheduler_tools = {name: authenticated_scheduler[name] for name in ("sbatch", "scancel", "scontrol")}
    submission = build_captured_replay_submission_receipt(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        replay_execution_manifest_path=str(manifest_path),
        replay_execution_manifest_sha256=manifest_sha,
        scheduler_client_environment=authenticated_scheduler["client_environment"],
        scheduler_tools=scheduler_tools,
        sbatch_argv=_sbatch_argv(
            manifest,
            manifest_path=manifest_path,
            manifest_sha=manifest_sha,
            comment=comment,
        ),
        parsable_stdout=f"{replay_job_id}\n",
        accepted_id_record={
            "path": str(accepted_path),
            "sha256": accepted_sha,
            "parsed_candidate_job_id": replay_job_id,
            "format": "ascii-positive-decimal-lf",
            "mode": "0400",
        },
        pre_release_scheduler_query=pre_release_ref,
        submitted_at_unix_ns=1_788_350_000_000_000_000,
    )
    submission_path, submission_sha = publish_captured_replay_submission_receipt(
        output=manifest["scheduler_submission"]["receipt"]["path"],
        document=submission,
        replay_execution_manifest=manifest,
        authenticated_source=source,
    )
    assert submission_path == Path(manifest["scheduler_submission"]["receipt"]["path"])

    job_root = (
        Path(pair["paths"]["results_root"])
        / "captured_replay"
        / "replay_job_state"
        / manifest["pair_id"]
        / manifest["attempt_id"]
        / f"{replay_job_id}-0"
    )
    pre_record = _record(manifest, comment=comment, phase="PRE", job_id=replay_job_id)
    _, pre_query_ref = _seal_query(
        manifest,
        authenticated_source=source,
        phase="PRE",
        raw_path=job_root / "queries/PRE.scontrol.raw",
        record=pre_record,
    )
    (job_root / "receipts").mkdir(mode=0o700)
    job = {
        "account": pair["campaign"]["slurm"]["account"],
        "name": manifest["scheduler_submission"]["identity"]["job_name"],
        "num_nodes": pair["campaign"]["nodes"],
        "partition": pair["campaign"]["slurm"]["partition"],
        "qos": pair["campaign"]["slurm"]["qos"],
        "gpus_per_node": 4,
        "restart_count": 0,
    }
    pre = build_captured_replay_pre_receipt(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        submission_receipt=submission,
        authenticated_job_id=replay_job_id,
        job=job,
        pre_scheduler_query=pre_query_ref,
    )
    pre_path, pre_sha = publish_captured_replay_pre_receipt(
        output=job_root / "receipts/PRE.json",
        document=pre,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        authenticated_source=source,
    )
    assert pre_path == job_root / "receipts/PRE.json"
    return (
        manifest,
        source,
        manifest_path,
        manifest_sha,
        submission,
        submission_sha,
        pre,
        pre_sha,
    )


def _publish_real_terminal_outputs(
    *,
    manifest: dict[str, Any],
    authenticated_source: AuthenticatedOffSourceCapture,
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    process: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, str]]:
    """Execute and publish the production scorer-only replay terminal graph."""
    output_root = Path(manifest["artifacts"]["outputs"]["directory"]["path"])
    output_root.mkdir(mode=0o700, parents=True)
    assert stat.S_IMODE(os.lstat(output_root).st_mode) == 0o700

    transport_source = load_strict_model_transport_replay_source(
        source=authenticated_source,
        replay_execution_manifest=manifest,
    )
    original_build_spec = child_runtime._build_spec

    def build_spec(**kwargs: Any) -> dict[str, Any]:
        kwargs["pair_id"] = manifest["pair_id"]
        kwargs["job_id"] = _JOB_ID
        return original_build_spec(**kwargs)

    monkeypatch.setattr(child_runtime, "_build_spec", build_spec)
    receipt_root = output_root / "strict_gym_child_runtime"
    receipt_root.mkdir(mode=0o700)
    score_session, _, _, score_run_helper = _score_finalizer_fixture(monkeypatch, receipt_root)
    for sequence in range(1, 5):
        (receipt_root / f"reasoning-score-call-{sequence:08d}.json").unlink()
    (receipt_root / "reasoning-score-closed.json").unlink()

    bootstrap = _bootstrap_module(monkeypatch)

    def normalize_reasoning_answer(answer: str) -> set[tuple[str, str]]:
        return {(answer, "role")} if answer else set()

    def frozen_score(*, answer: str, entry: dict[str, Any]) -> float:
        del answer, entry
        return 0.0

    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: frozen_score)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec=score_session.spec,
        spec_sha=score_session.spec_sha256,
        process={"pid": 123, "start_ticks": 777},
        frozen_score_fn=frozen_score,
        frozen_normalize_fn=normalize_reasoning_answer,
    )
    score_with_evidence = reasoning_gym.get_score_answer_fn("knights_knaves")
    expected_score_calls: list[dict[str, Any]] = []

    def post_verifier(
        rollout_index: int,
        generation_seed: int,
        derived_verifier_request: dict[str, Any],
    ) -> dict[str, Any]:
        source_entry = authenticated_source.transcript_bundle["entries"][rollout_index]
        assert generation_seed == source_entry["generation_seed"]
        assert derived_verifier_request == source_entry["derived_verifier_request"]
        response = copy.deepcopy(source_entry["verifier_response"])
        response["score"] = 0.0
        response["reward"] = 0.0
        return response

    def independent_score_check(
        rollout_index: int,
        derived_verifier_request: dict[str, Any],
        verifier_response: dict[str, Any],
    ) -> None:
        assert rollout_index == len(expected_score_calls)
        material = reasoning_gym_score_call_material(
            derived_verifier_request=derived_verifier_request,
            verifier_response=verifier_response,
        )
        assert score_with_evidence(answer=material["answer"], entry=material["entry"]) == material["float_result"]
        expected_score_calls.append(
            child_runtime.reasoning_score_call_expectation(
                task_name=material["task_name"],
                answer=material["answer"],
                entry=material["entry"],
                float_result=material["float_result"],
            )
        )

    def finalize_score_call_evidence() -> dict[str, Any]:
        terminal, digest = score_session.finalize_score_calls(expected_score_calls, run_helper=score_run_helper)
        assert terminal["call_count"] == 4
        return {
            "path": str(receipt_root / "reasoning-score-call-index.json"),
            "schema": child_runtime.STRICT_GYM_SCORE_CALL_INDEX_SCHEMA,
            "sha256": digest,
        }

    documents = execute_captured_replay_cohort(
        manifest=manifest,
        replay_execution_manifest_sha256=replay_execution_manifest_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        authenticated_job_id=_JOB_ID,
        driver_process=process,
        driver_scheduler_device_environment=_device_environment(),
        source_transcript_document=authenticated_source.transcript_bundle,
        source_main_ledger_document=authenticated_source.main_ledger,
        transport_source=transport_source,
        post_verifier=post_verifier,
        independent_score_check=independent_score_check,
        finalize_score_call_evidence=finalize_score_call_evidence,
    )

    references = {"reasoning_score_call_index": copy.deepcopy(documents.reasoning_score_call_index)}
    for name, document in (
        ("transcript_bundle", documents.transcript_bundle),
        ("replay_ledger", documents.replay_ledger),
    ):
        declaration = manifest["artifacts"]["outputs"][name]
        path, digest = publish_evidence_document(
            output=declaration["path"],
            document=document,
            trailing_lf=False,
        )
        references[name] = {
            "path": str(path),
            "schema": declaration["schema"],
            "sha256": digest,
        }

    consumption_declaration = manifest["artifacts"]["outputs"]["transport_consumption"]
    consumption_path, consumption_sha = publish_strict_model_transport_replay_consumption(
        output=consumption_declaration["path"],
        document=documents.transport_consumption,
    )
    references["transport_consumption"] = {
        "path": str(consumption_path),
        "schema": consumption_declaration["schema"],
        "sha256": consumption_sha,
    }
    assert set(references) == set(evidence.REPLAY_OUTPUT_KEYS)
    return references


def test_lifecycle_r1_submission_and_pre_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, source, _, _, submission, submission_sha, pre, pre_sha = _submission_and_pre(tmp_path, monkeypatch)
    assert set(submission) == set(REPLAY_SUBMISSION_ROOT_KEYS)
    assert "authenticated_job_id" not in submission
    assert set(pre) == set(REPLAY_PRE_ROOT_KEYS)
    assert pre["candidate_job_id"] == pre["authenticated_job_id"] == _JOB_ID
    assert pre["static_boundary"] == {name: manifest[name] for name in evidence.REPLAY_STATIC_BOUNDARY_KEYS}
    assert pre["runtime_attestation_contract"] == manifest["runtime_attestation_requirements"]
    assert pre["output_precondition"] == {
        "path": manifest["artifacts"]["outputs"]["directory"]["path"],
        "mode": "0700",
        "status": "absent",
    }
    assert not Path(pre["output_precondition"]["path"]).exists()
    submission_path = Path(manifest["scheduler_submission"]["receipt"]["path"])
    loaded_submission, actual_submission_sha = load_captured_replay_submission_receipt(
        path=submission_path,
        expected_sha256=submission_sha,
        replay_execution_manifest=manifest,
        authenticated_source=source,
    )
    assert loaded_submission == submission
    assert actual_submission_sha == submission_sha
    pre_path = Path(pre["driver"]["pre_receipt_path"])
    loaded_pre, actual_pre_sha = load_captured_replay_pre_receipt(
        path=pre_path,
        expected_sha256=pre_sha,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        authenticated_source=source,
    )
    assert loaded_pre == pre
    assert actual_pre_sha == pre_sha
    for path in (submission_path, pre_path):
        metadata = os.lstat(path)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 1
        assert path.read_bytes().endswith(b"\n")


def test_scheduler_query_derives_record_from_raw_and_rejects_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    monkeypatch.setattr(evidence, "_load_bound_runtime_tool_bytes", lambda *_a, **_k: b"tool")
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _, manifest_sha = publish_replay_execution_manifest(
        output=control / "manifest.json", document=manifest, authenticated_source=source
    )
    comment = (
        f"nemo-rl-strict-captured-replay-v1:{manifest['attempt_id']}:"
        f"{manifest['scheduler_submission']['nonce']}:{manifest_sha}"
    )
    record = _record(manifest, comment=comment, phase="PRE_RELEASE")
    raw_path = Path(manifest["scheduler_submission"]["accepted_id_record"]["path"]).parent / "PRE_RELEASE.scontrol.raw"
    raw_sha = _seal_bytes(raw_path, _query_raw(record))
    query = build_captured_replay_scheduler_query(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        phase="PRE_RELEASE",
        raw_output_path=str(raw_path),
        raw_output_sha256=raw_sha,
        record=record,
    )
    assert set(query) == set(REPLAY_SCHEDULER_QUERY_ROOT_KEYS)
    assert query["records"] == [record]
    assert query["argv"] == [
        manifest["runtime_tools"]["document"]["host"]["scontrol"]["path"],
        "show",
        "job",
        "--json",
        _JOB_ID,
    ]
    assert query["normalization"] == {
        "algorithm": "scontrol-show-job-json-v1",
        "complete": True,
        "duplicate_keys_rejected": True,
        "negative_zero_rejected": True,
        "nonfinite_numbers_rejected": True,
    }
    query_path = raw_path.with_name("PRE_RELEASE.scontrol-query.json")
    _, query_sha = publish_captured_replay_scheduler_query(
        output=query_path,
        document=query,
        replay_execution_manifest=manifest,
        authenticated_source=source,
    )
    loaded, actual = load_captured_replay_scheduler_query(
        path=query_path,
        expected_sha256=query_sha,
        replay_execution_manifest=manifest,
        authenticated_source=source,
    )
    assert loaded == query
    assert actual == query_sha

    mismatched = copy.deepcopy(record)
    mismatched["job_id"] = "82002"
    with pytest.raises(ValueError, match="differs from normalized raw"):
        build_captured_replay_scheduler_query(
            replay_execution_manifest=manifest,
            authenticated_source=source,
            phase="PRE_RELEASE",
            raw_output_path=str(raw_path),
            raw_output_sha256=raw_sha,
            record=mismatched,
        )

    duplicate_path = raw_path.with_name("duplicate.scontrol.raw")
    duplicate_sha = _seal_bytes(duplicate_path, _query_raw(record, duplicate_job_id=True))
    with pytest.raises(ValueError, match="duplicate"):
        build_captured_replay_scheduler_query(
            replay_execution_manifest=manifest,
            authenticated_source=source,
            phase="PRE_RELEASE",
            raw_output_path=str(duplicate_path),
            raw_output_sha256=duplicate_sha,
            record=record,
        )

    canonical_raw = _query_raw(record)
    decoded = json.loads(canonical_raw)
    assert canonical_raw == (
        json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert set(decoded) == {
        "errors",
        "jobs",
        "last_backfill",
        "last_update",
        "meta",
        "warnings",
    }
    assert decoded["errors"] == decoded["warnings"] == []
    assert len(decoded["jobs"]) == 1
    raw_job = decoded["jobs"][0]
    assert set(raw_job) == {
        "comment",
        "current_working_directory",
        "hold",
        "job_id",
        "job_state",
        "name",
        "restart_cnt",
        "state_reason",
        "user_id",
    }
    assert type(raw_job["job_id"]) is int
    assert type(raw_job["user_id"]) is int
    assert type(raw_job["restart_cnt"]) is int
    assert type(raw_job["hold"]) is bool
    assert raw_job["job_state"] == [record["job_state"]]

    for filename, poisoned_raw in (
        (
            "nonfinite.scontrol.raw",
            canonical_raw.replace(b'"restart_cnt":0', b'"restart_cnt":NaN', 1),
        ),
        (
            "negative-zero.scontrol.raw",
            canonical_raw.replace(b'"restart_cnt":0', b'"restart_cnt":-0', 1),
        ),
    ):
        poisoned_path = raw_path.with_name(filename)
        poisoned_sha = _seal_bytes(poisoned_path, poisoned_raw)
        with pytest.raises(ValueError):
            build_captured_replay_scheduler_query(
                replay_execution_manifest=manifest,
                authenticated_source=source,
                phase="PRE_RELEASE",
                raw_output_path=str(poisoned_path),
                raw_output_sha256=poisoned_sha,
                record=record,
            )

    for field in ("last_backfill", "last_update", "meta"):
        for index, invalid_value in enumerate((None, [], 0, False, "invalid")):
            poisoned_document = copy.deepcopy(decoded)
            poisoned_document[field] = invalid_value
            poisoned_raw = (
                json.dumps(
                    poisoned_document,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            poisoned_path = raw_path.with_name(f"{field}-{index}.scontrol.raw")
            poisoned_sha = _seal_bytes(poisoned_path, poisoned_raw)
            with pytest.raises(TypeError, match=rf"scheduler raw JSON {field} must be an exact object"):
                build_captured_replay_scheduler_query(
                    replay_execution_manifest=manifest,
                    authenticated_source=source,
                    phase="PRE_RELEASE",
                    raw_output_path=str(poisoned_path),
                    raw_output_sha256=poisoned_sha,
                    record=record,
                )

    with pytest.raises(ValueError, match="PRE scheduler record"):
        build_captured_replay_scheduler_query(
            replay_execution_manifest=manifest,
            authenticated_source=source,
            phase="PRE",
            raw_output_path=str(raw_path),
            raw_output_sha256=raw_sha,
            record=record,
        )

    post_record = _record(manifest, comment=comment, phase="POST")
    results_root = Path(manifest["pair"]["manifest"]["path"]).parent
    post_raw_parent = (
        results_root
        / "captured_replay/replay_job_state"
        / manifest["pair_id"]
        / manifest["attempt_id"]
        / f"{_JOB_ID}-0/queries"
    )
    for index, invalid_reason in enumerate((123, "\x1f", "\x7f")):
        poisoned_record = copy.deepcopy(post_record)
        poisoned_record["reason"] = invalid_reason
        poisoned_raw = _query_raw(poisoned_record)
        if invalid_reason == "\x7f":
            poisoned_raw = poisoned_raw.replace(b"\x7f", b"\\u007f")
        if invalid_reason in {"\x1f", "\x7f"}:
            assert b"\\u00" in poisoned_raw
        poisoned_path = post_raw_parent / f"POST-reason-{index}.scontrol.raw"
        poisoned_sha = _seal_bytes(poisoned_path, poisoned_raw)
        with pytest.raises((TypeError, ValueError), match="scheduler raw state_reason"):
            build_captured_replay_scheduler_query(
                replay_execution_manifest=manifest,
                authenticated_source=source,
                phase="POST",
                raw_output_path=str(poisoned_path),
                raw_output_sha256=poisoned_sha,
                record=poisoned_record,
            )


@pytest.mark.parametrize("pair_job_id", _PAIR_JOB_IDS)
def test_submission_rejects_coherent_pair_job_id_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pair_job_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_job_id reuses an authenticated Pair OFF/ON job ID",
    ):
        _submission_and_pre(
            tmp_path,
            monkeypatch,
            replay_job_id=pair_job_id,
        )


def test_submission_and_pre_reject_identity_and_exact_key_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source, _, _, submission, _, pre, _ = _submission_and_pre(tmp_path, monkeypatch)
    changed_submission = copy.deepcopy(submission)
    changed_submission["authenticated_job_id"] = _JOB_ID
    with pytest.raises(ValueError, match="keyset mismatch"):
        validate_captured_replay_submission_receipt(
            changed_submission,
            replay_execution_manifest=manifest,
            authenticated_source=source,
        )
    changed_submission = copy.deepcopy(submission)
    changed_submission["sbatch"]["argv"][0:2] = ["--hold", "--parsable"]
    with pytest.raises(ValueError, match="authoritative replay ordering"):
        validate_captured_replay_submission_receipt(
            changed_submission,
            replay_execution_manifest=manifest,
            authenticated_source=source,
        )
    authoritative_argv = submission["sbatch"]["argv"]
    pair_flags = (
        "--pair-manifest",
        "--pair-manifest-sha256",
        "--pair-submission-receipt",
        "--pair-submission-receipt-sha256",
        "--off-exit-receipt",
        "--off-exit-receipt-sha256",
    )
    poisoned_pair_anchors: list[list[str]] = []
    for flag in pair_flags:
        argv = list(authoritative_argv)
        offset = argv.index(flag)
        del argv[offset : offset + 2]
        poisoned_pair_anchors.append(argv)
    for flag in (
        "--pair-manifest",
        "--pair-submission-receipt",
        "--off-exit-receipt",
    ):
        argv = list(authoritative_argv)
        argv[argv.index(flag) + 1] = "/forged/pair-authority.json"
        poisoned_pair_anchors.append(argv)
    for flag in (
        "--pair-manifest-sha256",
        "--pair-submission-receipt-sha256",
        "--off-exit-receipt-sha256",
    ):
        argv = list(authoritative_argv)
        argv[argv.index(flag) + 1] = "f" * 64
        poisoned_pair_anchors.append(argv)
    argv = list(authoritative_argv)
    start = argv.index("--pair-manifest")
    anchors = argv[start : start + 8]
    argv[start : start + 8] = anchors[4:] + anchors[:4]
    poisoned_pair_anchors.append(argv)
    for argv in poisoned_pair_anchors:
        changed_submission = copy.deepcopy(submission)
        changed_submission["sbatch"]["argv"] = argv
        with pytest.raises(ValueError, match="authoritative replay ordering"):
            validate_captured_replay_submission_receipt(
                changed_submission,
                replay_execution_manifest=manifest,
                authenticated_source=source,
            )
    changed_pre = copy.deepcopy(pre)
    changed_pre["authenticated_job_id"] = "82002"
    with pytest.raises(ValueError, match="identity differs|job IDs differ"):
        validate_captured_replay_pre_receipt(
            changed_pre,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            authenticated_source=source,
        )

    for pair_job_id in _PAIR_JOB_IDS:
        with pytest.raises(
            ValueError,
            match="authenticated_job_id reuses an authenticated Pair OFF/ON job ID",
        ):
            build_captured_replay_pre_receipt(
                replay_execution_manifest=manifest,
                authenticated_source=source,
                submission_receipt=submission,
                authenticated_job_id=pair_job_id,
                job=pre["job"],
                pre_scheduler_query=pre["pre_scheduler_query"],
            )
    changed_pre = copy.deepcopy(pre)
    changed_pre["static_boundary"]["source"]["head"] = "f" * 40
    with pytest.raises(ValueError, match="static boundary differs"):
        validate_captured_replay_pre_receipt(
            changed_pre,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            authenticated_source=source,
        )


def test_lifecycle_r1_exit_and_post_index_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, source, _, manifest_sha, submission, submission_sha, pre, _ = _submission_and_pre(tmp_path, monkeypatch)
    assert manifest["wandb"] == {
        "enabled": False,
        "mode": "disabled",
        "reason": "scorer-only-replay-no-wandb-credentials-or-output",
    }
    job_root = Path(pre["driver"]["pre_receipt_path"]).parents[1]
    post_record = _record(manifest, comment=submission["comment"], phase="POST")
    _, post_query_ref = _seal_query(
        manifest,
        authenticated_source=source,
        phase="POST",
        raw_path=job_root / "queries/POST.scontrol.raw",
        record=post_record,
    )
    process = {
        "boot_id_sha256": _digest("replay-driver-boot"),
        "pid": 41001,
        "start_time_ticks": 51001,
    }
    outputs = _publish_real_terminal_outputs(
        manifest=manifest,
        authenticated_source=source,
        replay_execution_manifest_sha256=manifest_sha,
        submission_receipt_sha256=submission_sha,
        process=process,
        monkeypatch=monkeypatch,
    )
    score_index = json.loads(Path(outputs["reasoning_score_call_index"]["path"]).read_bytes())
    resource_receipt = json.loads(Path(score_index["resource_receipt"]["path"]).read_bytes())
    scorer_runtime = manifest["runtime_attestation_requirements"]["resource_scorer_child"]
    assert resource_receipt["distribution_versions"] == {
        **scorer_runtime["required_common_distributions"],
        "reasoning-gym": "0.1.25",
    }
    assert resource_receipt["module_versions"] == {
        **scorer_runtime["required_module_versions"],
        "reasoning_gym": "0.1.19",
    }
    exit_receipt = build_captured_replay_exit_receipt(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        submission_receipt=submission,
        pre_receipt=pre,
        post_scheduler_query=post_query_ref,
        driver_exit_code=0,
        hardware=_hardware_observation(manifest["runtime_tools"]["document"]["host"]["nvidia_smi"]),
        scheduler_device_environment=_device_environment(),
        driver_scheduler_device_environment=_device_environment(),
        driver_process=process,
        outputs=outputs,
    )
    assert set(exit_receipt) == set(REPLAY_EXIT_ROOT_KEYS)
    assert len(REPLAY_EXIT_ROOT_KEYS) == 24
    assert exit_receipt["schema"] == ("nemo-rl-strict-captured-replay-job-exit-receipt-v5")
    assert exit_receipt["hardware"]["schema"] == ("nemo-rl-strict-hardware-observation-v2")
    assert len(exit_receipt["hardware"]) == 8
    assert exit_receipt["driver_scheduler_device_environment"] == (exit_receipt["scheduler_device_environment"])
    assert exit_receipt["outputs"] == outputs
    assert exit_receipt["runtime_attestation"] == {
        "schema": evidence.REPLAY_RUNTIME_ATTESTATION_SCHEMA,
        "requirements": manifest["runtime_attestation_requirements"],
        **outputs,
    }
    exit_path, exit_sha = publish_captured_replay_exit_receipt(
        output=job_root / "receipts/EXIT.json",
        document=exit_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        authenticated_source=source,
    )
    loaded_exit, actual_exit_sha = load_captured_replay_exit_receipt(
        path=exit_path,
        expected_sha256=exit_sha,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        authenticated_source=source,
    )
    assert loaded_exit == exit_receipt
    assert actual_exit_sha == exit_sha
    assert exit_path.read_bytes().endswith(b"\n")

    post_index = build_captured_replay_evidence_index(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
    )
    assert set(post_index) == set(REPLAY_POST_INDEX_ROOT_KEYS)
    assert post_index["identity"] == {
        "candidate_job_id": _JOB_ID,
        "authenticated_job_id": _JOB_ID,
        "driver_process": process,
        "run_id": replay_run_id(
            environment=manifest["environment"],
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
    }
    index_path, index_sha = publish_captured_replay_evidence_index(
        output=manifest["artifacts"]["outputs"]["evidence_index"]["path"],
        document=post_index,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        authenticated_source=source,
    )
    loaded_index, actual_index_sha = load_captured_replay_evidence_index(
        path=index_path,
        expected_sha256=index_sha,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        authenticated_source=source,
    )
    assert loaded_index == post_index
    assert actual_index_sha == index_sha
    assert not index_path.read_bytes().endswith(b"\n")
    for path in (
        exit_path,
        index_path,
        *(Path(ref["path"]) for ref in outputs.values()),
    ):
        metadata = os.lstat(path)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 1

    changed_exit = copy.deepcopy(exit_receipt)
    changed_exit["runtime_attestation"]["unexpected"] = True
    with pytest.raises(ValueError, match="keyset mismatch"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_exit = copy.deepcopy(exit_receipt)
    changed_exit["outputs"]["transport_consumption"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256|bytes differ"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_exit = copy.deepcopy(exit_receipt)
    changed_exit["driver_scheduler_device_environment"]["cuda_visible_devices"] = "4,5,6,7"
    changed_exit["driver_scheduler_device_environment"]["gpu_device_ordinal"] = "4,5,6,7"
    with pytest.raises(ValueError, match="differs from wrapper observation"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_exit = copy.deepcopy(exit_receipt)
    for row in changed_exit["hardware"]["ordered_rows"]:
        row["raw"] = "NVIDIA A100, 580.126.20"
        row["gpu_model"] = "NVIDIA A100"
    changed_exit["hardware"]["gpu_model"] = "NVIDIA A100"
    changed_exit["hardware"]["raw_output_sha256"] = hashlib.sha256(
        ("NVIDIA A100, 580.126.20\n" * 4).encode("ascii")
    ).hexdigest()
    changed_exit["hardware"]["ordered_rows_sha256"] = evidence.domain_sha256(
        evidence.HARDWARE_ORDERED_ROWS_HASH_LABEL,
        changed_exit["hardware"]["ordered_rows"],
    )
    with pytest.raises(ValueError, match="requires NVIDIA GB200"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_exit = copy.deepcopy(exit_receipt)
    for row in changed_exit["hardware"]["ordered_rows"]:
        row["raw"] = "NVIDIA GB200 ,580.126.20"
    changed_exit["hardware"]["raw_output_sha256"] = hashlib.sha256(
        ("NVIDIA GB200 ,580.126.20\n" * 4).encode("ascii")
    ).hexdigest()
    changed_exit["hardware"]["ordered_rows_sha256"] = evidence.domain_sha256(
        evidence.HARDWARE_ORDERED_ROWS_HASH_LABEL,
        changed_exit["hardware"]["ordered_rows"],
    )
    with pytest.raises(ValueError, match="raw text differs"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    consumption_path = Path(outputs["transport_consumption"]["path"])
    original_consumption_bytes = consumption_path.read_bytes()
    forged_consumption = json.loads(original_consumption_bytes)
    forged_consumption["replay"]["scheduler_device_environment"].update(
        cuda_visible_devices="4,5,6,7",
        gpu_device_ordinal="4,5,6,7",
    )
    forged_consumption_bytes = evidence.canonical_ascii_json(forged_consumption)
    consumption_path.chmod(0o600)
    consumption_path.write_bytes(forged_consumption_bytes)
    consumption_path.chmod(0o400)
    try:
        changed_exit = copy.deepcopy(exit_receipt)
        changed_exit["outputs"]["transport_consumption"]["sha256"] = hashlib.sha256(
            forged_consumption_bytes
        ).hexdigest()
        with pytest.raises(ValueError, match="transport replay.*differs"):
            validate_captured_replay_exit_receipt(
                changed_exit,
                replay_execution_manifest=manifest,
                submission_receipt=submission,
                pre_receipt=pre,
                authenticated_source=source,
            )
    finally:
        consumption_path.chmod(0o600)
        consumption_path.write_bytes(original_consumption_bytes)
        consumption_path.chmod(0o400)
    changed_exit = copy.deepcopy(exit_receipt)
    changed_exit["hardware"]["nvidia_smi"]["path"] = str(
        Path(changed_exit["hardware"]["nvidia_smi"]["path"]).with_name("forged-nvidia-smi")
    )
    with pytest.raises(ValueError, match="differs from authenticated runtime tool"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_exit = copy.deepcopy(exit_receipt)
    changed_exit["hardware"]["nvidia_smi"]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="differs from authenticated runtime tool"):
        validate_captured_replay_exit_receipt(
            changed_exit,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )
    changed_index = copy.deepcopy(post_index)
    changed_index["identity"]["authenticated_job_id"] = "82002"
    with pytest.raises(ValueError, match="identity differs"):
        validate_captured_replay_evidence_index(
            changed_index,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            exit_receipt=exit_receipt,
            authenticated_source=source,
        )

    changed_index = copy.deepcopy(post_index)
    changed_index["identity"]["driver_process"]["start_time_ticks"] += 1
    with pytest.raises(ValueError, match="identity differs"):
        validate_captured_replay_evidence_index(
            changed_index,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            exit_receipt=exit_receipt,
            authenticated_source=source,
        )

    changed_index = copy.deepcopy(post_index)
    changed_index["exit_receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exit_receipt differs"):
        validate_captured_replay_evidence_index(
            changed_index,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            exit_receipt=exit_receipt,
            authenticated_source=source,
        )

    nvidia_smi_path = Path(exit_receipt["hardware"]["nvidia_smi"]["path"])
    nvidia_smi_path.chmod(0o700)
    nvidia_smi_path.write_bytes(b"forged-tool-nvidia-smi")
    nvidia_smi_path.chmod(0o500)
    with pytest.raises(ValueError, match="runtime tool bytes differ"):
        validate_captured_replay_exit_receipt(
            exit_receipt,
            replay_execution_manifest=manifest,
            submission_receipt=submission,
            pre_receipt=pre,
            authenticated_source=source,
        )


def test_scheduler_query_rejects_sha_and_named_path_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _, source = _manifest(str(tmp_path))
    monkeypatch.setattr(evidence, "_load_bound_runtime_tool_bytes", lambda *_a, **_k: b"tool")
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _, manifest_sha = publish_replay_execution_manifest(
        output=control / "manifest.json", document=manifest, authenticated_source=source
    )
    comment = (
        f"nemo-rl-strict-captured-replay-v1:{manifest['attempt_id']}:"
        f"{manifest['scheduler_submission']['nonce']}:{manifest_sha}"
    )
    record = _record(manifest, comment=comment, phase="PRE_RELEASE")
    raw_path = Path(manifest["scheduler_submission"]["accepted_id_record"]["path"]).parent / "PRE_RELEASE.scontrol.raw"
    raw_sha = _seal_bytes(raw_path, _query_raw(record))
    with pytest.raises(ValueError, match="raw evidence differs"):
        build_captured_replay_scheduler_query(
            replay_execution_manifest=manifest,
            authenticated_source=source,
            phase="PRE_RELEASE",
            raw_output_path=str(raw_path),
            raw_output_sha256="1" * 64,
            record=record,
        )

    real_stat = evidence.os.stat

    def drifting_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        metadata = real_stat(path, *args, **kwargs)
        if path != raw_path.name or kwargs.get("dir_fd") is None:
            return metadata
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_nlink=metadata.st_nlink,
            st_uid=metadata.st_uid,
            st_gid=metadata.st_gid,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns + 1,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    monkeypatch.setattr(evidence.os, "stat", drifting_stat)
    with pytest.raises(RuntimeError, match="changed during stable read"):
        evidence._load_lifecycle_raw_bytes(raw_path, expected_sha256=raw_sha, maximum=1024 * 1024)


def test_authoritative_exports_exclude_legacy_lifecycle_names() -> None:
    lifecycle_prefixes = (
        "build_captured_replay_",
        "load_captured_replay_",
        "publish_captured_replay_",
        "validate_captured_replay_",
    )
    assert all(name.startswith(lifecycle_prefixes) or name.startswith("REPLAY_") for name in evidence.__all__)
    for name in (
        "build_replay_submission_receipt",
        "validate_replay_submission_receipt",
        "build_replay_job_exit_receipt",
        "validate_replay_job_exit_receipt",
        "build_same_arm_replay_index",
        "validate_same_arm_replay_index",
        "build_cross_arm_parity_index",
        "validate_cross_arm_parity_index",
        "_legacy_build_replay_submission_receipt",
        "_legacy_validate_replay_submission_receipt",
    ):
        assert not hasattr(evidence, name)
