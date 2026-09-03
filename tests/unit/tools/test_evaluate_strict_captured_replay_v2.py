from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_PATH = ROOT / "examples/nemo_gym/nemotron-3.5-nano" / "evaluate_strict_captured_replay_v2.py"


def _load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "strict_captured_replay_v2_evaluator_tests",
        EVALUATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = _load_evaluator()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _ref(label: str) -> dict[str, str]:
    return {"path": f"/authority/{label}", "sha256": _sha(label)}


def _artifact(path: str, schema: str, label: str) -> dict[str, str]:
    return {"path": path, "schema": schema, "sha256": _sha(label)}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _rehash_request(request: dict[str, Any]) -> None:
    projection = dict(request)
    projection.pop("request_sha256")
    request["request_sha256"] = hashlib.sha256(EVALUATOR.REQUEST_HASH_DOMAIN + _canonical(projection)).hexdigest()


def _samples(environment: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(4):
        if environment == "citation":
            details: dict[str, Any] = {
                "expected": [f"[{index + 1}]"],
                "missing": [],
                "spurious": [],
                "passed": True,
            }
        else:
            details = {
                "matching_lines": 2,
                "min_matches": 2,
                "passed": True,
            }
        result.append(
            {
                "sample_index": index,
                "fixture_row_index": 0,
                "rollout_index": index,
                "generation_seed": index + 100,
                "model_transport_entry_sha256": _sha(f"entry-{index}"),
                "model_transport_request_body_sha256": _sha(f"request-{index}"),
                "model_transport_response_body_sha256": _sha(f"response-{index}"),
                "model_response_sha256": _sha(f"model-response-{index}"),
                "match_details": details,
                "raw_environment_reward": 1.0,
            }
        )
    return result


def _request(environment: str = "citation") -> dict[str, Any]:
    profile_id = EVALUATOR.PROFILE_BY_ENVIRONMENT[environment]
    pair_id = "pair-abc"
    results_root = "/results"
    request = {
        "schema": EVALUATOR.REQUEST_SCHEMA,
        "nonce": _sha("nonce"),
        "request_sha256": "0" * 64,
        "evaluator_program": _ref("evaluator-program"),
        "pair": {
            "pair_id": pair_id,
            "environment": environment,
            "profile_id": profile_id,
            "manifest": _ref("pair-manifest"),
            "submission_receipt": _ref("pair-submission"),
            "off_exit_receipt": _ref("off-exit"),
        },
        "attempts": {},
    }
    for ordinal, attempt in enumerate(EVALUATOR.ATTEMPT_NAMES, start=1):
        job_id = str(8000 + ordinal)
        receipt_root = f"{results_root}/captured_replay/replay_job_state/{pair_id}/" f"{attempt}/{job_id}-0/receipts"
        request["attempts"][attempt] = {
            "replay_execution_manifest": {
                "path": f"{results_root}/captured_replay/manifests/{pair_id}/{attempt}.json",
                "sha256": _sha(f"{attempt}-manifest"),
            },
            "submission_receipt_sha256": _sha(f"{attempt}-submission"),
            "candidate_job_id": job_id,
            "result_final_receipt": {
                "path": f"{receipt_root}/FINAL.json",
                "sha256": _sha(f"{attempt}-final"),
            },
        }
    _rehash_request(request)
    return request


def _snapshot(request: dict[str, Any], attempt: str) -> dict[str, Any]:
    pair = request["pair"]
    item = request["attempts"][attempt]
    result_root = f"/results/captured_replay/{attempt}"
    receipt_root = (
        f"/results/captured_replay/replay_job_state/{pair['pair_id']}/"
        f"{attempt}/{item['candidate_job_id']}-0/receipts"
    )
    number = 1 if attempt == "replay-1" else 2
    boot_id = f"12345678-1234-1234-1234-1234567890a{number}"
    return {
        "schema": EVALUATOR.SNAPSHOT_SCHEMA,
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "profile_id": pair["profile_id"],
        "attempt_id": attempt,
        "candidate_job_id": item["candidate_job_id"],
        "authenticated_job_id": item["candidate_job_id"],
        "run_id": hashlib.sha256(
            f"nemo-rl-strict-replay-v2:{pair['environment']}:{pair['pair_id']}:{attempt}".encode("ascii")
        ).hexdigest(),
        "driver_process": {
            "boot_id_sha256": hashlib.sha256((boot_id + "\n").encode("ascii")).hexdigest(),
            "pid": 1000 + number,
            "start_time_ticks": 2000 + number,
        },
        "scorer_process_identity": {
            "boot_id": boot_id,
            "hostname": f"node-{number}",
            "pid": 3000 + number,
            "start_ticks": 4000 + number,
        },
        "manifest": _artifact(
            item["replay_execution_manifest"]["path"],
            EVALUATOR.MANIFEST_SCHEMA,
            f"{attempt}-manifest",
        ),
        "submission_receipt": _artifact(
            f"/results/captured_replay/replay_submission_state/{pair['pair_id']}/" f"{attempt}/submission-receipt.json",
            EVALUATOR.SUBMISSION_SCHEMA,
            f"{attempt}-submission",
        ),
        "pre_receipt": _artifact(
            f"{receipt_root}/PRE.json",
            EVALUATOR.PRE_SCHEMA,
            f"{attempt}-pre",
        ),
        "exit_receipt": _artifact(
            f"{receipt_root}/EXIT.json",
            EVALUATOR.EXIT_SCHEMA,
            f"{attempt}-exit",
        ),
        "result_final_receipt": _artifact(
            f"{receipt_root}/FINAL.json",
            EVALUATOR.FINAL_SCHEMA,
            f"{attempt}-final",
        ),
        "result_root": result_root,
        "result_inventory": _artifact(
            f"{result_root}/result-inventory-v2.json",
            EVALUATOR.INVENTORY_SCHEMA,
            f"{attempt}-inventory",
        ),
        "evidence_index": _artifact(
            f"{result_root}/evidence-index.json",
            EVALUATOR.INDEX_SCHEMA,
            f"{attempt}-index",
        ),
        "outputs": {
            name: _artifact(
                f"{result_root}/{EVALUATOR.OUTPUT_PATHS[name]}",
                schema,
                f"{attempt}-{name}",
            )
            for name, schema in EVALUATOR.OUTPUT_SCHEMAS.items()
        },
        "samples": _samples(pair["environment"]),
    }


class _Consumed:
    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.__snapshot = snapshot

    @property
    def snapshot(self) -> dict[str, Any]:
        return self.__snapshot


def _api(request: dict[str, Any], snapshots: dict[str, dict[str, Any]] | None = None) -> tuple[Any, Any, Any]:
    material = snapshots or {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}

    def consume(**kwargs: Any) -> _Consumed:
        attempt = next(
            name
            for name in EVALUATOR.ATTEMPT_NAMES
            if request["attempts"][name]["candidate_job_id"] == kwargs["candidate_job_id"]
        )
        expected = request["attempts"][attempt]
        assert kwargs == {
            "pair_manifest_path": request["pair"]["manifest"]["path"],
            "pair_manifest_sha256": request["pair"]["manifest"]["sha256"],
            "pair_submission_receipt_path": request["pair"]["submission_receipt"]["path"],
            "pair_submission_receipt_sha256": request["pair"]["submission_receipt"]["sha256"],
            "trusted_off_exit_receipt_path": request["pair"]["off_exit_receipt"]["path"],
            "trusted_off_exit_receipt_sha256": request["pair"]["off_exit_receipt"]["sha256"],
            "replay_manifest_path": expected["replay_execution_manifest"]["path"],
            "replay_manifest_sha256": expected["replay_execution_manifest"]["sha256"],
            "submission_receipt_sha256": expected["submission_receipt_sha256"],
            "candidate_job_id": expected["candidate_job_id"],
            "result_final_receipt_path": expected["result_final_receipt"]["path"],
            "result_final_receipt_sha256": expected["result_final_receipt"]["sha256"],
            "expected_environment": request["pair"]["environment"],
            "expected_profile_id": request["pair"]["profile_id"],
        }
        return _Consumed(copy.deepcopy(material[attempt]))

    return (_Consumed, consume, vars(_Consumed)["snapshot"])


@pytest.mark.parametrize("environment", ("citation", "freeform"))
def test_accepts_exact_profile_bound_dual_attempts(environment: str) -> None:
    request = _request(environment)

    report = EVALUATOR.evaluate_authenticated_request(
        request,
        evaluator_program=request["evaluator_program"],
        coordinator_api=_api(request),
    )

    assert report["schema"] == EVALUATOR.REPORT_SCHEMA
    assert report["status"] == "authenticated"
    assert report["pair"]["environment"] == environment
    assert report["parity"]["status"] == "exact-match"
    assert report["parity"]["reward_vector"] == [1.0] * 4
    assert set(report["attempts"]) == set(EVALUATOR.ATTEMPT_NAMES)


def test_canonical_request_round_trip_and_hash() -> None:
    request = _request()
    raw = _canonical(request)

    assert EVALUATOR._validate_request(EVALUATOR._parse_request(raw)) == request
    assert EVALUATOR._request_sha256(request) == request["request_sha256"]


def test_request_grammar_matches_producer_safe_id_and_canonical_unicode_paths() -> None:
    request = _request()
    request["pair"]["pair_id"] = "_pair"
    request["pair"]["manifest"]["path"] = "/authority/réplay/pair-manifest"
    _rehash_request(request)

    assert EVALUATOR._validate_request(request) == request


@pytest.mark.parametrize(
    "raw",
    (
        b'{"a":1,"a":2}',
        b'{"value":NaN}',
        b'{"value":-0}',
        b' {"value":0}',
        b'{"value":0}\n',
        b'{"value":"\xc3\xa9"}',
    ),
)
def test_rejects_noncanonical_request_frames(raw: bytes) -> None:
    with pytest.raises(EVALUATOR.ReplayV2EvaluationError):
        EVALUATOR._parse_request(raw)


@pytest.mark.parametrize(
    ("environment", "profile"),
    (
        ("reasoning_gym", "reasoning-gym-v1"),
        ("citation", "freeform-regex-v1"),
        ("freeform", "citation-string-match-v1"),
    ),
)
def test_rejects_unsupported_or_crossed_profile(environment: str, profile: str) -> None:
    request = _request()
    request["pair"]["environment"] = environment
    request["pair"]["profile_id"] = profile
    _rehash_request(request)

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="dispatch"):
        EVALUATOR._validate_request(request)


def test_rejects_old_attempt_authority_or_extra_claims() -> None:
    request = _request()
    item = request["attempts"]["replay-1"]
    item["evidence_index"] = _ref("old-index")
    _rehash_request(request)

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="key set"):
        EVALUATOR._validate_request(request)


@pytest.mark.parametrize("member", ("path", "sha256"))
def test_rejects_relabelled_pair_authority(member: str) -> None:
    request = _request()
    request["pair"]["off_exit_receipt"][member] = request["pair"]["manifest"][member]
    _rehash_request(request)

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="Pair authority"):
        EVALUATOR._validate_request(request)


@pytest.mark.parametrize(
    "field",
    (
        "replay_execution_manifest",
        "submission_receipt_sha256",
        "result_final_receipt",
    ),
)
def test_rejects_reused_request_attempt_authority(field: str) -> None:
    request = _request()
    first = request["attempts"]["replay-1"][field]
    request["attempts"]["replay-2"][field] = copy.deepcopy(first)
    _rehash_request(request)

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="distinct"):
        EVALUATOR._validate_request(request)


def test_rejects_self_selected_program_authority() -> None:
    request = _request()
    authenticated = _ref("other-program")

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="deployment"):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=authenticated,
            coordinator_api=_api(request),
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda snapshots: snapshots["replay-1"].update(extra=True),
        lambda snapshots: snapshots["replay-1"].update(authenticated_job_id="9001"),
        lambda snapshots: snapshots["replay-1"]["manifest"].update(
            schema="nemo-rl-strict-captured-replay-execution-manifest-v3"
        ),
        lambda snapshots: snapshots["replay-1"]["samples"][0].update(generation_seed=True),
        lambda snapshots: snapshots["replay-1"]["samples"][0].update(raw_environment_reward=1),
    ),
)
def test_rejects_nonexact_snapshot_types_and_schemas(mutate: Any) -> None:
    request = _request()
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    mutate(snapshots)

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


@pytest.mark.parametrize(
    "field",
    (
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_root",
        "result_inventory",
        "evidence_index",
        "outputs",
    ),
)
def test_rejects_reused_attempt_authority(field: str) -> None:
    request = _request()
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    snapshots["replay-2"][field] = copy.deepcopy(snapshots["replay-1"][field])

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


@pytest.mark.parametrize(
    "field",
    (
        "pre_receipt",
        "exit_receipt",
        "result_inventory",
        "evidence_index",
    ),
)
def test_rejects_relabelled_attempt_artifact_digest(field: str) -> None:
    request = _request()
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    snapshots["replay-2"][field]["sha256"] = snapshots["replay-1"][field]["sha256"]

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match=rf"{field}\.sha256"):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


@pytest.mark.parametrize("output_name", tuple(EVALUATOR.OUTPUT_SCHEMAS))
def test_rejects_relabelled_attempt_output_digest(output_name: str) -> None:
    request = _request()
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    snapshots["replay-2"]["outputs"][output_name]["sha256"] = snapshots["replay-1"]["outputs"][output_name]["sha256"]

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match=output_name):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


def test_rejects_authenticated_sample_parity_mismatch() -> None:
    request = _request("freeform")
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    second = snapshots["replay-2"]["samples"][3]
    second["match_details"] = {
        "matching_lines": 1,
        "min_matches": 2,
        "passed": False,
    }
    second["raw_environment_reward"] = 0.0

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="sample evidence differs"):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


def test_rejects_invalid_profile_match_details_even_when_both_attempts_match() -> None:
    request = _request("citation")
    snapshots = {attempt: _snapshot(request, attempt) for attempt in EVALUATOR.ATTEMPT_NAMES}
    for snapshot in snapshots.values():
        snapshot["samples"][0]["match_details"] = {
            "expected": ["[1]"],
            "missing": ["[1]"],
            "spurious": [],
            "passed": True,
        }

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="passed differs"):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=_api(request, snapshots),
        )


def test_rejects_fake_or_missing_coordinator_api() -> None:
    request = _request()

    with pytest.raises(EVALUATOR.ReplayV2EvaluationError, match="authority"):
        EVALUATOR.evaluate_authenticated_request(
            request,
            evaluator_program=request["evaluator_program"],
            coordinator_api=(object(), lambda **_: object(), None),
        )


def test_main_rejects_when_authenticated_bootstrap_authority_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports: list[dict[str, Any]] = []
    monkeypatch.setattr(EVALUATOR, "_validate_runtime", lambda: None)
    monkeypatch.setattr(EVALUATOR, "_write_report", reports.append)
    monkeypatch.setattr(EVALUATOR, "_BOOTSTRAP_PROGRAM_REFERENCE", None)
    monkeypatch.setattr(EVALUATOR, "_BOOTSTRAP_COORDINATOR_API", None)

    assert EVALUATOR.main() == 1
    assert reports == [
        {
            "schema": EVALUATOR.REPORT_SCHEMA,
            "status": "rejected",
            "nonce": None,
            "request_sha256": None,
            "error": {
                "code": "missing_authority",
                "message": "authenticated evaluator bootstrap authority is absent",
            },
        }
    ]


def test_main_accepts_only_injected_program_and_coordinator_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("freeform")
    reports: list[dict[str, Any]] = []
    monkeypatch.setattr(EVALUATOR, "_validate_runtime", lambda: None)
    monkeypatch.setattr(EVALUATOR, "_read_request", lambda: _canonical(request))
    monkeypatch.setattr(EVALUATOR, "_write_report", reports.append)
    monkeypatch.setattr(
        EVALUATOR,
        "_BOOTSTRAP_PROGRAM_REFERENCE",
        copy.deepcopy(request["evaluator_program"]),
    )
    monkeypatch.setattr(EVALUATOR, "_BOOTSTRAP_COORDINATOR_API", _api(request))

    assert EVALUATOR.main() == 0
    assert len(reports) == 1
    assert reports[0]["status"] == "authenticated"
    assert reports[0]["pair"]["profile_id"] == "freeform-regex-v1"
