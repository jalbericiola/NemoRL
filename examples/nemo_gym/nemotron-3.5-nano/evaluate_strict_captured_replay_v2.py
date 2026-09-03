#!/usr/bin/env python3
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

"""Fresh-process acceptance evaluator for two strict replay V2 results.

This source is never a login-process trust bootstrap.  A release stager must
first authenticate the separately signed evaluator program manifest and then
execute these exact bytes under an isolated interpreter.  That bootstrap
injects the authenticated program reference and the exact coordinator
``ConsumedReplayResult`` type, ``consume_replay_result`` callable, and
``snapshot`` descriptor.  No NeMoRL module is imported by this file.

The injected coordinator seam owns all filesystem and producer lifecycle
authentication, including FINAL-before-result authority and the retained
thirteen-file sealed-result byte capability.  This evaluator independently
validates the detached snapshots, exact citation/freeform/reasoning profile semantics,
and the required distinctions and parity between replay-1 and replay-2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import sys
from typing import Any

REQUEST_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-request-v1"
REPORT_SCHEMA = "nemo-rl-strict-captured-replay-v2-evaluator-report-v1"
SNAPSHOT_SCHEMA = "nemo-rl-strict-captured-replay-authenticated-result-snapshot-v2"
REQUEST_HASH_DOMAIN = b"nemo-rl-strict-captured-replay-v2-evaluator-request-v1\0"
PARITY_HASH_DOMAIN = b"nemo-rl-strict-captured-replay-v2-evaluator-parity-v1\0"
MAX_REQUEST_BYTES = 64 * 1024
MAX_REPORT_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1 << 20

ATTEMPT_NAMES = ("replay-1", "replay-2")
PROFILE_BY_ENVIRONMENT = {
    "citation": "citation-string-match-v1",
    "freeform": "freeform-regex-v1",
    "reasoning_gym": "reasoning-gym-exact-match-v1",
}
EXACT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}

MANIFEST_SCHEMA = "nemo-rl-strict-captured-replay-execution-manifest-v4"
SUBMISSION_SCHEMA = "nemo-rl-strict-captured-replay-submission-receipt-v5"
PRE_SCHEMA = "nemo-rl-strict-captured-replay-job-pre-receipt-v3"
EXIT_SCHEMA = "nemo-rl-strict-captured-replay-job-exit-receipt-v6"
FINAL_SCHEMA = "nemo-rl-strict-captured-replay-result-final-receipt-v1"
INVENTORY_SCHEMA = "nemo-rl-strict-captured-replay-result-inventory-v2"
INDEX_SCHEMA = "nemo-rl-strict-captured-replay-evidence-index-v4"
SCORER_INDEX_SCHEMA = "nemo-rl-strict-format-verification-call-index-v1"
REASONING_SCORER_INDEX_SCHEMA = "nemo-rl-strict-reasoning-score-call-index-v1"
TRANSPORT_SCHEMA = "nemo-rl-strict-model-transport-replay-consumption-v3"
TRANSCRIPT_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
LEDGER_SCHEMA = "nemo-rl-strict-captured-replay-step1-ledger-v5"

SNAPSHOT_KEYS = frozenset(
    {
        "schema",
        "pair_id",
        "environment",
        "profile_id",
        "attempt_id",
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
        "samples",
    }
)
SAMPLE_KEYS = frozenset(
    {
        "sample_index",
        "fixture_row_index",
        "rollout_index",
        "generation_seed",
        "model_transport_entry_sha256",
        "model_transport_request_body_sha256",
        "model_transport_response_body_sha256",
        "model_response_sha256",
        "match_details",
        "raw_environment_reward",
    }
)
OUTPUT_SCHEMAS = {
    "scorer_call_index": SCORER_INDEX_SCHEMA,
    "transport_consumption": TRANSPORT_SCHEMA,
    "transcript_bundle": TRANSCRIPT_SCHEMA,
    "replay_ledger": LEDGER_SCHEMA,
}
OUTPUT_PATHS = {
    "scorer_call_index": "strict_gym_child_runtime/format-verification-call-index.json",
    "transport_consumption": "model-transport-replay-consumption.json",
    "transcript_bundle": "transcript-bundle.json",
    "replay_ledger": "replay-ledger.json",
}
SCORER_INDEX_SCHEMA_BY_ENVIRONMENT = {
    "citation": SCORER_INDEX_SCHEMA,
    "freeform": SCORER_INDEX_SCHEMA,
    "reasoning_gym": REASONING_SCORER_INDEX_SCHEMA,
}
SCORER_INDEX_PATH_BY_ENVIRONMENT = {
    "citation": OUTPUT_PATHS["scorer_call_index"],
    "freeform": OUTPUT_PATHS["scorer_call_index"],
    "reasoning_gym": "strict_gym_child_runtime/reasoning-score-call-index.json",
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PAIR_ID = re.compile(r"[A-Za-z0-9._-]{1,64}\Z", re.ASCII)
_JOB_ID = re.compile(r"[1-9][0-9]*\Z", re.ASCII)
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.ASCII,
)
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z", re.ASCII)

# Supplied only by the already-authenticated deployment bootstrap.  Importing
# this file directly is useful for pure tests but grants no acceptance power.
_BOOTSTRAP_PROGRAM_REFERENCE = globals().pop("_NEMO_RL_V2_EVALUATOR_PROGRAM_REFERENCE", None)
_BOOTSTRAP_COORDINATOR_API = globals().pop("_NEMO_RL_V2_COORDINATOR_API", None)


class ReplayV2EvaluationError(RuntimeError):
    """One fail-closed request, authority, or parity error."""

    def __init__(self, code: str, message: str) -> None:
        if type(code) is not str or _ERROR_CODE.fullmatch(code) is None:
            code = "internal_failure"
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise ReplayV2EvaluationError(code, message)


def _exact_dict(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail("invalid_evidence", f"{label} has the wrong exact key set")
    return value


def _clean_ascii(value: Any, label: str, *, maximum: int) -> str:
    if type(value) is not str or not 1 <= len(value) <= maximum:
        _fail("invalid_evidence", f"{label} must be a bounded exact string")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError:
        _fail("invalid_evidence", f"{label} must be ASCII")
    if any(byte < 0x20 or byte > 0x7E for byte in raw):
        _fail("invalid_evidence", f"{label} contains a control byte")
    return value


def _digest(value: Any, label: str, *, nonzero: bool = True) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _fail("invalid_evidence", f"{label} must be a lowercase SHA-256")
    if nonzero and value == "0" * 64:
        _fail("invalid_evidence", f"{label} must be nonzero")
    return value


def _canonical_path(value: Any, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\n" in value or "\r" in value:
        _fail("invalid_evidence", f"{label} must be one nonempty line without NUL")
    if not value.startswith("/") or value.startswith("//") or value.endswith("/") or posixpath.normpath(value) != value:
        _fail("invalid_evidence", f"{label} must be a canonical absolute path")
    return value


def _reference(value: Any, label: str, *, schema: str | None = None) -> dict[str, str]:
    keys = frozenset({"path", "sha256"} if schema is None else {"path", "schema", "sha256"})
    reference = _exact_dict(value, keys, label)
    result = {
        "path": _canonical_path(reference["path"], f"{label}.path"),
        "sha256": _digest(reference["sha256"], f"{label}.sha256"),
    }
    if schema is not None:
        if type(reference["schema"]) is not str or reference["schema"] != schema:
            _fail("invalid_evidence", f"{label}.schema differs")
        result["schema"] = schema
    return result


def _bounded_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("invalid_evidence", f"{label} must be an exact bounded integer")
    return value


def _job_id(value: Any, label: str) -> str:
    job_id = _clean_ascii(value, label, maximum=19)
    if _JOB_ID.fullmatch(job_id) is None or int(job_id) > (1 << 63) - 1:
        _fail("invalid_evidence", f"{label} must be a canonical positive decimal job ID")
    return job_id


def _validate_json_tree(value: Any, label: str, *, request: bool) -> None:
    kind = type(value)
    if value is None or kind in {bool, str, int}:
        return
    if kind is float:
        if request or not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            _fail(
                "invalid_request" if request else "invalid_evidence",
                f"{label} contains a forbidden float",
            )
        return
    if kind is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]", request=request)
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(
                    "invalid_request" if request else "invalid_evidence",
                    f"{label} has a non-string key",
                )
            _validate_json_tree(item, f"{label}.{key}", request=request)
        return
    _fail(
        "invalid_request" if request else "invalid_evidence",
        f"{label} contains a forbidden type",
    )


def _canonical_bytes(value: Any, label: str, *, request: bool = False) -> bytes:
    _validate_json_tree(value, label, request=request)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        _fail(
            "invalid_request" if request else "invalid_evidence",
            f"cannot encode {label}: {error}",
        )


def _pairs_to_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("invalid_request", "request contains a duplicate or non-string key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    _fail("invalid_request", f"request contains forbidden constant {value!r}")


def _parse_request(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_REQUEST_BYTES:
        _fail("invalid_request", "request frame is empty or too large")
    try:
        text = raw.decode("ascii")
        value = json.loads(text, object_pairs_hook=_pairs_to_dict, parse_constant=_reject_constant)
    except ReplayV2EvaluationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        _fail("invalid_request", f"request frame is invalid: {error}")
    if type(value) is not dict or _canonical_bytes(value, "request", request=True) != raw:
        _fail("invalid_request", "request must be canonical ASCII JSON with no LF")
    return value


def _request_sha256(request: dict[str, Any]) -> str:
    projection = dict(request)
    projection.pop("request_sha256", None)
    return hashlib.sha256(
        REQUEST_HASH_DOMAIN + _canonical_bytes(projection, "request projection", request=True)
    ).hexdigest()


def _validate_request(value: dict[str, Any]) -> dict[str, Any]:
    request = _exact_dict(
        value,
        frozenset(
            {
                "schema",
                "nonce",
                "request_sha256",
                "evaluator_program",
                "pair",
                "attempts",
            }
        ),
        "request",
    )
    if type(request["schema"]) is not str or request["schema"] != REQUEST_SCHEMA:
        _fail("invalid_request", "request schema differs")
    _digest(request["nonce"], "request nonce")
    supplied_sha = _digest(request["request_sha256"], "request SHA-256")
    if supplied_sha != _request_sha256(request):
        _fail("invalid_request", "request SHA-256 differs")
    _reference(request["evaluator_program"], "evaluator program")

    pair = _exact_dict(
        request["pair"],
        frozenset(
            {
                "pair_id",
                "environment",
                "profile_id",
                "manifest",
                "submission_receipt",
                "off_exit_receipt",
            }
        ),
        "request pair",
    )
    pair_id = _clean_ascii(pair["pair_id"], "request pair ID", maximum=64)
    if _PAIR_ID.fullmatch(pair_id) is None:
        _fail("invalid_request", "request pair ID differs")
    environment = _clean_ascii(pair["environment"], "request environment", maximum=32)
    profile_id = _clean_ascii(pair["profile_id"], "request profile ID", maximum=64)
    if PROFILE_BY_ENVIRONMENT.get(environment) != profile_id:
        _fail("invalid_request", "request environment/profile dispatch differs")
    pair_references = {
        name: _reference(pair[name], f"request pair {name}")
        for name in ("manifest", "submission_receipt", "off_exit_receipt")
    }
    for member in ("path", "sha256"):
        if len({reference[member] for reference in pair_references.values()}) != 3:
            _fail("invalid_request", f"request Pair authority {member}s must be distinct")

    attempts = _exact_dict(request["attempts"], frozenset(ATTEMPT_NAMES), "request attempts")
    observed_candidates: set[str] = set()
    observed_paths: set[str] = set()
    observed_manifest_digests: set[str] = set()
    observed_submission_digests: set[str] = set()
    observed_final_digests: set[str] = set()
    for attempt in ATTEMPT_NAMES:
        item = _exact_dict(
            attempts[attempt],
            frozenset(
                {
                    "replay_execution_manifest",
                    "submission_receipt_sha256",
                    "candidate_job_id",
                    "result_final_receipt",
                }
            ),
            f"request {attempt}",
        )
        manifest = _reference(
            item["replay_execution_manifest"],
            f"request {attempt} replay execution manifest",
        )
        submission_digest = _digest(
            item["submission_receipt_sha256"],
            f"request {attempt} submission receipt SHA-256",
        )
        candidate = _job_id(item["candidate_job_id"], f"request {attempt} candidate job ID")
        final = _reference(item["result_final_receipt"], f"request {attempt} FINAL receipt")
        if candidate in observed_candidates:
            _fail("invalid_request", "request replay candidate job IDs must be distinct")
        observed_candidates.add(candidate)
        for digest, observed, label in (
            (manifest["sha256"], observed_manifest_digests, "manifest"),
            (submission_digest, observed_submission_digests, "submission receipt"),
            (final["sha256"], observed_final_digests, "FINAL receipt"),
        ):
            if digest in observed:
                _fail(
                    "invalid_request",
                    f"request replay {label} digests must be distinct",
                )
            observed.add(digest)
        for path in (manifest["path"], final["path"]):
            if path in observed_paths:
                _fail("invalid_request", "request replay authority paths must be distinct")
            observed_paths.add(path)
    return request


def _artifact(value: Any, label: str, *, path: str, schema: str, sha256: str | None = None) -> dict[str, str]:
    reference = _reference(value, label, schema=schema)
    if reference["path"] != path or (sha256 is not None and reference["sha256"] != sha256):
        _fail("invalid_evidence", f"{label} authority differs")
    return reference


def _result_roots(result_root: str, attempt: str) -> tuple[str, str]:
    suffix = f"/captured_replay/{attempt}"
    if not result_root.endswith(suffix) or len(result_root) <= len(suffix):
        _fail("invalid_evidence", "snapshot result root differs from the attempt")
    return result_root[: -len(suffix)], suffix


def _validate_processes(snapshot: dict[str, Any]) -> None:
    driver = _exact_dict(
        snapshot["driver_process"],
        frozenset({"boot_id_sha256", "pid", "start_time_ticks"}),
        "driver process",
    )
    driver_boot = _digest(driver["boot_id_sha256"], "driver boot ID SHA-256")
    driver_pid = _bounded_int(driver["pid"], "driver PID", minimum=1, maximum=(1 << 31) - 1)
    _bounded_int(
        driver["start_time_ticks"],
        "driver start ticks",
        minimum=1,
        maximum=(1 << 63) - 1,
    )

    scorer = _exact_dict(
        snapshot["scorer_process_identity"],
        frozenset({"boot_id", "hostname", "pid", "start_ticks"}),
        "scorer process",
    )
    boot_id = _clean_ascii(scorer["boot_id"], "scorer boot ID", maximum=36)
    if _BOOT_ID.fullmatch(boot_id) is None:
        _fail("invalid_evidence", "scorer boot ID differs")
    hostname = scorer["hostname"]
    if type(hostname) is not str or not hostname or "\x00" in hostname or "\n" in hostname or "\r" in hostname:
        _fail("invalid_evidence", "scorer hostname must be one nonempty line")
    try:
        hostname_raw = hostname.encode("ascii", "strict")
    except UnicodeEncodeError:
        _fail("invalid_evidence", "scorer hostname must be ASCII")
    if len(hostname_raw) > 255:
        _fail("invalid_evidence", "scorer hostname is too long")
    scorer_pid = _bounded_int(scorer["pid"], "scorer PID", minimum=1, maximum=(1 << 31) - 1)
    _bounded_int(scorer["start_ticks"], "scorer start ticks", minimum=1, maximum=(1 << 63) - 1)
    if driver_boot != hashlib.sha256((boot_id + "\n").encode("ascii")).hexdigest():
        _fail("invalid_evidence", "driver and scorer boot identities differ")
    if driver_pid == scorer_pid:
        _fail("invalid_evidence", "driver and scorer process identities alias")


def _validate_samples(snapshot: dict[str, Any], environment: str) -> list[dict[str, Any]]:
    samples = snapshot["samples"]
    if type(samples) is not list or len(samples) != 4:
        _fail("invalid_evidence", "snapshot must contain exact K=4 samples")
    for index, value in enumerate(samples):
        sample = _exact_dict(value, SAMPLE_KEYS, f"sample {index}")
        for name, expected in (
            ("sample_index", index),
            ("fixture_row_index", 0),
            ("rollout_index", index),
        ):
            if type(sample[name]) is not int or sample[name] != expected:
                _fail("invalid_evidence", f"sample {index} {name} differs")
        _bounded_int(
            sample["generation_seed"],
            f"sample {index} generation seed",
            minimum=0,
            maximum=(1 << 63) - 1,
        )
        for name in (
            "model_transport_entry_sha256",
            "model_transport_request_body_sha256",
            "model_transport_response_body_sha256",
            "model_response_sha256",
        ):
            _digest(sample[name], f"sample {index} {name}")
        details = sample["match_details"]
        if environment == "citation":
            details = _exact_dict(
                details,
                frozenset({"expected", "missing", "spurious", "passed"}),
                f"citation sample {index} match details",
            )
            for name in ("expected", "missing", "spurious"):
                items = details[name]
                if type(items) is not list or any(type(item) is not str for item in items):
                    _fail("invalid_evidence", f"citation sample {index} {name} differs")
            passed = details["passed"]
            if type(passed) is not bool or passed is not (not details["missing"] and not details["spurious"]):
                _fail("invalid_evidence", f"citation sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif environment == "freeform":
            details = _exact_dict(
                details,
                frozenset({"matching_lines", "min_matches", "passed"}),
                f"freeform sample {index} match details",
            )
            matching = _bounded_int(
                details["matching_lines"],
                f"freeform sample {index} matches",
                minimum=0,
                maximum=(1 << 31) - 1,
            )
            minimum = _bounded_int(
                details["min_matches"],
                f"freeform sample {index} minimum",
                minimum=0,
                maximum=(1 << 31) - 1,
            )
            passed = details["passed"]
            if type(passed) is not bool or passed is not (matching >= minimum):
                _fail("invalid_evidence", f"freeform sample {index} passed differs")
            expected_reward = 1.0 if passed else 0.0
        elif environment == "reasoning_gym":
            details = _exact_dict(
                details,
                frozenset({"task_name", "score", "extracted_answer"}),
                f"reasoning sample {index} match details",
            )
            if type(details["task_name"]) is not str or details["task_name"] != "knights_knaves":
                _fail("invalid_evidence", f"reasoning sample {index} task name differs")
            score = details["score"]
            if (
                type(score) is not float
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
                or (score == 0.0 and math.copysign(1.0, score) < 0.0)
            ):
                _fail("invalid_evidence", f"reasoning sample {index} score differs")
            if type(details["extracted_answer"]) is not str:
                _fail("invalid_evidence", f"reasoning sample {index} extracted answer differs")
            expected_reward = score
        else:  # Closed by request dispatch.
            raise AssertionError("unreachable profile")
        reward = sample["raw_environment_reward"]
        if (
            type(reward) is not float
            or not math.isfinite(reward)
            or not 0.0 <= reward <= 1.0
            or (reward == 0.0 and math.copysign(1.0, reward) < 0.0)
            or reward != expected_reward
        ):
            _fail("invalid_evidence", f"sample {index} reward differs")
    return samples


def _validate_snapshot(
    value: Any,
    *,
    attempt: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    snapshot = _exact_dict(value, SNAPSHOT_KEYS, f"{attempt} snapshot")
    _validate_json_tree(snapshot, f"{attempt} snapshot", request=False)
    if type(snapshot["schema"]) is not str or snapshot["schema"] != SNAPSHOT_SCHEMA:
        _fail("invalid_evidence", f"{attempt} snapshot schema differs")
    pair = request["pair"]
    requested_attempt = request["attempts"][attempt]
    expected_values = {
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "profile_id": pair["profile_id"],
        "attempt_id": attempt,
        "candidate_job_id": requested_attempt["candidate_job_id"],
        "authenticated_job_id": requested_attempt["candidate_job_id"],
    }
    if any(type(snapshot[name]) is not str or snapshot[name] != expected for name, expected in expected_values.items()):
        _fail("invalid_evidence", f"{attempt} snapshot identity differs")
    expected_run_id = hashlib.sha256(
        f"nemo-rl-strict-replay-v2:{pair['environment']}:{pair['pair_id']}:{attempt}".encode("ascii")
    ).hexdigest()
    if type(snapshot["run_id"]) is not str or snapshot["run_id"] != expected_run_id:
        _fail("invalid_evidence", f"{attempt} run ID differs")
    _validate_processes(snapshot)

    result_root = _canonical_path(snapshot["result_root"], f"{attempt} result root")
    results_root, _ = _result_roots(result_root, attempt)
    manifest_path = f"{results_root}/captured_replay/manifests/{pair['pair_id']}/{attempt}.json"
    submission_path = (
        f"{results_root}/captured_replay/replay_submission_state/"
        f"{pair['pair_id']}/{attempt}/submission-receipt.json"
    )
    receipt_root = (
        f"{results_root}/captured_replay/replay_job_state/{pair['pair_id']}/{attempt}/"
        f"{requested_attempt['candidate_job_id']}-0/receipts"
    )
    _artifact(
        snapshot["manifest"],
        f"{attempt} manifest",
        path=manifest_path,
        schema=MANIFEST_SCHEMA,
        sha256=requested_attempt["replay_execution_manifest"]["sha256"],
    )
    if requested_attempt["replay_execution_manifest"]["path"] != manifest_path:
        _fail("invalid_evidence", f"{attempt} request manifest path differs")
    _artifact(
        snapshot["submission_receipt"],
        f"{attempt} submission receipt",
        path=submission_path,
        schema=SUBMISSION_SCHEMA,
        sha256=requested_attempt["submission_receipt_sha256"],
    )
    _artifact(
        snapshot["pre_receipt"],
        f"{attempt} PRE",
        path=f"{receipt_root}/PRE.json",
        schema=PRE_SCHEMA,
    )
    _artifact(
        snapshot["exit_receipt"],
        f"{attempt} EXIT",
        path=f"{receipt_root}/EXIT.json",
        schema=EXIT_SCHEMA,
    )
    _artifact(
        snapshot["result_final_receipt"],
        f"{attempt} FINAL",
        path=f"{receipt_root}/FINAL.json",
        schema=FINAL_SCHEMA,
        sha256=requested_attempt["result_final_receipt"]["sha256"],
    )
    if requested_attempt["result_final_receipt"]["path"] != f"{receipt_root}/FINAL.json":
        _fail("invalid_evidence", f"{attempt} request FINAL path differs")
    _artifact(
        snapshot["result_inventory"],
        f"{attempt} inventory",
        path=f"{result_root}/result-inventory-v2.json",
        schema=INVENTORY_SCHEMA,
    )
    _artifact(
        snapshot["evidence_index"],
        f"{attempt} evidence index",
        path=f"{result_root}/evidence-index.json",
        schema=INDEX_SCHEMA,
    )
    outputs = _exact_dict(snapshot["outputs"], frozenset(OUTPUT_SCHEMAS), f"{attempt} outputs")
    output_schemas = dict(OUTPUT_SCHEMAS)
    output_paths = dict(OUTPUT_PATHS)
    output_schemas["scorer_call_index"] = SCORER_INDEX_SCHEMA_BY_ENVIRONMENT[pair["environment"]]
    output_paths["scorer_call_index"] = SCORER_INDEX_PATH_BY_ENVIRONMENT[pair["environment"]]
    for name, schema in output_schemas.items():
        _artifact(
            outputs[name],
            f"{attempt} output {name}",
            path=f"{result_root}/{output_paths[name]}",
            schema=schema,
        )
    _validate_samples(snapshot, pair["environment"])
    # Round-trip through canonical JSON to detach any coordinator-owned aliases.
    return json.loads(_canonical_bytes(snapshot, f"{attempt} snapshot").decode("ascii"))


def _coordinator_snapshot(api: Any, request: dict[str, Any], attempt: str) -> dict[str, Any]:
    if type(api) is not tuple or len(api) != 3:
        _fail("missing_authority", "coordinator API authority is absent")
    result_type, consume, snapshot_descriptor = api
    if type(result_type) is not type or not callable(consume):
        _fail("missing_authority", "coordinator API authority differs")
    descriptor_get = getattr(snapshot_descriptor, "__get__", None)
    if not callable(descriptor_get):
        _fail("missing_authority", "coordinator snapshot descriptor differs")
    pair = request["pair"]
    item = request["attempts"][attempt]
    result = consume(
        pair_manifest_path=pair["manifest"]["path"],
        pair_manifest_sha256=pair["manifest"]["sha256"],
        pair_submission_receipt_path=pair["submission_receipt"]["path"],
        pair_submission_receipt_sha256=pair["submission_receipt"]["sha256"],
        trusted_off_exit_receipt_path=pair["off_exit_receipt"]["path"],
        trusted_off_exit_receipt_sha256=pair["off_exit_receipt"]["sha256"],
        replay_manifest_path=item["replay_execution_manifest"]["path"],
        replay_manifest_sha256=item["replay_execution_manifest"]["sha256"],
        submission_receipt_sha256=item["submission_receipt_sha256"],
        candidate_job_id=item["candidate_job_id"],
        result_final_receipt_path=item["result_final_receipt"]["path"],
        result_final_receipt_sha256=item["result_final_receipt"]["sha256"],
        expected_environment=pair["environment"],
        expected_profile_id=pair["profile_id"],
    )
    if type(result) is not result_type:
        _fail("invalid_evidence", f"{attempt} coordinator result type differs")
    try:
        snapshot = descriptor_get(result, result_type)
    except Exception as error:
        _fail("invalid_evidence", f"{attempt} coordinator snapshot access failed: {error}")
    return _validate_snapshot(snapshot, attempt=attempt, request=request)


def _snapshot_identity(snapshot: dict[str, Any], field: str) -> bytes:
    return _canonical_bytes(snapshot[field], f"snapshot {field}")


def _validate_distinct_and_parity(
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first = snapshots["replay-1"]
    second = snapshots["replay-2"]
    for field in (
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
    ):
        if _snapshot_identity(first, field) == _snapshot_identity(second, field):
            _fail("invalid_evidence", f"replay attempts reuse {field}")
    for field in (
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_inventory",
        "evidence_index",
    ):
        for member in ("path", "sha256"):
            if first[field][member] == second[field][member]:
                _fail(
                    "invalid_evidence",
                    f"replay attempts reuse {field}.{member}",
                )
    if first["result_root"] == second["result_root"]:
        _fail("invalid_evidence", "replay attempts reuse result_root")
    for output_name in OUTPUT_SCHEMAS:
        for member in ("path", "sha256"):
            if first["outputs"][output_name][member] == second["outputs"][output_name][member]:
                _fail(
                    "invalid_evidence",
                    f"replay attempts reuse outputs.{output_name}.{member}",
                )
    first_samples = _canonical_bytes(first["samples"], "replay-1 samples")
    second_samples = _canonical_bytes(second["samples"], "replay-2 samples")
    if first_samples != second_samples:
        _fail("replay_mismatch", "authenticated replay sample evidence differs")
    rewards = [sample["raw_environment_reward"] for sample in first["samples"]]
    return {
        "schema": "nemo-rl-strict-captured-replay-v2-parity-v1",
        "status": "exact-match",
        "samples_sha256": hashlib.sha256(PARITY_HASH_DOMAIN + first_samples).hexdigest(),
        "reward_vector": rewards,
    }


def evaluate_authenticated_request(
    request: dict[str, Any],
    *,
    evaluator_program: dict[str, str],
    coordinator_api: tuple[Any, Any, Any],
) -> dict[str, Any]:
    """Evaluate one already-decoded request under explicit bootstrap authority."""
    request = _validate_request(request)
    program = _reference(evaluator_program, "authenticated evaluator program")
    if request["evaluator_program"] != program:
        _fail(
            "missing_authority",
            "request evaluator program is not the authenticated deployment",
        )
    snapshots = {attempt: _coordinator_snapshot(coordinator_api, request, attempt) for attempt in ATTEMPT_NAMES}
    parity = _validate_distinct_and_parity(snapshots)
    return {
        "schema": REPORT_SCHEMA,
        "status": "authenticated",
        "nonce": request["nonce"],
        "request_sha256": request["request_sha256"],
        "evaluator_program": dict(program),
        "pair": {
            "pair_id": request["pair"]["pair_id"],
            "environment": request["pair"]["environment"],
            "profile_id": request["pair"]["profile_id"],
            "manifest_sha256": request["pair"]["manifest"]["sha256"],
            "submission_receipt_sha256": request["pair"]["submission_receipt"]["sha256"],
            "off_exit_receipt_sha256": request["pair"]["off_exit_receipt"]["sha256"],
        },
        "attempts": snapshots,
        "parity": parity,
    }


def _validate_runtime() -> None:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.no_user_site == 1
        and flags.ignore_environment == 1
        and flags.safe_path
        and flags.dont_write_bytecode == 1
    ):
        _fail(
            "missing_authority",
            "evaluator requires exact -I -S -B interpreter isolation",
        )
    if os.getcwd() != "/" or type(os.environ) is not os._Environ or dict(os.environ) != EXACT_ENVIRONMENT:
        _fail("missing_authority", "evaluator runtime boundary differs")
    if type(sys.argv) is not list or len(sys.argv) != 1:
        _fail("missing_authority", "evaluator argv boundary differs")
    logical_path = _canonical_path(sys.argv[0], "evaluator logical path")
    if type(__file__) is not str or logical_path != __file__:
        _fail("missing_authority", "evaluator logical path differs from its source")
    for descriptor in (0, 1, 2):
        try:
            os.fstat(descriptor)
        except OSError as error:
            _fail(
                "missing_authority",
                f"evaluator fd {descriptor} is unavailable: {error}",
            )
        if os.isatty(descriptor):
            _fail("missing_authority", f"evaluator fd {descriptor} must not be a TTY")


def _read_request() -> bytes:
    chunks: list[bytes] = []
    remaining = MAX_REQUEST_BYTES + 1
    while remaining:
        chunk = os.read(0, min(READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > MAX_REQUEST_BYTES:
        _fail("invalid_request", "request exceeds fixed bound")
    return raw


def _write_report(report: dict[str, Any]) -> None:
    raw = _canonical_bytes(report, "report")
    if not 1 <= len(raw) <= MAX_REPORT_BYTES:
        _fail("internal_failure", "report exceeds fixed bound")
    offset = 0
    while offset < len(raw):
        written = os.write(1, raw[offset:])
        if written <= 0:
            _fail("internal_failure", "report write made no progress")
        offset += written


def _rejected(error: ReplayV2EvaluationError, raw: bytes | None) -> dict[str, Any]:
    nonce: str | None = None
    request_sha: str | None = None
    if raw is not None:
        try:
            value = json.loads(raw.decode("ascii"))
            if type(value) is dict:
                if type(value.get("nonce")) is str and _HEX64.fullmatch(value["nonce"]):
                    nonce = value["nonce"]
                if type(value.get("request_sha256")) is str and _HEX64.fullmatch(value["request_sha256"]):
                    request_sha = value["request_sha256"]
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            pass
    message = str(error).encode("ascii", "replace").decode("ascii")[:512] or "rejected"
    return {
        "schema": REPORT_SCHEMA,
        "status": "rejected",
        "nonce": nonce,
        "request_sha256": request_sha,
        "error": {"code": error.code, "message": message},
    }


def main() -> int:
    raw: bytes | None = None
    try:
        _validate_runtime()
        if _BOOTSTRAP_PROGRAM_REFERENCE is None or _BOOTSTRAP_COORDINATOR_API is None:
            _fail(
                "missing_authority",
                "authenticated evaluator bootstrap authority is absent",
            )
        raw = _read_request()
        request = _validate_request(_parse_request(raw))
        report = evaluate_authenticated_request(
            request,
            evaluator_program=_BOOTSTRAP_PROGRAM_REFERENCE,
            coordinator_api=_BOOTSTRAP_COORDINATOR_API,
        )
    except ReplayV2EvaluationError as error:
        _write_report(_rejected(error, raw))
        return 1
    except Exception as error:  # Fail closed without traceback/stderr leakage.
        wrapped = ReplayV2EvaluationError("internal_failure", f"internal evaluator failure: {type(error).__name__}")
        _write_report(_rejected(wrapped, raw))
        return 1
    _write_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
