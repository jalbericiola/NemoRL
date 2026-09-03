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

"""Fail-closed K=4 citation/freeform scorer-only captured replay V2.

This module deliberately contains no model-generation path.  It consumes the
authenticated OFF transport material exactly once, sends the reconstructed
resource request directly to ``POST /verify``, and builds the transcript-v4 and
replay-ledger-v5 documents with the shared evidence utilities.

The external wrapper captures scheduler authentication and publishes PRE/EXIT;
the authenticated replay entrypoint validates that boundary and owns the
resource-child lifecycle.  This runtime receives only the closed scorer and
terminal callbacks, so it cannot substitute a login-side candidate ID or start
a policy/model server.
"""

from __future__ import annotations

import copy
import http.client
import json
import math
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_rl.utils.strict_captured_replay_evidence import (
    TRANSCRIPT_BUNDLE_SCHEMA,
    build_captured_replay_step1_ledger,
    build_transcript_bundle,
    canonical_ascii_json,
    document_sha256,
    model_response_token_geometry,
    replay_run_id,
    validate_captured_replay_source_join,
    validate_captured_replay_step1_ledger,
    validate_ledger_transcript_join,
    validate_transcript_bundle,
)
from nemo_rl.utils.strict_main_step_ledger import validate_main_step1_ledger

K4 = 4
MAX_HTTP_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_SIGNED_INT64 = (1 << 63) - 1
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_MATERIAL_KEYS = frozenset(
    {
        "rollout_index",
        "generation_seed",
        "model_response",
        "agent_run_request",
        "derived_verifier_request",
        "source_entry_sha256",
        "request_body_sha256",
        "response_body_sha256",
    }
)
_PENALTY_FLAGS = {
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


class StrictCapturedReplayError(RuntimeError):
    """A fail-closed replay boundary rejected its input or observation."""


class StrictModelTransportReplaySourceV3(Protocol):
    """Profile-bound exact-once OFF transport source used by this runtime."""

    def consume(self, *, rollout_index: int, generation_seed: int) -> Mapping[str, Any]:
        """Consume one logical source entry exactly once."""

    def record_fresh_verifier_result(
        self, *, rollout_index: int, verifier_response: Mapping[str, Any]
    ) -> None:
        """Validate and retain one fresh resource-server result."""

    def finalize(
        self,
        *,
        replay_execution_manifest_sha256: str,
        authenticated_job_id: str,
        process: Mapping[str, Any],
        scheduler_device_environment: Mapping[str, Any],
        format_verification_call_index_ref: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return terminal exact-once consumption evidence."""


VerifierPost = Callable[[int, int, Mapping[str, Any]], Mapping[str, Any]]
IndependentFormatCheck = Callable[[int, Mapping[str, Any], Mapping[str, Any]], None]
FinalizeFormatCallEvidence = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ReplayDocumentsV2:
    """The three runtime-owned documents declared by the replay manifest."""

    transcript_bundle: dict[str, Any]
    replay_ledger: dict[str, Any]
    transport_consumption: dict[str, Any]
    scorer_call_index: dict[str, Any]


def execute_profiled_captured_replay_cohort(
    *,
    manifest: Mapping[str, Any],
    replay_execution_manifest_sha256: str,
    submission_receipt_sha256: str,
    authenticated_job_id: str,
    driver_process: Mapping[str, Any],
    driver_scheduler_device_environment: Mapping[str, Any],
    source_transcript_document: Mapping[str, Any],
    source_main_ledger_document: Mapping[str, Any],
    transport_source: StrictModelTransportReplaySourceV3,
    post_verifier: VerifierPost,
    independent_format_check: IndependentFormatCheck | None,
    finalize_format_call_evidence: FinalizeFormatCallEvidence,
    expected_environment: str,
    expected_profile_id: str,
) -> ReplayDocumentsV2:
    """Execute one explicitly selected K=4 citation/freeform replay.

    ``manifest`` must already have passed
    :func:`validate_replay_execution_manifest` against the authenticated Pair.
    This function intentionally rechecks every source document and every leaf it
    consumes before producing output.
    """
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )

    manifest = _mapping(manifest, "manifest")
    _mapping(driver_process, "driver process")
    _mapping(
        driver_scheduler_device_environment,
        "driver scheduler device environment",
    )
    source_transcript_document = _mapping(
        source_transcript_document,
        "source transcript document",
    )
    source_main_ledger_document = _mapping(
        source_main_ledger_document,
        "source main ledger document",
    )
    profile = get_strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    if manifest.get("schema") != "nemo-rl-strict-captured-replay-execution-manifest-v4":
        raise StrictCapturedReplayError("profiled runtime requires manifest V4")
    pair_id = _safe_pair_id(manifest.get("pair_id"))
    environment = manifest.get("environment")
    if environment != profile.environment:
        raise StrictCapturedReplayError(
            "manifest environment differs from selected scorer profile"
        )
    if canonical_ascii_json(manifest.get("scorer_profile")) != canonical_ascii_json(
        _scorer_profile(profile)
    ):
        raise StrictCapturedReplayError(
            "manifest scorer_profile differs from closed registry"
        )
    attempt_id = manifest.get("attempt_id")
    if attempt_id not in {"replay-1", "replay-2"}:
        raise StrictCapturedReplayError("replay attempt is not admitted")
    if (
        manifest.get("arm") != "on"
        or manifest.get("mode") != "fresh_verifier_reward_replay"
    ):
        raise StrictCapturedReplayError("runtime accepts only ON scorer-only replay")
    _digest(replay_execution_manifest_sha256, "replay manifest SHA-256")
    _digest(submission_receipt_sha256, "submission receipt SHA-256")
    _job_id(authenticated_job_id)

    replay_contract = _mapping(manifest.get("replay_contract"), "replay_contract")
    policy_execution = _mapping(
        replay_contract.get("policy_execution"), "replay_contract.policy_execution"
    )
    if policy_execution != {
        "backward": False,
        "forward": False,
        "optimizer": False,
        "violation": "fail-closed",
    }:
        raise StrictCapturedReplayError("policy execution is forbidden in replay")
    if replay_contract.get("execution_scope") != "scorer-only":
        raise StrictCapturedReplayError("replay execution scope must be scorer-only")
    if manifest.get("wandb") != {
        "enabled": False,
        "mode": "disabled",
        "reason": "scorer-only-replay-no-wandb-credentials-or-output",
    }:
        raise StrictCapturedReplayError(
            "scorer-only replay requires the exact disabled W&B policy"
        )

    source_capture = _mapping(manifest.get("source_capture"), "source_capture")
    source_step1 = _mapping(
        source_capture.get("step1_evidence"), "source step1 evidence"
    )
    source_transcript_ref = _mapping(
        source_step1.get("transcript_bundle"), "source transcript reference"
    )
    source_ledger_ref = _mapping(
        source_step1.get("main_ledger"), "source ledger reference"
    )
    _close_document_reference(
        source_transcript_ref,
        source_transcript_document,
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
        name="source transcript",
    )
    _close_document_reference(
        source_ledger_ref,
        source_main_ledger_document,
        expected_schema="nemo-rl-strict-main-step1-ledger-v5",
        name="source main ledger",
    )
    validate_transcript_bundle(source_transcript_document)
    validate_main_step1_ledger(source_main_ledger_document)
    validate_ledger_transcript_join(
        ledger=source_main_ledger_document,
        transcript_bundle=source_transcript_document,
    )
    if (
        source_transcript_document.get("pair_id") != pair_id
        or source_transcript_document.get("environment") != environment
        or source_transcript_document.get("arm") != "off"
        or source_transcript_document.get("mode") != "observe"
        or source_transcript_document.get("attempt_id") is not None
    ):
        raise StrictCapturedReplayError("OFF source transcript identity differs")

    source_entries = source_transcript_document.get("entries")
    if type(source_entries) is not list or len(source_entries) != K4:
        raise StrictCapturedReplayError("OFF source transcript must contain exact K=4")
    generation = copy.deepcopy(source_transcript_document["generation"])
    entry_inputs: list[dict[str, Any]] = []
    for rollout_index, source_entry in enumerate(source_entries):
        source_entry = _mapping(source_entry, f"source entry {rollout_index}")
        generation_seed = source_entry.get("generation_seed")
        if type(generation_seed) is not int:
            raise StrictCapturedReplayError(
                "source generation seed is not an exact int"
            )
        material = _material_document(
            transport_source.consume(
                rollout_index=rollout_index, generation_seed=generation_seed
            ),
            rollout_index=rollout_index,
        )
        _validate_material_against_source(
            material=material,
            source_entry=source_entry,
            rollout_index=rollout_index,
            generation_seed=generation_seed,
        )
        derived_request = copy.deepcopy(material["derived_verifier_request"])
        verifier_response = _mapping(
            post_verifier(rollout_index, generation_seed, derived_request),
            f"fresh verifier response {rollout_index}",
        )
        if independent_format_check is None:
            raise StrictCapturedReplayError(
                "profiled replay requires an independent pinned format check"
            )
        independent_format_check(rollout_index, derived_request, verifier_response)
        transport_source.record_fresh_verifier_result(
            rollout_index=rollout_index,
            verifier_response=verifier_response,
        )
        reward = verifier_response.get("reward")
        if type(reward) is not float or not math.isfinite(reward):
            raise StrictCapturedReplayError(
                f"fresh verifier response {rollout_index} reward is not a finite JSON float"
            )
        if reward == 0.0 and math.copysign(1.0, reward) < 0:
            raise StrictCapturedReplayError(
                "fresh verifier reward must not be negative zero"
            )
        entry_inputs.append(
            {
                "sample_index": rollout_index,
                "fixture_row_index": 0,
                "rollout_index": rollout_index,
                "generation_seed": generation_seed,
                "generation_request": copy.deepcopy(source_entry["generation_request"]),
                "model_response": copy.deepcopy(material["model_response"]),
                "agent_run_request": copy.deepcopy(material["agent_run_request"]),
                "derived_verifier_request": derived_request,
                "verifier_response": copy.deepcopy(dict(verifier_response)),
                "raw_environment_reward": reward,
                "model_transport_entry_sha256": material["source_entry_sha256"],
                "model_transport_request_body_sha256": material["request_body_sha256"],
                "model_transport_response_body_sha256": material[
                    "response_body_sha256"
                ],
            }
        )

    transcript_bindings = _transcript_bindings(
        manifest=manifest,
        submission_receipt_sha256=submission_receipt_sha256,
        authenticated_job_id=authenticated_job_id,
    )
    transport_ref = _mapping(
        _mapping(source_step1.get("model_transport"), "source model transport").get(
            "bundle"
        ),
        "source transport bundle reference",
    )
    transcript = build_transcript_bundle(
        pair_id=pair_id,
        environment=environment,
        arm="on",
        mode="captured_replay",
        attempt_id=attempt_id,
        generation=generation,
        bindings=transcript_bindings,
        fixture_row=source_transcript_document["fixture_row"]["value"],
        model_transport_bundle=transport_ref,
        verifier_request_derivation=source_transcript_document[
            "verifier_request_derivation"
        ],
        entry_inputs=entry_inputs,
    )
    validate_transcript_bundle(transcript)
    validate_captured_replay_source_join(
        source_transcript_bundle=source_transcript_document,
        replay_transcript_bundle=transcript,
    )

    outputs = _mapping(
        _mapping(manifest.get("artifacts"), "artifacts").get("outputs"),
        "artifacts.outputs",
    )
    transcript_output = _declared_output(
        outputs.get("transcript_bundle"),
        expected_schema=TRANSCRIPT_BUNDLE_SCHEMA,
        name="transcript output",
    )
    transcript_ref = {
        "path": transcript_output["path"],
        "schema": TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": document_sha256(transcript, trailing_lf=False),
    }
    ledger_rows = _build_replay_ledger_rows(
        source_main_ledger=source_main_ledger_document,
        replay_transcript=transcript,
    )
    ledger_bindings = {
        **transcript_bindings,
        "restart_count": 0,
        "pair_campaign_sha256": manifest["pair"]["pair_campaign_sha256"],
        "pair_campaign_reward_and_advantage_sha256": manifest["pair"][
            "pair_campaign_reward_and_advantage_sha256"
        ],
        "process": copy.deepcopy(dict(driver_process)),
    }
    ledger = build_captured_replay_step1_ledger(
        pair_id=pair_id,
        environment=environment,
        attempt_id=attempt_id,
        source_main_ledger_sha256=source_ledger_ref["sha256"],
        source_transcript_bundle=source_transcript_ref,
        source_transcript_document=source_transcript_document,
        generation=generation,
        bindings=ledger_bindings,
        transcript_bundle=transcript_ref,
        transcript_document=transcript,
        row_inputs=ledger_rows,
    )
    validate_captured_replay_step1_ledger(
        ledger,
        source_transcript_document=source_transcript_document,
        transcript_document=transcript,
    )
    validate_ledger_transcript_join(ledger=ledger, transcript_bundle=transcript)

    scorer_call_index_ref = _mapping(
        finalize_format_call_evidence(),
        "format-verification terminal reference",
    )
    if set(scorer_call_index_ref) != {"path", "schema", "sha256"}:
        raise StrictCapturedReplayError(
            "format-verification terminal reference keyset differs"
        )
    if scorer_call_index_ref.get("schema") != profile.call_index_schema:
        raise StrictCapturedReplayError(
            "format-verification terminal reference schema differs"
        )
    _digest(
        scorer_call_index_ref.get("sha256"),
        "format-verification terminal SHA-256",
    )
    scorer_output = _declared_output(
        outputs.get("scorer_call_index"),
        expected_schema=profile.call_index_schema,
        name="scorer output",
    )
    if (
        scorer_call_index_ref.get("path") != scorer_output.get("path")
        or scorer_output.get("schema") != profile.call_index_schema
    ):
        raise StrictCapturedReplayError(
            "format-verification terminal reference differs from manifest output"
        )
    consumption = _mapping(
        transport_source.finalize(
            replay_execution_manifest_sha256=replay_execution_manifest_sha256,
            authenticated_job_id=authenticated_job_id,
            process=driver_process,
            scheduler_device_environment=driver_scheduler_device_environment,
            format_verification_call_index_ref=scorer_call_index_ref,
        ),
        "transport replay consumption",
    )
    return ReplayDocumentsV2(
        transcript_bundle=transcript,
        replay_ledger=ledger,
        transport_consumption=copy.deepcopy(dict(consumption)),
        scorer_call_index=copy.deepcopy(dict(scorer_call_index_ref)),
    )


def post_resource_verify(
    *,
    host: str,
    port: int,
    derived_verifier_request: Mapping[str, Any],
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Send one direct, non-redirecting ``POST /verify`` to the resource child."""
    if type(host) is not str or not host or any(ord(char) < 33 for char in host):
        raise StrictCapturedReplayError("resource scorer host is malformed")
    if type(port) is not int or not 1 <= port <= 65535:
        raise StrictCapturedReplayError("resource scorer port is malformed")
    if type(timeout_seconds) is not float or not 0.0 < timeout_seconds <= 900.0:
        raise StrictCapturedReplayError("resource scorer timeout is outside policy")
    payload = canonical_ascii_json(
        _mapping(derived_verifier_request, "derived verifier request")
    )
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        connection.request(
            "POST",
            "/verify",
            body=payload,
            headers={
                "Accept": "application/json",
                "Content-Length": str(len(payload)),
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise StrictCapturedReplayError(
                f"resource scorer returned HTTP status {response.status}"
            )
        raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
        if len(raw) > MAX_HTTP_RESPONSE_BYTES:
            raise StrictCapturedReplayError("resource scorer response exceeds 64 MiB")
        if response.read(1):
            raise StrictCapturedReplayError(
                "resource scorer response grew while reading"
            )
    finally:
        connection.close()
    return _strict_json_object(raw, name="resource scorer response")


def _scorer_profile(profile: Any) -> dict[str, Any]:
    """Project the complete closed profile into the manifest wire shape."""
    return {
        "environment": profile.environment,
        "profile_id": profile.profile_id,
        "verifier_type": profile.verifier_type,
        "method": profile.method,
        "resource_config_path_name": profile.resource_config_path_name,
        "disabled_config_path_name": profile.disabled_config_path_name,
        "resource_app": {
            "path": profile.resource_app_path,
            "sha256": profile.resource_app_sha256,
        },
        "resource_config": {
            "path": profile.resource_config_path,
            "sha256": profile.resource_config_sha256,
        },
        "requirements": {
            "path": profile.requirements_path,
            "sha256": profile.requirements_sha256,
        },
        "fixture": {
            "path": profile.fixture_path,
            "sha256": profile.fixture_sha256,
            "rows": profile.fixture_rows,
        },
        "call_schema": profile.call_schema,
        "closed_schema": profile.closed_schema,
        "call_index_schema": profile.call_index_schema,
    }


def _build_replay_ledger_rows(
    *,
    source_main_ledger: Mapping[str, Any],
    replay_transcript: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_rows = source_main_ledger.get("rows")
    entries = replay_transcript.get("entries")
    if type(source_rows) is not list or type(entries) is not list:
        raise StrictCapturedReplayError(
            "source ledger/replay transcript rows are malformed"
        )
    if len(source_rows) != K4 or len(entries) != K4:
        raise StrictCapturedReplayError("replay ledger geometry must be exact K=4")
    rewards = [_float32(entry["raw_environment_reward"]) for entry in entries]
    advantages = _expected_float32_rloo_advantages(rewards)
    rows: list[dict[str, Any]] = []
    for index, (source_row, entry) in enumerate(zip(source_rows, entries, strict=True)):
        source_row = _mapping(source_row, f"source ledger row {index}")
        prompt, completion, token_ids = model_response_token_geometry(
            entry["model_response"], name=f"replay entry {index}.model_response"
        )
        if (
            source_row.get("prompt_token_ids") != prompt
            or source_row.get("completion_token_ids") != completion
        ):
            raise StrictCapturedReplayError(
                f"source ledger row {index} token geometry differs from OFF transport"
            )
        reward = rewards[index]
        rows.append(
            {
                "sample_index": index,
                "sample_id": source_row["sample_id"],
                "shared_prefix_group_id": source_row["shared_prefix_group_id"],
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
                "penalty_flags": dict(_PENALTY_FLAGS),
                "verifier_reward": reward,
                "processed_reward": reward,
                "sample_mask": 1.0,
                "advantages": [advantages[index]] * len(token_ids),
                "valid_loss_tokens": len(completion),
                "total_tokens": len(token_ids),
            }
        )
    return rows


def _transcript_bindings(
    *,
    manifest: Mapping[str, Any],
    submission_receipt_sha256: str,
    authenticated_job_id: str,
) -> dict[str, Any]:
    pair = _mapping(manifest.get("pair"), "pair")
    contract = _mapping(manifest.get("replay_contract"), "replay_contract")
    scorer = _mapping(contract.get("gym_scorer"), "gym_scorer")
    resources = _mapping(scorer.get("resources"), "gym_scorer.resources")
    source_snapshot = _mapping(contract.get("source_snapshot"), "source_snapshot")
    if set(source_snapshot) != {"arm", "ref"} or source_snapshot.get("arm") != "on":
        raise StrictCapturedReplayError(
            "replay source snapshot is not the authenticated ON reference"
        )
    source_snapshot_ref = _mapping(source_snapshot.get("ref"), "source snapshot ref")
    return {
        "pair_manifest_sha256": pair["manifest"]["sha256"],
        "submission_receipt_sha256": submission_receipt_sha256,
        "job_id": authenticated_job_id,
        "run_id": replay_run_id(
            environment=manifest["environment"],
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
        "fixture_sha256": manifest["artifacts"]["fixture"]["sha256"],
        "verifier_source_sha256": resources["verifier_source"]["sha256"],
        "config_sha256": contract["selected_config"]["sha256"],
        "snapshot_manifest_sha256": source_snapshot_ref["manifest_sha256"],
    }


def _validate_material_against_source(
    *,
    material: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    rollout_index: int,
    generation_seed: int,
) -> None:
    expected = {
        "rollout_index": rollout_index,
        "generation_seed": generation_seed,
        "model_response": source_entry["model_response"],
        "agent_run_request": source_entry["agent_run_request"],
        "derived_verifier_request": source_entry["derived_verifier_request"],
        "source_entry_sha256": source_entry["model_transport_entry_sha256"],
        "request_body_sha256": source_entry["model_transport_request_body_sha256"],
        "response_body_sha256": source_entry["model_transport_response_body_sha256"],
    }
    if canonical_ascii_json(material) != canonical_ascii_json(expected):
        raise StrictCapturedReplayError(
            f"transport material {rollout_index} differs from authenticated OFF transcript"
        )


def _material_document(value: Any, *, rollout_index: int) -> dict[str, Any]:
    """Close the transport Mapping into an owned exact JSON object."""
    material = _mapping(value, f"transport material {rollout_index}")
    if set(material) != _MATERIAL_KEYS:
        raise StrictCapturedReplayError(
            f"transport material {rollout_index} keyset differs"
        )
    document = copy.deepcopy(dict(material))
    canonical_ascii_json(document)
    return document


def _close_document_reference(
    reference: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    expected_schema: str,
    name: str,
) -> None:
    if set(reference) != {"path", "schema", "sha256"}:
        raise StrictCapturedReplayError(f"{name} reference keyset differs")
    if (
        reference.get("schema") != expected_schema
        or document.get("schema") != expected_schema
    ):
        raise StrictCapturedReplayError(f"{name} schema differs")
    if reference.get("sha256") != document_sha256(document, trailing_lf=False):
        raise StrictCapturedReplayError(f"{name} digest does not bind document bytes")


def _strict_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise StrictCapturedReplayError(f"{name} contains non-finite {value}")

    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictCapturedReplayError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def parse_int(value: str) -> int:
        if value == "-0":
            raise StrictCapturedReplayError(f"{name} contains negative zero")
        return int(value)

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise StrictCapturedReplayError(f"{name} contains a non-finite float")
        if parsed == 0.0 and value.startswith("-"):
            raise StrictCapturedReplayError(f"{name} contains negative zero")
        return parsed

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_int=parse_int,
            parse_float=parse_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictCapturedReplayError(f"{name} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise StrictCapturedReplayError(f"{name} root must be an object")
    canonical_ascii_json(value)
    return value


def _expected_float32_rloo_advantages(rewards: Sequence[float]) -> list[float]:
    if len(rewards) != K4:
        raise StrictCapturedReplayError("RLOO closure requires exact K=4")
    rounded = [_float32(value) for value in rewards]
    result: list[float] = []
    for index, reward in enumerate(rounded):
        peers = [
            value for peer_index, value in enumerate(rounded) if peer_index != index
        ]
        peer_sum = _float32(_float32(peers[0] + peers[1]) + peers[2])
        mean = _float32(peer_sum / 3.0)
        squares = [_float32(value * value) for value in peers]
        square_sum = _float32(_float32(squares[0] + squares[1]) + squares[2])
        square_mean = _float32(square_sum / 3.0)
        variance = _float32(_float32(square_mean - _float32(mean * mean)) * 1.5)
        standard_deviation = _float32(math.sqrt(max(variance, 0.0)))
        delta = _float32(reward - mean)
        advantage = (
            _float32(delta / _float32(standard_deviation + 1e-6))
            if standard_deviation > 0.0
            else delta
        )
        result.append(_float32(min(20.0, max(-20.0, advantage))))
    return result


def _float32(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise StrictCapturedReplayError(
            "replay reward must be an exact finite JSON float"
        )
    rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    return 0.0 if rounded == 0.0 else rounded


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise StrictCapturedReplayError(f"{name} must be an exact JSON object")
    canonical_ascii_json(value)
    return value


def _declared_output(
    value: Any,
    *,
    expected_schema: str,
    name: str,
) -> Mapping[str, Any]:
    output = _mapping(value, name)
    if set(output) != {"path", "schema", "framing", "mode"}:
        raise StrictCapturedReplayError(f"{name} keyset differs")
    if (
        output.get("schema") != expected_schema
        or output.get("framing") != "canonical-ascii-json-no-lf"
        or output.get("mode") != "0400"
    ):
        raise StrictCapturedReplayError(f"{name} publication contract differs")
    return output


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 64
    ):
        raise StrictCapturedReplayError(f"{name} must be a nonzero lowercase SHA-256")
    return value


def _job_id(value: Any) -> str:
    if (
        type(value) is not str
        or not value.isdecimal()
        or str(int(value)) != value
        or not 0 < int(value) <= _MAX_SIGNED_INT64
    ):
        raise StrictCapturedReplayError(
            "authenticated job ID must be a canonical positive int63 string"
        )
    return value


def _safe_pair_id(value: Any) -> str:
    if (
        type(value) is not str
        or _PAIR_ID_RE.fullmatch(value) is None
        or len(value.encode("ascii")) > 64
    ):
        raise StrictCapturedReplayError("pair_id is not a bounded safe identifier")
    return value


__all__ = [
    "IndependentFormatCheck",
    "FinalizeFormatCallEvidence",
    "ReplayDocumentsV2",
    "StrictCapturedReplayError",
    "StrictModelTransportReplaySourceV3",
    "VerifierPost",
    "execute_profiled_captured_replay_cohort",
    "post_resource_verify",
]
