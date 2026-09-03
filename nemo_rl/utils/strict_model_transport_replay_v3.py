"""Profile-bound model-transport replay consumption.

This module is intentionally parallel to :mod:`strict_model_transport_replay`.
Importing the legacy module does not import the profile registry, and callers
must select one exact environment/profile pair before any V3 authority exists.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import posixpath
import re
import stat
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from nemo_rl.utils.strict_captured_replay_evidence import (
    HASH_DOMAIN,
    TRANSCRIPT_BUNDLE_SCHEMA,
    canonical_ascii_json,
    derive_nemo_gym_request_seed,
    domain_sha256,
    load_evidence_document,
    publish_evidence_document,
    validate_fresh_verifier_response,
    validate_ledger_transcript_join,
    validate_scheduler_device_environment,
    validate_transcript_bundle,
    validate_transcript_model_transport_join,
)
from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_LEDGER_SCHEMA,
    validate_main_step1_ledger,
)
from nemo_rl.utils.strict_model_transport import (
    K4_SAMPLES,
    MAX_BUNDLE_BYTES,
    MODEL_TRANSPORT_BUNDLE_SCHEMA,
    MODEL_TRANSPORT_CALL_SCHEMA,
    MODEL_TRANSPORT_MANIFEST_SCHEMA,
    load_attested_model_transport_policy,
    validate_model_transport_manifest,
)

if TYPE_CHECKING:
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
    )
    from nemo_rl.utils.strict_captured_replay_profiles import (
        StrictCapturedReplayProfile,
    )


MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA = (
    "nemo-rl-strict-model-transport-replay-consumption-v3"
)
MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA = (
    "nemo-rl-strict-model-transport-replay-consumption-entry-v1"
)
MODEL_TRANSPORT_REPLAY_MODE = "replay"
MODEL_TRANSPORT_REPLAY_STATUS = "complete-terminal"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_MAX_INT31 = (1 << 31) - 1
_MAX_INT63 = (1 << 63) - 1

_MATERIAL_KEYS = (
    "rollout_index",
    "generation_seed",
    "model_response",
    "agent_run_request",
    "derived_verifier_request",
    "source_entry_sha256",
    "request_body_sha256",
    "response_body_sha256",
)
_RAW_LOG_REF_KEYS = frozenset({"path", "record_schema", "record_count", "sha256"})
_ARTIFACT_REF_KEYS = frozenset({"path", "schema", "sha256"})
_SOURCE_KEYS = frozenset(
    {
        "arm",
        "authenticated_job_id",
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "main_ledger",
        "transcript_bundle",
        "transport_bundle",
        "transport_manifest",
        "raw_log",
        "ordered_entries_sha256",
    }
)
_REPLAY_KEYS = frozenset(
    {
        "attempt_id",
        "replay_execution_manifest_sha256",
        "authenticated_job_id",
        "process",
        "scheduler_device_environment",
        "scorer_evidence",
    }
)
_PROCESS_KEYS = frozenset({"boot_id_sha256", "pid", "start_time_ticks"})
_CONSUMPTION_ENTRY_KEYS = frozenset(
    {
        "schema",
        "rollout_index",
        "generation_seed",
        "source_model_transport_entry_sha256",
        "source_request_body_sha256",
        "source_response_body_sha256",
        "generation_request_sha256",
        "model_response_sha256",
        "agent_run_request_sha256",
        "derived_verifier_request_sha256",
        "fresh_verifier_response_sha256",
        "fresh_native_reward",
        "entry_sha256",
    }
)
_CONSUMPTION_BASE_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "status",
        "mode",
        "pair_id",
        "environment",
        "source",
        "replay",
        "entry_count",
        "entries",
        "ordered_entries_sha256",
    }
)

_SCORER_EVIDENCE_KEYS = frozenset(
    {"status", "terminal_index", "original_process_reaped"}
)
_CONSUMPTION_ROOT_KEYS = _CONSUMPTION_BASE_ROOT_KEYS | frozenset({"scorer_profile"})
_TRANSPORT_FILENAME = "model-transport-replay-consumption.json"


class StrictModelTransportReplayError(RuntimeError):
    """A profiled replay source was misused or failed terminal checks."""


@dataclass(frozen=True)
class ReplayVerifierMaterial(Mapping[str, Any]):
    """One detached OFF response and its reconstructed verifier request."""

    rollout_index: int
    generation_seed: int
    model_response: dict[str, Any]
    agent_run_request: dict[str, Any]
    derived_verifier_request: dict[str, Any]
    source_entry_sha256: str
    request_body_sha256: str
    response_body_sha256: str

    _KEYS: ClassVar[tuple[str, ...]] = _MATERIAL_KEYS

    def __post_init__(self) -> None:
        _logical_index(self.rollout_index, "material.rollout_index")
        _nonnegative_int63(self.generation_seed, "material.generation_seed")
        for name in (
            "source_entry_sha256",
            "request_body_sha256",
            "response_body_sha256",
        ):
            _digest(getattr(self, name), f"material.{name}")
        for name in (
            "model_response",
            "agent_run_request",
            "derived_verifier_request",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"material.{name} must be an object")
            copied = copy.deepcopy(dict(value))
            canonical_ascii_json(copied)
            object.__setattr__(self, name, copied)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return copy.deepcopy(getattr(self, key))


def _profile_v3(
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> StrictCapturedReplayProfile:
    """Resolve a profile lazily only for an explicitly invoked V3 API."""
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )

    profile = get_strict_captured_replay_profile(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    file_schemas = dict(
        zip(
            profile.result_files,
            profile.result_file_schemas,
            strict=True,
        )
    )
    if (
        len(profile.result_files) != 13
        or file_schemas.get(_TRANSPORT_FILENAME)
        != MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA
        or profile.scorer_terminal_index_path not in file_schemas
        or file_schemas.get(profile.scorer_terminal_index_path)
        != profile.call_index_schema
        or _TRANSPORT_FILENAME not in profile.result_anchor_paths
        or profile.scorer_terminal_index_path not in profile.result_anchor_paths
    ):
        raise RuntimeError("profile registry V3 result roster differs")
    return profile


def _scorer_profile_v3(
    profile: StrictCapturedReplayProfile,
) -> dict[str, Any]:
    """Project the complete closed scorer identity into the V3 wire shape."""
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


def _profile_reward(
    profile: StrictCapturedReplayProfile,
    reward: Any,
) -> float:
    if type(reward) is not float or not math.isfinite(reward):
        raise TypeError("profiled replay reward must be a finite float")
    if reward == 0.0 and math.copysign(1.0, reward) < 0.0:
        raise ValueError("profiled replay reward must not be negative zero")
    if profile.environment in {"citation", "freeform"}:
        if reward not in (0.0, 1.0):
            raise ValueError("profiled replay format reward must be binary")
    elif profile.environment == "reasoning_gym":
        if not 0.0 <= reward <= 1.0:
            raise ValueError("profiled replay reasoning reward is outside [0,1]")
    else:
        raise ValueError("unsupported profiled replay reward environment")
    return reward


class StrictModelTransportReplaySourceV3:
    """Exact-once source with one closed scorer profile."""

    def __init__(
        self,
        *,
        expected_environment: str,
        expected_profile_id: str,
        pair_id: str,
        attempt_id: str,
        replay_attempt_root: str,
        source_job_id: str,
        pair_manifest_sha256: str,
        source_submission_receipt_sha256: str,
        source_refs: Mapping[str, Any],
        materials: tuple[ReplayVerifierMaterial, ...],
        transcript_document: Mapping[str, Any],
        main_ledger_document: Mapping[str, Any],
    ) -> None:
        profile = _profile_v3(
            expected_environment=expected_environment,
            expected_profile_id=expected_profile_id,
        )
        if len(materials) != K4_SAMPLES:
            raise ValueError("profiled replay source requires exact K=4 material")
        self._pair_id = _pair_id(pair_id)
        self._attempt_id = _attempt_id(attempt_id)
        self._replay_attempt_root = str(
            _canonical_absolute_path(
                replay_attempt_root,
                "replay attempt root",
            )
        )
        self._source_job_id = _job_id(source_job_id, "source job_id")
        self._pair_manifest_sha256 = _digest(
            pair_manifest_sha256,
            "pair_manifest_sha256",
        )
        self._source_submission_receipt_sha256 = _digest(
            source_submission_receipt_sha256,
            "source_submission_receipt_sha256",
        )
        self._source_refs = copy.deepcopy(dict(source_refs))
        self._materials = tuple(materials)
        if [item.rollout_index for item in materials] != list(range(K4_SAMPLES)):
            raise ValueError("profiled replay materials are not logical 0..3")
        self._transcript_document = copy.deepcopy(dict(transcript_document))
        self._main_ledger_document = copy.deepcopy(dict(main_ledger_document))
        self._consumed: dict[int, ReplayVerifierMaterial] = {}
        self._fresh_results: dict[int, dict[str, Any]] = {}
        self._phase = "open"
        self._failure: str | None = None
        self._lock = threading.RLock()
        self._profile = profile

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    @property
    def source_transcript_document(self) -> dict[str, Any]:
        return copy.deepcopy(self._transcript_document)

    @property
    def source_main_ledger_document(self) -> dict[str, Any]:
        return copy.deepcopy(self._main_ledger_document)

    def consume(
        self,
        *,
        rollout_index: int,
        generation_seed: int,
    ) -> ReplayVerifierMaterial:
        """Consume one logical OFF entry; any misuse poisons the source."""
        with self._lock:
            self._require_open()
            try:
                index = _logical_index(rollout_index, "rollout_index")
                seed = _nonnegative_int63(generation_seed, "generation_seed")
                if index in self._consumed:
                    raise ValueError("logical replay entry was already consumed")
                retained = self._materials[index]
                if seed != retained.generation_seed:
                    raise ValueError("generation_seed differs from OFF source entry")
            except (TypeError, ValueError) as error:
                self._poison(str(error))
                raise StrictModelTransportReplayError(str(error)) from error
            self._consumed[index] = retained
            return ReplayVerifierMaterial(
                **{name: retained[name] for name in ReplayVerifierMaterial._KEYS}
            )

    def record_fresh_verifier_result(
        self,
        *,
        rollout_index: int,
        verifier_response: Mapping[str, Any],
    ) -> None:
        """Validate and bind one scorer result after consumption."""
        with self._lock:
            self._require_open()
            try:
                index = _logical_index(rollout_index, "rollout_index")
                if index not in self._consumed:
                    raise ValueError(
                        "fresh verifier result precedes source consumption"
                    )
                if index in self._fresh_results:
                    raise ValueError("fresh verifier result was already recorded")
                if not isinstance(verifier_response, Mapping):
                    raise TypeError("verifier_response must be an object")
                response = copy.deepcopy(dict(verifier_response))
                material = self._consumed[index]
                reward = validate_fresh_verifier_response(
                    environment=self._profile.environment,
                    agent_run_request=material.agent_run_request,
                    derived_verifier_request=(material.derived_verifier_request),
                    model_response=material.model_response,
                    verifier_response=response,
                )
                reward = _profile_reward(self._profile, reward)
                response_sha256 = domain_sha256(
                    "step1-verifier-response",
                    response,
                )
            except (TypeError, ValueError) as error:
                self._poison(str(error))
                raise StrictModelTransportReplayError(str(error)) from error
            self._fresh_results[index] = {
                "fresh_verifier_response_sha256": response_sha256,
                "fresh_native_reward": reward,
                "verifier_response": response,
            }

    def finalize(
        self,
        *,
        replay_execution_manifest_sha256: str,
        authenticated_job_id: str,
        process: Mapping[str, Any],
        scheduler_device_environment: Mapping[str, Any],
        scorer_call_index_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal V3 after the selected scorer's K=4 graph is reaped."""
        with self._lock:
            self._require_open()
            try:
                replay_sha = _digest(
                    replay_execution_manifest_sha256,
                    "replay_execution_manifest_sha256",
                )
                replay_job = _job_id(
                    authenticated_job_id,
                    "authenticated_job_id",
                )
                process_document = _process(process)
                device_environment = validate_scheduler_device_environment(
                    scheduler_device_environment
                )
                expected_indices = set(range(K4_SAMPLES))
                if set(self._consumed) != expected_indices:
                    raise ValueError("not all K=4 OFF entries were consumed")
                if set(self._fresh_results) != expected_indices:
                    raise ValueError("not all K=4 fresh verifier results were recorded")
                scorer_evidence = self._load_scorer_evidence(
                    scorer_call_index_ref,
                    authenticated_job_id=replay_job,
                )
                entries = [
                    self._consumption_entry(index) for index in range(K4_SAMPLES)
                ]
                document = {
                    "schema": MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA,
                    "hash_domain": HASH_DOMAIN,
                    "status": MODEL_TRANSPORT_REPLAY_STATUS,
                    "mode": MODEL_TRANSPORT_REPLAY_MODE,
                    "pair_id": self._pair_id,
                    "environment": self._profile.environment,
                    "scorer_profile": _scorer_profile_v3(self._profile),
                    "source": {
                        "arm": "off",
                        "authenticated_job_id": self._source_job_id,
                        "pair_manifest_sha256": self._pair_manifest_sha256,
                        "submission_receipt_sha256": (
                            self._source_submission_receipt_sha256
                        ),
                        **copy.deepcopy(self._source_refs),
                    },
                    "replay": {
                        "attempt_id": self._attempt_id,
                        "replay_execution_manifest_sha256": replay_sha,
                        "authenticated_job_id": replay_job,
                        "process": process_document,
                        "scheduler_device_environment": device_environment,
                        "scorer_evidence": scorer_evidence,
                    },
                    "entry_count": K4_SAMPLES,
                    "entries": entries,
                    "ordered_entries_sha256": domain_sha256(
                        "model-transport-replay-consumption-entries",
                        entries,
                    ),
                }
                validate_strict_model_transport_replay_consumption_v3(
                    document,
                    expected_environment=self._profile.environment,
                    expected_profile_id=self._profile.profile_id,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._poison(str(error))
                raise StrictModelTransportReplayError(str(error)) from error
            self._phase = MODEL_TRANSPORT_REPLAY_STATUS
            return copy.deepcopy(document)

    def _consumption_entry(self, index: int) -> dict[str, Any]:
        material = self._consumed[index]
        transcript_entry = self._transcript_document["entries"][index]
        entry: dict[str, Any] = {
            "schema": MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA,
            "rollout_index": index,
            "generation_seed": material.generation_seed,
            "source_model_transport_entry_sha256": (material.source_entry_sha256),
            "source_request_body_sha256": material.request_body_sha256,
            "source_response_body_sha256": material.response_body_sha256,
            "generation_request_sha256": transcript_entry["generation_request_sha256"],
            "model_response_sha256": transcript_entry["model_response_sha256"],
            "agent_run_request_sha256": transcript_entry["agent_run_request_sha256"],
            "derived_verifier_request_sha256": transcript_entry[
                "derived_verifier_request_sha256"
            ],
            "fresh_verifier_response_sha256": self._fresh_results[index][
                "fresh_verifier_response_sha256"
            ],
            "fresh_native_reward": self._fresh_results[index]["fresh_native_reward"],
        }
        entry["entry_sha256"] = domain_sha256(
            "model-transport-replay-consumption-entry",
            entry,
        )
        return entry

    def _load_format_scorer_evidence(
        self,
        reference: Mapping[str, Any],
        *,
        authenticated_job_id: str,
    ) -> dict[str, Any]:
        terminal_ref = _artifact_ref(
            reference,
            self._profile.call_index_schema,
            "format-verification call terminal index",
        )
        receipt_root = f"{self._replay_attempt_root}/strict_gym_child_runtime"
        expected_path = (
            f"{self._replay_attempt_root}/{self._profile.scorer_terminal_index_path}"
        )
        if terminal_ref["path"] != expected_path:
            raise ValueError("format-verification call-index path differs")

        from nemo_rl.environments.strict_gym_child_runtime_v2 import (
            format_verification_call_expectation,
            load_finalized_format_verification_call_index,
        )

        terminal, terminal_sha256 = load_finalized_format_verification_call_index(
            Path(terminal_ref["path"]),
            expected_sha256=terminal_ref["sha256"],
            expected_receipt_root=Path(receipt_root),
            expected_pair_id=self._pair_id,
            expected_job_id=authenticated_job_id,
            expected_environment=self._profile.environment,
            expected_profile_id=self._profile.profile_id,
        )
        if terminal_sha256 != terminal_ref["sha256"]:
            raise ValueError("format-verification call-index digest differs")
        if (
            terminal.get("schema") != self._profile.call_index_schema
            or terminal.get("environment") != self._profile.environment
            or terminal.get("profile_id") != self._profile.profile_id
            or terminal.get("scope") != "scorer-only"
            or terminal.get("pair_id") != self._pair_id
            or terminal.get("job_id") != authenticated_job_id
            or terminal.get("call_count") != K4_SAMPLES
            or type(terminal.get("call_count")) is not int
        ):
            raise ValueError("format-verification call-index identity differs")
        quiescence = terminal.get("quiescence")
        if (
            not isinstance(quiescence, Mapping)
            or quiescence.get("original_process_reaped") is not True
        ):
            raise ValueError("format-verification process was not reaped")
        calls = terminal.get("calls")
        if not isinstance(calls, list) or len(calls) != K4_SAMPLES:
            raise ValueError("format-verification call-index must contain K=4")
        for index, actual in enumerate(calls):
            material = self._materials[index]
            format_request = {
                name: copy.deepcopy(material.derived_verifier_request[name])
                for name in (
                    "responses_create_params",
                    "response",
                    "verifier",
                )
            }
            expected = format_verification_call_expectation(
                environment=self._profile.environment,
                derived_verifier_request=format_request,
                verifier_response=self._fresh_results[index]["verifier_response"],
            )
            if not isinstance(actual, Mapping):
                raise TypeError(
                    f"format-verification call-index entry {index + 1} "
                    "must be an object"
                )
            expected_record = {
                "sequence": index + 1,
                "method": expected["method"],
                "input": {
                    name: expected[name]
                    for name in (
                        "request_sha256",
                        "verifier_sha256",
                        "response_text_sha256",
                    )
                },
                "outcome": {
                    "kind": "returned",
                    "response_sha256": expected["response_sha256"],
                    "match_details_sha256": expected["match_details_sha256"],
                    "float_result": expected["float_result"],
                },
                "receipt": copy.deepcopy(actual.get("receipt")),
            }
            _exact_document(
                actual,
                expected_record,
                f"format-verification call-index entry {index + 1}",
            )
        return {
            "status": "authenticated",
            "terminal_index": terminal_ref,
            "original_process_reaped": True,
        }

    def _load_scorer_evidence(
        self,
        reference: Mapping[str, Any],
        *,
        authenticated_job_id: str,
    ) -> dict[str, Any]:
        if self._profile.environment in {"citation", "freeform"}:
            return self._load_format_scorer_evidence(
                reference,
                authenticated_job_id=authenticated_job_id,
            )
        if self._profile.environment == "reasoning_gym":
            if (
                self._profile.profile_id != "reasoning-gym-exact-match-v1"
                or self._profile.verifier_type != "score_answer"
                or self._profile.method != "KnightsKnavesDataset.score_answer"
            ):
                raise ValueError("reasoning scorer profile identity differs")
            return self._load_reasoning_scorer_evidence(
                reference,
                authenticated_job_id=authenticated_job_id,
            )
        raise ValueError("unsupported scorer evidence environment")

    def _load_reasoning_scorer_evidence(
        self,
        reference: Mapping[str, Any],
        *,
        authenticated_job_id: str,
    ) -> dict[str, Any]:
        terminal_ref = _artifact_ref(
            reference,
            self._profile.call_index_schema,
            "reasoning score-call terminal index",
        )
        receipt_root = f"{self._replay_attempt_root}/strict_gym_child_runtime"
        expected_path = (
            f"{self._replay_attempt_root}/{self._profile.scorer_terminal_index_path}"
        )
        if terminal_ref["path"] != expected_path:
            raise ValueError("reasoning score-call index path differs")

        from nemo_rl.environments.strict_gym_child_runtime_v2 import (
            load_finalized_reasoning_score_call_index,
        )

        terminal, terminal_sha256 = load_finalized_reasoning_score_call_index(
            Path(terminal_ref["path"]),
            expected_sha256=terminal_ref["sha256"],
            expected_receipt_root=Path(receipt_root),
            expected_pair_id=self._pair_id,
            expected_job_id=authenticated_job_id,
        )
        if terminal_sha256 != terminal_ref["sha256"]:
            raise ValueError("reasoning score-call index digest differs")
        if (
            terminal.get("schema") != self._profile.call_index_schema
            or terminal.get("environment") != "reasoning_gym"
            or terminal.get("scope") != "scorer-only"
            or terminal.get("pair_id") != self._pair_id
            or terminal.get("job_id") != authenticated_job_id
            or terminal.get("call_count") != K4_SAMPLES
            or type(terminal.get("call_count")) is not int
        ):
            raise ValueError("reasoning score-call index identity differs")
        quiescence = terminal.get("quiescence")
        if (
            not isinstance(quiescence, Mapping)
            or quiescence.get("original_process_reaped") is not True
        ):
            raise ValueError("reasoning scorer process was not reaped")
        calls = terminal.get("calls")
        if not isinstance(calls, list) or len(calls) != K4_SAMPLES:
            raise ValueError("reasoning score-call index must contain K=4")
        for index, actual in enumerate(calls):
            if not isinstance(actual, Mapping):
                raise TypeError(
                    f"reasoning score-call index entry {index + 1} must be an object"
                )
            expected = self._expected_reasoning_score_call(index)
            expected_record = {
                "sequence": index + 1,
                "task_name": expected["task_name"],
                "input": expected["input"],
                "float_result": expected["float_result"],
                "receipt": copy.deepcopy(actual.get("receipt")),
            }
            _exact_document(
                actual,
                expected_record,
                f"reasoning score-call index entry {index + 1}",
            )
        return {
            "status": "authenticated",
            "terminal_index": terminal_ref,
            "original_process_reaped": True,
        }

    def _expected_reasoning_score_call(self, rollout_index: int) -> dict[str, Any]:
        material = self._materials[rollout_index]
        request = material.derived_verifier_request
        response = self._fresh_results[rollout_index]["verifier_response"]
        metadata = request.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("reasoning scorer metadata must be an object")
        task_name = metadata.get("source_dataset")
        if type(task_name) is not str or task_name != "knights_knaves":
            raise ValueError("reasoning scorer task must be knights_knaves")
        scorer_entry = {
            "question": copy.deepcopy(request.get("question")),
            "answer": copy.deepcopy(request.get("answer")),
            "metadata": copy.deepcopy(dict(metadata)),
        }
        return {
            "task_name": task_name,
            "input": {
                "answer_sha256": hashlib.sha256(
                    canonical_ascii_json(response.get("extracted_answer"))
                ).hexdigest(),
                "entry_sha256": hashlib.sha256(
                    canonical_ascii_json(scorer_entry)
                ).hexdigest(),
            },
            "float_result": self._fresh_results[rollout_index]["fresh_native_reward"],
        }

    def _require_open(self) -> None:
        if self._phase != "open":
            suffix = f": {self._failure}" if self._failure is not None else ""
            raise StrictModelTransportReplayError(
                f"model transport replay source is {self._phase}{suffix}"
            )

    def _poison(self, reason: str) -> None:
        self._phase = "poisoned"
        self._failure = reason


def load_strict_model_transport_replay_source_v3(
    *,
    source: AuthenticatedOffSourceCapture,
    replay_execution_manifest: Mapping[str, Any],
    expected_environment: str,
    expected_profile_id: str,
) -> StrictModelTransportReplaySourceV3:
    """Reload one scorer source under explicit V3 manifest/profile authority."""
    profile = _profile_v3(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )

    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        AuthenticatedOffSourceCapture,
        validate_replay_execution_manifest_v2,
    )

    if not isinstance(source, AuthenticatedOffSourceCapture):
        raise TypeError("source must be an AuthenticatedOffSourceCapture")
    validate_replay_execution_manifest_v2(
        replay_execution_manifest,
        authenticated_source=source,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    _exact_document(
        replay_execution_manifest["source_capture"],
        source.source_capture,
        "authenticated replay source capture",
    )
    pair = _pair_id(replay_execution_manifest["pair_id"])
    environment = replay_execution_manifest["environment"]
    if environment != profile.environment:
        raise ValueError("replay source differs from expected scorer profile")
    if replay_execution_manifest.get("scorer_profile") != _scorer_profile_v3(profile):
        raise ValueError("replay source scorer profile differs from registry")
    attempt_id = _attempt_id(replay_execution_manifest["attempt_id"])
    pair_sha = _digest(
        source.pair_manifest_sha256,
        "pair_manifest_sha256",
    )
    receipt_sha = _digest(
        source.pair_submission_receipt_sha256,
        "source_submission_receipt_sha256",
    )
    if replay_execution_manifest["pair"]["manifest"]["sha256"] != pair_sha:
        raise ValueError("replay manifest Pair SHA differs from authenticated source")
    if replay_execution_manifest["pair"]["submission_receipt"]["sha256"] != receipt_sha:
        raise ValueError(
            "replay manifest Pair receipt SHA differs from authenticated source"
        )
    capture = source.source_capture
    job_id = _job_id(
        capture["authenticated_job"]["job_id"],
        "source authenticated_job_id",
    )
    step1 = capture["step1_evidence"]
    transport = step1["model_transport"]
    raw_reference = _raw_log_ref(transport["raw_log"])
    bundle_reference = _artifact_ref(
        transport["bundle"],
        MODEL_TRANSPORT_BUNDLE_SCHEMA,
        "bundle_ref",
    )
    manifest_reference = _artifact_ref(
        transport["manifest"],
        MODEL_TRANSPORT_MANIFEST_SCHEMA,
        "transport_manifest_ref",
    )
    transcript_reference = _artifact_ref(
        step1["transcript_bundle"],
        TRANSCRIPT_BUNDLE_SCHEMA,
        "transcript_ref",
    )
    main_ledger_reference = _artifact_ref(
        step1["main_ledger"],
        MAIN_STEP1_LEDGER_SCHEMA,
        "main ledger ref",
    )

    manifest = _load_artifact_document(manifest_reference)
    validate_model_transport_manifest(manifest)
    _close_source_identity(
        manifest,
        pair_id=pair,
        environment=environment,
        source_job_id=job_id,
        pair_manifest_sha256=pair_sha,
        source_submission_receipt_sha256=receipt_sha,
    )
    _exact_document(
        manifest["transport_capture"],
        raw_reference,
        "raw log ref",
    )
    _exact_document(
        manifest["transport_bundle"],
        bundle_reference,
        "bundle ref",
    )
    _exact_document(
        manifest["main_transcript_bundle"],
        transcript_reference,
        "transcript ref",
    )

    bundle = _load_artifact_document(bundle_reference)
    transcript = _load_artifact_document(transcript_reference)
    _exact_document(
        manifest["main_ledger"],
        main_ledger_reference,
        "main ledger ref",
    )
    main_ledger = _load_artifact_document(main_ledger_reference)
    raw_log = _load_immutable_bytes(
        Path(raw_reference["path"]),
        expected_sha256=raw_reference["sha256"],
        maximum=MAX_BUNDLE_BYTES,
    )
    _exact_document(
        manifest,
        source.transport_manifest,
        "authenticated manifest",
    )
    _exact_document(
        bundle,
        source.transport_bundle,
        "authenticated bundle",
    )
    _exact_document(
        transcript,
        source.transcript_bundle,
        "authenticated transcript",
    )
    _exact_document(
        main_ledger,
        source.main_ledger,
        "authenticated main ledger",
    )

    validate_transcript_bundle(transcript)
    validate_main_step1_ledger(main_ledger)
    _close_off_transcript(
        transcript,
        pair_id=pair,
        environment=environment,
        source_job_id=job_id,
        pair_manifest_sha256=pair_sha,
        source_submission_receipt_sha256=receipt_sha,
        bundle_ref=bundle_reference,
    )
    _close_off_ledger(
        main_ledger,
        transcript_ref=transcript_reference,
        pair_id=pair,
        environment=environment,
        source_job_id=job_id,
        pair_manifest_sha256=pair_sha,
        source_submission_receipt_sha256=receipt_sha,
    )
    validate_ledger_transcript_join(
        ledger=main_ledger,
        transcript_bundle=transcript,
    )

    model_path = _bundle_model_path(bundle)
    expected_policy_sha256 = replay_execution_manifest["pair"][
        "model_transport_policy_sha256"
    ]
    pair_policy = source.pair_manifest.get("model_transport")
    if not isinstance(pair_policy, Mapping):
        raise TypeError("authenticated Pair model_transport policy is missing")
    pair_source = source.pair_manifest.get("source")
    if not isinstance(pair_source, Mapping):
        raise TypeError("authenticated Pair source is missing")
    snapshots = pair_source.get("snapshots")
    if not isinstance(snapshots, Mapping) or not isinstance(
        snapshots.get("off"),
        Mapping,
    ):
        raise TypeError("authenticated Pair OFF snapshot is missing")
    source_root = _canonical_absolute_path(
        snapshots["off"].get("path"),
        "authenticated Pair OFF snapshot path",
    )
    policy = load_attested_model_transport_policy(
        model_transport_policy=pair_policy,
        expected_policy_sha256=expected_policy_sha256,
        source_root=source_root,
    )
    if manifest["model_transport_policy_sha256"] != expected_policy_sha256:
        raise ValueError(
            "OFF transport manifest policy differs from authenticated Pair"
        )
    validate_transcript_model_transport_join(
        transcript_bundle=transcript,
        model_transport_bundle=bundle,
        model_transport_policy=policy,
        model_path=model_path,
    )
    if bundle.get("arm") != "off":
        raise ValueError("captured replay source transport arm must be off")
    if manifest["capture_server"] != bundle.get("capture_server"):
        raise ValueError("transport manifest/bundle capture server differs")
    if manifest["entry_count"] != bundle.get("entry_count"):
        raise ValueError("transport manifest/bundle entry count differs")
    if manifest["ordered_entries_sha256"] != bundle.get("ordered_entries_sha256"):
        raise ValueError("transport manifest/bundle ordered digest differs")
    expected_raw = b"".join(
        canonical_ascii_json(entry) + b"\n" for entry in bundle["entries"]
    )
    if raw_log != expected_raw:
        raise ValueError("raw transport log differs from canonical bundle entries")
    if tuple(copy.deepcopy(bundle["entries"])) != tuple(source.transport_records):
        raise ValueError("raw transport records differ from authenticated source")

    materials = tuple(
        ReplayVerifierMaterial(
            rollout_index=index,
            generation_seed=transcript_entry["generation_seed"],
            model_response=transcript_entry["model_response"],
            agent_run_request=transcript_entry["agent_run_request"],
            derived_verifier_request=transcript_entry["derived_verifier_request"],
            source_entry_sha256=transport_entry["entry_sha256"],
            request_body_sha256=transport_entry["request_body_sha256"],
            response_body_sha256=transport_entry["response_body_sha256"],
        )
        for index, (transcript_entry, transport_entry) in enumerate(
            zip(transcript["entries"], bundle["entries"], strict=True)
        )
    )
    source_refs = {
        "main_ledger": main_ledger_reference,
        "transcript_bundle": transcript_reference,
        "transport_bundle": bundle_reference,
        "transport_manifest": manifest_reference,
        "raw_log": raw_reference,
        "ordered_entries_sha256": bundle["ordered_entries_sha256"],
    }
    return StrictModelTransportReplaySourceV3(
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
        pair_id=pair,
        attempt_id=attempt_id,
        replay_attempt_root=replay_execution_manifest["artifacts"]["outputs"][
            "directory"
        ]["path"],
        source_job_id=job_id,
        pair_manifest_sha256=pair_sha,
        source_submission_receipt_sha256=receipt_sha,
        source_refs=source_refs,
        materials=materials,
        transcript_document=transcript,
        main_ledger_document=main_ledger,
    )


def validate_strict_model_transport_replay_consumption_v3(
    document: Mapping[str, Any],
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> None:
    """Validate one terminal V3 document against an explicit scorer profile."""
    profile = _profile_v3(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    _exact_keys(
        document,
        _CONSUMPTION_ROOT_KEYS,
        "profiled replay consumption",
    )
    if document["schema"] != MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA:
        raise ValueError("unexpected profiled replay consumption schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected profiled replay consumption hash domain")
    if (
        document["status"] != MODEL_TRANSPORT_REPLAY_STATUS
        or document["mode"] != MODEL_TRANSPORT_REPLAY_MODE
    ):
        raise ValueError("profiled replay terminal mode/status differs")
    _pair_id(document["pair_id"])
    if document["environment"] != profile.environment:
        raise ValueError("profiled replay consumption environment differs")
    _exact_document(
        document["scorer_profile"],
        _scorer_profile_v3(profile),
        "profiled replay scorer profile",
    )

    source = document["source"]
    _exact_keys(
        source,
        _SOURCE_KEYS,
        "profiled replay consumption source",
    )
    if source["arm"] != "off":
        raise ValueError("profiled replay consumption source arm must be off")
    _job_id(
        source["authenticated_job_id"],
        "source.authenticated_job_id",
    )
    _digest(
        source["pair_manifest_sha256"],
        "source.pair_manifest_sha256",
    )
    _digest(
        source["submission_receipt_sha256"],
        "source.submission_receipt_sha256",
    )
    _artifact_ref(
        source["main_ledger"],
        MAIN_STEP1_LEDGER_SCHEMA,
        "source.main_ledger",
    )
    _artifact_ref(
        source["transport_bundle"],
        MODEL_TRANSPORT_BUNDLE_SCHEMA,
        "source.transport_bundle",
    )
    _artifact_ref(
        source["transport_manifest"],
        MODEL_TRANSPORT_MANIFEST_SCHEMA,
        "source.transport_manifest",
    )
    _raw_log_ref(source["raw_log"])
    _artifact_ref(
        source["transcript_bundle"],
        TRANSCRIPT_BUNDLE_SCHEMA,
        "source.transcript_bundle",
    )
    _digest(
        source["ordered_entries_sha256"],
        "source.ordered_entries_sha256",
    )

    replay = document["replay"]
    _exact_keys(
        replay,
        _REPLAY_KEYS,
        "profiled replay consumption replay",
    )
    _attempt_id(replay["attempt_id"])
    _digest(
        replay["replay_execution_manifest_sha256"],
        "replay.replay_execution_manifest_sha256",
    )
    _job_id(
        replay["authenticated_job_id"],
        "replay.authenticated_job_id",
    )
    _process(replay["process"])
    validate_scheduler_device_environment(replay["scheduler_device_environment"])
    scorer = replay["scorer_evidence"]
    _exact_keys(
        scorer,
        _SCORER_EVIDENCE_KEYS,
        "profiled replay scorer evidence",
    )
    if (
        scorer["status"] != "authenticated"
        or scorer["original_process_reaped"] is not True
    ):
        raise ValueError("profiled replay scorer evidence is not reaped/authenticated")
    terminal_ref = _artifact_ref(
        scorer["terminal_index"],
        profile.call_index_schema,
        "profiled replay scorer terminal index",
    )
    terminal_path = str(
        _canonical_absolute_path(
            terminal_ref["path"],
            "profiled replay scorer terminal-index path",
        )
    )
    if not terminal_path.endswith(f"/{profile.scorer_terminal_index_path}"):
        raise ValueError("profiled replay scorer terminal-index path differs")

    if type(document["entry_count"]) is not int or document["entry_count"] != 4:
        raise ValueError("profiled replay entry_count must be exact integer 4")
    entries = document["entries"]
    if not isinstance(entries, list) or len(entries) != K4_SAMPLES:
        raise ValueError("profiled replay entries must be exact K=4")
    for index, raw_entry in enumerate(entries):
        _exact_keys(
            raw_entry,
            _CONSUMPTION_ENTRY_KEYS,
            f"entry[{index}]",
        )
        if (
            raw_entry["rollout_index"] != index
            or type(raw_entry["rollout_index"]) is not int
        ):
            raise ValueError("profiled replay entries are not logical 0..3")
        if raw_entry["schema"] != MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA:
            raise ValueError("unexpected profiled replay consumption entry schema")
        _nonnegative_int63(
            raw_entry["generation_seed"],
            "generation_seed",
        )
        if raw_entry["generation_seed"] != derive_nemo_gym_request_seed(
            seed_base=42,
            fixture_row_index=0,
            rollout_index=index,
        ):
            raise ValueError("profiled replay generation seed does not close")
        for name in _CONSUMPTION_ENTRY_KEYS - {
            "schema",
            "rollout_index",
            "generation_seed",
            "fresh_native_reward",
        }:
            _digest(raw_entry[name], f"entry[{index}].{name}")
        reward = _profile_reward(profile, raw_entry["fresh_native_reward"])
        projection = {
            key: copy.deepcopy(value)
            for key, value in raw_entry.items()
            if key != "entry_sha256"
        }
        if raw_entry["entry_sha256"] != domain_sha256(
            "model-transport-replay-consumption-entry",
            projection,
        ):
            raise ValueError(f"profiled replay entry {index} digest differs")
    if document["ordered_entries_sha256"] != domain_sha256(
        "model-transport-replay-consumption-entries",
        entries,
    ):
        raise ValueError("profiled replay consumption ordered digest differs")
    canonical_ascii_json(document)


def publish_strict_model_transport_replay_consumption_v3(
    *,
    output: str | Path,
    document: Mapping[str, Any],
    expected_environment: str,
    expected_profile_id: str,
) -> tuple[Path, str]:
    """Publish one explicitly selected V3 file at its fixed roster path."""
    profile = _profile_v3(
        expected_environment=expected_environment,
        expected_profile_id=expected_profile_id,
    )
    validate_strict_model_transport_replay_consumption_v3(
        document,
        expected_environment=profile.environment,
        expected_profile_id=profile.profile_id,
    )
    terminal_path = _canonical_absolute_path(
        document["replay"]["scorer_evidence"]["terminal_index"]["path"],
        "profiled replay scorer terminal-index path",
    )
    expected_output = terminal_path.parent.parent / _TRANSPORT_FILENAME
    output_path = _canonical_absolute_path(
        output,
        "profiled replay consumption output",
    )
    if output_path != expected_output:
        raise ValueError("profiled replay consumption output path differs")
    return publish_evidence_document(
        output=output_path,
        document=document,
        trailing_lf=False,
    )


def _close_source_identity(
    manifest: Mapping[str, Any],
    *,
    pair_id: str,
    environment: str,
    source_job_id: str,
    pair_manifest_sha256: str,
    source_submission_receipt_sha256: str,
) -> None:
    expected = {
        "pair_id": pair_id,
        "environment": environment,
        "arm": "off",
        "authenticated_job_id": source_job_id,
        "pair_manifest_sha256": pair_manifest_sha256,
        "submission_receipt_sha256": source_submission_receipt_sha256,
    }
    for name, value in expected.items():
        if manifest.get(name) != value or type(manifest.get(name)) is not type(value):
            raise ValueError(f"transport manifest source {name} differs")


def _close_off_transcript(
    transcript: Mapping[str, Any],
    *,
    pair_id: str,
    environment: str,
    source_job_id: str,
    pair_manifest_sha256: str,
    source_submission_receipt_sha256: str,
    bundle_ref: Mapping[str, Any],
) -> None:
    if (
        transcript.get("pair_id") != pair_id
        or transcript.get("environment") != environment
        or transcript.get("arm") != "off"
        or transcript.get("mode") != "observe"
        or transcript.get("attempt_id") is not None
    ):
        raise ValueError("source transcript is not the exact OFF main cohort")
    bindings = transcript["bindings"]
    expected = {
        "job_id": source_job_id,
        "pair_manifest_sha256": pair_manifest_sha256,
        "submission_receipt_sha256": source_submission_receipt_sha256,
    }
    for name, value in expected.items():
        if bindings.get(name) != value:
            raise ValueError(f"source transcript binding {name} differs")
    _exact_document(
        transcript["model_transport_bundle"],
        bundle_ref,
        "transcript bundle ref",
    )


def _close_off_ledger(
    ledger: Mapping[str, Any],
    *,
    transcript_ref: Mapping[str, Any],
    pair_id: str,
    environment: str,
    source_job_id: str,
    pair_manifest_sha256: str,
    source_submission_receipt_sha256: str,
) -> None:
    if (
        ledger.get("pair_id") != pair_id
        or ledger.get("environment") != environment
        or ledger.get("arm") != "off"
        or ledger.get("mode") != "observe"
    ):
        raise ValueError("source main ledger is not the exact OFF main cohort")
    _exact_document(
        ledger["transcript_bundle"],
        transcript_ref,
        "ledger transcript",
    )
    bindings = ledger["bindings"]
    expected = {
        "job_id": source_job_id,
        "pair_manifest_sha256": pair_manifest_sha256,
        "submission_receipt_sha256": source_submission_receipt_sha256,
    }
    for name, value in expected.items():
        if bindings.get(name) != value:
            raise ValueError(f"source main ledger binding {name} differs")


def _bundle_model_path(bundle: Mapping[str, Any]) -> str:
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != K4_SAMPLES:
        raise ValueError("source model transport bundle must contain K=4 entries")
    first = entries[0]
    if not isinstance(first, Mapping):
        raise TypeError("source model transport entry must be an object")
    payload = first.get("request_payload")
    if not isinstance(payload, Mapping):
        raise TypeError("source model transport request payload must be an object")
    model_path = payload.get("model")
    if type(model_path) is not str or not model_path:
        raise ValueError("source model transport model path is malformed")
    return model_path


def _artifact_ref(
    value: Mapping[str, Any],
    expected_schema: str,
    name: str,
) -> dict[str, Any]:
    _exact_keys(value, _ARTIFACT_REF_KEYS, name)
    result = dict(value)
    result["path"] = str(_canonical_absolute_path(result["path"], f"{name}.path"))
    if result["schema"] != expected_schema:
        raise ValueError(f"{name}.schema differs")
    result["sha256"] = _digest(result["sha256"], f"{name}.sha256")
    return result


def _raw_log_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, _RAW_LOG_REF_KEYS, "raw_log_ref")
    result = dict(value)
    result["path"] = str(_canonical_absolute_path(result["path"], "raw_log_ref.path"))
    if result["record_schema"] != MODEL_TRANSPORT_CALL_SCHEMA:
        raise ValueError("raw_log_ref record schema differs")
    if type(result["record_count"]) is not int or result["record_count"] != 4:
        raise ValueError("raw_log_ref record count must be exact integer 4")
    result["sha256"] = _digest(result["sha256"], "raw_log_ref.sha256")
    return result


def _process(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, _PROCESS_KEYS, "process")
    result = dict(value)
    result["boot_id_sha256"] = _digest(
        result["boot_id_sha256"],
        "process.boot_id_sha256",
    )
    if type(result["pid"]) is not int or not 1 <= result["pid"] <= _MAX_INT31:
        raise ValueError("process.pid must be an exact positive int31")
    if (
        type(result["start_time_ticks"]) is not int
        or not 1 <= result["start_time_ticks"] <= _MAX_INT63
    ):
        raise ValueError("process.start_time_ticks must be an exact positive int63")
    return result


def _load_artifact_document(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Load canonical JSON and independently stable-read the named inode."""
    document, _ = load_evidence_document(
        path=reference["path"],
        expected_sha256=reference["sha256"],
        trailing_lf=False,
    )
    raw = _load_immutable_bytes(
        Path(reference["path"]),
        expected_sha256=reference["sha256"],
        maximum=MAX_BUNDLE_BYTES,
    )
    if raw != canonical_ascii_json(document):
        raise RuntimeError("artifact changed between canonical and stable reads")
    return document


def _load_immutable_bytes(
    path: Path,
    *,
    expected_sha256: str,
    maximum: int,
) -> bytes:
    expected = _digest(expected_sha256, "immutable bytes expected SHA-256")
    canonical = _canonical_absolute_path(path, "immutable bytes path")
    _reject_symlink_components(canonical)
    named_before = os.lstat(canonical)
    descriptor = os.open(
        canonical,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise RuntimeError(
                "raw transport evidence must be EUID-owned mode0400 nlink1"
            )
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        named_after = os.lstat(canonical)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)

    def fingerprint(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        len(raw) > maximum
        or fingerprint(named_before) != fingerprint(before)
        or fingerprint(before) != fingerprint(after)
        or fingerprint(after) != fingerprint(named_after)
    ):
        raise RuntimeError("raw transport evidence changed during stable read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("raw transport evidence SHA-256 differs")
    return raw


def _reject_symlink_components(path: Path) -> None:
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        if stat.S_ISLNK(os.lstat(cursor).st_mode):
            raise ValueError(f"secure path contains a symlink: {cursor}")


def _canonical_absolute_path(value: Any, name: str) -> Path:
    text = str(value) if isinstance(value, Path) else value
    if type(text) is not str or not text.startswith("/") or text.startswith("//"):
        raise ValueError(f"{name} must be a canonical absolute path")
    if text != posixpath.normpath(text) or "//" in text:
        raise ValueError(f"{name} must be a canonical absolute path")
    return Path(text)


def _exact_document(actual: Any, expected: Any, name: str) -> None:
    if canonical_ascii_json(actual) != canonical_ascii_json(expected):
        raise ValueError(f"{name} differs")


def _exact_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} keys differ")


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or _DIGEST_RE.fullmatch(value) is None
        or value == "0" * 64
    ):
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256 digest")
    return value


def _pair_id(value: Any) -> str:
    if type(value) is not str or _PAIR_ID_RE.fullmatch(value) is None:
        raise ValueError("pair_id is malformed")
    return value


def _job_id(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or str(int(value)) != value
        or not 1 <= int(value) <= _MAX_INT63
    ):
        raise ValueError(f"{name} must be a canonical positive decimal string")
    return value


def _logical_index(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value < K4_SAMPLES:
        raise ValueError(f"{name} must be an exact logical index 0..3")
    return value


def _attempt_id(value: Any) -> str:
    if type(value) is not str or value not in {"replay-1", "replay-2"}:
        raise ValueError("attempt_id must be replay-1 or replay-2")
    return value


def _nonnegative_int63(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_INT63:
        raise ValueError(f"{name} must be an exact nonnegative int63")
    return value


__all__ = [
    "MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA",
    "MODEL_TRANSPORT_REPLAY_CONSUMPTION_V3_SCHEMA",
    "MODEL_TRANSPORT_REPLAY_MODE",
    "MODEL_TRANSPORT_REPLAY_STATUS",
    "ReplayVerifierMaterial",
    "StrictModelTransportReplayError",
    "StrictModelTransportReplaySourceV3",
    "load_strict_model_transport_replay_source_v3",
    "publish_strict_model_transport_replay_consumption_v3",
    "validate_strict_model_transport_replay_consumption_v3",
]
