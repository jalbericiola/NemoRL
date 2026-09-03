"""Fail-closed consumption of authenticated OFF model-transport evidence.

Captured replay is a scorer-only operation.  This module therefore exposes the
already-captured model response and reconstructed verifier request, but contains
no model client, SimpleAgent client, or generation fallback.  A source entry can
be consumed once, must receive one validated fresh verifier result, and can only
then participate in the terminal consumption document.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import posixpath
import re
import signal
import stat
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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
from nemo_rl.utils.strict_captured_replay_manifest import (
    AuthenticatedOffSourceCapture,
    validate_replay_execution_manifest,
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


MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA = (
    "nemo-rl-strict-model-transport-replay-consumption-v2"
)
MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA = (
    "nemo-rl-strict-model-transport-replay-consumption-entry-v1"
)
MODEL_TRANSPORT_REPLAY_MODE = "replay"
MODEL_TRANSPORT_REPLAY_STATUS = "complete-terminal"
REASONING_SCORE_CALL_SCHEMA = "nemo-rl-strict-reasoning-score-call-v1"
REASONING_SCORE_CALL_INDEX_SCHEMA = "nemo-rl-strict-reasoning-score-call-index-v1"
REASONING_SCORE_CLOSED_SCHEMA = "nemo-rl-strict-reasoning-score-closed-v1"
STRICT_GYM_CHILD_SPEC_SCHEMA = "nemo-rl-strict-gym-child-spec-v1"
STRICT_GYM_CHILD_INDEX_SCHEMA = "nemo-rl-strict-gym-child-index-v1"
STRICT_GYM_CHILD_RECEIPT_SCHEMA = "nemo-rl-strict-gym-child-receipt-v1"
STRICT_GYM_CHILD_HASH_DOMAIN = "sha256-canonical-ascii-json-no-lf-v1"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_ENVIRONMENTS = frozenset({"citation", "freeform", "reasoning_gym"})
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
_SCORER_EVIDENCE_KEYS = frozenset({"status", "terminal_index"})
_SCORE_INDEX_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "environment",
        "scope",
        "pair_id",
        "job_id",
        "spec",
        "child_index",
        "resource_receipt",
        "score_closed",
        "quiescence",
        "call_count",
        "calls",
    }
)
_SCORE_INDEX_CALL_KEYS = frozenset(
    {"sequence", "task_name", "input", "float_result", "receipt"}
)
_SCORE_CALL_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "environment",
        "pair_id",
        "job_id",
        "spec_sha256",
        "process",
        "sequence",
        "task_name",
        "input",
        "outcome",
    }
)
_SCORE_INPUT_KEYS = frozenset({"answer_sha256", "entry_sha256"})
_SCORE_PROCESS_KEYS = frozenset({"pid", "start_ticks"})
_SCORE_CLOSED_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "environment",
        "pair_id",
        "job_id",
        "spec_sha256",
        "process",
        "call_count",
        "calls",
    }
)
_SCORE_CLOSED_CALL_REF_KEYS = frozenset({"sequence", "path", "sha256", "schema"})
_SCORE_QUIESCENCE_KEYS = frozenset(
    {
        "pid",
        "start_ticks",
        "child_termination_signal",
        "wrapper_pid",
        "wrapper_returncode",
        "original_process_reaped",
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
_CONSUMPTION_ROOT_KEYS = frozenset(
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


class StrictModelTransportReplayError(RuntimeError):
    """A replay source was misused or one of its terminal checks failed."""


@dataclass(frozen=True)
class ReplayVerifierMaterial(Mapping[str, Any]):
    """One immutable-by-ownership logical OFF response for fresh scoring.

    The class also implements the runtime's closed Mapping contract.  Nested
    payloads are copied both when the material is constructed and when accessed
    through the Mapping interface, so a replay consumer cannot mutate retained
    source authority.
    """

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


class StrictModelTransportReplaySource:
    """Exact-once K=4 source loaded from one authenticated OFF capture."""

    def __init__(
        self,
        *,
        pair_id: str,
        environment: str,
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
        if len(materials) != K4_SAMPLES:
            raise ValueError("replay source requires exact K=4 material")
        self._pair_id = pair_id
        self._environment = environment
        self._attempt_id = attempt_id
        self._replay_attempt_root = str(
            _canonical_absolute_path(replay_attempt_root, "replay attempt root")
        )
        self._source_job_id = source_job_id
        self._pair_manifest_sha256 = pair_manifest_sha256
        self._source_submission_receipt_sha256 = source_submission_receipt_sha256
        self._source_refs = copy.deepcopy(dict(source_refs))
        self._materials = tuple(materials)
        self._transcript_document = copy.deepcopy(dict(transcript_document))
        self._main_ledger_document = copy.deepcopy(dict(main_ledger_document))
        self._consumed: dict[int, ReplayVerifierMaterial] = {}
        self._fresh_results: dict[int, dict[str, Any]] = {}
        self._phase = "open"
        self._failure: str | None = None
        self._lock = threading.RLock()

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
        self, *, rollout_index: int, generation_seed: int
    ) -> ReplayVerifierMaterial:
        """Consume one logical source entry; any misuse poisons the source."""
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
        self, *, rollout_index: int, verifier_response: Mapping[str, Any]
    ) -> None:
        """Validate and bind one fresh resource-server result after consumption."""
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
                    environment=self._environment,
                    agent_run_request=material.agent_run_request,
                    derived_verifier_request=material.derived_verifier_request,
                    model_response=material.model_response,
                    verifier_response=response,
                )
                response_sha256 = domain_sha256("step1-verifier-response", response)
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
        reasoning_score_call_index_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal terminal evidence after all K=4 source/result pairs close."""
        with self._lock:
            self._require_open()
            try:
                replay_sha = _digest(
                    replay_execution_manifest_sha256,
                    "replay_execution_manifest_sha256",
                )
                replay_job = _job_id(authenticated_job_id, "authenticated_job_id")
                process_document = _process(process)
                device_environment = validate_scheduler_device_environment(
                    scheduler_device_environment
                )
                expected = set(range(K4_SAMPLES))
                if set(self._consumed) != expected:
                    raise ValueError("not all K=4 OFF entries were consumed")
                if set(self._fresh_results) != expected:
                    raise ValueError("not all K=4 fresh verifier results were recorded")
                scorer_evidence = self._load_reasoning_scorer_evidence(
                    reasoning_score_call_index_ref,
                    authenticated_job_id=replay_job,
                )
                entries = [self._consumption_entry(index) for index in range(4)]
                document = {
                    "schema": MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA,
                    "hash_domain": HASH_DOMAIN,
                    "status": MODEL_TRANSPORT_REPLAY_STATUS,
                    "mode": MODEL_TRANSPORT_REPLAY_MODE,
                    "pair_id": self._pair_id,
                    "environment": self._environment,
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
                        "model-transport-replay-consumption-entries", entries
                    ),
                }
                validate_strict_model_transport_replay_consumption(document)
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
            "source_model_transport_entry_sha256": material.source_entry_sha256,
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
            "model-transport-replay-consumption-entry", entry
        )
        return entry

    def _load_reasoning_scorer_evidence(
        self,
        reference: Mapping[str, Any],
        *,
        authenticated_job_id: str,
    ) -> dict[str, Any]:
        terminal_ref = _artifact_ref(
            reference,
            REASONING_SCORE_CALL_INDEX_SCHEMA,
            "reasoning score-call terminal index",
        )
        expected_receipt_root = f"{self._replay_attempt_root}/strict_gym_child_runtime"
        if terminal_ref["path"] != (
            f"{expected_receipt_root}/reasoning-score-call-index.json"
        ):
            raise ValueError("reasoning score-call index path differs")
        from nemo_rl.environments.strict_gym_child_runtime import (
            load_finalized_reasoning_score_call_index,
        )

        terminal, terminal_sha256 = load_finalized_reasoning_score_call_index(
            Path(terminal_ref["path"]),
            expected_sha256=terminal_ref["sha256"],
            expected_receipt_root=Path(expected_receipt_root),
            expected_pair_id=self._pair_id,
            expected_job_id=authenticated_job_id,
        )
        if terminal_sha256 != terminal_ref["sha256"]:
            raise ValueError("reasoning score-call loader digest differs")
        _exact_keys(terminal, _SCORE_INDEX_KEYS, "reasoning score-call index")
        if (
            terminal["schema"] != REASONING_SCORE_CALL_INDEX_SCHEMA
            or terminal["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
            or terminal["environment"] != "reasoning_gym"
            or terminal["scope"] != "scorer-only"
            or terminal["pair_id"] != self._pair_id
            or terminal["job_id"] != authenticated_job_id
            or type(terminal["call_count"]) is not int
            or terminal["call_count"] != K4_SAMPLES
        ):
            raise ValueError("reasoning score-call index identity differs")
        spec_ref = _artifact_ref(
            terminal["spec"], STRICT_GYM_CHILD_SPEC_SCHEMA, "score-call spec"
        )
        child_index_ref = _artifact_ref(
            terminal["child_index"],
            STRICT_GYM_CHILD_INDEX_SCHEMA,
            "score-call child index",
        )
        resource_ref = _artifact_ref(
            terminal["resource_receipt"],
            STRICT_GYM_CHILD_RECEIPT_SCHEMA,
            "score-call resource receipt",
        )
        score_closed_ref = _artifact_ref(
            terminal["score_closed"],
            REASONING_SCORE_CLOSED_SCHEMA,
            "score-call closure receipt",
        )
        expected_static_paths = {
            spec_ref["path"]: f"{expected_receipt_root}/spec.json",
            child_index_ref["path"]: f"{expected_receipt_root}/index.json",
            resource_ref["path"]: f"{expected_receipt_root}/resource.json",
            score_closed_ref["path"]: (
                f"{expected_receipt_root}/reasoning-score-closed.json"
            ),
        }
        if any(
            actual != expected for actual, expected in expected_static_paths.items()
        ):
            raise ValueError("reasoning score-call supporting artifact path differs")
        spec = _load_artifact_document(spec_ref)
        child_index = _load_artifact_document(child_index_ref)
        resource_receipt = _load_artifact_document(resource_ref)
        score_closed = _load_artifact_document(score_closed_ref)
        _close_scorer_identity(
            spec,
            schema=STRICT_GYM_CHILD_SPEC_SCHEMA,
            pair_id=self._pair_id,
            job_id=authenticated_job_id,
            name="reasoning score-call spec",
        )
        _close_scorer_identity(
            child_index,
            schema=STRICT_GYM_CHILD_INDEX_SCHEMA,
            pair_id=self._pair_id,
            job_id=authenticated_job_id,
            name="reasoning score-call child index",
        )
        _close_scorer_identity(
            resource_receipt,
            schema=STRICT_GYM_CHILD_RECEIPT_SCHEMA,
            pair_id=self._pair_id,
            job_id=authenticated_job_id,
            name="reasoning score-call resource receipt",
        )
        if child_index.get("spec") != spec_ref:
            raise ValueError("reasoning score-call child index spec ref differs")
        children = child_index.get("children")
        if not isinstance(children, list) or len(children) != 1:
            raise ValueError("reasoning score-call child index must select one child")
        child = children[0]
        if (
            not isinstance(child, Mapping)
            or child.get("role") != "resource"
            or child.get("receipt") != resource_ref
        ):
            raise ValueError("reasoning score-call resource selection differs")
        if resource_receipt.get("spec_sha256") != spec_ref["sha256"]:
            raise ValueError("reasoning score-call resource spec SHA differs")
        _exact_keys(
            score_closed,
            _SCORE_CLOSED_KEYS,
            "reasoning score-call closure receipt",
        )
        closed_process = _score_process(score_closed["process"])
        if (
            score_closed["schema"] != REASONING_SCORE_CLOSED_SCHEMA
            or score_closed["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
            or score_closed["environment"] != "reasoning_gym"
            or score_closed["pair_id"] != self._pair_id
            or score_closed["job_id"] != authenticated_job_id
            or score_closed["spec_sha256"] != spec_ref["sha256"]
            or type(score_closed["call_count"]) is not int
            or score_closed["call_count"] != K4_SAMPLES
        ):
            raise ValueError("reasoning score-call closure identity differs")
        closed_call_refs = score_closed["calls"]
        if (
            not isinstance(closed_call_refs, list)
            or len(closed_call_refs) != K4_SAMPLES
        ):
            raise ValueError("reasoning score-call closure must contain exact K=4")
        quiescence = _score_quiescence(terminal["quiescence"])
        if {
            "pid": quiescence["pid"],
            "start_ticks": quiescence["start_ticks"],
        } != closed_process:
            raise ValueError("reasoning score-call quiescence process differs")
        child_observation = child.get("observation")
        if not isinstance(child_observation, Mapping) or (
            child_observation.get("pid") != quiescence["pid"]
            or child_observation.get("start_ticks") != quiescence["start_ticks"]
            or child_observation.get("wrapper_pid") != quiescence["wrapper_pid"]
        ):
            raise ValueError("reasoning score-call quiescence startup join differs")
        calls = terminal["calls"]
        if not isinstance(calls, list) or len(calls) != K4_SAMPLES:
            raise ValueError("reasoning score-call index must contain exact K=4")
        observed_process: dict[str, Any] | None = None
        for rollout_index, index_call in enumerate(calls):
            sequence = rollout_index + 1
            _exact_keys(
                index_call,
                _SCORE_INDEX_CALL_KEYS,
                f"reasoning score-call index entry {sequence}",
            )
            receipt_ref = _artifact_ref(
                index_call["receipt"],
                REASONING_SCORE_CALL_SCHEMA,
                f"reasoning score-call receipt {sequence}",
            )
            closed_call_ref = _score_closed_call_ref(
                closed_call_refs[rollout_index], sequence=sequence
            )
            if closed_call_ref != {"sequence": sequence, **receipt_ref}:
                raise ValueError(f"reasoning score-call closure ref {sequence} differs")
            if receipt_ref["path"] != (
                f"{expected_receipt_root}/reasoning-score-call-{sequence:08d}.json"
            ):
                raise ValueError(
                    f"reasoning score-call receipt {sequence} path differs"
                )
            receipt = _load_artifact_document(receipt_ref)
            _exact_keys(
                receipt,
                _SCORE_CALL_KEYS,
                f"reasoning score-call receipt {sequence}",
            )
            expected = self._expected_reasoning_score_call(rollout_index)
            call_process = _score_process(receipt["process"])
            if call_process != closed_process:
                raise ValueError(
                    f"reasoning score-call receipt {sequence} closure process differs"
                )
            if observed_process is None:
                observed_process = call_process
            elif call_process != observed_process:
                raise ValueError("reasoning score-call process changed across K=4")
            resource_process = resource_receipt.get("process")
            if not isinstance(resource_process, Mapping) or call_process != {
                "pid": resource_process.get("pid"),
                "start_ticks": resource_process.get("start_ticks"),
            }:
                raise ValueError(
                    f"reasoning score-call receipt {sequence} process differs "
                    "from resource receipt"
                )
            if (
                receipt["schema"] != REASONING_SCORE_CALL_SCHEMA
                or receipt["hash_domain"] != STRICT_GYM_CHILD_HASH_DOMAIN
                or receipt["environment"] != "reasoning_gym"
                or receipt["pair_id"] != self._pair_id
                or receipt["job_id"] != authenticated_job_id
                or receipt["spec_sha256"] != spec_ref["sha256"]
                or receipt["sequence"] != sequence
                or type(receipt["sequence"]) is not int
                or receipt["task_name"] != expected["task_name"]
            ):
                raise ValueError(
                    f"reasoning score-call receipt {sequence} identity differs"
                )
            _exact_keys(
                receipt["input"],
                _SCORE_INPUT_KEYS,
                f"reasoning score-call receipt {sequence} input",
            )
            if receipt["input"] != expected["input"]:
                raise ValueError(
                    f"reasoning score-call receipt {sequence} input differs"
                )
            if receipt["outcome"] != {
                "kind": "returned",
                "float_result": expected["float_result"],
            }:
                raise ValueError(
                    f"reasoning score-call receipt {sequence} did not return reward"
                )
            expected_index_call = {
                "sequence": sequence,
                "task_name": expected["task_name"],
                "input": expected["input"],
                "float_result": expected["float_result"],
                "receipt": receipt_ref,
            }
            _exact_document(
                index_call,
                expected_index_call,
                f"reasoning score-call index entry {sequence}",
            )
        return {
            "status": "authenticated",
            "terminal_index": terminal_ref,
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


def load_strict_model_transport_replay_source(
    *,
    source: AuthenticatedOffSourceCapture,
    replay_execution_manifest: Mapping[str, Any],
) -> StrictModelTransportReplaySource:
    """Reload one authenticated OFF source under its Pair/replay authorities."""
    if not isinstance(source, AuthenticatedOffSourceCapture):
        raise TypeError("source must be an AuthenticatedOffSourceCapture")
    validate_replay_execution_manifest(
        replay_execution_manifest, authenticated_source=source
    )
    _exact_document(
        replay_execution_manifest["source_capture"],
        source.source_capture,
        "authenticated replay source capture",
    )
    pair = _pair_id(replay_execution_manifest["pair_id"])
    environment = replay_execution_manifest["environment"]
    if environment not in _ENVIRONMENTS:
        raise ValueError("replay source environment is not admitted")
    if environment != "reasoning_gym":
        raise ValueError("only reasoning_gym has authenticated scorer-call evidence")
    attempt_id = _attempt_id(replay_execution_manifest["attempt_id"])
    pair_sha = _digest(source.pair_manifest_sha256, "pair_manifest_sha256")
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
        capture["authenticated_job"]["job_id"], "source authenticated_job_id"
    )
    step1 = capture["step1_evidence"]
    transport = step1["model_transport"]
    raw_reference = _raw_log_ref(transport["raw_log"])
    bundle_reference = _artifact_ref(
        transport["bundle"], MODEL_TRANSPORT_BUNDLE_SCHEMA, "bundle_ref"
    )
    manifest_reference = _artifact_ref(
        transport["manifest"],
        MODEL_TRANSPORT_MANIFEST_SCHEMA,
        "transport_manifest_ref",
    )
    transcript_reference = _artifact_ref(
        step1["transcript_bundle"], TRANSCRIPT_BUNDLE_SCHEMA, "transcript_ref"
    )
    main_ledger_reference = _artifact_ref(
        step1["main_ledger"], MAIN_STEP1_LEDGER_SCHEMA, "main ledger ref"
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
    _exact_document(manifest["transport_capture"], raw_reference, "raw log ref")
    _exact_document(manifest["transport_bundle"], bundle_reference, "bundle ref")
    _exact_document(
        manifest["main_transcript_bundle"],
        transcript_reference,
        "transcript ref",
    )

    bundle = _load_artifact_document(bundle_reference)
    transcript = _load_artifact_document(transcript_reference)
    _exact_document(manifest["main_ledger"], main_ledger_reference, "main ledger ref")
    main_ledger = _load_artifact_document(main_ledger_reference)
    raw_log = _load_immutable_bytes(
        Path(raw_reference["path"]),
        expected_sha256=raw_reference["sha256"],
        maximum=MAX_BUNDLE_BYTES,
    )
    _exact_document(manifest, source.transport_manifest, "authenticated manifest")
    _exact_document(bundle, source.transport_bundle, "authenticated bundle")
    _exact_document(transcript, source.transcript_bundle, "authenticated transcript")
    _exact_document(main_ledger, source.main_ledger, "authenticated main ledger")

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
    validate_ledger_transcript_join(ledger=main_ledger, transcript_bundle=transcript)

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
        snapshots.get("off"), Mapping
    ):
        raise TypeError("authenticated Pair OFF snapshot is missing")
    source_root = _canonical_absolute_path(
        snapshots["off"].get("path"), "authenticated Pair OFF snapshot path"
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
    return StrictModelTransportReplaySource(
        pair_id=pair,
        environment=environment,
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


def validate_strict_model_transport_replay_consumption(
    document: Mapping[str, Any],
) -> None:
    """Validate the closed terminal consumption schema and all domain hashes."""
    _exact_keys(document, _CONSUMPTION_ROOT_KEYS, "replay consumption")
    if document["schema"] != MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA:
        raise ValueError("unexpected replay consumption schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected replay consumption hash domain")
    if (
        document["status"] != MODEL_TRANSPORT_REPLAY_STATUS
        or document["mode"] != MODEL_TRANSPORT_REPLAY_MODE
    ):
        raise ValueError("replay consumption terminal mode/status differs")
    _pair_id(document["pair_id"])
    if document["environment"] != "reasoning_gym":
        raise ValueError(
            "only reasoning_gym replay consumption has authenticated scorer evidence"
        )
    source = document["source"]
    _exact_keys(source, _SOURCE_KEYS, "replay consumption source")
    if source["arm"] != "off":
        raise ValueError("replay consumption source arm must be off")
    _job_id(source["authenticated_job_id"], "source.authenticated_job_id")
    _digest(source["pair_manifest_sha256"], "source.pair_manifest_sha256")
    _digest(
        source["submission_receipt_sha256"],
        "source.submission_receipt_sha256",
    )
    _artifact_ref(source["main_ledger"], MAIN_STEP1_LEDGER_SCHEMA, "source.main_ledger")
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
    _digest(source["ordered_entries_sha256"], "source.ordered_entries_sha256")
    replay = document["replay"]
    _exact_keys(replay, _REPLAY_KEYS, "replay consumption replay")
    _attempt_id(replay["attempt_id"])
    _digest(
        replay["replay_execution_manifest_sha256"],
        "replay.replay_execution_manifest_sha256",
    )
    _job_id(replay["authenticated_job_id"], "replay.authenticated_job_id")
    _process(replay["process"])
    validate_scheduler_device_environment(replay["scheduler_device_environment"])
    scorer = replay["scorer_evidence"]
    _exact_keys(scorer, _SCORER_EVIDENCE_KEYS, "replay scorer evidence")
    if scorer["status"] != "authenticated":
        raise ValueError("reasoning replay scorer evidence is not authenticated")
    _artifact_ref(
        scorer["terminal_index"],
        REASONING_SCORE_CALL_INDEX_SCHEMA,
        "replay scorer terminal index",
    )
    if type(document["entry_count"]) is not int or document["entry_count"] != 4:
        raise ValueError("replay consumption entry_count must be exact integer 4")
    entries = document["entries"]
    if not isinstance(entries, list) or len(entries) != K4_SAMPLES:
        raise ValueError("replay consumption entries must be exact K=4")
    for index, raw_entry in enumerate(entries):
        _exact_keys(raw_entry, _CONSUMPTION_ENTRY_KEYS, f"entry[{index}]")
        if (
            raw_entry["rollout_index"] != index
            or type(raw_entry["rollout_index"]) is not int
        ):
            raise ValueError("replay consumption entries are not logical 0..3")
        if raw_entry["schema"] != MODEL_TRANSPORT_REPLAY_CONSUMPTION_ENTRY_SCHEMA:
            raise ValueError("unexpected replay consumption entry schema")
        _nonnegative_int63(raw_entry["generation_seed"], "generation_seed")
        if raw_entry["generation_seed"] != derive_nemo_gym_request_seed(
            seed_base=42, fixture_row_index=0, rollout_index=index
        ):
            raise ValueError("replay consumption generation seed does not close")
        for name in _CONSUMPTION_ENTRY_KEYS - {
            "schema",
            "rollout_index",
            "generation_seed",
            "fresh_native_reward",
        }:
            _digest(raw_entry[name], f"entry[{index}].{name}")
        reward = raw_entry["fresh_native_reward"]
        if type(reward) is not float or not math.isfinite(reward):
            raise TypeError("replay consumption reward must be a finite float")
        if reward == 0.0 and math.copysign(1.0, reward) < 0.0:
            raise ValueError("replay consumption reward must not be negative zero")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("reasoning replay consumption reward is outside [0, 1]")
        projection = {
            key: copy.deepcopy(value)
            for key, value in raw_entry.items()
            if key != "entry_sha256"
        }
        if raw_entry["entry_sha256"] != domain_sha256(
            "model-transport-replay-consumption-entry", projection
        ):
            raise ValueError(f"replay consumption entry {index} digest differs")
    if document["ordered_entries_sha256"] != domain_sha256(
        "model-transport-replay-consumption-entries", entries
    ):
        raise ValueError("replay consumption ordered digest differs")
    canonical_ascii_json(document)


def publish_strict_model_transport_replay_consumption(
    *, output: str | Path, document: Mapping[str, Any]
) -> tuple[Path, str]:
    """Publish one exclusive, canonical no-LF, EUID-owned mode-0400 document."""
    validate_strict_model_transport_replay_consumption(document)
    return publish_evidence_document(
        output=output, document=document, trailing_lf=False
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
        transcript["model_transport_bundle"], bundle_ref, "transcript bundle ref"
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
    _exact_document(ledger["transcript_bundle"], transcript_ref, "ledger transcript")
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
    value: Mapping[str, Any], expected_schema: str, name: str
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
        result["boot_id_sha256"], "process.boot_id_sha256"
    )
    if type(result["pid"]) is not int or not 1 <= result["pid"] <= _MAX_INT31:
        raise ValueError("process.pid must be an exact positive int31")
    if (
        type(result["start_time_ticks"]) is not int
        or not 1 <= result["start_time_ticks"] <= _MAX_INT63
    ):
        raise ValueError("process.start_time_ticks must be an exact positive int63")
    return result


def _score_process(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, _SCORE_PROCESS_KEYS, "reasoning score process")
    result = dict(value)
    if type(result["pid"]) is not int or not 1 <= result["pid"] <= _MAX_INT31:
        raise ValueError("reasoning score process pid must be a positive int31")
    if (
        type(result["start_ticks"]) is not int
        or not 1 <= result["start_ticks"] <= _MAX_INT63
    ):
        raise ValueError("reasoning score process start_ticks must be a positive int63")
    return result


def _score_closed_call_ref(value: Any, *, sequence: int) -> dict[str, Any]:
    _exact_keys(
        value,
        _SCORE_CLOSED_CALL_REF_KEYS,
        f"reasoning score-call closure ref {sequence}",
    )
    if type(value["sequence"]) is not int or value["sequence"] != sequence:
        raise ValueError(
            f"reasoning score-call closure ref {sequence} sequence differs"
        )
    reference = _artifact_ref(
        {key: value[key] for key in _ARTIFACT_REF_KEYS},
        REASONING_SCORE_CALL_SCHEMA,
        f"reasoning score-call closure ref {sequence}",
    )
    return {"sequence": sequence, **reference}


def _score_quiescence(value: Any) -> dict[str, Any]:
    _exact_keys(value, _SCORE_QUIESCENCE_KEYS, "reasoning score quiescence")
    result = dict(value)
    process = _score_process(
        {"pid": result["pid"], "start_ticks": result["start_ticks"]}
    )
    result.update(process)
    if (
        type(result["wrapper_pid"]) is not int
        or not 2 <= result["wrapper_pid"] <= _MAX_INT31
    ):
        raise ValueError("reasoning score quiescence wrapper_pid is invalid")
    child_signal = result["child_termination_signal"]
    if type(child_signal) is not str or child_signal not in {
        "SIGINT",
        "SIGTERM",
        "SIGKILL",
    }:
        raise ValueError("reasoning score quiescence child signal is invalid")
    if type(result["wrapper_returncode"]) is not int:
        raise TypeError("reasoning score quiescence wrapper_returncode must be an int")
    if result["wrapper_pid"] != result["pid"] or (
        child_signal == "SIGKILL" and result["wrapper_returncode"] != -signal.SIGKILL
    ):
        raise ValueError("reasoning score quiescence wrapper process differs")
    if result["original_process_reaped"] is not True:
        raise ValueError("reasoning score quiescence process was not reaped")
    return result


def _close_scorer_identity(
    document: Mapping[str, Any],
    *,
    schema: str,
    pair_id: str,
    job_id: str,
    name: str,
) -> None:
    expected = {
        "schema": schema,
        "environment": "reasoning_gym",
        "pair_id": pair_id,
        "job_id": job_id,
    }
    if schema != STRICT_GYM_CHILD_RECEIPT_SCHEMA:
        expected["scope"] = "scorer-only"
    for key, value in expected.items():
        if document.get(key) != value or type(document.get(key)) is not type(value):
            raise ValueError(f"{name} {key} differs")
    if document.get("hash_domain") != STRICT_GYM_CHILD_HASH_DOMAIN:
        raise ValueError(f"{name} hash domain differs")


def _load_artifact_document(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Load canonical JSON and independently stable-read the same named inode."""
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


def _load_immutable_bytes(path: Path, *, expected_sha256: str, maximum: int) -> bytes:
    expected = _digest(expected_sha256, "immutable bytes expected SHA-256")
    canonical = _canonical_absolute_path(str(path), "immutable bytes path")
    _reject_symlink_components(canonical)
    named_before = os.lstat(canonical)
    descriptor = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
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
    fingerprint = lambda item: (
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
    if type(value) is not str or not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"{name} must be a canonical absolute path")
    if value != posixpath.normpath(value) or "//" in value:
        raise ValueError(f"{name} must be a canonical absolute path")
    return Path(value)


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
    if value not in {"replay-1", "replay-2"}:
        raise ValueError("attempt_id must be replay-1 or replay-2")
    return value


def _nonnegative_int63(value: Any, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_INT63:
        raise ValueError(f"{name} must be an exact nonnegative int63")
    return value


__all__ = [
    "MODEL_TRANSPORT_REPLAY_CONSUMPTION_SCHEMA",
    "MODEL_TRANSPORT_REPLAY_MODE",
    "MODEL_TRANSPORT_REPLAY_STATUS",
    "ReplayVerifierMaterial",
    "StrictModelTransportReplayError",
    "StrictModelTransportReplaySource",
    "load_strict_model_transport_replay_source",
    "publish_strict_model_transport_replay_consumption",
    "validate_strict_model_transport_replay_consumption",
]
