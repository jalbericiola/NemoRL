"""Fail-closed evidence helpers for strict step-1 model transport capture.

This module deliberately has no torch, Ray, FastAPI, or vLLM dependency.  The
model-owner route owns concurrent capture and the rollout finalizer owns the
state transition; this module owns their canonical evidence formats.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import re
import socket
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from nemo_rl.utils.strict_captured_replay_evidence import (
    HASH_DOMAIN,
    canonical_ascii_json,
    derive_nemo_gym_request_seed,
    domain_sha256,
    load_evidence_document,
    publish_evidence_document,
    validate_gym_model_response_r3,
)


MODEL_TRANSPORT_POLICY_SCHEMA = "nemo-rl-strict-model-transport-policy-v1"
MODEL_TRANSPORT_CALL_SCHEMA = "nemo-rl-strict-model-transport-call-v1"
MODEL_TRANSPORT_BUNDLE_SCHEMA = "nemo-rl-strict-model-transport-bundle-v1"
MODEL_TRANSPORT_MANIFEST_SCHEMA = "nemo-rl-strict-model-transport-manifest-v1"
MODEL_TRANSPORT_CAPTURE_OPEN_SCHEMA = "nemo-rl-strict-model-transport-capture-open-v1"
STEP1_TRANSCRIPT_BUNDLE_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
MAIN_STEP1_LEDGER_SCHEMA = "nemo-rl-strict-main-step1-ledger-v5"

MODEL_TRANSPORT_DIRECTORY = "strict_model_transport"
MODEL_TRANSPORT_LOG = "model-transport.jsonl"
MODEL_TRANSPORT_BUNDLE = "model-transport-bundle.json"
MODEL_TRANSPORT_MANIFEST = "model-transport-manifest.json"

MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 192 * 1024 * 1024
K4_SAMPLES = 4
MAX_INT31 = (1 << 31) - 1
MAX_INT63 = (1 << 63) - 1

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HOSTNAME_RE = re.compile(
    r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9.-]{0,253}[A-Za-z0-9])\Z"
)
_CHAT_ID_RE = re.compile(r"chatcmpl-[0-9a-f]{16}\Z")
_FINGERPRINT_RE = re.compile(r"vllm-0\.25\.1-[0-9A-Za-z._+-]{1,128}\Z")
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

_SOURCE_PATHS = {
    "collector": "nemo_rl/utils/strict_model_transport.py",
    "rollout_finalizer": "nemo_rl/experience/rollout_manager.py",
    "vllm_route": "nemo_rl/models/generation/vllm/vllm_worker_async.py",
}

_ACTIVATION = {
    "arm_environment": {"name": "STRICT_PAIR_ARM", "off": "off", "on": "on"},
    "config_key": "policy.generation.vllm_cfg.strict_model_transport",
    "environment_environment": "STRICT_PAIR_ENVIRONMENT",
    "main_mode": "capture",
    "pair_id_environment": "PAIR_ID",
    "replay_mode": "replay",
    "results_dir_environment": "RESULTS_DIR",
}

_CAPTURE_WINDOW = {
    "concurrency": "arrival-independent",
    "duplicate_or_retry": "reject",
    "fixture_row_index": 0,
    "logical_rollout_indices": [0, 1, 2, 3],
    "main_after_seal": (
        "reject-until-rollout-finalizer-attests-step1-complete-then-pass-through"
    ),
    "replay_after_seal": "reject-terminal",
    "sample_count": K4_SAMPLES,
    "seal": "atomic-after-four-successes",
    "seed_base": 42,
    "seed_derivation": "sha256-canonical-ascii-json-int63-v1",
    "step": 1,
}

_HTTP_POLICY = {
    "authorization": "excluded",
    "body_boundary": "http-body-bytes-only",
    "cookies": "excluded",
    "direct_python_generation_during_replay": "reject",
    "encoding": "utf-8",
    "endpoint_allowlist": [
        {"logical_count": K4_SAMPLES, "method": "POST", "path": "/v1/chat/completions"}
    ],
    "headers": "excluded",
    "max_bundle_bytes": MAX_BUNDLE_BYTES,
    "max_request_body_bytes": MAX_REQUEST_BODY_BYTES,
    "max_response_body_bytes": MAX_RESPONSE_BODY_BYTES,
    "probe_allowlist": [],
    "query": "forbidden",
    "request_media_type": "application/json",
    "response_media_type": "application/json",
    "response_status_code": 200,
    "streaming": False,
    "tokenize_count": 0,
    "unlisted_during_replay": "reject",
    "unlisted_during_window": "reject",
}

_ARTIFACTS = {
    "bundle": {
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
        "relative_path": f"{MODEL_TRANSPORT_DIRECTORY}/{MODEL_TRANSPORT_BUNDLE}",
        "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
    },
    "directory": {
        "inventory": [
            MODEL_TRANSPORT_LOG,
            MODEL_TRANSPORT_BUNDLE,
            MODEL_TRANSPORT_MANIFEST,
        ],
        "mode": "0700",
        "precondition": "absent-at-pre-runtime-creates-exclusively",
        "relative_path": MODEL_TRANSPORT_DIRECTORY,
    },
    "log": {
        "framing": "canonical-ascii-json-line-lf",
        "lines": K4_SAMPLES,
        "mode": "0400",
        "relative_path": f"{MODEL_TRANSPORT_DIRECTORY}/{MODEL_TRANSPORT_LOG}",
        "schema": MODEL_TRANSPORT_CALL_SCHEMA,
    },
    "manifest": {
        "framing": "canonical-ascii-json-no-lf",
        "mode": "0400",
        "relative_path": f"{MODEL_TRANSPORT_DIRECTORY}/{MODEL_TRANSPORT_MANIFEST}",
        "schema": MODEL_TRANSPORT_MANIFEST_SCHEMA,
    },
    "nlink": 1,
    "owner": "effective-uid",
    "publication": "o-excl-fsync-atomic-seal",
}

MODEL_TRANSPORT_ENDPOINT = {
    "method": "POST",
    "path": "/v1/chat/completions",
    "request_media_type": "application/json",
    "response_media_type": "application/json",
    "status_code": 200,
    "streaming": False,
}

_POLICY_KEYS = frozenset(
    {
        "activation",
        "arms",
        "artifacts",
        "capture_window",
        "enabled",
        "hash_domain",
        "http",
        "policy_sha256",
        "schema",
        "sources",
    }
)
_CAPTURE_SERVER_KEYS = frozenset(
    {"boot_id_sha256", "hostname", "pid", "server_instance_id", "start_time_ticks"}
)
_CALL_KEYS = frozenset(
    {
        "arrival_index",
        "entry_sha256",
        "generation_seed",
        "request_body_base64",
        "request_body_sha256",
        "request_payload",
        "request_payload_sha256",
        "response_body_base64",
        "response_body_sha256",
        "response_payload",
        "response_payload_sha256",
        "rollout_index",
        "schema",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "chat_template_kwargs",
        "logprobs",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "return_tokens_as_token_ids",
        "seed",
        "temperature",
        "top_logprobs",
        "top_p",
    }
)
_OUTER_GENERATION_REQUEST_KEYS = frozenset(
    {"input", "max_output_tokens", "metadata", "temperature", "top_p"}
)
_RESPONSE_KEYS = frozenset(
    {
        "choices",
        "created",
        "id",
        "kv_transfer_params",
        "metrics",
        "model",
        "object",
        "prompt_logprobs",
        "prompt_text",
        "prompt_token_ids",
        "service_tier",
        "system_fingerprint",
        "usage",
    }
)
_CHOICE_KEYS = frozenset(
    {
        "finish_reason",
        "index",
        "logprobs",
        "message",
        "routed_experts",
        "stop_reason",
        "token_ids",
    }
)
_MESSAGE_KEYS = frozenset(
    {
        "annotations",
        "audio",
        "content",
        "function_call",
        "generation_log_probs",
        "generation_token_ids",
        "prompt_token_ids",
        "reasoning",
        "refusal",
        "role",
    }
)
_USAGE_KEYS = frozenset(
    {"completion_tokens", "prompt_tokens", "prompt_tokens_details", "total_tokens"}
)
_TOKEN_LOGPROB_KEYS = frozenset({"bytes", "logprob", "token", "top_logprobs"})
_BUNDLE_KEYS = frozenset(
    {
        "arm",
        "capture_server",
        "capture_window",
        "endpoint",
        "entries",
        "entry_count",
        "environment",
        "hash_domain",
        "ordered_entries_sha256",
        "pair_id",
        "schema",
    }
)
_REF_KEYS = frozenset({"path", "schema", "sha256"})
_CAPTURE_REF_KEYS = frozenset({"path", "record_count", "record_schema", "sha256"})
_MANIFEST_KEYS = frozenset(
    {
        "arm",
        "authenticated_job_id",
        "capture_server",
        "entry_count",
        "environment",
        "hash_domain",
        "main_ledger",
        "main_transcript_bundle",
        "model_transport_policy_sha256",
        "ordered_entries_sha256",
        "pair_id",
        "pair_manifest_sha256",
        "schema",
        "submission_receipt_sha256",
        "transport_bundle",
        "transport_capture",
    }
)


@dataclass(frozen=True)
class ModelTransportTicket:
    """Opaque reservation returned by :meth:`begin_chat_call`."""

    arrival_index: int
    generation_seed: int
    nonce: str
    request_body: bytes
    request_payload: dict[str, Any]
    rollout_index: int


class StrictModelTransportCapture:
    """Thread-safe, fail-closed K=4 capture state machine for one vLLM owner."""

    def __init__(
        self,
        *,
        pair_id: str,
        environment: str,
        arm: str,
        results_dir: str | Path,
        model_path: str,
        model_transport_policy: Mapping[str, Any],
        capture_server: Mapping[str, Any] | None = None,
    ) -> None:
        validate_model_transport_policy(model_transport_policy)
        self._pair_id, self._environment, self._arm = _identity(
            pair_id, environment, arm
        )
        self._policy = copy.deepcopy(dict(model_transport_policy))
        self._model_path = _bounded_text(model_path, "model_path", 1_048_576)
        self._capture_server = _capture_server(
            observe_capture_server_identity()
            if capture_server is None
            else capture_server
        )
        self._directory = initialize_model_transport_directory(
            results_dir=results_dir, model_transport_policy=self._policy
        )
        self._spool = self._directory / "spool"
        os.mkdir(self._spool, stat.S_IRWXU)
        os.chmod(self._spool, stat.S_IRWXU, follow_symlinks=False)
        marker = {
            "arm": self._arm,
            "capture_server": self._capture_server,
            "created_at_unix_ns": time.time_ns(),
            "environment": self._environment,
            "hash_domain": HASH_DOMAIN,
            "model_transport_policy_sha256": self._policy["policy_sha256"],
            "pair_id": self._pair_id,
            "schema": MODEL_TRANSPORT_CAPTURE_OPEN_SCHEMA,
        }
        if marker["created_at_unix_ns"] <= 0:
            raise RuntimeError("capture-open timestamp is not positive")
        publish_evidence_document(
            output=self._directory / "capture-open.json",
            document=marker,
            trailing_lf=False,
        )
        self._lock = threading.Lock()
        self._phase = "open"
        self._poison: str | None = None
        self._reservations: dict[str, tuple[int, ModelTransportTicket]] = {}
        self._rollouts: set[int] = set()
        self._entries: dict[int, dict[str, Any]] = {}
        self._capture_ref: dict[str, Any] | None = None
        self._bundle_ref: dict[str, Any] | None = None
        self._bundle: dict[str, Any] | None = None

    @property
    def capture_server(self) -> dict[str, Any]:
        return copy.deepcopy(self._capture_server)

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def begin_chat_call(
        self,
        *,
        request_body: bytes,
        typed_seed: int,
        method: str,
        path: str,
        query: str,
        media_type: str,
        expected_request_payload: Mapping[str, Any] | None = None,
        expected_generation_input: Sequence[Mapping[str, Any]] | None = None,
    ) -> ModelTransportTicket | None:
        """Reserve one unique logical step-1 request, or pass later main traffic."""
        self._validate_endpoint(
            method=method, path=path, query=query, media_type=media_type
        )
        with self._lock:
            if self._poison is not None:
                raise RuntimeError(
                    f"strict model transport capture is poisoned: {self._poison}"
                )
            if self._phase == "pass-through":
                return None
            if self._phase != "open":
                self._poison_locked(
                    "chat call arrived after seal but before attestation"
                )
        raw = _body(request_body, "request_body", MAX_REQUEST_BODY_BYTES)
        payload = _strict_json_object(raw, "request body")
        seed = _bounded_int(typed_seed, "typed_seed", 0, MAX_INT63)
        if payload.get("seed") != seed or type(payload.get("seed")) is not int:
            self._fail("typed seed differs from exact raw request seed")
        rollout_by_seed = {
            derive_nemo_gym_request_seed(
                seed_base=42, fixture_row_index=0, rollout_index=index
            ): index
            for index in range(K4_SAMPLES)
        }
        if seed not in rollout_by_seed:
            self._fail("chat request is not one of the frozen logical step-1 seeds")
        rollout_index = rollout_by_seed[seed]
        _validate_request_payload(
            payload,
            generation_seed=seed,
            model_path=self._model_path,
            expected_generation_input=expected_generation_input,
        )
        if expected_request_payload is not None:
            _exact_json(payload, expected_request_payload, "request raw/typed payload")
        with self._lock:
            if self._poison is not None:
                raise RuntimeError(
                    f"strict model transport capture is poisoned: {self._poison}"
                )
            if self._phase != "open":
                self._poison_locked(
                    "chat call arrived after seal but before attestation"
                )
            if rollout_index in self._rollouts:
                self._poison_locked(
                    f"duplicate or retry for logical rollout {rollout_index}"
                )
            arrival_index = len(self._rollouts)
            if arrival_index >= K4_SAMPLES:
                self._poison_locked("more than four logical requests arrived")
            nonce = domain_sha256(
                "model-transport-ticket",
                {
                    "arrival_index": arrival_index,
                    "capture_server": self._capture_server,
                    "request_body_sha256": hashlib.sha256(raw).hexdigest(),
                    "rollout_index": rollout_index,
                },
            )
            ticket = ModelTransportTicket(
                arrival_index=arrival_index,
                generation_seed=seed,
                nonce=nonce,
                request_body=raw,
                request_payload=payload,
                rollout_index=rollout_index,
            )
            spool_path = self._spool / f"call-{rollout_index}.json"
            fd = os.open(
                spool_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                stat.S_IRUSR,
            )
            os.fchmod(fd, stat.S_IRUSR)
            self._rollouts.add(rollout_index)
            self._reservations[nonce] = (fd, ticket)
            return ticket

    def record_success(
        self,
        ticket: ModelTransportTicket,
        *,
        response_body: bytes,
        status_code: int,
        media_type: str,
        streaming: bool,
        expected_response_payload: Mapping[str, Any] | None = None,
        expected_generation_input: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Complete a reservation and seal the bundle on the fourth success."""
        if (
            type(status_code) is not int
            or status_code != 200
            or type(media_type) is not str
            or media_type != "application/json"
            or type(streaming) is not bool
            or streaming
        ):
            self.record_failure(
                ticket, reason="non-success/non-JSON/streaming response"
            )
            raise RuntimeError(
                "strict transport only accepts nonstreaming JSON HTTP 200"
            )
        with self._lock:
            if self._poison is not None:
                raise RuntimeError(
                    f"strict model transport capture is poisoned: {self._poison}"
                )
            reservation = self._reservations.pop(ticket.nonce, None)
            if reservation is None or reservation[1] != ticket:
                self._poison_locked("unknown, reused, or forged capture ticket")
            fd = reservation[0]
            try:
                entry = build_model_transport_call(
                    pair_id=self._pair_id,
                    environment=self._environment,
                    arm=self._arm,
                    capture_server=self._capture_server,
                    rollout_index=ticket.rollout_index,
                    generation_seed=ticket.generation_seed,
                    arrival_index=ticket.arrival_index,
                    model_path=self._model_path,
                    request_body=ticket.request_body,
                    response_body=response_body,
                    expected_generation_input=expected_generation_input,
                    expected_request_payload=ticket.request_payload,
                    expected_response_payload=expected_response_payload,
                )
                _write_fd(fd, canonical_ascii_json(entry) + b"\n")
                os.fsync(fd)
            except Exception:
                self._poison = "call completion validation/publication failed"
                raise
            finally:
                os.close(fd)
            self._entries[ticket.rollout_index] = entry
            if len(self._entries) == K4_SAMPLES:
                if self._reservations or set(self._entries) != set(range(K4_SAMPLES)):
                    self._poison_locked(
                        "cannot close capture window without four completed logical calls"
                    )
                self._phase = "full-set-awaiting-finalizer"
            return copy.deepcopy(entry)

    def record_failure(
        self, ticket: ModelTransportTicket, *, reason: str = "model call failed"
    ) -> None:
        with self._lock:
            reservation = self._reservations.pop(ticket.nonce, None)
            if reservation is not None:
                os.close(reservation[0])
            self._poison = _ascii(reason, "capture failure reason", 512)

    def record_unmatched_failure(self, *, reason: str) -> None:
        """Poison an active window when failure occurs before a ticket exists."""
        failure = _ascii(reason, "unmatched capture failure reason", 512)
        with self._lock:
            if self._phase == "pass-through":
                return
            if self._poison is None:
                self._poison = failure

    def guard_unlisted(self, *, method: str, path: str) -> None:
        if method == "POST" and path == "/v1/chat/completions":
            return
        with self._lock:
            if self._phase == "pass-through" and self._poison is None:
                return
        self._fail(f"unlisted model transport endpoint {method} {path}")

    def attest_step1_complete(
        self,
        *,
        expected_generation_requests: Sequence[Mapping[str, Any]],
        expected_model_responses: Sequence[Mapping[str, Any]],
        expected_request_body_sha256s: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Verify outer joins, remove transients, and enable later main traffic."""
        with self._lock:
            if self._poison is not None:
                raise RuntimeError(
                    f"strict model transport capture is poisoned: {self._poison}"
                )
            if self._phase != "full-set-awaiting-finalizer":
                raise RuntimeError(
                    "strict model transport capture has not completed K=4"
                )
            try:
                expected_generation_inputs = _generation_inputs(
                    expected_generation_requests
                )
                entries = [self._entries[index] for index in range(K4_SAMPLES)]
                expected_spool = {f"call-{index}.json" for index in range(K4_SAMPLES)}
                if {item.name for item in self._spool.iterdir()} != expected_spool:
                    raise RuntimeError("capture spool inventory changed")
                for index, entry in enumerate(entries):
                    spool_raw = _load_immutable_bytes(
                        self._spool / f"call-{index}.json", maximum=MAX_BUNDLE_BYTES
                    )
                    if spool_raw != canonical_ascii_json(entry) + b"\n":
                        raise RuntimeError(
                            f"capture spool entry {index} differs from owner memory"
                        )
                prepublication_inventory = {"capture-open.json", "spool"}
                if {
                    item.name for item in self._directory.iterdir()
                } != prepublication_inventory:
                    raise RuntimeError(
                        "pre-publication transport directory inventory changed"
                    )
                bundle = build_model_transport_bundle(
                    pair_id=self._pair_id,
                    environment=self._environment,
                    arm=self._arm,
                    model_transport_policy=self._policy,
                    capture_server=self._capture_server,
                    entries=entries,
                    model_path=self._model_path,
                    expected_generation_inputs=expected_generation_inputs,
                )
                validate_model_transport_generation_request_join(
                    bundle, generation_requests=expected_generation_requests
                )
                validate_model_transport_model_response_join(
                    bundle,
                    generation_requests=expected_generation_requests,
                    model_responses=expected_model_responses,
                )
                if expected_request_body_sha256s is not None:
                    if len(expected_request_body_sha256s) != K4_SAMPLES:
                        raise ValueError(
                            "expected_request_body_sha256s must contain four digests"
                        )
                    for index, digest in enumerate(expected_request_body_sha256s):
                        if bundle["entries"][index]["request_body_sha256"] != _digest(
                            digest, f"expected_request_body_sha256s[{index}]"
                        ):
                            raise ValueError(
                                "captured request body differs from finalizer evidence"
                            )
                capture_ref, bundle_ref = publish_model_transport_capture(
                    transport_directory=self._directory,
                    bundle=bundle,
                    model_transport_policy=self._policy,
                    model_path=self._model_path,
                    expected_generation_inputs=expected_generation_inputs,
                )
                finalizing_inventory = {
                    "capture-open.json",
                    "spool",
                    MODEL_TRANSPORT_LOG,
                    MODEL_TRANSPORT_BUNDLE,
                }
                if {
                    item.name for item in self._directory.iterdir()
                } != finalizing_inventory:
                    raise RuntimeError(
                        "pre-attestation transport directory inventory changed"
                    )
                for index in range(K4_SAMPLES):
                    os.unlink(self._spool / f"call-{index}.json")
                os.rmdir(self._spool)
                os.unlink(self._directory / "capture-open.json")
                directory_fd = os.open(
                    self._directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self._capture_ref = capture_ref
                self._bundle_ref = bundle_ref
                self._bundle = bundle
                self._phase = "pass-through"
                return (
                    copy.deepcopy(capture_ref),
                    copy.deepcopy(bundle_ref),
                    copy.deepcopy(bundle),
                )
            except Exception:
                if self._poison is None:
                    self._poison = "step-1 finalizer attestation/publication failed"
                raise

    def _validate_endpoint(
        self, *, method: str, path: str, query: str, media_type: str
    ) -> None:
        if (
            type(method) is not str
            or method != "POST"
            or type(path) is not str
            or path != "/v1/chat/completions"
            or type(query) is not str
            or query != ""
            or type(media_type) is not str
            or media_type != "application/json"
        ):
            self._fail(
                "model transport endpoint/request media/query differs from policy"
            )

    def _poison_locked(self, reason: str) -> None:
        self._poison = reason
        raise RuntimeError(reason)

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._poison = reason
        raise RuntimeError(reason)


def observe_capture_server_identity(
    *, proc_root: str | Path = "/proc", hostname: str | None = None
) -> dict[str, Any]:
    """Observe the source-attested vLLM actor process tuple."""
    root = Path(proc_root)
    boot_raw = (root / "sys/kernel/random/boot_id").read_bytes()
    try:
        boot_text = boot_raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("capture server boot_id is not ASCII") from error
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", boot_text
    ):
        raise RuntimeError("capture server boot_id is malformed")
    stat_text = (root / "self/stat").read_text(encoding="ascii")
    close = stat_text.rfind(")")
    if close < 0:
        raise RuntimeError("capture server /proc/self/stat is malformed")
    fields_after_comm = stat_text[close + 2 :].split()
    if len(fields_after_comm) < 20:
        raise RuntimeError("capture server /proc/self/stat is incomplete")
    start_time_ticks = int(fields_after_comm[19], 10)
    observed_hostname = socket.gethostname() if hostname is None else hostname
    base = {
        "boot_id_sha256": hashlib.sha256(boot_raw).hexdigest(),
        "hostname": observed_hostname,
        "pid": os.getpid(),
        "start_time_ticks": start_time_ticks,
    }
    base["server_instance_id"] = domain_sha256("model-transport-server-instance", base)
    return _capture_server(base)


def load_attested_model_transport_policy(
    *,
    model_transport_policy: Mapping[str, Any],
    expected_policy_sha256: str,
    source_root: str | Path,
) -> dict[str, Any]:
    """Validate the policy digest and stable-hash all three attested sources."""
    validate_model_transport_policy(model_transport_policy)
    expected = _digest(expected_policy_sha256, "expected_policy_sha256")
    if model_transport_policy["policy_sha256"] != expected:
        raise ValueError("runtime model transport policy differs from expected digest")
    root = Path(_absolute_path(str(source_root), "source_root"))
    for name, expected_path in _SOURCE_PATHS.items():
        source = model_transport_policy["sources"][name]
        if source["path"] != expected_path:
            raise ValueError(f"runtime source path for {name} changed")
        actual = _stable_file_sha256(root / expected_path)
        if actual != source["sha256"]:
            raise ValueError(f"runtime source bytes for {name} differ from Pair policy")
    return copy.deepcopy(dict(model_transport_policy))


def load_runtime_model_transport_policy(
    *, expected_policy_sha256: str, source_root: str | Path
) -> dict[str, Any]:
    """Rebuild and authenticate R2 directly from the three installed sources."""
    root = Path(_absolute_path(str(source_root), "source_root"))
    source_digests = {
        name: _stable_file_sha256(root / relative)
        for name, relative in _SOURCE_PATHS.items()
    }
    policy = build_model_transport_policy(
        collector_sha256=source_digests["collector"],
        vllm_route_sha256=source_digests["vllm_route"],
        rollout_finalizer_sha256=source_digests["rollout_finalizer"],
    )
    return load_attested_model_transport_policy(
        model_transport_policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        source_root=root,
    )


def build_model_transport_policy(
    *,
    collector_sha256: str,
    vllm_route_sha256: str,
    rollout_finalizer_sha256: str,
) -> dict[str, Any]:
    """Build the exact symmetric static Pair policy."""
    sources = {
        "collector": {
            "path": _SOURCE_PATHS["collector"],
            "sha256": _digest(collector_sha256, "collector_sha256"),
        },
        "rollout_finalizer": {
            "path": _SOURCE_PATHS["rollout_finalizer"],
            "sha256": _digest(rollout_finalizer_sha256, "rollout_finalizer_sha256"),
        },
        "vllm_route": {
            "path": _SOURCE_PATHS["vllm_route"],
            "sha256": _digest(vllm_route_sha256, "vllm_route_sha256"),
        },
    }
    document: dict[str, Any] = {
        "activation": copy.deepcopy(_ACTIVATION),
        "arms": ["off", "on"],
        "artifacts": copy.deepcopy(_ARTIFACTS),
        "capture_window": copy.deepcopy(_CAPTURE_WINDOW),
        "enabled": True,
        "hash_domain": HASH_DOMAIN,
        "http": copy.deepcopy(_HTTP_POLICY),
        "schema": MODEL_TRANSPORT_POLICY_SCHEMA,
        "sources": sources,
    }
    document["policy_sha256"] = domain_sha256("model-transport-policy", document)
    validate_model_transport_policy(document)
    return document


def validate_model_transport_policy(document: Mapping[str, Any]) -> None:
    _exact_keys(document, _POLICY_KEYS, "model transport policy")
    projection = {
        key: copy.deepcopy(value)
        for key, value in document.items()
        if key != "policy_sha256"
    }
    expected = {
        "activation": _ACTIVATION,
        "arms": ["off", "on"],
        "artifacts": _ARTIFACTS,
        "capture_window": _CAPTURE_WINDOW,
        "enabled": True,
        "hash_domain": HASH_DOMAIN,
        "http": _HTTP_POLICY,
        "schema": MODEL_TRANSPORT_POLICY_SCHEMA,
    }
    for key, value in expected.items():
        _exact_json(projection.get(key), value, f"model transport policy {key}")
    sources = _mapping(projection.get("sources"), "model transport policy sources")
    _exact_keys(sources, frozenset(_SOURCE_PATHS), "model transport policy sources")
    for label, path in _SOURCE_PATHS.items():
        source = _mapping(sources[label], f"model transport policy source {label}")
        _exact_keys(
            source,
            frozenset({"path", "sha256"}),
            f"model transport policy source {label}",
        )
        if source["path"] != path:
            raise ValueError(f"model transport policy source {label} path changed")
        _digest(source["sha256"], f"model transport policy source {label} sha256")
    expected_sha = domain_sha256("model-transport-policy", projection)
    if document["policy_sha256"] != expected_sha:
        raise ValueError("model transport policy digest does not close")
    canonical_ascii_json(document)


def build_model_transport_call(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    capture_server: Mapping[str, Any],
    rollout_index: int,
    generation_seed: int,
    arrival_index: int,
    model_path: str,
    request_body: bytes,
    response_body: bytes,
    expected_generation_input: Sequence[Mapping[str, Any]] | None = None,
    expected_request_payload: Mapping[str, Any] | None = None,
    expected_response_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one raw-body call record.

    ``expected_*_payload`` lets the route bind the exact typed object it
    dispatched/returned to the raw UTF-8 body.  Closed campaign field checks
    are additionally enforced by transcript/evaluator joins.
    """
    _identity(pair_id, environment, arm)
    capture = _capture_server(capture_server)
    logical_index = _bounded_int(rollout_index, "rollout_index", 0, 3)
    arrival = _bounded_int(arrival_index, "arrival_index", 0, 3)
    seed = _bounded_int(generation_seed, "generation_seed", 0, MAX_INT63)
    expected_seed = derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=logical_index
    )
    if seed != expected_seed:
        raise ValueError("generation_seed does not close to logical rollout index")
    request_raw = _body(request_body, "request_body", MAX_REQUEST_BODY_BYTES)
    response_raw = _body(response_body, "response_body", MAX_RESPONSE_BODY_BYTES)
    request_payload = _strict_json_object(request_raw, "request body")
    response_payload = _strict_json_object(response_raw, "response body")
    _validate_request_payload(
        request_payload,
        generation_seed=seed,
        model_path=model_path,
        expected_generation_input=expected_generation_input,
    )
    _validate_response_payload(
        response_payload,
        request_payload=request_payload,
    )
    if expected_request_payload is not None:
        _exact_json(
            request_payload, expected_request_payload, "request raw/typed payload"
        )
    if expected_response_payload is not None:
        _exact_json(
            response_payload, expected_response_payload, "response raw/typed payload"
        )
    entry: dict[str, Any] = {
        "arrival_index": arrival,
        "generation_seed": seed,
        "request_body_base64": base64.b64encode(request_raw).decode("ascii"),
        "request_body_sha256": hashlib.sha256(request_raw).hexdigest(),
        "request_payload": request_payload,
        "request_payload_sha256": domain_sha256(
            "model-transport-request-payload", request_payload
        ),
        "response_body_base64": base64.b64encode(response_raw).decode("ascii"),
        "response_body_sha256": hashlib.sha256(response_raw).hexdigest(),
        "response_payload": response_payload,
        "response_payload_sha256": domain_sha256(
            "model-transport-response-payload", response_payload
        ),
        "rollout_index": logical_index,
        "schema": MODEL_TRANSPORT_CALL_SCHEMA,
    }
    entry["entry_sha256"] = _entry_sha256(
        pair_id=pair_id,
        environment=environment,
        arm=arm,
        capture_server=capture,
        entry=entry,
    )
    validate_model_transport_call(
        entry,
        pair_id=pair_id,
        environment=environment,
        arm=arm,
        capture_server=capture,
        model_path=model_path,
        expected_generation_input=expected_generation_input,
    )
    return entry


def validate_model_transport_call(
    document: Mapping[str, Any],
    *,
    pair_id: str,
    environment: str,
    arm: str,
    capture_server: Mapping[str, Any],
    model_path: str,
    expected_generation_input: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    _identity(pair_id, environment, arm)
    capture = _capture_server(capture_server)
    _exact_keys(document, _CALL_KEYS, "model transport call")
    if document["schema"] != MODEL_TRANSPORT_CALL_SCHEMA:
        raise ValueError("unexpected model transport call schema")
    index = _bounded_int(document["rollout_index"], "rollout_index", 0, 3)
    _bounded_int(document["arrival_index"], "arrival_index", 0, 3)
    seed = _bounded_int(document["generation_seed"], "generation_seed", 0, MAX_INT63)
    if seed != derive_nemo_gym_request_seed(
        seed_base=42, fixture_row_index=0, rollout_index=index
    ):
        raise ValueError("generation_seed does not close to logical rollout index")
    request_body = _decode_body(
        document["request_body_base64"], "request body", MAX_REQUEST_BODY_BYTES
    )
    response_body = _decode_body(
        document["response_body_base64"], "response body", MAX_RESPONSE_BODY_BYTES
    )
    if document["request_body_sha256"] != hashlib.sha256(request_body).hexdigest():
        raise ValueError("request body digest does not close")
    if document["response_body_sha256"] != hashlib.sha256(response_body).hexdigest():
        raise ValueError("response body digest does not close")
    request_payload = _strict_json_object(request_body, "request body")
    response_payload = _strict_json_object(response_body, "response body")
    _validate_request_payload(
        request_payload,
        generation_seed=seed,
        model_path=model_path,
        expected_generation_input=expected_generation_input,
    )
    _validate_response_payload(response_payload, request_payload=request_payload)
    _exact_json(document["request_payload"], request_payload, "request payload")
    _exact_json(document["response_payload"], response_payload, "response payload")
    if document["request_payload_sha256"] != domain_sha256(
        "model-transport-request-payload", request_payload
    ):
        raise ValueError("request payload digest does not close")
    if document["response_payload_sha256"] != domain_sha256(
        "model-transport-response-payload", response_payload
    ):
        raise ValueError("response payload digest does not close")
    if document["entry_sha256"] != _entry_sha256(
        pair_id=pair_id,
        environment=environment,
        arm=arm,
        capture_server=capture,
        entry=document,
    ):
        raise ValueError("model transport entry digest does not close")
    canonical_ascii_json(document)


def build_model_transport_bundle(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    model_transport_policy: Mapping[str, Any],
    capture_server: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    model_path: str,
    expected_generation_inputs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    validate_model_transport_policy(model_transport_policy)
    _identity(pair_id, environment, arm)
    capture = _capture_server(capture_server)
    materialized = [copy.deepcopy(dict(item)) for item in entries]
    document = {
        "arm": arm,
        "capture_server": capture,
        "capture_window": copy.deepcopy(model_transport_policy["capture_window"]),
        "endpoint": copy.deepcopy(MODEL_TRANSPORT_ENDPOINT),
        "entries": materialized,
        "entry_count": len(materialized),
        "environment": environment,
        "hash_domain": HASH_DOMAIN,
        "pair_id": pair_id,
        "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
    }
    document["ordered_entries_sha256"] = domain_sha256(
        "model-transport-ordered-entries", materialized
    )
    validate_model_transport_bundle(
        document,
        model_transport_policy=model_transport_policy,
        model_path=model_path,
        expected_generation_inputs=expected_generation_inputs,
    )
    return document


def validate_model_transport_bundle(
    document: Mapping[str, Any],
    *,
    model_transport_policy: Mapping[str, Any],
    model_path: str,
    expected_generation_inputs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> None:
    validate_model_transport_policy(model_transport_policy)
    _exact_keys(document, _BUNDLE_KEYS, "model transport bundle")
    if document["schema"] != MODEL_TRANSPORT_BUNDLE_SCHEMA:
        raise ValueError("unexpected model transport bundle schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected model transport bundle hash domain")
    pair_id, environment, arm = _identity(
        document["pair_id"], document["environment"], document["arm"]
    )
    _exact_json(document["endpoint"], MODEL_TRANSPORT_ENDPOINT, "transport endpoint")
    _exact_json(
        document["capture_window"],
        model_transport_policy["capture_window"],
        "transport capture window",
    )
    capture = _capture_server(document["capture_server"])
    if (
        type(document["entry_count"]) is not int
        or document["entry_count"] != K4_SAMPLES
    ):
        raise ValueError("model transport bundle must contain exactly four entries")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != K4_SAMPLES:
        raise ValueError("model transport bundle entries must be a four-element array")
    entries: list[dict[str, Any]] = []
    if (
        expected_generation_inputs is not None
        and len(expected_generation_inputs) != K4_SAMPLES
    ):
        raise ValueError("expected_generation_inputs must contain exactly four arrays")
    arrivals: set[int] = set()
    for index, raw in enumerate(raw_entries):
        entry = _mapping(raw, f"model transport entry {index}")
        validate_model_transport_call(
            entry,
            pair_id=pair_id,
            environment=environment,
            arm=arm,
            capture_server=capture,
            model_path=model_path,
            expected_generation_input=(
                None
                if expected_generation_inputs is None
                else expected_generation_inputs[index]
            ),
        )
        if entry["rollout_index"] != index:
            raise ValueError("model transport entries are not in logical 0..3 order")
        arrivals.add(entry["arrival_index"])
        entries.append(copy.deepcopy(dict(entry)))
    if arrivals != set(range(K4_SAMPLES)):
        raise ValueError("arrival_index values must be a diagnostic 0..3 permutation")
    fingerprints = {
        entry["response_payload"]["system_fingerprint"] for entry in entries
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "all K=4 responses must carry one byte-identical runtime fingerprint"
        )
    expected_ordered = domain_sha256("model-transport-ordered-entries", entries)
    if document["ordered_entries_sha256"] != expected_ordered:
        raise ValueError("ordered model transport entries digest does not close")
    payload = canonical_ascii_json(document)
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("model transport bundle exceeds the frozen byte limit")


def validate_model_transport_generation_request_join(
    bundle: Mapping[str, Any], *, generation_requests: Sequence[Mapping[str, Any]]
) -> None:
    """Bind each raw ChatCompletion request to its outer Gym request object."""
    if len(generation_requests) != K4_SAMPLES:
        raise ValueError("generation_requests must contain exactly four objects")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != K4_SAMPLES:
        raise ValueError("model transport bundle does not contain four entries")
    for index, raw_outer in enumerate(generation_requests):
        outer = _mapping(raw_outer, f"generation_requests[{index}]")
        _exact_keys(
            outer,
            _OUTER_GENERATION_REQUEST_KEYS,
            f"generation_requests[{index}]",
        )
        raw_payload = _mapping(
            entries[index].get("request_payload"),
            f"model transport request payload {index}",
        )
        expected_projection = {
            "input": raw_payload["messages"],
            "max_output_tokens": raw_payload["max_tokens"],
            "metadata": raw_payload["metadata"],
            "temperature": raw_payload["temperature"],
            "top_p": raw_payload["top_p"],
        }
        _exact_json(
            outer,
            expected_projection,
            f"outer/raw generation request {index}",
        )


def validate_model_transport_model_response_join(
    bundle: Mapping[str, Any],
    *,
    generation_requests: Sequence[Mapping[str, Any]],
    model_responses: Sequence[Mapping[str, Any]],
) -> None:
    """Bind raw ChatCompletion semantics to Gym's pinned Responses conversion."""
    if len(generation_requests) != K4_SAMPLES or len(model_responses) != K4_SAMPLES:
        raise ValueError(
            "generation_requests/model_responses must each contain four objects"
        )
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != K4_SAMPLES:
        raise ValueError("model transport bundle does not contain four entries")
    for index, (raw_request, raw_response) in enumerate(
        zip(generation_requests, model_responses, strict=True)
    ):
        generation_request = _mapping(raw_request, f"generation_requests[{index}]")
        model_response = _mapping(raw_response, f"model_responses[{index}]")
        validate_gym_model_response_r3(
            model_response,
            generation_request=generation_request,
            name=f"model_responses[{index}]",
        )
        request_payload = _mapping(
            entries[index]["request_payload"],
            f"model transport request payload {index}",
        )
        response_payload = _mapping(
            entries[index]["response_payload"],
            f"model transport response payload {index}",
        )
        choice = response_payload["choices"][0]
        raw_message = choice["message"]
        reasoning = raw_message["reasoning"]
        combined = (f"<think>{reasoning}</think>" if reasoning else "") + (
            raw_message["content"] or ""
        )
        reasoning_matches = _THINK_TAG_RE.findall(combined)
        cleaned = _THINK_TAG_RE.sub("", combined)
        actual_output = model_response["output"]
        expected_output: list[dict[str, Any]] = []
        if reasoning_matches:
            expected_output.append(
                {
                    "content": None,
                    "encrypted_content": None,
                    "id": actual_output[0]["id"],
                    "summary": [
                        {"text": text, "type": "summary_text"}
                        for text in reasoning_matches
                    ],
                    "type": "reasoning",
                }
            )
        if cleaned or not expected_output:
            item_index = len(expected_output)
            expected_output.append(
                {
                    "content": [
                        {
                            "annotations": [],
                            "logprobs": None,
                            "text": cleaned,
                            "type": "output_text",
                        }
                    ],
                    "id": actual_output[item_index]["id"],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            )
        expected_output[-1].update(
            {
                "generation_log_probs": copy.deepcopy(
                    raw_message["generation_log_probs"]
                ),
                "generation_token_ids": copy.deepcopy(
                    raw_message["generation_token_ids"]
                ),
                "prompt_token_ids": copy.deepcopy(raw_message["prompt_token_ids"]),
                "routed_experts": None,
            }
        )
        finish_reason = choice["finish_reason"]
        incomplete = (
            {"reason": "max_output_tokens"} if finish_reason == "length" else None
        )
        raw_usage = response_payload["usage"]
        expected = {
            "background": None,
            "conversation": None,
            "created_at": float(response_payload["created"]),
            "error": None,
            "id": model_response["id"],
            "incomplete_details": incomplete,
            "instructions": None,
            "max_output_tokens": generation_request["max_output_tokens"],
            "max_tool_calls": None,
            "metadata": copy.deepcopy(generation_request["metadata"]),
            "model": request_payload["model"],
            "object": "response",
            "output": expected_output,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "reasoning": None,
            "safety_identifier": None,
            "service_tier": None,
            "status": "incomplete" if incomplete is not None else "completed",
            "temperature": generation_request["temperature"],
            "text": None,
            "tool_choice": "auto",
            "tools": [],
            "top_logprobs": None,
            "top_p": generation_request["top_p"],
            "truncation": None,
            "usage": {
                "input_tokens": raw_usage["prompt_tokens"],
                "input_tokens_details": {"cached_tokens": None},
                "output_tokens": raw_usage["completion_tokens"],
                "output_tokens_details": {"reasoning_tokens": None},
                "total_tokens": raw_usage["total_tokens"],
            },
            "user": None,
        }
        _exact_json(
            model_response,
            expected,
            f"raw ChatCompletion/Gym model response projection {index}",
        )


def publish_model_transport_capture(
    *,
    transport_directory: str | Path,
    bundle: Mapping[str, Any],
    model_transport_policy: Mapping[str, Any],
    model_path: str,
    expected_generation_inputs: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the canonical four-line log and no-LF bundle exclusively."""
    validate_model_transport_bundle(
        bundle,
        model_transport_policy=model_transport_policy,
        model_path=model_path,
        expected_generation_inputs=expected_generation_inputs,
    )
    directory = _transport_directory(transport_directory)
    entries = bundle["entries"]
    log_bytes = b"".join(canonical_ascii_json(entry) + b"\n" for entry in entries)
    log_path = directory / MODEL_TRANSPORT_LOG
    log_sha = _publish_bytes(output=log_path, payload=log_bytes)
    bundle_path = directory / MODEL_TRANSPORT_BUNDLE
    _, bundle_sha = publish_evidence_document(
        output=bundle_path, document=bundle, trailing_lf=False
    )
    return (
        {
            "path": str(log_path),
            "record_count": K4_SAMPLES,
            "record_schema": MODEL_TRANSPORT_CALL_SCHEMA,
            "sha256": log_sha,
        },
        {
            "path": str(bundle_path),
            "schema": MODEL_TRANSPORT_BUNDLE_SCHEMA,
            "sha256": bundle_sha,
        },
    )


def load_finalized_model_transport_capture(
    *,
    results_dir: str | Path,
    expected_bundle_ref: Mapping[str, Any],
    expected_policy_sha256: str,
    source_root: str | Path,
    model_path: str,
    expected_generation_requests: Sequence[Mapping[str, Any]],
    expected_model_responses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reload the immutable post-attestation capture without ambient state."""
    policy = load_runtime_model_transport_policy(
        expected_policy_sha256=expected_policy_sha256, source_root=source_root
    )
    root = Path(_absolute_path(str(results_dir), "results_dir"))
    directory = _transport_directory(root / MODEL_TRANSPORT_DIRECTORY)
    expected_bundle_path = directory / MODEL_TRANSPORT_BUNDLE
    bundle_ref = _artifact_ref(
        expected_bundle_ref, MODEL_TRANSPORT_BUNDLE_SCHEMA, "expected_bundle_ref"
    )
    if bundle_ref["path"] != str(expected_bundle_path):
        raise ValueError("model transport bundle path does not match RESULTS_DIR")
    bundle, actual_bundle_sha = load_evidence_document(
        path=expected_bundle_path,
        expected_sha256=bundle_ref["sha256"],
        trailing_lf=False,
    )
    if actual_bundle_sha != bundle_ref["sha256"]:
        raise AssertionError("unreachable bundle digest mismatch")
    validate_model_transport_bundle(
        bundle,
        model_transport_policy=policy,
        model_path=model_path,
        expected_generation_inputs=_generation_inputs(expected_generation_requests),
    )
    validate_model_transport_generation_request_join(
        bundle, generation_requests=expected_generation_requests
    )
    validate_model_transport_model_response_join(
        bundle,
        generation_requests=expected_generation_requests,
        model_responses=expected_model_responses,
    )
    log_path = directory / MODEL_TRANSPORT_LOG
    log_raw = _load_immutable_bytes(log_path, maximum=MAX_BUNDLE_BYTES)
    expected_log = b"".join(
        canonical_ascii_json(entry) + b"\n" for entry in bundle["entries"]
    )
    if log_raw != expected_log:
        raise ValueError("model transport raw log differs from bundle entries")
    raw_log_ref = {
        "path": str(log_path),
        "record_count": K4_SAMPLES,
        "record_schema": MODEL_TRANSPORT_CALL_SCHEMA,
        "sha256": hashlib.sha256(log_raw).hexdigest(),
    }
    return policy, bundle, raw_log_ref, bundle_ref


def initialize_model_transport_directory(
    *, results_dir: str | Path, model_transport_policy: Mapping[str, Any]
) -> Path:
    """Exclusively create the arm-private transport directory, without parents."""
    validate_model_transport_policy(model_transport_policy)
    root = Path(_absolute_path(str(results_dir), "results_dir"))
    root_metadata = os.lstat(root)
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("results_dir must be a real directory")
    if root_metadata.st_uid != os.geteuid():
        raise PermissionError("results_dir must be EUID-owned")
    target = root / MODEL_TRANSPORT_DIRECTORY
    try:
        os.mkdir(target, stat.S_IRWXU)
    except FileExistsError as error:
        raise FileExistsError(
            "strict model transport directory must be absent before runtime"
        ) from error
    os.chmod(target, stat.S_IRWXU, follow_symlinks=False)
    metadata = os.lstat(target)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeError("strict model transport directory creation did not close")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return target


def build_model_transport_manifest(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    pair_manifest_sha256: str,
    authenticated_job_id: str,
    submission_receipt_sha256: str,
    capture_server: Mapping[str, Any],
    main_transcript_bundle: Mapping[str, Any],
    main_ledger: Mapping[str, Any],
    transport_bundle: Mapping[str, Any],
    transport_capture: Mapping[str, Any],
    model_transport_policy_sha256: str,
    entry_count: int,
    ordered_entries_sha256: str,
) -> dict[str, Any]:
    _identity(pair_id, environment, arm)
    document = {
        "arm": arm,
        "authenticated_job_id": _job_id(authenticated_job_id, "authenticated_job_id"),
        "capture_server": _capture_server(capture_server),
        "entry_count": _bounded_int(entry_count, "entry_count", K4_SAMPLES, K4_SAMPLES),
        "environment": environment,
        "hash_domain": HASH_DOMAIN,
        "main_ledger": _artifact_ref(
            main_ledger, MAIN_STEP1_LEDGER_SCHEMA, "main_ledger"
        ),
        "main_transcript_bundle": _artifact_ref(
            main_transcript_bundle,
            STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
            "main_transcript_bundle",
        ),
        "model_transport_policy_sha256": _digest(
            model_transport_policy_sha256, "model_transport_policy_sha256"
        ),
        "ordered_entries_sha256": _digest(
            ordered_entries_sha256, "ordered_entries_sha256"
        ),
        "pair_id": pair_id,
        "pair_manifest_sha256": _digest(pair_manifest_sha256, "pair_manifest_sha256"),
        "schema": MODEL_TRANSPORT_MANIFEST_SCHEMA,
        "submission_receipt_sha256": _digest(
            submission_receipt_sha256, "submission_receipt_sha256"
        ),
        "transport_bundle": _artifact_ref(
            transport_bundle, MODEL_TRANSPORT_BUNDLE_SCHEMA, "transport_bundle"
        ),
        "transport_capture": _capture_ref(transport_capture),
    }
    validate_model_transport_manifest(document)
    return document


def validate_model_transport_manifest(document: Mapping[str, Any]) -> None:
    _exact_keys(document, _MANIFEST_KEYS, "model transport manifest")
    if document["schema"] != MODEL_TRANSPORT_MANIFEST_SCHEMA:
        raise ValueError("unexpected model transport manifest schema")
    if document["hash_domain"] != HASH_DOMAIN:
        raise ValueError("unexpected model transport manifest hash domain")
    _identity(document["pair_id"], document["environment"], document["arm"])
    _job_id(document["authenticated_job_id"], "authenticated_job_id")
    _bounded_int(document["entry_count"], "entry_count", K4_SAMPLES, K4_SAMPLES)
    _capture_server(document["capture_server"])
    for name in (
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "model_transport_policy_sha256",
        "ordered_entries_sha256",
    ):
        _digest(document[name], name)
    _artifact_ref(
        document["main_transcript_bundle"],
        STEP1_TRANSCRIPT_BUNDLE_SCHEMA,
        "main_transcript_bundle",
    )
    _artifact_ref(document["main_ledger"], MAIN_STEP1_LEDGER_SCHEMA, "main_ledger")
    _artifact_ref(
        document["transport_bundle"], MODEL_TRANSPORT_BUNDLE_SCHEMA, "transport_bundle"
    )
    _capture_ref(document["transport_capture"])
    canonical_ascii_json(document)


def publish_model_transport_manifest(
    *, transport_directory: str | Path, manifest: Mapping[str, Any]
) -> tuple[Path, str]:
    validate_model_transport_manifest(manifest)
    directory = _transport_directory(transport_directory)
    return publish_evidence_document(
        output=directory / MODEL_TRANSPORT_MANIFEST,
        document=manifest,
        trailing_lf=False,
    )


def _validate_request_payload(
    payload: Mapping[str, Any],
    *,
    generation_seed: int,
    model_path: str,
    expected_generation_input: Sequence[Mapping[str, Any]] | None,
) -> None:
    _exact_keys(payload, _REQUEST_KEYS, "chat completion request payload")
    template = _mapping(payload["chat_template_kwargs"], "chat_template_kwargs")
    _exact_json(
        template,
        {"enable_thinking": True, "truncate_history_thinking": False},
        "chat_template_kwargs",
    )
    if payload["logprobs"] is not True or type(payload["logprobs"]) is not bool:
        raise ValueError("request logprobs must be true")
    if (
        payload["return_tokens_as_token_ids"] is not True
        or type(payload["return_tokens_as_token_ids"]) is not bool
    ):
        raise ValueError("request return_tokens_as_token_ids must be true")
    _bounded_int(payload["max_tokens"], "request max_tokens", 768, 768)
    _bounded_int(payload["top_logprobs"], "request top_logprobs", 0, 0)
    _bounded_int(payload["seed"], "request seed", generation_seed, generation_seed)
    for name in ("temperature", "top_p"):
        if type(payload[name]) is not float or payload[name] != 1.0:
            raise ValueError(f"request {name} must be exact float 1.0")
    if type(model_path) is not str or not model_path:
        raise ValueError("model_path must be nonempty text")
    if payload["model"] != model_path or type(payload["model"]) is not str:
        raise ValueError("request model must equal the authenticated Pair model path")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("request messages must contain exactly one message")
    message = _mapping(messages[0], "request message")
    _exact_keys(message, frozenset({"content", "role"}), "request message")
    if message["role"] != "user" or type(message["role"]) is not str:
        raise ValueError("request message role must be user")
    _bounded_text(message["content"], "request message content", MAX_REQUEST_BODY_BYTES)
    if expected_generation_input is not None:
        if not isinstance(expected_generation_input, (list, tuple)):
            raise TypeError("expected_generation_input must be an array")
        _exact_json(
            messages, list(expected_generation_input), "request/outer generation input"
        )
    metadata = _mapping(payload["metadata"], "request metadata")
    _exact_keys(metadata, frozenset({"extra_body"}), "request metadata")
    expected_extra_body = json.dumps(
        {"seed": generation_seed}, sort_keys=True, separators=(",", ":")
    )
    if (
        metadata["extra_body"] != expected_extra_body
        or type(metadata["extra_body"]) is not str
    ):
        raise ValueError(
            "request metadata.extra_body is not the exact compact seed object"
        )


def _validate_response_payload(
    payload: Mapping[str, Any], *, request_payload: Mapping[str, Any]
) -> None:
    _exact_keys(payload, _RESPONSE_KEYS, "chat completion response payload")
    for name in (
        "kv_transfer_params",
        "metrics",
        "prompt_logprobs",
        "prompt_text",
        "prompt_token_ids",
        "service_tier",
    ):
        if payload[name] is not None:
            raise ValueError(f"response {name} must be null")
    if payload["object"] != "chat.completion" or type(payload["object"]) is not str:
        raise ValueError("response object must be chat.completion")
    if (
        payload["model"] != request_payload["model"]
        or type(payload["model"]) is not str
    ):
        raise ValueError("response model differs from request model")
    _bounded_int(payload["created"], "response created", 1, MAX_INT63)
    if type(payload["id"]) is not str or not _CHAT_ID_RE.fullmatch(payload["id"]):
        raise ValueError("response id does not match the frozen ChatCompletion form")
    fingerprint = payload["system_fingerprint"]
    if fingerprint is not None and (
        type(fingerprint) is not str or not _FINGERPRINT_RE.fullmatch(fingerprint)
    ):
        raise ValueError("response system_fingerprint differs from pinned vLLM 0.25.1")
    choices = payload["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response choices must contain exactly one choice")
    choice = _mapping(choices[0], "response choice")
    _exact_keys(choice, _CHOICE_KEYS, "response choice")
    _bounded_int(choice["index"], "response choice index", 0, 0)
    if type(choice["finish_reason"]) is not str or choice["finish_reason"] not in {
        "length",
        "stop",
    }:
        raise ValueError("response finish_reason must be stop or length")
    for name in ("routed_experts", "token_ids"):
        if choice[name] is not None:
            raise ValueError(f"response choice {name} must be null")
    stop_reason = choice["stop_reason"]
    if stop_reason is not None:
        if type(stop_reason) is str:
            _bounded_text_allow_empty(stop_reason, "response choice stop_reason", 4096)
        elif type(stop_reason) is int:
            _bounded_int(stop_reason, "response choice stop_reason", 0, MAX_INT31)
        else:
            raise TypeError(
                "response choice stop_reason must be null, UTF-8 text, or int31"
            )
    message = _mapping(choice["message"], "response choice message")
    _exact_keys(message, _MESSAGE_KEYS, "response choice message")
    for name in ("annotations", "audio", "function_call", "refusal"):
        if message[name] is not None:
            raise ValueError(f"response message {name} must be null")
    if message["role"] != "assistant" or type(message["role"]) is not str:
        raise ValueError("response message role must be assistant")
    if message["content"] is not None:
        _bounded_text_allow_empty(
            message["content"], "response message content", MAX_RESPONSE_BODY_BYTES
        )
    if message["reasoning"] is not None:
        _bounded_text_allow_empty(
            message["reasoning"],
            "response message reasoning",
            MAX_RESPONSE_BODY_BYTES,
        )
    prompt_tokens = _int_array(
        message["prompt_token_ids"], "response prompt_token_ids", 1, 131_072, MAX_INT31
    )
    generation_tokens = _int_array(
        message["generation_token_ids"],
        "response generation_token_ids",
        1,
        131_072,
        MAX_INT31,
    )
    if len(prompt_tokens) + len(generation_tokens) > 131_072:
        raise ValueError(
            "response prompt and generation tokens exceed the accumulated cap"
        )
    generation_log_probs = _number_array(
        message["generation_log_probs"],
        "response generation_log_probs",
        len(generation_tokens),
    )
    logprobs = _mapping(choice["logprobs"], "response choice logprobs")
    _exact_keys(logprobs, frozenset({"content"}), "response choice logprobs")
    content = logprobs["content"]
    if not isinstance(content, list) or len(content) != len(generation_tokens):
        raise ValueError(
            "response choice logprobs content does not match generation length"
        )
    for index, raw_item in enumerate(content):
        item = _mapping(raw_item, f"response token logprob {index}")
        _exact_keys(item, _TOKEN_LOGPROB_KEYS, f"response token logprob {index}")
        byte_values = _int_array(
            item["bytes"], f"response token logprob {index} bytes", 1, 4096, 255
        )
        if not byte_values:
            raise AssertionError("unreachable empty byte vector")
        _number(item["logprob"], f"response token logprob {index} logprob")
        _exact_json(
            item["logprob"],
            generation_log_probs[index],
            f"response token logprob {index} mirrored logprob",
        )
        token_text = _bounded_text(
            item["token"], f"response token logprob {index} token", 4096
        )
        if token_text != f"token_id:{generation_tokens[index]}":
            raise ValueError(f"response token logprob {index} token id does not close")
        if item["top_logprobs"] != [] or not isinstance(item["top_logprobs"], list):
            raise ValueError(
                f"response token logprob {index} top_logprobs must be empty"
            )
    usage = _mapping(payload["usage"], "response usage")
    _exact_keys(usage, _USAGE_KEYS, "response usage")
    if usage["prompt_tokens_details"] is not None:
        raise ValueError("response usage prompt_tokens_details must be null")
    expected_usage = {
        "completion_tokens": len(generation_tokens),
        "prompt_tokens": len(prompt_tokens),
        "total_tokens": len(prompt_tokens) + len(generation_tokens),
    }
    for name, expected in expected_usage.items():
        _bounded_int(usage[name], f"response usage {name}", expected, expected)


def _entry_sha256(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    capture_server: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> str:
    projection = {
        key: copy.deepcopy(value)
        for key, value in entry.items()
        if key != "entry_sha256"
    }
    return domain_sha256(
        "model-transport-entry",
        {
            "arm": arm,
            "capture_server": copy.deepcopy(dict(capture_server)),
            "endpoint": copy.deepcopy(MODEL_TRANSPORT_ENDPOINT),
            "entry": projection,
            "environment": environment,
            "pair_id": pair_id,
        },
    )


def _strict_json_object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be UTF-8") from error

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} has duplicate object key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    canonical_ascii_json(value)
    return copy.deepcopy(value)


def _capture_server(value: Mapping[str, Any]) -> dict[str, Any]:
    server = _mapping(value, "capture_server")
    _exact_keys(server, _CAPTURE_SERVER_KEYS, "capture_server")
    result = {
        "boot_id_sha256": _digest(
            server["boot_id_sha256"], "capture_server.boot_id_sha256"
        ),
        "hostname": _ascii(server["hostname"], "capture_server.hostname", 255),
        "pid": _bounded_int(server["pid"], "capture_server.pid", 1, MAX_INT31),
        "start_time_ticks": _bounded_int(
            server["start_time_ticks"], "capture_server.start_time_ticks", 1, MAX_INT63
        ),
    }
    if not _HOSTNAME_RE.fullmatch(result["hostname"]) or len(result["hostname"]) > 255:
        raise ValueError("capture_server.hostname is not a closed hostname")
    expected_instance_id = domain_sha256("model-transport-server-instance", result)
    actual_instance_id = _safe_id(
        server["server_instance_id"], "capture_server.server_instance_id"
    )
    if actual_instance_id != expected_instance_id:
        raise ValueError(
            "capture_server.server_instance_id does not close over process identity"
        )
    result["server_instance_id"] = actual_instance_id
    return result


def _artifact_ref(value: Mapping[str, Any], schema: str, name: str) -> dict[str, Any]:
    ref = _mapping(value, name)
    _exact_keys(ref, _REF_KEYS, name)
    if ref["schema"] != schema:
        raise ValueError(f"{name}.schema is not {schema}")
    return {
        "path": _absolute_path(ref["path"], f"{name}.path"),
        "schema": schema,
        "sha256": _digest(ref["sha256"], f"{name}.sha256"),
    }


def _capture_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    ref = _mapping(value, "transport_capture")
    _exact_keys(ref, _CAPTURE_REF_KEYS, "transport_capture")
    if ref["record_schema"] != MODEL_TRANSPORT_CALL_SCHEMA:
        raise ValueError("transport_capture.record_schema changed")
    return {
        "path": _absolute_path(ref["path"], "transport_capture.path"),
        "record_count": _bounded_int(
            ref["record_count"],
            "transport_capture.record_count",
            K4_SAMPLES,
            K4_SAMPLES,
        ),
        "record_schema": MODEL_TRANSPORT_CALL_SCHEMA,
        "sha256": _digest(ref["sha256"], "transport_capture.sha256"),
    }


def _transport_directory(value: str | Path) -> Path:
    path = Path(_absolute_path(str(value), "transport_directory"))
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("transport_directory must be a real directory")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError("transport_directory must be EUID-owned mode 0700")
    return path


def _publish_bytes(*, output: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    candidate = output.with_name(f".{output.name}.candidate")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd: int | None = None
    created = False
    try:
        fd = os.open(candidate, flags, stat.S_IRUSR)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing transport capture")
            view = view[written:]
        os.fchmod(fd, stat.S_IRUSR)
        os.fsync(fd)
        metadata = os.fstat(fd)
        if (
            metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise RuntimeError("transport capture candidate inode changed")
        os.close(fd)
        fd = None
        try:
            os.link(candidate, output, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"transport capture output already exists: {output}"
            ) from error
        os.unlink(candidate)
        created = False
        directory_fd = os.open(
            output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if created:
            try:
                os.unlink(candidate)
            except FileNotFoundError:
                pass
    return digest


def _write_fd(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while publishing model transport evidence")
        view = view[written:]


def _stable_file_sha256(path: Path) -> str:
    _reject_symlink_components(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"attested source is not regular: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise RuntimeError(f"attested source changed during read: {path}")
    return digest.hexdigest()


def _load_immutable_bytes(path: Path, *, maximum: int) -> bytes:
    _reject_symlink_components(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise RuntimeError("transport evidence must be EUID-owned mode0400 nlink1")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if len(raw) > maximum or (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("transport evidence changed during read")
    return raw


def _reject_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("secure path must be absolute")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        metadata = os.lstat(cursor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"secure path contains a symlink: {cursor}")


def _decode_body(value: Any, name: str, maximum: int) -> bytes:
    if type(value) is not str or not value:
        raise TypeError(f"{name}_base64 must be nonempty ASCII text")
    try:
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        raise ValueError(f"{name}_base64 is not canonical RFC4648 base64") from error
    if base64.b64encode(decoded) != encoded:
        raise ValueError(f"{name}_base64 is not canonical RFC4648 base64")
    return _body(decoded, name, maximum)


def _body(value: Any, name: str, maximum: int) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{name} must be bytes")
    if not value or len(value) > maximum:
        raise ValueError(f"{name} must contain 1..{maximum} bytes")
    return value


def _identity(pair_id: Any, environment: Any, arm: Any) -> tuple[str, str, str]:
    pair = _safe_id(pair_id, "pair_id")
    if environment not in {"reasoning_gym", "citation", "freeform"}:
        raise ValueError("environment is not one of the three frozen environments")
    if arm not in {"off", "on"}:
        raise ValueError("arm must be off or on")
    return pair, environment, arm


def _generation_inputs(
    generation_requests: Sequence[Mapping[str, Any]],
) -> list[Sequence[Mapping[str, Any]]]:
    if len(generation_requests) != K4_SAMPLES:
        raise ValueError("generation_requests must contain exactly four objects")
    result: list[Sequence[Mapping[str, Any]]] = []
    for index, raw in enumerate(generation_requests):
        request = _mapping(raw, f"generation_requests[{index}]")
        inputs = request.get("input")
        if not isinstance(inputs, list):
            raise TypeError(f"generation_requests[{index}].input must be an array")
        result.append(inputs)
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{name} has unexpected keys")


def _exact_json(actual: Any, expected: Any, name: str) -> None:
    if canonical_ascii_json(actual) != canonical_ascii_json(expected):
        raise ValueError(f"{name} does not match exactly")


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value) or value == "0" * 64:
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256 digest")
    return value


def _safe_id(value: Any, name: str) -> str:
    if type(value) is not str or not _SAFE_ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a bounded safe identifier")
    return value


def _job_id(value: Any, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[1-9][0-9]{0,18}", value):
        raise ValueError(f"{name} must be a canonical positive decimal string")
    if int(value, 10) > MAX_INT63:
        raise ValueError(f"{name} exceeds signed int63")
    return value


def _ascii(value: Any, name: str, maximum: int) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be bounded nonempty ASCII")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be ASCII") from error
    return value


def _bounded_text(value: Any, name: str, maximum: int) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(
            f"{name} must be nonempty UTF-8 text of at most {maximum} bytes"
        )
    return value


def _bounded_text_allow_empty(value: Any, name: str, maximum: int) -> str:
    if type(value) is not str or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be UTF-8 text of at most {maximum} bytes")
    return value


def _int_array(
    value: Any, name: str, minimum_length: int, maximum_length: int, maximum_value: int
) -> list[int]:
    if (
        not isinstance(value, list)
        or not minimum_length <= len(value) <= maximum_length
    ):
        raise ValueError(
            f"{name} must have {minimum_length}..{maximum_length} elements"
        )
    return [
        _bounded_int(item, f"{name}[{index}]", 0, maximum_value)
        for index, item in enumerate(value)
    ]


def _number(value: Any, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact finite JSON float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value == 0 and math.copysign(1.0, value) < 0:
        raise ValueError(f"{name} must not be negative zero")
    return value


def _number_array(value: Any, name: str, exact_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != exact_length:
        raise ValueError(f"{name} must have exactly {exact_length} elements")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _absolute_path(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise ValueError(f"{name} must be an absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or str(path) != value
    ):
        raise ValueError(f"{name} must be canonical and absolute")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum},{maximum}]")
    return value
