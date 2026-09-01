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

"""Fail-closed replay of captured NeMo Gym chat-completion responses."""

import copy
import hashlib
import json
import math
import os
import re
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

REPLAY_BUNDLE_ENV = "NEMORL_VLLM_REPLAY_BUNDLE"
REPLAY_BUNDLE_SHA256_ENV = "NEMORL_VLLM_REPLAY_BUNDLE_SHA256"
REPLAY_BUNDLE_SCHEMA = "nemorl-gym-captured-cohort-replay-v1"
REPLAY_MARKER_PREFIX = "NEMORL_VLLM_REPLAY "
REPLAY_READY_MARKER_PREFIX = "NEMORL_VLLM_REPLAY_READY "
STRICT_REPLAY_ENTRIES = 4

_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOKEN_ID_PATTERN = re.compile(r"token_id:(?P<token_id>[0-9]+)")

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
type JsonObject = dict[str, JsonValue]


class VllmReplayError(RuntimeError):
    """Base class for replay failures."""


class VllmReplayConfigurationError(VllmReplayError):
    """The configured replay bundle is unsafe or invalid."""


class VllmReplayDirectGenerationError(VllmReplayError):
    """A direct generation API cannot honor required captured replay."""


class VllmReplayRequestError(VllmReplayError):
    """A replay-enabled HTTP request cannot be served from the bundle."""

    status_code: int = 409
    error_code: str = "replay_request_rejected"


class VllmReplayMissError(VllmReplayRequestError):
    """No captured response exists for the canonical request."""

    error_code = "replay_miss"


class VllmReplayReuseError(VllmReplayRequestError):
    """The captured response for this request was already consumed."""

    error_code = "replay_reuse"


class VllmReplayStreamingError(VllmReplayRequestError):
    """Streaming is deliberately unsupported in captured-cohort replay."""

    status_code = 400
    error_code = "replay_streaming_unsupported"


class VllmReplayInvalidRequestError(VllmReplayRequestError):
    """The raw HTTP request is not strict JSON and cannot be replayed."""

    status_code = 400
    error_code = "replay_invalid_request"


@dataclass(frozen=True)
class _ReplayEntry:
    seed: int
    request_sha256: str
    response: JsonObject
    response_sha256: str


@dataclass(frozen=True)
class VllmReplayConfigContract:
    """Validated, environment-independent replay intent from ``vllm_cfg``."""

    required: bool
    bundle_sha256: str | None


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize JSON exactly as NeMo Gym hashes captured HTTP payloads."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not strict JSON") from error
    return encoded.encode("utf-8")


def sha256_json(value: JsonValue) -> str:
    """Return the NeMo Gym canonical JSON SHA256 digest for ``value``."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_object_keys(
    pairs: Sequence[tuple[str, JsonValue]],
) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise VllmReplayConfigurationError(
                "replay bundle contains a duplicate JSON object key"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise VllmReplayConfigurationError(
        f"replay bundle contains non-finite JSON constant {value}"
    )


def _decode_bundle_json(raw: bytes) -> JsonValue:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VllmReplayConfigurationError(
            "replay bundle is not valid UTF-8"
        ) from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_json_constant,
        )
    except VllmReplayConfigurationError:
        raise
    except json.JSONDecodeError as error:
        raise VllmReplayConfigurationError("replay bundle is not valid JSON") from error


def _decode_request_json(raw: bytes) -> JsonObject:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VllmReplayInvalidRequestError(
            "replay request body is not valid UTF-8"
        ) from error

    def reject_request_duplicate_keys(
        pairs: Sequence[tuple[str, JsonValue]],
    ) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise VllmReplayInvalidRequestError(
                    "replay request body contains a duplicate JSON object key"
                )
            result[key] = value
        return result

    def reject_request_constant(_value: str) -> None:
        raise VllmReplayInvalidRequestError(
            "replay request body contains a non-finite JSON value"
        )

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_request_duplicate_keys,
            parse_constant=reject_request_constant,
        )
    except VllmReplayInvalidRequestError:
        raise
    except json.JSONDecodeError as error:
        raise VllmReplayInvalidRequestError(
            "replay request body is not valid JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise VllmReplayInvalidRequestError("replay request body must be a JSON object")
    return decoded


def _require_exact_keys(
    value: Mapping[str, JsonValue], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise VllmReplayConfigurationError(
            f"{context} must contain exactly {sorted(expected)}"
        )


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise VllmReplayConfigurationError(
            f"{context} must be a lowercase SHA256 digest"
        )
    return value


def _require_data_parallel_size(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise VllmReplayConfigurationError(f"{context} must be a positive integer")
    return value


def validate_vllm_replay_config(
    vllm_cfg: Mapping[str, object],
    *,
    data_parallel_size: int | None = None,
) -> VllmReplayConfigContract:
    """Validate replay intent independently of replay environment variables."""
    required_value = vllm_cfg.get("replay_required", False)
    if type(required_value) is not bool:
        raise VllmReplayConfigurationError("vllm_cfg.replay_required must be a boolean")
    required = required_value

    configured_sha256_value = vllm_cfg.get("replay_bundle_sha256")
    if configured_sha256_value is None:
        configured_sha256 = None
    else:
        configured_sha256 = _require_sha256(
            configured_sha256_value, "vllm_cfg.replay_bundle_sha256"
        )

    if required:
        if configured_sha256 is None:
            raise VllmReplayConfigurationError(
                "vllm_cfg.replay_bundle_sha256 is required when replay_required=true"
            )
        if vllm_cfg.get("async_engine") is not True:
            raise VllmReplayConfigurationError(
                "required replay needs vllm_cfg.async_engine=true"
            )
        if vllm_cfg.get("expose_http_server") is not True:
            raise VllmReplayConfigurationError(
                "required replay needs vllm_cfg.expose_http_server=true"
            )
    elif configured_sha256 is not None:
        raise VllmReplayConfigurationError(
            "vllm_cfg.replay_bundle_sha256 must be null or absent when "
            "replay_required=false"
        )

    if data_parallel_size is not None:
        parsed_data_parallel_size = _require_data_parallel_size(
            data_parallel_size, "replay data-parallel size"
        )
        if required and parsed_data_parallel_size != 1:
            raise VllmReplayConfigurationError(
                "required replay supports only data-parallel size 1 because "
                "response claims are actor-local"
            )

    return VllmReplayConfigContract(
        required=required,
        bundle_sha256=configured_sha256,
    )


def reject_vllm_direct_async_generation_during_replay(
    vllm_cfg: Mapping[str, object],
    *,
    method_name: str,
    replay_bundle_loaded: bool = False,
) -> None:
    """Reject direct async generation that cannot claim an HTTP replay entry.

    Captured responses are keyed by the canonical raw request body received at
    ``/v1/chat/completions``. The direct token and text APIs do not carry that
    request envelope, so routing them to the live engine while replay is active
    would silently mix fresh samples into a deterministic captured cohort.

    Args:
        vllm_cfg: vLLM generation configuration containing replay intent.
        method_name: Direct async API being guarded.
        replay_bundle_loaded: Whether the worker already holds a replay bundle.

    Raises:
        VllmReplayDirectGenerationError: If required replay is configured or a
            replay bundle is already loaded in the worker.
    """
    contract = validate_vllm_replay_config(vllm_cfg)
    if not contract.required and not replay_bundle_loaded:
        return
    replay_state = (
        "vllm_cfg.replay_required=true"
        if contract.required
        else "a replay bundle is loaded"
    )
    raise VllmReplayDirectGenerationError(
        f"{method_name} is unavailable while {replay_state}: "
        "captured-cohort replay can only be claimed through "
        "/v1/chat/completions. Refusing to sample from the live vLLM engine."
    )


def validate_vllm_replay_worker_topology(
    vllm_cfg: Mapping[str, object],
    vllm_kwargs: Mapping[str, object],
    *,
    outer_data_parallel_size: int,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Attest that all worker-visible DP signals are one for replay.

    ``outer_data_parallel_size`` is injected from ``VllmGeneration.dp_size``;
    the other signals cover vLLM's optional internal data-parallel modes.
    """
    contract = validate_vllm_replay_config(
        vllm_cfg, data_parallel_size=outer_data_parallel_size
    )
    environment = os.environ if environ is None else environ
    replay_enabled = bool(
        contract.required
        or environment.get(REPLAY_BUNDLE_ENV)
        or environment.get(REPLAY_BUNDLE_SHA256_ENV)
    )
    parsed_outer_size = _require_data_parallel_size(
        outer_data_parallel_size, "NeMo-RL generation data-parallel size"
    )
    if not replay_enabled:
        return parsed_outer_size

    tensor_parallel_size = _require_data_parallel_size(
        vllm_cfg.get("tensor_parallel_size"), "vllm_cfg.tensor_parallel_size"
    )
    expert_parallel_size = _require_data_parallel_size(
        vllm_cfg.get("expert_parallel_size"), "vllm_cfg.expert_parallel_size"
    )
    if expert_parallel_size > tensor_parallel_size:
        if expert_parallel_size % tensor_parallel_size != 0:
            raise VllmReplayConfigurationError(
                "vllm_cfg.expert_parallel_size must be divisible by "
                "tensor_parallel_size"
            )
        expert_derived_size = expert_parallel_size // tensor_parallel_size
    else:
        expert_derived_size = 1

    dp_signals: dict[str, int] = {
        "NeMo-RL generation": parsed_outer_size,
        "expert-parallel-derived vLLM": expert_derived_size,
    }
    for key in ("data_parallel_size", "data_parallel_size_local"):
        if key in vllm_kwargs:
            dp_signals[f"vllm_kwargs.{key}"] = _require_data_parallel_size(
                vllm_kwargs[key], f"vllm_kwargs.{key}"
            )
    environment_dp_size = environment.get("VLLM_DP_SIZE")
    if environment_dp_size is not None:
        try:
            parsed_environment_dp_size: object = int(environment_dp_size)
        except ValueError as error:
            raise VllmReplayConfigurationError(
                "VLLM_DP_SIZE must be a positive integer during replay"
            ) from error
        dp_signals["VLLM_DP_SIZE"] = _require_data_parallel_size(
            parsed_environment_dp_size, "VLLM_DP_SIZE"
        )

    non_unit_signals = {name: value for name, value in dp_signals.items() if value != 1}
    if non_unit_signals:
        details = ", ".join(
            f"{name}={value}" for name, value in sorted(non_unit_signals.items())
        )
        raise VllmReplayConfigurationError(
            "captured-cohort replay requires every data-parallel size to be 1 "
            f"because response claims are actor-local; got {details}"
        )
    return parsed_outer_size


def validate_vllm_replay_environment_contract(
    vllm_cfg: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> VllmReplayConfigContract:
    """Reject environment-only replay before worker-path selection."""
    contract = validate_vllm_replay_config(vllm_cfg)
    environment = os.environ if environ is None else environ
    forwarded_environment = vllm_cfg.get("env_vars", {})
    if not isinstance(forwarded_environment, Mapping):
        raise VllmReplayConfigurationError("vllm_cfg.env_vars must be a mapping")
    replay_environment_present = bool(
        environment.get(REPLAY_BUNDLE_ENV)
        or environment.get(REPLAY_BUNDLE_SHA256_ENV)
        or forwarded_environment.get(REPLAY_BUNDLE_ENV)
        or forwarded_environment.get(REPLAY_BUNDLE_SHA256_ENV)
    )
    if replay_environment_present and not contract.required:
        raise VllmReplayConfigurationError(
            "replay bundle environment variables require vllm_cfg.replay_required=true"
        )
    return contract


def _require_int(value: JsonValue, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise VllmReplayConfigurationError(f"{context} must be an integer >= {minimum}")
    return value


def _require_float(value: JsonValue, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VllmReplayConfigurationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VllmReplayConfigurationError(f"{context} must be finite")
    return result


def _require_int_list(value: JsonValue, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise VllmReplayConfigurationError(f"{context} must be a list")
    return tuple(
        _require_int(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _require_float_list(value: JsonValue, context: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise VllmReplayConfigurationError(f"{context} must be a list")
    return tuple(
        _require_float(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _validate_request_payload(
    request: JsonObject, context: str, *, expected_seed: int
) -> None:
    seed = _require_int(request.get("seed"), f"{context}.seed")
    if seed != expected_seed:
        raise VllmReplayConfigurationError(
            f"{context}.seed must equal entry seed {expected_seed}"
        )
    if request.get("logprobs") is not True:
        raise VllmReplayConfigurationError(f"{context}.logprobs must be exactly true")
    if request.get("return_tokens_as_token_ids") is not True:
        raise VllmReplayConfigurationError(
            f"{context}.return_tokens_as_token_ids must be exactly true"
        )
    _require_int(request.get("max_tokens"), f"{context}.max_tokens", minimum=1)
    if type(request.get("top_logprobs")) is not int or request.get("top_logprobs") != 0:
        raise VllmReplayConfigurationError(f"{context}.top_logprobs must be exactly 0")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise VllmReplayConfigurationError(
            f"{context}.messages must be a nonempty list"
        )


def _validate_chat_completion_response(response: JsonObject, context: str) -> None:
    required_keys = {"id", "object", "created", "model", "choices"}
    if not required_keys.issubset(response):
        raise VllmReplayConfigurationError(
            f"{context} is missing required ChatCompletion response fields"
        )
    if not isinstance(response["id"], str) or not response["id"]:
        raise VllmReplayConfigurationError(f"{context}.id must be a nonempty string")
    if response["object"] != "chat.completion":
        raise VllmReplayConfigurationError(
            f"{context}.object must be 'chat.completion'"
        )
    _require_int(response["created"], f"{context}.created")
    if not isinstance(response["model"], str) or not response["model"]:
        raise VllmReplayConfigurationError(f"{context}.model must be a nonempty string")

    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise VllmReplayConfigurationError(
            f"{context}.choices must contain exactly one choice"
        )
    choice = choices[0]
    choice_context = f"{context}.choices[0]"
    if not isinstance(choice, dict):
        raise VllmReplayConfigurationError(f"{choice_context} must be a JSON object")
    if not {"index", "message", "finish_reason"}.issubset(choice):
        raise VllmReplayConfigurationError(
            f"{choice_context} is missing required fields"
        )
    if type(choice["index"]) is not int or choice["index"] != 0:
        raise VllmReplayConfigurationError(f"{choice_context}.index must be exactly 0")
    finish_reason = choice["finish_reason"]
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise VllmReplayConfigurationError(
            f"{choice_context}.finish_reason must be a string or null"
        )
    message = choice["message"]
    if not isinstance(message, dict):
        raise VllmReplayConfigurationError(
            f"{choice_context}.message must be a JSON object"
        )
    if not isinstance(message.get("role"), str):
        raise VllmReplayConfigurationError(
            f"{choice_context}.message.role must be a string"
        )

    generation_ids = _require_int_list(
        message.get("generation_token_ids"),
        f"{choice_context}.message.generation_token_ids",
    )
    generation_logprobs = _require_float_list(
        message.get("generation_log_probs"),
        f"{choice_context}.message.generation_log_probs",
    )
    if not generation_ids:
        raise VllmReplayConfigurationError(f"{context} has an empty completion")
    if len(generation_ids) != len(generation_logprobs):
        raise VllmReplayConfigurationError(
            f"{context} generation token/logprob lengths differ"
        )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise VllmReplayConfigurationError(
            f"{choice_context}.logprobs must be a JSON object"
        )
    content = logprobs.get("content")
    if not isinstance(content, list) or len(content) != len(generation_ids):
        raise VllmReplayConfigurationError(
            f"{choice_context}.logprobs.content must have {len(generation_ids)} entries"
        )
    content_ids: list[int] = []
    content_logprobs: list[float] = []
    for item_index, item in enumerate(content):
        item_context = f"{choice_context}.logprobs.content[{item_index}]"
        if not isinstance(item, dict):
            raise VllmReplayConfigurationError(f"{item_context} must be a JSON object")
        token = item.get("token")
        match = _TOKEN_ID_PATTERN.fullmatch(token) if isinstance(token, str) else None
        if match is None:
            raise VllmReplayConfigurationError(
                f"{item_context}.token must use exact token_id:<integer> form"
            )
        content_ids.append(int(match.group("token_id")))
        content_logprobs.append(
            _require_float(item.get("logprob"), f"{item_context}.logprob")
        )
        byte_values = item.get("bytes")
        if byte_values is not None:
            parsed_bytes = _require_int_list(byte_values, f"{item_context}.bytes")
            if any(value > 255 for value in parsed_bytes):
                raise VllmReplayConfigurationError(
                    f"{item_context}.bytes contains a value > 255"
                )
        if item.get("top_logprobs") != []:
            raise VllmReplayConfigurationError(
                f"{item_context}.top_logprobs must be empty"
            )

    if tuple(content_ids) != generation_ids:
        raise VllmReplayConfigurationError(
            f"{context} generation token IDs disagree between message and "
            "logprobs.content"
        )
    if tuple(content_logprobs) != generation_logprobs:
        raise VllmReplayConfigurationError(
            f"{context} generation logprobs disagree between message and "
            "logprobs.content"
        )

    for owner_context, owner in (
        (choice_context, choice),
        (f"{choice_context}.message", message),
    ):
        optional_ids = owner.get("token_ids")
        if (
            optional_ids is not None
            and _require_int_list(optional_ids, f"{owner_context}.token_ids")
            != generation_ids
        ):
            raise VllmReplayConfigurationError(
                f"{owner_context}.token_ids disagrees with generation_token_ids"
            )

    prompt_ids = _require_int_list(
        message.get("prompt_token_ids"), f"{choice_context}.message.prompt_token_ids"
    )
    if not prompt_ids:
        raise VllmReplayConfigurationError(f"{context} has an empty prompt token list")
    top_prompt_ids = response.get("prompt_token_ids")
    if (
        top_prompt_ids is not None
        and _require_int_list(top_prompt_ids, f"{context}.prompt_token_ids")
        != prompt_ids
    ):
        raise VllmReplayConfigurationError(
            f"{context}.prompt_token_ids disagrees with message.prompt_token_ids"
        )
    if response.get("prompt_logprobs") is not None:
        raise VllmReplayConfigurationError(f"{context}.prompt_logprobs must be null")

    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise VllmReplayConfigurationError(f"{context}.usage must be a JSON object")
    completion_tokens = _require_int(
        usage.get("completion_tokens"), f"{context}.usage.completion_tokens"
    )
    prompt_tokens = _require_int(
        usage.get("prompt_tokens"), f"{context}.usage.prompt_tokens"
    )
    total_tokens = _require_int(
        usage.get("total_tokens"), f"{context}.usage.total_tokens"
    )
    if completion_tokens != len(generation_ids) or prompt_tokens != len(prompt_ids):
        raise VllmReplayConfigurationError(
            f"{context}.usage token counts disagree with transported token arrays"
        )
    if total_tokens != prompt_tokens + completion_tokens:
        raise VllmReplayConfigurationError(
            f"{context}.usage.total_tokens is inconsistent"
        )


def _file_fingerprint(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_immutable_bundle(path: Path) -> bytes:
    if not path.is_absolute():
        raise VllmReplayConfigurationError(
            f"{REPLAY_BUNDLE_ENV} must name an absolute path"
        )
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise VllmReplayConfigurationError(
            "replay bundle cannot be inspected"
        ) from error
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise VllmReplayConfigurationError(
            "replay bundle must be a regular non-symlink file"
        )
    if path_stat.st_nlink != 1:
        raise VllmReplayConfigurationError(
            "replay bundle must have exactly one filesystem link"
        )
    if path_stat.st_mode & 0o222:
        raise VllmReplayConfigurationError(
            "replay bundle must have no filesystem write bits"
        )
    if path_stat.st_size <= 0 or path_stat.st_size > _MAX_BUNDLE_BYTES:
        raise VllmReplayConfigurationError(
            f"replay bundle size must be in [1, {_MAX_BUNDLE_BYTES}] bytes"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VllmReplayConfigurationError("replay bundle cannot be opened") from error
    try:
        opened_stat = os.fstat(descriptor)
        if _file_fingerprint(opened_stat) != _file_fingerprint(path_stat):
            raise VllmReplayConfigurationError(
                "replay bundle changed while it was being opened"
            )
        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            read_size = min(1024 * 1024, _MAX_BUNDLE_BYTES + 1 - bytes_read)
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _MAX_BUNDLE_BYTES:
                raise VllmReplayConfigurationError(
                    f"replay bundle exceeds {_MAX_BUNDLE_BYTES} bytes"
                )
        final_stat = os.fstat(descriptor)
    except OSError as error:
        raise VllmReplayConfigurationError("replay bundle cannot be read") from error
    finally:
        os.close(descriptor)

    if _file_fingerprint(final_stat) != _file_fingerprint(opened_stat):
        raise VllmReplayConfigurationError(
            "replay bundle changed while it was being read"
        )
    raw = b"".join(chunks)
    if len(raw) != final_stat.st_size:
        raise VllmReplayConfigurationError(
            "replay bundle size changed while it was being read"
        )
    return raw


def _default_marker_sink(marker: str) -> None:
    print(marker, flush=True)


class VllmReplayBundle:
    """An immutable, content-addressed, single-use captured response cohort."""

    def __init__(
        self,
        *,
        bundle_sha256: str,
        entries: dict[str, _ReplayEntry],
        required: bool = False,
        marker_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.bundle_sha256 = bundle_sha256
        self.required = required
        self._entries = entries
        self._consumed: set[str] = set()
        self._hits = 0
        self._misses = 0
        self._reuses = 0
        self._streaming_rejections = 0
        self._lock = threading.Lock()
        self._marker_sink = marker_sink or _default_marker_sink

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        required: bool = False,
        configured_sha256: str | None = None,
        marker_sink: Callable[[str], None] | None = None,
    ) -> Self | None:
        """Load the configured bundle once, or return ``None`` for live mode."""
        if type(required) is not bool:
            raise VllmReplayConfigurationError("replay required flag must be boolean")
        if configured_sha256 is not None:
            configured_sha256 = _require_sha256(
                configured_sha256, "configured replay bundle SHA256"
            )
        if required and configured_sha256 is None:
            raise VllmReplayConfigurationError(
                "configured replay bundle SHA256 is required for required replay"
            )
        if not required and configured_sha256 is not None:
            raise VllmReplayConfigurationError(
                "configured replay bundle SHA256 requires replay required mode"
            )

        environment = os.environ if environ is None else environ
        bundle_path = environment.get(REPLAY_BUNDLE_ENV)
        environment_sha256 = environment.get(REPLAY_BUNDLE_SHA256_ENV)
        if not required:
            if bundle_path or environment_sha256:
                raise VllmReplayConfigurationError(
                    "replay bundle environment variables require "
                    "vllm_cfg.replay_required=true"
                )
            return None
        if not bundle_path:
            raise VllmReplayConfigurationError(
                f"{REPLAY_BUNDLE_ENV} is required by vllm_cfg.replay_required"
            )
        if environment_sha256 and environment_sha256 != configured_sha256:
            raise VllmReplayConfigurationError(
                f"{REPLAY_BUNDLE_SHA256_ENV} does not match "
                "vllm_cfg.replay_bundle_sha256"
            )
        expected_sha256 = configured_sha256
        assert expected_sha256 is not None
        return cls.from_path(
            Path(bundle_path),
            expected_sha256,
            required=required,
            marker_sink=marker_sink,
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        expected_sha256: str,
        *,
        required: bool = False,
        marker_sink: Callable[[str], None] | None = None,
    ) -> Self:
        """Load and fully validate an immutable replay bundle."""
        expected_sha256 = _require_sha256(expected_sha256, "bundle SHA256 pin")
        raw = _read_immutable_bundle(path)
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise VllmReplayConfigurationError(
                "replay bundle SHA256 does not match its configured pin"
            )

        decoded = _decode_bundle_json(raw)
        if not isinstance(decoded, dict):
            raise VllmReplayConfigurationError(
                "replay bundle top level must be a JSON object"
            )
        if raw != canonical_json_bytes(decoded):
            raise VllmReplayConfigurationError(
                "replay bundle file must use canonical NeMo Gym JSON encoding"
            )
        _require_exact_keys(decoded, {"schema", "entries"}, "replay bundle")
        if decoded["schema"] != REPLAY_BUNDLE_SCHEMA:
            raise VllmReplayConfigurationError("unsupported replay bundle schema")
        raw_entries = decoded["entries"]
        if (
            not isinstance(raw_entries, list)
            or len(raw_entries) != STRICT_REPLAY_ENTRIES
        ):
            raise VllmReplayConfigurationError(
                f"replay bundle must contain exactly K={STRICT_REPLAY_ENTRIES} entries"
            )

        entries: dict[str, _ReplayEntry] = {}
        for entry_index, raw_entry in enumerate(raw_entries):
            context = f"replay bundle entries[{entry_index}]"
            if not isinstance(raw_entry, dict):
                raise VllmReplayConfigurationError(f"{context} must be a JSON object")
            _require_exact_keys(
                raw_entry,
                {
                    "seed",
                    "request_payload",
                    "request_sha256",
                    "raw_response",
                    "response_sha256",
                },
                context,
            )
            seed = _require_int(raw_entry["seed"], f"{context}.seed")
            if seed != entry_index:
                raise VllmReplayConfigurationError(
                    "replay bundle entries must be ordered by seeds 0..3; "
                    f"index {entry_index} has seed {seed}"
                )
            request = raw_entry["request_payload"]
            response = raw_entry["raw_response"]
            if not isinstance(request, dict):
                raise VllmReplayConfigurationError(
                    f"{context}.request_payload must be a JSON object"
                )
            _validate_request_payload(
                request, f"{context}.request_payload", expected_seed=seed
            )
            if not isinstance(response, dict):
                raise VllmReplayConfigurationError(
                    f"{context}.raw_response must be a JSON object"
                )
            request_sha256 = _require_sha256(
                raw_entry["request_sha256"], f"{context}.request_sha256"
            )
            response_sha256 = _require_sha256(
                raw_entry["response_sha256"], f"{context}.response_sha256"
            )
            if sha256_json(request) != request_sha256:
                raise VllmReplayConfigurationError(
                    f"{context} request SHA256 does not match request_payload"
                )
            if sha256_json(response) != response_sha256:
                raise VllmReplayConfigurationError(
                    f"{context} response SHA256 does not match raw_response"
                )
            _validate_chat_completion_response(response, f"{context}.raw_response")
            if request_sha256 in entries:
                raise VllmReplayConfigurationError(
                    "replay bundle contains a duplicate request SHA256"
                )
            entries[request_sha256] = _ReplayEntry(
                seed=seed,
                request_sha256=request_sha256,
                response=response,
                response_sha256=response_sha256,
            )
        return cls(
            bundle_sha256=actual_sha256,
            entries=entries,
            required=required,
            marker_sink=marker_sink,
        )

    def replay_json_bytes(self, raw_request: bytes, *, streaming: bool) -> JsonObject:
        """Return the request's one recorded response without a live fallback."""
        try:
            request = _decode_request_json(raw_request)
            request_sha256 = sha256_json(request)
        except ValueError as error:
            raise VllmReplayInvalidRequestError(
                "replay request body is not strict JSON"
            ) from error
        seed_value = request.get("seed")
        seed = seed_value if type(seed_value) is int and seed_value >= 0 else None

        with self._lock:
            if streaming:
                self._streaming_rejections += 1
                self._emit_marker_locked(
                    event="streaming_rejected",
                    request_sha256=request_sha256,
                    response_sha256=None,
                    seed=seed,
                )
                raise VllmReplayStreamingError(
                    "captured-cohort replay does not support streaming requests"
                )

            entry = self._entries.get(request_sha256)
            if entry is None:
                self._misses += 1
                self._emit_marker_locked(
                    event="miss",
                    request_sha256=request_sha256,
                    response_sha256=None,
                    seed=seed,
                )
                raise VllmReplayMissError(
                    f"captured-cohort replay miss for request_sha256={request_sha256}"
                )
            if request_sha256 in self._consumed:
                self._reuses += 1
                self._emit_marker_locked(
                    event="reuse",
                    request_sha256=request_sha256,
                    response_sha256=entry.response_sha256,
                    seed=entry.seed,
                )
                raise VllmReplayReuseError(
                    "captured-cohort response already used for "
                    f"request_sha256={request_sha256}"
                )

            self._consumed.add(request_sha256)
            self._hits += 1
            self._emit_marker_locked(
                event="hit",
                request_sha256=request_sha256,
                response_sha256=entry.response_sha256,
                seed=entry.seed,
            )
            return copy.deepcopy(entry.response)

    def state_summary(self) -> JsonObject:
        """Return a secret-free receipt of replay progress."""
        with self._lock:
            entries = len(self._entries)
            hits = self._hits
            return {
                "bundle_sha256": self.bundle_sha256,
                "enabled": True,
                "entries": entries,
                "hits": hits,
                "misses": self._misses,
                "pending": entries - hits,
                "required": self.required,
                "reuses": self._reuses,
                "schema": REPLAY_BUNDLE_SCHEMA,
                "streaming_rejections": self._streaming_rejections,
            }

    def _emit_marker_locked(
        self,
        *,
        event: str,
        request_sha256: str,
        response_sha256: str | None,
        seed: int | None,
    ) -> None:
        marker: JsonObject = {
            "bundle_sha256": self.bundle_sha256,
            "event": event,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "seed": seed,
        }
        self._marker_sink(
            REPLAY_MARKER_PREFIX + canonical_json_bytes(marker).decode("utf-8")
        )


def load_vllm_replay_from_config(
    vllm_cfg: Mapping[str, object],
    *,
    data_parallel_size: int,
    actor_identity: str,
    environ: Mapping[str, str] | None = None,
    marker_sink: Callable[[str], None] | None = None,
) -> VllmReplayBundle | None:
    """Load replay using independent config intent and attest its topology."""
    if not isinstance(actor_identity, str) or not actor_identity:
        raise VllmReplayConfigurationError(
            "replay actor identity must be a nonempty string"
        )
    contract = validate_vllm_replay_config(
        vllm_cfg, data_parallel_size=data_parallel_size
    )
    replay = VllmReplayBundle.from_environment(
        environ,
        required=contract.required,
        configured_sha256=contract.bundle_sha256,
        marker_sink=marker_sink,
    )
    if replay is None:
        return None

    parsed_data_parallel_size = _require_data_parallel_size(
        data_parallel_size, "replay data-parallel size"
    )
    if parsed_data_parallel_size != 1:
        raise VllmReplayConfigurationError(
            "captured-cohort replay supports only data-parallel size 1 because "
            "response claims are actor-local"
        )

    state = replay.state_summary()
    ready_marker: JsonObject = {
        "actor_identity": actor_identity,
        "bundle_sha256": replay.bundle_sha256,
        "data_parallel_size": parsed_data_parallel_size,
        "entries": state["entries"],
        "loaded": True,
        "required": replay.required,
    }
    sink = marker_sink or _default_marker_sink
    sink(
        REPLAY_READY_MARKER_PREFIX + canonical_json_bytes(ready_marker).decode("utf-8")
    )
    return replay
