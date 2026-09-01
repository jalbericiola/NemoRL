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

"""CPU tests for captured-cohort replay at the vLLM HTTP boundary."""

import copy
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nemo_rl.utils.vllm_replay_bundle import (
    REPLAY_BUNDLE_ENV,
    REPLAY_BUNDLE_SCHEMA,
    REPLAY_BUNDLE_SHA256_ENV,
    REPLAY_MARKER_PREFIX,
    REPLAY_READY_MARKER_PREFIX,
    STRICT_REPLAY_ENTRIES,
    VllmReplayBundle,
    VllmReplayConfigurationError,
    VllmReplayInvalidRequestError,
    VllmReplayMissError,
    VllmReplayReuseError,
    VllmReplayStreamingError,
    canonical_json_bytes,
    load_vllm_replay_from_config,
    sha256_json,
    validate_vllm_replay_config,
    validate_vllm_replay_environment_contract,
    validate_vllm_replay_worker_topology,
)


def _request(seed: int = 0) -> dict:
    return {
        "messages": [{"role": "user", "content": "café λ"}],
        "logprobs": True,
        "max_tokens": 8,
        "model": "model",
        "return_tokens_as_token_ids": True,
        "seed": seed,
        "stream": False,
        "temperature": 1.0,
        "top_logprobs": 0,
        "top_p": 1.0,
    }


def _response(seed: int = 0) -> dict:
    prompt_ids = [100, 101]
    generation_ids = [200 + seed, 300 + seed]
    generation_logprobs = [-0.1, -0.2]
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": {
                    "content": [
                        {
                            "bytes": [65 + index],
                            "logprob": generation_logprobs[index],
                            "token": f"token_id:{token_id}",
                            "top_logprobs": [],
                        }
                        for index, token_id in enumerate(generation_ids)
                    ]
                },
                "message": {
                    "content": f"answer-{seed}",
                    "generation_log_probs": generation_logprobs,
                    "generation_token_ids": generation_ids,
                    "prompt_token_ids": prompt_ids,
                    "role": "assistant",
                    "token_ids": generation_ids,
                },
                "token_ids": generation_ids,
            }
        ],
        "created": 1,
        "id": f"chatcmpl-{seed}",
        "model": "model",
        "object": "chat.completion",
        "prompt_logprobs": None,
        "prompt_token_ids": prompt_ids,
        "usage": {
            "completion_tokens": len(generation_ids),
            "prompt_tokens": len(prompt_ids),
            "total_tokens": len(generation_ids) + len(prompt_ids),
        },
    }


def _entry(seed: int = 0) -> dict:
    request = _request(seed)
    response = _response(seed)
    return {
        "raw_response": response,
        "request_payload": request,
        "request_sha256": sha256_json(request),
        "response_sha256": sha256_json(response),
        "seed": seed,
    }


def _bundle(*entries: dict) -> dict:
    return {
        "entries": (
            list(entries)
            if entries
            else [_entry(seed) for seed in range(STRICT_REPLAY_ENTRIES)]
        ),
        "schema": REPLAY_BUNDLE_SCHEMA,
    }


def _bundle_with_entry(seed: int, entry: dict) -> dict:
    entries = [_entry(index) for index in range(STRICT_REPLAY_ENTRIES)]
    entries[seed] = entry
    return _bundle(*entries)


def _write_readonly_bundle(
    tmp_path: Path,
    bundle: dict | None = None,
    *,
    raw: bytes | None = None,
    filename: str = "replay.json",
) -> tuple[Path, str]:
    path = tmp_path / filename
    content = raw if raw is not None else canonical_json_bytes(bundle or _bundle())
    path.write_bytes(content)
    path.chmod(0o444)
    return path, hashlib.sha256(content).hexdigest()


def _load(
    tmp_path: Path,
    bundle: dict | None = None,
    *,
    markers: list[str] | None = None,
) -> tuple[VllmReplayBundle, Path, str]:
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, bundle)
    replay = VllmReplayBundle.from_path(
        path,
        bundle_sha256,
        marker_sink=None if markers is None else markers.append,
    )
    return replay, path, bundle_sha256


def _parse_marker(marker: str) -> dict:
    assert marker.startswith(REPLAY_MARKER_PREFIX)
    return json.loads(marker.removeprefix(REPLAY_MARKER_PREFIX))


def _required_config(bundle_sha256: str) -> dict:
    return {
        "async_engine": True,
        "expert_parallel_size": 1,
        "expose_http_server": True,
        "replay_bundle_sha256": bundle_sha256,
        "replay_required": True,
        "tensor_parallel_size": 4,
    }


def test_default_off_keeps_live_mode() -> None:
    assert VllmReplayBundle.from_environment({}) is None
    assert VllmReplayBundle.from_environment({REPLAY_BUNDLE_ENV: ""}) is None


def test_sha_pin_without_bundle_is_a_configuration_error() -> None:
    with pytest.raises(VllmReplayConfigurationError, match="replay_required=true"):
        VllmReplayBundle.from_environment({REPLAY_BUNDLE_SHA256_ENV: "0" * 64})


def test_environment_loads_exact_pinned_bundle(tmp_path: Path) -> None:
    path, bundle_sha256 = _write_readonly_bundle(tmp_path)
    replay = VllmReplayBundle.from_environment(
        {
            REPLAY_BUNDLE_ENV: os.fspath(path),
            REPLAY_BUNDLE_SHA256_ENV: bundle_sha256,
        },
        required=True,
        configured_sha256=bundle_sha256,
        marker_sink=lambda _marker: None,
    )
    assert replay is not None
    assert replay.state_summary()["bundle_sha256"] == bundle_sha256


def test_hit_hashes_noncanonical_raw_json_like_gym(tmp_path: Path) -> None:
    markers: list[str] = []
    replay, _, bundle_sha256 = _load(tmp_path, markers=markers)
    request = _request()
    raw_request = json.dumps(request, ensure_ascii=False, indent=2).encode("utf-8")

    response = replay.replay_json_bytes(raw_request, streaming=False)

    assert response == _response()
    assert response is not _response()
    assert len(markers) == 1
    assert _parse_marker(markers[0]) == {
        "bundle_sha256": bundle_sha256,
        "event": "hit",
        "request_sha256": sha256_json(request),
        "response_sha256": sha256_json(_response()),
        "seed": 0,
    }
    assert replay.state_summary() == {
        "bundle_sha256": bundle_sha256,
        "enabled": True,
        "entries": 4,
        "hits": 1,
        "misses": 0,
        "pending": 3,
        "required": False,
        "reuses": 0,
        "schema": REPLAY_BUNDLE_SCHEMA,
        "streaming_rejections": 0,
    }


def test_miss_fails_closed_without_leaking_request(tmp_path: Path) -> None:
    markers: list[str] = []
    replay, _, bundle_sha256 = _load(tmp_path, markers=markers)
    request = _request(seed=9)
    request["messages"][0]["content"] = "do-not-log-this-secret"

    with pytest.raises(VllmReplayMissError) as error:
        replay.replay_json_bytes(canonical_json_bytes(request), streaming=False)

    marker = markers[0]
    assert "do-not-log-this-secret" not in marker
    assert "do-not-log-this-secret" not in str(error.value)
    assert _parse_marker(marker) == {
        "bundle_sha256": bundle_sha256,
        "event": "miss",
        "request_sha256": sha256_json(request),
        "response_sha256": None,
        "seed": 9,
    }
    assert replay.state_summary()["misses"] == 1
    assert replay.state_summary()["pending"] == 4


def test_response_is_single_use(tmp_path: Path) -> None:
    markers: list[str] = []
    replay, _, _ = _load(tmp_path, markers=markers)
    raw_request = canonical_json_bytes(_request())
    replay.replay_json_bytes(raw_request, streaming=False)

    with pytest.raises(VllmReplayReuseError):
        replay.replay_json_bytes(raw_request, streaming=False)

    assert [_parse_marker(marker)["event"] for marker in markers] == ["hit", "reuse"]
    assert replay.state_summary()["hits"] == 1
    assert replay.state_summary()["reuses"] == 1


def test_duplicate_bundle_json_key_is_rejected(tmp_path: Path) -> None:
    raw = (
        b'{"entries":[],"schema":"nemorl-gym-captured-cohort-replay-v1",'
        b'"schema":"nemorl-gym-captured-cohort-replay-v1"}'
    )
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, raw=raw)

    with pytest.raises(VllmReplayConfigurationError, match="duplicate JSON"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_entries_must_be_exactly_ordered_seeds_zero_through_three(
    tmp_path: Path,
) -> None:
    entry = _entry()
    path, bundle_sha256 = _write_readonly_bundle(
        tmp_path,
        _bundle(entry, copy.deepcopy(entry), _entry(seed=2), _entry(seed=3)),
    )

    with pytest.raises(VllmReplayConfigurationError, match="ordered by seeds 0..3"):
        VllmReplayBundle.from_path(path, bundle_sha256)


@pytest.mark.parametrize("entry_count", [0, 1, 3, 5])
def test_bundle_requires_exactly_four_entries(tmp_path: Path, entry_count: int) -> None:
    bundle = {
        "entries": [_entry(seed) for seed in range(entry_count)],
        "schema": REPLAY_BUNDLE_SCHEMA,
    }
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, bundle)

    with pytest.raises(VllmReplayConfigurationError, match="exactly K=4"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_duplicate_raw_request_key_is_rejected(tmp_path: Path) -> None:
    replay, _, _ = _load(tmp_path, markers=[])

    with pytest.raises(VllmReplayInvalidRequestError, match="duplicate JSON"):
        replay.replay_json_bytes(b'{"seed":0,"seed":0}', streaming=False)


def test_bundle_file_digest_tamper_is_rejected(tmp_path: Path) -> None:
    path, original_sha256 = _write_readonly_bundle(tmp_path)
    path.chmod(0o644)
    tampered = _bundle()
    tampered["entries"][0]["raw_response"]["id"] = "tampered"
    path.write_bytes(canonical_json_bytes(tampered))
    path.chmod(0o444)

    with pytest.raises(VllmReplayConfigurationError, match="configured pin"):
        VllmReplayBundle.from_path(path, original_sha256)


@pytest.mark.parametrize("digest_field", ["request_sha256", "response_sha256"])
def test_per_entry_digest_tamper_is_rejected(tmp_path: Path, digest_field: str) -> None:
    entry = _entry()
    entry[digest_field] = "0" * 64
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, _bundle_with_entry(0, entry))

    with pytest.raises(VllmReplayConfigurationError, match="SHA256 does not match"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_invalid_response_schema_is_rejected(tmp_path: Path) -> None:
    entry = _entry()
    del entry["raw_response"]["choices"]
    entry["response_sha256"] = sha256_json(entry["raw_response"])
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, _bundle_with_entry(0, entry))

    with pytest.raises(VllmReplayConfigurationError, match="ChatCompletion"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_noncanonical_bundle_encoding_is_rejected(tmp_path: Path) -> None:
    raw = json.dumps(_bundle(), indent=2, ensure_ascii=False).encode("utf-8")
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, raw=raw)

    with pytest.raises(VllmReplayConfigurationError, match="canonical"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_writable_bundle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "writable.json"
    raw = canonical_json_bytes(_bundle())
    path.write_bytes(raw)
    path.chmod(0o644)

    with pytest.raises(VllmReplayConfigurationError, match="write bits"):
        VllmReplayBundle.from_path(path, hashlib.sha256(raw).hexdigest())


def test_symlink_bundle_is_rejected(tmp_path: Path) -> None:
    target, bundle_sha256 = _write_readonly_bundle(tmp_path, filename="target.json")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(VllmReplayConfigurationError, match="non-symlink"):
        VllmReplayBundle.from_path(link, bundle_sha256)


def test_nonregular_bundle_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "bundle-dir"
    directory.mkdir()
    directory.chmod(0o555)

    with pytest.raises(VllmReplayConfigurationError, match="regular non-symlink"):
        VllmReplayBundle.from_path(directory, "0" * 64)


def test_hardlinked_bundle_is_rejected(tmp_path: Path) -> None:
    target, bundle_sha256 = _write_readonly_bundle(tmp_path, filename="target.json")
    link = tmp_path / "hardlink.json"
    os.link(target, link)

    with pytest.raises(VllmReplayConfigurationError, match="one filesystem link"):
        VllmReplayBundle.from_path(target, bundle_sha256)


def test_streaming_rejection_does_not_consume_response(tmp_path: Path) -> None:
    markers: list[str] = []
    replay, _, _ = _load(tmp_path, markers=markers)
    raw_request = canonical_json_bytes(_request())

    with pytest.raises(VllmReplayStreamingError):
        replay.replay_json_bytes(raw_request, streaming=True)
    assert replay.state_summary()["pending"] == 4

    assert replay.replay_json_bytes(raw_request, streaming=False) == _response()
    assert [_parse_marker(marker)["event"] for marker in markers] == [
        "streaming_rejected",
        "hit",
    ]


def test_loaded_bundle_is_an_in_memory_snapshot(tmp_path: Path) -> None:
    replay, path, _ = _load(tmp_path, markers=[])
    path.chmod(0o644)
    replacement = _entry(seed=0)
    replacement["raw_response"]["id"] = "changed-after-load"
    replacement["response_sha256"] = sha256_json(replacement["raw_response"])
    path.write_bytes(canonical_json_bytes(_bundle_with_entry(0, replacement)))
    path.chmod(0o444)

    assert replay.replay_json_bytes(
        canonical_json_bytes(_request(seed=0)), streaming=False
    ) == _response(seed=0)


def test_concurrent_claim_has_exactly_one_hit(tmp_path: Path) -> None:
    markers: list[str] = []
    replay, _, _ = _load(tmp_path, markers=markers)
    raw_request = canonical_json_bytes(_request())

    def claim() -> str:
        try:
            replay.replay_json_bytes(raw_request, streaming=False)
        except VllmReplayReuseError:
            return "reuse"
        return "hit"

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(lambda _index: claim(), range(32)))

    assert outcomes.count("hit") == 1
    assert outcomes.count("reuse") == 31
    assert replay.state_summary()["hits"] == 1
    assert replay.state_summary()["reuses"] == 31
    assert [_parse_marker(marker)["event"] for marker in markers].count("hit") == 1


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        canonical_json_bytes({"value": float("nan")})


def test_request_capture_contract_is_enforced(tmp_path: Path) -> None:
    entry = _entry()
    entry["request_payload"]["logprobs"] = False
    entry["request_sha256"] = sha256_json(entry["request_payload"])
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, _bundle_with_entry(0, entry))

    with pytest.raises(VllmReplayConfigurationError, match="logprobs.*exactly true"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_response_token_and_logprob_alignment_is_enforced(tmp_path: Path) -> None:
    entry = _entry()
    entry["raw_response"]["choices"][0]["logprobs"]["content"][0]["token"] = (
        "token_id:999"
    )
    entry["response_sha256"] = sha256_json(entry["raw_response"])
    path, bundle_sha256 = _write_readonly_bundle(tmp_path, _bundle_with_entry(0, entry))

    with pytest.raises(VllmReplayConfigurationError, match="token IDs disagree"):
        VllmReplayBundle.from_path(path, bundle_sha256)


def test_required_config_fails_when_both_bundle_environment_values_are_lost() -> None:
    with pytest.raises(VllmReplayConfigurationError, match=REPLAY_BUNDLE_ENV):
        load_vllm_replay_from_config(
            _required_config("0" * 64),
            data_parallel_size=1,
            actor_identity="pid:123",
            environ={},
            marker_sink=lambda _marker: None,
        )


def test_required_config_digest_is_authoritative_and_emits_ready_marker(
    tmp_path: Path,
) -> None:
    path, bundle_sha256 = _write_readonly_bundle(tmp_path)
    markers: list[str] = []

    replay = load_vllm_replay_from_config(
        _required_config(bundle_sha256),
        data_parallel_size=1,
        actor_identity="pid:123",
        environ={REPLAY_BUNDLE_ENV: os.fspath(path)},
        marker_sink=markers.append,
    )

    assert replay is not None
    assert replay.required is True
    assert len(markers) == 1
    assert markers[0].startswith(REPLAY_READY_MARKER_PREFIX)
    assert json.loads(markers[0].removeprefix(REPLAY_READY_MARKER_PREFIX)) == {
        "actor_identity": "pid:123",
        "bundle_sha256": bundle_sha256,
        "data_parallel_size": 1,
        "entries": 4,
        "loaded": True,
        "required": True,
    }


def test_required_config_rejects_disagreeing_environment_digest(
    tmp_path: Path,
) -> None:
    path, bundle_sha256 = _write_readonly_bundle(tmp_path)
    with pytest.raises(VllmReplayConfigurationError, match="does not match"):
        load_vllm_replay_from_config(
            _required_config(bundle_sha256),
            data_parallel_size=1,
            actor_identity="pid:123",
            environ={
                REPLAY_BUNDLE_ENV: os.fspath(path),
                REPLAY_BUNDLE_SHA256_ENV: "0" * 64,
            },
            marker_sink=lambda _marker: None,
        )


def test_live_config_cannot_carry_a_replay_digest() -> None:
    with pytest.raises(VllmReplayConfigurationError, match="must be null or absent"):
        validate_vllm_replay_config(
            {"replay_required": False, "replay_bundle_sha256": "0" * 64}
        )


@pytest.mark.parametrize(
    "environment",
    [
        {REPLAY_BUNDLE_ENV: "/bundle.json"},
        {REPLAY_BUNDLE_SHA256_ENV: "0" * 64},
    ],
)
def test_environment_cannot_enable_replay_without_required_config(
    environment: dict[str, str],
) -> None:
    with pytest.raises(VllmReplayConfigurationError, match="replay_required=true"):
        validate_vllm_replay_environment_contract(
            {"replay_required": False}, environ=environment
        )


def test_forwarded_actor_environment_cannot_bypass_required_config() -> None:
    with pytest.raises(VllmReplayConfigurationError, match="replay_required=true"):
        validate_vllm_replay_environment_contract(
            {
                "env_vars": {REPLAY_BUNDLE_ENV: "/bundle.json"},
                "replay_required": False,
            },
            environ={},
        )


@pytest.mark.parametrize("required_field", ["async_engine", "expose_http_server"])
def test_required_replay_needs_async_http_worker(required_field: str) -> None:
    config = _required_config("0" * 64)
    config[required_field] = False

    with pytest.raises(VllmReplayConfigurationError, match=required_field):
        validate_vllm_replay_config(config)


def test_required_replay_rejects_outer_data_parallelism() -> None:
    with pytest.raises(VllmReplayConfigurationError, match="data-parallel size 1"):
        validate_vllm_replay_config(_required_config("0" * 64), data_parallel_size=2)


@pytest.mark.parametrize(
    ("vllm_kwargs", "environ"),
    [
        ({"data_parallel_size": 2}, {}),
        ({"data_parallel_size_local": 2}, {}),
        ({}, {"VLLM_DP_SIZE": "2"}),
    ],
)
def test_replay_rejects_every_worker_internal_data_parallel_signal(
    vllm_kwargs: dict, environ: dict[str, str]
) -> None:
    environment = {REPLAY_BUNDLE_ENV: "/not-opened", **environ}
    with pytest.raises(VllmReplayConfigurationError, match="requires every.*size.*1"):
        validate_vllm_replay_worker_topology(
            _required_config("0" * 64),
            vllm_kwargs,
            outer_data_parallel_size=1,
            environ=environment,
        )


def test_replay_rejects_expert_parallel_derived_vllm_dp() -> None:
    config = _required_config("0" * 64)
    config["tensor_parallel_size"] = 2
    config["expert_parallel_size"] = 4

    with pytest.raises(VllmReplayConfigurationError, match="expert-parallel-derived"):
        validate_vllm_replay_worker_topology(
            config,
            {},
            outer_data_parallel_size=1,
            environ={REPLAY_BUNDLE_ENV: "/not-opened"},
        )


def test_live_worker_topology_is_unchanged_when_replay_is_off() -> None:
    assert (
        validate_vllm_replay_worker_topology(
            {},
            {"data_parallel_size": 2},
            outer_data_parallel_size=2,
            environ={},
        )
        == 2
    )
