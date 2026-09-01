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

"""Chat-template kwargs must reach every consumer that renders a conversation.

The async HTTP server builds three objects that each render chat messages:
OnlineRenderer, OpenAIServingChat and ServingTokenization. They do not share
one copy -- the chat serving builds its reasoning parser from its own, and the
tokenize path passes its own into preprocess_chat -- so a value handed to only
one makes /tokenize render differently from /v1/chat/completions on the same
conversation.

These tests drive the real _setup_vllm_openai_api_server against a fake vLLM
module tree and inspect what each consumer was constructed with.
"""

import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_rl.models.generation.vllm.vllm_worker_async as vllm_worker_async
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.vllm_replay_bundle import (
    REPLAY_BUNDLE_ENV,
    REPLAY_BUNDLE_SHA256_ENV,
    VllmReplayConfigurationError,
    VllmReplayMissError,
)

# The server subclasses each of these (NeMoRLOnlineRenderer, and so on), so the
# recording list is bound explicitly rather than looked up through the instance.
# A class attribute would be shadowed by the subclass and the construction would
# be recorded somewhere the assertions never look.
_BUILT: dict[str, list] = {"renderer": [], "chat": [], "tokenize": []}


def _recorder(slot: str):
    class _Stub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _BUILT[slot].append(self)

    return _Stub


_OnlineRenderer = _recorder("renderer")
_OpenAIServingChat = _recorder("chat")
_ServingTokenization = _recorder("tokenize")


class _FakeApp:
    """Minimal FastAPI stand-in: the server only registers routes on it."""

    def __init__(self):
        self.routes = []

    def _register(self, path):
        def decorator(fn):
            self.routes.append((path, fn))
            return fn

        return decorator

    def post(self, path, **_kwargs):
        return self._register(path)

    def get(self, path, **_kwargs):
        return self._register(path)


def _install_fake_vllm(monkeypatch):
    """Stub exactly the vLLM surface _setup_vllm_openai_api_server imports."""
    for name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.engine",
        "vllm.entrypoints.openai.models",
        "vllm.entrypoints.serve",
        "vllm.entrypoints.serve.tokenize",
        "vllm.reasoning",
        "vllm.renderers",
        "vllm.tool_parsers",
        "vllm.v1",
        "vllm.v1.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    def placeholder(name):
        return type(name, (), {})

    module(
        "vllm.entrypoints.chat_utils", load_chat_template=MagicMock(return_value=None)
    )
    module(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionRequest=placeholder("ChatCompletionRequest"),
        ChatCompletionResponse=placeholder("ChatCompletionResponse"),
    )
    module(
        "vllm.entrypoints.openai.chat_completion.serving",
        OpenAIServingChat=_OpenAIServingChat,
    )
    module(
        "vllm.entrypoints.openai.engine.protocol",
        ErrorResponse=placeholder("ErrorResponse"),
    )
    module(
        "vllm.entrypoints.openai.models.protocol",
        BaseModelPath=lambda **kwargs: kwargs,
    )
    module(
        "vllm.entrypoints.openai.models.serving",
        OpenAIServingModels=MagicMock(),
    )
    module(
        "vllm.entrypoints.serve.tokenize.protocol",
        TokenizeChatRequest=placeholder("TokenizeChatRequest"),
        TokenizeCompletionRequest=placeholder("TokenizeCompletionRequest"),
        TokenizeResponse=placeholder("TokenizeResponse"),
    )
    module(
        "vllm.entrypoints.serve.tokenize.serving",
        ServingTokenization=_ServingTokenization,
    )
    module("vllm.renderers.online_renderer", OnlineRenderer=_OnlineRenderer)
    module(
        "vllm.exceptions",
        VLLMValidationError=type("VLLMValidationError", (Exception,), {}),
    )
    module(
        "vllm.reasoning.abs_reasoning_parsers",
        ReasoningParserManager=type(
            "ReasoningParserManager", (), {"import_reasoning_parser": MagicMock()}
        ),
    )
    module(
        "vllm.tool_parsers.abstract_tool_parser",
        ToolParserManager=type(
            "ToolParserManager", (), {"import_tool_parser": MagicMock()}
        ),
    )
    module("vllm.v1.engine.async_llm", logger=MagicMock())

    for built in _BUILT.values():
        built.clear()


def _build_server_context(monkeypatch, serving_chat_kwargs, replay_bundle=None):
    """Run the real server setup and return its worker and registered app."""
    _install_fake_vllm(monkeypatch)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "vllm_cfg": {"http_server_serving_chat_kwargs": serving_chat_kwargs},
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )
    worker._vllm_replay_bundle = replay_bundle

    app = _FakeApp()
    worker._setup_vllm_openai_api_server(app)
    return worker, app


def _build_server(monkeypatch, serving_chat_kwargs):
    """Run the real server setup and hand back the three consumer stubs."""
    _build_server_context(monkeypatch, serving_chat_kwargs)
    return _BUILT["renderer"], _BUILT["chat"], _BUILT["tokenize"]


def _chat_handler(app):
    return dict(app.routes)["/v1/chat/completions"]


def _request(**overrides):
    values = {
        "stream": False,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _raw_request(payload=None):
    payload = payload or {"seed": 0, "temperature": 1.0, "top_p": 1.0}
    return SimpleNamespace(
        body=AsyncMock(
            return_value=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    )


@pytest.mark.parametrize(
    "serving_chat_kwargs",
    [
        {"default_chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ],
    ids=["native-name", "legacy-name"],
)
def test_both_spellings_reach_all_three_consumers(monkeypatch, serving_chat_kwargs):
    renderer, serving_chat, tokenization = _build_server(
        monkeypatch, dict(serving_chat_kwargs)
    )

    expected = {"enable_thinking": False}
    assert renderer[0].kwargs["default_chat_template_kwargs"] == expected
    assert serving_chat[0].kwargs["default_chat_template_kwargs"] == expected
    assert tokenization[0].kwargs["default_chat_template_kwargs"] == expected


def test_legacy_spelling_does_not_survive_into_serving_chat(monkeypatch):
    """The legacy key must be renamed, not merely read.

    The kwargs bag is splatted into OpenAIServingChat, which rejects an
    argument it does not declare, so leaving chat_template_kwargs behind is a
    TypeError at construction.
    """
    _, serving_chat, _ = _build_server(
        monkeypatch, {"chat_template_kwargs": {"enable_thinking": False}}
    )

    assert "chat_template_kwargs" not in serving_chat[0].kwargs


def test_native_spelling_wins_and_legacy_is_dropped(monkeypatch):
    """Both spellings present: native wins, legacy is removed.

    Reading these as ``pop(native) or pop(legacy)`` short-circuits on a truthy
    native value and leaves the legacy key in the bag.
    """
    _, serving_chat, _ = _build_server(
        monkeypatch,
        {
            "default_chat_template_kwargs": {"enable_thinking": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    chat_kwargs = serving_chat[0].kwargs
    assert chat_kwargs["default_chat_template_kwargs"] == {"enable_thinking": True}
    assert "chat_template_kwargs" not in chat_kwargs


def test_absent_kwargs_render_as_empty_dict(monkeypatch):
    """Neither spelling given: consumers get {} rather than None.

    preprocess_chat splats this, so None raises instead of letting the template
    apply its own defaults.
    """
    renderer, _, tokenization = _build_server(monkeypatch, {})

    assert renderer[0].kwargs["default_chat_template_kwargs"] == {}
    assert tokenization[0].kwargs["default_chat_template_kwargs"] == {}


def test_default_off_reaches_live_chat_engine_without_reading_body(monkeypatch):
    _, app = _build_server_context(monkeypatch, {})
    live_generator = MagicMock()
    live_generator.__aiter__.return_value = iter(())
    serving_chat = _BUILT["chat"][0]
    serving_chat.create_chat_completion = AsyncMock(return_value=live_generator)
    raw_request = _raw_request()

    asyncio.run(_chat_handler(app)(_request(), raw_request))

    serving_chat.create_chat_completion.assert_awaited_once()
    raw_request.body.assert_not_awaited()


def test_replay_hit_returns_before_live_chat_engine(monkeypatch):
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "recorded", "role": "assistant"},
            }
        ],
        "created": 1,
        "id": "chatcmpl-recorded",
        "model": "model",
        "object": "chat.completion",
    }
    replay_bundle = MagicMock()
    replay_bundle.replay_json_bytes.return_value = response
    _, app = _build_server_context(monkeypatch, {}, replay_bundle=replay_bundle)
    serving_chat = _BUILT["chat"][0]
    serving_chat.create_chat_completion = AsyncMock()
    raw_request = _raw_request()

    http_response = asyncio.run(_chat_handler(app)(_request(), raw_request))

    assert http_response.status_code == 200
    assert json.loads(http_response.body) == response
    replay_bundle.replay_json_bytes.assert_called_once_with(
        raw_request.body.return_value, streaming=False
    )
    serving_chat.create_chat_completion.assert_not_awaited()


def test_sampling_assertions_run_before_replay(monkeypatch):
    replay_bundle = MagicMock()
    _, app = _build_server_context(monkeypatch, {}, replay_bundle=replay_bundle)

    with pytest.raises(AssertionError, match="matches neither"):
        asyncio.run(
            _chat_handler(app)(
                _request(temperature=0.5),
                _raw_request({"seed": 0, "temperature": 0.5, "top_p": 1.0}),
            )
        )

    replay_bundle.replay_json_bytes.assert_not_called()


def test_replay_miss_is_http_error_without_live_fallback(monkeypatch):
    replay_bundle = MagicMock()
    replay_bundle.replay_json_bytes.side_effect = VllmReplayMissError(
        "captured-cohort replay miss for request_sha256=" + "0" * 64
    )
    _, app = _build_server_context(monkeypatch, {}, replay_bundle=replay_bundle)
    serving_chat = _BUILT["chat"][0]
    serving_chat.create_chat_completion = AsyncMock()

    http_response = asyncio.run(_chat_handler(app)(_request(), _raw_request()))

    assert http_response.status_code == 409
    assert json.loads(http_response.body)["error"]["code"] == "replay_miss"
    serving_chat.create_chat_completion.assert_not_awaited()


def _required_worker_config() -> dict:
    return {
        "_replay_data_parallel_size": 1,
        "vllm_cfg": {
            "async_engine": True,
            "expert_parallel_size": 1,
            "expose_http_server": True,
            "replay_bundle_sha256": "0" * 64,
            "replay_required": True,
            "tensor_parallel_size": 4,
        },
        "vllm_kwargs": {},
    }


def test_required_replay_startup_fails_before_base_worker_when_bundle_env_is_lost(
    monkeypatch,
):
    monkeypatch.delenv(REPLAY_BUNDLE_ENV, raising=False)
    monkeypatch.delenv(REPLAY_BUNDLE_SHA256_ENV, raising=False)
    base_init = MagicMock()
    monkeypatch.setattr(
        vllm_worker_async.BaseVllmGenerationWorker, "__init__", base_init
    )

    with pytest.raises(VllmReplayConfigurationError, match=REPLAY_BUNDLE_ENV):
        VllmAsyncGenerationWorkerImpl(
            _required_worker_config(), bundle_indices=[0, 1, 2, 3]
        )

    base_init.assert_not_called()


def test_required_replay_is_loaded_and_attested_before_base_worker(monkeypatch):
    monkeypatch.setenv(REPLAY_BUNDLE_ENV, "/immutable/replay.json")
    replay = MagicMock()
    load_replay = MagicMock(return_value=replay)
    monkeypatch.setattr(vllm_worker_async, "load_vllm_replay_from_config", load_replay)
    observed_replay_at_base_init = []

    def fake_base_init(worker, *_args, **_kwargs):
        observed_replay_at_base_init.append(worker._vllm_replay_bundle)
        worker.is_model_owner = True

    monkeypatch.setattr(
        vllm_worker_async.BaseVllmGenerationWorker, "__init__", fake_base_init
    )
    config = _required_worker_config()

    worker = VllmAsyncGenerationWorkerImpl(
        config, bundle_indices=[0, 1, 2, 3], defer_model_load=False
    )

    assert worker._vllm_replay_bundle is replay
    assert observed_replay_at_base_init == [replay]
    load_replay.assert_called_once()
    load_call = load_replay.call_args
    assert load_call.args == (config["vllm_cfg"],)
    assert load_call.kwargs["data_parallel_size"] == 1
    assert load_call.kwargs["actor_identity"].startswith("pid:")
