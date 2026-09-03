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

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
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
        self.middleware_handlers = []

    def _register(self, path):
        def decorator(fn):
            self.routes.append((path, fn))
            return fn

        return decorator

    def post(self, path, **_kwargs):
        return self._register(path)

    def get(self, path, **_kwargs):
        return self._register(path)

    def middleware(self, middleware_type):
        assert middleware_type == "http"

        def decorator(fn):
            self.middleware_handlers.append(fn)
            return fn

        return decorator


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


def _build_server(monkeypatch, serving_chat_kwargs):
    """Run the real server setup and hand back the three consumer stubs."""
    _install_fake_vllm(monkeypatch)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_cfg": {"http_server_serving_chat_kwargs": serving_chat_kwargs},
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )
    worker._strict_model_transport_capture = None
    worker._strict_model_transport_policy = None

    worker._setup_vllm_openai_api_server(_FakeApp())
    return _BUILT["renderer"], _BUILT["chat"], _BUILT["tokenize"]


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


def test_disabled_strict_transport_is_inert_and_needs_no_runtime_token(monkeypatch):
    _install_fake_vllm(monkeypatch)
    monkeypatch.delenv(
        "EXPECTED_STRICT_PAIR_MODEL_TRANSPORT_POLICY_SHA256", raising=False
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_cfg": {
            "http_server_serving_chat_kwargs": {},
            "strict_model_transport": "disabled",
        },
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )
    worker._strict_model_transport_capture = None
    worker._strict_model_transport_policy = None
    app = _FakeApp()

    worker._setup_vllm_openai_api_server(app)

    assert worker._strict_model_transport_capture is None
    assert worker._strict_model_transport_policy is None
    assert app.middleware_handlers == []


@pytest.mark.asyncio
async def test_strict_route_captures_exact_request_and_returned_response_bytes(
    monkeypatch,
):
    _install_fake_vllm(monkeypatch)

    class Capture:
        def __init__(self):
            self.unmatched_failures = []

        def begin_chat_call(self, **kwargs):
            self.begin = kwargs
            return "ticket"

        def record_success(self, ticket, **kwargs):
            self.success = (ticket, kwargs)

        def record_failure(self, ticket, **kwargs):
            self.failure = (ticket, kwargs)

        def record_unmatched_failure(self, **kwargs):
            self.unmatched_failures.append(kwargs)

        def guard_unlisted(self, **kwargs):
            raise AssertionError(f"unexpected unlisted call: {kwargs}")

    capture = Capture()
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "vllm_cfg": {"http_server_serving_chat_kwargs": {}},
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )
    worker._strict_model_transport_capture = capture
    worker._strict_model_transport_policy = {"policy_sha256": "a" * 64}
    app = _FakeApp()
    worker._setup_vllm_openai_api_server(app)
    chat = _BUILT["chat"][0]
    response_type = sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionResponse
    raw_generator = response_type()

    async def generate(_request, _raw_request):
        return raw_generator

    chat.create_chat_completion = generate
    response_payload = {"wire": "exact"}
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.model_dump_chat_response_with_dynamic_message_fields",
        lambda generator: response_payload,
    )
    raw_body = b'{"seed":123,"top_k":null}'

    class RawRequest:
        method = "POST"
        url = SimpleNamespace(path="/v1/chat/completions", query="")
        headers = {"content-type": "application/json"}

        async def body(self):
            return raw_body

    request = SimpleNamespace(
        top_k=None,
        top_p=1.0,
        temperature=1.0,
        seed=123,
    )
    route = dict(app.routes)["/v1/chat/completions"]

    response = await route(request, RawRequest())

    assert request.top_k == -1
    assert capture.begin["request_body"] == raw_body
    assert capture.begin["typed_seed"] == 123
    assert capture.success[0] == "ticket"
    assert capture.success[1]["response_body"] == response.body
    assert capture.success[1]["expected_response_payload"] == response_payload

    boundary = app.middleware_handlers[0]

    async def accepted_before_attestation(_raw_request):
        return SimpleNamespace(status_code=200)

    accepted_response = await boundary(RawRequest(), accepted_before_attestation)
    assert accepted_response.status_code == 200
    assert capture.unmatched_failures == []

    async def rejected_before_handler(_raw_request):
        return SimpleNamespace(status_code=422)

    rejected_response = await boundary(RawRequest(), rejected_before_handler)
    assert rejected_response.status_code == 422
    assert capture.unmatched_failures == [
        {"reason": "chat request returned non-success before attestation"}
    ]

    async def raised_before_handler(_raw_request):
        raise RuntimeError("validation middleware failed")

    with pytest.raises(RuntimeError, match="validation middleware failed"):
        await boundary(RawRequest(), raised_before_handler)
    assert capture.unmatched_failures[-1] == {
        "reason": "HTTP request failed before a successful capture"
    }

    unlisted_request = RawRequest()
    unlisted_request.url = SimpleNamespace(path="/tokenize", query="")
    with pytest.raises(AssertionError, match="unexpected unlisted call"):
        await boundary(unlisted_request, accepted_before_attestation)
