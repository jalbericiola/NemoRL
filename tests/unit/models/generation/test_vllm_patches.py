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

"""Guards for the vLLM source patches that had no coverage.

The two port patches ship their own suites. These cover the remaining two:

* ``_patch_vllm_tool_parser_namespace_tool`` is the most load-bearing patch in
  the repo -- it is the only thing that makes vLLM 0.25.1 importable against
  the pinned ``openai==2.6.1``. If upstream reorders that import block the
  patch logs a warning and returns, and every engine then dies on
  ``import vllm.tool_parsers``. So the anchor needs pinning.
* the ``VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`` merge replaced the old
  ``ADDITIONAL_ENV_VARS`` file patch and is what now carries
  ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` and every user ``extra_env_vars`` to the
  Ray workers. Being additive rather than clobbering is the whole point of the
  rewrite, and it is pure string handling, so it is cheap to pin.
"""

import ast
import builtins
import logging
import os
import sys
import types

import pytest

from nemo_rl.models.generation.vllm import patches
from tests.unit.models.generation.vllm_patch_source_utils import (
    patch_snippets,
    write_unpatched_copy,
)

_TOOL_PARSER_SOURCE = "tool_parsers/utils.py"
_PATCH_FN = "_patch_vllm_tool_parser_namespace_tool"
_MARKER = "except ImportError:  # openai < 2.25.0 predates namespace tools"
_RADIO_SOURCE = "model_executor/models/radio.py"
_RADIO_PATCH_FN = "_patch_vllm_radio_layerscale_loader"
_RADIO_MARKER = "initializer_factor = self.config.initializer_factor"


@pytest.fixture
def patched_tool_parser_source(tmp_path, monkeypatch):
    """The installed tool_parsers/utils.py, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_TOOL_PARSER_SOURCE, _PATCH_FN, tmp_path / "utils.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    assert patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    return copied


@pytest.fixture
def patched_radio_source(tmp_path, monkeypatch):
    """The installed vLLM RADIO loader, unpatched then patched in tmp."""
    copied = write_unpatched_copy(_RADIO_SOURCE, _RADIO_PATCH_FN, tmp_path / "radio.py")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(copied))
    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))
    return copied


@pytest.mark.vllm
def test_namespace_tool_patch_anchor_still_matches_installed_vllm(
    patched_tool_parser_source,
):
    """A source edit becomes a silent no-op if upstream reorders the import."""
    content = patched_tool_parser_source.read_text()
    assert _MARKER in content, (
        "the NamespaceTool compat patch did not apply to the installed vLLM; "
        "its anchor import block has probably changed upstream. Every vLLM "
        "engine will fail to import tool_parsers against the pinned openai."
    )
    ast.parse(content)  # the edit must leave valid Python


@pytest.mark.vllm
def test_namespace_tool_patch_is_idempotent(patched_tool_parser_source, monkeypatch):
    """Every worker on a node runs the patch against the same file."""
    before = patched_tool_parser_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_tool_parser_source)
    )
    assert patches._patch_vllm_tool_parser_namespace_tool(logging.getLogger(__name__))
    assert patched_tool_parser_source.read_text() == before


def test_namespace_tool_patch_reports_unknown_source(monkeypatch, tmp_path, caplog):
    """A moved source anchor must be a hard result, not a warning-only no-op."""
    tool_parser_source = tmp_path / "utils.py"
    tool_parser_source.write_text("from openai.types.responses import FunctionTool\n")
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(tool_parser_source)
    )

    with caplog.at_level(logging.ERROR):
        applied = patches._patch_vllm_tool_parser_namespace_tool(
            logging.getLogger(__name__)
        )

    assert applied is False
    assert "expected import block not found" in caplog.text


def test_namespace_tool_compat_fails_closed(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm.tool_parsers.utils", raising=False)
    monkeypatch.setattr(patches, "_namespace_tool_compat_installed_path", None)
    monkeypatch.setattr(
        patches,
        "version",
        lambda _distribution: patches._NAMESPACE_TOOL_COMPAT_VLLM_VERSION,
    )
    monkeypatch.setattr(
        patches, "_patch_vllm_tool_parser_namespace_tool", lambda _logger: False
    )

    with pytest.raises(RuntimeError, match="before importing vLLM"):
        patches._ensure_vllm_tool_parser_namespace_tool()


def test_namespace_tool_compat_rejects_wrong_vllm_version(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm.tool_parsers.utils", raising=False)
    monkeypatch.setattr(patches, "_namespace_tool_compat_installed_path", None)
    monkeypatch.setattr(patches, "version", lambda _distribution: "0.25.2")
    monkeypatch.setattr(
        patches,
        "_patch_vllm_tool_parser_namespace_tool",
        lambda _logger: pytest.fail("source must not be edited for an unknown version"),
    )

    with pytest.raises(RuntimeError, match="pinned to vLLM 0.25.1, but found 0.25.2"):
        patches._ensure_vllm_tool_parser_namespace_tool()


def test_namespace_tool_compat_rejects_late_install(monkeypatch):
    monkeypatch.setattr(patches, "_namespace_tool_compat_installed_path", None)
    monkeypatch.setattr(
        patches,
        "version",
        lambda _distribution: patches._NAMESPACE_TOOL_COMPAT_VLLM_VERSION,
    )
    monkeypatch.setitem(
        sys.modules, "vllm.tool_parsers.utils", types.ModuleType("tool_parser")
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_tool_parser_namespace_tool",
        lambda _logger: pytest.fail("late source mutation must not be attempted"),
    )

    with pytest.raises(RuntimeError, match="imported before"):
        patches._ensure_vllm_tool_parser_namespace_tool()


def test_namespace_tool_compat_allows_repeat_after_safe_import(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "vllm.tool_parsers.utils", raising=False)
    tool_parser_source = tmp_path / "utils.py"
    tool_parser_source.write_text("patched\n")
    patch_calls = []
    monkeypatch.setattr(patches, "_namespace_tool_compat_installed_path", None)
    monkeypatch.setattr(
        patches,
        "version",
        lambda _distribution: patches._NAMESPACE_TOOL_COMPAT_VLLM_VERSION,
    )
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(tool_parser_source)
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_tool_parser_namespace_tool",
        lambda _logger: patch_calls.append("patch") or True,
    )

    first_logger = patches._ensure_vllm_tool_parser_namespace_tool()
    loaded_module = types.ModuleType("vllm.tool_parsers.utils")
    loaded_module.__file__ = str(tool_parser_source)
    monkeypatch.setitem(sys.modules, "vllm.tool_parsers.utils", loaded_module)
    second_logger = patches._ensure_vllm_tool_parser_namespace_tool()

    assert second_logger is first_logger
    assert patch_calls == ["patch"]


def test_namespace_tool_compat_rejects_repeat_from_other_origin(monkeypatch, tmp_path):
    installed_source = tmp_path / "installed" / "utils.py"
    loaded_source = tmp_path / "loaded" / "utils.py"
    installed_source.parent.mkdir()
    loaded_source.parent.mkdir()
    installed_source.write_text("patched\n")
    loaded_source.write_text("other\n")
    monkeypatch.setattr(
        patches, "_namespace_tool_compat_installed_path", str(installed_source)
    )
    monkeypatch.setattr(
        patches,
        "version",
        lambda _distribution: patches._NAMESPACE_TOOL_COMPAT_VLLM_VERSION,
    )
    loaded_module = types.ModuleType("vllm.tool_parsers.utils")
    loaded_module.__file__ = str(loaded_source)
    monkeypatch.setitem(sys.modules, "vllm.tool_parsers.utils", loaded_module)

    with pytest.raises(RuntimeError, match="unverified path"):
        patches._ensure_vllm_tool_parser_namespace_tool()


def test_locked_file_patch_replaces_atomically_and_preserves_mode(
    monkeypatch, tmp_path
):
    source = tmp_path / "source.py"
    source.write_text("old\n")
    source.chmod(0o640)
    real_replace = os.replace
    replacements = []

    def recording_replace(source_path, destination_path):
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(patches.os, "replace", recording_replace)

    with patches._locked_file_patch(str(source)) as (content, write_back):
        assert content == "old\n"
        write_back("new\n")

    assert source.read_text() == "new\n"
    assert source.stat().st_mode & 0o777 == 0o640
    assert len(replacements) == 1
    assert replacements[0][1] == str(source)
    assert not list(tmp_path.glob(".source.py.patch-*"))


def test_apply_patches_installs_namespace_guard_before_first_vllm_import(monkeypatch):
    """The compatibility source edit must precede vLLM package initialization."""
    events = []

    monkeypatch.setattr(
        patches,
        "_ensure_vllm_tool_parser_namespace_tool",
        lambda: events.append("namespace-guard"),
    )
    monkeypatch.setattr(
        patches, "_patch_vllm_init_workers_ray", lambda _python, _env: True
    )
    for patch_name in (
        "_patch_vllm_llama_eagle3_own_lm_head",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
        "_patch_vllm_radio_layerscale_loader",
    ):
        monkeypatch.setattr(patches, patch_name, lambda _logger: None)

    fake_envs = types.ModuleType("vllm.envs")
    fake_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = True
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = logging.getLogger
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.envs = fake_envs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm" or name.startswith("vllm."):
            if not any(event.startswith("import:") for event in events):
                assert events == ["namespace-guard"]
            events.append(f"import:{name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    patches._apply_vllm_patches("python")

    assert events[:2] == ["namespace-guard", "import:vllm.envs"]


def test_apply_patches_repeats_after_safe_tool_parser_import(monkeypatch, tmp_path):
    """A second worker setup in one process must recognize its own safe install."""
    monkeypatch.delitem(sys.modules, "vllm.tool_parsers.utils", raising=False)
    old_snippet, new_snippet = patch_snippets(_PATCH_FN)
    tool_parser_source = tmp_path / "utils.py"
    tool_parser_source.write_text(old_snippet)
    monkeypatch.setattr(patches, "_namespace_tool_compat_installed_path", None)
    monkeypatch.setattr(
        patches,
        "version",
        lambda _distribution: patches._NAMESPACE_TOOL_COMPAT_VLLM_VERSION,
    )
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(tool_parser_source)
    )
    monkeypatch.setattr(
        patches, "_patch_vllm_init_workers_ray", lambda _python, _env: True
    )
    for patch_name in (
        "_patch_vllm_llama_eagle3_own_lm_head",
        "_patch_vllm_ray_executor_v2_tcpstore_port",
        "_patch_vllm_shm_broadcast_bind_retry",
        "_patch_vllm_radio_layerscale_loader",
    ):
        monkeypatch.setattr(patches, patch_name, lambda _logger: None)

    fake_envs = types.ModuleType("vllm.envs")
    fake_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND = True
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = logging.getLogger
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.envs = fake_envs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger)

    patches._apply_vllm_patches("python")
    first_content = tool_parser_source.read_text()
    assert new_snippet in first_content

    loaded_module = types.ModuleType("vllm.tool_parsers.utils")
    loaded_module.__file__ = str(tool_parser_source)
    monkeypatch.setitem(sys.modules, "vllm.tool_parsers.utils", loaded_module)
    patches._apply_vllm_patches("python")

    assert tool_parser_source.read_text() == first_content


def test_direct_source_compat_does_not_import_vllm_before_patching(monkeypatch):
    """Raw ``vllm.LLM`` callers use this helper before their first import."""
    events = []
    monkeypatch.setattr(
        patches,
        "_ensure_vllm_tool_parser_namespace_tool",
        lambda: events.append("namespace-guard") or logging.getLogger(__name__),
    )
    monkeypatch.setattr(
        patches,
        "_patch_vllm_radio_layerscale_loader",
        lambda _logger: events.append("radio-patch"),
    )

    real_import = builtins.__import__

    def reject_vllm_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert not (name == "vllm" or name.startswith("vllm."))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_vllm_import)

    patches.ensure_vllm_source_compat()

    assert events == ["namespace-guard", "radio-patch"]


@pytest.mark.vllm
def test_namespace_tool_stub_never_matches(patched_tool_parser_source):
    """The stub must be a plain class, so isinstance() is always False.

    All upstream uses are ``isinstance(tool, NamespaceTool)`` guarding a
    namespace-tools branch, so degrading to "no namespace tools" is correct for
    a client that cannot construct them -- but only if nothing can be an
    instance of the stub.
    """
    namespace: dict = {}
    tree = ast.parse(patched_tool_parser_source.read_text())
    stub = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NamespaceTool"
    )
    exec(compile(ast.Module(body=[stub], type_ignores=[]), "<stub>", "exec"), namespace)
    stub_cls = namespace["NamespaceTool"]
    for value in ({}, "tool", 0, None, object()):
        assert not isinstance(value, stub_cls)


@pytest.mark.vllm
def test_radio_layerscale_patch_anchor_still_matches_installed_vllm(
    patched_radio_source,
):
    """Pin the vLLM 0.25.1 RADIO loader shape used by the source patch."""
    content = patched_radio_source.read_text()
    assert _RADIO_MARKER in content
    assert "Skip layer-scale entries that vLLM doesn't use" not in content
    ast.parse(content)


@pytest.mark.vllm
def test_radio_layerscale_patch_loads_explicit_and_initializes_folded_weights(
    patched_radio_source,
):
    content = patched_radio_source.read_text()
    assert 'vllm_key = f"model.encoder.layers.{layer_idx}.{suffix}"' in content
    assert 'name.endswith((".ls1", ".ls2"))' in content
    assert "param.data.fill_(initializer_factor)" in content
    assert "loaded_params.add(name)" in content


@pytest.mark.vllm
def test_radio_layerscale_patch_is_idempotent(patched_radio_source, monkeypatch):
    before = patched_radio_source.read_text()
    monkeypatch.setattr(
        patches, "_get_vllm_file", lambda _relative: str(patched_radio_source)
    )

    patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert patched_radio_source.read_text() == before


def test_radio_layerscale_patch_warns_on_unknown_source(monkeypatch, tmp_path, caplog):
    radio_source = tmp_path / "radio.py"
    radio_source.write_text("class RadioModel:\n    pass\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _relative: str(radio_source))

    with caplog.at_level(logging.WARNING):
        patches._patch_vllm_radio_layerscale_loader(logging.getLogger(__name__))

    assert radio_source.read_text() == "class RadioModel:\n    pass\n"
    assert "vLLM 0.25.1 source shape was not found" in caplog.text


@pytest.mark.parametrize(
    "existing,extra,expected",
    [
        (None, None, "RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        ("", ["MY_VAR"], "MY_VAR,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # A value the caller already set must survive, not be clobbered.
        ("PRESET", ["MY_VAR"], "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV"),
        # Duplicates collapse and surrounding whitespace is stripped.
        (
            " PRESET , MY_VAR ",
            ["MY_VAR"],
            "MY_VAR,PRESET,RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        ),
    ],
)
def test_ray_extra_env_vars_merge_is_additive(
    monkeypatch, tmp_path, existing, extra, expected
):
    """vLLM 0.25 replaced the ADDITIONAL_ENV_VARS source patch with this hook.

    It must add to whatever the caller already set rather than overwrite it --
    otherwise user ``extra_env_vars`` silently stop reaching the Ray workers.
    """
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    if existing is None:
        monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)
    else:
        monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", existing)

    patches._patch_vllm_init_workers_ray("py", extra)

    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == expected


def test_init_workers_ray_reports_a_missing_anchor(monkeypatch, tmp_path):
    """A reshaped call site must not be reported as a successful patch."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray_renamed(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))
    monkeypatch.delenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", raising=False)

    assert patches._patch_vllm_init_workers_ray("py", None) is False
    # The env merge still has to happen; it is independent of the file patch.
    assert os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == (
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV"
    )


def test_init_workers_ray_reports_success_and_is_idempotent(monkeypatch, tmp_path):
    """Patching twice against the same file still reports success."""
    ray_executor = tmp_path / "ray_executor.py"
    ray_executor.write_text("self._init_workers_ray(placement_group)\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _r: str(ray_executor))

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    once = ray_executor.read_text()
    assert 'runtime_env={"py_executable": "py-exec"}' in once

    assert patches._patch_vllm_init_workers_ray("py-exec", None) is True
    assert ray_executor.read_text() == once
