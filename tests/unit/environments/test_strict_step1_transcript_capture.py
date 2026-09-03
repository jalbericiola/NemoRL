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

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.environments.nemo_gym import (
    NemoGym,
    _STRICT_VERIFIER_CHILD_PROBE_PROGRAM,
    _attest_strict_verifier_derivation_runtime,
    _probe_strict_verifier_derivation_child_runtime,
    _validate_strict_verifier_derivation_runtime,
)
from nemo_rl.experience.interfaces import NEMO_GYM_STRICT_TRANSCRIPT_KEY


def test_strict_transcript_is_copied_before_nemo_gym_postprocess(monkeypatch) -> None:
    monkeypatch.setenv("STRICT_PAIR_LAUNCH_MODE", "submit")
    monkeypatch.setenv("STRICT_PAIR_ARM", "off")
    versions = {
        "openai_version": "2.6.1",
        "pydantic_version": "2.13.4",
    }
    process_index = {"schema": "nemo-rl-strict-gym-child-index-v1"}
    process_index_sha256 = "a" * 64

    async def run() -> None:
        request_params = {
            "input": [{"role": "user", "content": "question"}],
            "metadata": {"extra_body": '{"seed":7}'},
        }
        row = {
            "_rowidx": 0,
            "_ng_rollout_index": 0,
            "agent_ref": {"name": "test-agent"},
            "responses_create_params": deepcopy(request_params),
        }
        raw_result = {
            # Gym returns the Pydantic-expanded Responses request here.  The
            # replayable generation request must instead remain the exact
            # compact request that was dispatched in ``row``.
            "responses_create_params": {
                **deepcopy(request_params),
                "background": None,
                "input": [{"role": "user", "content": "question", "type": "message"}],
            },
            "response": {"id": "chatcmpl-1", "output": []},
            "reward": 1.0,
        }
        original_row = deepcopy(row)
        original_result = deepcopy(raw_result)

        class RolloutCollectionHelper:
            def run_examples(self, examples, head_server_config):
                del examples, head_server_config

                async def completed():
                    return row, raw_result

                return [completed()]

        class MockGym:
            cfg = {}
            rch = RolloutCollectionHelper()
            head_server_config = object()
            _tokenizer = object()
            _strict_verifier_derivation_runtime = versions
            _strict_verifier_derivation_child_diagnostic = {
                "runtime": versions,
                "in_process": {
                    "index": process_index,
                    "index_sha256": process_index_sha256,
                },
            }
            _strict_gym_child_process_index = process_index
            _strict_gym_child_process_index_sha256 = process_index_sha256
            _strict_main_step1_enabled_at_spinup = True

            @staticmethod
            def _require_spinup() -> None:
                return None

            @staticmethod
            def _postprocess_nemo_gym_to_nemo_rl_result(
                result_row,
                result,
                tokenizer,
                *,
                include_initial_multimodal_data,
            ):
                del result_row, tokenizer, include_initial_multimodal_data
                result["reward"] = 0.0
                result["response"]["postprocessed"] = True
                return {"message_log": []}

        streamed = []
        async for item in NemoGym.__ray_metadata__.modified_class.run_rollouts(
            MockGym(), [row], "strict-test"
        ):
            streamed.append(item)

        transcript = streamed[0][1][NEMO_GYM_STRICT_TRANSCRIPT_KEY]
        derived_verifier_request = deepcopy(original_row)
        derived_verifier_request["responses_create_params"] = original_result[
            "responses_create_params"
        ]
        derived_verifier_request["response"] = original_result["response"]
        assert transcript == {
            "generation_request": original_row["responses_create_params"],
            "model_response": original_result["response"],
            "agent_run_request": original_row,
            "derived_verifier_request": derived_verifier_request,
            "verifier_response": original_result,
            "verifier_request_derivation_runtime": {
                "openai_version": "2.6.1",
                "pydantic_version": "2.13.4",
            },
        }
        assert (
            transcript["generation_request"]
            != transcript["verifier_response"]["responses_create_params"]
        )
        assert "postprocessed" not in transcript["model_response"]
        assert (
            transcript["derived_verifier_request"]["response"]
            == transcript["model_response"]
        )
        assert transcript["verifier_response"]["reward"] == 1.0
        assert raw_result["reward"] == 0.0
        assert raw_result["response"]["postprocessed"] is True

    asyncio.run(run())


def test_strict_rollout_rejects_before_run_examples_without_child_attestation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STRICT_PAIR_LAUNCH_MODE", "submit")
    monkeypatch.setenv("STRICT_PAIR_ARM", "off")

    class RolloutCollectionHelper:
        called = False

        def run_examples(self, examples, head_server_config):
            del examples, head_server_config
            self.called = True
            raise AssertionError("run_examples must not precede child attestation")

    helper = RolloutCollectionHelper()

    class MockGym:
        cfg = {}
        rch = helper
        head_server_config = object()
        _tokenizer = object()
        _strict_verifier_derivation_runtime = None
        _strict_verifier_derivation_child_diagnostic = None
        _strict_gym_child_process_index = None
        _strict_gym_child_process_index_sha256 = None
        _strict_main_step1_enabled_at_spinup = True

        @staticmethod
        def _require_spinup() -> None:
            return None

    async def run() -> None:
        stream = NemoGym.__ray_metadata__.modified_class.run_rollouts(
            MockGym(), [{}], "strict-test"
        )
        with pytest.raises(RuntimeError, match="was not attested during Gym spinup"):
            await anext(stream)

    asyncio.run(run())
    assert helper.called is False


def test_strict_rollout_rejects_activation_drift_before_run_examples(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STRICT_PAIR_LAUNCH_MODE", "submit")
    monkeypatch.setenv("STRICT_PAIR_ARM", "off")

    class RolloutCollectionHelper:
        called = False

        def run_examples(self, examples, head_server_config):
            del examples, head_server_config
            self.called = True
            raise AssertionError("run_examples must not follow strict-mode drift")

    helper = RolloutCollectionHelper()

    class MockGym:
        cfg = {}
        rch = helper
        head_server_config = object()
        _tokenizer = object()
        _strict_main_step1_enabled_at_spinup = False

        @staticmethod
        def _require_spinup() -> None:
            return None

    async def run() -> None:
        stream = NemoGym.__ray_metadata__.modified_class.run_rollouts(
            MockGym(), [{}], "strict-test"
        )
        with pytest.raises(RuntimeError, match="activation changed after"):
            await anext(stream)

    asyncio.run(run())
    assert helper.called is False


def _install_spinup_fakes(
    monkeypatch,
    *,
    strict_enabled: bool,
    attest_error: BaseException | None = None,
    cleanup_error: BaseException | None = None,
) -> tuple[SimpleNamespace, list[str]]:
    from nemo_rl.environments import nemo_gym as nemo_gym_module
    from nemo_rl.environments import strict_gym_child_runtime

    events: list[str] = []

    class ParserConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class RunHelper:
        def __init__(self):
            events.append("run-helper-init")

        def start(self, *, global_config_dict_parser_config):
            del global_config_dict_parser_config
            if strict_enabled:
                assert os.environ.get("STRICT_TEST_CHILD_LAUNCH") == "1"
            else:
                assert "STRICT_TEST_CHILD_LAUNCH" not in os.environ
            events.append("run-helper-start")

        def shutdown(self):
            events.append("run-helper-shutdown")
            if cleanup_error is not None:
                raise cleanup_error

    class RolloutCollectionHelper:
        def __init__(self):
            events.append("rollout-helper-init")

    class BaseServerConfig:
        def __init__(self, *, host, port):
            assert host == "127.0.0.1"
            assert port == 43210
            events.append("base-server-config")

    class Session:
        receipt_root = Path("/strict-test-receipts")

        @contextmanager
        def launch_environment(self):
            events.append("launch-enter")
            previous = os.environ.get("STRICT_TEST_CHILD_LAUNCH")
            os.environ["STRICT_TEST_CHILD_LAUNCH"] = "1"
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("STRICT_TEST_CHILD_LAUNCH", None)
                else:
                    os.environ["STRICT_TEST_CHILD_LAUNCH"] = previous
                events.append("launch-exit")

        def attest_started(self, run_helper):
            assert isinstance(run_helper, RunHelper)
            assert "STRICT_TEST_CHILD_LAUNCH" not in os.environ
            events.append("process-attest")
            if attest_error is not None:
                raise attest_error
            return {"schema": "nemo-rl-strict-gym-child-index-v1"}, "a" * 64

    def prepare(*, scope):
        assert scope == "main"
        events.append("prepare")
        return Session()

    def derivation_attest(run_helper):
        assert isinstance(run_helper, RunHelper)
        events.append("derivation-attest")
        return (
            {"openai_version": "2.6.1", "pydantic_version": "2.13.4"},
            {"runtime": "supporting"},
        )

    monkeypatch.setattr(
        nemo_gym_module, "strict_main_step1_enabled", lambda: strict_enabled
    )
    monkeypatch.setattr(nemo_gym_module, "_get_node_ip_local", lambda: "127.0.0.1")
    monkeypatch.setattr(
        nemo_gym_module, "_get_free_port_local", lambda low, high: 43210
    )
    monkeypatch.setattr(nemo_gym_module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        nemo_gym_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(gcs_address="ray://strict-test"),
    )
    monkeypatch.setattr(
        nemo_gym_module,
        "_attest_strict_verifier_derivation_runtime",
        derivation_attest,
    )
    monkeypatch.setattr(
        strict_gym_child_runtime, "prepare_strict_gym_child_runtime", prepare
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_gym.cli",
        SimpleNamespace(
            GlobalConfigDictParserConfig=ParserConfig,
            RunHelper=RunHelper,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_gym.rollout_collection",
        SimpleNamespace(RolloutCollectionHelper=RolloutCollectionHelper),
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_gym.server_utils",
        SimpleNamespace(
            HEAD_SERVER_KEY_NAME="head_server",
            BaseServerConfig=BaseServerConfig,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        SimpleNamespace(DictConfig=lambda value: value),
    )

    actor = SimpleNamespace(
        cfg={
            "model_name": "strict-test-model",
            "base_urls": ["http://strict-test-vllm"],
        },
        rh=None,
        rch=None,
        head_server_config=None,
        node_ip=None,
        head_server_port=None,
        _strict_verifier_derivation_runtime=None,
        _strict_verifier_derivation_child_diagnostic=None,
        _strict_gym_child_process_index=None,
        _strict_gym_child_process_index_sha256=None,
        _strict_main_step1_enabled_at_spinup=None,
    )
    return actor, events


def test_strict_spinup_attests_live_children_before_rollout_helper(
    monkeypatch,
) -> None:
    actor, events = _install_spinup_fakes(monkeypatch, strict_enabled=True)

    NemoGym.__ray_metadata__.modified_class._spinup(actor)

    assert events == [
        "prepare",
        "run-helper-init",
        "launch-enter",
        "run-helper-start",
        "launch-exit",
        "process-attest",
        "derivation-attest",
        "base-server-config",
        "rollout-helper-init",
    ]
    assert actor._strict_main_step1_enabled_at_spinup is True
    assert actor._strict_gym_child_process_index == {
        "schema": "nemo-rl-strict-gym-child-index-v1"
    }
    assert actor._strict_gym_child_process_index_sha256 == "a" * 64
    assert (
        actor._strict_verifier_derivation_child_diagnostic["in_process"]["index_sha256"]
        == "a" * 64
    )


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_strict_spinup_attestation_failure_cleans_without_masking(
    monkeypatch, cleanup_fails: bool
) -> None:
    primary = RuntimeError("process receipt invalid")
    cleanup = RuntimeError("cleanup failed") if cleanup_fails else None
    actor, events = _install_spinup_fakes(
        monkeypatch,
        strict_enabled=True,
        attest_error=primary,
        cleanup_error=cleanup,
    )

    with pytest.raises(RuntimeError, match="process receipt invalid") as caught:
        NemoGym.__ray_metadata__.modified_class._spinup(actor)

    assert caught.value is primary
    assert events == [
        "prepare",
        "run-helper-init",
        "launch-enter",
        "run-helper-start",
        "launch-exit",
        "process-attest",
        "run-helper-shutdown",
    ]
    assert actor.rh is None
    assert actor.rch is None
    assert actor.head_server_config is None
    assert actor._strict_gym_child_process_index is None
    if cleanup_fails:
        assert any("cleanup failed" in note for note in caught.value.__notes__)


def test_nonstrict_spinup_does_not_prepare_child_attestation(monkeypatch) -> None:
    actor, events = _install_spinup_fakes(monkeypatch, strict_enabled=False)

    NemoGym.__ray_metadata__.modified_class._spinup(actor)

    assert events == [
        "run-helper-init",
        "run-helper-start",
        "base-server-config",
        "rollout-helper-init",
    ]
    assert actor._strict_main_step1_enabled_at_spinup is False
    assert actor._strict_gym_child_process_index is None


@pytest.mark.parametrize(
    ("versions", "missing", "match"),
    [
        (
            {"openai": "2.7.2", "pydantic": "2.13.4"},
            None,
            "openai expected '2.6.1', got '2.7.2'",
        ),
        (
            {"openai": "2.6.1"},
            "pydantic",
            "requires installed distribution 'pydantic'==2.13.4",
        ),
        (
            {"openai": 261, "pydantic": "2.13.4"},
            None,
            "openai expected '2.6.1', got 261",
        ),
    ],
)
def test_strict_verifier_derivation_rejects_wrong_or_missing_runtime(
    monkeypatch, versions: dict[str, object], missing: str | None, match: str
) -> None:
    def installed_version(distribution: str):
        if distribution == missing:
            raise PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.importlib_metadata.version",
        installed_version,
    )

    with pytest.raises(RuntimeError, match=match):
        _validate_strict_verifier_derivation_runtime()


def test_strict_verifier_derivation_rejects_actor_child_runtime_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym._validate_strict_verifier_derivation_runtime",
        lambda: {"openai_version": "2.6.1", "pydantic_version": "2.13.4"},
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym._probe_strict_verifier_derivation_child_runtime",
        lambda helper: (
            {"openai_version": "2.7.2", "pydantic_version": "2.13.4"},
            {"runtime": "child"},
        ),
    )

    with pytest.raises(
        RuntimeError, match="actor and SimpleAgent child runtimes differ"
    ):
        _attest_strict_verifier_derivation_runtime(object())


def _child_probe_fixture(
    monkeypatch,
    tmp_path: Path,
    *,
    instances: list[SimpleNamespace] | None = None,
    venv_root: Path | None = None,
) -> tuple[SimpleNamespace, Path, list[list[str]]]:
    import nemo_gym as nemo_gym_package
    from nemo_gym import global_config
    from nemo_gym.cli import setup_command
    from nemo_rl.environments import nemo_gym as nemo_gym_module

    gym_source_root = tmp_path / "Gym"
    instance_dir = gym_source_root / "responses_api_agents" / "simple_agent"
    instance_dir.mkdir(parents=True)
    source_bytes = b"authenticated SimpleAgent test source\n"
    (instance_dir / "app.py").write_bytes(source_bytes)
    if venv_root is None:
        # Production uses an external /opt/gym_venvs-style root.  Gym appends
        # the server type/name before .venv.
        venv_root = tmp_path / "gym_venvs"
    selected_venv = venv_root / "responses_api_agents" / "simple_agent" / ".venv"
    bin_dir = selected_venv / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python"
    interpreter.symlink_to(sys.executable)
    if instances is None:
        instances = [
            SimpleNamespace(
                server_type="responses_api_agents",
                name="reasoning_gym_simple_agent",
                config_path="reasoning_gym_simple_agent",
                dir_path=instance_dir,
                pid=123,
            )
        ]
    helper = SimpleNamespace(_server_instance_display_configs=instances)
    calls: list[list[str]] = []
    monkeypatch.setattr(nemo_gym_package, "PARENT_DIR", gym_source_root)
    monkeypatch.setattr(
        nemo_gym_module, "_STRICT_VERIFIER_GYM_SOURCE_ROOT", gym_source_root
    )
    monkeypatch.setattr(
        nemo_gym_module,
        "_STRICT_VERIFIER_CHILD_SOURCE_SHA256",
        hashlib.sha256(source_bytes).hexdigest(),
    )
    monkeypatch.setattr(nemo_gym_module, "_STRICT_VERIFIER_CHILD_VENV_ROOT", venv_root)
    monkeypatch.setattr(
        global_config,
        "get_global_config_dict",
        lambda: {global_config.UV_VENV_DIR_KEY_NAME: str(venv_root)},
    )
    monkeypatch.setattr(
        setup_command,
        "get_venv_path",
        lambda dir_path, global_config_dict: selected_venv,
    )

    def completed_run(argv, *, check, capture_output, timeout):
        del check, capture_output, timeout
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(
                    {
                        "openai_version": "2.6.1",
                        "pydantic_version": "2.13.4",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr("nemo_rl.environments.nemo_gym.subprocess.run", completed_run)
    return helper, interpreter, calls


def test_strict_verifier_derivation_probes_selected_child_venv(
    monkeypatch, tmp_path: Path
) -> None:
    helper, interpreter, calls = _child_probe_fixture(monkeypatch, tmp_path)

    runtime, diagnostic = _probe_strict_verifier_derivation_child_runtime(helper)

    assert runtime == {
        "openai_version": "2.6.1",
        "pydantic_version": "2.13.4",
    }
    assert calls == [
        [
            str(interpreter),
            "-I",
            "-c",
            _STRICT_VERIFIER_CHILD_PROBE_PROGRAM,
        ]
    ]
    assert diagnostic["server_type"] == "responses_api_agents"
    assert diagnostic["server_name"] == "reasoning_gym_simple_agent"
    assert diagnostic["gym_source_root"] == str(tmp_path / "Gym")
    assert (
        diagnostic["simple_agent_source_sha256"]
        == hashlib.sha256(b"authenticated SimpleAgent test source\n").hexdigest()
    )
    assert diagnostic["interpreter"] == str(interpreter)
    assert diagnostic["configured_venv_root"] == str(tmp_path / "gym_venvs")
    assert diagnostic["resolved_interpreter"] == str(Path(sys.executable).resolve())
    assert diagnostic["runtime"] == runtime
    assert diagnostic["interpreter_lstat"]["st_ino"] == interpreter.lstat().st_ino
    assert (
        diagnostic["interpreter_stat"]["st_ino"] == Path(sys.executable).stat().st_ino
    )


@pytest.mark.parametrize(
    ("stdout", "returncode", "stderr", "match"),
    [
        (
            b'{"openai_version":"2.7.2","pydantic_version":"2.13.4"}\n',
            0,
            b"",
            "child runtime differs",
        ),
        (
            b'{"openai_version":261,"pydantic_version":"2.13.4"}\n',
            0,
            b"",
            "versions must be exact strings",
        ),
        (
            b"",
            1,
            b"PackageNotFoundError: pydantic",
            "package probe failed closed",
        ),
        (
            b'{ "openai_version":"2.6.1","pydantic_version":"2.13.4" }\n',
            0,
            b"",
            "output is not canonical JSON-LF",
        ),
        (
            b'{"openai_version":"2.6.1","openai_version":"2.6.1",'
            b'"pydantic_version":"2.13.4"}\n',
            0,
            b"",
            "output is not strict ASCII JSON",
        ),
        (b"x" * 4097, 0, b"", "package probe failed closed"),
    ],
)
def test_strict_verifier_derivation_rejects_child_probe_failures(
    monkeypatch,
    tmp_path: Path,
    stdout: bytes,
    returncode: int,
    stderr: bytes,
    match: str,
) -> None:
    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(RuntimeError, match=match):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_ambiguous_child(
    monkeypatch, tmp_path: Path
) -> None:
    instances = [
        SimpleNamespace(
            server_type="responses_api_agents",
            name=f"agent-{index}",
            dir_path=tmp_path,
        )
        for index in range(2)
    ]
    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path, instances=instances)

    with pytest.raises(RuntimeError, match="exactly one .* child; found 2"):
        _probe_strict_verifier_derivation_child_runtime(helper)


@pytest.mark.parametrize("instances", [None, (), []])
def test_strict_verifier_derivation_rejects_missing_child_metadata(
    instances: object,
) -> None:
    helper = SimpleNamespace(_server_instance_display_configs=instances)

    with pytest.raises(
        RuntimeError, match="server instance metadata|exactly one .* child; found 0"
    ):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_child_venv_outside_configured_root(
    monkeypatch, tmp_path: Path
) -> None:
    from nemo_gym.cli import setup_command

    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    outside = tmp_path / "outside" / ".venv"
    outside.mkdir(parents=True)
    monkeypatch.setattr(
        setup_command,
        "get_venv_path",
        lambda dir_path, global_config_dict: outside,
    )

    with pytest.raises(RuntimeError, match="child venv could not be resolved"):
        _probe_strict_verifier_derivation_child_runtime(helper)


@pytest.mark.parametrize(
    "selected",
    [
        Path("/tmp/gym_venvs/responses_api_agents/../simple_agent/.venv"),
        Path("/tmp/gym_venvs/responses_api_agents/wrong_agent/.venv"),
        Path("/tmp/gym_venvs"),
    ],
)
def test_strict_verifier_derivation_rejects_wrong_child_venv_path(
    monkeypatch, tmp_path: Path, selected: Path
) -> None:
    from nemo_gym.cli import setup_command

    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    configured_root = tmp_path / "gym_venvs"
    relative = selected.relative_to("/tmp/gym_venvs")
    selected_under_test_root = configured_root / relative
    monkeypatch.setattr(
        setup_command,
        "get_venv_path",
        lambda dir_path, global_config_dict: selected_under_test_root,
    )

    with pytest.raises(RuntimeError, match="child venv could not be resolved"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_relative_configured_root(
    monkeypatch, tmp_path: Path
) -> None:
    from nemo_gym import global_config

    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        global_config,
        "get_global_config_dict",
        lambda: {global_config.UV_VENV_DIR_KEY_NAME: "relative/gym_venvs"},
    )

    with pytest.raises(RuntimeError, match="child venv could not be resolved"):
        _probe_strict_verifier_derivation_child_runtime(helper)


@pytest.mark.parametrize("alias_kind", ["double-slash", "dot", "suffix-dot", "path"])
def test_strict_verifier_derivation_rejects_nonexact_configured_root_alias(
    monkeypatch, tmp_path: Path, alias_kind: str
) -> None:
    from nemo_gym import global_config

    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    canonical = tmp_path / "gym_venvs"
    aliases = {
        "double-slash": f"{tmp_path}//gym_venvs",
        "dot": f"{tmp_path}/./gym_venvs",
        "suffix-dot": f"{canonical}/.",
        "path": canonical,
    }
    monkeypatch.setattr(
        global_config,
        "get_global_config_dict",
        lambda: {global_config.UV_VENV_DIR_KEY_NAME: aliases[alias_kind]},
    )

    with pytest.raises(RuntimeError, match="child venv could not be resolved"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_wrong_configured_root(
    monkeypatch, tmp_path: Path
) -> None:
    from nemo_gym import global_config

    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    wrong_root = tmp_path / "wrong-gym-venvs"
    wrong_root.mkdir()
    monkeypatch.setattr(
        global_config,
        "get_global_config_dict",
        lambda: {global_config.UV_VENV_DIR_KEY_NAME: str(wrong_root)},
    )

    with pytest.raises(RuntimeError, match="child venv could not be resolved"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_foreign_same_suffix_child_source(
    monkeypatch, tmp_path: Path
) -> None:
    wrong_dir = tmp_path / "foreign" / "responses_api_agents" / "simple_agent"
    wrong_dir.mkdir(parents=True)
    (wrong_dir / "app.py").write_bytes(b"hostile shadow SimpleAgent\n")
    instance = SimpleNamespace(
        server_type="responses_api_agents",
        name="reasoning_gym_simple_agent",
        dir_path=wrong_dir,
    )
    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path, instances=[instance])

    with pytest.raises(RuntimeError, match="differs from the authenticated Gym source"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_changed_simple_agent_source(
    monkeypatch, tmp_path: Path
) -> None:
    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)
    source_path = tmp_path / "Gym/responses_api_agents/simple_agent/app.py"
    source_path.write_bytes(b"changed after authenticated source selection\n")

    with pytest.raises(RuntimeError, match="source differs from the authenticated"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_child_venv_symlink_escape(
    monkeypatch, tmp_path: Path
) -> None:
    helper, interpreter, _ = _child_probe_fixture(monkeypatch, tmp_path)
    selected_venv = interpreter.parents[1]
    interpreter.unlink()
    interpreter.parent.rmdir()
    selected_venv.rmdir()
    outside_venv = tmp_path / "outside-venv"
    outside_bin = outside_venv / "bin"
    outside_bin.mkdir(parents=True)
    (outside_bin / "python").symlink_to(sys.executable)
    selected_venv.symlink_to(outside_venv, target_is_directory=True)

    with pytest.raises(RuntimeError, match="venv is outside the configured venv root"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_missing_child_interpreter(
    monkeypatch, tmp_path: Path
) -> None:
    helper, interpreter, _ = _child_probe_fixture(monkeypatch, tmp_path)
    interpreter.unlink()

    with pytest.raises(RuntimeError, match="child interpreter is unavailable"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_nonexecutable_child_interpreter(
    monkeypatch, tmp_path: Path
) -> None:
    helper, interpreter, _ = _child_probe_fixture(monkeypatch, tmp_path)
    interpreter.unlink()
    interpreter.write_bytes(b"not executable")
    interpreter.chmod(0o600)

    with pytest.raises(RuntimeError, match="not an executable regular file"):
        _probe_strict_verifier_derivation_child_runtime(helper)


def test_strict_verifier_derivation_rejects_child_probe_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    helper, _, _ = _child_probe_fixture(monkeypatch, tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("nemo_rl.environments.nemo_gym.subprocess.run", timeout)

    with pytest.raises(RuntimeError, match="package probe could not complete"):
        _probe_strict_verifier_derivation_child_runtime(helper)
