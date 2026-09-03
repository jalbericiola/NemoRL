# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ENTRYPOINT = _REPOSITORY_ROOT / "examples/nemo_gym/run_strict_captured_replay_v2.py"


def _load_entrypoint() -> Any:
    name = "_strict_captured_replay_v2_unit_entrypoint"
    spec = importlib.util.spec_from_file_location(name, _ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


@pytest.fixture(scope="module")
def runner() -> Any:
    return _load_entrypoint()


def test_entrypoint_has_no_eager_repository_import_and_exact_v2_roster(
    runner: Any,
) -> None:
    tree = ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not any(
            alias.name.startswith(("nemo_rl", "nemo_gym", "ray", "omegaconf"))
            for alias in node.names
        )
        for node in top_level_imports
    )
    assert runner._PROGRAM_PATHS == {
        "entrypoint": "examples/nemo_gym/run_strict_captured_replay_v2.py",
        "evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence_v2.py",
        "gym_child_bootstrap": "nemo_rl/environments/_strict_gym_child_bootstrap_v2/sitecustomize.py",
        "gym_child_runtime": "nemo_rl/environments/strict_gym_child_runtime_v2.py",
        "job_wrapper": "strict_pair_replay_job_wrapper_v2.sh",
        "legacy_evidence_utility": "nemo_rl/utils/strict_captured_replay_evidence.py",
        "main_step_ledger": "nemo_rl/utils/strict_main_step_ledger.py",
        "manifest_utility": "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
        "model_transport_utility": "nemo_rl/utils/strict_model_transport.py",
        "profile_registry": "nemo_rl/utils/strict_captured_replay_profiles.py",
        "raw_transport_owner": "nemo_rl/utils/strict_model_transport_replay_v3.py",
        "result_sealer": "nemo_rl/utils/strict_captured_replay_seal_v2.py",
        "runtime": "nemo_rl/algorithms/strict_captured_replay_runtime_v2.py",
        "submission_launcher": "strict_pair_replay_launch_v2.sh",
    }
    assert len(runner._PROGRAM_PATHS) == 14


def test_entrypoint_pair79_roster_matches_v2_manifest(runner: Any) -> None:
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        REPLAY_SLURM_EXPORT_SCHEMA,
        SLURM_EXPORT_ALLOWED_NAMES,
    )

    assert runner._SLURM_EXPORT_ALLOWED_NAMES == SLURM_EXPORT_ALLOWED_NAMES
    assert len(runner._SLURM_EXPORT_ALLOWED_NAMES) == 79
    assert runner._REPLAY_SLURM_EXPORT_SCHEMA == REPLAY_SLURM_EXPORT_SCHEMA
    assert REPLAY_SLURM_EXPORT_SCHEMA.endswith("-v2")


def test_exact_v4_lifecycle_root_keysets(runner: Any) -> None:
    assert runner._MANIFEST_ROOT_KEYS == frozenset(
        {
            "arm",
            "artifacts",
            "attempt_id",
            "container_entry_boundary",
            "deployment",
            "environment",
            "execution_environment",
            "hash_domain",
            "mode",
            "pair",
            "pair_id",
            "replay_contract",
            "runtime_attestation_requirements",
            "runtime_tools",
            "scheduler_submission",
            "schema",
            "scorer_profile",
            "slurm_export_boundary",
            "source",
            "source_capture",
            "wandb",
        }
    )
    assert runner._PRE_ROOT_KEYS == frozenset(
        {
            "arm",
            "attempt_id",
            "authenticated_job_id",
            "candidate_job_id",
            "driver",
            "environment",
            "execution_source_root",
            "job",
            "mode",
            "output_precondition",
            "pair_id",
            "phase",
            "post_verified",
            "pre_scheduler_query",
            "replay_execution_manifest",
            "runtime_attestation_contract",
            "schema",
            "scorer_profile",
            "static_boundary",
            "status",
            "submission_receipt",
        }
    )
    assert runner._REPLAY_SUBMISSION_ROOT_KEYS == frozenset(
        {
            "accepted_id_record",
            "arm",
            "attempt_id",
            "candidate_job_id",
            "comment",
            "environment",
            "job_name",
            "job_wrapper",
            "mode",
            "pair_id",
            "phase",
            "pre_release_scheduler_query",
            "replay_execution_manifest",
            "replay_source_snapshot",
            "sbatch",
            "scheduler_client_environment",
            "scheduler_tools",
            "schema",
            "scorer_profile",
            "slurm_export_boundary",
            "status",
            "submission_contract",
            "submission_launcher",
            "submission_nonce",
            "submitted_at_unix_ns",
            "submitter_euid",
        }
    )


def test_cli_and_slurm_job_argv_are_explicit_and_frozen(runner: Any) -> None:
    arguments = [
        "--replay-driver-phase",
        "--replay-manifest",
        "/strict/replay.json",
        "--replay-manifest-sha256",
        "a" * 64,
        "--pre-receipt",
        "/strict/pre.json",
        "--pre-receipt-sha256",
        "b" * 64,
        "--environment",
        "citation",
        "--profile-id",
        "citation-string-match-v1",
    ]
    parsed = runner._parser().parse_args(arguments)
    assert parsed.environment == "citation"
    assert parsed.profile_id == "citation-string-match-v1"
    with pytest.raises(SystemExit):
        runner._parser().parse_args(arguments[:-2])

    function_tree = ast.parse(
        inspect.getsource(runner._load_authenticated_slurm_export)
    )
    templates = []
    for node in ast.walk(function_tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "job_argv_template":
                templates.append(ast.literal_eval(value))
    assert templates == [
        [
            "--pair-manifest",
            "{pair_manifest_path}",
            "--pair-manifest-sha256",
            "{pair_manifest_sha256}",
            "--pair-submission-receipt",
            "{pair_submission_receipt_path}",
            "--pair-submission-receipt-sha256",
            "{pair_submission_receipt_sha256}",
            "--off-exit-receipt",
            "{trusted_off_exit_receipt_path}",
            "--off-exit-receipt-sha256",
            "{trusted_off_exit_receipt_sha256}",
            "--replay-manifest",
            "{replay_manifest_path}",
            "--replay-manifest-sha256",
            "{replay_manifest_sha256}",
            "--environment",
            "{environment}",
            "--profile-id",
            "{profile_id}",
        ]
    ]
    assert len(templates[0]) == 20


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_only_closed_format_profiles_are_admitted(
    runner: Any,
    environment: str,
    profile_id: str,
) -> None:
    profile = runner._closed_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert profile["environment"] == environment
    assert profile["profile_id"] == profile_id
    assert profile["method"] in {"_verify_string_match", "_verify_regex"}
    assert profile["fixture"]["rows"] == 5
    assert profile["fixture"]["path"] == (
        f"tests/unit/tools/data/{environment}_example.jsonl"
    )
    assert (
        profile["fixture"]["sha256"]
        == {
            "citation": (
                "d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc"
            ),
            "freeform": (
                "8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399"
            ),
        }[environment]
    )
    with pytest.raises(runner.StrictCapturedReplayEntrypointError):
        runner._closed_profile(
            expected_environment=environment,
            expected_profile_id="wrong-profile",
        )


def _clear_program_modules(monkeypatch: pytest.MonkeyPatch, runner: Any) -> None:
    for module_name in runner._PROGRAM_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)


def test_required_program_origin_check_rejects_missing_and_poisoned_modules(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_program_modules(monkeypatch, runner)
    root = tmp_path / "snapshot"
    root.mkdir()
    expected_path = root / "runtime.py"
    expected_path.write_bytes(b"authenticated runtime\n")
    expected_path.chmod(0o444)
    program = {
        "runtime": {
            "path": "runtime.py",
            "sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        }
    }
    required = frozenset({"runtime"})
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="not imported",
    ):
        runner._verify_imported_program_modules(
            execution_source_root=root,
            program=program,
            required_program_names=required,
        )

    module_name = "nemo_rl.algorithms.strict_captured_replay_runtime_v2"
    authenticated = types.ModuleType(module_name)
    authenticated.__file__ = str(expected_path)
    monkeypatch.setitem(sys.modules, module_name, authenticated)
    runner._verify_imported_program_modules(
        execution_source_root=root,
        program=program,
        required_program_names=required,
    )

    poison_path = tmp_path / "poison.py"
    poison_path.write_bytes(expected_path.read_bytes())
    poison_path.chmod(0o444)
    authenticated.__file__ = str(poison_path)
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="differs from authenticated program",
    ):
        runner._verify_imported_program_modules(
            execution_source_root=root,
            program=program,
            required_program_names=required,
        )


class _FakeParserConfig:
    def __init__(self, *, initial_global_config_dict: Any, **kwargs: Any) -> None:
        self.initial = initial_global_config_dict
        self.kwargs = kwargs


class _FakeDictConfig(dict[str, Any]):
    pass


def _install_fake_gym_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gym_root: Path,
    environment: str,
    profile: dict[str, Any],
    poison: bool,
) -> None:
    gym_module = types.ModuleType("nemo_gym")
    gym_module.__path__ = []
    gym_module.PARENT_DIR = gym_root
    cli_module = types.ModuleType("nemo_gym.cli")
    cli_module.GlobalConfigDictParserConfig = _FakeParserConfig
    global_module = types.ModuleType("nemo_gym.global_config")

    disabled_name = profile["disabled_config_path_name"]
    resource_name = profile["resource_config_path_name"]
    reserved_names = {
        "cache_dir",
        "config_paths",
        "default_host",
        "head_server",
        "model_endpoint_readiness_timeout_seconds",
        "observability_enabled",
        "port_range_high",
        "port_range_low",
        "ray_head_node_address",
        "results_dir",
        "skip_venv_if_present",
        "uv_cache_dir",
        "uv_venv_dir",
    }
    global_module.NEMO_GYM_RESERVED_TOP_LEVEL_KEYS = frozenset(reserved_names)
    global_module.maybe_get_global_config_dict = lambda: None

    def resolve(
        *, global_config_dict_parser_config: _FakeParserConfig
    ) -> dict[str, Any]:
        initial = dict(global_config_dict_parser_config.initial)
        assert initial[disabled_name] == {"_delete_key": "responses_api_agents"}
        leaf: dict[str, Any] = {"entrypoint": "app.py"}
        if poison:
            leaf["responses_api_models"] = {"poison": True}
        return {
            **{name: initial[name] for name in reserved_names},
            disabled_name: {},
            resource_name: {"resources_servers": {"format_verification": leaf}},
        }

    global_module.get_global_config_dict = resolve
    omega_module = types.ModuleType("omegaconf")
    omega_module.DictConfig = _FakeDictConfig
    monkeypatch.setitem(sys.modules, "nemo_gym", gym_module)
    monkeypatch.setitem(sys.modules, "nemo_gym.cli", cli_module)
    monkeypatch.setitem(sys.modules, "nemo_gym.global_config", global_module)
    monkeypatch.setitem(sys.modules, "omegaconf", omega_module)


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_reduced_parser_keeps_exactly_one_resource_and_no_agent_or_model(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: str,
    profile_id: str,
) -> None:
    profile = runner._closed_profile(
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    gym_root = tmp_path / runner._GYM_SOURCE_RELATIVE
    config_path = gym_root / profile["resource_config"]["path"]
    config_path.parent.mkdir(parents=True)
    config_path.write_text("authenticated: yaml\n", encoding="ascii")
    _install_fake_gym_modules(
        monkeypatch,
        gym_root=gym_root,
        environment=environment,
        profile=profile,
        poison=False,
    )
    hash_checks: list[str] = []

    def stable_hash(path: Path, *, name: str) -> str:
        assert path == config_path
        hash_checks.append(name)
        return profile["resource_config"]["sha256"]

    monkeypatch.setattr(runner, "_stable_regular_sha256", stable_hash)
    manifest = {
        "scorer_profile": profile,
        "replay_contract": {
            "gym_scorer": {
                "launcher": {
                    "log_wrapper": "forbidden",
                    "resource_only_config": None,
                    "config_path_name": profile["resource_config_path_name"],
                },
                "resources": {"config": profile["resource_config"]},
            }
        },
        "execution_environment": {"attempt": {"persistent_cache": "/strict/cache"}},
        "artifacts": {"outputs": {"directory": {"path": "/strict/output"}}},
    }
    ray_module = types.SimpleNamespace(
        get_runtime_context=lambda: types.SimpleNamespace(gcs_address="127.0.0.1:6379")
    )
    parser_config = runner._build_format_resource_only_parser_config(
        manifest=manifest,
        execution_source_root=tmp_path,
        ray_module=ray_module,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert type(parser_config) is _FakeParserConfig
    assert parser_config.initial[profile["disabled_config_path_name"]] == {
        "_delete_key": "responses_api_agents"
    }
    assert hash_checks == [
        "selected format Gym config",
        "post-parse selected format Gym config",
    ]


def test_reduced_parser_poison_rejects_nested_model_server(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = runner._closed_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    gym_root = tmp_path / runner._GYM_SOURCE_RELATIVE
    config_path = gym_root / profile["resource_config"]["path"]
    config_path.parent.mkdir(parents=True)
    config_path.write_text("authenticated: yaml\n", encoding="ascii")
    _install_fake_gym_modules(
        monkeypatch,
        gym_root=gym_root,
        environment="citation",
        profile=profile,
        poison=True,
    )
    monkeypatch.setattr(
        runner,
        "_stable_regular_sha256",
        lambda *args, **kwargs: profile["resource_config"]["sha256"],
    )
    manifest = {
        "scorer_profile": profile,
        "replay_contract": {
            "gym_scorer": {
                "launcher": {
                    "log_wrapper": "forbidden",
                    "resource_only_config": None,
                    "config_path_name": profile["resource_config_path_name"],
                },
                "resources": {"config": profile["resource_config"]},
            }
        },
        "execution_environment": {"attempt": {"persistent_cache": "/strict/cache"}},
        "artifacts": {"outputs": {"directory": {"path": "/strict/output"}}},
    }
    with pytest.raises(
        runner.StrictCapturedReplayEntrypointError,
        match="agent or model server",
    ):
        runner._build_format_resource_only_parser_config(
            manifest=manifest,
            execution_source_root=tmp_path,
            ray_module=types.SimpleNamespace(
                get_runtime_context=lambda: types.SimpleNamespace(
                    gcs_address="127.0.0.1:6379"
                )
            ),
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_child_lifecycle_uses_exact_parser_type_and_finalizes_before_transport(
    runner: Any,
) -> None:
    child_source = inspect.getsource(
        runner.run_profiled_replay_with_authenticated_resource_child
    )
    assert (
        "type(resource_parser_config) is not GlobalConfigDictParserConfig"
        in child_source
    )
    assert 'prepare_strict_gym_child_runtime(scope="scorer-only")' in child_source
    assert "session.finalize_format_verification_calls(" in child_source
    runtime_source = inspect.getsource(runner.run_from_authenticated_wrapper)
    assert (
        "finalize_format_call_evidence=finalize_format_call_evidence" in runtime_source
    )


def test_primary_failure_and_shutdown_poison_force_authenticated_reap(
    runner: Any,
) -> None:
    class Process:
        pid = 4242

        def __init__(self) -> None:
            self.alive = True
            self.reaped = False
            self.kill_calls = 0

        def poll(self) -> int | None:
            return None if self.alive else -2

        def kill(self) -> None:
            self.kill_calls += 1
            self.alive = False

        def wait(self, timeout: float) -> int:
            assert timeout in {0, runner._RUN_HELPER_REAP_TIMEOUT_SECONDS}
            assert not self.alive
            self.reaped = True
            return -2

    class HeadThread:
        def __init__(self) -> None:
            self.alive = True

        def join(self, timeout: float) -> None:
            assert timeout == runner._RUN_HELPER_REAP_TIMEOUT_SECONDS
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    process = Process()
    head_server = types.SimpleNamespace(should_exit=False)
    head_thread = HeadThread()

    class RunHelper:
        _processes = {"citation_format": process}
        _head_server = head_server
        _head_server_thread = head_thread

        @staticmethod
        def shutdown() -> None:
            raise RuntimeError("shutdown poison")

    terminate_calls: list[tuple[int, int]] = []

    def terminate(pid: int, start_ticks: int) -> None:
        terminate_calls.append((pid, start_ticks))
        process.alive = False

    def process_stat(pid: int) -> tuple[int, int]:
        assert pid == process.pid
        if process.alive:
            return 1, 12345
        raise FileNotFoundError(pid)

    child_runtime = types.SimpleNamespace(
        _terminate_authenticated_process=terminate,
        _process_stat=process_stat,
    )
    session = types.SimpleNamespace(
        spec={"targets": [{"config_path": "citation_format"}]}
    )
    run_helper = RunHelper()
    primary = ValueError("primary replay failure")

    def fail_with_cleanup() -> None:
        try:
            raise primary
        finally:
            runner._shutdown_profiled_resource_child(
                run_helper=run_helper,
                session=session,
                child_runtime=child_runtime,
                authenticated_process_identity=(process.pid, 12345),
                primary_failure=sys.exc_info()[1],
            )

    with pytest.raises(ValueError, match="primary replay failure") as raised:
        fail_with_cleanup()
    assert raised.value is primary
    assert terminate_calls == [(process.pid, 12345)]
    assert process.reaped
    assert process.kill_calls == 0
    assert not head_thread.alive
    assert head_server.should_exit is True
    assert run_helper._processes == {}
    assert run_helper._head_server is None
    assert run_helper._head_server_thread is None
    assert any("shutdown poison" in note for note in primary.__notes__)


def test_shutdown_and_unverified_live_child_fail_closed_with_primary(
    runner: Any,
) -> None:
    class Process:
        pid = 5252

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            raise AssertionError("authenticated cleanup must not use Popen.kill")

        @staticmethod
        def wait(timeout: float) -> None:
            assert timeout == runner._RUN_HELPER_REAP_TIMEOUT_SECONDS
            return None

    process = Process()

    class RunHelper:
        _processes = {"freeform_formatting": process}
        _head_server = None
        _head_server_thread = None

        @staticmethod
        def shutdown() -> None:
            raise RuntimeError("shutdown poison")

    child_runtime = types.SimpleNamespace(
        _terminate_authenticated_process=lambda pid, start_ticks: None,
        _process_stat=lambda pid: (1, 9876),
    )
    session = types.SimpleNamespace(
        spec={"targets": [{"config_path": "freeform_formatting"}]}
    )
    primary = ValueError("primary replay failure")

    with pytest.raises(BaseExceptionGroup) as raised:
        runner._shutdown_profiled_resource_child(
            run_helper=RunHelper(),
            session=session,
            child_runtime=child_runtime,
            authenticated_process_identity=(process.pid, 9876),
            primary_failure=primary,
        )
    assert raised.value.exceptions[0] is primary
    assert "shutdown poison" in repr(raised.value.exceptions[1])
    assert "reaped terminal state" in repr(raised.value.exceptions[2])


def test_normal_shutdown_cannot_hide_saved_live_child_identity(runner: Any) -> None:
    class Process:
        pid = 6262

        def __init__(self) -> None:
            self.waited = False

        def poll(self) -> int | None:
            return -15 if self.waited else None

        def kill(self) -> None:
            raise AssertionError("authenticated cleanup must not use Popen.kill")

        def wait(self, timeout: float) -> int:
            assert timeout == runner._RUN_HELPER_REAP_TIMEOUT_SECONDS
            self.waited = True
            return -15

    process = Process()

    class RunHelper:
        def __init__(self) -> None:
            self._processes = {"citation_format": process}
            self._head_server = None
            self._head_server_thread = None

        def shutdown(self) -> None:
            self._processes = {}

    terminate_calls: list[tuple[int, int]] = []

    def terminate(pid: int, start_ticks: int) -> None:
        terminate_calls.append((pid, start_ticks))

    child_runtime = types.SimpleNamespace(
        _terminate_authenticated_process=terminate,
        _process_stat=lambda pid: (1, 13579),
    )
    session = types.SimpleNamespace(
        spec={"targets": [{"config_path": "citation_format"}]}
    )
    primary = ValueError("primary replay failure")
    run_helper = RunHelper()

    with pytest.raises(BaseExceptionGroup) as raised:
        runner._shutdown_profiled_resource_child(
            run_helper=run_helper,
            session=session,
            child_runtime=child_runtime,
            authenticated_process_identity=(process.pid, 13579),
            primary_failure=primary,
        )
    assert raised.value.exceptions[0] is primary
    assert "remained live after authenticated reap" in repr(raised.value.exceptions[1])
    assert terminate_calls == [(process.pid, 13579)]
    assert run_helper._processes == {}


def test_finalizer_failure_after_clearing_run_helper_uses_captured_handles(
    runner: Any,
) -> None:
    class Process:
        pid = 7373

        def __init__(self) -> None:
            self.alive = True
            self.reaped = False

        def poll(self) -> int | None:
            return None if self.alive else -15

        def kill(self) -> None:
            raise AssertionError("authenticated cleanup must not use Popen.kill")

        def wait(self, timeout: float) -> int:
            assert timeout == runner._RUN_HELPER_REAP_TIMEOUT_SECONDS
            assert not self.alive
            self.reaped = True
            return -15

    class HeadThread:
        def __init__(self) -> None:
            self.alive = True

        def join(self, timeout: float) -> None:
            assert timeout == runner._RUN_HELPER_REAP_TIMEOUT_SECONDS
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

    process = Process()
    head_server = types.SimpleNamespace(should_exit=False)
    head_thread = HeadThread()

    class RunHelper:
        _processes: dict[str, Any] = {}
        _head_server = None
        _head_server_thread = None

        @staticmethod
        def shutdown() -> None:
            return None

    terminate_calls: list[tuple[int, int]] = []

    def terminate(pid: int, start_ticks: int) -> None:
        terminate_calls.append((pid, start_ticks))
        process.alive = False

    def process_stat(pid: int) -> tuple[int, int]:
        assert pid == process.pid
        if process.alive:
            return 1, 24680
        raise FileNotFoundError(pid)

    child_runtime = types.SimpleNamespace(
        _terminate_authenticated_process=terminate,
        _process_stat=process_stat,
    )
    session = types.SimpleNamespace(
        spec={"targets": [{"config_path": "citation_format"}]}
    )
    run_helper = RunHelper()
    primary = RuntimeError("finalizer cleared RunHelper fields then failed")

    def fail_with_captured_cleanup() -> None:
        try:
            raise primary
        finally:
            runner._shutdown_profiled_resource_child(
                run_helper=run_helper,
                session=session,
                child_runtime=child_runtime,
                authenticated_process_identity=(process.pid, 24680),
                primary_failure=sys.exc_info()[1],
                captured_resource_process=process,
                captured_head_server=head_server,
                captured_head_server_thread=head_thread,
            )

    with pytest.raises(RuntimeError, match="finalizer cleared") as raised:
        fail_with_captured_cleanup()
    assert raised.value is primary
    assert terminate_calls == [(process.pid, 24680)]
    assert process.reaped
    assert head_server.should_exit is True
    assert not head_thread.alive
    assert run_helper._processes == {}
    assert run_helper._head_server is None
    assert run_helper._head_server_thread is None
