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

import ast
import asyncio
import importlib.machinery
import json
import os
import runpy
import shlex
import socket
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemo_rl.environments import strict_gym_child_runtime_v2 as runtime

BOOTSTRAP = Path(runtime.__file__).parent / "_strict_gym_child_bootstrap_v2" / "sitecustomize.py"


def _bootstrap_module(monkeypatch) -> SimpleNamespace:
    monkeypatch.delenv("NRL_STRICT_GYM_CHILD_SPEC_PATH", raising=False)
    return SimpleNamespace(**runpy.run_path(str(BOOTSTRAP), run_name="_strict_gym_bootstrap_test"))


def test_bootstrap_is_loaded_by_real_python_startup(tmp_path: Path) -> None:
    disabled_pycache = tmp_path / "does-not-exist"
    environment = dict(os.environ)
    environment.pop("NRL_STRICT_GYM_CHILD_SPEC_PATH", None)
    environment.update(
        {
            "PYTHONPATH": str(BOOTSTRAP.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONPYCACHEPREFIX": str(disabled_pycache),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", "import sitecustomize; print(sitecustomize.__file__)"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert Path(completed.stdout.strip()).resolve() == BOOTSTRAP.resolve()
    assert not disabled_pycache.exists()


def test_direct_runner_app_compile_does_not_inherit_future_annotations() -> None:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    app_compile_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compile"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "_strict_gym_app_payload"
    ]

    assert len(app_compile_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in app_compile_calls[0].keywords}
    assert isinstance(keywords.get("dont_inherit"), ast.Constant)
    assert keywords["dont_inherit"].value is True

    # This test module intentionally enables future annotations. The child app
    # does not, so its concrete config annotation must retain its own semantics.
    namespace: dict[str, Any] = {}
    code = compile(
        b"class Config: pass\nclass App:\n    config: Config\n",
        "<strict-direct-runner-app>",
        "exec",
        dont_inherit=True,
    )
    exec(code, namespace)
    assert namespace["App"].__annotations__["config"] is namespace["Config"]

    future_namespace: dict[str, Any] = {}
    future_code = compile(
        b"from __future__ import annotations\n" b"class Config: pass\nclass App:\n    config: Config\n",
        "<strict-direct-runner-future-app>",
        "exec",
        dont_inherit=True,
    )
    exec(future_code, future_namespace)
    assert future_namespace["App"].__annotations__["config"] == "Config"


def test_pidfd_wrappers_use_pinned_linux_syscall_fallback(monkeypatch) -> None:
    opened: list[int] = []
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(runtime.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(runtime.signal, "pidfd_send_signal", None, raising=False)
    monkeypatch.setattr(
        runtime,
        "_linux_pidfd_open_syscall",
        lambda pid: opened.append(pid) or 91,
    )
    monkeypatch.setattr(
        runtime,
        "_linux_pidfd_send_signal_syscall",
        lambda pidfd, signal_number: sent.append((pidfd, signal_number)),
    )

    assert runtime._pidfd_open(123) == 91
    runtime._pidfd_send_signal(91, runtime.signal.SIGINT)
    assert opened == [123]
    assert sent == [(91, runtime.signal.SIGINT)]


def test_authenticated_termination_uses_pidfd_across_escalation(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []
    closed: list[int] = []

    class Poller:
        def __init__(self) -> None:
            self.poll_count = 0

        def register(self, pidfd: int, events: int) -> None:
            assert pidfd == 73
            assert events == (runtime.select.POLLIN | runtime.select.POLLHUP | runtime.select.POLLERR)

        def poll(self, timeout_ms: int) -> list[tuple[int, int]]:
            assert timeout_ms == 5_000
            self.poll_count += 1
            if self.poll_count == 1:
                return []
            return [(73, runtime.select.POLLIN)]

    monkeypatch.setattr(runtime, "_pidfd_open", lambda pid: 73)
    monkeypatch.setattr(
        runtime,
        "_pidfd_send_signal",
        lambda pidfd, signal_number: signals.append((pidfd, signal_number)),
    )
    monkeypatch.setattr(runtime, "_process_stat", lambda pid: (50, 777))
    monkeypatch.setattr(runtime.select, "poll", Poller)
    monkeypatch.setattr(runtime.os, "close", closed.append)

    assert runtime._terminate_authenticated_process(123, 777) == "SIGTERM"
    assert signals == [(73, runtime.signal.SIGINT), (73, runtime.signal.SIGTERM)]
    assert closed == [73]


def _spec_and_target(
    environment: str = "citation",
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = runtime._target_matrix(environment, runtime.STRICT_GYM_ROOT, scope="main")
    spec = runtime._build_spec(
        environment=environment,
        scope="main",
        pair_id="strict-pair-1",
        job_id="12345",
        results_dir=Path("/strict/results/off"),
        receipt_root=Path("/strict/results/off/strict_gym_child_runtime"),
        bootstrap_root=Path("/opt/nemo-rl/nemo_rl/environments/bootstrap"),
        bootstrap_sha256="a" * 64,
        targets=targets,
    )
    return spec, targets[0]


def _receipt(
    spec: dict[str, Any], target: dict[str, Any], *, num_workers: int | None = None
) -> tuple[dict[str, Any], SimpleNamespace]:
    boot_id = "12345678-1234-1234-1234-123456789abc"
    instance = SimpleNamespace(
        config_path=target["config_path"],
        server_type=target["server_type"],
        name=target["server_name"],
        entrypoint=target["entrypoint"],
        host="127.0.0.1",
        port=5123,
    )
    target_record = {
        name: target[name]
        for name in (
            "role",
            "config_path",
            "server_type",
            "server_name",
            "component_dir",
            "entrypoint",
            "source_path",
            "source_sha256",
            "config_path_source",
            "config_sha256",
            "requirements_path",
            "requirements_sha256",
            "venv",
            "interpreter",
        )
    }
    scorer = target["scorer"]
    if scorer is not None:
        scorer = dict(scorer)
        purelib = Path(target["venv"]) / "lib/python3.13/site-packages"
        scorer.update(
            {
                "package_root": str(purelib / "reasoning_gym"),
                "package_resolved_root": str(purelib / "reasoning_gym"),
                "module_origin": str(purelib / scorer["module_origin_relative_to_purelib"]),
                "module_resolved_origin": str(purelib / scorer["module_origin_relative_to_purelib"]),
                "resolver_origin": str(purelib / scorer["resolver_origin_relative_to_purelib"]),
                "resolver_resolved_origin": str(purelib / scorer["resolver_origin_relative_to_purelib"]),
                "origin": str(purelib / scorer["origin_relative_to_purelib"]),
                "resolved_origin": str(purelib / scorer["origin_relative_to_purelib"]),
            }
        )
    direct = spec["scope"] == "scorer-only"
    bootstrap_source = str(Path(spec["bootstrap"]["root"]) / spec["bootstrap"]["filename"])
    receipt = {
        "schema": runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
        "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": spec["environment"],
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "stage": ("isolated-runner-pre-entrypoint" if direct else "sitecustomize-pre-entrypoint"),
        "spec_sha256": runtime._sha256_bytes(runtime.canonical_ascii_json(spec)),
        "target": target_record,
        "server": {
            "config_path": target["config_path"],
            "server_type": target["server_type"],
            "server_name": target["server_name"],
            "entrypoint": target["entrypoint"],
            "host": "127.0.0.1",
            "port": 5123,
            "num_workers": num_workers,
        },
        "process": {
            "pid": 123,
            "ppid": 50,
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "cwd": target["component_dir"],
            "sys_executable": target["interpreter"],
            "sys_prefix": target["venv"],
            "sys_base_prefix": "/usr/local",
            "proc_exe": "/usr/local/bin/python3.13",
            "sys_argv": ([bootstrap_source, target["source_path"]] if direct else [target["entrypoint"]]),
            "proc_argv": (
                [
                    target["interpreter"],
                    "-I",
                    "-S",
                    "-B",
                    bootstrap_source,
                    target["source_path"],
                ]
                if direct
                else ["python", target["entrypoint"]]
            ),
            "start_ticks": 777,
            "boot_id": boot_id,
            "hostname": socket.gethostname(),
        },
        "distribution_versions": target["distribution_versions"],
        "module_versions": target["module_versions"],
        "scorer": scorer,
    }
    return receipt, instance


def _patch_live_process(monkeypatch, target: dict[str, Any]) -> None:
    monkeypatch.setattr(runtime, "_boot_id", lambda: "12345678-1234-1234-1234-123456789abc")
    monkeypatch.setattr(runtime, "_process_stat", lambda pid: (50, 777))
    monkeypatch.setattr(runtime, "_process_is_descendant", lambda pid, ancestor: True)
    monkeypatch.setattr(runtime, "_process_descendant_identities", lambda pid: [])
    monkeypatch.setattr(runtime, "_process_argv", lambda pid: ["python", target["entrypoint"]])
    monkeypatch.setattr(runtime, "_listening_socket_inodes", lambda pid, host, port: [88])
    monkeypatch.setattr(
        runtime.os,
        "readlink",
        lambda path: ("/usr/local/bin/python3.13" if str(path).endswith("/exe") else target["component_dir"]),
    )


def test_descendant_scan_includes_children_of_nonleader_threads(monkeypatch, tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"

    def add_process(pid: int, tasks: dict[int, bytes]) -> None:
        process_root = proc_root / str(pid)
        (process_root / "task").mkdir(parents=True)
        (process_root / "status").write_bytes(f"Name:\ttest\nTgid:\t{pid}\n".encode())
        for task, children in tasks.items():
            task_root = process_root / "task" / str(task)
            task_root.mkdir()
            (task_root / "status").write_bytes(f"Name:\ttest\nTgid:\t{pid}\n".encode())
            (task_root / "children").write_bytes(children)

    add_process(100, {100: b"", 101: b"200\n"})
    add_process(200, {200: b""})

    def process_stat(pid: int, *, proc_root: Path) -> tuple[int, int]:
        assert proc_root == tmp_path / "proc"
        assert pid == 200
        return 100, 777

    monkeypatch.setattr(runtime, "_process_stat", process_stat)

    assert runtime._process_descendant_identities(100, proc_root=proc_root) == [(200, 777)]


def _write_immutable(path: Path, document: dict[str, Any]) -> bytes:
    payload = runtime.canonical_ascii_json(document)
    path.write_bytes(payload)
    path.chmod(0o400)
    return payload


def _score_finalizer_fixture(monkeypatch, tmp_path: Path) -> tuple[
    runtime.StrictGymChildRuntimeSession,
    list[dict[str, Any]],
    list[dict[str, Any]],
    SimpleNamespace,
]:
    tmp_path.chmod(0o700)
    gym_root = tmp_path.parent / f"{tmp_path.name}-gym"
    component_dir = gym_root / "resources_servers/reasoning_gym"
    component_dir.mkdir(parents=True)
    monkeypatch.setattr(runtime, "STRICT_GYM_ROOT", gym_root)
    monkeypatch.setattr(runtime, "STRICT_GYM_VENV_ROOT", tmp_path.parent / f"{tmp_path.name}-venvs")
    targets = runtime._target_matrix("reasoning_gym", runtime.STRICT_GYM_ROOT, scope="scorer-only")
    target = targets[0]
    scorer = target["scorer"]
    assert scorer is not None
    purelib = Path(target["venv"]) / "lib/python3.13/site-packages"
    (purelib / "reasoning_gym").mkdir(parents=True)
    for relative_name in (
        scorer["module_origin_relative_to_purelib"],
        scorer["resolver_origin_relative_to_purelib"],
        scorer["origin_relative_to_purelib"],
    ):
        scorer_path = purelib / relative_name
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        scorer_path.touch()
    spec = runtime._build_spec(
        environment="reasoning_gym",
        scope="scorer-only",
        pair_id="strict-pair-1",
        job_id="12345",
        results_dir=tmp_path.parent,
        receipt_root=tmp_path,
        bootstrap_root=Path("/opt/nemo-rl/nemo_rl/environments/bootstrap"),
        bootstrap_sha256="a" * 64,
        targets=targets,
    )
    monkeypatch.setattr(
        runtime,
        "_require_sealed_bootstrap_root",
        lambda: (
            Path(spec["bootstrap"]["root"]),
            spec["bootstrap"]["sha256"],
        ),
    )
    spec_payload = _write_immutable(tmp_path / "spec.json", spec)
    receipt, _ = _receipt(spec, target)
    resource_payload = _write_immutable(tmp_path / "resource.json", receipt)
    observation = {
        "pid": 123,
        "start_ticks": 777,
        "wrapper_pid": 123,
        "host": "127.0.0.1",
        "port": 5123,
        "listener_socket_inodes": [88],
    }
    child_index = {
        "schema": runtime.STRICT_GYM_CHILD_INDEX_SCHEMA,
        "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "scope": "scorer-only",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "gym": spec["gym"],
        "spec": {
            "path": str(tmp_path / "spec.json"),
            "sha256": runtime._sha256_bytes(spec_payload),
            "schema": runtime.STRICT_GYM_CHILD_SPEC_SCHEMA,
        },
        "children": [
            {
                "role": "resource",
                "config_path": "reasoning_gym",
                "receipt": {
                    "path": str(tmp_path / "resource.json"),
                    "sha256": runtime._sha256_bytes(resource_payload),
                    "schema": runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
                },
                "observation": observation,
            }
        ],
    }
    child_index_payload = _write_immutable(tmp_path / "index.json", child_index)
    expected_calls: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    call_refs: list[dict[str, Any]] = []
    for sequence in range(1, 5):
        answer = f"answer-{sequence}"
        entry = {
            "question": f"question-{sequence}",
            "metadata": {"source_dataset": "knights_knaves"},
        }
        reward = float(sequence % 2)
        expected = runtime.reasoning_score_call_expectation(
            task_name="knights_knaves",
            answer=answer,
            entry=entry,
            float_result=reward,
        )
        expected_calls.append(expected)
        document = {
            "schema": runtime.STRICT_GYM_SCORE_CALL_SCHEMA,
            "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": "reasoning_gym",
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec_sha256": runtime._sha256_bytes(spec_payload),
            "process": {"pid": 123, "start_ticks": 777},
            "sequence": sequence,
            "task_name": "knights_knaves",
            "input": {
                "answer_sha256": expected["answer_sha256"],
                "entry_sha256": expected["entry_sha256"],
            },
            "outcome": {"kind": "returned", "float_result": reward},
        }
        documents.append(document)
        call_path = tmp_path / f"reasoning-score-call-{sequence:08d}.json"
        call_payload = _write_immutable(call_path, document)
        call_refs.append(
            {
                "sequence": sequence,
                "path": str(call_path),
                "sha256": runtime._sha256_bytes(call_payload),
                "schema": runtime.STRICT_GYM_SCORE_CALL_SCHEMA,
            }
        )
    closed = {
        "schema": runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA,
        "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "spec_sha256": runtime._sha256_bytes(spec_payload),
        "process": {"pid": 123, "start_ticks": 777},
        "call_count": 4,
        "calls": call_refs,
    }
    _write_immutable(tmp_path / "reasoning-score-closed.json", closed)
    state = {"running": True}

    def process_stat(pid: int) -> tuple[int, int]:
        if not state["running"]:
            raise ProcessLookupError(pid)
        return 50, 777

    monkeypatch.setattr(runtime, "_boot_id", lambda: "12345678-1234-1234-1234-123456789abc")
    monkeypatch.setattr(runtime, "_process_stat", process_stat)
    monkeypatch.setattr(runtime, "_process_is_descendant", lambda pid, ancestor: True)
    monkeypatch.setattr(runtime, "_process_descendant_identities", lambda pid: [])
    monkeypatch.setattr(runtime, "_process_argv", lambda pid: ["python", target["entrypoint"]])
    monkeypatch.setattr(runtime, "_listening_socket_inodes", lambda pid, host, port: [88])
    monkeypatch.setattr(
        runtime,
        "_validate_receipt",
        lambda document, spec, target, instance, wrapper_pid: observation,
    )
    monkeypatch.setattr(
        runtime,
        "_terminate_authenticated_process",
        lambda pid, start_ticks: state.update(running=False) or "SIGINT",
    )
    wrapper = SimpleNamespace(pid=123, returncode=None)
    wrapper.poll = lambda: wrapper.returncode
    instance = SimpleNamespace(
        config_path=target["config_path"],
        server_type=target["server_type"],
        name=target["server_name"],
        entrypoint=target["entrypoint"],
        host="127.0.0.1",
        port=5123,
        dir_path=target["component_dir"],
    )
    run_helper = SimpleNamespace(
        _server_instance_display_configs=[instance],
        _processes={target["config_path"]: wrapper},
    )

    def shutdown() -> None:
        state["running"] = False
        wrapper.returncode = -2
        run_helper._processes = {}

    run_helper.shutdown = shutdown
    session = runtime.StrictGymChildRuntimeSession(
        environment="reasoning_gym",
        scope="scorer-only",
        receipt_root=tmp_path,
        spec_path=tmp_path / "spec.json",
        spec_sha256=runtime._sha256_bytes(spec_payload),
        bootstrap_root=Path(spec["bootstrap"]["root"]),
        bootstrap_sha256=spec["bootstrap"]["sha256"],
        spec=spec,
    )
    object.__setattr__(session, "_started_index", child_index)
    object.__setattr__(session, "_started_index_sha256", runtime._sha256_bytes(child_index_payload))
    return session, expected_calls, documents, run_helper


def _format_finalizer_fixture(monkeypatch, tmp_path: Path) -> tuple[
    runtime.StrictGymChildRuntimeSession,
    list[dict[str, Any]],
    list[dict[str, Any]],
    SimpleNamespace,
]:
    tmp_path.chmod(0o700)
    gym_root = tmp_path.parent / f"{tmp_path.name}-format-gym"
    (gym_root / "resources_servers/format_verification").mkdir(parents=True)
    monkeypatch.setattr(runtime, "STRICT_GYM_ROOT", gym_root)
    monkeypatch.setattr(
        runtime,
        "STRICT_GYM_VENV_ROOT",
        tmp_path.parent / f"{tmp_path.name}-format-venvs",
    )
    targets = runtime._target_matrix("citation", runtime.STRICT_GYM_ROOT, scope="scorer-only")
    target = targets[0]
    assert target["scorer"] is None
    spec = runtime._build_spec(
        environment="citation",
        scope="scorer-only",
        pair_id="strict-pair-1",
        job_id="12345",
        results_dir=tmp_path.parent,
        receipt_root=tmp_path,
        bootstrap_root=Path("/opt/nemo-rl/nemo_rl/environments/bootstrap"),
        bootstrap_sha256="a" * 64,
        targets=targets,
    )
    monkeypatch.setattr(
        runtime,
        "_require_sealed_bootstrap_root",
        lambda: (Path(spec["bootstrap"]["root"]), spec["bootstrap"]["sha256"]),
    )
    spec_payload = _write_immutable(tmp_path / "spec.json", spec)
    receipt, _ = _receipt(spec, target)
    resource_payload = _write_immutable(tmp_path / "resource.json", receipt)
    observation = {
        "pid": 123,
        "start_ticks": 777,
        "wrapper_pid": 123,
        "host": "127.0.0.1",
        "port": 5123,
        "listener_socket_inodes": [88],
    }
    child_index = {
        "schema": runtime.STRICT_GYM_CHILD_INDEX_SCHEMA,
        "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "citation",
        "scope": "scorer-only",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "gym": spec["gym"],
        "spec": {
            "path": str(tmp_path / "spec.json"),
            "sha256": runtime._sha256_bytes(spec_payload),
            "schema": runtime.STRICT_GYM_CHILD_SPEC_SCHEMA,
        },
        "children": [
            {
                "role": "resource",
                "config_path": target["config_path"],
                "receipt": {
                    "path": str(tmp_path / "resource.json"),
                    "sha256": runtime._sha256_bytes(resource_payload),
                    "schema": runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
                },
                "observation": observation,
            }
        ],
    }
    child_index_payload = _write_immutable(tmp_path / "index.json", child_index)
    expected_calls: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    call_refs: list[dict[str, Any]] = []
    for sequence in range(1, 5):
        marker_present = sequence % 2 == 1
        text = "answer [1]" if marker_present else "answer"
        request = {
            "responses_create_params": {"model": "policy"},
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ]
            },
            "verifier": {
                "type": "string_match",
                "expected_markers": ["[1]"],
                "patterns": [r"\[[0-9]+\]"],
            },
        }
        details = {
            "expected": ["[1]"],
            "missing": [] if marker_present else ["[1]"],
            "spurious": [],
            "passed": marker_present,
        }
        response = {
            **request,
            "reward": 1.0 if marker_present else 0.0,
            "match_details": details,
        }
        expected = runtime.format_verification_call_expectation(
            environment="citation",
            derived_verifier_request=request,
            verifier_response=response,
        )
        expected_calls.append(expected)
        document = {
            "schema": runtime.STRICT_GYM_FORMAT_CALL_SCHEMA,
            "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": "citation",
            "profile_id": "citation-string-match-v1",
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec_sha256": runtime._sha256_bytes(spec_payload),
            "process": {"pid": 123, "start_ticks": 777},
            "sequence": sequence,
            "method": "_verify_string_match",
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
        }
        documents.append(document)
        call_path = tmp_path / f"format-verification-call-{sequence:08d}.json"
        call_payload = _write_immutable(call_path, document)
        call_refs.append(
            {
                "sequence": sequence,
                "path": str(call_path),
                "sha256": runtime._sha256_bytes(call_payload),
                "schema": runtime.STRICT_GYM_FORMAT_CALL_SCHEMA,
            }
        )
    closed = {
        "schema": runtime.STRICT_GYM_FORMAT_CLOSED_SCHEMA,
        "hash_domain": runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "citation",
        "profile_id": "citation-string-match-v1",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "spec_sha256": runtime._sha256_bytes(spec_payload),
        "process": {"pid": 123, "start_ticks": 777},
        "call_count": 4,
        "calls": call_refs,
    }
    _write_immutable(tmp_path / "format-verification-closed.json", closed)
    state = {"running": True}

    def process_stat(pid: int) -> tuple[int, int]:
        if not state["running"]:
            raise ProcessLookupError(pid)
        return 50, 777

    monkeypatch.setattr(runtime, "_boot_id", lambda: "12345678-1234-1234-1234-123456789abc")
    monkeypatch.setattr(runtime, "_process_stat", process_stat)
    monkeypatch.setattr(runtime, "_process_is_descendant", lambda pid, ancestor: True)
    monkeypatch.setattr(runtime, "_process_descendant_identities", lambda pid: [])
    monkeypatch.setattr(runtime, "_listening_socket_inodes", lambda pid, host, port: [88])
    monkeypatch.setattr(
        runtime,
        "_validate_receipt",
        lambda document, spec, target, instance, wrapper_pid: observation,
    )
    monkeypatch.setattr(
        runtime,
        "_terminate_authenticated_process",
        lambda pid, start_ticks: state.update(running=False) or "SIGINT",
    )
    wrapper = SimpleNamespace(pid=123, returncode=None)
    wrapper.poll = lambda: wrapper.returncode
    instance = SimpleNamespace(
        config_path=target["config_path"],
        server_type=target["server_type"],
        name=target["server_name"],
        entrypoint=target["entrypoint"],
        host="127.0.0.1",
        port=5123,
        dir_path=target["component_dir"],
    )
    run_helper = SimpleNamespace(
        _server_instance_display_configs=[instance],
        _processes={target["config_path"]: wrapper},
    )

    def shutdown() -> None:
        state["running"] = False
        wrapper.returncode = -2
        run_helper._processes = {}

    run_helper.shutdown = shutdown
    session = runtime.StrictGymChildRuntimeSession(
        environment="citation",
        scope="scorer-only",
        receipt_root=tmp_path,
        spec_path=tmp_path / "spec.json",
        spec_sha256=runtime._sha256_bytes(spec_payload),
        bootstrap_root=Path(spec["bootstrap"]["root"]),
        bootstrap_sha256=spec["bootstrap"]["sha256"],
        spec=spec,
    )
    object.__setattr__(session, "_started_index", child_index)
    object.__setattr__(
        session,
        "_started_index_sha256",
        runtime._sha256_bytes(child_index_payload),
    )
    return session, expected_calls, documents, run_helper


def _format_payload_roster(root: Path) -> tuple[tuple[str, bytes], ...]:
    filenames = (
        "index.json",
        *(f"format-verification-call-{sequence:08d}.json" for sequence in range(1, 5)),
        "format-verification-call-index.json",
        "format-verification-closed.json",
        "resource.json",
        "spec.json",
    )
    return tuple((f"strict_gym_child_runtime/{filename}", (root / filename).read_bytes()) for filename in filenames)


@pytest.mark.parametrize("environment", ["reasoning_gym", "citation", "freeform"])
def test_target_selection_main_and_scorer_only(environment: str) -> None:
    main = runtime._target_matrix(environment, runtime.STRICT_GYM_ROOT, scope="main")
    scorer_only = runtime._target_matrix(environment, runtime.STRICT_GYM_ROOT, scope="scorer-only")

    assert [target["role"] for target in main] == ["resource", "simple_agent"]
    assert [target["role"] for target in scorer_only] == ["resource"]
    if environment == "reasoning_gym":
        assert main[0]["config_path_source"].endswith("/configs/reasoning_gym.yaml")
        assert scorer_only[0]["config_path_source"].endswith("/configs/resources_only.yaml")
        assert scorer_only[0]["config_sha256"] == ("e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68")
    else:
        assert scorer_only == [main[0]]
    assert main[0]["component_dir"].startswith(f"{runtime.STRICT_GYM_ROOT}/")
    assert main[1]["interpreter"] == ("/opt/gym_venvs/responses_api_agents/simple_agent/.venv/bin/python")
    assert main[1]["distribution_versions"] == {
        "nemo-gym": "0.5.0rc0",
        "openai": "2.6.1",
        "pydantic": "2.13.4",
        "pydantic-core": "2.46.4",
    }


def test_reasoning_metadata_and_module_literals_are_separate() -> None:
    target = runtime._target_matrix("reasoning_gym", runtime.STRICT_GYM_ROOT, scope="scorer-only")[0]

    assert target["distribution_versions"]["nemo-gym"] == "0.5.0rc0"
    assert target["module_versions"]["nemo_gym"] == "0.5.1"
    assert target["distribution_versions"]["ray"] == "2.56.1"
    assert target["module_versions"]["ray"] == "2.56.1"
    assert target["distribution_versions"]["reasoning-gym"] == "0.1.25"
    assert target["module_versions"]["reasoning_gym"] == "0.1.19"
    assert target["scorer"]["callable"].endswith("KnightsKnavesDataset.score_answer")


def test_canonical_artifact_round_trip_and_noncanonical_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_bytes(runtime.canonical_ascii_json({"b": 2, "a": 1}))
    path.chmod(0o400)

    value, payload = runtime._load_canonical_document(path)
    assert value == {"a": 1, "b": 2}
    assert payload == b'{"a":1,"b":2}'

    path.chmod(0o600)
    path.write_text('{ "a": 1, "b": 2 }', encoding="ascii")
    path.chmod(0o400)
    with pytest.raises(ValueError, match="not canonical JSON"):
        runtime._load_canonical_document(path)


def test_canonical_artifact_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    target.chmod(0o400)
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        runtime._load_canonical_document(link)


def test_launch_environment_is_exclusive_and_restores_parent(monkeypatch, tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    session = runtime.StrictGymChildRuntimeSession(
        environment="citation",
        scope="main",
        receipt_root=tmp_path,
        spec_path=spec_path,
        spec_sha256="a" * 64,
        bootstrap_root=tmp_path / "bootstrap",
        bootstrap_sha256="b" * 64,
        spec={},
    )
    monkeypatch.setenv("PYTHONPATH", "/ambient/path")
    monkeypatch.delenv("NRL_STRICT_GYM_CHILD_SPEC_PATH", raising=False)

    with session.launch_environment():
        assert os.environ["PYTHONPATH"] == str(session.bootstrap_root)
        assert os.environ["NRL_STRICT_GYM_CHILD_SPEC_PATH"] == str(spec_path)
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        assert os.environ["PYTHONNOUSERSITE"] == "1"
        assert os.environ["PYTHONSAFEPATH"] == "1"
        assert os.environ["PYTHONPYCACHEPREFIX"].endswith("/__pycache_disabled__")

    assert os.environ["PYTHONPATH"] == "/ambient/path"
    assert "NRL_STRICT_GYM_CHILD_SPEC_PATH" not in os.environ


def test_scorer_only_launch_rejects_ray_initialization_inside_hook(monkeypatch, tmp_path: Path) -> None:
    session = runtime.StrictGymChildRuntimeSession(
        environment="reasoning_gym",
        scope="scorer-only",
        receipt_root=tmp_path,
        spec_path=tmp_path / "spec.json",
        spec_sha256="a" * 64,
        bootstrap_root=tmp_path,
        bootstrap_sha256="b" * 64,
        spec={},
    )
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(is_initialized=lambda: False))

    with pytest.raises(RuntimeError, match="Ray initialized before hook injection"):
        with session.launch_environment():
            pytest.fail("launch context must not be entered")


def test_scorer_only_launch_uses_exact_isolated_process_and_restores_globals(monkeypatch, tmp_path: Path) -> None:
    component = tmp_path / "component"
    component.mkdir()
    target = {
        "config_path": "reasoning_gym",
        "component_dir": str(component),
        "entrypoint": "app.py",
        "source_path": str(component / "app.py"),
        "venv": str(tmp_path / "venv"),
        "interpreter": str(tmp_path / "venv/bin/python"),
    }
    session = runtime.StrictGymChildRuntimeSession(
        environment="reasoning_gym",
        scope="scorer-only",
        receipt_root=tmp_path,
        spec_path=tmp_path / "spec.json",
        spec_sha256="a" * 64,
        bootstrap_root=tmp_path / "bootstrap",
        bootstrap_sha256="b" * 64,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "results_dir": str(tmp_path),
            "targets": [target],
        },
    )
    ray_module = ModuleType("ray")
    ray_module.is_initialized = lambda: True
    gym_package = ModuleType("nemo_gym")
    gym_package.__path__ = []
    cli_package = ModuleType("nemo_gym.cli")
    cli_package.__path__ = []
    env_module = ModuleType("nemo_gym.cli.env")
    setup_module = ModuleType("nemo_gym.cli.setup_command")

    def original_setup(*args, **kwargs):
        raise AssertionError("original setup must be replaced")

    def original_run(*args, **kwargs):
        raise AssertionError("original run must be replaced")

    env_module.setup_env_command = original_setup
    env_module.run_command = original_run
    setup_module.setup_env_command = original_setup
    setup_module.run_command = original_run
    setup_module.get_venv_path = lambda directory, config: Path(target["venv"])
    omega_module = ModuleType("omegaconf")
    omega_module.OmegaConf = SimpleNamespace(to_yaml=lambda config: "dry_run: false\nskip_venv_if_present: true\n")
    gym_package.cli = cli_package
    cli_package.env = env_module
    cli_package.setup_command = setup_module
    for name, module in {
        "ray": ray_module,
        "nemo_gym": gym_package,
        "nemo_gym.cli": cli_package,
        "nemo_gym.cli.env": env_module,
        "nemo_gym.cli.setup_command": setup_module,
        "omegaconf": omega_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    captured: dict[str, Any] = {}

    def fake_popen(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(pid=321)

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("PYTHONWARNINGS", "error::RuntimeWarning")
    config = {
        "skip_venv_if_present": True,
        "dry_run": False,
        "uv_venv_dir": str(runtime.STRICT_GYM_VENV_ROOT),
    }
    yaml = omega_module.OmegaConf.to_yaml(config)
    sentinel = "NRL_STRICT_DIRECT_SCORER_LAUNCH_V1"
    command = (
        f"{sentinel} \\\n"
        f"    && NEMO_GYM_CONFIG_DICT={shlex.quote(yaml)} \\\n"
        "    NEMO_GYM_CONFIG_PATH=reasoning_gym \\\n"
        "    python app.py"
    )

    with session.launch_environment():
        assert env_module.setup_env_command(component, config, "reasoning_gym") == sentinel
        process = env_module.run_command(command, component, server_name="reasoning_gym")
        assert process.pid == 321

    assert env_module.setup_env_command is original_setup
    assert env_module.run_command is original_run
    assert captured["argv"] == [
        target["interpreter"],
        "-I",
        "-S",
        "-B",
        str(session.bootstrap_root / "sitecustomize.py"),
        target["source_path"],
    ]
    assert captured["cwd"] == component
    assert captured["stdout"] is sys.stdout
    assert captured["stderr"] is sys.stderr
    assert "PYTHONWARNINGS" not in captured["env"]
    assert "PYTHONPATH" not in captured["env"]
    assert captured["env"]["NRL_STRICT_GYM_DIRECT_RUNNER"] == "1"
    assert captured["env"]["PATH"] == "/usr/bin:/bin"


def test_receipt_joins_live_child_wrapper_and_listener(monkeypatch) -> None:
    spec, target = _spec_and_target()
    receipt, instance = _receipt(spec, target)
    _patch_live_process(monkeypatch, target)

    observed = runtime._validate_receipt(
        receipt,
        spec=spec,
        target=target,
        instance=instance,
        wrapper_pid=40,
    )

    assert observed == {
        "pid": 123,
        "start_ticks": 777,
        "wrapper_pid": 40,
        "host": "127.0.0.1",
        "port": 5123,
        "listener_socket_inodes": [88],
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda receipt: receipt["server"].update(num_workers=True), "one uvicorn"),
        (lambda receipt: receipt["process"].update(uid=-1), "UID differs"),
        (
            lambda receipt: receipt["process"].update(start_ticks=778),
            "PID was reused",
        ),
    ],
)
def test_receipt_rejects_hostile_process_claims(monkeypatch, mutation, match) -> None:
    spec, target = _spec_and_target()
    receipt, instance = _receipt(spec, target)
    _patch_live_process(monkeypatch, target)
    mutation(receipt)

    with pytest.raises(ValueError, match=match):
        runtime._validate_receipt(
            receipt,
            spec=spec,
            target=target,
            instance=instance,
            wrapper_pid=40,
        )


def test_receipt_rejects_process_without_selected_listener(monkeypatch) -> None:
    spec, target = _spec_and_target()
    receipt, instance = _receipt(spec, target)
    _patch_live_process(monkeypatch, target)
    monkeypatch.setattr(runtime, "_listening_socket_inodes", lambda pid, host, port: [])

    with pytest.raises(ValueError, match="does not own"):
        runtime._validate_receipt(
            receipt,
            spec=spec,
            target=target,
            instance=instance,
            wrapper_pid=40,
        )


def test_exact_root_rejects_foreign_same_suffix(tmp_path: Path) -> None:
    hostile = tmp_path / "responses_api_agents" / "simple_agent"
    hostile.mkdir(parents=True)

    with pytest.raises(ValueError, match="authenticated deployment mount"):
        runtime._validate_pinned_gym_root(hostile, [], scope="scorer-only")


def test_scorer_no_site_accepts_only_exact_inert_q_image_pth_inventory(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    purelib.chmod(0o755)

    runtime._validate_purelib_pth_inventory(purelib, scope="main")
    bootstrap._validate_scorer_no_site_pth_inventory(purelib)
    purelib.chmod(0o700)
    with pytest.raises(ValueError, match="purelib identity differs"):
        runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    with pytest.raises(ValueError, match="purelib identity differs"):
        bootstrap._validate_scorer_no_site_pth_inventory(purelib)
    purelib.chmod(0o755)
    archive = tmp_path / "archive"
    archive.mkdir()
    editable_target = archive / "editable.pth"
    editable_payload = b"import exact_editable"
    editable_target.write_bytes(editable_payload)
    editable_target.chmod(0o600)
    coverage_target = archive / "coverage.pth"
    coverage_payload = b"import exact_coverage\n"
    coverage_target.write_bytes(coverage_payload)
    coverage_target.chmod(0o600)
    virtualenv_payload = b"import _virtualenv"
    inventory = {
        "__editable__.nemo_gym-0.5.0rc0.pth": {
            "kind": "symlink",
            "link_target": str(editable_target),
            "size": len(editable_payload),
            "sha256": runtime._sha256_bytes(editable_payload),
        },
        "_virtualenv.pth": {
            "kind": "regular",
            "size": len(virtualenv_payload),
            "sha256": runtime._sha256_bytes(virtualenv_payload),
        },
        "a1_coverage.pth": {
            "kind": "symlink",
            "link_target": str(coverage_target),
            "size": len(coverage_payload),
            "sha256": runtime._sha256_bytes(coverage_payload),
        },
    }
    monkeypatch.setattr(runtime, "_SCORER_NO_SITE_PTH_FILES", inventory)
    monkeypatch.setitem(
        bootstrap._validate_scorer_no_site_pth_inventory.__globals__,
        "_SCORER_NO_SITE_PTH_FILES",
        inventory,
    )
    editable_hook = purelib / "__editable__.nemo_gym-0.5.0rc0.pth"
    editable_hook.symlink_to(editable_target)
    virtualenv_hook = purelib / "_virtualenv.pth"
    virtualenv_hook.write_bytes(virtualenv_payload)
    virtualenv_hook.chmod(0o600)
    (purelib / "a1_coverage.pth").symlink_to(coverage_target)
    runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    bootstrap._validate_scorer_no_site_pth_inventory(purelib)

    virtualenv_hook.chmod(0o644)
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        bootstrap._validate_scorer_no_site_pth_inventory(purelib)
    virtualenv_hook.chmod(0o600)

    with pytest.raises(ValueError, match="untrusted pre-sitecustomize"):
        runtime._validate_purelib_pth_inventory(purelib, scope="main")

    editable_hook.unlink()
    editable_hook.write_bytes(editable_payload)
    editable_hook.chmod(0o600)
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        bootstrap._validate_scorer_no_site_pth_inventory(purelib)

    editable_hook.unlink()
    duplicate_target = archive / "duplicate-editable.pth"
    duplicate_target.write_bytes(editable_payload)
    duplicate_target.chmod(0o600)
    editable_hook.symlink_to(duplicate_target)
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        bootstrap._validate_scorer_no_site_pth_inventory(purelib)

    editable_hook.unlink()
    editable_hook.symlink_to(editable_target)
    coverage_target.write_bytes(b"import hostile\n")
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        runtime._validate_purelib_pth_inventory(purelib, scope="scorer-only")
    with pytest.raises(ValueError, match=r"\.pth identity differs"):
        bootstrap._validate_scorer_no_site_pth_inventory(purelib)


def test_bootstrap_rejects_tampered_static_target(monkeypatch) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    spec, target = _spec_and_target()
    target = json.loads(json.dumps(target))
    target["source_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="source_sha256"):
        bootstrap._validate_target(spec, target)


def test_direct_bootstrap_removes_and_locks_hostile_component_import_path(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib = tmp_path / "purelib"
    component = tmp_path / "component"
    gym_root = tmp_path / "Gym"
    for directory in (purelib, component, gym_root):
        directory.mkdir()
    (component / "hostile_component_module.py").write_text("raise RuntimeError('must not import')\n", encoding="ascii")
    nemo_gym = ModuleType("nemo_gym")
    nemo_gym._augment_sys_path = lambda: None
    monkeypatch.setitem(bootstrap._seal_direct_runner_sys_path.__globals__, "_GYM_ROOT", gym_root)
    original_path = list(sys.path)
    isolated = ["/isolated/stdlib", "/isolated/lib-dynload"]
    try:
        sys.path[:] = [*isolated, str(purelib), str(component), str(gym_root)]
        safe_path = bootstrap._seal_direct_runner_sys_path(isolated, purelib, component, nemo_gym)

        assert sys.path == [*isolated, str(purelib), str(gym_root)]
        assert safe_path == tuple(sys.path)
        assert importlib.machinery.PathFinder.find_spec("hostile_component_module", sys.path) is None
        sys.path.append(str(component))
        with pytest.raises(ValueError, match="changed after authentication"):
            nemo_gym._augment_sys_path()
    finally:
        sys.path[:] = original_path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda paths: paths[1:],
        lambda paths: [*paths, paths[0]],
        lambda paths: [paths[1], paths[0], *paths[2:]],
        lambda paths: ["/hostile", *paths],
    ],
)
def test_direct_bootstrap_rejects_changed_ray_thirdparty_path(monkeypatch, tmp_path: Path, mutate) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib = tmp_path / "purelib"
    gym_root = tmp_path / "Gym"
    ray_thirdparty = purelib / "ray/thirdparty_files"
    ray_archive = tmp_path / "ray-archive"
    for directory in (gym_root, ray_thirdparty, ray_archive):
        directory.mkdir(parents=True)
    ray_init = ray_archive / "__init__.py"
    ray_init.write_text("# ray\n", encoding="ascii")
    ray_module = ModuleType("ray")
    ray_module.__file__ = str(ray_init)
    ray_module.__version__ = bootstrap._SCORER_RAY_VERSION
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(
        bootstrap._seal_direct_runner_ray_sys_path.__globals__,
        "_SCORER_RAY_ARCHIVE_ROOT",
        ray_archive,
    )
    isolated = ["/isolated/stdlib", "/isolated/lib-dynload"]
    safe_path = (*isolated, str(purelib), str(gym_root))
    ray_inputs = {"root": purelib / "ray", "thirdparty": ray_thirdparty, "sources": {}}
    monkeypatch.setitem(
        bootstrap._seal_direct_runner_ray_sys_path.__globals__,
        "_authenticate_direct_runner_ray_inputs",
        lambda selected_purelib: ray_inputs,
    )
    expected = [str(ray_thirdparty), *safe_path]
    original_path = list(sys.path)
    try:
        sys.path[:] = mutate(expected)
        with pytest.raises(ValueError, match="Ray changed the isolated scorer import path"):
            bootstrap._seal_direct_runner_ray_sys_path(safe_path, purelib, ray_inputs)
    finally:
        sys.path[:] = original_path


def test_direct_bootstrap_strips_exact_ray_thirdparty_path(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib = tmp_path / "purelib"
    gym_root = tmp_path / "Gym"
    ray_thirdparty = purelib / "ray/thirdparty_files"
    ray_archive = tmp_path / "ray-archive"
    for directory in (gym_root, ray_thirdparty, ray_archive):
        directory.mkdir(parents=True)
    ray_init = ray_archive / "__init__.py"
    ray_init.write_text("# ray\n", encoding="ascii")
    ray_module = ModuleType("ray")
    ray_module.__file__ = str(ray_init)
    ray_module.__version__ = bootstrap._SCORER_RAY_VERSION
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(
        bootstrap._seal_direct_runner_ray_sys_path.__globals__,
        "_SCORER_RAY_ARCHIVE_ROOT",
        ray_archive,
    )
    isolated = ["/isolated/stdlib", "/isolated/lib-dynload"]
    safe_path = (*isolated, str(purelib), str(gym_root))
    ray_inputs = {"root": purelib / "ray", "thirdparty": ray_thirdparty, "sources": {}}
    monkeypatch.setitem(
        bootstrap._seal_direct_runner_ray_sys_path.__globals__,
        "_authenticate_direct_runner_ray_inputs",
        lambda selected_purelib: ray_inputs,
    )
    original_path = list(sys.path)
    try:
        sys.path[:] = [str(ray_thirdparty), *safe_path]
        bootstrap._seal_direct_runner_ray_sys_path(safe_path, purelib, ray_inputs)
        assert sys.path == list(safe_path)
    finally:
        sys.path[:] = original_path


def _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path: Path):
    purelib = tmp_path / "purelib"
    ray_root = purelib / "ray"
    thirdparty = ray_root / "thirdparty_files"
    archive_root = tmp_path / "ray-archive"
    hardlink_root = tmp_path / "ray-archive-hardlinks"
    thirdparty.mkdir(parents=True)
    archive_root.mkdir()
    hardlink_root.mkdir()
    identities = {}
    for relative, payload in {
        "__init__.py": b"# pinned ray\n",
        "_version.py": b'version = "2.56.1"\n',
    }.items():
        target = archive_root / relative
        target.write_bytes(payload)
        target.chmod(0o600)
        (hardlink_root / relative).hardlink_to(target)
        (ray_root / relative).symlink_to(target)
        identities[relative] = {
            "size": len(payload),
            "sha256": runtime._sha256_bytes(payload),
        }
    globals_dict = bootstrap._authenticate_direct_runner_ray_inputs.__globals__
    monkeypatch.setitem(globals_dict, "_SCORER_RAY_ARCHIVE_ROOT", archive_root)
    monkeypatch.setitem(globals_dict, "_SCORER_RAY_SOURCES", identities)
    monkeypatch.setattr(
        globals_dict["importlib"].metadata,
        "version",
        lambda name: bootstrap._SCORER_RAY_VERSION if name == "ray" else "unexpected",
    )
    return purelib, thirdparty


def test_direct_bootstrap_authenticates_exact_ray_inputs(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib, thirdparty = _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path)

    authenticated = bootstrap._authenticate_direct_runner_ray_inputs(purelib)

    assert authenticated["thirdparty"] == thirdparty
    assert set(authenticated["sources"]) == {"__init__.py", "_version.py"}


def test_direct_bootstrap_rejects_symlinked_ray_thirdparty_path(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib, thirdparty = _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path)
    thirdparty.rmdir()
    replacement = tmp_path / "replacement-thirdparty"
    replacement.mkdir()
    thirdparty.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(ValueError, match="Ray installation differs"):
        bootstrap._authenticate_direct_runner_ray_inputs(purelib)


def test_direct_bootstrap_rejects_changed_ray_source(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib, _thirdparty = _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path)
    ray_init = (purelib / "ray/__init__.py").resolve(strict=True)
    ray_init.write_bytes(b"# changed ray\n")
    ray_init.chmod(0o600)

    with pytest.raises(ValueError, match="Ray source identity differs"):
        bootstrap._authenticate_direct_runner_ray_inputs(purelib)


@pytest.mark.parametrize("link_count", [1, 3])
def test_direct_bootstrap_rejects_changed_ray_source_link_count(monkeypatch, tmp_path: Path, link_count: int) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib, _thirdparty = _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path)
    ray_init = (purelib / "ray/__init__.py").resolve(strict=True)
    hardlink = tmp_path / "ray-archive-hardlinks/__init__.py"
    if link_count == 1:
        hardlink.unlink()
    else:
        (tmp_path / "third-ray-init-hardlink").hardlink_to(ray_init)

    assert ray_init.stat().st_nlink == link_count
    with pytest.raises(ValueError, match="Ray source identity differs"):
        bootstrap._authenticate_direct_runner_ray_inputs(purelib)


@pytest.mark.parametrize("mode", [0o644, 0o664, 0o646])
def test_direct_bootstrap_rejects_changed_ray_source_mode(monkeypatch, tmp_path: Path, mode: int) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    purelib, _thirdparty = _direct_ray_auth_fixture(bootstrap, monkeypatch, tmp_path)
    ray_init = (purelib / "ray/__init__.py").resolve(strict=True)
    ray_init.chmod(mode)

    with pytest.raises(ValueError, match="Ray source identity differs"):
        bootstrap._authenticate_direct_runner_ray_inputs(purelib)


def _test_reasoning_normalize(answer: str) -> set[tuple[str, str]]:
    return {(answer, "role")} if answer else set()


def _format_hook_fixture(monkeypatch, tmp_path: Path, *, environment: str = "citation"):
    bootstrap = _bootstrap_module(monkeypatch)
    gym_root = tmp_path / "Gym"
    base_path = gym_root / "nemo_gym/base_resources_server.py"
    judge_path = gym_root / "nemo_gym/judge.py"
    app_path = gym_root / "resources_servers/format_verification/app.py"
    base_path.parent.mkdir(parents=True)
    app_path.parent.mkdir(parents=True)
    base_payload = b"# pinned base fixture\n"
    judge_payload = b"# pinned judge fixture\n"
    app_source = b"""import re

class FormatVerificationVerifyRequest:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload

class FormatVerificationVerifyResponse:
    def __init__(self, **payload):
        self.payload = payload

    def model_dump(self):
        return self.payload

class FormatVerificationResourcesServer:
    async def verify(self, body: FormatVerificationVerifyRequest) -> FormatVerificationVerifyResponse:
        request = body.model_dump()
        text = self._extract_assistant_text(request["response"])
        if request["verifier"]["type"] == "regex":
            reward, details = self._verify_regex(text, request["verifier"])
        else:
            reward, details = self._verify_string_match(text, request["verifier"])
        return FormatVerificationVerifyResponse(**request, reward=reward, match_details=details)

    @staticmethod
    def _extract_assistant_text(response):
        return "".join(
            part["text"]
            for item in response["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )

    @staticmethod
    def _verify_regex(text, verifier):
        patterns = [re.compile(pattern) for pattern in verifier["verify_regex"]]
        count = sum(1 for line in text.split("\\n") if any(pattern.search(line) for pattern in patterns))
        details = {"matching_lines": count, "min_matches": verifier["verify_min_matches"], "passed": count >= verifier["verify_min_matches"]}
        return (1.0 if details["passed"] else 0.0), details

    @staticmethod
    def _verify_string_match(text, verifier):
        missing = [marker for marker in verifier["expected_markers"] if marker not in text]
        expected = set(verifier["expected_markers"])
        spurious = [match.group(0) for pattern in verifier["patterns"] for match in re.finditer(pattern, text) if match.group(0) not in expected]
        details = {"expected": verifier["expected_markers"], "missing": missing, "spurious": spurious, "passed": not missing and not spurious}
        return (1.0 if details["passed"] else 0.0), details
"""
    base_path.write_bytes(base_payload)
    judge_path.write_bytes(judge_payload)
    app_path.write_bytes(app_source)

    judge_module = ModuleType("strict_format_judge_fixture")
    exec(
        compile(
            b"""import functools
class JudgeError(Exception):
    pass
class JSONResponse:
    pass
def jsonable_encoder(value):
    return value
def judge_failsafe(verify_fn):
    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        return await verify_fn(*args, **kwargs)
    return wrapper
""",
            str(judge_path),
            "exec",
            dont_inherit=True,
        ),
        judge_module.__dict__,
    )
    judge_module.__file__ = str(judge_path)
    base_module = ModuleType("strict_format_base_fixture")
    base_module.__file__ = str(base_path)
    base_module.judge_failsafe = judge_module.judge_failsafe
    app_globals: dict[str, Any] = {}
    exec(
        compile(app_source, str(app_path), "exec", dont_inherit=True),
        app_globals,
    )
    server_type = app_globals["FormatVerificationResourcesServer"]
    request_type = app_globals["FormatVerificationVerifyRequest"]
    globals_dict = bootstrap._install_format_score_evidence.__globals__
    monkeypatch.setitem(globals_dict, "_GYM_ROOT", gym_root)
    monkeypatch.setitem(
        globals_dict,
        "_FORMAT_BASE_SOURCE",
        {
            "path": "nemo_gym/base_resources_server.py",
            "sha256": runtime._sha256_bytes(base_payload),
        },
    )
    monkeypatch.setitem(
        globals_dict,
        "_FORMAT_JUDGE_SOURCE",
        {
            "path": "nemo_gym/judge.py",
            "sha256": runtime._sha256_bytes(judge_payload),
        },
    )
    published: list[tuple[str, dict[str, Any]]] = []

    def publish(root, filename, document):
        published.append((filename, json.loads(json.dumps(document))))
        return runtime._sha256_bytes(bootstrap._canonical(document))

    monkeypatch.setitem(globals_dict, "_publish", publish)
    bootstrap._install_format_score_evidence(
        base_module,
        judge_module,
        spec={
            "environment": environment,
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
            "targets": [
                {
                    "source_sha256": runtime._sha256_bytes(app_source),
                }
            ],
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        app_path=app_path,
    )
    endpoint = base_module.judge_failsafe(server_type().verify)
    payload = {
        "responses_create_params": {},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer [1]"}],
                }
            ]
        },
        "verifier": (
            {
                "type": "string_match",
                "expected_markers": ["[1]"],
                "patterns": [r"\[[0-9]+\]"],
            }
            if environment == "citation"
            else {
                "type": "regex",
                "pattern_id": "answer-line",
                "verify_regex": [r"^answer"],
                "verify_min_matches": 1,
            }
        ),
    }
    return endpoint, request_type, payload, published


def test_format_hook_records_exact_typed_fastapi_call(monkeypatch, tmp_path: Path) -> None:
    endpoint, request_type, payload, published = _format_hook_fixture(monkeypatch, tmp_path)

    returned = asyncio.run(endpoint(body=request_type(payload)))

    assert returned.model_dump()["reward"] == 1.0
    assert [name for name, _ in published] == ["format-verification-call-00000001.json"]
    receipt = published[0][1]
    assert receipt["schema"] == runtime.STRICT_GYM_FORMAT_CALL_SCHEMA
    assert receipt["environment"] == "citation"
    assert receipt["profile_id"] == "citation-string-match-v1"
    assert receipt["outcome"]["kind"] == "returned"


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_format_hook_closes_exact_k4_and_rejects_fifth(
    monkeypatch,
    tmp_path: Path,
    environment: str,
    profile_id: str,
) -> None:
    endpoint, request_type, payload, published = _format_hook_fixture(
        monkeypatch,
        tmp_path,
        environment=environment,
    )

    for _ in range(4):
        asyncio.run(endpoint(body=request_type(payload)))

    assert [name for name, _ in published] == [
        "format-verification-call-00000001.json",
        "format-verification-call-00000002.json",
        "format-verification-call-00000003.json",
        "format-verification-call-00000004.json",
        "format-verification-closed.json",
    ]
    closed = published[-1][1]
    assert closed["profile_id"] == profile_id
    assert closed["call_count"] == 4
    assert [reference["sequence"] for reference in closed["calls"]] == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="closed evidence stream"):
        asyncio.run(endpoint(body=request_type(payload)))
    assert len(published) == 5


def test_format_expectation_rejects_negative_zero_reward() -> None:
    request = {
        "responses_create_params": {},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "no marker"}],
                }
            ]
        },
        "verifier": {
            "type": "string_match",
            "expected_markers": ["[1]"],
            "patterns": [r"\[[0-9]+\]"],
        },
    }
    response = {
        **request,
        "reward": -0.0,
        "match_details": {
            "expected": ["[1]"],
            "missing": ["[1]"],
            "spurious": [],
            "passed": False,
        },
    }

    with pytest.raises(ValueError, match="independent result"):
        runtime.format_verification_call_expectation(
            environment="citation",
            derived_verifier_request=request,
            verifier_response=response,
        )


def test_freeform_expectation_binds_exact_regex_result_and_rejects_inline_prose() -> None:
    request = {
        "responses_create_params": {},
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Title: exact\nBody"}],
                }
            ]
        },
        "verifier": {
            "type": "regex",
            "pattern_id": "title",
            "verify_regex": [r"^Title:"],
            "verify_min_matches": 1,
        },
    }
    response = {
        **request,
        "reward": 1.0,
        "match_details": {
            "matching_lines": 1,
            "min_matches": 1,
            "passed": True,
        },
    }

    expectation = runtime.format_verification_call_expectation(
        environment="freeform",
        derived_verifier_request=request,
        verifier_response=response,
    )

    assert expectation["profile_id"] == "freeform-regex-v1"
    assert expectation["method"] == "_verify_regex"
    assert expectation["float_result"] == 1.0
    request["verifier"]["type"] = "inline_prose"
    response["verifier"]["type"] = "inline_prose"
    with pytest.raises(ValueError, match="identity differs"):
        runtime.format_verification_call_expectation(
            environment="freeform",
            derived_verifier_request=request,
            verifier_response=response,
        )


def test_format_finalizer_publishes_reaped_call_index(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)

    terminal, digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)

    assert terminal["schema"] == runtime.STRICT_GYM_FORMAT_CALL_INDEX_SCHEMA
    assert terminal["profile_id"] == "citation-string-match-v1"
    assert terminal["call_count"] == 4
    assert terminal["quiescence"]["original_process_reaped"] is True
    assert digest == runtime._sha256_bytes((tmp_path / "format-verification-call-index.json").read_bytes())


def test_format_payload_validator_matches_path_loader_without_io(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    terminal, digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)
    payload_roster = _format_payload_roster(tmp_path)

    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("pure format payload validation performed I/O")

    monkeypatch.setattr(runtime, "_load_canonical_document", unexpected_io)
    monkeypatch.setattr(runtime, "_require_sealed_bootstrap_root", unexpected_io)
    monkeypatch.setattr(runtime.os, "open", unexpected_io)
    monkeypatch.setattr(Path, "iterdir", unexpected_io)
    monkeypatch.setattr(Path, "lstat", unexpected_io)

    admitted, admitted_digest = runtime.validate_finalized_format_verification_call_index_payloads(
        payload_roster,
        expected_sha256=digest,
        expected_receipt_root=tmp_path,
        expected_bootstrap_root=session.bootstrap_root,
        expected_bootstrap_sha256=session.bootstrap_sha256,
        expected_pair_id="strict-pair-1",
        expected_job_id="12345",
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )

    assert admitted == terminal
    assert admitted_digest == digest


def test_format_payload_validator_requires_exact_roster_and_bootstrap_identity(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    _terminal, digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)
    payload_roster = _format_payload_roster(tmp_path)
    arguments = {
        "expected_sha256": digest,
        "expected_receipt_root": tmp_path,
        "expected_bootstrap_root": session.bootstrap_root,
        "expected_bootstrap_sha256": session.bootstrap_sha256,
        "expected_pair_id": "strict-pair-1",
        "expected_job_id": "12345",
        "expected_environment": "citation",
        "expected_profile_id": "citation-string-match-v1",
    }
    renamed = list(payload_roster)
    renamed[0] = ("strict_gym_child_runtime/not-index.json", renamed[0][1])
    mutable_payload = list(payload_roster)
    mutable_payload[0] = (mutable_payload[0][0], bytearray(mutable_payload[0][1]))
    poisons = (
        list(payload_roster),
        payload_roster[:-1],
        tuple(reversed(payload_roster)),
        tuple(renamed),
        tuple(mutable_payload),
    )
    for poison in poisons:
        with pytest.raises(ValueError, match="payload roster differs"):
            runtime.validate_finalized_format_verification_call_index_payloads(poison, **arguments)

    for boundary_poison in (
        {"expected_bootstrap_root": Path("/different/bootstrap")},
        {"expected_bootstrap_sha256": "b" * 64},
    ):
        with pytest.raises(ValueError, match="finalized format spec differs"):
            runtime.validate_finalized_format_verification_call_index_payloads(
                payload_roster, **(arguments | boundary_poison)
            )


def test_format_payload_validator_rejects_coherently_rehashed_negative_zero(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    terminal, _digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)
    payloads = dict(_format_payload_roster(tmp_path))

    call_name = "strict_gym_child_runtime/format-verification-call-00000001.json"
    call = json.loads(payloads[call_name])
    call["outcome"]["float_result"] = -0.0
    payloads[call_name] = runtime.canonical_ascii_json(call)
    call_digest = runtime._sha256_bytes(payloads[call_name])

    closed_name = "strict_gym_child_runtime/format-verification-closed.json"
    closed = json.loads(payloads[closed_name])
    closed["calls"][0]["sha256"] = call_digest
    payloads[closed_name] = runtime.canonical_ascii_json(closed)
    closed_digest = runtime._sha256_bytes(payloads[closed_name])

    terminal["calls"][0]["outcome"]["float_result"] = -0.0
    terminal["calls"][0]["receipt"]["sha256"] = call_digest
    terminal["verification_closed"]["sha256"] = closed_digest
    terminal_name = "strict_gym_child_runtime/format-verification-call-index.json"
    payloads[terminal_name] = runtime.canonical_ascii_json(terminal)
    terminal_digest = runtime._sha256_bytes(payloads[terminal_name])
    poisoned_roster = tuple((name, payloads[name]) for name, _payload in _format_payload_roster(tmp_path))

    with pytest.raises(ValueError, match="finalized format call record 1"):
        runtime.validate_finalized_format_verification_call_index_payloads(
            poisoned_roster,
            expected_sha256=terminal_digest,
            expected_receipt_root=tmp_path,
            expected_bootstrap_root=session.bootstrap_root,
            expected_bootstrap_sha256=session.bootstrap_sha256,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "sys_base_prefix_int",
        "sys_base_prefix_relative",
        "sys_base_prefix_noncanonical",
        "sys_base_prefix_venv",
        "proc_exe_int",
        "proc_exe_relative",
        "proc_exe_noncanonical",
    ),
)
def test_format_payload_validator_rejects_coherently_rehashed_process_paths(
    monkeypatch, tmp_path: Path, mutation: str
) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    terminal, _digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)
    payloads = dict(_format_payload_roster(tmp_path))
    resource_name = "strict_gym_child_runtime/resource.json"
    resource = json.loads(payloads[resource_name])
    if mutation == "sys_base_prefix_int":
        resource["process"]["sys_base_prefix"] = 1
    elif mutation == "sys_base_prefix_relative":
        resource["process"]["sys_base_prefix"] = "usr/local"
    elif mutation == "sys_base_prefix_noncanonical":
        resource["process"]["sys_base_prefix"] = "/usr/local/../local"
    elif mutation == "sys_base_prefix_venv":
        resource["process"]["sys_base_prefix"] = resource["process"]["sys_prefix"]
    elif mutation == "proc_exe_int":
        resource["process"]["proc_exe"] = 1
    elif mutation == "proc_exe_relative":
        resource["process"]["proc_exe"] = "usr/local/bin/python3.13"
    else:
        resource["process"]["proc_exe"] = "/usr/local/../local/bin/python3.13"
    payloads[resource_name] = runtime.canonical_ascii_json(resource)
    resource_digest = runtime._sha256_bytes(payloads[resource_name])

    index_name = "strict_gym_child_runtime/index.json"
    index = json.loads(payloads[index_name])
    index["children"][0]["receipt"]["sha256"] = resource_digest
    payloads[index_name] = runtime.canonical_ascii_json(index)
    index_digest = runtime._sha256_bytes(payloads[index_name])

    terminal["resource_receipt"]["sha256"] = resource_digest
    terminal["child_index"]["sha256"] = index_digest
    terminal_name = "strict_gym_child_runtime/format-verification-call-index.json"
    payloads[terminal_name] = runtime.canonical_ascii_json(terminal)
    terminal_digest = runtime._sha256_bytes(payloads[terminal_name])
    poisoned_roster = tuple((name, payloads[name]) for name, _payload in _format_payload_roster(tmp_path))

    with pytest.raises(ValueError, match="resource process paths|resource receipt"):
        runtime.validate_finalized_format_verification_call_index_payloads(
            poisoned_roster,
            expected_sha256=terminal_digest,
            expected_receipt_root=tmp_path,
            expected_bootstrap_root=session.bootstrap_root,
            expected_bootstrap_sha256=session.bootstrap_sha256,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


def test_format_path_loader_delegates_owned_payload_roster(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    terminal, digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)
    validator = runtime.validate_finalized_format_verification_call_index_payloads
    observed: list[tuple[tuple[tuple[str, bytes], ...], dict[str, Any]]] = []

    def spy(payload_roster, **kwargs):
        observed.append((payload_roster, kwargs))
        return validator(payload_roster, **kwargs)

    monkeypatch.setattr(
        runtime,
        "validate_finalized_format_verification_call_index_payloads",
        spy,
    )

    admitted = runtime.load_finalized_format_verification_call_index(
        tmp_path / "format-verification-call-index.json",
        expected_sha256=digest,
        expected_receipt_root=tmp_path,
        expected_pair_id="strict-pair-1",
        expected_job_id="12345",
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )

    assert admitted == (terminal, digest)
    assert len(observed) == 1
    payload_roster, arguments = observed[0]
    assert payload_roster == _format_payload_roster(tmp_path)
    assert arguments["expected_bootstrap_root"] == session.bootstrap_root
    assert arguments["expected_bootstrap_sha256"] == session.bootstrap_sha256


def test_format_finalizer_rejects_negative_zero_expected_call(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    expected[0]["float_result"] = -0.0

    with pytest.raises(ValueError, match="expected format verification call 1"):
        session.finalize_format_verification_calls(expected, run_helper=run_helper)


def test_format_offline_loader_rejects_coherently_rehashed_negative_zero(monkeypatch, tmp_path: Path) -> None:
    session, expected, _documents, run_helper = _format_finalizer_fixture(monkeypatch, tmp_path)
    terminal, _digest = session.finalize_format_verification_calls(expected, run_helper=run_helper)

    def replace_document(filename: str, document: dict[str, Any]) -> str:
        path = tmp_path / filename
        path.chmod(0o600)
        payload = runtime.canonical_ascii_json(document)
        path.write_bytes(payload)
        path.chmod(0o400)
        return runtime._sha256_bytes(payload)

    call_name = "format-verification-call-00000001.json"
    call = json.loads((tmp_path / call_name).read_bytes())
    call["outcome"]["float_result"] = -0.0
    call_digest = replace_document(call_name, call)
    closed_name = "format-verification-closed.json"
    closed = json.loads((tmp_path / closed_name).read_bytes())
    closed["calls"][0]["sha256"] = call_digest
    closed_digest = replace_document(closed_name, closed)
    terminal["calls"][0]["outcome"]["float_result"] = -0.0
    terminal["calls"][0]["receipt"]["sha256"] = call_digest
    terminal["verification_closed"]["sha256"] = closed_digest
    terminal_digest = replace_document("format-verification-call-index.json", terminal)

    with pytest.raises(ValueError, match="finalized format call record 1"):
        runtime.load_finalized_format_verification_call_index(
            tmp_path / "format-verification-call-index.json",
            expected_sha256=terminal_digest,
            expected_receipt_root=tmp_path,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
            expected_environment="citation",
            expected_profile_id="citation-string-match-v1",
        )


@pytest.mark.parametrize(
    "poison",
    ["positional", "custom", "request_subclass", "dict_subclass", "list_subclass"],
)
def test_format_hook_rejects_nonexact_request_boundary(monkeypatch, tmp_path: Path, poison: str) -> None:
    endpoint, request_type, payload, _published = _format_hook_fixture(monkeypatch, tmp_path)

    if poison == "positional":
        invoke = lambda: endpoint(request_type(payload))
        match = "invocation differs"
    elif poison == "custom":
        invoke = lambda: endpoint(body=SimpleNamespace(model_dump=lambda: payload))
        match = "request type differs"
    elif poison == "request_subclass":

        class RequestSubclass(request_type):
            pass

        invoke = lambda: endpoint(body=RequestSubclass(payload))
        match = "request type differs"
    elif poison == "dict_subclass":

        class DictSubclass(dict):
            pass

        invoke = lambda: endpoint(body=request_type(DictSubclass(payload)))
        match = "typed request dump differs"
    else:

        class ListSubclass(list):
            pass

        poisoned = json.loads(json.dumps(payload))
        poisoned["response"]["output"] = ListSubclass(poisoned["response"]["output"])
        invoke = lambda: endpoint(body=request_type(poisoned))
        match = "non-exact JSON value"

    with pytest.raises(TypeError, match=match):
        asyncio.run(invoke())


def test_reasoning_score_instrumentation_records_success(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[tuple[str, dict[str, Any]]] = []

    def publish(root, filename, document):
        published.append((filename, document))
        return "d" * 64

    monkeypatch.setitem(
        bootstrap._publish.__globals__,
        "_publish",
        publish,
    )
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: lambda *, answer, entry: 1.0)
    spec = {
        "environment": "reasoning_gym",
        "pair_id": "pair-1",
        "job_id": "123",
        "receipt_root": str(tmp_path),
    }
    process = {"pid": 12, "start_ticks": 34}

    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec=spec,
        spec_sha="a" * 64,
        process=process,
        frozen_score_fn=lambda *, answer, entry: 1.0,
        frozen_normalize_fn=_test_reasoning_normalize,
    )
    score = reasoning_gym.get_score_answer_fn("knights_knaves")(
        answer="sage", entry={"answer": "sage", "question": "q"}
    )

    assert score == 1.0
    assert published[0][0] == "reasoning-score-call-00000001.json"
    assert published[0][1]["outcome"] == {
        "kind": "returned",
        "float_result": 1.0,
    }


def test_reasoning_score_instrumentation_records_exception(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []

    def publish(root, filename, document):
        published.append(document)
        return "d" * 64

    monkeypatch.setitem(
        bootstrap._publish.__globals__,
        "_publish",
        publish,
    )

    def fail(*, answer, entry):
        raise LookupError("hidden by pinned resource app")

    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: fail)
    spec = {
        "environment": "reasoning_gym",
        "pair_id": "pair-1",
        "job_id": "123",
        "receipt_root": str(tmp_path),
    }

    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec=spec,
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=fail,
        frozen_normalize_fn=_test_reasoning_normalize,
    )
    with pytest.raises(LookupError, match="hidden"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry={"answer": "sage", "question": "q"})

    assert published[0]["outcome"] == {
        "kind": "exception",
        "phase": "score",
        "type": "builtins.LookupError",
    }


@pytest.mark.parametrize(
    "entry",
    [
        {"question": "q"},
        {"answer": None, "question": "q"},
    ],
)
def test_reasoning_score_instrumentation_rejects_inputs_hidden_by_pinned_scorer(
    monkeypatch, tmp_path: Path, entry: dict[str, Any]
) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []

    def publish(root, filename, document):
        published.append(json.loads(json.dumps(document)))
        return "d" * 64

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        # The pinned scorer returns 0.0 after swallowing either malformed entry.
        frozen_score_fn=lambda *, answer, entry: 0.0,
        frozen_normalize_fn=_test_reasoning_normalize,
    )

    with pytest.raises(TypeError, match="entry answer is not an exact string"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry=entry)

    assert published[0]["outcome"] == {
        "kind": "exception",
        "phase": "score",
        "type": "builtins.TypeError",
    }


def test_reasoning_score_instrumentation_rejects_swallowed_internal_failure(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []

    def publish(root, filename, document):
        published.append(json.loads(json.dumps(document)))
        return "d" * 64

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=lambda *, answer, entry: 0.0,
        frozen_normalize_fn=_test_reasoning_normalize,
    )

    with pytest.raises(ValueError, match="authenticated normalization"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry={"answer": "sage", "question": "q"})

    assert published[0]["outcome"] == {
        "kind": "exception",
        "phase": "score",
        "type": "builtins.ValueError",
    }


def test_reasoning_score_instrumentation_admits_valid_zero_without_exception(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []

    def publish(root, filename, document):
        published.append(json.loads(json.dumps(document)))
        return "d" * 64

    def score(*, answer, entry):
        oracle = _test_reasoning_normalize(entry["answer"])
        candidate = _test_reasoning_normalize(answer)
        if oracle == candidate:
            return 1.0
        return 0.0

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=score,
        frozen_normalize_fn=_test_reasoning_normalize,
    )

    result = reasoning_gym.get_score_answer_fn("knights_knaves")(
        answer="candidate", entry={"answer": "oracle", "question": "q"}
    )

    assert result == 0.0
    assert published[0]["outcome"] == {"kind": "returned", "float_result": 0.0}


def test_reasoning_score_instrumentation_rejects_caught_exception_at_valid_zero(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []
    normalization_calls = 0

    def publish(root, filename, document):
        published.append(json.loads(json.dumps(document)))
        return "d" * 64

    def stateful_normalize(answer: str) -> set[tuple[str, str]]:
        nonlocal normalization_calls
        normalization_calls += 1
        if normalization_calls == 3:
            raise LookupError("swallowed inside pinned scorer")
        return {(answer, "role")}

    def swallowing_score(*, answer, entry):
        try:
            oracle = stateful_normalize(entry["answer"])
            candidate = stateful_normalize(answer)
            if oracle == candidate:
                return 1.0
        except Exception:
            pass
        return 0.0

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=swallowing_score,
        frozen_normalize_fn=stateful_normalize,
    )

    with pytest.raises(RuntimeError, match="raised a caught exception"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(
            answer="candidate", entry={"answer": "oracle", "question": "q"}
        )

    assert published[0]["outcome"] == {
        "kind": "exception",
        "phase": "score",
        "type": "builtins.RuntimeError",
    }


def test_reasoning_normalizer_staticmethod_identity_is_plain_function() -> None:
    class Dataset:
        @staticmethod
        def normalize(answer: str) -> set[tuple[str, str]]:
            return {(answer, "role")}

    instance = Dataset()
    assert instance.normalize is Dataset.normalize
    assert getattr(instance.normalize, "__func__", None) is None


def test_reasoning_score_instrumentation_rejects_receiver_normalizer_rebind(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)

    class Dataset:
        @staticmethod
        def _normalize_answer(answer: str) -> set[tuple[str, str]]:
            return {(answer, "role")}

        def score_answer(self, *, answer, entry):
            return float(self._normalize_answer(answer) == self._normalize_answer(entry["answer"]))

    monkeypatch.setitem(
        bootstrap._publish.__globals__,
        "_publish",
        lambda root, filename, document: "d" * 64,
    )
    dataset = Dataset()
    frozen_normalize = dataset._normalize_answer
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=dataset.score_answer,
        frozen_normalize_fn=frozen_normalize,
    )
    dataset._normalize_answer = lambda answer: {(answer, "hostile")}

    with pytest.raises(RuntimeError, match="semantics changed"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry={"answer": "sage"})


def test_reasoning_score_instrumentation_freezes_callable_and_closes_exact_k4(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[tuple[str, dict[str, Any]]] = []
    resolver_calls = 0

    def publish(root, filename, document):
        published.append((filename, json.loads(json.dumps(document))))
        return runtime._sha256_bytes(bootstrap._canonical(document))

    def mutable_resolver(task_name):
        nonlocal resolver_calls
        resolver_calls += 1
        return lambda *, answer, entry: 0.0

    def hard_exit(code: int) -> None:
        raise RuntimeError(f"hard-exit-{code}")

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    monkeypatch.setattr(os, "_exit", hard_exit)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=mutable_resolver)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=lambda *, answer, entry: 1.0,
        frozen_normalize_fn=_test_reasoning_normalize,
    )

    scores = [
        reasoning_gym.get_score_answer_fn("knights_knaves")(
            answer=f"answer-{index}",
            entry={"answer": f"answer-{index}"},
        )
        for index in range(4)
    ]

    assert scores == [1.0, 1.0, 1.0, 1.0]
    assert resolver_calls == 0
    assert [name for name, _ in published] == [
        "reasoning-score-call-00000001.json",
        "reasoning-score-call-00000002.json",
        "reasoning-score-call-00000003.json",
        "reasoning-score-call-00000004.json",
        "reasoning-score-closed.json",
    ]
    assert published[-1][1]["schema"] == runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA
    assert [item["sequence"] for item in published[-1][1]["calls"]] == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="hard-exit-80"):
        reasoning_gym.get_score_answer_fn("knights_knaves")


def test_reasoning_score_instrumentation_serializes_blocked_fifth_behind_close(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[str] = []
    fourth_entered = threading.Event()
    release_fourth = threading.Event()
    fifth_started = threading.Event()
    thread_errors: list[str] = []
    invocation = 0

    def publish(root, filename, document):
        published.append(filename)
        return runtime._sha256_bytes(bootstrap._canonical(document))

    def frozen(*, answer, entry):
        nonlocal invocation
        invocation += 1
        if invocation == 4:
            fourth_entered.set()
            assert release_fourth.wait(timeout=5)
        return 1.0

    def hard_exit(code: int) -> None:
        raise RuntimeError(f"hard-exit-{code}")

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    monkeypatch.setattr(os, "_exit", hard_exit)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=frozen,
        frozen_normalize_fn=_test_reasoning_normalize,
    )
    for index in range(3):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer=str(index), entry={"answer": str(index)})

    fourth = threading.Thread(
        target=lambda: reasoning_gym.get_score_answer_fn("knights_knaves")(answer="fourth", entry={"answer": "fourth"})
    )

    def fifth_call() -> None:
        fifth_started.set()
        try:
            reasoning_gym.get_score_answer_fn("knights_knaves")
        except RuntimeError as error:
            thread_errors.append(str(error))

    fifth = threading.Thread(target=fifth_call)
    fourth.start()
    assert fourth_entered.wait(timeout=5)
    fifth.start()
    assert fifth_started.wait(timeout=5)
    release_fourth.set()
    fourth.join(timeout=5)
    fifth.join(timeout=5)

    assert not fourth.is_alive() and not fifth.is_alive()
    assert thread_errors == ["hard-exit-80"]
    assert published[-1] == "reasoning-score-closed.json"
    assert not any(name.endswith("00000005.json") for name in published)


def test_reasoning_score_instrumentation_hashes_precall_deep_copy(monkeypatch, tmp_path: Path) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    published: list[dict[str, Any]] = []

    def publish(root, filename, document):
        published.append(json.loads(json.dumps(document)))
        return "d" * 64

    def mutating_scorer(*, answer, entry):
        entry["answer"] = "mutated"
        return 0.0

    monkeypatch.setitem(bootstrap._publish.__globals__, "_publish", publish)
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    entry = {"answer": "original"}
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=mutating_scorer,
        frozen_normalize_fn=_test_reasoning_normalize,
    )
    reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry=entry)

    assert entry == {"answer": "mutated"}
    assert published[0]["input"]["entry_sha256"] == runtime._sha256_bytes(b'{"answer":"original"}')


@pytest.mark.parametrize("bad_result", [1, True, -0.0, float("inf")])
def test_reasoning_score_instrumentation_rejects_nonexact_float_result(
    monkeypatch,
    tmp_path: Path,
    bad_result: Any,
) -> None:
    bootstrap = _bootstrap_module(monkeypatch)
    monkeypatch.setitem(
        bootstrap._publish.__globals__,
        "_publish",
        lambda root, name, document: "d" * 64,
    )
    reasoning_gym = SimpleNamespace(get_score_answer_fn=lambda task_name: None)
    bootstrap._install_reasoning_score_evidence(
        reasoning_gym,
        spec={
            "environment": "reasoning_gym",
            "pair_id": "pair-1",
            "job_id": "123",
            "receipt_root": str(tmp_path),
        },
        spec_sha="a" * 64,
        process={"pid": 12, "start_ticks": 34},
        frozen_score_fn=lambda *, answer, entry: bad_result,
        frozen_normalize_fn=_test_reasoning_normalize,
    )

    with pytest.raises(ValueError, match="not an admitted exact float"):
        reasoning_gym.get_score_answer_fn("knights_knaves")(answer="sage", entry={"answer": "sage"})


def test_reasoning_score_expectation_binds_exact_input() -> None:
    expectation = runtime.reasoning_score_call_expectation(
        task_name="knights_knaves",
        answer="sage",
        entry={"question": "q"},
        float_result=0.5,
    )

    assert expectation == {
        "task_name": "knights_knaves",
        "answer_sha256": runtime._sha256_bytes(b'"sage"'),
        "entry_sha256": runtime._sha256_bytes(b'{"question":"q"}'),
        "float_result": 0.5,
    }
    with pytest.raises(ValueError, match="finite float"):
        runtime.reasoning_score_call_expectation(
            task_name="knights_knaves",
            answer="sage",
            entry={"question": "q"},
            float_result=-0.0,
        )


def test_scorer_only_k4_finalizer_publishes_terminal_index(monkeypatch, tmp_path: Path) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)

    terminal, digest = session.finalize_score_calls(expected, run_helper=run_helper)

    assert terminal["schema"] == runtime.STRICT_GYM_SCORE_CALL_INDEX_SCHEMA
    assert terminal["call_count"] == 4
    assert [item["sequence"] for item in terminal["calls"]] == [1, 2, 3, 4]
    assert terminal["score_closed"]["schema"] == runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA
    assert terminal["quiescence"]["original_process_reaped"] is True
    assert terminal["quiescence"]["child_termination_signal"] == "SIGINT"
    assert run_helper._processes == {}
    assert len(digest) == 64
    assert (tmp_path / "reasoning-score-call-index.json").stat().st_mode & 0o777 == 0o400
    reloaded, reloaded_digest = runtime.load_finalized_reasoning_score_call_index(
        tmp_path / "reasoning-score-call-index.json",
        expected_sha256=digest,
        expected_receipt_root=tmp_path,
        expected_pair_id="strict-pair-1",
        expected_job_id="12345",
    )
    assert reloaded == terminal
    assert reloaded_digest == digest


def test_finalized_score_loader_requires_external_digest_and_exact_inventory(monkeypatch, tmp_path: Path) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    _, digest = session.finalize_score_calls(expected, run_helper=run_helper)
    terminal_path = tmp_path / "reasoning-score-call-index.json"

    with pytest.raises(ValueError, match="caller-carried SHA256"):
        runtime.load_finalized_reasoning_score_call_index(
            terminal_path,
            expected_sha256="f" * 64,
            expected_receipt_root=tmp_path,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
        )

    _write_immutable(tmp_path / "unbound.json", {"unexpected": True})
    with pytest.raises(ValueError, match="inventory differs"):
        runtime.load_finalized_reasoning_score_call_index(
            terminal_path,
            expected_sha256=digest,
            expected_receipt_root=tmp_path,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
        )


def test_finalized_score_loader_binds_current_sealed_bootstrap(monkeypatch, tmp_path: Path) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    _, digest = session.finalize_score_calls(expected, run_helper=run_helper)
    monkeypatch.setattr(
        runtime,
        "_require_sealed_bootstrap_root",
        lambda: (Path("/different/sealed/bootstrap"), "b" * 64),
    )

    with pytest.raises(ValueError, match="finalized score-call spec differs"):
        runtime.load_finalized_reasoning_score_call_index(
            tmp_path / "reasoning-score-call-index.json",
            expected_sha256=digest,
            expected_receipt_root=tmp_path,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("closed_sequence_bool", "closed ref 1 differs"),
        ("closed_pid_float", "closed receipt differs"),
        ("quiescence_pid_float", "quiescence differs"),
        ("server_port_float", "resource receipt differs"),
        ("sys_base_prefix_int", "resource receipt differs"),
        ("scorer_resolved_origin_alias", "scorer paths differ"),
    ],
)
def test_finalized_score_loader_rejects_coherently_rehashed_type_and_path_aliases(
    monkeypatch, tmp_path: Path, mutation: str, match: str
) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    session.finalize_score_calls(expected, run_helper=run_helper)

    def load(filename: str) -> dict[str, Any]:
        return runtime._load_canonical_document(tmp_path / filename)[0]

    def replace_file(filename: str, document: dict[str, Any]) -> bytes:
        path = tmp_path / filename
        path.chmod(0o600)
        return _write_immutable(path, document)

    terminal = load("reasoning-score-call-index.json")
    if mutation.startswith("closed_"):
        closed = load("reasoning-score-closed.json")
        if mutation == "closed_sequence_bool":
            closed["calls"][0]["sequence"] = True
        else:
            closed["process"]["pid"] = 123.0
        closed_payload = replace_file("reasoning-score-closed.json", closed)
        terminal["score_closed"]["sha256"] = runtime._sha256_bytes(closed_payload)
    elif mutation == "quiescence_pid_float":
        terminal["quiescence"]["pid"] = 123.0
    else:
        resource = load("resource.json")
        if mutation == "server_port_float":
            resource["server"]["port"] = 5123.0
        elif mutation == "sys_base_prefix_int":
            resource["process"]["sys_base_prefix"] = 1
        else:
            resource["scorer"]["resolved_origin"] = resource["scorer"]["resolver_resolved_origin"]
        resource_payload = replace_file("resource.json", resource)
        index = load("index.json")
        index["children"][0]["receipt"]["sha256"] = runtime._sha256_bytes(resource_payload)
        index_payload = replace_file("index.json", index)
        terminal["resource_receipt"]["sha256"] = runtime._sha256_bytes(resource_payload)
        terminal["child_index"]["sha256"] = runtime._sha256_bytes(index_payload)
    terminal_payload = replace_file("reasoning-score-call-index.json", terminal)

    with pytest.raises(ValueError, match=match):
        runtime.load_finalized_reasoning_score_call_index(
            tmp_path / "reasoning-score-call-index.json",
            expected_sha256=runtime._sha256_bytes(terminal_payload),
            expected_receipt_root=tmp_path,
            expected_pair_id="strict-pair-1",
            expected_job_id="12345",
        )


def test_scorer_only_k4_finalizer_rejects_caught_exception(monkeypatch, tmp_path: Path) -> None:
    session, expected, documents, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    first_path = tmp_path / "reasoning-score-call-00000001.json"
    first_path.chmod(0o600)
    documents[0]["outcome"] = {
        "kind": "exception",
        "phase": "score_or_float",
        "type": "builtins.LookupError",
    }
    _write_immutable(first_path, documents[0])

    with pytest.raises(ValueError, match="reasoning score outcome has the wrong keyset"):
        session.finalize_score_calls(expected, run_helper=run_helper)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("answer_sha256", "f" * 64, "input mismatch"),
        ("entry_sha256", "e" * 64, "input mismatch"),
        ("float_result", 0.25, "did not return the expected reward"),
    ],
)
def test_scorer_only_k4_finalizer_rejects_expected_call_mismatch(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: str | float,
    match: str,
) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    expected[0][field] = value

    with pytest.raises(ValueError, match=match):
        session.finalize_score_calls(expected, run_helper=run_helper)


@pytest.mark.parametrize(
    ("sequence", "section", "field", "value", "match"),
    [
        (1, "process", "start_ticks", 777.0, "process mismatch"),
        (1, "outcome", "float_result", 1, "expected reward"),
        (2, "outcome", "float_result", -0.0, "expected reward"),
    ],
)
def test_scorer_only_k4_finalizer_rejects_nonexact_receipt_types(
    monkeypatch,
    tmp_path: Path,
    sequence: int,
    section: str,
    field: str,
    value: Any,
    match: str,
) -> None:
    session, expected, documents, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    call_path = tmp_path / f"reasoning-score-call-{sequence:08d}.json"
    call_path.chmod(0o600)
    documents[sequence - 1][section][field] = value
    _write_immutable(call_path, documents[sequence - 1])

    with pytest.raises(ValueError, match=match):
        session.finalize_score_calls(expected, run_helper=run_helper)


@pytest.mark.parametrize("mutation", ["extra", "gap", "missing_closed"])
def test_scorer_only_k4_finalizer_rejects_nonexact_call_inventory(
    monkeypatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    session, expected, documents, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    if mutation == "extra":
        extra = json.loads(json.dumps(documents[-1]))
        extra["sequence"] = 5
        _write_immutable(tmp_path / "reasoning-score-call-00000005.json", extra)
    elif mutation == "gap":
        (tmp_path / "reasoning-score-call-00000004.json").unlink()
    else:
        (tmp_path / "reasoning-score-closed.json").unlink()

    with pytest.raises(RuntimeError, match="extra, gap, or missing"):
        session.finalize_score_calls(expected, run_helper=run_helper)


@pytest.mark.parametrize("failure", ["pid_reuse", "dead", "listener_lost"])
def test_scorer_only_k4_finalizer_rejects_stale_resource_process(
    monkeypatch,
    tmp_path: Path,
    failure: str,
) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    if failure == "pid_reuse":
        monkeypatch.setattr(runtime, "_process_stat", lambda pid: (50, 778))
    elif failure == "dead":

        def missing(pid: int) -> tuple[int, int]:
            raise ProcessLookupError(pid)

        monkeypatch.setattr(runtime, "_process_stat", missing)
    else:
        monkeypatch.setattr(runtime, "_listening_socket_inodes", lambda pid, host, port: [])

    expected_exception = ProcessLookupError if failure == "dead" else RuntimeError
    with pytest.raises(expected_exception):
        session.finalize_score_calls(expected, run_helper=run_helper)


@pytest.mark.parametrize(
    ("field", "value"),
    [("scope", "main"), ("environment", "citation")],
)
def test_score_finalizer_rejects_wrong_scope_or_environment(
    monkeypatch,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    session = replace(session, **{field: value})

    with pytest.raises(RuntimeError, match="only valid for scorer-only reasoning replay"):
        session.finalize_score_calls(expected, run_helper=run_helper)


def test_scorer_only_k4_finalizer_rejects_second_finalization(monkeypatch, tmp_path: Path) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    session.finalize_score_calls(expected, run_helper=run_helper)

    with pytest.raises(RuntimeError, match="already finalized"):
        session.finalize_score_calls(expected, run_helper=run_helper)


def test_scorer_only_k4_finalizer_closes_child_receipt_reference(monkeypatch, tmp_path: Path) -> None:
    session, expected, _, run_helper = _score_finalizer_fixture(monkeypatch, tmp_path)
    index_path = tmp_path / "index.json"
    child_index, _ = runtime._load_canonical_document(index_path)
    resource_path = tmp_path / "resource.json"
    resource, _ = runtime._load_canonical_document(resource_path)
    resource["job_id"] = "54321"
    resource_path.chmod(0o600)
    replacement_payload = _write_immutable(resource_path, resource)
    child_index["children"][0]["receipt"]["sha256"] = runtime._sha256_bytes(replacement_payload)
    index_path.chmod(0o600)
    _write_immutable(index_path, child_index)

    with pytest.raises(RuntimeError, match="retained startup attestation"):
        session.finalize_score_calls(expected, run_helper=run_helper)
