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

"""Build a genuine strict captured-replay v3 graph for consumer tests.

This module intentionally reuses the lifecycle test's production-backed
builders instead of synthesizing schema-shaped placeholders.  Evaluator and
sealer tests can therefore share one exact manifest-v3/submission-v4/PRE-v2/
EXIT-v5/index-v3 fixture and the corresponding result-inventory-v1 roster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

import nemo_rl.utils.strict_captured_replay_evidence as evidence
import nemo_rl.utils.strict_captured_replay_manifest as replay_manifest_module
from nemo_rl.environments import strict_gym_child_runtime as child_runtime
from nemo_rl.utils.strict_captured_replay_evidence import (
    build_captured_replay_evidence_index,
    build_captured_replay_exit_receipt,
    publish_captured_replay_evidence_index,
    publish_captured_replay_exit_receipt,
)
from nemo_rl.utils.strict_captured_replay_manifest import (
    REPLAY_EXECUTION_MANIFEST_SCHEMA,
    AuthenticatedOffSourceCapture,
)
from nemo_rl.utils.strict_model_transport_replay import (
    validate_strict_model_transport_replay_consumption,
)
from nemo_rl.utils.strict_captured_replay_seal import (
    RESULT_ANCHOR_ALLOWLIST,
    RESULT_FILE_ALLOWLIST,
    RESULT_FILE_SCHEMA_ALLOWLIST,
)
from tests.unit.environments import test_strict_gym_child_runtime as child_fixtures
from tests.unit.utils import test_strict_captured_replay_lifecycle_v3 as lifecycle
from tests.unit.utils import test_strict_captured_replay_manifest as manifest_fixtures

__all__ = [
    "CanonicalLifecycleDocument",
    "StrictCapturedReplayV3CompatPair",
    "StrictCapturedReplayV3CompatFixture",
    "build_strict_captured_replay_v3_compat_pair",
    "build_strict_captured_replay_v3_compat_fixture",
]

_ORIGINAL_LIFECYCLE_BOOTSTRAP_MODULE = lifecycle._bootstrap_module
_ORIGINAL_LIFECYCLE_EXECUTE_CAPTURED_REPLAY_COHORT = lifecycle.execute_captured_replay_cohort
_ORIGINAL_PATH_RESOLVE = Path.resolve

_SCORER_SYS_BASE_PREFIX = "/root/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu"
_SCORER_RESOLVED_PACKAGE_ROOT = "/root/.cache/uv/archive-v0/vjPtRmNH2jG9KTGj/reasoning_gym"


@dataclass(frozen=True)
class CanonicalLifecycleDocument:
    """One lifecycle document together with its exact reference and bytes."""

    document: dict[str, Any]
    reference: dict[str, str]
    raw: bytes


@dataclass(frozen=True)
class StrictCapturedReplayV3CompatFixture:
    """Complete genuine lifecycle and ordered terminal-result compatibility data."""

    authenticated_source: AuthenticatedOffSourceCapture
    result_root: Path
    scorer_scaffold_root: Path
    manifest: CanonicalLifecycleDocument
    submission: CanonicalLifecycleDocument
    pre: CanonicalLifecycleDocument
    exit: CanonicalLifecycleDocument
    evidence_index: CanonicalLifecycleDocument
    outputs: dict[str, dict[str, str]]
    result_roster: tuple[tuple[str, bytes], ...]
    result_anchors: dict[str, str]


@dataclass(frozen=True)
class StrictCapturedReplayV3CompatPair:
    """Two distinct replay attempts backed by one Pair/OFF authority."""

    authenticated_source: AuthenticatedOffSourceCapture
    replay_1: StrictCapturedReplayV3CompatFixture
    replay_2: StrictCapturedReplayV3CompatFixture


def _prepare_shared_replay_static(
    authority_root: Path,
    *,
    authenticated_source: AuthenticatedOffSourceCapture,
    attempt_id: Literal["replay-1", "replay-2"],
) -> None:
    """Publish one attempt while retaining a common Pair submission parent."""

    manifest_fixtures._seal_replay_export(
        authority_root,
        pair=authenticated_source.pair_manifest,
        attempt_id=attempt_id,
    )
    contract = replay_manifest_module.build_replay_submission_contract(
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
        submission_nonce=manifest_fixtures._digest(f"replay-nonce-{attempt_id}"),
    )
    contract_parent = (
        authority_root
        / "results"
        / "captured_replay"
        / "replay_submission_state"
        / authenticated_source.pair_manifest["pair_id"]
    )
    contract_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    contract_parent.chmod(0o700)
    replay_manifest_module.publish_replay_submission_contract(
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
        document=contract,
    )
    replay_manifest_module.load_authenticated_replay_static_inputs(
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
    )


def _canonical_document(
    *,
    document: dict[str, Any],
    path: Path,
    schema: str,
    sha256: str,
    trailing_lf: bool,
) -> CanonicalLifecycleDocument:
    raw = path.read_bytes()
    expected = evidence.canonical_ascii_json(document) + (b"\n" if trailing_lf else b"")
    if raw != expected:
        raise AssertionError(f"lifecycle fixture bytes are not canonical: {path}")
    if document.get("schema") != schema:
        raise AssertionError(f"lifecycle fixture schema differs: {path}")
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise AssertionError(f"lifecycle fixture digest differs: {path}")
    return CanonicalLifecycleDocument(
        document=copy.deepcopy(document),
        reference={"path": str(path), "schema": schema, "sha256": sha256},
        raw=raw,
    )


def _result_roster(result_root: Path) -> tuple[tuple[str, bytes], ...]:
    expected_root_entries = {
        "evidence-index.json",
        "model-transport-replay-consumption.json",
        "replay-ledger.json",
        "strict_gym_child_runtime",
        "transcript-bundle.json",
    }
    actual_root_entries = {path.name for path in result_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise AssertionError("captured-replay result root differs from the closed inventory")
    expected_child_entries = {
        relative.split("/", 1)[1]
        for relative in RESULT_FILE_ALLOWLIST
        if relative.startswith("strict_gym_child_runtime/")
    }
    actual_child_entries = {path.name for path in (result_root / "strict_gym_child_runtime").iterdir()}
    if actual_child_entries != expected_child_entries:
        raise AssertionError("strict Gym result directory differs from the closed inventory")

    roster = tuple((relative, (result_root / relative).read_bytes()) for relative in RESULT_FILE_ALLOWLIST)
    if len(roster) != 13 or tuple(relative for relative, _ in roster) != RESULT_FILE_ALLOWLIST:
        raise AssertionError("captured-replay result roster differs from the exact 13-file allowlist")

    for (relative, raw), schema in zip(roster, RESULT_FILE_SCHEMA_ALLOWLIST, strict=True):
        if type(relative) is not str or type(raw) is not bytes:
            raise AssertionError("result roster must contain only exact str/bytes pairs")
        try:
            document = json.loads(raw.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise AssertionError(f"result fixture is not strict ASCII JSON: {relative}") from error
        if type(document) is not dict or evidence.canonical_ascii_json(document) != raw:
            raise AssertionError(f"result fixture is not canonical ASCII JSON without LF: {relative}")
        if document.get("schema") != schema:
            raise AssertionError(f"result fixture schema differs: {relative}")

    score_relative = "strict_gym_child_runtime/reasoning-score-call-index.json"
    score_index = json.loads(dict(roster)[score_relative].decode("ascii"))
    quiescence = score_index.get("quiescence")
    if type(quiescence) is not dict or quiescence.get("original_process_reaped") is not True:
        raise AssertionError("reasoning scorer call index does not prove original_process_reaped=true")
    if type(quiescence.get("wrapper_returncode")) is not int:
        raise AssertionError("reasoning scorer call index lacks an exact wrapper return code")
    return roster


def _replace_immutable_json(path: Path, document: dict[str, Any]) -> tuple[bytes, str]:
    """Replace one fixture-owned canonical document and restore mode 0400."""

    if not path.is_file() or path.is_symlink():
        raise AssertionError(f"scorer fixture member is not one regular file: {path}")
    payload = child_runtime.canonical_ascii_json(document)
    path.chmod(0o600)
    path.write_bytes(payload)
    path.chmod(0o400)
    return payload, hashlib.sha256(payload).hexdigest()


def _container_scorer_identity(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Derive the scorer spec identity from one authenticated replay contract."""

    scorer_contract = manifest["replay_contract"]["gym_scorer"]
    source = scorer_contract["source"]
    source_root = scorer_contract["source_root"]
    launcher = scorer_contract["launcher"]
    resources = scorer_contract["resources"]
    runtime = scorer_contract["runtime"]
    gym_root = source_root["container_path"]
    if gym_root != "/opt/nemo-rl/3rdparty/Gym-workspace/Gym":
        raise AssertionError("compat fixture scorer container root differs")
    venv_root = "/opt/gym_venvs"
    venv = f"{venv_root}/resources_servers/reasoning_gym/.venv"
    component = f"{gym_root}/resources_servers/reasoning_gym"
    distributions = copy.deepcopy(runtime["required_common_distributions"])
    distributions["reasoning-gym"] = runtime["scorer_pin"]["required_distribution_version"]
    modules = copy.deepcopy(runtime["required_module_versions"])
    modules["reasoning_gym"] = runtime["scorer_pin"]["module_internal_version_literal"]
    scorer = copy.deepcopy(child_runtime._RESOURCE_TARGETS["reasoning_gym"]["scorer"])
    target = {
        "role": "resource",
        "server_type": "resources_servers",
        "server_name": "reasoning_gym",
        "component_relative": "resources_servers/reasoning_gym",
        "component_dir": component,
        "entrypoint": "app.py",
        "source_relative": "resources_servers/reasoning_gym/app.py",
        "source_path": f"{component}/app.py",
        "source_sha256": runtime["selected_resource_app"]["sha256"],
        "config_path": "reasoning_gym",
        "resource_config_path": None,
        "config_relative": ("resources_servers/reasoning_gym/configs/resources_only.yaml"),
        "config_path_source": f"{component}/configs/resources_only.yaml",
        "config_sha256": launcher["resource_only_config"]["sha256"],
        "requirements_relative": "resources_servers/reasoning_gym/requirements.txt",
        "requirements_path": f"{component}/requirements.txt",
        "requirements_sha256": resources["requirements"]["sha256"],
        "venv": venv,
        "interpreter": f"{venv}/bin/python",
        "receipt_filename": "resource.json",
        "distribution_versions": distributions,
        "module_versions": modules,
        "scorer": scorer,
    }
    gym = {
        "git_commit": source["gitlink_commit"],
        "tree": source["tree"],
        "root": gym_root,
        "venv_root": venv_root,
        "sources": {
            "nemo_gym/__init__.py": child_runtime._GYM_SOURCE_PINS["nemo_gym/__init__.py"],
            "nemo_gym/global_config.py": launcher["allocator"]["sha256"],
            "nemo_gym/cli/setup_command.py": launcher["setup"]["sha256"],
            "nemo_gym/cli/env.py": launcher["generator"]["sha256"],
        },
    }
    bootstrap_program = manifest["replay_contract"]["program"]["gym_child_bootstrap"]
    bootstrap_relative = Path(bootstrap_program["path"])
    if (
        bootstrap_relative.is_absolute()
        or bootstrap_relative.name != "sitecustomize.py"
        or ".." in bootstrap_relative.parts
    ):
        raise AssertionError("compat fixture scorer bootstrap path differs")
    bootstrap_root = str(Path("/opt/nemo-rl") / bootstrap_relative.parent)
    return gym, target, bootstrap_root, bootstrap_program["sha256"]


def _normalize_finalized_scorer_graph(
    *,
    manifest: dict[str, Any],
    authenticated_job_id: str,
    receipt_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Project a locally exercised scorer graph into its admitted container view."""

    old_terminal = json.loads((receipt_root / "reasoning-score-call-index.json").read_text(encoding="ascii"))
    old_resource = json.loads((receipt_root / "resource.json").read_text(encoding="ascii"))
    old_child_index = json.loads((receipt_root / "index.json").read_text(encoding="ascii"))
    if old_terminal.get("call_count") != 4 or len(old_terminal.get("calls", [])) != 4:
        raise AssertionError("local scorer finalizer did not close exact K=4")

    gym, target, bootstrap_root, bootstrap_sha256 = _container_scorer_identity(manifest)
    spec = {
        "schema": child_runtime.STRICT_GYM_CHILD_SPEC_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "scope": "scorer-only",
        "pair_id": manifest["pair_id"],
        "job_id": authenticated_job_id,
        "gym": gym,
        "results_dir": str(receipt_root.parent),
        "receipt_root": str(receipt_root),
        "bootstrap": {
            "root": bootstrap_root,
            "filename": "sitecustomize.py",
            "sha256": bootstrap_sha256,
        },
        "targets": [target],
    }
    spec_path = receipt_root / "spec.json"
    _, spec_sha256 = _replace_immutable_json(spec_path, spec)

    old_process = old_resource["process"]
    bootstrap_source = f"{bootstrap_root}/sitecustomize.py"
    process = {
        "pid": old_process["pid"],
        "ppid": old_process["ppid"],
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "cwd": target["component_dir"],
        "sys_executable": target["interpreter"],
        "sys_prefix": target["venv"],
        "sys_base_prefix": _SCORER_SYS_BASE_PREFIX,
        "proc_exe": target["interpreter"],
        "sys_argv": [bootstrap_source, target["source_path"]],
        "proc_argv": [
            target["interpreter"],
            "-I",
            "-S",
            "-B",
            bootstrap_source,
            target["source_path"],
        ],
        "start_ticks": old_process["start_ticks"],
        "boot_id": old_process["boot_id"],
        "hostname": old_process["hostname"],
    }
    server = {
        "config_path": target["config_path"],
        "server_type": target["server_type"],
        "server_name": target["server_name"],
        "entrypoint": target["entrypoint"],
        "host": "127.0.0.1",
        "port": old_resource["server"]["port"],
        "num_workers": old_resource["server"]["num_workers"],
    }
    target_record_keys = (
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
    purelib = f"{target['venv']}/lib/python3.13/site-packages"
    scorer_static = copy.deepcopy(target["scorer"])
    scorer = {
        **scorer_static,
        "package_root": f"{purelib}/reasoning_gym",
        "package_resolved_root": _SCORER_RESOLVED_PACKAGE_ROOT,
        "module_origin": (f"{purelib}/{scorer_static['module_origin_relative_to_purelib']}"),
        "module_resolved_origin": f"{_SCORER_RESOLVED_PACKAGE_ROOT}/__init__.py",
        "resolver_origin": (f"{purelib}/{scorer_static['resolver_origin_relative_to_purelib']}"),
        "resolver_resolved_origin": f"{_SCORER_RESOLVED_PACKAGE_ROOT}/factory.py",
        "origin": f"{purelib}/{scorer_static['origin_relative_to_purelib']}",
        "resolved_origin": (f"{_SCORER_RESOLVED_PACKAGE_ROOT}/logic/knights_knaves.py"),
    }
    resource = {
        "schema": child_runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "pair_id": manifest["pair_id"],
        "job_id": authenticated_job_id,
        "stage": "isolated-runner-pre-entrypoint",
        "spec_sha256": spec_sha256,
        "target": {name: copy.deepcopy(target[name]) for name in target_record_keys},
        "server": server,
        "process": process,
        "distribution_versions": copy.deepcopy(target["distribution_versions"]),
        "module_versions": copy.deepcopy(target["module_versions"]),
        "scorer": scorer,
    }
    resource_path = receipt_root / "resource.json"
    _, resource_sha256 = _replace_immutable_json(resource_path, resource)

    observation = {
        "pid": process["pid"],
        "start_ticks": process["start_ticks"],
        "wrapper_pid": process["pid"],
        "host": server["host"],
        "port": server["port"],
        "listener_socket_inodes": copy.deepcopy(
            old_child_index["children"][0]["observation"]["listener_socket_inodes"]
        ),
    }

    def reference(path: Path, schema: str, sha256: str) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256, "schema": schema}

    spec_ref = reference(spec_path, child_runtime.STRICT_GYM_CHILD_SPEC_SCHEMA, spec_sha256)
    resource_ref = reference(
        resource_path,
        child_runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
        resource_sha256,
    )
    child_index = {
        "schema": child_runtime.STRICT_GYM_CHILD_INDEX_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "scope": "scorer-only",
        "pair_id": manifest["pair_id"],
        "job_id": authenticated_job_id,
        "gym": gym,
        "spec": spec_ref,
        "children": [
            {
                "role": "resource",
                "config_path": target["config_path"],
                "receipt": resource_ref,
                "observation": observation,
            }
        ],
    }
    child_index_path = receipt_root / "index.json"
    _, child_index_sha256 = _replace_immutable_json(child_index_path, child_index)

    process_projection = {
        "pid": process["pid"],
        "start_ticks": process["start_ticks"],
    }
    call_refs: list[dict[str, Any]] = []
    terminal_calls: list[dict[str, Any]] = []
    for sequence, old_record in enumerate(old_terminal["calls"], start=1):
        call = {
            "schema": child_runtime.STRICT_GYM_SCORE_CALL_SCHEMA,
            "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": "reasoning_gym",
            "pair_id": manifest["pair_id"],
            "job_id": authenticated_job_id,
            "spec_sha256": spec_sha256,
            "process": process_projection,
            "sequence": sequence,
            "task_name": old_record["task_name"],
            "input": copy.deepcopy(old_record["input"]),
            "outcome": {
                "kind": "returned",
                "float_result": old_record["float_result"],
            },
        }
        call_path = receipt_root / f"reasoning-score-call-{sequence:08d}.json"
        _, call_sha256 = _replace_immutable_json(call_path, call)
        call_ref = reference(call_path, child_runtime.STRICT_GYM_SCORE_CALL_SCHEMA, call_sha256)
        call_refs.append({"sequence": sequence, **call_ref})
        terminal_calls.append(
            {
                "sequence": sequence,
                "task_name": old_record["task_name"],
                "input": copy.deepcopy(old_record["input"]),
                "float_result": old_record["float_result"],
                "receipt": call_ref,
            }
        )
    closed = {
        "schema": child_runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "pair_id": manifest["pair_id"],
        "job_id": authenticated_job_id,
        "spec_sha256": spec_sha256,
        "process": process_projection,
        "call_count": 4,
        "calls": call_refs,
    }
    closed_path = receipt_root / "reasoning-score-closed.json"
    _, closed_sha256 = _replace_immutable_json(closed_path, closed)
    terminal = {
        "schema": child_runtime.STRICT_GYM_SCORE_CALL_INDEX_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "scope": "scorer-only",
        "pair_id": manifest["pair_id"],
        "job_id": authenticated_job_id,
        "spec": spec_ref,
        "child_index": reference(
            child_index_path,
            child_runtime.STRICT_GYM_CHILD_INDEX_SCHEMA,
            child_index_sha256,
        ),
        "resource_receipt": resource_ref,
        "score_closed": reference(closed_path, child_runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA, closed_sha256),
        "quiescence": copy.deepcopy(old_terminal["quiescence"]),
        "call_count": 4,
        "calls": terminal_calls,
    }
    terminal_path = receipt_root / "reasoning-score-call-index.json"
    _, terminal_sha256 = _replace_immutable_json(terminal_path, terminal)
    return terminal, terminal_sha256, target, bootstrap_root


def _activate_container_scorer_loader_projection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: dict[str, Any],
    target: dict[str, Any],
    bootstrap_root: str,
) -> None:
    """Narrowly project image-owned paths for the producer's offline loader."""

    purelib = f"{target['venv']}/lib/python3.13/site-packages"
    scorer = target["scorer"]
    resolved_paths = {
        _SCORER_SYS_BASE_PREFIX: _SCORER_SYS_BASE_PREFIX,
        f"{purelib}/reasoning_gym": _SCORER_RESOLVED_PACKAGE_ROOT,
        f"{purelib}/{scorer['module_origin_relative_to_purelib']}": (f"{_SCORER_RESOLVED_PACKAGE_ROOT}/__init__.py"),
        f"{purelib}/{scorer['resolver_origin_relative_to_purelib']}": (f"{_SCORER_RESOLVED_PACKAGE_ROOT}/factory.py"),
        f"{purelib}/{scorer['origin_relative_to_purelib']}": (
            f"{_SCORER_RESOLVED_PACKAGE_ROOT}/logic/knights_knaves.py"
        ),
    }

    def projected_resolve(path: Path, strict: bool = False) -> Path:
        replacement = resolved_paths.get(str(path))
        if replacement is not None:
            return Path(replacement)
        return _ORIGINAL_PATH_RESOLVE(path, strict=strict)

    _, _, _, bootstrap_sha256 = _container_scorer_identity(manifest)
    monkeypatch.setattr(Path, "resolve", projected_resolve)
    monkeypatch.setattr(
        child_runtime,
        "STRICT_GYM_ROOT",
        Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym"),
    )
    monkeypatch.setattr(child_runtime, "STRICT_GYM_VENV_ROOT", Path("/opt/gym_venvs"))
    monkeypatch.setitem(
        child_runtime._RESOURCE_TARGETS["reasoning_gym"],
        "requirements_sha256",
        target["requirements_sha256"],
    )
    monkeypatch.setattr(
        child_runtime,
        "_require_sealed_bootstrap_root",
        lambda: (Path(bootstrap_root), bootstrap_sha256),
    )


def _external_score_finalizer_fixture(
    monkeypatch: pytest.MonkeyPatch,
    receipt_root: Path,
    *,
    scaffold_root: Path,
    scorer_pid: int,
    scorer_start_ticks: int,
    scorer_port: int,
    scorer_listener_inode: int,
) -> tuple[
    child_runtime.StrictGymChildRuntimeSession,
    list[dict[str, Any]],
    list[dict[str, Any]],
    SimpleNamespace,
]:
    """Adapt the genuine scorer fixture with dependencies outside result_root."""

    receipt_root.chmod(0o700)
    scaffold_root.mkdir(mode=0o700, exist_ok=True)
    gym_root = scaffold_root / "gym"
    component_dir = gym_root / "resources_servers/reasoning_gym"
    component_dir.mkdir(parents=True, exist_ok=True)
    venv_root = scaffold_root / "venvs"
    monkeypatch.setattr(child_runtime, "STRICT_GYM_ROOT", gym_root)
    monkeypatch.setattr(child_runtime, "STRICT_GYM_VENV_ROOT", venv_root)
    targets = child_runtime._target_matrix(
        "reasoning_gym",
        child_runtime.STRICT_GYM_ROOT,
        scope="scorer-only",
    )
    target = targets[0]
    scorer = target["scorer"]
    if scorer is None:
        raise AssertionError("reasoning Gym scorer target is absent")
    purelib = Path(target["venv"]) / "lib/python3.13/site-packages"
    (purelib / "reasoning_gym").mkdir(parents=True, exist_ok=True)
    for relative_name in (
        scorer["module_origin_relative_to_purelib"],
        scorer["resolver_origin_relative_to_purelib"],
        scorer["origin_relative_to_purelib"],
    ):
        scorer_path = purelib / relative_name
        scorer_path.parent.mkdir(parents=True, exist_ok=True)
        scorer_path.touch()

    spec = child_runtime._build_spec(
        environment="reasoning_gym",
        scope="scorer-only",
        pair_id="strict-pair-1",
        job_id="12345",
        results_dir=receipt_root.parent,
        receipt_root=receipt_root,
        bootstrap_root=Path("/opt/nemo-rl/nemo_rl/environments/bootstrap"),
        bootstrap_sha256="a" * 64,
        targets=targets,
    )
    monkeypatch.setattr(
        child_runtime,
        "_require_sealed_bootstrap_root",
        lambda: (
            Path(spec["bootstrap"]["root"]),
            spec["bootstrap"]["sha256"],
        ),
    )
    spec_payload = child_fixtures._write_immutable(receipt_root / "spec.json", spec)
    receipt, _ = child_fixtures._receipt(spec, target)
    receipt["server"]["port"] = scorer_port
    receipt["process"]["pid"] = scorer_pid
    receipt["process"]["start_ticks"] = scorer_start_ticks
    resource_payload = child_fixtures._write_immutable(
        receipt_root / "resource.json",
        receipt,
    )
    observation = {
        "pid": scorer_pid,
        "start_ticks": scorer_start_ticks,
        "wrapper_pid": scorer_pid,
        "host": "127.0.0.1",
        "port": scorer_port,
        "listener_socket_inodes": [scorer_listener_inode],
    }
    child_index = {
        "schema": child_runtime.STRICT_GYM_CHILD_INDEX_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "scope": "scorer-only",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "gym": spec["gym"],
        "spec": {
            "path": str(receipt_root / "spec.json"),
            "sha256": child_runtime._sha256_bytes(spec_payload),
            "schema": child_runtime.STRICT_GYM_CHILD_SPEC_SCHEMA,
        },
        "children": [
            {
                "role": "resource",
                "config_path": target["config_path"],
                "receipt": {
                    "path": str(receipt_root / "resource.json"),
                    "sha256": child_runtime._sha256_bytes(resource_payload),
                    "schema": child_runtime.STRICT_GYM_CHILD_RECEIPT_SCHEMA,
                },
                "observation": observation,
            }
        ],
    }
    child_index_payload = child_fixtures._write_immutable(
        receipt_root / "index.json",
        child_index,
    )
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
        expected = child_runtime.reasoning_score_call_expectation(
            task_name="knights_knaves",
            answer=answer,
            entry=entry,
            float_result=reward,
        )
        expected_calls.append(expected)
        document = {
            "schema": child_runtime.STRICT_GYM_SCORE_CALL_SCHEMA,
            "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
            "environment": "reasoning_gym",
            "pair_id": spec["pair_id"],
            "job_id": spec["job_id"],
            "spec_sha256": child_runtime._sha256_bytes(spec_payload),
            "process": {"pid": scorer_pid, "start_ticks": scorer_start_ticks},
            "sequence": sequence,
            "task_name": "knights_knaves",
            "input": {
                "answer_sha256": expected["answer_sha256"],
                "entry_sha256": expected["entry_sha256"],
            },
            "outcome": {"kind": "returned", "float_result": reward},
        }
        documents.append(document)
        call_path = receipt_root / f"reasoning-score-call-{sequence:08d}.json"
        call_payload = child_fixtures._write_immutable(call_path, document)
        call_refs.append(
            {
                "sequence": sequence,
                "path": str(call_path),
                "sha256": child_runtime._sha256_bytes(call_payload),
                "schema": child_runtime.STRICT_GYM_SCORE_CALL_SCHEMA,
            }
        )
    closed = {
        "schema": child_runtime.STRICT_GYM_SCORE_CLOSED_SCHEMA,
        "hash_domain": child_runtime.STRICT_GYM_CHILD_HASH_DOMAIN,
        "environment": "reasoning_gym",
        "pair_id": spec["pair_id"],
        "job_id": spec["job_id"],
        "spec_sha256": child_runtime._sha256_bytes(spec_payload),
        "process": {"pid": scorer_pid, "start_ticks": scorer_start_ticks},
        "call_count": 4,
        "calls": call_refs,
    }
    child_fixtures._write_immutable(
        receipt_root / "reasoning-score-closed.json",
        closed,
    )
    state = {"running": True}

    def process_stat(pid: int) -> tuple[int, int]:
        if not state["running"]:
            raise ProcessLookupError(pid)
        return 50, scorer_start_ticks

    monkeypatch.setattr(
        child_runtime,
        "_boot_id",
        lambda: "12345678-1234-1234-1234-123456789abc",
    )
    monkeypatch.setattr(child_runtime, "_process_stat", process_stat)
    monkeypatch.setattr(
        child_runtime,
        "_process_is_descendant",
        lambda pid, ancestor: True,
    )
    monkeypatch.setattr(
        child_runtime,
        "_process_descendant_identities",
        lambda pid: [],
    )
    monkeypatch.setattr(
        child_runtime,
        "_process_argv",
        lambda pid: ["python", target["entrypoint"]],
    )
    monkeypatch.setattr(
        child_runtime,
        "_listening_socket_inodes",
        lambda pid, host, port: [scorer_listener_inode],
    )
    monkeypatch.setattr(
        child_runtime,
        "_validate_receipt",
        lambda document, spec, target, instance, wrapper_pid: observation,
    )
    monkeypatch.setattr(
        child_runtime,
        "_terminate_authenticated_process",
        lambda pid, start_ticks: state.update(running=False) or "SIGINT",
    )
    wrapper = SimpleNamespace(pid=scorer_pid, returncode=None)
    wrapper.poll = lambda: wrapper.returncode
    instance = SimpleNamespace(
        config_path=target["config_path"],
        server_type=target["server_type"],
        name=target["server_name"],
        entrypoint=target["entrypoint"],
        host="127.0.0.1",
        port=scorer_port,
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
    session = child_runtime.StrictGymChildRuntimeSession(
        environment="reasoning_gym",
        scope="scorer-only",
        receipt_root=receipt_root,
        spec_path=receipt_root / "spec.json",
        spec_sha256=child_runtime._sha256_bytes(spec_payload),
        bootstrap_root=Path(spec["bootstrap"]["root"]),
        bootstrap_sha256=spec["bootstrap"]["sha256"],
        spec=spec,
    )
    object.__setattr__(session, "_started_index", child_index)
    object.__setattr__(
        session,
        "_started_index_sha256",
        child_runtime._sha256_bytes(child_index_payload),
    )
    return session, expected_calls, documents, run_helper


def _build_attempt(
    *,
    operation_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated_source: AuthenticatedOffSourceCapture,
    authority_root: Path,
    scorer_scaffold_root: Path,
    attempt_id: Literal["replay-1", "replay-2"],
    replay_job_id: str,
    driver_pid: int,
    driver_start_ticks: int,
    scorer_pid: int,
    scorer_start_ticks: int,
    scorer_port: int,
    scorer_listener_inode: int,
) -> StrictCapturedReplayV3CompatFixture:
    def trusted_test_container(pair: dict[str, Any]) -> dict[str, Any]:
        return {
            **pair["artifacts"]["container"],
            "owner_uid": 153493,
            "owner_gid": 30,
        }

    monkeypatch.setattr(
        replay_manifest_module,
        "_stable_container_asset_identity",
        trusted_test_container,
    )

    _prepare_shared_replay_static(
        authority_root,
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
    )
    manifest_document = replay_manifest_module.build_replay_execution_manifest(
        authenticated_source=authenticated_source,
        attempt_id=attempt_id,
    )

    def attempt_manifest(
        root: str,
    ) -> tuple[dict[str, object], dict[str, object], AuthenticatedOffSourceCapture]:
        if Path(root) != operation_root:
            raise AssertionError("lifecycle fixture used the wrong operation root")
        return manifest_document, authenticated_source.pair_manifest, authenticated_source

    monkeypatch.setattr(lifecycle, "_manifest", attempt_manifest)
    monkeypatch.setattr(lifecycle, "_JOB_ID", replay_job_id)

    def bootstrap_module(fixture_monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
        bootstrap = _ORIGINAL_LIFECYCLE_BOOTSTRAP_MODULE(fixture_monkeypatch)
        original_install = bootstrap._install_reasoning_score_evidence

        def install_reasoning_score_evidence(*args: Any, **kwargs: Any) -> Any:
            kwargs["process"] = {
                "pid": scorer_pid,
                "start_ticks": scorer_start_ticks,
            }
            return original_install(*args, **kwargs)

        bootstrap._install_reasoning_score_evidence = install_reasoning_score_evidence
        return bootstrap

    monkeypatch.setattr(lifecycle, "_bootstrap_module", bootstrap_module)

    (
        loaded_manifest,
        loaded_source,
        manifest_path,
        manifest_sha256,
        submission_document,
        submission_sha256,
        pre_document,
        pre_sha256,
    ) = lifecycle._submission_and_pre(
        operation_root,
        monkeypatch,
        replay_job_id=replay_job_id,
    )
    if loaded_manifest is not manifest_document or loaded_source is not authenticated_source:
        raise AssertionError("lifecycle fixture replaced the shared Pair/OFF authority")

    job_root = Path(pre_document["driver"]["pre_receipt_path"]).parents[1]
    post_record = lifecycle._record(
        manifest_document,
        comment=submission_document["comment"],
        phase="POST",
        job_id=replay_job_id,
    )
    _, post_query_ref = lifecycle._seal_query(
        manifest_document,
        authenticated_source=authenticated_source,
        phase="POST",
        raw_path=job_root / "queries/POST.scontrol.raw",
        record=post_record,
    )
    driver_process = {
        "boot_id_sha256": lifecycle._digest("replay-driver-boot"),
        "pid": driver_pid,
        "start_time_ticks": driver_start_ticks,
    }

    def score_finalizer_fixture(
        fixture_monkeypatch: pytest.MonkeyPatch,
        receipt_root: Path,
    ) -> tuple[
        child_runtime.StrictGymChildRuntimeSession,
        list[dict[str, Any]],
        list[dict[str, Any]],
        SimpleNamespace,
    ]:
        return _external_score_finalizer_fixture(
            fixture_monkeypatch,
            receipt_root,
            scaffold_root=scorer_scaffold_root,
            scorer_pid=scorer_pid,
            scorer_start_ticks=scorer_start_ticks,
            scorer_port=scorer_port,
            scorer_listener_inode=scorer_listener_inode,
        )

    monkeypatch.setattr(
        lifecycle,
        "_score_finalizer_fixture",
        score_finalizer_fixture,
    )

    def execute_with_container_scorer_graph(**kwargs: Any) -> Any:
        documents = _ORIGINAL_LIFECYCLE_EXECUTE_CAPTURED_REPLAY_COHORT(**kwargs)
        receipt_root = Path(manifest_document["artifacts"]["outputs"]["reasoning_score_call_index"]["path"]).parent
        terminal, terminal_sha256, target, bootstrap_root = _normalize_finalized_scorer_graph(
            manifest=manifest_document,
            authenticated_job_id=replay_job_id,
            receipt_root=receipt_root,
        )
        _activate_container_scorer_loader_projection(
            monkeypatch,
            manifest=manifest_document,
            target=target,
            bootstrap_root=bootstrap_root,
        )
        admitted_terminal, admitted_sha256 = child_runtime.load_finalized_reasoning_score_call_index(
            receipt_root / "reasoning-score-call-index.json",
            expected_sha256=terminal_sha256,
            expected_receipt_root=receipt_root,
            expected_pair_id=manifest_document["pair_id"],
            expected_job_id=replay_job_id,
        )
        if admitted_terminal != terminal or admitted_sha256 != terminal_sha256:
            raise AssertionError("container-view scorer graph admission differs")
        terminal_ref = {
            "path": str(receipt_root / "reasoning-score-call-index.json"),
            "schema": child_runtime.STRICT_GYM_SCORE_CALL_INDEX_SCHEMA,
            "sha256": terminal_sha256,
        }
        consumption = copy.deepcopy(documents.transport_consumption)
        consumption["replay"]["scorer_evidence"]["terminal_index"] = terminal_ref
        validate_strict_model_transport_replay_consumption(consumption)
        return type(documents)(
            transcript_bundle=copy.deepcopy(documents.transcript_bundle),
            replay_ledger=copy.deepcopy(documents.replay_ledger),
            transport_consumption=consumption,
            reasoning_score_call_index=terminal_ref,
        )

    monkeypatch.setattr(
        lifecycle,
        "execute_captured_replay_cohort",
        execute_with_container_scorer_graph,
    )
    outputs = lifecycle._publish_real_terminal_outputs(
        manifest=manifest_document,
        authenticated_source=authenticated_source,
        replay_execution_manifest_sha256=manifest_sha256,
        submission_receipt_sha256=submission_sha256,
        process=driver_process,
        monkeypatch=monkeypatch,
    )

    exit_document = build_captured_replay_exit_receipt(
        replay_execution_manifest=manifest_document,
        authenticated_source=authenticated_source,
        submission_receipt=submission_document,
        pre_receipt=pre_document,
        post_scheduler_query=post_query_ref,
        driver_exit_code=0,
        hardware=lifecycle._hardware_observation(manifest_document["runtime_tools"]["document"]["host"]["nvidia_smi"]),
        scheduler_device_environment=lifecycle._device_environment(),
        driver_scheduler_device_environment=lifecycle._device_environment(),
        driver_process=driver_process,
        outputs=outputs,
    )
    exit_path, exit_sha256 = publish_captured_replay_exit_receipt(
        output=job_root / "receipts/EXIT.json",
        document=exit_document,
        replay_execution_manifest=manifest_document,
        submission_receipt=submission_document,
        pre_receipt=pre_document,
        authenticated_source=authenticated_source,
    )

    index_document = build_captured_replay_evidence_index(
        replay_execution_manifest=manifest_document,
        authenticated_source=authenticated_source,
        submission_receipt=submission_document,
        pre_receipt=pre_document,
        exit_receipt=exit_document,
    )
    index_path, index_sha256 = publish_captured_replay_evidence_index(
        output=manifest_document["artifacts"]["outputs"]["evidence_index"]["path"],
        document=index_document,
        replay_execution_manifest=manifest_document,
        submission_receipt=submission_document,
        pre_receipt=pre_document,
        exit_receipt=exit_document,
        authenticated_source=authenticated_source,
    )

    manifest = _canonical_document(
        document=manifest_document,
        path=manifest_path,
        schema=REPLAY_EXECUTION_MANIFEST_SCHEMA,
        sha256=manifest_sha256,
        trailing_lf=False,
    )
    submission = _canonical_document(
        document=submission_document,
        path=Path(manifest_document["scheduler_submission"]["receipt"]["path"]),
        schema=evidence.REPLAY_SUBMISSION_RECEIPT_SCHEMA,
        sha256=submission_sha256,
        trailing_lf=True,
    )
    pre = _canonical_document(
        document=pre_document,
        path=Path(pre_document["driver"]["pre_receipt_path"]),
        schema=evidence.REPLAY_JOB_PRE_RECEIPT_SCHEMA,
        sha256=pre_sha256,
        trailing_lf=True,
    )
    exit_fixture = _canonical_document(
        document=exit_document,
        path=exit_path,
        schema=evidence.REPLAY_JOB_EXIT_RECEIPT_SCHEMA,
        sha256=exit_sha256,
        trailing_lf=True,
    )
    index = _canonical_document(
        document=index_document,
        path=index_path,
        schema=evidence.REPLAY_POST_INDEX_SCHEMA,
        sha256=index_sha256,
        trailing_lf=False,
    )

    result_root = Path(manifest_document["artifacts"]["outputs"]["directory"]["path"])
    roster = _result_roster(result_root)
    anchors = {
        relative: hashlib.sha256(raw).hexdigest() for relative, raw in roster if relative in RESULT_ANCHOR_ALLOWLIST
    }
    if set(anchors) != set(RESULT_ANCHOR_ALLOWLIST):
        raise AssertionError("result fixture terminal anchor set differs")

    return StrictCapturedReplayV3CompatFixture(
        authenticated_source=authenticated_source,
        result_root=result_root,
        scorer_scaffold_root=scorer_scaffold_root,
        manifest=manifest,
        submission=submission,
        pre=pre,
        exit=exit_fixture,
        evidence_index=index,
        outputs=copy.deepcopy(outputs),
        result_roster=roster,
        result_anchors=anchors,
    )


def _fixture_root(tmp_path: Path, *, name: str) -> Path:
    if not isinstance(tmp_path, Path) or not tmp_path.is_absolute():
        raise TypeError("tmp_path must be one absolute pathlib.Path")
    tmp_path = tmp_path.resolve(strict=True)
    root = tmp_path / name
    root.mkdir(mode=0o700)
    return root


def _authenticated_fixture_source(authority_root: Path) -> AuthenticatedOffSourceCapture:
    authority_root.mkdir(mode=0o700)
    source_kwargs = manifest_fixtures._authenticated_source_fixture(authority_root)
    return replay_manifest_module.load_authenticated_off_source_capture(**source_kwargs)


def _attempt_identity(attempt_id: Literal["replay-1", "replay-2"]) -> dict[str, int | str]:
    if attempt_id == "replay-1":
        return {
            "replay_job_id": "82001",
            "driver_pid": 41001,
            "driver_start_ticks": 51001,
            "scorer_pid": 42001,
            "scorer_start_ticks": 52001,
            "scorer_port": 5123,
            "scorer_listener_inode": 61001,
        }
    return {
        "replay_job_id": "82002",
        "driver_pid": 41002,
        "driver_start_ticks": 51002,
        "scorer_pid": 42002,
        "scorer_start_ticks": 52002,
        "scorer_port": 5124,
        "scorer_listener_inode": 61002,
    }


def _build_scoped_attempt(
    *,
    scope_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    authenticated_source: AuthenticatedOffSourceCapture,
    authority_root: Path,
    scorer_scaffold_root: Path,
    attempt_id: Literal["replay-1", "replay-2"],
) -> StrictCapturedReplayV3CompatFixture:
    operation_root = scope_root / "operations" / attempt_id
    operation_root.mkdir(mode=0o700, parents=True)
    identity = _attempt_identity(attempt_id)
    return _build_attempt(
        operation_root=operation_root,
        monkeypatch=monkeypatch,
        authenticated_source=authenticated_source,
        authority_root=authority_root,
        scorer_scaffold_root=scorer_scaffold_root,
        attempt_id=attempt_id,
        replay_job_id=str(identity["replay_job_id"]),
        driver_pid=int(identity["driver_pid"]),
        driver_start_ticks=int(identity["driver_start_ticks"]),
        scorer_pid=int(identity["scorer_pid"]),
        scorer_start_ticks=int(identity["scorer_start_ticks"]),
        scorer_port=int(identity["scorer_port"]),
        scorer_listener_inode=int(identity["scorer_listener_inode"]),
    )


def build_strict_captured_replay_v3_compat_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attempt_id: Literal["replay-1", "replay-2"] = "replay-1",
) -> StrictCapturedReplayV3CompatFixture:
    """Build one exact production-backed replay-v3 fixture graph."""

    if type(attempt_id) is not str or attempt_id not in {"replay-1", "replay-2"}:
        raise ValueError("attempt_id must be exactly replay-1 or replay-2")
    scope_root = _fixture_root(tmp_path, name=f"strict-captured-replay-v3-{attempt_id}")
    authority_root = scope_root / "authority"
    authenticated_source = _authenticated_fixture_source(authority_root)
    return _build_scoped_attempt(
        scope_root=scope_root,
        monkeypatch=monkeypatch,
        authenticated_source=authenticated_source,
        authority_root=authority_root,
        scorer_scaffold_root=scope_root / "scorer-scaffold",
        attempt_id=attempt_id,
    )


def build_strict_captured_replay_v3_compat_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> StrictCapturedReplayV3CompatPair:
    """Build replay-1/replay-2 with shared Pair/OFF and distinct live identities."""

    scope_root = _fixture_root(tmp_path, name="strict-captured-replay-v3-pair")
    authority_root = scope_root / "authority"
    authenticated_source = _authenticated_fixture_source(authority_root)
    scorer_scaffold_root = scope_root / "scorer-scaffold"
    replay_1 = _build_scoped_attempt(
        scope_root=scope_root,
        monkeypatch=monkeypatch,
        authenticated_source=authenticated_source,
        authority_root=authority_root,
        scorer_scaffold_root=scorer_scaffold_root,
        attempt_id="replay-1",
    )
    replay_2 = _build_scoped_attempt(
        scope_root=scope_root,
        monkeypatch=monkeypatch,
        authenticated_source=authenticated_source,
        authority_root=authority_root,
        scorer_scaffold_root=scorer_scaffold_root,
        attempt_id="replay-2",
    )
    if replay_1.authenticated_source is not authenticated_source:
        raise AssertionError("replay-1 does not retain the shared Pair/OFF authority")
    if replay_2.authenticated_source is not authenticated_source:
        raise AssertionError("replay-2 does not retain the shared Pair/OFF authority")
    return StrictCapturedReplayV3CompatPair(
        authenticated_source=authenticated_source,
        replay_1=replay_1,
        replay_2=replay_2,
    )
