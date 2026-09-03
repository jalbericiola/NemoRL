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

"""Static and fail-closed checks for the parallel profiled replay shell family."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
V2_SCRIPTS = (
    REPO_ROOT / "strict_pair_replay_launch_v2.sh",
    REPO_ROOT / "strict_pair_replay_job_wrapper_v2.sh",
)
LEGACY_SHA256 = {
    "strict_pair_replay_launch.sh": ("e36554c3f1d269d647f444b79176a7f21812f33e8e5164486ce375f28c0ff0ad"),
    "strict_pair_replay_job_wrapper.sh": ("ec682724562d60ced2153ca1ba712e00f8492f8a4ec2d3af48f930a7ff34ad11"),
}


def _embedded_python(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return source.split("<<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]


def _dict_assignments(source: str) -> dict[str, dict[str, str]]:
    """Evaluate only literal dict assignments and prior-name dict unpacking."""
    result: dict[str, dict[str, str]] = {}
    for node in ast.parse(source).body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(node.value, ast.Dict)
        ):
            continue
        value: dict[str, str] = {}
        supported = True
        for key, member in zip(node.value.keys, node.value.values, strict=True):
            if key is None:
                if not isinstance(member, ast.Name) or member.id not in result:
                    supported = False
                    break
                value.update(result[member.id])
                continue
            try:
                literal_key = ast.literal_eval(key)
                literal_value = ast.literal_eval(member)
            except (TypeError, ValueError):
                supported = False
                break
            if type(literal_key) is not str or type(literal_value) is not str:
                supported = False
                break
            value[literal_key] = literal_value
        if supported:
            result[node.targets[0].id] = value
    return result


def _closure_function(source: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "authenticate_program_closure":
            return node
    raise AssertionError("authenticate_program_closure is missing")


def _named_functions(source: str, names: set[str]) -> list[ast.FunctionDef]:
    result = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in result} == names
    return result


def _named_assignment_expression(source: str, name: str) -> ast.expr:
    matches = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_v2_scripts_are_shell_and_embedded_python_syntax_clean() -> None:
    for path in V2_SCRIPTS:
        assert stat.S_IMODE(path.stat().st_mode) & 0o111
        subprocess.run(["bash", "-n", str(path)], check=True)
        compile(_embedded_python(path), str(path), "exec")


@pytest.mark.parametrize("script", V2_SCRIPTS, ids=lambda path: path.name)
@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "freeform-regex-v1"),
        ("freeform", "citation-string-match-v1"),
        ("reasoning_gym", "citation-string-match-v1"),
    ],
)
def test_v2_shell_rejects_noncanonical_profile_pair_before_bootstrap(
    script: Path,
    environment: str,
    profile_id: str,
) -> None:
    digest = "a" * 64
    argv = [
        "/bin/bash",
        str(script),
        "--pair-manifest",
        "/pair.json",
        "--pair-manifest-sha256",
        digest,
        "--pair-submission-receipt",
        "/pair-submission.json",
        "--pair-submission-receipt-sha256",
        digest,
        "--off-exit-receipt",
        "/off-exit.json",
        "--off-exit-receipt-sha256",
        digest,
        "--replay-manifest",
        "/replay.json",
        "--replay-manifest-sha256",
        digest,
        "--environment",
        environment,
        "--profile-id",
        profile_id,
    ]
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={},
        timeout=10,
    )
    assert completed.returncode == 2
    assert b"environment/profile pair is not admitted" in completed.stderr
    assert b"bootstrap sha256sum" not in completed.stderr


def test_v2_script_program_maps_exactly_match_manifest_v2() -> None:
    manifest_source = (REPO_ROOT / "nemo_rl/utils/strict_captured_replay_manifest_v2.py").read_text(encoding="utf-8")
    expected = _dict_assignments(manifest_source)["REPLAY_PROGRAM_V2_PATHS"]
    assert len(expected) == 14
    assert expected["profile_registry"] == ("nemo_rl/utils/strict_captured_replay_profiles.py")
    for path in V2_SCRIPTS:
        actual = _dict_assignments(_embedded_python(path))["PROGRAM_PATHS"]
        assert actual == expected


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_launcher_builds_exact_twenty_token_profiled_wrapper_tail(
    environment: str,
    profile_id: str,
) -> None:
    source = _embedded_python(V2_SCRIPTS[0])
    function = _named_functions(source, {"build_submission_argv"})[0]
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "pair_manifest": {
            "campaign": {
                "nodes": 1,
                "slurm": {"account": "a", "partition": "p", "qos": "q"},
            }
        },
        "canonical_absolute_path": lambda value, _label: Path(value),
    }
    exec(compile(module, str(V2_SCRIPTS[0]), "exec"), namespace)
    manifest = {
        "execution_environment": {"attempt": {"operational": {"slurm": "/state/slurm"}}},
        "replay_contract": {"program": {"job_wrapper": {"path": "strict_pair_replay_job_wrapper_v2.sh"}}},
        "scheduler_submission": {"identity": {"job_name": "replay-job"}},
        "slurm_export_boundary": {"path": "/state/export.env"},
    }
    values = {
        "pair_manifest_path": Path("/authority/pair.json"),
        "pair_manifest_sha256": "1" * 64,
        "pair_submission_receipt_path": Path("/authority/pair-submission.json"),
        "pair_submission_receipt_sha256": "2" * 64,
        "trusted_off_exit_receipt_path": Path("/authority/off-exit.json"),
        "trusted_off_exit_receipt_sha256": "3" * 64,
        "manifest_path": Path("/authority/replay.json"),
        "manifest_sha256": "4" * 64,
        "expected_environment": environment,
        "expected_profile_id": profile_id,
    }
    argv = namespace["build_submission_argv"](
        manifest=manifest,
        snapshot_root=Path("/snapshot"),
        comment="comment",
        slurm_export_fd=71,
        job_wrapper_fd=72,
        **values,
    )
    assert argv[17:19] == ["--export-file=71", "/proc/self/fd/72"]
    assert argv[-20:] == [
        "--pair-manifest",
        str(values["pair_manifest_path"]),
        "--pair-manifest-sha256",
        values["pair_manifest_sha256"],
        "--pair-submission-receipt",
        str(values["pair_submission_receipt_path"]),
        "--pair-submission-receipt-sha256",
        values["pair_submission_receipt_sha256"],
        "--off-exit-receipt",
        str(values["trusted_off_exit_receipt_path"]),
        "--off-exit-receipt-sha256",
        values["trusted_off_exit_receipt_sha256"],
        "--replay-manifest",
        str(values["manifest_path"]),
        "--replay-manifest-sha256",
        values["manifest_sha256"],
        "--environment",
        environment,
        "--profile-id",
        profile_id,
    ]


def test_launcher_submits_only_sealed_wrapper_and_export_descriptors() -> None:
    source = _embedded_python(V2_SCRIPTS[0])
    assert 'authenticated_programs["job_wrapper"]' in source
    assert 'sealed_memfd(wrapper_raw, label="authenticated job wrapper")' in source
    assert 'sealed_memfd(slurm_export_raw, label="authenticated Slurm export")' in source
    assert "pass_fds=(slurm_export_fd, job_wrapper_fd)" in source
    assert 'host_tools["scontrol"]["path"]' in source
    assert '"batch_script"' in source
    assert "controller_script.stdout != wrapper_raw" in source
    assert 'f"--export-file={slurm_export_fd}"' in source
    assert 'f"/proc/self/fd/{job_wrapper_fd}"' in source


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed memfd transport is the Linux HSG production path",
)
def test_launcher_sealed_descriptors_survive_wrapper_name_poison(
    tmp_path: Path,
) -> None:
    source = _embedded_python(V2_SCRIPTS[0])
    function = _named_functions(source, {"sealed_memfd"})[0]

    def fail(message: str) -> None:
        raise AssertionError(message)

    namespace = {
        "hashlib": hashlib,
        "fcntl": fcntl,
        "os": os,
        "stat": stat,
        "fail": fail,
    }
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(V2_SCRIPTS[0]), "exec"), namespace)
    good_wrapper = b"#!/bin/bash -p\nprintf authenticated\\n\n"
    good_export = b"PATH=/usr/bin:/bin\0"
    wrapper_path = tmp_path / "strict_pair_replay_job_wrapper_v2.sh"
    marker = tmp_path / "poison-marker"
    wrapper_path.write_bytes(good_wrapper)
    wrapper_fd = namespace["sealed_memfd"](good_wrapper, label="wrapper")
    export_fd = namespace["sealed_memfd"](good_export, label="export")
    wrapper_path.write_text(
        f"#!/bin/bash -p\ntouch {marker}\n",
        encoding="ascii",
    )
    tail = tuple(f"token-{index}" for index in range(20))
    stub = (
        "import os,sys;"
        "e=int(sys.argv[1]);w=sys.argv[2];"
        "assert os.read(e,1048576)==b'PATH=/usr/bin:/bin\\0';"
        "assert open(w,'rb').read()==b'#!/bin/bash -p\\nprintf authenticated\\\\n\\n';"
        "assert sys.argv[3:]==[f'token-{i}' for i in range(20)]"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                stub,
                str(export_fd),
                f"/proc/self/fd/{wrapper_fd}",
                *tail,
            ],
            pass_fds=(export_fd, wrapper_fd),
            close_fds=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        os.close(export_fd)
        os.close(wrapper_fd)
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_wrapper_authenticates_exact_hsg_spool_copy_and_retains_driver_bytes() -> None:
    shell = V2_SCRIPTS[1].read_text(encoding="utf-8")
    source = _embedded_python(V2_SCRIPTS[1])
    assert "^/cm/local/apps/slurm/var/spool/job[0-9]+/slurm_script$" in shell
    assert 'Path("/cm/local/apps/slurm/var/spool")' in source
    assert 'f"job{int(job_id, 10):05d}" / "slurm_script"' in source
    assert "parent_before.st_gid != os.getgid()" in source
    assert "executed_wrapper_raw != wrapper_raw" in source
    assert "executed_script != wrapper_path" not in source
    bootstrap = ast.literal_eval(_named_assignment_expression(source, "RETAINED_ENTRYPOINT_BOOTSTRAP"))
    compile(bootstrap, "<retained-entrypoint-bootstrap>", "exec")
    assert "anonymous_immutable_input(" in source
    assert "stdin_descriptor=retained_entrypoint_fd" in source
    assert source.count('        "-A",') == 1


def test_launcher_preflights_the_exact_attempt_scoped_cleanup_targets() -> None:
    source = _embedded_python(V2_SCRIPTS[0])
    expected_suffixes = (
        "CLEANUP_CANCEL.stderr",
        "CLEANUP_CANCEL.stdout",
        "CLEANUP_POST.scontrol.raw",
        "CLEANUP_POST.scontrol.stderr",
        "CLEANUP_PRE.scontrol.raw",
        "CLEANUP_PRE.scontrol.stderr",
        "UNKNOWN_CANDIDATE.sbatch.stderr",
        "UNKNOWN_CANDIDATE.sbatch.stdout",
        "cleanup-report.json",
    )
    suffix_expression = _named_assignment_expression(source, "CLEANUP_ARTIFACT_SUFFIXES")
    assert ast.literal_eval(suffix_expression) == expected_suffixes

    observed: list[tuple[Path, str]] = []
    poisoned: Path | None = None

    def fail(message: str) -> None:
        raise SystemExit(message)

    def require_absent(path: Path, *, label: str) -> None:
        observed.append((path, label))
        if path == poisoned:
            raise RuntimeError(f"poisoned existing output: {path}")

    namespace: dict[str, Any] = {
        "CLEANUP_ARTIFACT_SUFFIXES": expected_suffixes,
        "Path": Path,
        "fail": fail,
        "require_absent": require_absent,
    }
    functions = _named_functions(
        source,
        {"cleanup_artifact_path", "require_cleanup_artifacts_absent"},
    )
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(V2_SCRIPTS[0]), "exec"), namespace)

    root = Path("/state/launcher")
    for attempt_id in ("replay-1", "replay-2"):
        observed.clear()
        namespace["require_cleanup_artifacts_absent"](
            submission_parent=root,
            attempt_id=attempt_id,
        )
        expected_paths = tuple(root / f"{attempt_id}.{suffix}" for suffix in expected_suffixes)
        assert tuple(path for path, _ in observed) == expected_paths
        assert tuple(label for _, label in observed) == tuple(
            f"launcher cleanup artifact {suffix}" for suffix in expected_suffixes
        )
        assert root / "cleanup-report.json" not in expected_paths
        assert root / "UNKNOWN_CANDIDATE.sbatch.stdout" not in expected_paths
        assert root / "UNKNOWN_CANDIDATE.sbatch.stderr" not in expected_paths
        for poisoned in expected_paths:
            with pytest.raises(RuntimeError, match="poisoned existing output"):
                namespace["require_cleanup_artifacts_absent"](
                    submission_parent=root,
                    attempt_id=attempt_id,
                )
        poisoned = None

    class StringSubclass(str):
        pass

    with pytest.raises(SystemExit, match="cleanup artifact attempt differs"):
        namespace["cleanup_artifact_path"](
            submission_parent=root,
            attempt_id=StringSubclass("replay-1"),
            suffix="cleanup-report.json",
        )
    with pytest.raises(SystemExit, match="cleanup artifact suffix is not admitted"):
        namespace["cleanup_artifact_path"](
            submission_parent=root,
            attempt_id="replay-1",
            suffix=StringSubclass("cleanup-report.json"),
        )


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    [
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ],
)
def test_wrapper_builds_exact_profiled_driver_argv(
    environment: str,
    profile_id: str,
) -> None:
    source = _embedded_python(V2_SCRIPTS[1])
    function = _named_functions(source, {"build_driver_argv"})[0]
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)

    def fail(message: str) -> None:
        raise SystemExit(message)

    def required_mapping(value: Any, label: str) -> dict[str, Any]:
        if type(value) is not dict:
            fail(f"{label} is not an object")
        return value

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "fail": fail,
        "program": {"entrypoint": {"path": "examples/nemo_gym/run_strict_captured_replay_v2.py"}},
        "required_mapping": required_mapping,
        "snapshot_root": Path("/snapshot"),
    }
    exec(compile(module, str(V2_SCRIPTS[1]), "exec"), namespace)
    manifest = {
        "environment": environment,
        "scorer_profile": {"profile_id": profile_id},
    }
    argv = namespace["build_driver_argv"](
        manifest=manifest,
        manifest_path=Path("/authority/replay.json"),
        manifest_sha256="1" * 64,
        pre_receipt_path=Path("/state/PRE.json"),
        pre_receipt_sha256="2" * 64,
        expected_environment=environment,
        expected_profile_id=profile_id,
    )
    assert argv == [
        "/snapshot/examples/nemo_gym/run_strict_captured_replay_v2.py",
        "--replay-driver-phase",
        "--replay-manifest",
        "/authority/replay.json",
        "--replay-manifest-sha256",
        "1" * 64,
        "--pre-receipt",
        "/state/PRE.json",
        "--pre-receipt-sha256",
        "2" * 64,
        "--environment",
        environment,
        "--profile-id",
        profile_id,
    ]
    with pytest.raises(SystemExit, match="driver environment/profile differs"):
        namespace["build_driver_argv"](
            manifest=manifest,
            manifest_path=Path("/authority/replay.json"),
            manifest_sha256="1" * 64,
            pre_receipt_path=Path("/state/PRE.json"),
            pre_receipt_sha256="2" * 64,
            expected_environment=environment,
            expected_profile_id="wrong-profile",
        )


@pytest.mark.parametrize("poison", ["path", "sha256"])
@pytest.mark.parametrize("script", V2_SCRIPTS, ids=lambda path: path.name)
def test_profile_registry_poison_is_rejected_by_full_closure_before_import(
    script: Path,
    poison: str,
) -> None:
    source = _embedded_python(script)
    assignments = _dict_assignments(source)
    program_paths = assignments["PROGRAM_PATHS"]
    snapshot = {relative: "a" * 64 for relative in program_paths.values()}
    program = {name: {"path": relative, "sha256": snapshot[relative]} for name, relative in program_paths.items()}
    program["profile_registry"][poison] = "nemo_rl/utils/poisoned_profiles.py" if poison == "path" else "b" * 64
    authenticated: list[str] = []

    def fail(message: str) -> None:
        raise SystemExit(message)

    def required_mapping(value: Any, label: str) -> dict[str, Any]:
        if type(value) is not dict:
            fail(f"{label} is not an object")
        return value

    def authenticated_program(
        snapshot_root: Path,
        snapshot_manifest: dict[str, str],
        *,
        name: str,
        require_executable: bool,
    ) -> tuple[Path, str, bytes]:
        del snapshot_manifest, require_executable
        authenticated.append(name)
        return snapshot_root / program_paths[name], "a" * 64, b"authenticated"

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "PROGRAM_NAMES": frozenset(program_paths),
        "PROGRAM_PATHS": program_paths,
        "authenticated_program": authenticated_program,
        "fail": fail,
        "required_mapping": required_mapping,
    }
    function = _closure_function(source)
    ast.fix_missing_locations(function)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"),
        namespace,
    )
    with pytest.raises(SystemExit, match="profile_registry differs"):
        namespace["authenticate_program_closure"](
            Path("/authenticated/snapshot"),
            snapshot,
            program,
            executable_name=("submission_launcher" if "launch_v2" in script.name else "job_wrapper"),
        )
    assert "profile_registry" not in authenticated

    tree = ast.parse(source)
    closure_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "authenticate_program_closure"
    ]
    import_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "import_authenticated_module"
    ]
    assert closure_calls and import_calls
    assert min(closure_calls) < min(import_calls)


@pytest.mark.parametrize("script", V2_SCRIPTS, ids=lambda path: path.name)
def test_mutated_transitive_marker_bytes_are_rejected_before_execution(
    tmp_path: Path,
    script: Path,
) -> None:
    source = _embedded_python(script)
    program_paths = _dict_assignments(source)["PROGRAM_PATHS"]
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    snapshot: dict[str, str] = {}
    for name, relative in program_paths.items():
        path = snapshot_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = f"SEALED_{name} = True\n".encode("ascii")
        path.write_bytes(raw)
        path.chmod(0o500 if name in {"submission_launcher", "job_wrapper"} else 0o400)
        snapshot[relative] = hashlib.sha256(raw).hexdigest()
    program = {name: {"path": relative, "sha256": snapshot[relative]} for name, relative in program_paths.items()}
    marker = tmp_path / "marker"
    profile_path = snapshot_root / program_paths["profile_registry"]
    profile_path.chmod(0o600)
    profile_path.write_text(f"open({str(marker)!r}, 'wb').close()\n", encoding="ascii")
    profile_path.chmod(0o400)

    def fail(message: str) -> None:
        raise SystemExit(message)

    def stable_evidence_bytes(
        path: Path,
        *,
        label: str,
        exact_mode: int | None,
    ) -> bytes:
        del label, exact_mode
        return path.read_bytes()

    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "PROGRAM_NAMES": frozenset(program_paths),
        "PROGRAM_PATHS": program_paths,
        "hashlib": hashlib,
        "stat": stat,
        "fail": fail,
        "stable_evidence_bytes": stable_evidence_bytes,
    }
    functions = _named_functions(
        source,
        {
            "required_mapping",
            "authenticated_program",
            "authenticate_program_closure",
        },
    )
    ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(script), "exec"),
        namespace,
    )
    with pytest.raises(SystemExit, match="profile_registry bytes differ"):
        namespace["authenticate_program_closure"](
            snapshot_root,
            snapshot,
            program,
            executable_name=("submission_launcher" if "launch_v2" in script.name else "job_wrapper"),
        )
    assert not marker.exists()


def test_v2_scripts_use_only_profiled_lifecycle_and_generic_scorer_names() -> None:
    launcher = V2_SCRIPTS[0].read_text(encoding="utf-8")
    wrapper = V2_SCRIPTS[1].read_text(encoding="utf-8")
    combined = launcher + wrapper
    for api in (
        "load_replay_execution_manifest_v2",
        "build_captured_replay_submission_receipt_v2",
        "publish_captured_replay_submission_receipt_v2",
        "load_captured_replay_submission_receipt_v2",
        "build_captured_replay_pre_receipt_v2",
        "publish_captured_replay_pre_receipt_v2",
        "build_captured_replay_exit_receipt_v2",
        "publish_captured_replay_exit_receipt_v2",
        "build_captured_replay_evidence_index_v2",
        "publish_captured_replay_evidence_index_v2",
        "publish_sealed_result_v2_with_authority",
        "build_captured_replay_result_final_receipt_v2",
        "publish_captured_replay_result_final_receipt_v2",
    ):
        assert api in combined
    assert "reasoning_score_call_index" not in wrapper
    assert "strict_gym_child_runtime/format-verification-call-index.json" in wrapper
    assert "RESULT_INVENTORY_V2_SCHEMA" in wrapper
    assert "RESULT_INVENTORY_V2_FILENAME" in wrapper
    assert '"--hold"' in launcher
    assert '"--dependency=singleton"' in launcher
    assert launcher.index("publish_captured_replay_submission_receipt_v2") < launcher.index(
        '"release", candidate_job_id'
    )


def test_wrapper_preflights_and_publishes_external_final_after_sealing() -> None:
    source = _embedded_python(V2_SCRIPTS[1])
    tree = ast.parse(source)

    def expression_matches(value: ast.expr, expected: str) -> bool:
        return ast.dump(value, include_attributes=False) == ast.dump(
            ast.parse(expected, mode="eval").body,
            include_attributes=False,
        )

    receipts_path_assignment = _named_assignment_expression(source, "receipts_dir")
    assert expression_matches(receipts_path_assignment, "job_root / 'receipts'")

    final_path_assignment = _named_assignment_expression(source, "final_receipt_path")
    assert expression_matches(final_path_assignment, "receipts_dir / 'FINAL.json'")

    preflight_loops = [
        node
        for node in tree.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [member.id if isinstance(member, ast.Name) else None for member in node.target.elts] == ["path", "label"]
    ]
    assert len(preflight_loops) == 1
    preflight_paths = {
        member.elts[0].id
        for member in preflight_loops[0].iter.elts
        if isinstance(member, ast.Tuple) and isinstance(member.elts[0], ast.Name)
    }
    assert "final_receipt_path" in preflight_paths

    calls: dict[str, list[ast.Call]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue
        calls.setdefault(name, []).append(node)

    singleton_names = (
        "publish_sealed_result_v2_with_authority",
        "build_captured_replay_result_final_receipt_v2",
        "publish_captured_replay_result_final_receipt_v2",
        "print",
    )
    assert all(len(calls[name]) == 1 for name in singleton_names)
    assert [calls[name][0].lineno for name in singleton_names] == sorted(
        calls[name][0].lineno for name in singleton_names
    )
    assert len(calls["run_driver"]) == 1
    assert preflight_loops[0].lineno < calls["run_driver"][0].lineno

    authority_call = calls["publish_sealed_result_v2_with_authority"][0]
    authority_arguments = {keyword.arg: keyword.value for keyword in authority_call.keywords}
    expected_authority_arguments = {
        "result_root": "str(output_root)",
        "anchored_sha256": "anchored_sha256",
        "expected_environment": "expected_environment",
        "expected_profile_id": "expected_profile_id",
    }
    assert authority_arguments.keys() == expected_authority_arguments.keys()
    assert all(
        expression_matches(authority_arguments[name], value) for name, value in expected_authority_arguments.items()
    )
    build_call = calls["build_captured_replay_result_final_receipt_v2"][0]
    publish_call = calls["publish_captured_replay_result_final_receipt_v2"][0]
    lifecycle_arguments = {
        "replay_execution_manifest": "manifest",
        "authenticated_source": "authenticated_source",
        "expected_environment": "expected_environment",
        "expected_profile_id": "expected_profile_id",
        "submission_receipt": "submission_document",
        "pre_receipt": "pre_document",
        "exit_receipt": "exit_document",
        "evidence_index": "evidence_index_document",
        "verified_result": "verified_result",
    }
    build_arguments = {keyword.arg: keyword.value for keyword in build_call.keywords}
    assert build_arguments.keys() == lifecycle_arguments.keys()
    assert all(expression_matches(build_arguments[name], value) for name, value in lifecycle_arguments.items())
    publish_arguments = {keyword.arg: keyword.value for keyword in publish_call.keywords}
    expected_publish_arguments = {
        "output": "final_receipt_path",
        "document": "final_document",
        **lifecycle_arguments,
    }
    assert publish_arguments.keys() == expected_publish_arguments.keys()
    assert all(expression_matches(publish_arguments[name], value) for name, value in expected_publish_arguments.items())

    call_targets: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if isinstance(node.value.func, ast.Attribute):
            call_name = node.value.func.attr
        else:
            continue
        assert len(node.targets) == 1
        target = node.targets[0]
        if isinstance(target, ast.Name):
            call_targets[call_name] = (target.id,)
        elif isinstance(target, ast.Tuple):
            assert all(isinstance(member, ast.Name) for member in target.elts)
            call_targets[call_name] = tuple(member.id for member in target.elts if isinstance(member, ast.Name))
    assert call_targets["publish_sealed_result_v2_with_authority"] == (
        "result_inventory_path",
        "result_inventory_sha",
        "verified_result",
    )
    assert call_targets["build_captured_replay_result_final_receipt_v2"] == ("final_document",)
    assert call_targets["publish_captured_replay_result_final_receipt_v2"] == (
        "published_final_path",
        "final_receipt_sha",
    )

    final_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and expression_matches(
            node.test,
            "published_final_path != final_receipt_path",
        )
    ]
    assert len(final_guards) == 1
    assert len(final_guards[0].body) == 1
    assert isinstance(final_guards[0].body[0], ast.Expr)
    guard_call = final_guards[0].body[0].value
    assert isinstance(guard_call, ast.Call)
    assert isinstance(guard_call.func, ast.Name)
    assert guard_call.func.id == "fail"
    assert ast.literal_eval(guard_call.args[0]) == (
        "published FINAL receipt path differs from authenticated job receipt root"
    )

    forbidden_post_cap_calls = {
        "load_evidence_document",
        "open",
        "output_reference",
        "stable_evidence_bytes",
    }
    post_cap_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and authority_call.lineno < node.lineno < publish_call.lineno
    ]
    for call in post_cap_calls:
        if isinstance(call.func, ast.Attribute):
            name = call.func.attr
        elif isinstance(call.func, ast.Name):
            name = call.func.id
        else:
            continue
        assert name not in forbidden_post_cap_calls
        assert not name.startswith("read_")

    print_call = calls["print"][0]
    assert len(print_call.args) == 1
    assert print_call.keywords == []
    dumps_call = print_call.args[0]
    assert isinstance(dumps_call, ast.Call)
    assert isinstance(dumps_call.func, ast.Attribute)
    assert dumps_call.func.attr == "dumps"
    output_document = dumps_call.args[0]
    assert isinstance(output_document, ast.Dict)
    output_arguments = {
        ast.literal_eval(key): value
        for key, value in zip(
            output_document.keys,
            output_document.values,
            strict=True,
        )
    }
    expected_output_arguments = {
        "attempt_id": "manifest['attempt_id']",
        "authenticated_job_id": "live_job_id",
        "pair_id": "manifest['pair_id']",
        "pre_receipt_sha256": "pre_receipt_sha",
        "exit_receipt_sha256": "exit_receipt_sha",
        "evidence_index_sha256": "evidence_index_sha",
        "result_inventory_path": "result_inventory_path",
        "result_inventory_sha256": "result_inventory_sha",
        "final_receipt_path": "str(published_final_path)",
        "final_receipt_sha256": "final_receipt_sha",
    }
    assert output_arguments.keys() == expected_output_arguments.keys()
    assert all(expression_matches(output_arguments[name], value) for name, value in expected_output_arguments.items())
    assert len(dumps_call.keywords) == 2
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in dumps_call.keywords} == {
        "sort_keys": True,
        "separators": (",", ":"),
    }


def test_no_repo_import_can_run_before_complete_program_authentication() -> None:
    for path in V2_SCRIPTS:
        tree = ast.parse(_embedded_python(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("nemo_rl", "nemo_gym")) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("nemo_rl", "nemo_gym"))
        source = _embedded_python(path)
        assert source.index("authenticated_programs = authenticate_program_closure(") < source.index(
            "runner = import_authenticated_module("
        )
        legacy_import = source.index('"nemo_rl.utils.strict_captured_replay_evidence",\n' "    legacy_evidence_path,")
        registry_import = source.index(
            '"nemo_rl.utils.strict_captured_replay_profiles",\n' "    profile_registry_path,"
        )
        manifest_import = source.index('"nemo_rl.utils.strict_captured_replay_manifest_v2",')
        assert legacy_import < manifest_import
        assert registry_import < manifest_import


def test_legacy_replay_shell_bytes_are_invariant() -> None:
    for relative, expected_sha256 in LEGACY_SHA256.items():
        raw = (REPO_ROOT / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
