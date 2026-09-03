from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCH_SCRIPT = _REPO_ROOT / "strict_pair_replay_launch.sh"
_WRAPPER_SCRIPT = _REPO_ROOT / "strict_pair_replay_job_wrapper.sh"
_SCRIPTS = (
    _LAUNCH_SCRIPT,
    _WRAPPER_SCRIPT,
)
_CANONICAL_GPU_UUIDS = ",".join(
    (
        "GPU-00000000-0000-0000-0000-000000000001",
        "GPU-00000000-0000-0000-0000-000000000002",
        "GPU-00000000-0000-0000-0000-000000000003",
        "GPU-00000000-0000-0000-0000-000000000004",
    )
)


def _embedded_python(script_path: Path) -> str:
    text = script_path.read_text(encoding="utf-8")
    marker = "<<'PY'\n"
    start = text.index(marker) + len(marker)
    end = text.rindex("\nPY\n")
    return text[start:end]


def _embedded_function(
    script_path: Path,
    function_name: str,
    *,
    extra_globals: dict[str, Any] | None = None,
):
    module = ast.parse(_embedded_python(script_path), filename=str(script_path))
    function_def = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    isolated = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Any": Any, "Path": Path}
    if extra_globals is not None:
        namespace.update(extra_globals)
    exec(compile(isolated, filename=str(script_path), mode="exec"), namespace)
    return namespace[function_name]


def _run_shell_script(script_path: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["bash", str(script_path), *args],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={"PATH": os.environ["PATH"]},
    )


def _raise_runtime_error(message: str) -> None:
    raise RuntimeError(message)


def _make_recording_write_exclusive() -> tuple[dict[str, bytes], Any]:
    written: dict[str, bytes] = {}

    def write_exclusive(path: Path, payload: bytes, *, mode: int) -> str:
        if mode != 0o400:
            raise AssertionError(f"unexpected mode: {mode:o}")
        written[str(path)] = payload
        return hashlib.sha256(payload).hexdigest()

    return written, write_exclusive


def _load_written_json(written: dict[str, bytes], path: str) -> dict[str, Any]:
    return json.loads(written[path].decode("ascii"))


class _FakeCompletedProcess:
    def __init__(self, *, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.args: list[str] = []


class _FakeSubprocess:
    DEVNULL = object()
    PIPE = object()

    def __init__(self, completed: _FakeCompletedProcess) -> None:
        self._completed = completed
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, argv: list[str], **kwargs: Any) -> _FakeCompletedProcess:
        self.calls.append((argv, kwargs))
        self._completed.args = list(argv)
        return self._completed


class _SequentialFakeSubprocess:
    DEVNULL = object()
    PIPE = object()

    def __init__(self, completed: list[_FakeCompletedProcess]) -> None:
        self._completed = list(completed)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, argv: list[str], **kwargs: Any) -> _FakeCompletedProcess:
        self.calls.append((argv, kwargs))
        if not self._completed:
            raise AssertionError("unexpected subprocess.run call")
        completed = self._completed.pop(0)
        completed.args = list(argv)
        return completed


class _ForeignContainerOS:
    """Delegate byte I/O while projecting the pinned foreign asset owner."""

    O_RDONLY = os.O_RDONLY
    O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
    O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
    W_OK = os.W_OK

    def __init__(
        self,
        *,
        owner_uid: int = 153493,
        owner_gid: int = 30,
        effective_uid: int = 14568,
        effectively_writable: bool = False,
    ) -> None:
        self.owner_uid = owner_uid
        self.owner_gid = owner_gid
        self.effective_uid = effective_uid
        self.effectively_writable = effectively_writable

    def _project(self, metadata: os.stat_result) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_nlink=metadata.st_nlink,
            st_uid=self.owner_uid,
            st_gid=self.owner_gid,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    def geteuid(self) -> int:
        return self.effective_uid

    def lstat(self, path: Path) -> types.SimpleNamespace:
        return self._project(os.lstat(path))

    def fstat(self, descriptor: int) -> types.SimpleNamespace:
        return self._project(os.fstat(descriptor))

    def access(self, path: Path, mode: int, *, effective_ids: bool) -> bool:
        assert mode == os.W_OK
        assert effective_ids is True
        return self.effectively_writable

    open = staticmethod(os.open)
    read = staticmethod(os.read)
    close = staticmethod(os.close)


class ReplayShellSyntaxTests(unittest.TestCase):
    def _launcher_cleanup_function(
        self,
        *,
        fake_subprocess: Any,
        normalize_scheduler_query: Any,
        verify_scheduler_client_environment: Any,
        write_exclusive: Any,
    ) -> Any:
        return _embedded_function(
            _LAUNCH_SCRIPT,
            "cleanup_authenticated_candidate",
            extra_globals={
                "CLEANUP_REPORT_SCHEMA": ("nemo-rl-strict-captured-replay-launcher-cleanup-report-v1"),
                "json": json,
                "normalize_scheduler_query": normalize_scheduler_query,
                "snapshot_root": Path("/snapshot"),
                "subprocess": fake_subprocess,
                "verify_scheduler_client_environment": (verify_scheduler_client_environment),
                "write_exclusive": write_exclusive,
            },
        )

    def _unknown_candidate_report_function(self, *, write_exclusive: Any) -> Any:
        return _embedded_function(
            _LAUNCH_SCRIPT,
            "persist_unknown_candidate_report",
            extra_globals={
                "CLEANUP_REPORT_SCHEMA": ("nemo-rl-strict-captured-replay-launcher-cleanup-report-v1"),
                "Optional": Optional,
                "json": json,
                "subprocess": subprocess,
                "write_exclusive": write_exclusive,
            },
        )

    def test_replay_shell_scripts_pass_bash_n(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", *(str(path) for path in _SCRIPTS)],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr, b"")

    def test_replay_shell_scripts_embed_compilable_python(self) -> None:
        for script_path in _SCRIPTS:
            compile(_embedded_python(script_path), str(script_path), "exec")

    def test_replay_shell_scripts_use_no_site_bootstrap(self) -> None:
        for script_path in _SCRIPTS:
            text = script_path.read_text(encoding="utf-8")
            self.assertIn('exec "${STRICT_REPLAY_PYTHON}" -I -S -B -', text)
            self.assertIn("sys.flags.no_site != 1", text)

    def test_replay_shell_scripts_join_bootstrap_tools_to_pair_authority(self) -> None:
        def required_mapping(value: Any, label: str) -> dict[str, Any]:
            if type(value) is not dict:
                raise RuntimeError(f"{label} is not an object")
            return value

        sha_path = Path("/trusted/sha256sum")
        sha_digest = "a" * 64
        python_path = Path("/trusted/python")
        python_digest = "b" * 64
        authority = {
            "runtime_tools": {
                "bootstrap_sha256sum": {
                    "path": str(sha_path),
                    "sha256": sha_digest,
                },
                "document": {
                    "host": {
                        "python": {
                            "path": str(python_path),
                            "sha256": python_digest,
                        }
                    }
                },
            }
        }
        poisons = (
            ("bootstrap_sha256sum", "path", "/different/sha256sum", "sha256sum"),
            ("bootstrap_sha256sum", "sha256", "c" * 64, "sha256sum"),
            ("python", "path", "/different/python", "Python"),
            ("python", "sha256", "d" * 64, "Python"),
        )
        for script_path in _SCRIPTS:
            join = _embedded_function(
                script_path,
                "require_pair_bootstrap_runtime_join",
                extra_globals={
                    "fail": _raise_runtime_error,
                    "required_mapping": required_mapping,
                },
            )
            join(
                authority,
                sha_tool_path=sha_path,
                sha_tool_sha256=sha_digest,
                host_python_path=python_path,
                host_python_sha256=python_digest,
            )
            for section, field, poisoned_value, message in poisons:
                poisoned = json.loads(json.dumps(authority))
                if section == "python":
                    poisoned["runtime_tools"]["document"]["host"]["python"][field] = poisoned_value
                else:
                    poisoned["runtime_tools"][section][field] = poisoned_value
                with self.subTest(script=script_path.name, section=section, field=field):
                    with self.assertRaisesRegex(RuntimeError, message):
                        join(
                            poisoned,
                            sha_tool_path=sha_path,
                            sha_tool_sha256=sha_digest,
                            host_python_path=python_path,
                            host_python_sha256=python_digest,
                        )

    def test_replay_shell_scripts_reject_legacy_four_arg_cli(self) -> None:
        for script_path in _SCRIPTS:
            completed = _run_shell_script(
                script_path,
                "--replay-manifest",
                "/tmp/replay.json",
                "--replay-manifest-sha256",
                "0" * 64,
            )
            self.assertEqual(completed.returncode, 2, script_path.name)
            self.assertIn(b"expected --pair-manifest PATH", completed.stderr)

    def test_replay_shell_scripts_reject_missing_off_exit_anchor(self) -> None:
        for script_path in _SCRIPTS:
            completed = _run_shell_script(
                script_path,
                "--pair-manifest",
                "/tmp/pair.json",
                "--pair-manifest-sha256",
                "1" * 64,
                "--pair-submission-receipt",
                "/tmp/pair-receipt.json",
                "--pair-submission-receipt-sha256",
                "2" * 64,
                "--replay-manifest",
                "/tmp/replay.json",
                "--replay-manifest-sha256",
                "3" * 64,
            )
            self.assertEqual(completed.returncode, 2, script_path.name)
            self.assertIn(b"--off-exit-receipt PATH", completed.stderr)

    def test_replay_shell_scripts_validate_off_exit_anchor(self) -> None:
        common = [
            "--pair-manifest",
            "/tmp/pair.json",
            "--pair-manifest-sha256",
            "1" * 64,
            "--pair-submission-receipt",
            "/tmp/pair-receipt.json",
            "--pair-submission-receipt-sha256",
            "2" * 64,
        ]
        suffix = [
            "--replay-manifest",
            "/tmp/replay.json",
            "--replay-manifest-sha256",
            "4" * 64,
        ]
        for script_path in _SCRIPTS:
            relative = _run_shell_script(
                script_path,
                *common,
                "--off-exit-receipt",
                "relative-EXIT.json",
                "--off-exit-receipt-sha256",
                "3" * 64,
                *suffix,
            )
            self.assertEqual(relative.returncode, 2, script_path.name)
            self.assertIn(
                b"OFF EXIT receipt path must be one absolute line",
                relative.stderr,
            )

            malformed_sha = _run_shell_script(
                script_path,
                *common,
                "--off-exit-receipt",
                "/tmp/off/EXIT.json",
                "--off-exit-receipt-sha256",
                "A" * 64,
                *suffix,
            )
            self.assertEqual(malformed_sha.returncode, 2, script_path.name)
            self.assertIn(
                b"OFF EXIT receipt SHA-256 is malformed",
                malformed_sha.stderr,
            )

    def test_replay_shell_scripts_validate_sixteen_arg_positions(self) -> None:
        for script_path in _SCRIPTS:
            completed = _run_shell_script(
                script_path,
                "--pair-manifest",
                "/tmp/pair.json",
                "--pair-manifest-sha256",
                "1" * 64,
                "--pair-submission-receipt",
                "/tmp/pair-receipt.json",
                "--pair-submission-receipt-sha256",
                "2" * 64,
                "--off-exit-receipt",
                "/tmp/off/EXIT.json",
                "--off-exit-receipt-sha256",
                "3" * 64,
                "--replay-manifest",
                "relative-replay.json",
                "--replay-manifest-sha256",
                "4" * 64,
            )
            self.assertEqual(completed.returncode, 2, script_path.name)
            self.assertIn(
                b"manifest path must be one absolute line",
                completed.stderr,
            )

    def test_launcher_build_submission_argv_has_pair_anchor_tail(self) -> None:
        build_submission_argv = _embedded_function(
            _LAUNCH_SCRIPT,
            "build_submission_argv",
            extra_globals={
                "pair_manifest": {
                    "campaign": {
                        "nodes": 1,
                        "slurm": {
                            "account": "acct",
                            "partition": "batch",
                            "qos": "normal",
                        },
                    }
                },
                "canonical_absolute_path": lambda value, label: Path(value),
            },
        )
        manifest = {
            "attempt_id": "attempt-1",
            "scheduler_submission": {
                "accepted_id_record": {
                    "path": ("/results/replay_submission_state/pair-1/attempt-1/" "accepted.job-id")
                },
                "identity": {"job_name": "strict-job"},
            },
            "execution_environment": {"attempt": {"operational": {"slurm": "/operational/pair-1/attempt-1/slurm"}}},
            "replay_contract": {"program": {"job_wrapper": {"path": "strict_pair_replay_job_wrapper.sh"}}},
            "slurm_export_boundary": {"path": "/results/attempt-1.env"},
        }
        argv = build_submission_argv(
            manifest=manifest,
            pair_manifest_path=Path("/control/PAIR_MANIFEST.json"),
            pair_manifest_sha256="a" * 64,
            pair_submission_receipt_path=Path("/control/PAIR_SUBMISSION_RECEIPT.json"),
            pair_submission_receipt_sha256="b" * 64,
            trusted_off_exit_receipt_path=Path("/results/off/job-receipts/EXIT.json"),
            trusted_off_exit_receipt_sha256="d" * 64,
            manifest_path=Path("/control/REPLAY_EXECUTION_MANIFEST.json"),
            manifest_sha256="c" * 64,
            snapshot_root=Path("/snapshot"),
            comment="comment",
        )
        self.assertEqual(
            argv[-16:],
            [
                "--pair-manifest",
                "/control/PAIR_MANIFEST.json",
                "--pair-manifest-sha256",
                "a" * 64,
                "--pair-submission-receipt",
                "/control/PAIR_SUBMISSION_RECEIPT.json",
                "--pair-submission-receipt-sha256",
                "b" * 64,
                "--off-exit-receipt",
                "/results/off/job-receipts/EXIT.json",
                "--off-exit-receipt-sha256",
                "d" * 64,
                "--replay-manifest",
                "/control/REPLAY_EXECUTION_MANIFEST.json",
                "--replay-manifest-sha256",
                "c" * 64,
            ],
        )

    def test_replay_shell_scripts_rehash_authenticated_slurm_conf_per_scheduler_call(
        self,
    ) -> None:
        for script_path in _SCRIPTS:
            verify_calls: list[dict[str, Any]] = []
            fake_subprocess = _SequentialFakeSubprocess(
                [
                    _FakeCompletedProcess(returncode=0, stdout=b"ok"),
                    _FakeCompletedProcess(returncode=0, stdout=b"ok"),
                ]
            )
            run_scheduler_command = _embedded_function(
                script_path,
                "run_scheduler_command",
                extra_globals={
                    "snapshot_root": Path("/snapshot"),
                    "subprocess": fake_subprocess,
                    "verify_scheduler_client_environment": (
                        lambda client_environment: verify_calls.append(client_environment)
                    ),
                },
            )
            client_environment = {
                "variables": {
                    "SLURM_CONF": {
                        "path": "/etc/slurm.conf",
                        "sha256": "f" * 64,
                    }
                }
            }
            env = {"LC_ALL": "C", "SLURM_CONF": "/etc/slurm.conf"}
            run_scheduler_command(
                ["/usr/bin/scontrol", "show", "job", "--json", "123"],
                env=env,
                client_environment=client_environment,
                label="first",
            )
            run_scheduler_command(
                ["/usr/bin/scontrol", "show", "job", "--json", "123"],
                env=env,
                client_environment=client_environment,
                label="second",
            )
            self.assertEqual(
                verify_calls,
                [client_environment, client_environment],
            )

    def test_replay_shell_scripts_reject_unclean_post_scheduler_reason(self) -> None:
        for script_path in _SCRIPTS:
            normalize_scheduler_query = _embedded_function(
                script_path,
                "normalize_scheduler_query",
                extra_globals={
                    "CLEANUP_TERMINAL_STATES": frozenset({"CANCELLED", "COMPLETED", "FAILED"}),
                    "MAX_INT63": (1 << 63) - 1,
                    "canonical_absolute_path": (lambda value, label: Path(value)),
                    "fail": _raise_runtime_error,
                    "json": json,
                    "parse_float": float,
                    "parse_integer": int,
                    "reject_constant": _raise_runtime_error,
                    "reject_duplicate": dict,
                },
            )
            for reason in (123, "None\x01", "None\x7f"):
                raw = (
                    json.dumps(
                        {
                            "errors": [],
                            "jobs": [
                                {
                                    "comment": "strict-comment",
                                    "current_working_directory": "/snapshot",
                                    "hold": False,
                                    "job_id": 123,
                                    "job_state": ["RUNNING"],
                                    "name": "strict-job",
                                    "restart_cnt": 0,
                                    "state_reason": reason,
                                    "user_id": 42,
                                }
                            ],
                            "last_backfill": {},
                            "last_update": {},
                            "meta": {},
                            "warnings": [],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                with self.assertRaises(RuntimeError, msg=(script_path.name, reason)):
                    normalize_scheduler_query(raw, phase="POST")

    def test_launcher_cleanup_cancels_released_pending_candidate_and_persists_report(
        self,
    ) -> None:
        phases: list[str] = []
        verify_calls: list[dict[str, Any]] = []
        fake_subprocess = _SequentialFakeSubprocess(
            [
                _FakeCompletedProcess(returncode=0, stdout=b"cleanup"),
                _FakeCompletedProcess(returncode=0, stdout=b""),
                _FakeCompletedProcess(returncode=0, stdout=b"rollback"),
            ]
        )
        written, write_exclusive = _make_recording_write_exclusive()

        def normalize_scheduler_query(raw: bytes, *, phase: str) -> dict[str, Any]:
            phases.append(phase)
            if raw == b"cleanup":
                return {
                    "job_id": "123",
                    "job_name": "strict-job",
                    "comment": "strict-comment",
                    "user_id": "42",
                    "work_dir": "/snapshot",
                    "job_state": "PENDING",
                    "reason": "Priority",
                    "held": False,
                    "restart_count": 0,
                }
            if raw == b"rollback":
                return {
                    "job_id": "123",
                    "job_name": "strict-job",
                    "comment": "strict-comment",
                    "user_id": "42",
                    "work_dir": "/snapshot",
                    "job_state": "CANCELLED",
                    "reason": "Cancelled",
                    "held": False,
                    "restart_count": 0,
                }
            raise AssertionError(f"unexpected raw payload: {raw!r}")

        cleanup_authenticated_candidate = self._launcher_cleanup_function(
            fake_subprocess=fake_subprocess,
            normalize_scheduler_query=normalize_scheduler_query,
            verify_scheduler_client_environment=(lambda client_environment: verify_calls.append(client_environment)),
            write_exclusive=write_exclusive,
        )
        client_environment = {
            "variables": {
                "SLURM_CONF": {
                    "path": "/etc/slurm.conf",
                    "sha256": "f" * 64,
                }
            }
        }
        result = cleanup_authenticated_candidate(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            host_tools={
                "scontrol": {"path": "/usr/bin/scontrol"},
                "scancel": {"path": "/usr/bin/scancel"},
            },
            client_environment=client_environment,
            submission_env={"LC_ALL": "C"},
            candidate_job_id="123",
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertEqual(
            phases,
            ["CLEANUP", "ROLLBACK"],
        )
        self.assertEqual(
            verify_calls,
            [client_environment, client_environment, client_environment],
        )
        self.assertEqual(result["status"], "cleanup-confirmed")
        self.assertTrue(result["confirmed"])
        self.assertTrue(report["confirmed"])
        self.assertEqual(report["status"], "cleanup-confirmed")
        self.assertEqual(
            report["pre_cancel_query"]["normalized_record"]["job_state"],
            "PENDING",
        )
        self.assertEqual(
            [call[0] for call in fake_subprocess.calls],
            [
                ["/usr/bin/scontrol", "show", "job", "--json", "123"],
                ["/usr/bin/scancel", "123"],
                ["/usr/bin/scontrol", "show", "job", "--json", "123"],
            ],
        )

    def test_launcher_cleanup_cancels_held_pending_candidate_and_persists_report(
        self,
    ) -> None:
        fake_subprocess = _SequentialFakeSubprocess(
            [
                _FakeCompletedProcess(returncode=0, stdout=b"cleanup"),
                _FakeCompletedProcess(returncode=0, stdout=b""),
                _FakeCompletedProcess(returncode=0, stdout=b"rollback"),
            ]
        )
        written, write_exclusive = _make_recording_write_exclusive()

        def normalize_scheduler_query(raw: bytes, *, phase: str) -> dict[str, Any]:
            if raw == b"cleanup":
                return {
                    "job_id": "123",
                    "job_name": "strict-job",
                    "comment": "strict-comment",
                    "user_id": "42",
                    "work_dir": "/snapshot",
                    "job_state": "PENDING",
                    "reason": "JobHeldUser",
                    "held": True,
                    "restart_count": 0,
                }
            if raw == b"rollback":
                return {
                    "job_id": "123",
                    "job_name": "strict-job",
                    "comment": "strict-comment",
                    "user_id": "42",
                    "work_dir": "/snapshot",
                    "job_state": "CANCELLED",
                    "reason": "Cancelled",
                    "held": False,
                    "restart_count": 0,
                }
            raise AssertionError(f"unexpected raw payload: {raw!r}")

        cleanup_authenticated_candidate = self._launcher_cleanup_function(
            fake_subprocess=fake_subprocess,
            normalize_scheduler_query=normalize_scheduler_query,
            verify_scheduler_client_environment=lambda client_environment: None,
            write_exclusive=write_exclusive,
        )
        result = cleanup_authenticated_candidate(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            host_tools={
                "scontrol": {"path": "/usr/bin/scontrol"},
                "scancel": {"path": "/usr/bin/scancel"},
            },
            client_environment={
                "variables": {
                    "SLURM_CONF": {
                        "path": "/etc/slurm.conf",
                        "sha256": "f" * 64,
                    }
                }
            },
            submission_env={"LC_ALL": "C"},
            candidate_job_id="123",
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertEqual(result["status"], "cleanup-confirmed")
        self.assertTrue(result["confirmed"])
        self.assertTrue(report["confirmed"])
        self.assertEqual(
            report["pre_cancel_query"]["normalized_record"]["reason"],
            "JobHeldUser",
        )

    def test_launcher_cleanup_refuses_cancel_on_identity_mismatch_and_persists_report(
        self,
    ) -> None:
        fake_subprocess = _SequentialFakeSubprocess([_FakeCompletedProcess(returncode=0, stdout=b"cleanup")])
        written, write_exclusive = _make_recording_write_exclusive()

        cleanup_authenticated_candidate = self._launcher_cleanup_function(
            fake_subprocess=fake_subprocess,
            normalize_scheduler_query=lambda raw, *, phase: {
                "job_id": "123",
                "job_name": "other-job",
                "comment": "strict-comment",
                "user_id": "42",
                "work_dir": "/snapshot",
                "job_state": "PENDING",
                "reason": "Priority",
                "held": False,
                "restart_count": 0,
            },
            verify_scheduler_client_environment=lambda client_environment: None,
            write_exclusive=write_exclusive,
        )
        result = cleanup_authenticated_candidate(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            host_tools={
                "scontrol": {"path": "/usr/bin/scontrol"},
                "scancel": {"path": "/usr/bin/scancel"},
            },
            client_environment={
                "variables": {
                    "SLURM_CONF": {
                        "path": "/etc/slurm.conf",
                        "sha256": "f" * 64,
                    }
                }
            },
            submission_env={"LC_ALL": "C"},
            candidate_job_id="123",
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["status"], "cleanup-identity-mismatch")
        self.assertEqual(report["status"], "cleanup-identity-mismatch")
        self.assertIsNone(report["cancellation"])
        self.assertEqual(
            [call[0] for call in fake_subprocess.calls],
            [["/usr/bin/scontrol", "show", "job", "--json", "123"]],
        )

    def test_launcher_cleanup_reports_unconfirmed_post_cancel_state(self) -> None:
        fake_subprocess = _SequentialFakeSubprocess(
            [
                _FakeCompletedProcess(returncode=0, stdout=b"cleanup"),
                _FakeCompletedProcess(returncode=0, stdout=b""),
                _FakeCompletedProcess(returncode=0, stdout=b"rollback"),
            ]
        )
        written, write_exclusive = _make_recording_write_exclusive()

        def normalize_scheduler_query(raw: bytes, *, phase: str) -> dict[str, Any]:
            if raw == b"cleanup":
                return {
                    "job_id": "123",
                    "job_name": "strict-job",
                    "comment": "strict-comment",
                    "user_id": "42",
                    "work_dir": "/snapshot",
                    "job_state": "RUNNING",
                    "reason": "None",
                    "held": False,
                    "restart_count": 0,
                }
            if raw == b"rollback":
                raise RuntimeError("post query parse failed")
            raise AssertionError(f"unexpected raw payload: {raw!r}")

        cleanup_authenticated_candidate = self._launcher_cleanup_function(
            fake_subprocess=fake_subprocess,
            normalize_scheduler_query=normalize_scheduler_query,
            verify_scheduler_client_environment=lambda client_environment: None,
            write_exclusive=write_exclusive,
        )
        result = cleanup_authenticated_candidate(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            host_tools={
                "scontrol": {"path": "/usr/bin/scontrol"},
                "scancel": {"path": "/usr/bin/scancel"},
            },
            client_environment={
                "variables": {
                    "SLURM_CONF": {
                        "path": "/etc/slurm.conf",
                        "sha256": "f" * 64,
                    }
                }
            },
            submission_env={"LC_ALL": "C"},
            candidate_job_id="123",
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["status"], "cleanup-post-query-unconfirmed")
        self.assertEqual(report["status"], "cleanup-post-query-unconfirmed")
        self.assertEqual(
            report["post_cancel_query"]["normalization_error"],
            "RuntimeError: post query parse failed",
        )
        self.assertIn("123", report["manual_recovery_hint"])
        self.assertIn("/usr/bin/scancel 123", report["manual_recovery_hint"])

    def test_launcher_persists_unknown_candidate_report_for_malformed_stdout(
        self,
    ) -> None:
        written, write_exclusive = _make_recording_write_exclusive()
        persist_unknown_candidate_report = self._unknown_candidate_report_function(write_exclusive=write_exclusive)
        result = persist_unknown_candidate_report(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            sbatch_argv=["/usr/bin/sbatch", "--parsable", "--hold"],
            completed=_FakeCompletedProcess(
                returncode=0,
                stdout=b"123;unexpected-cluster\n",
                stderr=b"",
            ),
            error=ValueError("sbatch stdout is not one job ID"),
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertEqual(result["status"], "cleanup-candidate-id-unknown")
        self.assertIsNone(report["candidate_job_id"])
        self.assertFalse(report["confirmed"])
        self.assertEqual(report["sbatch"]["status"], 0)
        self.assertEqual(
            written[report["sbatch"]["stdout"]["path"]],
            b"123;unexpected-cluster\n",
        )
        self.assertEqual(report["expected_identity"]["comment"], "strict-comment")
        self.assertIn("Never treat the captured stdout", report["manual_recovery_hint"])

    def test_launcher_persists_unknown_candidate_report_for_sbatch_timeout(
        self,
    ) -> None:
        written, write_exclusive = _make_recording_write_exclusive()
        persist_unknown_candidate_report = self._unknown_candidate_report_function(write_exclusive=write_exclusive)
        timeout = subprocess.TimeoutExpired(
            cmd=["/usr/bin/sbatch", "--parsable", "--hold"],
            timeout=30,
            output=b"partial-stdout",
            stderr=b"partial-stderr",
        )
        result = persist_unknown_candidate_report(
            submission_parent=Path("/submission"),
            attempt_id="attempt-1",
            pair_id="pair-1",
            sbatch_argv=["/usr/bin/sbatch", "--parsable", "--hold"],
            completed=None,
            error=timeout,
            expected_job_name="strict-job",
            expected_comment="strict-comment",
            expected_user_id="42",
            expected_work_dir=Path("/snapshot"),
        )
        report = _load_written_json(written, result["report_path"])
        self.assertIsNone(report["sbatch"]["status"])
        self.assertEqual(written[report["sbatch"]["stdout"]["path"]], b"partial-stdout")
        self.assertEqual(written[report["sbatch"]["stderr"]["path"]], b"partial-stderr")
        self.assertIn("TimeoutExpired", report["exception"])

    def test_launcher_main_routes_unknown_candidate_failures_to_durable_report(
        self,
    ) -> None:
        module = ast.parse(_embedded_python(_LAUNCH_SCRIPT))
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "persist_unknown_candidate_report"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertIsInstance(keywords["completed"], ast.Name)
        self.assertEqual(keywords["completed"].id, "completed")
        self.assertIsInstance(keywords["error"], ast.Name)
        self.assertEqual(keywords["error"].id, "error")

    def test_replay_shell_scripts_use_authenticated_module_imports(self) -> None:
        for script_path in _SCRIPTS:
            text = _embedded_python(script_path)
            self.assertIn("import_authenticated_module(", text)
            self.assertNotIn(
                'importlib.import_module("nemo_rl.utils.strict_captured_replay_manifest")',
                text,
            )
            self.assertNotIn(
                'importlib.import_module("nemo_rl.utils.strict_captured_replay_evidence")',
                text,
            )

    def test_wrapper_scheduler_device_env_rejects_missing_cuda_visible_devices(
        self,
    ) -> None:
        observed_scheduler_device_environment = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_scheduler_device_environment",
            extra_globals={"os": os, "fail": _raise_runtime_error},
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES is required"):
                observed_scheduler_device_environment()

    def test_wrapper_scheduler_device_env_maps_exact_five_names_to_six_keys(
        self,
    ) -> None:
        observed_scheduler_device_environment = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_scheduler_device_environment",
            extra_globals={"os": os, "fail": _raise_runtime_error},
        )
        source = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "GPU_DEVICE_ORDINAL": "0,1,2,3",
            "NVIDIA_VISIBLE_DEVICES": _CANONICAL_GPU_UUIDS,
            "ROCR_VISIBLE_DEVICES": "4,5,6,7",
            "ZE_AFFINITY_MASK": "0.0,1.0,2.0,3.0",
        }
        with mock.patch.dict(os.environ, source, clear=True):
            self.assertEqual(
                observed_scheduler_device_environment(),
                {
                    "schema": "nemo-rl-strict-scheduler-device-environment-v1",
                    "cuda_visible_devices": "0,1,2,3",
                    "gpu_device_ordinal": "0,1,2,3",
                    "nvidia_visible_devices": _CANONICAL_GPU_UUIDS,
                    "rocr_visible_devices": "4,5,6,7",
                    "ze_affinity_mask": "0.0,1.0,2.0,3.0",
                },
            )

    def test_wrapper_validates_device_boundary_before_hardware_and_reuses_it(
        self,
    ) -> None:
        module = ast.parse(_embedded_python(_WRAPPER_SCRIPT))
        assignments = {
            target.id: node
            for node in module.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        device_assignment = assignments["driver_scheduler_device_environment"]
        hardware_assignment = assignments["driver_hardware"]
        driver_assignment = assignments["driver_result"]
        exit_assignment = assignments["exit_document"]
        driver_reported_assignment = assignments["driver_reported_device_environment"]
        self.assertLess(device_assignment.lineno, hardware_assignment.lineno)
        self.assertLess(hardware_assignment.lineno, driver_assignment.lineno)
        self.assertLess(driver_assignment.lineno, exit_assignment.lineno)
        self.assertLess(driver_assignment.lineno, driver_reported_assignment.lineno)
        self.assertLess(driver_reported_assignment.lineno, exit_assignment.lineno)
        self.assertIsInstance(device_assignment.value, ast.Call)
        validator = device_assignment.value.func
        self.assertIsInstance(validator, ast.Attribute)
        self.assertEqual(validator.attr, "_validate_scheduler_device_environment")
        for assignment in (driver_assignment, exit_assignment):
            self.assertIsInstance(assignment.value, ast.Call)
            keyword = next(item for item in assignment.value.keywords if item.arg == "scheduler_device_environment")
            self.assertIsInstance(keyword.value, ast.Name)
            self.assertEqual(
                keyword.value.id,
                "driver_scheduler_device_environment",
            )
        driver_keyword = next(
            item for item in exit_assignment.value.keywords if item.arg == "driver_scheduler_device_environment"
        )
        self.assertIsInstance(driver_keyword.value, ast.Name)
        self.assertEqual(driver_keyword.value.id, "driver_reported_device_environment")

    def test_runtime_device_names_are_disjoint_from_exact_pair74(self) -> None:
        from nemo_rl.utils.strict_captured_replay_manifest import (
            SLURM_EXPORT_ALLOWED_NAMES,
        )

        runtime_device_names = {
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "NVIDIA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "ZE_AFFINITY_MASK",
        }
        self.assertEqual(len(SLURM_EXPORT_ALLOWED_NAMES), 74)
        self.assertTrue(runtime_device_names.isdisjoint(SLURM_EXPORT_ALLOWED_NAMES))

    def test_wrapper_build_driver_env_includes_scheduler_device_variables(self) -> None:
        build_driver_env = _embedded_function(
            _WRAPPER_SCRIPT,
            "build_driver_env",
            extra_globals={
                "manifest_utility": types.SimpleNamespace(
                    _load_replay_slurm_export=lambda *, source, attempt_id: (
                        None,
                        None,
                        [
                            ("PAIR_ID", b"pair-1"),
                            ("RESULTS_DIR", b"/results"),
                            ("BASE_LOG_DIR", b""),
                        ],
                    )
                )
            },
        )
        env = build_driver_env(
            {"attempt_id": "attempt-1"},
            authenticated_source=object(),
            job_id="123",
            scheduler_device_environment={
                "cuda_visible_devices": "0,1,2,3",
                "gpu_device_ordinal": "0,1,2,3",
                "nvidia_visible_devices": _CANONICAL_GPU_UUIDS,
                "rocr_visible_devices": None,
                "ze_affinity_mask": None,
            },
        )
        self.assertEqual(env["STRICT_PAIR_BOUND_JOB_ID"], "123")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(env["GPU_DEVICE_ORDINAL"], "0,1,2,3")
        self.assertEqual(
            env["NVIDIA_VISIBLE_DEVICES"],
            _CANONICAL_GPU_UUIDS,
        )
        self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        self.assertNotIn("ZE_AFFINITY_MASK", env)

    def test_wrapper_run_driver_linux_env_i_forwards_scheduler_cuda_visible_devices(
        self,
    ) -> None:
        events: list[str] = []

        class EventingSubprocess(_FakeSubprocess):
            def run(self, argv: list[str], **kwargs: Any) -> _FakeCompletedProcess:
                events.append("srun")
                return super().run(argv, **kwargs)

        manifest_utility = types.SimpleNamespace(
            _load_replay_slurm_export=lambda *, source, attempt_id: (
                None,
                None,
                [
                    ("PAIR_ID", b"pair-1"),
                    ("RESULTS_DIR", b"/results"),
                    ("BASE_LOG_DIR", b""),
                ],
            )
        )
        build_driver_env = _embedded_function(
            _WRAPPER_SCRIPT,
            "build_driver_env",
            extra_globals={"manifest_utility": manifest_utility},
        )
        build_driver_argv = _embedded_function(
            _WRAPPER_SCRIPT,
            "build_driver_argv",
            extra_globals={
                "program": {"entrypoint": {"path": "examples/nemo_gym/run_strict_captured_replay.py"}},
                "snapshot_root": Path("/snapshot"),
            },
        )
        container_mounts = _embedded_function(
            _WRAPPER_SCRIPT,
            "container_mounts",
            extra_globals={"snapshot_root": Path("/snapshot")},
        )
        fake_subprocess = EventingSubprocess(_FakeCompletedProcess(returncode=0, stdout=b"driver-ok"))
        container_hashes: list[tuple[str, str, int, int]] = []
        scheduler_verifications: list[dict[str, Any]] = []
        run_driver = _embedded_function(
            _WRAPPER_SCRIPT,
            "run_driver",
            extra_globals={
                "authenticated_source": object(),
                "build_driver_argv": build_driver_argv,
                "build_driver_env": build_driver_env,
                "container_mounts": container_mounts,
                "LINUX_SRUN_PATH": Path("/usr/bin/srun"),
                "LINUX_SRUN_SHA256": "d" * 64,
                "live_job_id": "123",
                "manifest_path": Path("/control/replay-manifest.json"),
                "manifest_sha256": "e" * 64,
                "program": {"entrypoint": {"path": "examples/nemo_gym/run_strict_captured_replay.py"}},
                "snapshot_root": Path("/snapshot"),
                "canonical_absolute_path": lambda value, label: Path(value),
                "stable_container_image_sha256": (
                    lambda path, *, label, expected_sha256, expected_owner_uid, expected_owner_gid: (
                        events.append(label),
                        container_hashes.append(
                            (
                                str(path),
                                expected_sha256,
                                expected_owner_uid,
                                expected_owner_gid,
                            )
                        ),
                    )[-1]
                ),
                "stable_tool_bytes": lambda path, *, label, expected_sha256: b"tool",
                "subprocess": fake_subprocess,
                "sys": types.SimpleNamespace(platform="linux"),
                "verify_scheduler_client_environment": (
                    lambda value: (
                        events.append("slurm-client"),
                        scheduler_verifications.append(value),
                    )[-1]
                ),
            },
        )
        client_environment = {"variables": {"SLURM_CONF": {"path": "/etc/slurm.conf", "sha256": "a" * 64}}}
        submission_env = {"LC_ALL": "C", "SLURM_CONF": "/etc/slurm.conf"}
        run_driver(
            manifest={
                "attempt_id": "attempt-1",
                "runtime_tools": {"document": {"container": {"python": {"path": "/usr/bin/python3"}}}},
                "replay_contract": {
                    "gym_scorer": {
                        "container": {
                            "path": "/container.sqsh",
                            "sha256": "c" * 64,
                            "owner_uid": 153493,
                            "owner_gid": 30,
                        }
                    }
                },
            },
            pair_manifest={
                "paths": {
                    "results_root": "/results",
                    "cache_root": "/cache",
                    "hf_home": "/hf",
                }
            },
            pre_receipt_path=Path("/control/PRE.json"),
            pre_receipt_sha256="f" * 64,
            job={"account": "acct", "partition": "batch"},
            scheduler_device_environment={
                "cuda_visible_devices": "0,1,2,3",
                "gpu_device_ordinal": "0,1,2,3",
                "nvidia_visible_devices": _CANONICAL_GPU_UUIDS,
                "rocr_visible_devices": None,
                "ze_affinity_mask": None,
            },
            client_environment=client_environment,
            submission_env=submission_env,
        )
        argv = fake_subprocess.calls[0][0]
        env_start = argv.index("-i") + 1
        python_index = argv.index("/usr/bin/python3")
        self.assertEqual(
            argv[env_start:python_index],
            sorted(
                (
                    "BASE_LOG_DIR=",
                    "CUDA_VISIBLE_DEVICES=0,1,2,3",
                    "GPU_DEVICE_ORDINAL=0,1,2,3",
                    f"NVIDIA_VISIBLE_DEVICES={_CANONICAL_GPU_UUIDS}",
                    "PAIR_ID=pair-1",
                    "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
                    "RESULTS_DIR=/results",
                    "STRICT_PAIR_BOUND_JOB_ID=123",
                )
            ),
        )
        self.assertFalse(any(value.startswith("ROCR_VISIBLE_DEVICES=") for value in argv))
        self.assertFalse(any(value.startswith("ZE_AFFINITY_MASK=") for value in argv))
        self.assertEqual(
            events,
            [
                "authenticated replay container image before srun",
                "slurm-client",
                "srun",
                "authenticated replay container image after srun",
            ],
        )
        self.assertEqual(
            container_hashes,
            [
                ("/container.sqsh", "c" * 64, 153493, 30),
                ("/container.sqsh", "c" * 64, 153493, 30),
            ],
        )
        self.assertEqual(scheduler_verifications, [client_environment])
        self.assertEqual(fake_subprocess.calls[0][1]["env"], submission_env)

    def test_wrapper_stable_container_hash_rejects_current_byte_mutation(
        self,
    ) -> None:
        foreign_os = _ForeignContainerOS()
        stable_container_image_sha256 = _embedded_function(
            _WRAPPER_SCRIPT,
            "stable_container_image_sha256",
            extra_globals={
                "DIGEST_RE": re.compile(r"[0-9a-f]{64}\Z"),
                "canonical_absolute_path": lambda value, label: Path(value),
                "fail": _raise_runtime_error,
                "hashlib": hashlib,
                "os": foreign_os,
                "stat": __import__("stat"),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "replay.sqsh"
            image.write_bytes(b"authenticated-container")
            image.chmod(0o644)
            expected = hashlib.sha256(b"authenticated-container").hexdigest()
            self.assertEqual(
                stable_container_image_sha256(
                    image,
                    label="replay container",
                    expected_sha256=expected,
                    expected_owner_uid=153493,
                    expected_owner_gid=30,
                ),
                expected,
            )
            image.chmod(0o644)
            image.write_bytes(b"mutated-container")
            image.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "digest differs"):
                stable_container_image_sha256(
                    image,
                    label="replay container",
                    expected_sha256=expected,
                    expected_owner_uid=153493,
                    expected_owner_gid=30,
                )

    def test_wrapper_stable_container_hash_rejects_hardlink_alias(self) -> None:
        foreign_os = _ForeignContainerOS()
        stable_container_image_sha256 = _embedded_function(
            _WRAPPER_SCRIPT,
            "stable_container_image_sha256",
            extra_globals={
                "DIGEST_RE": re.compile(r"[0-9a-f]{64}\Z"),
                "canonical_absolute_path": lambda value, label: Path(value),
                "fail": _raise_runtime_error,
                "hashlib": hashlib,
                "os": foreign_os,
                "stat": __import__("stat"),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "replay.sqsh"
            alias = Path(directory) / "replay-alias.sqsh"
            image.write_bytes(b"authenticated-container")
            image.chmod(0o644)
            os.link(image, alias)
            expected = hashlib.sha256(b"authenticated-container").hexdigest()
            with self.assertRaisesRegex(RuntimeError, "exactly one hard link"):
                stable_container_image_sha256(
                    image,
                    label="replay container",
                    expected_sha256=expected,
                    expected_owner_uid=153493,
                    expected_owner_gid=30,
                )

    def test_wrapper_container_admits_only_named_foreign_effective_readonly_asset(
        self,
    ) -> None:
        cases = (
            (
                "wrong-actual-uid",
                _ForeignContainerOS(owner_uid=153494),
                0o644,
                153493,
                30,
                "authenticated regular host file",
            ),
            (
                "wrong-actual-gid",
                _ForeignContainerOS(owner_gid=31),
                0o644,
                153493,
                30,
                "authenticated regular host file",
            ),
            (
                "root-replay-process",
                _ForeignContainerOS(effective_uid=0),
                0o644,
                153493,
                30,
                "publisher must be foreign",
            ),
            (
                "publisher-replay-process",
                _ForeignContainerOS(effective_uid=153493),
                0o644,
                153493,
                30,
                "publisher must be foreign",
            ),
            (
                "group-writable",
                _ForeignContainerOS(),
                0o664,
                153493,
                30,
                "group/other writable",
            ),
            (
                "other-writable",
                _ForeignContainerOS(),
                0o646,
                153493,
                30,
                "group/other writable",
            ),
            (
                "effective-write-access",
                _ForeignContainerOS(effectively_writable=True),
                0o644,
                153493,
                30,
                "writable by the replay process",
            ),
            (
                "wrong-declared-uid",
                _ForeignContainerOS(),
                0o644,
                153494,
                30,
                "publisher identity is not admitted",
            ),
            (
                "bool-declared-uid",
                _ForeignContainerOS(),
                0o644,
                True,
                30,
                "publisher identity is not admitted",
            ),
            (
                "wrong-declared-gid",
                _ForeignContainerOS(),
                0o644,
                153493,
                31,
                "publisher identity is not admitted",
            ),
            (
                "bool-declared-gid",
                _ForeignContainerOS(),
                0o644,
                153493,
                False,
                "publisher identity is not admitted",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "replay.sqsh"
            image.write_bytes(b"authenticated-container")
            expected = hashlib.sha256(image.read_bytes()).hexdigest()
            for name, foreign_os, mode, owner_uid, owner_gid, message in cases:
                with self.subTest(name=name):
                    image.chmod(mode)
                    stable_container_image_sha256 = _embedded_function(
                        _WRAPPER_SCRIPT,
                        "stable_container_image_sha256",
                        extra_globals={
                            "DIGEST_RE": re.compile(r"[0-9a-f]{64}\Z"),
                            "canonical_absolute_path": (lambda value, label: Path(value)),
                            "fail": _raise_runtime_error,
                            "hashlib": hashlib,
                            "os": foreign_os,
                            "stat": __import__("stat"),
                        },
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        stable_container_image_sha256(
                            image,
                            label="replay container",
                            expected_sha256=expected,
                            expected_owner_uid=owner_uid,
                            expected_owner_gid=owner_gid,
                        )

    def test_wrapper_rejects_container_mutation_after_successful_srun(self) -> None:
        run_driver = _embedded_function(
            _WRAPPER_SCRIPT,
            "run_driver",
            extra_globals={
                "authenticated_source": object(),
                "build_driver_argv": lambda **kwargs: ["/entrypoint.py"],
                "build_driver_env": lambda *args, **kwargs: {"PAIR_ID": "pair-1"},
                "canonical_absolute_path": lambda value, label: Path(value),
                "container_mounts": lambda **kwargs: "/snapshot:/snapshot:ro",
                "LINUX_SRUN_PATH": Path("/usr/bin/srun"),
                "LINUX_SRUN_SHA256": "d" * 64,
                "live_job_id": "123",
                "manifest_path": Path("/control/replay-manifest.json"),
                "manifest_sha256": "e" * 64,
                "program": {"entrypoint": {"path": "entrypoint.py"}},
                "snapshot_root": Path("/snapshot"),
                "stable_tool_bytes": lambda *args, **kwargs: b"tool",
                "subprocess": _FakeSubprocess(_FakeCompletedProcess(returncode=0, stdout=b"driver-ok")),
                "sys": types.SimpleNamespace(platform="linux"),
                "verify_scheduler_client_environment": lambda value: None,
                "stable_container_image_sha256": self._post_srun_mutation_gate(),
            },
        )
        with self.assertRaisesRegex(RuntimeError, "post-srun container mutation"):
            run_driver(
                manifest={
                    "attempt_id": "attempt-1",
                    "runtime_tools": {"document": {"container": {"python": {"path": "/usr/bin/python3"}}}},
                    "replay_contract": {
                        "gym_scorer": {
                            "container": {
                                "path": "/container.sqsh",
                                "sha256": "c" * 64,
                                "owner_uid": 153493,
                                "owner_gid": 30,
                            }
                        }
                    },
                },
                pair_manifest={
                    "paths": {
                        "results_root": "/results",
                        "cache_root": "/cache",
                        "hf_home": "/hf",
                    }
                },
                pre_receipt_path=Path("/control/PRE.json"),
                pre_receipt_sha256="f" * 64,
                job={"account": "acct", "partition": "batch"},
                scheduler_device_environment={
                    "cuda_visible_devices": "0,1,2,3",
                    "gpu_device_ordinal": None,
                    "nvidia_visible_devices": None,
                    "rocr_visible_devices": None,
                    "ze_affinity_mask": None,
                },
                client_environment={"variables": {}},
                submission_env={"LC_ALL": "C"},
            )

    def _post_srun_mutation_gate(self) -> Any:
        calls = 0

        def gate(
            path: Path,
            *,
            label: str,
            expected_sha256: str,
            expected_owner_uid: int,
            expected_owner_gid: int,
        ) -> str:
            nonlocal calls
            del path, label, expected_sha256
            self.assertEqual((expected_owner_uid, expected_owner_gid), (153493, 30))
            calls += 1
            if calls == 2:
                raise RuntimeError("post-srun container mutation")
            return "c" * 64

        return gate

    def test_wrapper_observed_hardware_requires_exactly_four_rows(self) -> None:
        fake_subprocess = _FakeSubprocess(
            _FakeCompletedProcess(
                returncode=0,
                stdout=(b"NVIDIA GB200, 580.126.20\n" b"NVIDIA GB200, 580.126.20\n" b"NVIDIA GB200, 580.126.20\n"),
            )
        )
        observed_hardware = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_hardware",
            extra_globals={
                "canonical_absolute_path": lambda value, label: Path(value),
                "fail": _raise_runtime_error,
                "manifest": {
                    "runtime_tools": {
                        "document": {
                            "host": {
                                "nvidia_smi": {
                                    "path": "/usr/bin/nvidia-smi",
                                    "sha256": "d" * 64,
                                }
                            }
                        }
                    }
                },
                "stable_tool_bytes": lambda path, *, label, expected_sha256: b"tool",
                "subprocess": fake_subprocess,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "exactly 4 GPU rows"):
            observed_hardware()
        self.assertEqual(len(fake_subprocess.calls), 1)

    def test_wrapper_observed_hardware_retains_exact_order_and_digests(self) -> None:
        from nemo_rl.utils.strict_captured_replay_evidence import domain_sha256

        raw_output = b"NVIDIA GB200, 580.126.20\n" * 4
        fake_subprocess = _FakeSubprocess(_FakeCompletedProcess(returncode=0, stdout=raw_output))
        observed_hardware = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_hardware",
            extra_globals={
                "canonical_absolute_path": lambda value, label: Path(value),
                "evidence_utility": types.SimpleNamespace(domain_sha256=domain_sha256),
                "fail": _raise_runtime_error,
                "hashlib": hashlib,
                "manifest": {
                    "runtime_tools": {
                        "document": {
                            "host": {
                                "nvidia_smi": {
                                    "path": "/usr/bin/nvidia-smi",
                                    "sha256": "d" * 64,
                                }
                            }
                        }
                    }
                },
                "stable_tool_bytes": lambda path, *, label, expected_sha256: b"tool",
                "subprocess": fake_subprocess,
            },
        )
        hardware = observed_hardware()
        self.assertEqual(hardware["gpu_row_count"], 4)
        self.assertEqual([row["index"] for row in hardware["ordered_rows"]], [0, 1, 2, 3])
        self.assertEqual(hardware["raw_output_sha256"], hashlib.sha256(raw_output).hexdigest())
        self.assertEqual(
            hardware["ordered_rows_sha256"],
            domain_sha256(
                "captured-replay-nvidia-smi-ordered-rows-v1",
                hardware["ordered_rows"],
            ),
        )

    def test_wrapper_observed_hardware_rejects_noncanonical_row_spacing(self) -> None:
        fake_subprocess = _FakeSubprocess(
            _FakeCompletedProcess(
                returncode=0,
                stdout=b"NVIDIA GB200 ,580.126.20\n" * 4,
            )
        )
        observed_hardware = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_hardware",
            extra_globals={
                "canonical_absolute_path": lambda value, label: Path(value),
                "fail": _raise_runtime_error,
                "manifest": {
                    "runtime_tools": {
                        "document": {
                            "host": {
                                "nvidia_smi": {
                                    "path": "/usr/bin/nvidia-smi",
                                    "sha256": "d" * 64,
                                }
                            }
                        }
                    }
                },
                "stable_tool_bytes": lambda path, *, label, expected_sha256: b"tool",
                "subprocess": fake_subprocess,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "GPU row formatting differs"):
            observed_hardware()

    def test_wrapper_observed_hardware_has_no_darwin_bypass(self) -> None:
        fake_subprocess = _FakeSubprocess(
            _FakeCompletedProcess(
                returncode=0,
                stdout=(
                    b"NVIDIA GB200, 580.126.20\n"
                    b"NVIDIA GB200, 580.126.20\n"
                    b"NVIDIA A100, 580.126.20\n"
                    b"NVIDIA GB200, 580.126.20\n"
                ),
            )
        )
        observed_hardware = _embedded_function(
            _WRAPPER_SCRIPT,
            "observed_hardware",
            extra_globals={
                "canonical_absolute_path": lambda value, label: Path(value),
                "fail": _raise_runtime_error,
                "manifest": {
                    "runtime_tools": {
                        "document": {
                            "host": {
                                "nvidia_smi": {
                                    "path": "/usr/bin/nvidia-smi",
                                    "sha256": "e" * 64,
                                }
                            }
                        }
                    }
                },
                "stable_tool_bytes": lambda path, *, label, expected_sha256: b"tool",
                "subprocess": fake_subprocess,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "compute hardware differs from required"):
            observed_hardware()
        self.assertEqual(len(fake_subprocess.calls), 1)
