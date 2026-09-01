from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"
BEGIN = "# BEGIN RAY_WORKER_SESSION_LATEST_RESOLVER\n"
END = "# END RAY_WORKER_SESSION_LATEST_RESOLVER\n"
MANIFEST_BEGIN = "# BEGIN RAY_RUNTIME_ARTIFACT_MANIFEST_HELPER\n"
MANIFEST_END = "# END RAY_RUNTIME_ARTIFACT_MANIFEST_HELPER\n"


def ray_command_source(command: str) -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    start = f"{command}_cmd=$(cat <<EOF\n"
    if source.count(start) != 1:
        raise AssertionError(f"{command} command heredoc must be unique")
    body = source.split(start, 1)[1]
    end = "\nEOF\n)\n"
    if body.count(end) < 1:
        raise AssertionError(f"{command} command heredoc must be terminated")
    return body.split(end, 1)[0]


def bash_function_source(source: str, function: str) -> str:
    start = f"{function}() {{\n"
    if source.count(start) != 1:
        raise AssertionError(f"{function} definition must be unique")
    body = source.split(start, 1)[1]
    end = "\n}\n"
    if end not in body:
        raise AssertionError(f"{function} definition must be terminated")
    return start + body.split(end, 1)[0] + end


def resolver_source() -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise AssertionError("Ray worker session resolver sentinels must be unique")
    return source.split(BEGIN, 1)[1].split(END, 1)[0]


def runtime_manifest_source() -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    if source.count(MANIFEST_BEGIN) != 1 or source.count(MANIFEST_END) != 1:
        raise AssertionError("Ray runtime artifact helper sentinels must be unique")
    return MANIFEST_BEGIN + source.split(MANIFEST_BEGIN, 1)[1].split(MANIFEST_END, 1)[0]


def write_runtime_evidence(
    log_dir: Path,
    *,
    job_id: str,
    restart_count: int,
    session: str,
    expected_workers: int,
) -> None:
    attestation_dir = log_dir / "RAY_SESSION_ATTESTATIONS"
    completion_dir = log_dir / "RAY_SESSION_COMPLETIONS"
    attestation_dir.mkdir(parents=True)
    completion_dir.mkdir()
    (log_dir / "RAY_SESSION_NAME").write_text(f"{session}\n", encoding="utf-8")

    head_node = "head-node"
    (attestation_dir / "head").write_text(
        f"role=head\nnode={head_node}\nsession={session}\n",
        encoding="utf-8",
    )
    (completion_dir / "head").write_text(
        "schema=ray-session-completion-v1\n"
        "role=head\n"
        f"job_id={job_id}\n"
        f"restart_count={restart_count}\n"
        f"node={head_node}\n"
        f"session={session}\n"
        "exit_code=0\n"
        "sync=complete\n",
        encoding="utf-8",
    )
    for worker_index in range(expected_workers):
        worker_node = f"worker-node-{worker_index}"
        (attestation_dir / f"worker-{worker_index}").write_text(
            "role=worker\n"
            f"worker_index={worker_index}\n"
            f"node={worker_node}\n"
            f"session={session}\n",
            encoding="utf-8",
        )
        (completion_dir / f"worker-{worker_index}").write_text(
            "schema=ray-session-completion-v1\n"
            "role=worker\n"
            f"worker_index={worker_index}\n"
            f"job_id={job_id}\n"
            f"restart_count={restart_count}\n"
            f"node={worker_node}\n"
            f"session={session}\n"
            "exit_code=0\n"
            "sync=complete\n",
            encoding="utf-8",
        )

    (log_dir / "RAY_JOB_COMPLETION").write_text(
        "schema=ray-session-completion-v1\n"
        "role=job\n"
        f"job_id={job_id}\n"
        f"restart_count={restart_count}\n"
        f"session={session}\n"
        f"expected_workers={expected_workers}\n"
        "status=complete\n"
        "exit_code=0\n",
        encoding="utf-8",
    )
    (log_dir / "ray-head.log").write_text("head finalized\n", encoding="utf-8")
    (log_dir / "ray-driver.log").write_text("driver finalized\n", encoding="utf-8")
    worker_width = len(str(expected_workers))
    for worker_index in range(expected_workers):
        (log_dir / f"ray-worker-{worker_index:0{worker_width}d}.log").write_text(
            f"worker {worker_index} finalized\n",
            encoding="utf-8",
        )


def run_runtime_manifest(
    log_dir: Path,
    *,
    job_id: str,
    restart_count: int,
    session: str,
    expected_workers: int,
) -> subprocess.CompletedProcess[str]:
    manifest = log_dir / "RAY_RUNTIME_ARTIFACTS.json"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            runtime_manifest_source(),
            str(log_dir),
            str(manifest),
            job_id,
            str(restart_count),
            session,
            str(expected_workers),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def make_runtime_evidence_writable(log_dir: Path) -> None:
    """Restore directory write bits so TemporaryDirectory can clean up."""
    for directory in (
        log_dir,
        log_dir / "RAY_SESSION_ATTESTATIONS",
        log_dir / "RAY_SESSION_COMPLETIONS",
    ):
        if directory.exists():
            directory.chmod(0o755)


def run_resolver(
    path: Path, *, prelude: str | None = None
) -> subprocess.CompletedProcess[str]:
    if prelude is None:
        # macOS readlink lacks GNU -e.  This shim implements the production
        # all-components-exist contract while retaining the host's -f resolver.
        prelude = r"""readlink() {
  [[ "$1" == "-e" && "$2" == "--" ]] || return 64
  [[ -e "$3" ]] || return 1
  /usr/bin/readlink -f "$3"
}"""
    script = (
        "set -euo pipefail\n"
        + resolver_source()
        + "\n"
        + prelude
        + '\nresolve_ray_session_latest "$1"\n'
    )
    return subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script, "resolver", str(path)],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        check=False,
        capture_output=True,
        text=True,
    )


class RayWorkerSessionLatestResolverTest(unittest.TestCase):
    def test_driver_identity_survives_slurm_environment_purge(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        head = ray_command_source("head")
        preserve_id = 'readonly NRL_SLURM_JOB_ID="$SLURM_JOB_ID"'
        preserve_name = 'readonly NRL_SLURM_JOB_NAME="$SLURM_JOB_NAME"'
        purge = "'/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print \\$1}'"

        self.assertEqual(source.count(preserve_id), 1)
        self.assertEqual(source.count(preserve_name), 1)
        self.assertLess(source.index(preserve_id), source.index("head_cmd=$(cat <<EOF"))
        self.assertLess(
            source.index(preserve_name), source.index("head_cmd=$(cat <<EOF")
        )
        self.assertIn("export NRL_SLURM_JOB_ID NRL_SLURM_JOB_NAME", source)
        self.assertIn(purge, head)
        self.assertIn('bash "$DRIVER_COMMAND_FILE"', head)

    def test_absent_path_does_not_call_gnu_like_readlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "session_latest"
            self.assertFalse(latest.exists())
            result = run_resolver(
                latest,
                prelude='readlink() { [[ "$1" == "-e" ]] || printf "%s\\n" "$3"; return 1; }',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "\n")

    def test_real_symlink_resolves_to_concrete_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session_2026-08-29_00-00-00_1_1"
            session.mkdir()
            latest = root / "session_latest"
            latest.symlink_to(session, target_is_directory=True)
            result = run_resolver(latest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), session.resolve())

    def test_dangling_symlink_waits_without_calling_readlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "session_latest"
            latest.symlink_to(root / "session_not_created", target_is_directory=True)
            result = run_resolver(
                latest,
                prelude='readlink() { [[ "$1" == "-e" ]] || printf "%s\\n" "$3"; return 1; }',
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "\n")

    def test_reserved_regular_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "session_latest"
            latest.mkdir()
            result = run_resolver(latest)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")

    def test_expandable_ray_command_heredocs_contain_no_backticks(self) -> None:
        head = ray_command_source("head")
        worker = ray_command_source("worker")
        self.assertNotIn("`", head)
        self.assertNotIn("`", worker)

    def test_head_functions_are_not_exported_to_ray_runtime_env(self) -> None:
        head = ray_command_source("head")

        # Ray snapshots the daemon's complete environment when it constructs an
        # actor runtime_env.  Exporting a Bash function therefore injects a
        # BASH_FUNC_* variable into every actor, even though the function is only
        # a head-shell implementation detail.
        self.assertNotIn("export -f", head)
        exit_function = bash_function_source(head, "exit-dramatically").replace(
            r"\$", "$"
        )
        result = subprocess.run(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                exit_function + "/usr/bin/env\n",
            ],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("BASH_FUNC_", result.stdout)

    def test_head_sidecars_are_pid_tracked_stopped_and_reaped(self) -> None:
        head = ray_command_source("head")
        pid_by_sidecar = {
            "monitor-sidecar": "monitor_sidecar_pid",
            "log-sync-sidecar": "log_sync_sidecar_pid",
            "ray-status-sidecar": "ray_status_sidecar_pid",
        }
        for sidecar, pid_variable in pid_by_sidecar.items():
            launch = f"{sidecar} &\n{pid_variable}=\\$!"
            self.assertEqual(head.count(launch), 1)

        stop_function = bash_function_source(head, "stop-head-sidecars").replace(
            r"\$", "$"
        )
        self.assertIn('kill -TERM "$pid"', stop_function)
        self.assertIn('wait "$pid"', stop_function)

        # Exercise the exact production cleanup function against three real
        # asynchronous children.  The static wait assertion above distinguishes
        # explicit reaping from merely signalling the processes.
        assignments = "\n".join(
            f"busy_sidecar &\n{pid_variable}=$!"
            for pid_variable in pid_by_sidecar.values()
        )
        script = (
            "set -euo pipefail\n"
            "busy_sidecar() { while :; do :; done; }\n"
            + stop_function
            + assignments
            + "\nstop-head-sidecars\n"
            + '[[ -z "$(jobs -pr)" ]]\n'
            + "printf 'HEAD_SIDECARS_REAPED\\n'\n"
        )
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", script],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "HEAD_SIDECARS_REAPED\n")

        finalizer = bash_function_source(head, "finalize-head-log-sync")
        self.assertLess(
            finalizer.index("stop-head-sidecars"),
            finalizer.index("atomic-final-head-logs"),
        )

    def test_worker_sidecars_are_pid_tracked_stopped_and_reaped(self) -> None:
        worker = ray_command_source("worker")
        pid_by_sidecar = {
            "monitor-sidecar": "worker_monitor_sidecar_pid",
            "log-sync-sidecar": "worker_log_sync_sidecar_pid",
        }
        for sidecar, pid_variable in pid_by_sidecar.items():
            launch = f"{sidecar} &\n{pid_variable}=\\$!"
            self.assertEqual(worker.count(launch), 1)

        stop_function = bash_function_source(worker, "stop-worker-sidecars").replace(
            r"\$", "$"
        )
        self.assertIn('kill -TERM "$pid"', stop_function)
        self.assertIn('wait "$pid"', stop_function)

        assignments = "\n".join(
            f"busy_sidecar &\n{pid_variable}=$!"
            for pid_variable in pid_by_sidecar.values()
        )
        script = (
            "set -euo pipefail\n"
            "busy_sidecar() { while :; do :; done; }\n"
            + stop_function
            + assignments
            + "\nstop-worker-sidecars\n"
            + '[[ -z "$(jobs -pr)" ]]\n'
            + "printf 'WORKER_SIDECARS_REAPED\\n'\n"
        )
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", script],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "WORKER_SIDECARS_REAPED\n")

        finalizer = bash_function_source(worker, "finalize-worker-log-sync")
        self.assertLess(
            finalizer.index("stop-worker-sidecars"),
            finalizer.index("atomic-final-worker-logs"),
        )

    def test_declare_f_injection_survives_expandable_heredoc(self) -> None:
        render_script = "set -euo pipefail\n" + resolver_source() + r"""
RAY_WORKER_SESSION_LATEST_RESOLVER="$(declare -f resolve_ray_session_latest)"
readlink() { printf 'RENDER_EXECUTED\n' >&2; return 99; }
cat <<EOF
$RAY_WORKER_SESSION_LATEST_RESOLVER
EOF
"""
        rendered = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", render_script],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertEqual(rendered.stderr, "")
        self.assertEqual(rendered.stdout.count("resolve_ray_session_latest ()"), 1)
        self.assertEqual(rendered.stdout.count('readlink -e -- "$latest_path"'), 1)
        syntax = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-n"],
            input=rendered.stdout,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)


class RayRuntimeArtifactManifestTest(unittest.TestCase):
    def test_success_manifest_binds_and_seals_exact_runtime_evidence(self) -> None:
        job_id = "72416"
        restart_count = 2
        session = "session_2026-08-31_00-00-00_72416_1"
        expected_workers = 2
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72416-2-logs"
            write_runtime_evidence(
                log_dir,
                job_id=job_id,
                restart_count=restart_count,
                session=session,
                expected_workers=expected_workers,
            )
            try:
                result = run_runtime_manifest(
                    log_dir,
                    job_id=job_id,
                    restart_count=restart_count,
                    session=session,
                    expected_workers=expected_workers,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

                manifest_path = log_dir / "RAY_RUNTIME_ARTIFACTS.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["schema"], "ray-runtime-artifacts-v1")
                self.assertEqual(manifest["manifest_path"], str(manifest_path))
                self.assertEqual(manifest["job_id"], job_id)
                self.assertEqual(manifest["restart_count"], restart_count)
                self.assertEqual(manifest["session"], session)
                self.assertEqual(manifest["expected_workers"], expected_workers)
                self.assertEqual(manifest["artifact_count"], 12)
                self.assertEqual(
                    [artifact["name"] for artifact in manifest["artifacts"]],
                    [
                        "session_name",
                        "attestation/head",
                        "attestation/worker-0",
                        "attestation/worker-1",
                        "completion/head",
                        "completion/worker-0",
                        "completion/worker-1",
                        "job_completion",
                        "log/head",
                        "log/driver",
                        "log/worker-0",
                        "log/worker-1",
                    ],
                )
                self.assertEqual(
                    manifest["integrity"],
                    {
                        "artifact_mode": "0444",
                        "artifacts_sealed": True,
                        "artifacts_fsynced": True,
                        "directory_mode": "0555",
                        "directories_sealed": True,
                        "directories_fsynced": True,
                        "manifest_mode": "0444",
                        "manifest_sealed": True,
                        "manifest_fsynced": True,
                        "atomic_publish": True,
                    },
                )

                for artifact in manifest["artifacts"]:
                    artifact_path = Path(artifact["path"])
                    artifact_stat = artifact_path.stat(follow_symlinks=False)
                    self.assertTrue(stat.S_ISREG(artifact_stat.st_mode))
                    self.assertEqual(stat.S_IMODE(artifact_stat.st_mode), 0o444)
                    self.assertEqual(artifact["mode"], "0444")
                    self.assertEqual(artifact["size"], artifact_stat.st_size)
                    self.assertEqual(artifact["device"], artifact_stat.st_dev)
                    self.assertEqual(artifact["inode"], artifact_stat.st_ino)
                    self.assertEqual(artifact["mtime_ns"], artifact_stat.st_mtime_ns)
                    self.assertEqual(
                        artifact["sha256"],
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    )

                for identity in manifest["directories"].values():
                    directory_path = Path(identity["path"])
                    directory_stat = directory_path.stat(follow_symlinks=False)
                    self.assertTrue(stat.S_ISDIR(directory_stat.st_mode))
                    self.assertEqual(stat.S_IMODE(directory_stat.st_mode), 0o555)
                    self.assertEqual(identity["mode"], "0555")
                    self.assertEqual(identity["device"], directory_stat.st_dev)
                    self.assertEqual(identity["inode"], directory_stat.st_ino)
                    self.assertNotIn("mtime_ns", identity)

                self.assertEqual(
                    stat.S_IMODE(manifest_path.stat().st_mode),
                    0o444,
                )
                self.assertFalse(
                    list(log_dir.glob(".RAY_RUNTIME_ARTIFACTS.json.*.tmp"))
                )
            finally:
                make_runtime_evidence_writable(log_dir)

    def test_missing_bound_artifact_fails_without_success_manifest(self) -> None:
        job_id = "72417"
        restart_count = 0
        session = "session_2026-08-31_00-00-00_72417_1"
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72417-logs"
            write_runtime_evidence(
                log_dir,
                job_id=job_id,
                restart_count=restart_count,
                session=session,
                expected_workers=1,
            )
            (log_dir / "ray-driver.log").unlink()
            try:
                result = run_runtime_manifest(
                    log_dir,
                    job_id=job_id,
                    restart_count=restart_count,
                    session=session,
                    expected_workers=1,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ray-driver.log", result.stderr)
                self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())
                self.assertFalse(
                    list(log_dir.glob(".RAY_RUNTIME_ARTIFACTS.json.*.tmp"))
                )
            finally:
                make_runtime_evidence_writable(log_dir)

    def test_multi_digit_worker_count_binds_zero_padded_complete_set(self) -> None:
        job_id = "72419"
        restart_count = 0
        session = "session_2026-08-31_00-00-00_72419_1"
        expected_workers = 12
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72419-logs"
            write_runtime_evidence(
                log_dir,
                job_id=job_id,
                restart_count=restart_count,
                session=session,
                expected_workers=expected_workers,
            )
            try:
                result = run_runtime_manifest(
                    log_dir,
                    job_id=job_id,
                    restart_count=restart_count,
                    session=session,
                    expected_workers=expected_workers,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                manifest = json.loads(
                    (log_dir / "RAY_RUNTIME_ARTIFACTS.json").read_text(encoding="utf-8")
                )
                artifacts = {
                    artifact["name"]: artifact for artifact in manifest["artifacts"]
                }
                self.assertEqual(manifest["artifact_count"], 42)
                self.assertEqual(
                    Path(artifacts["log/worker-0"]["path"]).name,
                    "ray-worker-00.log",
                )
                self.assertEqual(
                    Path(artifacts["log/worker-11"]["path"]).name,
                    "ray-worker-11.log",
                )
                self.assertEqual(
                    len(
                        [
                            name
                            for name in artifacts
                            if name.startswith("attestation/worker-")
                        ]
                    ),
                    expected_workers,
                )
                self.assertEqual(
                    len(
                        [
                            name
                            for name in artifacts
                            if name.startswith("completion/worker-")
                        ]
                    ),
                    expected_workers,
                )
            finally:
                make_runtime_evidence_writable(log_dir)

    def test_exit_handler_is_the_only_success_publication_boundary(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        exit_handler = bash_function_source(source, "_nrl_finish_exit")
        success_guard = 'if [[ "$rc" -eq 0 ]]; then'
        publish_call = (
            'publish-runtime-artifact-manifest "$session_name" "$expected_workers"'
        )
        self.assertEqual(exit_handler.count(success_guard), 1)
        self.assertEqual(exit_handler.count(publish_call), 1)
        self.assertLess(
            exit_handler.index(success_guard), exit_handler.index(publish_call)
        )
        self.assertLess(
            exit_handler.index(publish_call),
            exit_handler.index('completion_status="complete"'),
        )
        self.assertEqual(source.count(publish_call), 1)

        noninteractive = source.rsplit('if [[ -n "$COMMAND" ]]; then', 1)[1].split(
            "else\n  # Interactive:", 1
        )[0]
        self.assertLess(
            noninteractive.rindex("write-job-completion-receipt"),
            noninteractive.rindex('exit "$final_exit_code"'),
        )

    def test_exit_handler_forces_nonzero_when_publication_fails(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        exit_handler = bash_function_source(source, "_nrl_finish_exit")
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72418-logs"
            log_dir.mkdir()
            (log_dir / "RAY_SESSION_NAME").write_text(
                "session_2026-08-31_00-00-00_72418_1\n",
                encoding="utf-8",
            )
            script = (
                "set -euo pipefail\n"
                "log_phase() { :; }\n"
                "publish-runtime-artifact-manifest() { return 1; }\n"
                f"{exit_handler}\n"
                "LOG_DIR=$1\n"
                "SLURM_JOB_ID=72418\n"
                "SLURM_JOB_NUM_NODES=2\n"
                "JOB_RESTART_COUNT=0\n"
                "_nrl_finish_exit 0 TEST\n"
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "exit-handler",
                    str(log_dir),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(
                "Failed to publish sealed Ray runtime artifact manifest", result.stderr
            )
            self.assertIn("status=failed exit_code=1", result.stdout)
            self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())

    def test_exit_handler_never_publishes_for_preexisting_failure(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        exit_handler = bash_function_source(source, "_nrl_finish_exit")
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72420-logs"
            log_dir.mkdir()
            (log_dir / "RAY_SESSION_NAME").write_text(
                "session_2026-08-31_00-00-00_72420_1\n",
                encoding="utf-8",
            )
            script = (
                "set -euo pipefail\n"
                "log_phase() { :; }\n"
                'publish-runtime-artifact-manifest() { touch "$LOG_DIR/RAY_RUNTIME_ARTIFACTS.json"; }\n'
                f"{exit_handler}\n"
                "LOG_DIR=$1\n"
                "SLURM_JOB_ID=72420\n"
                "SLURM_JOB_NUM_NODES=2\n"
                "JOB_RESTART_COUNT=0\n"
                "_nrl_finish_exit 17 TEST\n"
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "exit-handler",
                    str(log_dir),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertIn("status=failed exit_code=17", result.stdout)
            self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())


if __name__ == "__main__":
    unittest.main()
