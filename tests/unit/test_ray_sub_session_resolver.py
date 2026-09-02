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

    head_logs = log_dir / "ray" / session / "logs"
    (head_logs / "events").mkdir(parents=True)
    (head_logs / "head.out").write_text("head runtime log\n", encoding="utf-8")
    (head_logs / "events" / "head.json").write_text(
        '{"role":"head"}\n', encoding="utf-8"
    )
    for worker_index in range(expected_workers):
        worker_logs = log_dir / "ray" / f"worker-node-{worker_index}" / session / "logs"
        (worker_logs / "events").mkdir(parents=True)
        (worker_logs / "worker.out").write_text(
            f"worker {worker_index} runtime log\n", encoding="utf-8"
        )
        (worker_logs / "events" / "worker.json").write_text(
            f'{{"worker":{worker_index}}}\n', encoding="utf-8"
        )


def run_runtime_manifest(
    log_dir: Path,
    *,
    job_id: str,
    restart_count: int,
    session: str,
    expected_workers: int,
    source: str | None = None,
) -> subprocess.CompletedProcess[str]:
    manifest = log_dir / "RAY_RUNTIME_ARTIFACTS.json"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            source if source is not None else runtime_manifest_source(),
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
    if not log_dir.exists():
        return
    for root, directories, files in os.walk(log_dir):
        root_path = Path(root)
        root_path.chmod(0o755)
        for directory in directories:
            (root_path / directory).chmod(0o755)
        for filename in files:
            (root_path / filename).chmod(0o644)


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
        self.assertIn(
            'stop-background-processes 10 "head sidecars" TERM KILL 0', stop_function
        )
        source = RAY_SUB.read_text(encoding="utf-8")
        collect_processes = bash_function_source(source, "collect-process-tree-pids")
        stop_processes = bash_function_source(source, "stop-background-processes")

        # Exercise the exact production cleanup function against three real
        # asynchronous children.  The static wait assertion above distinguishes
        # explicit reaping from merely signalling the processes.
        assignments = "\n".join(
            f"busy_sidecar &\n{pid_variable}=$!"
            for pid_variable in pid_by_sidecar.values()
        )
        script = (
            "set -euo pipefail\n"
            "pgrep() { return 1; }\n"
            "busy_sidecar() { while :; do :; done; }\n"
            + collect_processes
            + stop_processes
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
        ordered_finalization_steps = (
            "write-worker-finalize-request-from-head",
            'touch "$LOG_DIR/ENDED"',
            "stop-head-sidecars",
            '"$RAY_FINAL_SYNC_WATCHDOG_SECONDS" ray stop',
            "atomic-final-head-logs",
        )
        offsets = [finalizer.index(step) for step in ordered_finalization_steps]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn('[[ "\\$ray_quiesced" -eq 1 ]]', finalizer)

    def test_head_finalizer_quiesces_ray_before_snapshot_and_preserves_status(
        self,
    ) -> None:
        head = ray_command_source("head")
        finalizer = bash_function_source(head, "finalize-head-log-sync").replace(
            r"\$", "$"
        )

        scenarios = (
            ("successful driver", 0, 0, 0, True, True),
            ("failed driver", 37, 0, 37, False, True),
            ("failed quiescence", 0, 9, 1, True, False),
        )
        for (
            label,
            driver_status,
            ray_stop_status,
            expected_status,
            expect_worker_request,
            expect_snapshot,
        ) in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                log_dir = Path(directory)
                session_name = "session_2026-09-01_00-00-00_72421_1"
                (log_dir / "RAY_SESSION_NAME").write_text(
                    f"{session_name}\n", encoding="utf-8"
                )
                events = log_dir / "events"

                script = (
                    "set -euo pipefail\n"
                    + finalizer
                    + r"""
LOG_DIR="$1"
NRL_TEST_DRIVER_STATUS="$2"
NRL_TEST_RAY_STOP_STATUS="$3"
RAY_FINAL_SYNC_WATCHDOG_SECONDS=7
head_node_name=head-node
events="$LOG_DIR/events"
writer_ready="$LOG_DIR/writer-ready"
writer_stopped="$LOG_DIR/writer-stopped"

stop-head-sidecars() {
  [[ -e "$LOG_DIR/ENDED" ]] || printf 'cleanup-before-marker\n' >> "$events"
  printf 'stop-sidecars\n' >> "$events"
}
provenance-file-matches() { return 0; }
write-worker-finalize-request-from-head() {
  printf 'worker-request:%s\n' "$1" >> "$events"
}
run-final-sync-command() {
  local timeout_seconds="$1"
  shift
  printf 'deadline:%s:%s\n' "$timeout_seconds" "$*" >> "$events"
  "$@"
}
ray() {
  [[ "$#" -eq 1 && "$1" == stop ]]
  printf 'ray-stop\n' >> "$events"
  kill -TERM "$writer_pid"
  wait "$writer_pid" 2>/dev/null || true
  : > "$writer_stopped"
  return "$NRL_TEST_RAY_STOP_STATUS"
}
atomic-final-head-logs() {
  if kill -0 "$writer_pid" 2>/dev/null || [[ ! -e "$writer_stopped" ]]; then
    printf 'copy-before-quiescence\n' >> "$events"
    return 81
  fi
  printf 'snapshot\n' >> "$events"
}
write-head-completion-receipt() {
  printf 'receipt:%s:%s\n' "$1" "$2" >> "$events"
}
writer() {
  trap 'exit 0' TERM
  : > "$writer_ready"
  while :; do sleep 0.01; done
}
writer &
writer_pid=$!
while [[ ! -e "$writer_ready" ]]; do :; done
finalize-head-log-sync "$NRL_TEST_DRIVER_STATUS"
"""
                )
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        script,
                        "head-finalizer",
                        str(log_dir),
                        str(driver_status),
                        str(ray_stop_status),
                    ],
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                self.assertEqual(result.returncode, expected_status, result.stderr)
                recorded_events = events.read_text(encoding="utf-8").splitlines()
                self.assertNotIn("cleanup-before-marker", recorded_events)
                self.assertEqual(
                    f"worker-request:{session_name}" in recorded_events,
                    expect_worker_request,
                )
                if expect_worker_request:
                    self.assertLess(
                        recorded_events.index(f"worker-request:{session_name}"),
                        recorded_events.index("stop-sidecars"),
                    )
                self.assertIn("deadline:7:ray stop", recorded_events)
                self.assertIn("ray-stop", recorded_events)
                self.assertNotIn("copy-before-quiescence", recorded_events)
                self.assertEqual("snapshot" in recorded_events, expect_snapshot)
                self.assertEqual(
                    any(event.startswith("receipt:") for event in recorded_events),
                    expect_snapshot,
                )
                if expect_snapshot:
                    self.assertLess(
                        recorded_events.index("ray-stop"),
                        recorded_events.index("snapshot"),
                    )
                    self.assertIn(
                        f"receipt:{driver_status}:{session_name}", recorded_events
                    )
                self.assertTrue((log_dir / "ENDED").is_file())

    def test_submit_accepts_only_authenticated_head_finalization_handoff(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        functions = "".join(
            bash_function_source(source, function)
            for function in (
                "provenance-file-matches",
                "validate-worker-finalize-request",
                "controlled-finalization-request-valid",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            session_name = "session_2026-08-31_00-00-00_72421_1"
            session_file = log_dir / "RAY_SESSION_NAME"
            request_file = log_dir / "WORKER_FINALIZE_REQUEST"
            session_file.write_text(f"{session_name}\n", encoding="utf-8")
            request_file.write_text(
                "schema=ray-session-completion-v1\n"
                "job_id=72421\n"
                "restart_count=0\n"
                f"session={session_name}\n"
                "action=finalize-workers\n",
                encoding="utf-8",
            )
            script = (
                "set -euo pipefail\n"
                f"{functions}\n"
                'LOG_DIR="$1"\n'
                'WORKER_FINALIZE_REQUEST="$LOG_DIR/WORKER_FINALIZE_REQUEST"\n'
                "SLURM_JOB_ID=72421\n"
                "JOB_RESTART_COUNT=0\n"
                "controlled-finalization-request-valid\n"
            )

            valid = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "validate-finalization-handoff",
                    str(log_dir),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)

            request_file.write_text("action=finalize-workers\n", encoding="utf-8")
            malformed = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "validate-finalization-handoff",
                    str(log_dir),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(malformed.returncode, 0)

            session_file.unlink()
            session_file.symlink_to(log_dir / "attacker-controlled-session")
            symlinked = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "validate-finalization-handoff",
                    str(log_dir),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(symlinked.returncode, 0)

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
        self.assertIn(
            'stop-background-processes 10 "worker sidecars" TERM KILL 0',
            stop_function,
        )
        source = RAY_SUB.read_text(encoding="utf-8")
        collect_processes = bash_function_source(source, "collect-process-tree-pids")
        stop_processes = bash_function_source(source, "stop-background-processes")

        assignments = "\n".join(
            f"busy_sidecar &\n{pid_variable}=$!"
            for pid_variable in pid_by_sidecar.values()
        )
        script = (
            "set -euo pipefail\n"
            "pgrep() { return 1; }\n"
            "busy_sidecar() { while :; do :; done; }\n"
            + collect_processes
            + stop_processes
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
        self.assertLess(
            finalizer.index('"$RAY_FINAL_SYNC_WATCHDOG_SECONDS" ray stop'),
            finalizer.index("atomic-final-worker-logs"),
        )

    def test_worker_finalizer_quiesces_when_main_exits_before_ended(self) -> None:
        worker = ray_command_source("worker")
        stop_sidecars = bash_function_source(worker, "stop-worker-sidecars").replace(
            r"\$", "$"
        )
        source = RAY_SUB.read_text(encoding="utf-8")
        collect_processes = bash_function_source(source, "collect-process-tree-pids")
        stop_processes = bash_function_source(source, "stop-background-processes")
        finalizer = (
            bash_function_source(worker, "finalize-worker-log-sync")
            .replace(r"\$", "$")
            .replace("/tmp/ray", "$NRL_TEST_RAY_ROOT")
        )

        scenarios = (
            ("successful worker", 0, 0, 0, True),
            ("failed worker", 37, 0, 37, True),
            ("failed quiescence", 0, 9, 1, False),
        )
        for (
            label,
            worker_status,
            ray_stop_status,
            expected_status,
            expect_snapshot,
        ) in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log_dir = root / "logs"
                log_dir.mkdir()
                ray_root = root / "ray-root"
                session_name = "session_2026-09-01_00-00-00_72421_3"
                (ray_root / session_name / "logs").mkdir(parents=True)
                (log_dir / "RAY_SESSION_NAME").write_text(
                    f"{session_name}\n", encoding="utf-8"
                )
                probe_ready = root / "worker-session-probe-ready"
                probe_ready.touch()
                events = root / "events"

                script = (
                    "set -euo pipefail\n"
                    "pgrep() { return 1; }\n"
                    + collect_processes
                    + stop_processes
                    + stop_sidecars
                    + finalizer
                    + r"""
NRL_TEST_ROOT="$1"
NRL_TEST_WORKER_STATUS="$2"
NRL_TEST_RAY_STOP_STATUS="$3"
NRL_TEST_RAY_ROOT="$NRL_TEST_ROOT/ray-root"
LOG_DIR="$NRL_TEST_ROOT/logs"
WORKER_PROCID=3
WORKER_NODE=worker-node-3
WORKER_SESSION_PROBE_READY="$NRL_TEST_ROOT/worker-session-probe-ready"
RAY_FINAL_SYNC_WATCHDOG_SECONDS=7
events="$NRL_TEST_ROOT/events"
monitor_ready="$NRL_TEST_ROOT/monitor-ready"
writer_ready="$NRL_TEST_ROOT/writer-ready"
writer_stopped="$NRL_TEST_ROOT/writer-stopped"
worker_log_sync_sidecar_pid=""

monitor_waiter() {
  trap 'printf "monitor-reaped\n" >> "$events"; exit 0' TERM
  : > "$monitor_ready"
  while [[ ! -e "$LOG_DIR/ENDED" ]]; do sleep 0.01; done
  printf 'monitor-saw-ended\n' >> "$events"
}
monitor_waiter &
worker_monitor_sidecar_pid=$!

ray_writer() {
  trap 'exit 0' TERM
  : > "$writer_ready"
  while :; do sleep 0.01; done
}
ray_writer &
ray_writer_pid=$!
while [[ ! -e "$monitor_ready" || ! -e "$writer_ready" ]]; do :; done

resolve_ray_session_latest() {
  printf '%s\n' "$NRL_TEST_RAY_ROOT/$(<"$LOG_DIR/RAY_SESSION_NAME")"
}
provenance-file-matches() { return 0; }
run-final-sync-command() {
  local timeout_seconds="$1"
  shift
  printf 'deadline:%s:%s\n' "$timeout_seconds" "$*" >> "$events"
  "$@"
}
ray() {
  [[ "$#" -eq 1 && "$1" == stop ]]
  printf 'ray-stop\n' >> "$events"
  kill -TERM "$ray_writer_pid"
  wait "$ray_writer_pid" 2>/dev/null || true
  : > "$writer_stopped"
  return "$NRL_TEST_RAY_STOP_STATUS"
}
atomic-final-worker-logs() {
  if kill -0 "$ray_writer_pid" 2>/dev/null || [[ ! -e "$writer_stopped" ]]; then
    printf 'snapshot-before-quiescence\n' >> "$events"
    return 81
  fi
  printf 'snapshot\n' >> "$events"
}
write-worker-completion-receipt() {
  printf 'receipt:%s:%s\n' "$1" "$2" >> "$events"
}
finalize-worker-log-sync "$NRL_TEST_WORKER_STATUS"
"""
                )
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        script,
                        "worker-finalizer",
                        str(root),
                        str(worker_status),
                        str(ray_stop_status),
                    ],
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                self.assertEqual(result.returncode, expected_status, result.stderr)
                recorded_events = events.read_text(encoding="utf-8").splitlines()
                self.assertIn("monitor-reaped", recorded_events)
                self.assertNotIn("monitor-saw-ended", recorded_events)
                self.assertIn("deadline:7:ray stop", recorded_events)
                self.assertIn("ray-stop", recorded_events)
                self.assertNotIn("snapshot-before-quiescence", recorded_events)
                self.assertEqual("snapshot" in recorded_events, expect_snapshot)
                self.assertEqual(
                    any(event.startswith("receipt:") for event in recorded_events),
                    expect_snapshot,
                )
                if expect_snapshot:
                    self.assertLess(
                        recorded_events.index("monitor-reaped"),
                        recorded_events.index("ray-stop"),
                    )
                    self.assertLess(
                        recorded_events.index("ray-stop"),
                        recorded_events.index("snapshot"),
                    )
                    self.assertIn(
                        f"receipt:{worker_status}:{session_name}", recorded_events
                    )
                self.assertFalse((log_dir / "ENDED").exists())
                self.assertFalse(probe_ready.exists())

    def test_declare_f_injection_survives_expandable_heredoc(self) -> None:
        render_script = (
            "set -euo pipefail\n"
            + resolver_source()
            + r"""
RAY_WORKER_SESSION_LATEST_RESOLVER="$(declare -f resolve_ray_session_latest)"
readlink() { printf 'RENDER_EXECUTED\n' >&2; return 99; }
cat <<EOF
$RAY_WORKER_SESSION_LATEST_RESOLVER
EOF
"""
        )
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
    def test_manifest_publication_runs_python_under_external_watchdog(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        run_bounded = bash_function_source(source, "run-final-sync-command")
        publish = bash_function_source(source, "publish-runtime-artifact-manifest")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable_dir = root / "bin"
            executable_dir.mkdir()
            tracker = root / "tracker"
            tracker.mkdir()

            timeout_executable = executable_dir / "timeout"
            timeout_executable.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                '[[ "$1" == "--kill-after=10s" ]]\n'
                '[[ "$2" == "7s" ]]\n'
                "shift 2\n"
                'printf "%s\\n" "$@" > "$NRL_TEST_TRACKER/timeout-argv"\n'
                'exec "$@"\n',
                encoding="utf-8",
            )
            timeout_executable.chmod(0o700)
            python_executable = executable_dir / "python3"
            python_executable.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                'printf "%s\\n" "$@" > "$NRL_TEST_TRACKER/python-argv"\n',
                encoding="utf-8",
            )
            python_executable.chmod(0o700)

            script = (
                "set -euo pipefail\n"
                + run_bounded
                + publish
                + r"""
RAY_FINAL_SYNC_WATCHDOG_SECONDS=7
RUNTIME_ARTIFACT_MANIFEST_HELPER="$1/helper.py"
LOG_DIR="$1/logs"
RAY_RUNTIME_ARTIFACT_MANIFEST="$LOG_DIR/RAY_RUNTIME_ARTIFACTS.json"
SLURM_JOB_ID=72425
JOB_RESTART_COUNT=3
mkdir -p "$LOG_DIR"
publish-runtime-artifact-manifest session_test 2
"""
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "manifest-watchdog",
                    str(root),
                ],
                env={
                    "PATH": f"{executable_dir}:/usr/bin:/bin",
                    "NRL_TEST_TRACKER": str(tracker),
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            timeout_argv = (tracker / "timeout-argv").read_text(encoding="utf-8")
            self.assertTrue(timeout_argv.startswith("python3\n"), timeout_argv)
            python_argv = (tracker / "python-argv").read_text(encoding="utf-8")
            self.assertEqual(
                python_argv.splitlines(),
                [
                    str(root / "helper.py"),
                    str(root / "logs"),
                    str(root / "logs" / "RAY_RUNTIME_ARTIFACTS.json"),
                    "72425",
                    "3",
                    "session_test",
                    "2",
                ],
            )

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
                self.assertEqual(manifest["artifact_count"], 18)
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
                        "final_log/head/events/head.json",
                        "final_log/head/head.out",
                        "final_log/worker-0/events/worker.json",
                        "final_log/worker-0/worker.out",
                        "final_log/worker-1/events/worker.json",
                        "final_log/worker-1/worker.out",
                    ],
                )
                self.assertEqual(manifest["final_log_directory_count"], 12)
                self.assertEqual(
                    manifest["integrity"],
                    {
                        "artifact_mode": "0444",
                        "artifacts_sealed": True,
                        "artifacts_fsynced": True,
                        "directory_mode": "0555",
                        "directories_sealed": True,
                        "directories_fsynced": True,
                        "final_logs_inventoried": True,
                        "final_logs_sealed": True,
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

                for identity in manifest["final_log_directories"]:
                    directory_path = Path(identity["path"])
                    directory_stat = directory_path.stat(follow_symlinks=False)
                    self.assertTrue(stat.S_ISDIR(directory_stat.st_mode))
                    self.assertEqual(stat.S_IMODE(directory_stat.st_mode), 0o555)
                    self.assertEqual(identity["mode"], "0555")
                    self.assertEqual(identity["device"], directory_stat.st_dev)
                    self.assertEqual(identity["inode"], directory_stat.st_ino)
                    self.assertTrue(identity["name"].startswith("final_log"))

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

    def test_unexpected_or_symlinked_final_log_tree_fails_closed(self) -> None:
        job_id = "72422"
        session = "session_2026-09-01_00-00-00_72422_1"
        for corruption in ("unexpected-root", "symlinked-entry"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                log_dir = Path(directory) / "72422-logs"
                write_runtime_evidence(
                    log_dir,
                    job_id=job_id,
                    restart_count=0,
                    session=session,
                    expected_workers=1,
                )
                if corruption == "unexpected-root":
                    (log_dir / "ray" / "stale-node").mkdir()
                else:
                    head_logs = log_dir / "ray" / session / "logs"
                    (head_logs / "linked.log").symlink_to(head_logs / "head.out")
                try:
                    result = run_runtime_manifest(
                        log_dir,
                        job_id=job_id,
                        restart_count=0,
                        session=session,
                        expected_workers=1,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())
                    self.assertFalse((log_dir / "RAY_JOB_COMPLETION").exists())
                finally:
                    make_runtime_evidence_writable(log_dir)

    def test_late_manifest_failure_removes_transaction_success_receipt(self) -> None:
        job_id = "72423"
        session = "session_2026-09-01_00-00-00_72423_1"
        source = runtime_manifest_source()
        publication_boundary = "        manifest_published = True\n\n        os.fchmod(log_fd, DIRECTORY_MODE)"
        self.assertEqual(source.count(publication_boundary), 1)
        source = source.replace(
            publication_boundary,
            "        manifest_published = True\n"
            '        raise RuntimeError("injected post-rename failure")\n\n'
            "        os.fchmod(log_fd, DIRECTORY_MODE)",
        )
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72423-logs"
            write_runtime_evidence(
                log_dir,
                job_id=job_id,
                restart_count=0,
                session=session,
                expected_workers=1,
            )
            try:
                result = run_runtime_manifest(
                    log_dir,
                    job_id=job_id,
                    restart_count=0,
                    session=session,
                    expected_workers=1,
                    source=source,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("injected post-rename failure", result.stderr)
                self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())
                self.assertFalse((log_dir / "RAY_JOB_COMPLETION").exists())
                self.assertFalse(
                    list(log_dir.glob(".RAY_RUNTIME_ARTIFACTS.json.*.tmp"))
                )
            finally:
                make_runtime_evidence_writable(log_dir)

    def test_post_inventory_final_log_mutation_invalidates_commit(self) -> None:
        job_id = "72424"
        session = "session_2026-09-01_00-00-00_72424_1"
        source = runtime_manifest_source()
        publication_boundary = "        manifest_published = True\n\n        os.fchmod(log_fd, DIRECTORY_MODE)"
        self.assertEqual(source.count(publication_boundary), 1)
        mutation = (
            "        manifest_published = True\n"
            '        mutation_path = os.path.join(log_dir, "ray", session, "logs", "head.out")\n'
            "        os.chmod(mutation_path, 0o644)\n"
            '        with open(mutation_path, "ab") as mutation_file:\n'
            '            mutation_file.write(b"late mutation\\n")\n\n'
            "        os.fchmod(log_fd, DIRECTORY_MODE)"
        )
        source = source.replace(publication_boundary, mutation)
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "72424-logs"
            write_runtime_evidence(
                log_dir,
                job_id=job_id,
                restart_count=0,
                session=session,
                expected_workers=0,
            )
            try:
                result = run_runtime_manifest(
                    log_dir,
                    job_id=job_id,
                    restart_count=0,
                    session=session,
                    expected_workers=0,
                    source=source,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("identity changed", result.stderr)
                self.assertFalse((log_dir / "RAY_RUNTIME_ARTIFACTS.json").exists())
                self.assertFalse((log_dir / "RAY_JOB_COMPLETION").exists())
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
                self.assertEqual(manifest["artifact_count"], 68)
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
                    len([name for name in artifacts if name.startswith("final_log/")]),
                    2 * (expected_workers + 1),
                )
                self.assertIn("final_log/worker-11/events/worker.json", artifacts)
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
        receipt_call = (
            'write-job-completion-receipt "$session_name" "$expected_workers"'
        )
        self.assertEqual(exit_handler.count(success_guard), 2)
        self.assertEqual(exit_handler.count(receipt_call), 1)
        self.assertEqual(exit_handler.count(publish_call), 1)
        self.assertLess(
            exit_handler.index(receipt_call), exit_handler.index(publish_call)
        )
        self.assertLess(
            exit_handler.index(publish_call),
            exit_handler.index('completion_status="complete"'),
        )
        self.assertEqual(source.count(publish_call), 1)
        self.assertEqual(source.count(receipt_call), 1)

        noninteractive = source.rsplit('if [[ -n "$COMMAND" ]]; then', 1)[1].split(
            "else\n  # Interactive:", 1
        )[0]
        self.assertNotIn("write-job-completion-receipt", noninteractive)
        self.assertNotIn("publish-runtime-artifact-manifest", noninteractive)

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
                'write-job-completion-receipt() { : > "$RAY_JOB_COMPLETION_RECEIPT"; }\n'
                "publish-runtime-artifact-manifest() { return 1; }\n"
                'cleanup-incomplete-runtime-publication() { rm -f "$RAY_JOB_COMPLETION_RECEIPT" "$RAY_RUNTIME_ARTIFACT_MANIFEST"; }\n'
                f"{exit_handler}\n"
                "LOG_DIR=$1\n"
                'RAY_JOB_COMPLETION_RECEIPT="$LOG_DIR/RAY_JOB_COMPLETION"\n'
                'RAY_RUNTIME_ARTIFACT_MANIFEST="$LOG_DIR/RAY_RUNTIME_ARTIFACTS.json"\n'
                "RAY_FINAL_SYNC_WATCHDOG_SECONDS=1\n"
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
            self.assertFalse((log_dir / "RAY_JOB_COMPLETION").exists())

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
                'write-job-completion-receipt() { touch "$LOG_DIR/writer-called"; }\n'
                'publish-runtime-artifact-manifest() { touch "$LOG_DIR/publisher-called"; }\n'
                "cleanup-incomplete-runtime-publication() { return 88; }\n"
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
            self.assertFalse((log_dir / "writer-called").exists())
            self.assertFalse((log_dir / "publisher-called").exists())


if __name__ == "__main__":
    unittest.main()
