from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"
POLICY_BEGIN = "# BEGIN RAY_FINAL_SYNC_WATCHDOG_POLICY\n"
POLICY_END = "# END RAY_FINAL_SYNC_WATCHDOG_POLICY\n"
HELPER_BEGIN = "cat > \"$FINAL_SYNC_HELPER_SCRIPT\" <<'FINAL_SYNC_HELPER_EOF'\n"
HELPER_END = "\nFINAL_SYNC_HELPER_EOF\n"


def watchdog_policy_source() -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    if source.count(POLICY_BEGIN) != 1 or source.count(POLICY_END) != 1:
        raise AssertionError("Ray final-sync watchdog policy sentinels must be unique")
    return source.split(POLICY_BEGIN, 1)[1].split(POLICY_END, 1)[0]


def final_sync_helper_source() -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    if source.count(HELPER_BEGIN) != 1:
        raise AssertionError("Ray final-sync helper heredoc must be unique")
    body = source.split(HELPER_BEGIN, 1)[1]
    if HELPER_END not in body:
        raise AssertionError("Ray final-sync helper heredoc must be terminated")
    return body.split(HELPER_END, 1)[0]


def bash_function_source(source: str, function: str) -> str:
    start = f"{function}() {{\n"
    if source.count(start) != 1:
        raise AssertionError(f"{function} definition must be unique")
    body = source.split(start, 1)[1]
    end = "\n}\n"
    if end not in body:
        raise AssertionError(f"{function} definition must be terminated")
    return start + body.split(end, 1)[0] + end


def indented_bash_function_source(source: str, function: str, indent: str) -> str:
    start = f"{indent}{function}() {{\n"
    if source.count(start) != 1:
        raise AssertionError(f"{function} definition must be unique")
    body = source.split(start, 1)[1]
    end = f"\n{indent}}}\n"
    if end not in body:
        raise AssertionError(f"{function} definition must be terminated")
    return textwrap.dedent(start + body.split(end, 1)[0] + end)


def run_watchdog_policy(
    node_count: str,
    base_seconds: str = "300",
    per_node_seconds: str = "15",
    maximum_seconds: str = "3600",
) -> subprocess.CompletedProcess[str]:
    script = (
        "set -euo pipefail\n"
        + watchdog_policy_source()
        + '\nray-final-sync-watchdog-seconds "$1" "$2" "$3" "$4"\n'
    )
    return subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
            "watchdog-policy",
            node_count,
            base_seconds,
            per_node_seconds,
            maximum_seconds,
        ],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        check=False,
        capture_output=True,
        text=True,
    )


class RayFinalSyncScalingTest(unittest.TestCase):
    def test_watchdog_keeps_prior_floor_and_scales_for_large_allocations(self) -> None:
        expected_seconds = {
            "1": "300",
            "10": "435",
            "16": "525",
            "22": "615",
            "86": "1575",
            "1000": "3600",
        }
        for node_count, expected in expected_seconds.items():
            with self.subTest(node_count=node_count):
                result = run_watchdog_policy(node_count)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"{expected}\n")

    def test_watchdog_policy_rejects_unbounded_or_malformed_values(self) -> None:
        invalid_arguments = (
            ("0", "300", "15", "3600"),
            ("16", "0", "15", "3600"),
            ("16", "300", "-1", "3600"),
            ("16", "300", "08", "3600"),
            ("16", "300", "15", "299"),
            ("16", "300", "15", "86401"),
            ("999999999999999999999", "300", "15", "3600"),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                result = run_watchdog_policy(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_worker_watchdog_consumes_computed_policy_not_fixed_literal(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        self.assertNotIn('sleep "$RAY_FINAL_SYNC_WATCHDOG_SECONDS"', source)
        self.assertNotIn('sleep 300\n      if kill -0 "$worker_pid"', source)
        self.assertNotIn("worker_watchdog_pid", source)
        self.assertEqual(source.count(".RAY_FINAL_SYNC_LOCK"), 0)
        self.assertIn(
            '"\\$source_logs" "\\$final_logs" head "$RAY_FINAL_SYNC_WATCHDOG_SECONDS"',
            source,
        )
        self.assertIn(
            '"worker-\\$WORKER_PROCID" \\\n    "$RAY_FINAL_SYNC_WATCHDOG_SECONDS"',
            source,
        )
        self.assertIn(
            '"$RAY_FINAL_SYNC_WATCHDOG_SECONDS" ray stop',
            source,
        )
        self.assertIn(
            'timeout --kill-after=10s "${timeout_seconds}s" "$@"',
            source,
        )
        self.assertNotIn("timeout --foreground", source)
        self.assertIn(
            '"$head_pid" "$head_finalization_observed" \\\n'
            '    "$RAY_FINAL_SYNC_WATCHDOG_SECONDS" "head finalization" \\\n'
            "    10 USR1 TERM",
            source,
        )
        self.assertIn(
            '"$worker_pid" 1 "$RAY_FINAL_SYNC_WATCHDOG_SECONDS" \\\n'
            '      "worker final-sync receipts" 10 USR1 TERM',
            source,
        )
        self.assertIn("trap 'finalize-head-log-sync 143' USR1", source)
        self.assertIn("trap 'finalize-worker-log-sync 143' USR1", source)
        self.assertIn("trap stop-sandbox-tree USR1", source)
        self.assertIn(
            '10 "sandbox teardown" USR1 TERM 1 "${SRUN_PIDS["sandbox"]}"',
            source,
        )
        self.assertIn(
            'if [[ -f "$LOG_DIR/ENDED" ]]; then\n'
            "      head_finalization_observed=1\n"
            "      break",
            source,
        )
        self.assertIn(
            'waiting for ${description}" >&2',
            source,
        )
        self.assertIn(
            'kill "-$force_signal" "${target_tree_pids[@]}"',
            source,
        )
        self.assertIn(
            'run-final-sync-command "$RAY_FINAL_SYNC_WATCHDOG_SECONDS" \\\n'
            '    python3 "$RUNTIME_ARTIFACT_MANIFEST_HELPER"',
            source,
        )
        self.assertIn(
            "RAY_FINAL_SYNC_BASE_TIMEOUT_SECONDS=${RAY_FINAL_SYNC_BASE_TIMEOUT_SECONDS:-300}",
            source,
        )
        self.assertIn(
            "RAY_FINAL_SYNC_PER_NODE_TIMEOUT_SECONDS=${RAY_FINAL_SYNC_PER_NODE_TIMEOUT_SECONDS:-15}",
            source,
        )
        self.assertIn(
            "RAY_FINAL_SYNC_MAX_TIMEOUT_SECONDS=${RAY_FINAL_SYNC_MAX_TIMEOUT_SECONDS:-3600}",
            source,
        )

    def test_sandbox_srun_must_exit_zero_instead_of_being_cancelled(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        teardown_start = source.index(
            "  # Sandbox sidecars are not provenance-bearing."
        )
        teardown_end = source.index("\n\n  set -e", teardown_start)
        teardown = source[teardown_start:teardown_end]

        self.assertNotIn('kill "${SRUN_PIDS["sandbox"]}"', teardown)
        self.assertNotIn(
            'wait "${SRUN_PIDS["sandbox"]}" 2>/dev/null || true', teardown
        )
        self.assertIn(
            '10 "sandbox teardown" USR1 TERM 1 "${SRUN_PIDS["sandbox"]}"',
            teardown,
        )
        self.assertIn("DerivedExitCode", teardown)

        # The forwarded graceful signal is successful only after node-local
        # sandbox descendants have stopped without a forced cleanup.
        sandbox_stop = indented_bash_function_source(
            source, "stop-sandbox-tree", "      "
        )
        self.assertIn("trap - USR1", sandbox_stop)
        self.assertIn("exit 0", sandbox_stop)
        self.assertIn(
            "collection_failed == 1 || forced == 1 || reap_failed == 1",
            sandbox_stop,
        )
        self.assertIn("while :; do sleep 1; done", sandbox_stop)

    def test_head_watchdog_enforces_deadline_and_preserves_head_status(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        process_tree = bash_function_source(source, "collect-process-tree-pids")
        watchdog = bash_function_source(source, "wait-for-finalization")
        script = (
            "set -euo pipefail\n"
            + process_tree
            + watchdog
            + r"""
mode="$1"
tracker="$2"
if [[ "$mode" == blocked ]]; then
  blocked_head() {
    trap ': > "$tracker/terminated"; exit 0' TERM
    : > "$tracker/ready"
    while :; do sleep 0.05; done
  }
  blocked_head &
  head_pid=$!
  while [[ ! -e "$tracker/ready" ]]; do :; done
  timeout_seconds=1
elif [[ "$mode" == term-resistant ]]; then
  resistant_head() {
    trap ': > "$tracker/terminated"' TERM
    : > "$tracker/ready"
    while :; do sleep 0.05; done
  }
  resistant_head &
  head_pid=$!
  while [[ ! -e "$tracker/ready" ]]; do :; done
  timeout_seconds=1
else
  (exit 37) &
  head_pid=$!
  timeout_seconds=5
fi
set +e
wait-for-finalization "$head_pid" 1 "$timeout_seconds" "test finalization" 1
head_status=$?
set -e
printf '%s\n' "$head_status"
"""
        )

        for mode, expected_status, expect_termination in (
            ("blocked", "1\n", True),
            ("term-resistant", "137\n", True),
            ("failed", "37\n", False),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                tracker = Path(directory)
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        script,
                        "head-watchdog",
                        mode,
                        str(tracker),
                    ],
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=8,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected_status)
                self.assertEqual((tracker / "terminated").is_file(), expect_termination)

    def test_secondary_worker_failure_preserves_exact_driver_status(self) -> None:
        merge = bash_function_source(
            RAY_SUB.read_text(encoding="utf-8"), "merge-finalization-exit-code"
        )
        script = (
            "set -euo pipefail\n"
            + merge
            + r"""
set +e
merge-finalization-exit-code "$1" "$2"
merged_status=$?
set -e
printf '%s\n' "$merged_status"
"""
        )
        for primary, secondary, expected in (
            ("37", "1", "37\n"),
            ("0", "1", "1\n"),
            ("0", "0", "0\n"),
        ):
            with self.subTest(primary=primary, secondary=secondary):
                result = subprocess.run(
                    [
                        "/bin/bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        script,
                        "merge-status",
                        primary,
                        secondary,
                    ],
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_term_resistant_sidecar_tree_is_force_killed_within_bound(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        collect = bash_function_source(source, "collect-process-tree-pids")
        stop = bash_function_source(source, "stop-background-processes")
        script = (
            "set -euo pipefail\n"
            + collect
            + stop
            + r"""
tracker="$1"
resistant_child() {
  trap ': > "$tracker/child-term"' TERM
  : > "$tracker/child-ready"
  while :; do sleep 0.05; done
}
sidecar() {
  trap ': > "$tracker/root-term"; exit 0' TERM
  resistant_child &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$tracker/child-pid"
  wait "$child_pid"
}
sidecar &
NRL_TEST_ROOT_PID=$!
while [[ ! -e "$tracker/child-ready" ]]; do :; done
NRL_TEST_CHILD_PID=$(<"$tracker/child-pid")
pgrep() {
  [[ "$1" == -P ]]
  if [[ "$2" == "$NRL_TEST_ROOT_PID" ]]; then
    printf '%s\n' "$NRL_TEST_CHILD_PID"
    return 0
  fi
  return 1
}
set +e
stop-background-processes 1 "test sidecar" TERM KILL 0 "$NRL_TEST_ROOT_PID"
stop_status=$?
set -e
printf '%s\n' "$stop_status"
if kill -0 "$NRL_TEST_CHILD_PID" 2>/dev/null; then exit 90; fi
"""
        )
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory)
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "sidecar-watchdog",
                    str(tracker),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "1\n")
            self.assertTrue((tracker / "root-term").is_file())
            self.assertTrue((tracker / "child-term").is_file())

    def test_process_enumeration_failure_cannot_report_cleanup_success(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        collect = bash_function_source(source, "collect-process-tree-pids")
        stop = bash_function_source(source, "stop-background-processes")
        script = (
            "set -euo pipefail\n"
            + collect
            + stop
            + r"""
pgrep() { return 2; }
trap_exit() { trap 'exit 0' TERM; while :; do sleep 0.05; done; }
trap_exit &
target_pid=$!
set +e
stop-background-processes 1 "enumeration failure" TERM KILL 0 "$target_pid"
stop_status=$?
set -e
printf '%s\n' "$stop_status"
"""
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
        self.assertEqual(result.stdout, "1\n")
        self.assertIn("Could not enumerate", result.stderr)

    def test_strict_cleanup_rejects_an_already_failed_sandbox_process(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        collect = bash_function_source(source, "collect-process-tree-pids")
        stop = bash_function_source(source, "stop-background-processes")
        script = (
            "set -euo pipefail\n"
            + collect
            + stop
            + r"""
pgrep() { return 1; }
(exit 42) &
target_pid=$!
# Let the asynchronous process finish while retaining its status for wait.
for _ in {1..100}; do
  kill -0 "$target_pid" 2>/dev/null || break
  sleep 0.01
done
set +e
stop-background-processes 1 "sandbox teardown" USR1 TERM 1 "$target_pid"
stop_status=$?
set -e
printf '%s\n' "$stop_status"
"""
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
        self.assertEqual(result.stdout, "1\n")
        self.assertIn("sandbox teardown process", result.stderr)
        self.assertIn("exited nonzero: 42", result.stderr)

    def test_sandbox_task_rejects_forced_or_incomplete_tree_cleanup(self) -> None:
        source = RAY_SUB.read_text(encoding="utf-8")
        collect = indented_bash_function_source(source, "sandbox-tree-pids", "      ")
        stop = indented_bash_function_source(source, "stop-sandbox-tree", "      ")
        self.assertEqual(stop.count("while :; do sleep 1; done"), 1)
        stop = stop.replace("while :; do sleep 1; done", "return 91")
        script = (
            "set -euo pipefail\n"
            + collect
            + stop
            + r"""
tracker="$1"
resistant_child() {
  trap ': > "$tracker/child-term"' TERM
  : > "$tracker/child-ready"
  while :; do /bin/sleep 0.05; done
}
sandbox_root() {
  trap ': > "$tracker/root-term"; exit 0' TERM
  resistant_child &
  child_pid=$!
  printf '%s\n' "$child_pid" > "$tracker/child-pid"
  wait "$child_pid"
}
sandbox_root &
SANDBOX_PID=$!
while [[ ! -e "$tracker/child-ready" ]]; do :; done
NRL_TEST_CHILD_PID=$(<"$tracker/child-pid")
pgrep() {
  [[ "$1" == -P ]]
  if [[ "$2" == "$SANDBOX_PID" ]]; then
    printf '%s\n' "$NRL_TEST_CHILD_PID"
    return 0
  fi
  return 1
}
sleep() { SECONDS=$((SECONDS + 10)); /bin/sleep 0.2; }
stop-sandbox-tree
"""
        )
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory)
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "sandbox-tree",
                    str(tracker),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 91, result.stderr)
            self.assertTrue((tracker / "root-term").is_file())
            self.assertTrue((tracker / "child-term").is_file())

    def test_twenty_two_slow_targets_copy_concurrently_and_publish_atomically(
        self,
    ) -> None:
        worker_count = 22
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = root / "tracker"
            tracker.mkdir()
            targets: list[Path] = []
            for worker_index in range(worker_count):
                source = root / "sources" / str(worker_index) / "logs"
                source.mkdir(parents=True)
                (source / "worker.log").write_text(
                    f"worker-{worker_index}\n", encoding="utf-8"
                )
                targets.append(
                    root
                    / "ray"
                    / f"worker-node-{worker_index}"
                    / "session_scale_test"
                    / "logs"
                )

            script = (
                "set -euo pipefail\n"
                + final_sync_helper_source()
                + r"""
NRL_PROVENANCE_LOG_DIR="$1"
NRL_TEST_TRACKER="$2"
worker_count="$3"
export NRL_PROVENANCE_LOG_DIR NRL_TEST_TRACKER

# Hold every copy at the same barrier. A cluster-global publication lock would
# allow only one marker to appear and make this contract time out.
rsync() {
  local source destination source_parent worker_index
  while (( "$#" > 2 )); do shift; done
  source="$1"
  destination="$2"
  source_parent="${source%/}"
  source_parent="${source_parent%/logs}"
  worker_index="${source_parent##*/}"
  : > "$NRL_TEST_TRACKER/started.$worker_index"
  while [[ ! -e "$NRL_TEST_TRACKER/release" ]]; do sleep 0.01; done
  cp -a -- "${source%/}/." "${destination%/}/"
}

pids=()
for (( worker_index=0; worker_index<worker_count; worker_index++ )); do
  atomic-final-ray-logs \
    "$NRL_PROVENANCE_LOG_DIR/sources/$worker_index/logs" \
    "$NRL_PROVENANCE_LOG_DIR/ray/worker-node-$worker_index/session_scale_test/logs" \
    "worker-$worker_index" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
"""
            )
            process = subprocess.Popen(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "final-sync",
                    str(root),
                    str(tracker),
                    str(worker_count),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            while len(list(tracker.glob("started.*"))) < worker_count:
                if process.poll() is not None or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)

            started_count = len(list(tracker.glob("started.*")))
            for target in targets:
                self.assertFalse(target.exists(), f"published before copy: {target}")

            (tracker / "release").touch()
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(started_count, worker_count, stderr)
            self.assertEqual(process.returncode, 0, f"{stdout}\n{stderr}")

            for worker_index, target in enumerate(targets):
                self.assertEqual(
                    (target / "worker.log").read_text(encoding="utf-8"),
                    f"worker-{worker_index}\n",
                )
                parent_entries = {path.name for path in target.parent.iterdir()}
                self.assertEqual(parent_entries, {"logs"})

    def test_duplicate_target_and_copy_failure_preserve_fail_closed_behavior(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "new.log").write_text("new\n", encoding="utf-8")
            target = root / "ray" / "worker-node" / "session_test" / "logs"
            target.mkdir(parents=True)
            (target / "old.log").write_text("old\n", encoding="utf-8")

            duplicate_script = (
                "set -euo pipefail\n"
                + final_sync_helper_source()
                + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
if atomic-final-ray-logs "$2" "$3" worker-0; then
  exit 90
fi
"""
            )
            duplicate = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    duplicate_script,
                    "duplicate-target",
                    str(root),
                    str(source),
                    str(target),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
            self.assertEqual({path.name for path in target.iterdir()}, {"old.log"})
            self.assertEqual((target / "old.log").read_text(encoding="utf-8"), "old\n")

            (target / "old.log").unlink()
            target.rmdir()
            failure_script = (
                "set -euo pipefail\n"
                + final_sync_helper_source()
                + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
rsync() { return 17; }
if atomic-final-ray-logs "$2" "$3" worker-0; then
  exit 91
fi
"""
            )
            failure = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    failure_script,
                    "copy-failure",
                    str(root),
                    str(source),
                    str(target),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(failure.returncode, 0, failure.stderr)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def test_copy_deadline_failure_preserves_fail_closed_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "worker.log").write_text("worker\n", encoding="utf-8")
            target = root / "ray" / "worker-node" / "session_test" / "logs"
            script = (
                "set -euo pipefail\n"
                + final_sync_helper_source()
                + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
timeout() { return 124; }
if atomic-final-ray-logs "$2" "$3" worker-0 1; then
  exit 92
fi
"""
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "copy-deadline",
                    str(root),
                    str(source),
                    str(target),
                ],
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])

    def test_blocking_copy_is_terminated_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "head.log").write_text("head\n", encoding="utf-8")
            target = root / "ray" / "session_test" / "logs"
            tracker = root / "tracker"
            tracker.mkdir()
            executable_dir = root / "bin"
            executable_dir.mkdir()

            # Keep this test portable to macOS while exercising the exact
            # production timeout command line. The fake copy would spin for
            # three seconds without the helper's deadline, and records TERM so
            # the assertion distinguishes a timeout from an ordinary failure.
            timeout_executable = executable_dir / "timeout"
            timeout_executable.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                '[[ "$1" == "--kill-after=10s" ]]\n'
                '[[ "$2" == "1s" ]]\n'
                "shift 2\n"
                '"$@" &\n'
                "command_pid=$!\n"
                "(\n"
                "  for _ in {1..300}; do\n"
                '    [[ -e "$NRL_TEST_TRACKER/copy-started" ]] && break\n'
                "    sleep 0.01\n"
                "  done\n"
                '  [[ -e "$NRL_TEST_TRACKER/copy-started" ]] || exit 97\n'
                '  if kill -0 "$command_pid" 2>/dev/null; then\n'
                '    : > "$NRL_TEST_TRACKER/deadline-fired"\n'
                '    kill -TERM "$command_pid"\n'
                "  fi\n"
                ") &\n"
                "watchdog_pid=$!\n"
                "set +e\n"
                'wait "$command_pid"\n'
                "command_rc=$?\n"
                "set -e\n"
                'kill "$watchdog_pid" 2>/dev/null || true\n'
                'wait "$watchdog_pid" 2>/dev/null || true\n'
                'if [[ -e "$NRL_TEST_TRACKER/deadline-fired" ]]; then exit 124; fi\n'
                'exit "$command_rc"\n',
                encoding="utf-8",
            )
            timeout_executable.chmod(0o700)

            rsync_executable = executable_dir / "rsync"
            rsync_executable.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "trap ': > \"$NRL_TEST_TRACKER/copy-terminated\"; exit 143' TERM\n"
                ': > "$NRL_TEST_TRACKER/copy-started"\n'
                "deadline=$((SECONDS + 3))\n"
                "while (( SECONDS < deadline )); do :; done\n"
                "exit 96\n",
                encoding="utf-8",
            )
            rsync_executable.chmod(0o700)

            script = (
                "set -euo pipefail\n"
                + final_sync_helper_source()
                + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
if atomic-final-ray-logs "$2" "$3" head 1; then
  exit 93
fi
"""
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    script,
                    "blocking-copy",
                    str(root),
                    str(source),
                    str(target),
                ],
                env={
                    "NRL_TEST_TRACKER": str(tracker),
                    "PATH": f"{executable_dir}:/usr/bin:/bin",
                },
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((tracker / "copy-started").is_file())
            self.assertTrue((tracker / "deadline-fired").is_file())
            self.assertTrue((tracker / "copy-terminated").is_file())
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
