from __future__ import annotations

import os
import subprocess
import tempfile
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


def run_watchdog_policy(
    node_count: str,
    base_seconds: str = "300",
    per_node_seconds: str = "15",
    maximum_seconds: str = "3600",
) -> subprocess.CompletedProcess[str]:
    script = (
        "set -euo pipefail\n" + watchdog_policy_source() + '\nray-final-sync-watchdog-seconds "$1" "$2" "$3" "$4"\n'
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
        self.assertEqual(
            source.count('sleep "$RAY_FINAL_SYNC_WATCHDOG_SECONDS"'),
            1,
        )
        self.assertNotIn('sleep 300\n      if kill -0 "$worker_pid"', source)
        self.assertEqual(source.count(".RAY_FINAL_SYNC_LOCK"), 0)
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
                (source / "worker.log").write_text(f"worker-{worker_index}\n", encoding="utf-8")
                targets.append(root / "ray" / f"worker-node-{worker_index}" / "session_scale_test" / "logs")

            script = "set -euo pipefail\n" + final_sync_helper_source() + r"""
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

            duplicate_script = "set -euo pipefail\n" + final_sync_helper_source() + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
if atomic-final-ray-logs "$2" "$3" worker-0; then
  exit 90
fi
"""
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
            failure_script = "set -euo pipefail\n" + final_sync_helper_source() + r"""
NRL_PROVENANCE_LOG_DIR="$1"
export NRL_PROVENANCE_LOG_DIR
rsync() { return 17; }
if atomic-final-ray-logs "$2" "$3" worker-0; then
  exit 91
fi
"""
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


if __name__ == "__main__":
    unittest.main()
