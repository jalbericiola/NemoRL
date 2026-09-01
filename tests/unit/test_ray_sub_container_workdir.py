from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"
POLICY_BEGIN = "# BEGIN RAY_CONTAINER_WORKDIR_POLICY\n"
POLICY_END = "# END RAY_CONTAINER_WORKDIR_POLICY\n"


def container_workdir_policy_source() -> str:
    source = RAY_SUB.read_text(encoding="utf-8")
    if source.count(POLICY_BEGIN) != 1 or source.count(POLICY_END) != 1:
        raise AssertionError("Ray container-workdir policy sentinels must be unique")
    return source.split(POLICY_BEGIN, 1)[1].split(POLICY_END, 1)[0]


def run_container_workdir_policy(
    path: str, *, use_submit_dir_default: bool = False
) -> subprocess.CompletedProcess[str]:
    if use_submit_dir_default:
        input_assignment = 'unset CONTAINER_CWD\nSLURM_SUBMIT_DIR="$1"\n'
    else:
        input_assignment = 'CONTAINER_CWD="$1"\nSLURM_SUBMIT_DIR=/fallback\n'
    script = (
        "set -euo pipefail\n"
        + input_assignment
        + container_workdir_policy_source()
        + "\n"
        # Exercise the same whitespace-separated argument representation used
        # by COMMON_SRUN_ARGS. Accepted punctuation must remain one literal
        # argv entry; dangerous splitting/expansion characters stay rejected.
        + 'COMMON_SRUN_ARGS=" --container-workdir=$CONTAINER_CWD"\n'
        + "set -- $COMMON_SRUN_ARGS\n"
        + '[[ "$#" -eq 1 ]]\n'
        + 'printf "%s\\n" "$1"\n'
    )
    return subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script, "policy", path],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        check=False,
        capture_output=True,
        text=True,
    )


class RayContainerWorkdirPolicyTest(unittest.TestCase):
    def test_safe_normalized_paths_with_restored_punctuation_are_allowed(self) -> None:
        safe_paths = (
            "/lustre/sweeps/lr=1e-6+mtp@owner:run41",
            "/versions/build@sha:deadbeef/model+optimizer=te",
            "/opt/nemo-rl/.cache_under-score",
        )
        for path in safe_paths:
            with self.subTest(path=path):
                result = run_container_workdir_policy(path)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"--container-workdir={path}\n")

    def test_submit_directory_default_accepts_restored_punctuation(self) -> None:
        path = "/lustre/runs/lr=1e-6+mtp@team:repro"
        result = run_container_workdir_policy(path, use_submit_dir_default=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"--container-workdir={path}\n")

    def test_non_normalized_or_shell_active_text_remains_rejected(self) -> None:
        unsafe_paths = (
            "",
            "relative/path",
            "/",
            "/trailing/",
            "/double//slash",
            "/./child",
            "/parent/../child",
            "/parent/.",
            "/parent/..",
            "/space here",
            "/tab\there",
            "/newline\nhere",
            "/semicolon;touch",
            "/ampersand&command",
            "/pipe|command",
            "/redirect>file",
            "/dollar$HOME",
            "/subshell$(id)",
            "/backtick`id`",
            "/single'quote",
            '/double"quote',
            "/glob*star",
            "/glob?mark",
            "/bracket[abc]",
            "/back\\slash",
            "/" + "a" * 4096,
        )
        for path in unsafe_paths:
            with self.subTest(path=path):
                result = run_container_workdir_policy(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("CONTAINER_CWD must be", result.stderr)

    def test_policy_mentions_every_permitted_punctuation_character(self) -> None:
        source = container_workdir_policy_source()
        for character in ("'/'", "'-'", "'+'", "'='", "'@'", "':'"):
            with self.subTest(character=character):
                self.assertIn(character, source)


if __name__ == "__main__":
    unittest.main()
