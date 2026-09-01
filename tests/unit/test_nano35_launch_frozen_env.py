from __future__ import annotations

import os
import re
import shlex
import subprocess
import unittest
from pathlib import Path
from typing import Mapping

NEMO_ROOT = Path(__file__).resolve().parents[2]
NANO35_LAUNCH = NEMO_ROOT / "examples" / "nemo_gym" / "nemotron-3.5-nano" / "nano35_launch.sh"
PINNED_PROJECT_ENVIRONMENT = "/opt/nemo_rl_venv"
PINNED_UV = "/root/.local/bin/uv"
HF_CACHE_BUILDER_BEGIN = "# BEGIN NANO35_OPTIONAL_HF_CACHE_ENV_BUILDER\n"
HF_CACHE_BUILDER_END = "# END NANO35_OPTIONAL_HF_CACHE_ENV_BUILDER\n"


def optional_hf_cache_env_builder() -> str:
    source = NANO35_LAUNCH.read_text(encoding="utf-8")
    if source.count(HF_CACHE_BUILDER_BEGIN) != 1 or source.count(HF_CACHE_BUILDER_END) != 1:
        raise AssertionError("nano35 HF cache builder sentinels must be unique")
    return source.split(HF_CACHE_BUILDER_BEGIN, 1)[1].split(HF_CACHE_BUILDER_END, 1)[0]


def training_command_assignment() -> str:
    source = NANO35_LAUNCH.read_text(encoding="utf-8")
    start_marker = 'TRAIN_CMD="'
    end_marker = '\n\nexport COMMAND="${TRAIN_CMD}"'
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise AssertionError("nano35 TRAIN_CMD assignment markers must be unique")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def render_training_command(
    variable_values: Mapping[str, str] | None = None,
) -> str:
    assignment = training_command_assignment()
    names = sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", assignment)))
    values = {name: f"value-for-{name.lower()}" for name in names}
    values.update(
        {
            "CODE_ROOT": "/opt/nemo-rl",
            "TRAIN_ENTRYPOINT": "./examples/run_grpo_single_controller.py",
            "VLLM_ENV_SOURCE": "",
        }
    )
    if variable_values is not None:
        values.update({name: value for name, value in variable_values.items() if name in names})
    definitions = "\n".join(f"{name}={shlex.quote(values[name])}" for name in names)
    script = (
        "set -euo pipefail\n"
        f"{definitions}\n"
        f"{optional_hf_cache_env_builder()}\n"
        "set -- policy.shared_prefix_training.mode=train\n"
        f"{assignment}\n"
        'printf "%s\\n" "$TRAIN_CMD"\n'
    )
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if variable_values is not None:
        environment.update(variable_values)
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.rstrip("\n")


class Nano35FrozenDriverLaunchTest(unittest.TestCase):
    def test_warm_command_omits_empty_optional_hf_cache_paths(self) -> None:
        command = render_training_command(
            {
                "HF_HOME": "/shared/hf cache/team+mtp@owner:run=41",
                "HF_MODULES_CACHE": "",
                "HF_HUB_CACHE": "/shared/hf cache/hub",
                "HF_DATASETS_CACHE": "/shared/hf cache/datasets",
            }
        )
        tokens = shlex.split(command)

        self.assertIn(
            "HF_HOME=/shared/hf cache/team+mtp@owner:run=41",
            tokens,
        )
        self.assertIn("HF_HUB_CACHE=/shared/hf cache/hub", tokens)
        self.assertIn("HF_DATASETS_CACHE=/shared/hf cache/datasets", tokens)
        self.assertFalse(any(token.startswith("HF_MODULES_CACHE=") for token in tokens))

    def test_warm_command_omits_all_absent_or_empty_hf_cache_paths(self) -> None:
        command = render_training_command(
            {
                "HF_HOME": "",
                "HF_MODULES_CACHE": "",
                "HF_HUB_CACHE": "",
                "HF_DATASETS_CACHE": "",
            }
        )
        tokens = shlex.split(command)

        for name in (
            "HF_HOME",
            "HF_MODULES_CACHE",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
        ):
            with self.subTest(name=name):
                self.assertFalse(any(token.startswith(f"{name}=") for token in tokens))

    def test_cold_command_retains_fixed_hf_cache_paths_and_offline_mode(self) -> None:
        command = render_training_command(
            {
                "HF_HOME": "/tmp/nemo_rl_hf_home",
                "HF_MODULES_CACHE": "/tmp/nemo_rl_hf_modules_cache",
                "HF_HUB_CACHE": "/tmp/nemo_rl_hf_home/hub",
                "HF_DATASETS_CACHE": "/tmp/nemo_rl_hf_home/datasets",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        tokens = shlex.split(command)

        self.assertIn("HF_HOME=/tmp/nemo_rl_hf_home", tokens)
        self.assertIn(
            "HF_MODULES_CACHE=/tmp/nemo_rl_hf_modules_cache",
            tokens,
        )
        self.assertIn("HF_HUB_CACHE=/tmp/nemo_rl_hf_home/hub", tokens)
        self.assertIn(
            "HF_DATASETS_CACHE=/tmp/nemo_rl_hf_home/datasets",
            tokens,
        )
        self.assertIn("HF_HUB_OFFLINE=1", tokens)
        self.assertIn("TRANSFORMERS_OFFLINE=1", tokens)

    def test_training_command_pins_no_sync_uv_runtime(self) -> None:
        command = render_training_command()
        tokens = shlex.split(command)
        uv_index = tokens.index(PINNED_UV)

        self.assertEqual(
            tokens[uv_index - 1],
            f"UV_PROJECT_ENVIRONMENT={PINNED_PROJECT_ENVIRONMENT}",
        )
        self.assertEqual(
            tokens[uv_index : uv_index + 4],
            [
                PINNED_UV,
                "run",
                "--no-sync",
                "./examples/run_grpo_single_controller.py",
            ],
        )
        self.assertNotIn("uv", tokens)

    def test_training_command_has_no_syncing_uv_fallback(self) -> None:
        assignment = training_command_assignment()
        self.assertEqual(assignment.count(PINNED_UV), 1)
        self.assertEqual(assignment.count("UV_PROJECT_ENVIRONMENT="), 1)
        self.assertEqual(assignment.count("run --no-sync"), 1)
        self.assertNotRegex(assignment, r"(?:^|[ \t])uv[ \t]+run(?:[ \t]|$)")
        self.assertNotIn("NRL_IGNORE_VERSION_MISMATCH", assignment)

    def test_launcher_remains_valid_bash(self) -> None:
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-n", str(NANO35_LAUNCH)],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
