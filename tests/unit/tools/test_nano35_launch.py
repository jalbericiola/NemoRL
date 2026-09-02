# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"


def _run_nano_launcher(*hydra_overrides: str, **env_overrides: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        config = root / "single-env.yaml"
        config.write_text("cluster: {}\n", encoding="utf-8")
        ray_sub = root / "ray.sub"
        ray_sub.write_text("#!/bin/bash\n", encoding="utf-8")

        env = {
            "HOME": str(root),
            "PATH": os.environ["PATH"],
            "DRY_RUN": "1",
            "USE_SNAPSHOT": "0",
            "EXP_NAME": "nano35-colocated-launcher-test",
            "CONFIG_PATH": str(config),
            "MODEL_PATH": "test-policy-model",
            "TRAIN_PATH": str(root / "train.jsonl"),
            "VAL_PATH": str(root / "validation.jsonl"),
            "CONTAINER": "test-container",
            "SANDBOX_CONTAINER": "test-sandbox-container",
            "PERSISTENT_CACHE": str(root / "cache"),
            "RESULTS_DIR": str(root / "results"),
            "RAY_SUB": str(ray_sub),
            "BATCH_SCRIPT": str(ray_sub),
            "SLURM_PARTITION": "test-partition",
            "SLURM_ACCOUNT": "test-account",
            "NUM_TRAIN_NODES": "1",
            "NUM_GEN_NODES": "0",
            "NUM_GYM_NODES": "0",
            "SEGMENT_SIZE": "1",
            "GPUS_PER_NODE": "4",
            "COLOCATED_GENERATION": "1",
            "TRAIN_ENTRYPOINT": "./examples/run_grpo_single_controller.py",
        }
        env.update({key: str(value) for key, value in env_overrides.items()})
        return subprocess.run(
            ["bash", str(LAUNCHER), "swe", *hydra_overrides],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )


class Nano35LaunchContractTest(unittest.TestCase):
    def test_colocated_dry_run_builds_exact_one_node_ipc_topology(self):
        result = _run_nano_launcher("grpo.max_num_steps=1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Nemotron 3.5 Nano", result.stdout)
        self.assertIn("(1-node)", result.stdout)
        self.assertIn(
            "vLLM gen:  colocated on the training node "
            "(1 node, 4 shared GPUs, CUDA-IPC refit)",
            result.stdout,
        )
        self.assertIn(
            "Ray head:  shared on the GPU node (DEDICATED_RAY_HEAD=0)",
            result.stdout,
        )
        for exact_override in (
            "cluster.num_nodes=1",
            "cluster.segment_size=1",
            "cluster.gpus_per_node=4",
            "policy.generation.backend=vllm",
            "policy.generation.colocated.enabled=true",
            "policy.generation.colocated.resources.num_nodes=1",
            "policy.generation.colocated.resources.gpus_per_node=4",
            "++policy.generation.refit_transport=null",
            "env.nemo_gym.num_gpu_nodes=0",
            "grpo.max_num_steps=1",
        ):
            with self.subTest(exact_override=exact_override):
                self.assertIn(exact_override, result.stdout)
        self.assertNotIn(
            "policy.generation.colocated.resources.num_nodes=0", result.stdout
        )

    def test_non_colocated_zero_generation_nodes_keeps_existing_rejection(self):
        result = _run_nano_launcher(COLOCATED_GENERATION="0")

        self.assertEqual(result.returncode, 1)
        self.assertIn("NUM_GEN_NODES must be > 0 (got 0)", result.stderr)
        self.assertIn(
            "allowed only with the exact COLOCATED_GENERATION=1", result.stderr
        )

    def test_non_colocated_positive_generation_shape_is_unchanged(self):
        result = _run_nano_launcher(
            "grpo.max_num_steps=1",
            COLOCATED_GENERATION="0",
            NUM_GEN_NODES="1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(2-node)", result.stdout)
        self.assertIn("vLLM gen:  1  (4 GPUs)", result.stdout)
        self.assertIn("cluster.num_nodes=2", result.stdout)
        self.assertIn(
            "policy.generation.colocated.resources.num_nodes=1", result.stdout
        )
        self.assertNotIn("policy.generation.backend=vllm", result.stdout)
        self.assertNotIn("policy.generation.colocated.enabled=true", result.stdout)
        self.assertNotIn("++policy.generation.refit_transport=null", result.stdout)

    def test_colocated_mode_rejects_every_shape_mismatch(self):
        for key, value in (
            ("NUM_TRAIN_NODES", "2"),
            ("NUM_GEN_NODES", "1"),
            ("NUM_GYM_NODES", "1"),
            ("SEGMENT_SIZE", "2"),
            ("GPUS_PER_NODE", "8"),
        ):
            with self.subTest(key=key, value=value):
                result = _run_nano_launcher(**{key: value})
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    "COLOCATED_GENERATION=1 requires the exact one-node/four-GPU shape",
                    result.stderr,
                )
                self.assertIn(
                    "NUM_TRAIN_NODES=1 NUM_GEN_NODES=0 NUM_GYM_NODES=0",
                    result.stderr,
                )

    def test_colocated_mode_rejects_dedicated_ray_head(self):
        result = _run_nano_launcher(DEDICATED_RAY_HEAD="1")

        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "COLOCATED_GENERATION=1 requires DEDICATED_RAY_HEAD=0",
            result.stderr,
        )

    def test_colocated_mode_rejects_protected_hydra_overrides(self):
        for override in (
            "cluster.num_nodes=2",
            "+cluster.gpus_per_node=8",
            "++cluster.segment_size=2",
            "policy.generation.backend=sglang",
            "policy.generation.colocated.enabled=false",
            "policy.generation.colocated.resources.num_nodes=0",
            "policy.generation.colocated.resources.gpus_per_node=2",
            "policy.generation.refit_transport=nccl_reshard",
            "~policy.generation.colocated",
            "env.nemo_gym.num_gpu_nodes=1",
            "+env.nemo_gym.num_gpu_nodes=1",
            "++env.nemo_gym.num_gpu_nodes=1",
            "~env.nemo_gym.num_gpu_nodes",
        ):
            with self.subTest(override=override):
                result = _run_nano_launcher(override)
                self.assertEqual(result.returncode, 1)
                self.assertIn(
                    "forbids overriding launcher-owned Hydra key", result.stderr
                )

    def test_colocated_mode_rejects_non_boolean_selector(self):
        result = _run_nano_launcher(COLOCATED_GENERATION="true")

        self.assertEqual(result.returncode, 1)
        self.assertIn("COLOCATED_GENERATION must be exactly 0 or 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
