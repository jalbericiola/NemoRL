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

"""Measure the numerical floor of repeated captured-citation GPU arms.

This fail-closed diagnostic is deliberately separate from the authoritative
OFF/ON parity harness.  ``off-forward`` and ``on-forward`` each load the model
once, snapshot every Python/NumPy/Torch/MCore RNG state, and execute serial and
default-overlap repetitions after restoring that snapshot before every probe.
They also execute a serial probe after an overlap primer in the same worker to
detect persistent stream/cache contamination.

Optional ``*-step-a``/``*-step-b`` arms reload the checkpoint in fresh process
groups and execute exactly one NeMo-RL optimizer step.  The comparator first
checks rank-local contracts and replicated-rank identity, then reports the
observed same-arm logprob, loss, gradient, and optimizer-update floors.  It
does not weaken or replace the independent OFF/ON acceptance thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import ray
import torch

def _import_parity_module(import_module: Callable[[str], Any] = importlib.import_module) -> Any:
    """Prefer the immutable /opt mount; use the repository module locally."""
    top_level_name = "shared_prefix_captured_citation_parity"
    try:
        return import_module(top_level_name)
    except ModuleNotFoundError as error:
        # Do not conceal a dependency failure raised while importing a present
        # top-level harness. Only its actual absence permits the local fallback.
        if error.name != top_level_name:
            raise
    return import_module("tools.model_diagnostics.shared_prefix_captured_citation_parity")


parity = _import_parity_module()


SCHEMA = "nemorl-shared-prefix-captured-citation-same-arm-v1"
BASE_SCHEMA = parity.SCHEMA
FORWARD_RUNS = ("off-forward", "on-forward")
STEP_RUNS = ("off-step-a", "off-step-b", "on-step-a", "on-step-b")
ALL_RUNS = FORWARD_RUNS + STEP_RUNS
EFFECTIVE_MCORE_SEED = 1234
FORWARD_PROBES = (
    "serial-a",
    "serial-b",
    "overlap-a",
    "overlap-b",
    "overlap-primer",
    "serial-after-overlap",
)


def _run_spec(
    run_id: str,
) -> tuple[Literal["off", "on"], Literal["forward", "a", "b"], Literal["forward", "step"]]:
    if run_id not in ALL_RUNS:
        raise parity.ParityError(f"unsupported same-arm run id: {run_id!r}")
    family: Literal["off", "on"] = "off" if run_id.startswith("off") else "on"
    repeat: Literal["forward", "a", "b"] = (
        "forward" if run_id.endswith("-forward") else ("a" if run_id.endswith("-a") else "b")
    )
    phase: Literal["forward", "step"] = "step" if "-step-" in run_id else "forward"
    return family, repeat, phase


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _rng_state_value(value: Any) -> Any:
    """Convert RNG tracker state to canonical, non-address-bearing evidence."""
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _tensor_sha256(value)}
    if isinstance(value, torch.Generator):
        return {"generator_state_sha256": _tensor_sha256(value.get_state())}
    if isinstance(value, Mapping):
        return {str(key): _rng_state_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_rng_state_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise parity.ParityError(f"unsupported RNG tracker state type: {type(value).__name__}")


def _clone_rng_state_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, torch.Generator):
        clone_state = getattr(value, "clone_state", None)
        if not callable(clone_state):
            raise parity.ParityError("graph-safe RNG generator lacks clone_state")
        return clone_state()
    if isinstance(value, Mapping):
        return {key: _clone_rng_state_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_rng_state_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_rng_state_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise parity.ParityError(f"unsupported RNG state clone type: {type(value).__name__}")


def _capture_rng_state() -> dict[str, Any]:
    from megatron.core.tensor_parallel.random import get_all_rng_states

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state().clone(),
        "mcore_cuda_tracker": _clone_rng_state_value(get_all_rng_states()),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda", "mcore_cuda_tracker"}:
        raise parity.ParityError("RNG restore state has an invalid schema")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].clone())
    torch.cuda.set_rng_state(state["torch_cuda"].clone())
    get_cuda_rng_tracker().set_states(_clone_rng_state_value(state["mcore_cuda_tracker"]))


def _rng_fingerprint(state: dict[str, Any] | None = None) -> dict[str, Any]:
    captured = _capture_rng_state() if state is None else state
    numpy_state = captured["numpy"]
    value = {
        "python": _rng_state_value(captured["python"]),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys_sha256": hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": _rng_state_value(captured["torch_cpu"]),
        "torch_cuda": _rng_state_value(captured["torch_cuda"]),
        "mcore_cuda_tracker": _rng_state_value(captured["mcore_cuda_tracker"]),
    }
    return {
        "sha256": parity._sha256_bytes(parity._canonical_json_bytes(value)),
        "components": value,
    }


def _run_repeat(arguments: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != parity.WORLD_SIZE:
        raise parity.ParityError(f"run-repeat requires torchrun WORLD_SIZE={parity.WORLD_SIZE}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or not 0 <= rank < parity.WORLD_SIZE:
        raise parity.ParityError(f"invalid rank identity rank={rank} local_rank={local_rank}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != parity.WORLD_SIZE:
        raise parity.ParityError(
            f"run-repeat requires exactly four visible CUDA devices, got {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise parity.ParityError("run-repeat requires CUDA BF16 support")

    family, repeat, phase = _run_spec(arguments.run_id)
    model_path = Path(arguments.model_path)
    if not model_path.is_absolute() or not (model_path / "config.json").is_file():
        raise parity.ParityError(f"model path is not an absolute HF checkpoint: {model_path}")
    repo_root = Path(arguments.repo_root)
    try:
        resolved_repo_root = repo_root.resolve(strict=True)
    except OSError as error:
        raise parity.ParityError(f"cannot resolve NeMo repo root {repo_root}: {error}") from error
    if (
        not repo_root.is_absolute()
        or not repo_root.is_dir()
        or repo_root.is_symlink()
        or resolved_repo_root != repo_root
    ):
        raise parity.ParityError(f"NeMo repo root must be an absolute canonical non-symlink directory: {repo_root}")
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute() or not output_dir.is_dir():
        raise parity.ParityError(f"output directory must already exist and be absolute: {output_dir}")
    output_path = output_dir / f"{arguments.run_id}.rank{rank}.json"
    if output_path.exists() or output_path.is_symlink():
        raise parity.ParityError(f"refusing to replace same-arm evidence: {output_path}")

    rows, batch_summary = parity._read_captured_rows(
        Path(arguments.batch), expected_sha256=arguments.expected_batch_sha256
    )
    shared_prefix_mode: Literal["observe", "train"] = "observe" if family == "off" else "train"
    policy_config, loss_config, config_sha256 = parity._build_policy_and_loss_config(
        str(model_path), repo_root=repo_root, shared_prefix_mode=shared_prefix_mode
    )
    batch_preparation = parity._preflight_batch_preparation(
        rows, policy_config=policy_config, shared_prefix_mode=shared_prefix_mode
    )

    from megatron.core.models.hybrid import shared_prefix_fused
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.models.megatron.setup import destroy_parallel_state
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    # R18 used the default overlapping stream path and disabled only fused merge.
    if shared_prefix_fused._SP_STREAMS is not True:
        raise parity.ParityError("same-arm R18 diagnostic requires default NRL_SP_STREAMS=1")
    if os.environ.get("NRL_SP_FUSED_MERGE") != "0":
        raise parity.ParityError("same-arm R18 diagnostic requires NRL_SP_FUSED_MERGE=0")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    ray.get_gpu_ids = lambda: [local_rank]
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    worker = None
    restore_runtime_counters = None
    original_stream_overlap = shared_prefix_fused._SP_STREAMS
    optimizer_probe: dict[str, Any] = {"calls": 0}
    try:
        worker = MegatronPolicyWorkerImpl(
            config=policy_config,
            tokenizer=tokenizer,
            # Match R18 construction exactly even for the forward-only probes;
            # the optimizer remains lazy and is never stepped in that phase.
            init_optimizer=True,
            init_reference_model=False,
            worker_sharding_annotations=parity._worker_sharding(),
        )
        model_config = worker._get_model_config()
        topology = {
            "world_size": torch.distributed.get_world_size(),
            "tensor_parallel_size": model_config.tensor_model_parallel_size,
            "context_parallel_size": model_config.context_parallel_size,
            "sequence_parallel": model_config.sequence_parallel,
            "expert_parallel_size": model_config.expert_model_parallel_size,
            "expert_tensor_parallel_size": model_config.expert_tensor_parallel_size,
            "pipeline_parallel_size": model_config.pipeline_model_parallel_size,
            "mtp_num_layers": model_config.mtp_num_layers,
            "mtp_use_repeated_layer": model_config.mtp_use_repeated_layer,
            "mtp_detach_heads": model_config.mtp_detach_heads,
        }
        if topology != parity._topology():
            raise parity.ParityError(f"runtime topology mismatch: {topology}")
        if model_config.num_moe_experts != 128:
            raise parity.ParityError(f"Nano must construct 128 experts, got {model_config.num_moe_experts}")
        if worker.megatron_cfg.rng.seed != EFFECTIVE_MCORE_SEED:
            raise parity.ParityError(
                "exact R18 Bridge/MCore RNG seed differs: "
                f"{worker.megatron_cfg.rng.seed} != {EFFECTIVE_MCORE_SEED}"
            )
        expected_enabled = family == "on"
        if worker._shared_prefix_training_enabled is not expected_enabled:
            raise parity.ParityError(
                "worker shared-prefix activation differs from repeat contract: "
                f"expected={expected_enabled} actual={worker._shared_prefix_training_enabled}"
            )
        runtime_counts, runtime_implementations, restore_runtime_counters = (
            parity._install_shared_prefix_runtime_counters()
        )
        evidence: dict[str, Any] = {
            "schema": SCHEMA,
            "base_schema": BASE_SCHEMA,
            "run_id": arguments.run_id,
            # Compatibility identity consumed by the immutable R18 gradient
            # and optimizer validators reused by the optional step comparator.
            "arm": family,
            "arm_family": family,
            "repeat": repeat,
            "phase": phase,
            "rank": rank,
            "local_rank": local_rank,
            "batch": batch_summary,
            "config_sha256": config_sha256,
            "shared_prefix_mode": shared_prefix_mode,
            "topology": topology,
            "repo_root": str(repo_root),
            "model_path": str(model_path),
            "requested_harness_seed": arguments.seed,
            "effective_mcore_seed": worker.megatron_cfg.rng.seed,
            "batch_preparation": batch_preparation,
            "worker_shared_prefix_training_enabled": worker._shared_prefix_training_enabled,
            "runtime_path_implementations": runtime_implementations,
            "stream_overlap_default": original_stream_overlap,
            "optimizer_master_reconstruction_self_test": parity._master_remainder_reconstruction_self_test(),
        }

        if phase == "forward":
            initial_rng_state = _capture_rng_state()
            initial_rng_sha256 = _rng_fingerprint(initial_rng_state)["sha256"]
            probes: dict[str, Any] = {}

            def forward_probe(probe: str, *, stream_overlap: bool) -> None:
                _restore_rng_state(initial_rng_state)
                shared_prefix_fused._SP_STREAMS = stream_overlap
                rng_before = _rng_fingerprint()
                calls_before = parity._runtime_path_snapshot(runtime_counts)
                selected_logprobs = parity._selected_logprobs(
                    worker,
                    rows,
                    policy_config=policy_config,
                    shared_prefix_mode=shared_prefix_mode,
                )
                torch.cuda.synchronize()
                rng_after = _rng_fingerprint()
                calls_after = parity._runtime_path_snapshot(runtime_counts)
                call_delta = {
                    name: calls_after[name] - calls_before[name] for name in parity.RUNTIME_PATH_COUNTERS
                }
                if expected_enabled and any(call_delta[name] <= 0 for name in parity.RUNTIME_PATH_COUNTERS):
                    raise parity.ParityError(
                        f"shared-prefix forward probe {probe} skipped runtime paths: {call_delta}"
                    )
                if not expected_enabled and any(call_delta.values()):
                    raise parity.ParityError(f"dense forward probe {probe} called shared-prefix paths: {call_delta}")
                probes[probe] = {
                    "stream_overlap": stream_overlap,
                    "rng_before": rng_before,
                    "rng_after": rng_after,
                    "runtime_path_call_delta": call_delta,
                    "selected_logprobs": selected_logprobs,
                }

            # Serial baselines run before any overlap probe. The final serial
            # probe runs after an overlap primer without rebuilding the worker.
            forward_probe("serial-a", stream_overlap=False)
            forward_probe("serial-b", stream_overlap=False)
            forward_probe("overlap-a", stream_overlap=True)
            forward_probe("overlap-b", stream_overlap=True)
            forward_probe("overlap-primer", stream_overlap=True)
            forward_probe("serial-after-overlap", stream_overlap=False)
            if tuple(probes) != FORWARD_PROBES:
                raise parity.ParityError(f"forward probes executed in an unexpected order: {tuple(probes)}")
            evidence.update(
                {
                    "initial_rng_sha256": initial_rng_sha256,
                    "forward_probe_order": list(FORWARD_PROBES),
                    "forward_probes": probes,
                }
            )
        else:
            rng_before = _rng_fingerprint()
            selected_before = parity._selected_logprobs(
                worker, rows, policy_config=policy_config, shared_prefix_mode=shared_prefix_mode
            )
            torch.cuda.synchronize()
            rng_after_pre = _rng_fingerprint()
            calls_after_pre = parity._runtime_path_snapshot(runtime_counts)
            parity._require_runtime_path_counts(
                calls_after_pre, shared_prefix_enabled=expected_enabled, phase="pre-logprob"
            )
            evidence.update(
                {
                    "runtime_path_calls": {"after_pre_logprob": calls_after_pre},
                    "rng": {"before_pre_logprob": rng_before, "after_pre_logprob": rng_after_pre},
                    "selected_logprobs_before": selected_before,
                }
            )
            original_step = worker.optimizer.step

            def step_with_evidence(*args: Any, **kwargs: Any) -> Any:
                optimizer_probe["calls"] += 1
                if optimizer_probe["calls"] != 1:
                    raise parity.ParityError("optimizer.step was called more than once")
                torch.cuda.synchronize()
                optimizer_probe["gradients"] = parity._snapshot_finalized_gradients(worker.model)
                optimizer_probe["optimizer_state_initialization"] = parity._initialize_optimizer_state_for_probe(
                    worker.model, worker.optimizer, worker.scheduler
                )
                optimizer_probe["optimizer_masters_before"] = parity._snapshot_optimizer_masters(
                    worker.model, worker.optimizer
                )
                result = original_step(*args, **kwargs)
                torch.cuda.synchronize()
                optimizer_probe["optimizer_masters_after"] = parity._snapshot_optimizer_masters(
                    worker.model, worker.optimizer
                )
                return result

            worker.optimizer.step = step_with_evidence
            scheduler_before = worker.scheduler.num_steps
            worker.begin_train_step(loss_fn=ClippedPGLossFn(loss_config), gbs=parity.BATCH_SIZE, mbs=1)
            try:
                train_batch, _ = parity._prepare_batch_for_worker(
                    parity._build_batch(rows),
                    policy_config=policy_config,
                    shared_prefix_mode=shared_prefix_mode,
                    stage="train",
                )
                worker.train_microbatch(train_batch)
                metrics = worker.finish_train_step()
            except Exception:
                worker.abort_train_step()
                raise
            calls_after_train = parity._runtime_path_snapshot(runtime_counts)
            parity._require_runtime_path_counts(
                calls_after_train,
                shared_prefix_enabled=expected_enabled,
                phase="train",
                previous=calls_after_pre,
            )
            if optimizer_probe["calls"] != 1:
                raise parity.ParityError(f"expected one optimizer step, got {optimizer_probe['calls']}")
            if worker.scheduler.num_steps != scheduler_before + parity.BATCH_SIZE:
                raise parity.ParityError(
                    "scheduler did not advance by the captured global batch size: "
                    f"before={scheduler_before} after={worker.scheduler.num_steps}"
                )
            evidence.update(
                {
                    "training": parity._metric_evidence(metrics),
                    "gradients": optimizer_probe["gradients"],
                    "optimizer_state_initialization": optimizer_probe["optimizer_state_initialization"],
                    "optimizer_masters_before": optimizer_probe["optimizer_masters_before"],
                    "optimizer_masters_after": optimizer_probe["optimizer_masters_after"],
                    "selected_logprobs_after": parity._selected_logprobs(
                        worker, rows, policy_config=policy_config, shared_prefix_mode=shared_prefix_mode
                    ),
                }
            )
            torch.cuda.synchronize()
            evidence["rng"]["after_post_logprob"] = _rng_fingerprint()
            calls_after_post = parity._runtime_path_snapshot(runtime_counts)
            parity._require_runtime_path_counts(
                calls_after_post,
                shared_prefix_enabled=expected_enabled,
                phase="post-logprob",
                previous=calls_after_train,
            )
            evidence["runtime_path_calls"].update(
                {"after_train": calls_after_train, "after_post_logprob": calls_after_post}
            )

        parity._atomic_json(output_path, evidence)
        torch.distributed.barrier()
        if rank == 0:
            print(
                "NEMORL_CAPTURED_CITATION_SAME_ARM_REPEAT_GREEN "
                f"run_id={arguments.run_id} phase={phase} batch_sha256={batch_summary['sha256']} "
                f"selected_tokens={batch_summary['selected_tokens']} config_sha256={config_sha256}",
                flush=True,
            )
    finally:
        shared_prefix_fused._SP_STREAMS = original_stream_overlap
        if restore_runtime_counters is not None:
            restore_runtime_counters()
        if worker is not None:
            worker.shutdown()
        destroy_parallel_state()


def _load_repeat(root: Path, run_id: str) -> list[dict[str, Any]]:
    family, repeat, phase = _run_spec(run_id)
    values: list[dict[str, Any]] = []
    for rank in range(parity.WORLD_SIZE):
        path = root / f"{run_id}.rank{rank}.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
            raise parity.ParityError(f"missing, symlinked, or writable repeat evidence: {path}")
        with path.open(encoding="utf-8", errors="strict") as stream:
            value = json.load(
                stream,
                object_pairs_hook=parity._no_duplicate_keys,
                parse_constant=parity._reject_constant,
            )
        if not isinstance(value, dict):
            raise parity.ParityError(f"repeat evidence is not an object: {path}")
        if (
            value.get("schema") != SCHEMA
            or value.get("base_schema") != BASE_SCHEMA
            or value.get("run_id") != run_id
            or value.get("arm") != family
            or value.get("arm_family") != family
            or value.get("repeat") != repeat
            or value.get("phase") != phase
            or value.get("rank") != rank
            or value.get("local_rank") != rank
        ):
            raise parity.ParityError(f"repeat identity mismatch in {path}")
        values.append(value)
    return values


def _validate_repeat(value: dict[str, Any], *, run_id: str, rank: int) -> dict[str, Any]:
    family, _repeat, phase = _run_spec(run_id)
    common = {
        "schema",
        "base_schema",
        "run_id",
        "arm",
        "arm_family",
        "repeat",
        "phase",
        "rank",
        "local_rank",
        "batch",
        "config_sha256",
        "shared_prefix_mode",
        "topology",
        "repo_root",
        "model_path",
        "requested_harness_seed",
        "effective_mcore_seed",
        "batch_preparation",
        "worker_shared_prefix_training_enabled",
        "runtime_path_implementations",
        "stream_overlap_default",
        "optimizer_master_reconstruction_self_test",
    }
    forward = {"initial_rng_sha256", "forward_probe_order", "forward_probes"}
    training = {
        "runtime_path_calls",
        "rng",
        "selected_logprobs_before",
        "training",
        "gradients",
        "optimizer_state_initialization",
        "optimizer_masters_before",
        "optimizer_masters_after",
        "selected_logprobs_after",
    }
    parity._exact_dict(
        value,
        keys=common | (training if phase == "step" else forward),
        label=f"{run_id} rank {rank}",
    )
    if value["stream_overlap_default"] is not True:
        raise parity.ParityError(f"{run_id} rank {rank} did not begin on the exact R18 overlap path")
    if value["shared_prefix_mode"] != ("observe" if family == "off" else "train"):
        raise parity.ParityError(f"{run_id} rank {rank} has invalid shared-prefix mode")
    if value["worker_shared_prefix_training_enabled"] is not (family == "on"):
        raise parity.ParityError(f"{run_id} rank {rank} has invalid worker activation")
    if parity._int(value["requested_harness_seed"], label=f"{run_id} rank {rank} requested seed") != 42:
        raise parity.ParityError(f"{run_id} rank {rank} requested harness seed must be 42")
    if (
        parity._int(value["effective_mcore_seed"], label=f"{run_id} rank {rank} MCore seed")
        != EFFECTIVE_MCORE_SEED
    ):
        raise parity.ParityError(
            f"{run_id} rank {rank} effective MCore seed must be {EFFECTIVE_MCORE_SEED}"
        )
    parity._validated_sha256(value["config_sha256"], label=f"{run_id} rank {rank} config")
    batch = parity._validated_batch_summary(value["batch"], arm=family, rank=rank)
    preparation = parity._validated_batch_preparation_evidence(
        value["batch_preparation"], arm=family, rank=rank, batch=batch
    )
    parity._validated_topology_evidence(value["topology"], arm=family, rank=rank)
    if value["optimizer_master_reconstruction_self_test"] != parity._master_remainder_reconstruction_self_test():
        raise parity.ParityError(f"{run_id} rank {rank} has invalid master reconstruction self-test")
    for path_key in ("repo_root", "model_path"):
        if not isinstance(value[path_key], str) or not Path(value[path_key]).is_absolute():
            raise parity.ParityError(f"{run_id} rank {rank} {path_key} must be absolute")

    implementations = value["runtime_path_implementations"]
    if implementations != {
        "hybrid_stack": "forward_hybrid_stack_shared_prefix",
        "mamba_cp": "_forward_mamba_layer_shared_prefix_cp",
        "attention_cp": "flash_composed_forest_attention_cp",
    }:
        raise parity.ParityError(f"{run_id} rank {rank} runtime implementations differ")
    if phase == "forward":
        initial_rng_sha256 = parity._validated_sha256(
            value["initial_rng_sha256"], label=f"{run_id} rank {rank} initial RNG"
        )
        probes = value["forward_probes"]
        if value["forward_probe_order"] != list(FORWARD_PROBES):
            raise parity.ParityError(f"{run_id} rank {rank} forward probe execution order differs")
        if not isinstance(probes, dict) or set(probes) != set(FORWARD_PROBES):
            raise parity.ParityError(f"{run_id} rank {rank} forward probe order/schema differs")
        expected_streams = {
            "serial-a": False,
            "serial-b": False,
            "overlap-a": True,
            "overlap-b": True,
            "overlap-primer": True,
            "serial-after-overlap": False,
        }
        for probe_name, probe_value in probes.items():
            probe = parity._exact_dict(
                probe_value,
                keys={
                    "stream_overlap",
                    "rng_before",
                    "rng_after",
                    "runtime_path_call_delta",
                    "selected_logprobs",
                },
                label=f"{run_id} rank {rank} {probe_name}",
            )
            if probe["stream_overlap"] is not expected_streams[probe_name]:
                raise parity.ParityError(f"{run_id} rank {rank} {probe_name} stream mode differs")
            for rng_position in ("rng_before", "rng_after"):
                fingerprint = parity._exact_dict(
                    probe[rng_position],
                    keys={"sha256", "components"},
                    label=f"{run_id} rank {rank} {probe_name} {rng_position}",
                )
                parity._validated_sha256(
                    fingerprint["sha256"], label=f"{run_id} rank {rank} {probe_name} {rng_position}"
                )
                if fingerprint["sha256"] != parity._sha256_bytes(
                    parity._canonical_json_bytes(fingerprint["components"])
                ):
                    raise parity.ParityError(
                        f"{run_id} rank {rank} {probe_name} {rng_position} digest differs"
                    )
            if probe["rng_before"]["sha256"] != initial_rng_sha256:
                raise parity.ParityError(f"{run_id} rank {rank} {probe_name} did not restore initial RNG")
            deltas = probe["runtime_path_call_delta"]
            if not isinstance(deltas, dict) or set(deltas) != set(parity.RUNTIME_PATH_COUNTERS):
                raise parity.ParityError(f"{run_id} rank {rank} {probe_name} path deltas differ")
            validated_deltas = {
                name: parity._int(deltas[name], label=f"{run_id} {probe_name} {name}")
                for name in parity.RUNTIME_PATH_COUNTERS
            }
            if family == "off" and any(validated_deltas.values()):
                raise parity.ParityError(f"{run_id} rank {rank} {probe_name} called shared-prefix paths")
            if family == "on" and any(value <= 0 for value in validated_deltas.values()):
                raise parity.ParityError(f"{run_id} rank {rank} {probe_name} skipped shared-prefix paths")
            parity._logprob_metrics(
                probe["selected_logprobs"],
                probe["selected_logprobs"],
                label=f"{run_id} rank {rank} {probe_name} selected logprobs",
            )
        return {"batch": batch, "batch_preparation": preparation, "forward_probes": probes}

    expected_phases = {
        "after_pre_logprob",
        "after_train",
        "after_post_logprob",
    }
    raw_calls = value["runtime_path_calls"]
    if not isinstance(raw_calls, dict) or set(raw_calls) != expected_phases:
        raise parity.ParityError(f"{run_id} rank {rank} runtime counter phases differ")
    calls: dict[str, dict[str, int]] = {}
    for call_phase, raw_counts in raw_calls.items():
        if not isinstance(raw_counts, dict) or set(raw_counts) != set(parity.RUNTIME_PATH_COUNTERS):
            raise parity.ParityError(f"{run_id} rank {rank} {call_phase} counters differ")
        calls[call_phase] = {
            name: parity._int(raw_counts[name], label=f"{run_id} {call_phase} {name}")
            for name in parity.RUNTIME_PATH_COUNTERS
        }
    pre = calls["after_pre_logprob"]
    if family == "off" and any(count for counts in calls.values() for count in counts.values()):
        raise parity.ParityError(f"{run_id} rank {rank} dense path called shared-prefix helpers")
    if family == "on" and any(pre[name] <= 0 for name in parity.RUNTIME_PATH_COUNTERS):
        raise parity.ParityError(f"{run_id} rank {rank} skipped a shared-prefix helper")
    if family == "on":
        train = calls["after_train"]
        post = calls["after_post_logprob"]
        if any(not pre[name] < train[name] < post[name] for name in parity.RUNTIME_PATH_COUNTERS):
            raise parity.ParityError(f"{run_id} rank {rank} shared-prefix counters did not increase")

    rng = value["rng"]
    expected_rng_phases = {"before_pre_logprob", "after_pre_logprob"}
    expected_rng_phases.add("after_post_logprob")
    if not isinstance(rng, dict) or set(rng) != expected_rng_phases:
        raise parity.ParityError(f"{run_id} rank {rank} RNG phases differ")
    for rng_phase, fingerprint in rng.items():
        item = parity._exact_dict(
            fingerprint, keys={"sha256", "components"}, label=f"{run_id} rank {rank} RNG {rng_phase}"
        )
        parity._validated_sha256(item["sha256"], label=f"{run_id} rank {rank} RNG {rng_phase}")
        if item["sha256"] != parity._sha256_bytes(parity._canonical_json_bytes(item["components"])):
            raise parity.ParityError(f"{run_id} rank {rank} RNG {rng_phase} digest differs")
    result: dict[str, Any] = {"batch": batch, "batch_preparation": preparation, "calls": calls}
    result["optimizer_state_initialization"] = parity._validated_optimizer_state_initialization(
        value["optimizer_state_initialization"], arm=family, rank=rank
    )
    return result


def _record(report: dict[str, Any], label: str, passed: bool, details: Any) -> None:
    parity._record_check(report, label=label, passed=passed, details=details)


def _same_arm_pair(
    report: dict[str, Any],
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    family: Literal["off", "on"],
) -> dict[str, Any]:
    pair_report: dict[str, Any] = {"ranks": {}}
    before_floors: list[dict[str, Any]] = []
    after_floors: list[dict[str, Any]] = []
    for rank, (a_value, b_value) in enumerate(zip(first, second, strict=True)):
        before = parity._logprob_metrics(
            a_value["selected_logprobs_before"],
            b_value["selected_logprobs_before"],
            label=f"{family} step rank {rank} repeat pre-logprobs",
        )
        before["bitwise_equal"] = a_value["selected_logprobs_before"] == b_value["selected_logprobs_before"]
        before_floors.append(before)
        _record(
            report,
            f"{family} step rank {rank} pre-logprob repeat floor",
            before["max_sequence_mean_exp_abs_delta"] < parity.LOGPROB_FILTER_LIMIT
            and before["relative_l2"] < 0.03
            and before["cosine"] > 0.995,
            before,
        )
        rank_report: dict[str, Any] = {"selected_logprobs_before": before}
        after = parity._logprob_metrics(
            a_value["selected_logprobs_after"],
            b_value["selected_logprobs_after"],
            label=f"{family} step rank {rank} repeat post-logprobs",
        )
        after["bitwise_equal"] = a_value["selected_logprobs_after"] == b_value["selected_logprobs_after"]
        after_floors.append(after)
        _record(
            report,
            f"{family} step rank {rank} post-logprob repeat floor",
            after["max_sequence_mean_exp_abs_delta"] < parity.LOGPROB_FILTER_LIMIT
            and after["relative_l2"] < 0.03
            and after["cosine"] > 0.995,
            after,
        )
        training_a = a_value["training"]
        training_b = b_value["training"]
        loss_delta = abs(
            parity._finite_float(training_a["global_loss"], label="repeat A global loss")
            - parity._finite_float(training_b["global_loss"], label="repeat B global loss")
        )
        rank_report["global_loss_abs_delta"] = loss_delta
        _record(report, f"{family} step rank {rank} RL loss repeat floor", loss_delta <= 1.0e-6, loss_delta)
        rank_report["families"] = {}
        for gradient_family in parity.REQUIRED_GRADIENT_FAMILIES:
            a_norms, a_samples = parity._family_gradient_vectors(a_value, gradient_family)
            b_norms, b_samples = parity._family_gradient_vectors(b_value, gradient_family)
            norms = parity._vector_metrics(
                a_norms, b_norms, label=f"{family} rank {rank} {gradient_family} repeat gradient norms"
            )
            samples = parity._vector_metrics(
                a_samples, b_samples, label=f"{family} rank {rank} {gradient_family} repeat gradient samples"
            )
            rank_report["families"][gradient_family] = {"norms": norms, "samples": samples}
            _record(
                report,
                f"{family} step rank {rank} {gradient_family} gradient repeat floor",
                norms["relative_l2"] < 0.05
                and norms["cosine"] > 0.995
                and samples["relative_l2"] < 0.10
                and samples["cosine"] > 0.99,
                rank_report["families"][gradient_family],
            )
        pair_report["ranks"][str(rank)] = rank_report

    def floor(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "max_abs": max(item["max_abs"] for item in items),
            "max_relative_l2": max(item["relative_l2"] for item in items),
            "min_cosine": min(item["cosine"] for item in items),
            "max_sequence_mean_exp_abs_delta": max(
                item["max_sequence_mean_exp_abs_delta"] for item in items
            ),
            "all_bitwise_equal": all(item["bitwise_equal"] for item in items),
        }

    pair_report["selected_logprobs_before_floor"] = floor(before_floors)
    pair_report["selected_logprobs_after_floor"] = floor(after_floors)
    pair_report["optimizer_families"] = {}
    for gradient_family in parity.REQUIRED_GRADIENT_FAMILIES:
        aggregate = parity._aggregate_family_optimizer_updates(first, second, gradient_family)
        pair_report["optimizer_families"][gradient_family] = aggregate
        update = aggregate["update_projection"]
        _record(
            report,
            f"{family} step global {gradient_family} optimizer repeat floor",
            aggregate["initial_exact_match"]
            and update["relative_l2"] < 0.10
            and update["cosine"] > 0.99,
            aggregate,
        )
    return pair_report


def _forward_same_worker(
    report: dict[str, Any],
    values: list[dict[str, Any]],
    *,
    family: Literal["off", "on"],
) -> dict[str, Any]:
    comparisons = {
        "R19_SERIAL_REPEAT": ("serial-a", "serial-b"),
        "R19_OVERLAP_REPEAT": ("overlap-a", "overlap-b"),
        "R19_SERIAL_VS_OVERLAP": ("serial-a", "overlap-a"),
        "R19_SERIAL_AFTER_OVERLAP": ("serial-a", "serial-after-overlap"),
    }
    result: dict[str, Any] = {"ranks": {}, "floors": {label: [] for label in comparisons}}
    for rank, value in enumerate(values):
        probes = value["forward_probes"]
        rank_report: dict[str, Any] = {}
        before_hashes = {probes[name]["rng_before"]["sha256"] for name in FORWARD_PROBES}
        after_hashes = {
            "serial-repeat": {
                probes["serial-a"]["rng_after"]["sha256"],
                probes["serial-b"]["rng_after"]["sha256"],
            },
            "overlap-repeat": {
                probes["overlap-a"]["rng_after"]["sha256"],
                probes["overlap-b"]["rng_after"]["sha256"],
            },
            "serial-vs-overlap": {
                probes["serial-a"]["rng_after"]["sha256"],
                probes["overlap-a"]["rng_after"]["sha256"],
            },
            "serial-after-overlap": {
                probes["serial-a"]["rng_after"]["sha256"],
                probes["serial-after-overlap"]["rng_after"]["sha256"],
            },
        }
        rng_invariant = len(before_hashes) == 1 and all(len(hashes) == 1 for hashes in after_hashes.values())
        _record(
            report,
            f"R19_RNG_INVARIANCE {family} rank {rank}",
            rng_invariant,
            {
                "before": sorted(before_hashes),
                "after": {name: sorted(hashes) for name, hashes in after_hashes.items()},
            },
        )
        rank_report["R19_RNG_INVARIANCE"] = rng_invariant
        for label, (reference_name, candidate_name) in comparisons.items():
            reference = probes[reference_name]["selected_logprobs"]
            candidate = probes[candidate_name]["selected_logprobs"]
            metrics = parity._logprob_metrics(
                reference,
                candidate,
                label=f"{label} {family} rank {rank}",
            )
            metrics["bitwise_equal"] = reference == candidate
            passed = (
                metrics["max_sequence_mean_exp_abs_delta"] < parity.LOGPROB_FILTER_LIMIT
                and metrics["relative_l2"] < 0.03
                and metrics["cosine"] > 0.995
            )
            _record(report, f"{label} {family} rank {rank}", passed, metrics)
            rank_report[label] = metrics
            result["floors"][label].append(metrics)
        result["ranks"][str(rank)] = rank_report

    for label, metrics in tuple(result["floors"].items()):
        result["floors"][label] = {
            "max_abs": max(item["max_abs"] for item in metrics),
            "max_relative_l2": max(item["relative_l2"] for item in metrics),
            "min_cosine": min(item["cosine"] for item in metrics),
            "max_sequence_mean_exp_abs_delta": max(
                item["max_sequence_mean_exp_abs_delta"] for item in metrics
            ),
            "all_bitwise_equal": all(item["bitwise_equal"] for item in metrics),
        }
    return result


def _compare(arguments: argparse.Namespace) -> None:
    root = Path(arguments.evidence_dir)
    output = Path(arguments.output)
    if not root.is_absolute() or not root.is_dir():
        raise parity.ParityError(f"evidence directory must be absolute: {root}")
    if not output.is_absolute() or output.parent != root or output.exists() or output.is_symlink():
        raise parity.ParityError("comparison output must be a new absolute direct child of evidence directory")
    expected_runs = FORWARD_RUNS if arguments.mode == "forward-only" else ALL_RUNS
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "base_schema": BASE_SCHEMA,
        "status": "RED",
        "mode": arguments.mode,
        "topology": parity._topology(),
        "thresholds": {
            "logprob_max_sequence_mean_exp_abs_delta": parity.LOGPROB_FILTER_LIMIT,
            "logprob_relative_l2": 0.03,
            "logprob_cosine": 0.995,
            "gradient_norm_relative_l2": 0.05,
            "gradient_norm_cosine": 0.995,
            "gradient_sample_relative_l2": 0.10,
            "gradient_sample_cosine": 0.99,
            "optimizer_update_relative_l2": 0.10,
            "optimizer_update_cosine": 0.99,
        },
        "checks": {},
        "failures": [],
        "same_arm": {},
    }
    runs: dict[str, list[dict[str, Any]]] = {}
    contracts: dict[str, list[dict[str, Any]]] = {}
    for run_id in expected_runs:
        try:
            runs[run_id] = _load_repeat(root, run_id)
            contracts[run_id] = []
            for rank, value in enumerate(runs[run_id]):
                contracts[run_id].append(_validate_repeat(value, run_id=run_id, rank=rank))
                _record(report, f"{run_id} rank {rank} evidence contract", True, contracts[run_id][-1])
        except (parity.ParityError, OSError, ValueError, TypeError, KeyError) as error:
            runs.pop(run_id, None)
            contracts.pop(run_id, None)
            _record(report, f"load and validate {run_id}", False, str(error))
    if len(runs) != len(expected_runs) or len(contracts) != len(expected_runs):
        parity._atomic_json(output, report)
        raise parity.ParityError(f"sealed RED same-arm report at {output}")

    invariant_keys = (
        "batch",
        "config_sha256",
        "topology",
        "repo_root",
        "model_path",
        "requested_harness_seed",
        "effective_mcore_seed",
    )
    leader = runs[expected_runs[0]][0]
    for run_id in expected_runs:
        values = runs[run_id]
        for rank, value in enumerate(values):
            for key in invariant_keys:
                _record(
                    report,
                    f"{run_id} rank {rank} invariant {key}",
                    value.get(key) == leader.get(key),
                    {"leader": leader.get(key), "value": value.get(key)},
                )
            _record(
                report,
                f"{run_id} rank0/rank{rank} batch preparation",
                contracts[run_id][rank]["batch_preparation"] == contracts[run_id][0]["batch_preparation"],
                "batch preparation must be bit-identical across replicated TP/CP ranks",
            )
            if value["phase"] == "forward":
                for probe_name in FORWARD_PROBES:
                    _record(
                        report,
                        f"{run_id} {probe_name} rank0/rank{rank} selected logprobs",
                        value["forward_probes"][probe_name]["selected_logprobs"]
                        == values[0]["forward_probes"][probe_name]["selected_logprobs"],
                        "selected logprobs must be bit-identical after PP broadcast",
                    )
            else:
                _record(
                    report,
                    f"{run_id} rank0/rank{rank} selected pre-logprobs",
                    value["selected_logprobs_before"] == values[0]["selected_logprobs_before"],
                    "selected logprobs must be bit-identical after PP broadcast",
                )
                _record(
                    report,
                    f"{run_id} rank0/rank{rank} selected post-logprobs",
                    value["selected_logprobs_after"] == values[0]["selected_logprobs_after"],
                    "selected post-step logprobs must be bit-identical after PP broadcast",
                )
    for family in ("off", "on"):
        try:
            report["same_arm"][f"{family}-forward"] = _forward_same_worker(
                report,
                runs[f"{family}-forward"],
                family=family,
            )
        except (parity.ParityError, ValueError, TypeError, KeyError, IndexError) as error:
            _record(report, f"{family} same-worker forward comparison", False, str(error))
        if arguments.mode == "forward-plus-one-step":
            a_id = f"{family}-step-a"
            b_id = f"{family}-step-b"
            for rank in range(parity.WORLD_SIZE):
                a_rng = runs[a_id][rank]["rng"]["before_pre_logprob"]["sha256"]
                b_rng = runs[b_id][rank]["rng"]["before_pre_logprob"]["sha256"]
                _record(
                    report,
                    f"{family} step rank {rank} initial RNG repeat identity",
                    a_rng == b_rng,
                    {"a": a_rng, "b": b_rng},
                )
            try:
                report["same_arm"][f"{family}-step"] = _same_arm_pair(
                    report,
                    runs[a_id],
                    runs[b_id],
                    family=family,
                )
            except (parity.ParityError, ValueError, TypeError, KeyError, IndexError) as error:
                _record(report, f"{family} one-step repeat comparison", False, str(error))

    report["status"] = "GREEN" if not report["failures"] else "RED"
    parity._atomic_json(output, report)
    if report["status"] != "GREEN":
        print(
            f"NEMORL_CAPTURED_CITATION_SAME_ARM_RED failures={len(report['failures'])} report={output}",
            flush=True,
        )
        raise parity.ParityError(f"sealed RED same-arm report at {output}")
    floors = {
        key: (
            value["floors"]
            if key.endswith("-forward")
            else value["selected_logprobs_before_floor"]
        )
        for key, value in report["same_arm"].items()
    }
    print(
        "NEMORL_CAPTURED_CITATION_SAME_ARM_GREEN "
        f"mode={arguments.mode} batch_sha256={leader['batch']['sha256']} "
        f"floors={json.dumps(floors, sort_keys=True, separators=(',', ':'))} report={output}",
        flush=True,
    )


def _self_test_compare() -> None:
    mounted_sentinel = object()
    local_sentinel = object()
    import_calls: list[str] = []

    def namespace_without_local_module(name: str) -> Any:
        import_calls.append(name)
        if name == "shared_prefix_captured_citation_parity":
            return mounted_sentinel
        # This is the exact Deployment-Q hazard: the package namespace exists,
        # but it cannot provide the requested parity module.
        raise ImportError("cannot import name 'shared_prefix_captured_citation_parity'")

    if _import_parity_module(namespace_without_local_module) is not mounted_sentinel or import_calls != [
        "shared_prefix_captured_citation_parity"
    ]:
        raise parity.ParityError("mounted top-level parity import was not preferred")

    def local_fallback(name: str) -> Any:
        if name == "shared_prefix_captured_citation_parity":
            raise ModuleNotFoundError("top-level parity is absent", name=name)
        if name == "tools.model_diagnostics.shared_prefix_captured_citation_parity":
            return local_sentinel
        raise AssertionError(name)

    if _import_parity_module(local_fallback) is not local_sentinel:
        raise parity.ParityError("repository parity fallback did not resolve")

    def dependency_failure(name: str) -> Any:
        if name == "shared_prefix_captured_citation_parity":
            raise ModuleNotFoundError("mounted parity dependency is absent", name="ray")
        raise AssertionError("dependency failure must not fall back to a different harness")

    try:
        _import_parity_module(dependency_failure)
    except ModuleNotFoundError as error:
        if error.name != "ray":
            raise parity.ParityError("wrong dependency failure propagated") from error
    else:
        raise parity.ParityError("mounted parity dependency failure was concealed")

    exact = [[-1.0, -2.0], [-3.0], [-4.0], [-5.0]]
    metrics = parity._logprob_metrics(exact, exact, label="same-arm exact self-test")
    if metrics["max_abs"] != 0.0 or metrics["relative_l2"] != 0.0 or metrics["cosine"] != 1.0:
        raise parity.ParityError("same-arm comparator exact self-test failed")
    changed = [[-1.1, -2.0], [-3.0], [-4.0], [-5.0]]
    changed_metrics = parity._logprob_metrics(exact, changed, label="same-arm changed self-test")
    if not changed_metrics["max_abs"] > 0.0:
        raise parity.ParityError("same-arm comparator failed to detect a changed vector")
    for run_id in ALL_RUNS:
        family, repeat, phase = _run_spec(run_id)
        if run_id != f"{family}{'-step' if phase == 'step' else ''}-{repeat}":
            raise parity.ParityError(f"same-arm run-id self-test failed for {run_id}")
    # Exercise the complete one-step comparator control flow with lightweight
    # validators. In particular, pin the R18 helpers' required ``arm`` identity
    # so an arm_family-only evidence schema cannot silently regress.
    step_evidence = {
        "arm": "off",
        "selected_logprobs_before": exact,
        "selected_logprobs_after": exact,
        "training": {"global_loss": 1.0},
    }
    first = [dict(step_evidence) for _ in range(parity.WORLD_SIZE)]
    second = [dict(step_evidence) for _ in range(parity.WORLD_SIZE)]
    original_gradient_vectors = parity._family_gradient_vectors
    original_aggregate = parity._aggregate_family_optimizer_updates

    def fake_gradient_vectors(evidence: dict[str, Any], _family: str) -> tuple[list[float], list[float]]:
        if evidence.get("arm") != "off":
            raise parity.ParityError("one-step self-test lost R18 arm compatibility identity")
        return [1.0], [1.0]

    def fake_aggregate(
        a_values: list[dict[str, Any]],
        b_values: list[dict[str, Any]],
        _family: str,
    ) -> dict[str, Any]:
        if any(item.get("arm") != "off" for item in a_values + b_values):
            raise parity.ParityError("one-step aggregate lost R18 arm compatibility identity")
        return {
            "initial_exact_match": True,
            "update_projection": {"relative_l2": 0.0, "cosine": 1.0},
        }

    try:
        parity._family_gradient_vectors = fake_gradient_vectors
        parity._aggregate_family_optimizer_updates = fake_aggregate
        step_report: dict[str, Any] = {"checks": {}, "failures": []}
        result = _same_arm_pair(step_report, first, second, family="off")
        if step_report["failures"] or set(result["optimizer_families"]) != set(
            parity.REQUIRED_GRADIENT_FAMILIES
        ):
            raise parity.ParityError("one-step comparator control-flow self-test failed")
    finally:
        parity._family_gradient_vectors = original_gradient_vectors
        parity._aggregate_family_optimizer_updates = original_aggregate
    print(
        "CAPTURED_CITATION_SAME_ARM_SELF_TEST_GREEN "
        f"runs={len(ALL_RUNS)} changed_max_abs={changed_metrics['max_abs']:.9g} "
        "import_resolution=green step_comparator=green",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-repeat")
    run.add_argument("--run-id", choices=ALL_RUNS, required=True)
    run.add_argument("--repo-root", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--batch", required=True)
    run.add_argument("--expected-batch-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--evidence-dir", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--mode", choices=("forward-only", "forward-plus-one-step"), required=True)
    subparsers.add_parser("self-test-compare")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run-repeat":
            _run_repeat(arguments)
        elif arguments.command == "compare":
            _compare(arguments)
        else:
            _self_test_compare()
    except parity.ParityError as error:
        print(f"CAPTURED_CITATION_SAME_ARM_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
