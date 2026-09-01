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

"""Fail-closed layer and routing isolation for captured citation parity.

This v4 diagnostic composes the sealed v3 captured-citation harness instead of
changing its evidence contract.  Every public arm retains its original v3 JSON
and adds one compact diagnostic sidecar with:

* row-semantic hidden-state projections after every Hybrid backbone layer;
* exact per-token top-k expert sets for every decoder MoE router;
* distributed-optimizer shard intervals, including TP/CP rank coordinates; and
* explicit attestation that the v3 TE state-initialization workaround is not a
  native optimizer-readiness gate.

The fourth, forward-only ``on-reference-cp-replay`` arm keeps Mamba replay and
replaces CP forest attention with an intentionally expensive gathered reference:
Q/K/V are all-gathered in sequence space, canonicalized from CP zigzag order,
evaluated by the non-CP exact forest primitive, and sliced back to the caller's
CP shard.  It never runs backward or an optimizer step.

``compare`` first preserves the complete v3 comparison, then fails closed on
layer drift, any router-ID difference, replay/reference logprob thresholds,
cross-rank optimizer interval gaps or overlaps, changed-parameter-set mismatch,
and gradient/update parity for every trainable family (including
``embedding_output`` and ``other``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, cast

import torch


def _load_paired_module(
    sibling: Path,
    *,
    package_name: str,
    module_name: str,
) -> Any:
    """Prefer an immutable mounted sibling, falling back to a local package."""
    if sibling.is_symlink():
        raise RuntimeError(f"paired harness must not be a symlink: {sibling}")
    if sibling.exists():
        if not sibling.is_file():
            raise RuntimeError(f"paired harness is not a regular file: {sibling}")
        specification = importlib.util.spec_from_file_location(module_name, sibling)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot load paired harness from {sibling}")
        module = importlib.util.module_from_spec(specification)
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except Exception:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            raise
        return module
    return importlib.import_module(package_name)


_BASE_PATH = Path(__file__).with_name("shared_prefix_captured_citation_parity.py")
v3 = _load_paired_module(
    _BASE_PATH,
    package_name="tools.model_diagnostics.shared_prefix_captured_citation_parity",
    module_name="_nemorl_shared_prefix_captured_citation_parity",
)


SCHEMA = "nemorl-shared-prefix-captured-citation-drift-v4"
PUBLIC_ARMS = ("off", "on", "on-mamba-replay", "on-reference-cp-replay")
FORWARD_ONLY_ARMS = ("on-mamba-replay", "on-reference-cp-replay")
LAYER_RELATIVE_L2_LIMIT = 0.03
LAYER_COSINE_LIMIT = 0.995
ROUTER_MISMATCH_LIMIT = 0
FEATURE_DIMENSIONS = 4
_DECODER_LAYER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)$")
_DECODER_ROUTER_RE = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.mlp\.router$")


class DriftError(v3.ParityError):
    """The isolation evidence or its distributed reconstruction is invalid."""


def _all_gather_cat(tensor: torch.Tensor, *, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """All-gather one detached sequence shard and concatenate in group-rank order."""
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return tensor
    outputs = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(outputs, tensor.contiguous(), group=group)
    return torch.cat(outputs, dim=0)


def _undo_cp_zigzag(rank_major: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert rank-major CP zigzag shards into canonical sequence order."""
    if cp_size < 1 or rank_major.shape[0] % (2 * cp_size):
        raise DriftError("CP rank-major length must be divisible by 2 * CP size")
    chunks = torch.chunk(rank_major, chunks=2 * cp_size, dim=0)
    order = [2 * index for index in range(cp_size)] + [2 * cp_size - 2 * index - 1 for index in range(cp_size)]
    return torch.cat([chunks[index] for index in order], dim=0)


def _redo_cp_zigzag(canonical: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Convert canonical sequence order into rank-major CP zigzag shards."""
    if cp_size < 1 or canonical.shape[0] % (2 * cp_size):
        raise DriftError("CP canonical length must be divisible by 2 * CP size")
    chunks = torch.chunk(canonical, chunks=2 * cp_size, dim=0)
    order: list[int | None] = [None] * (2 * cp_size)
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, 2 * cp_size))
    if any(index is None for index in order):
        raise DriftError("failed to construct inverse CP zigzag permutation")
    return torch.cat([chunks[cast(int, index)] for index in order], dim=0)


def _reference_cp_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    forest: Any,
    *,
    cp_group: torch.distributed.ProcessGroup,
    scale: float | None = None,
) -> torch.Tensor:
    """Gathered forward-only reference for CP forest attention."""
    if torch.is_grad_enabled() or any(tensor.requires_grad for tensor in (query, key, value)):
        raise DriftError("reference CP attention is forward-only and forbids autograd")
    if any(tensor.ndim != 4 or tensor.shape[1] != 1 for tensor in (query, key, value)):
        raise DriftError("reference CP attention requires Q/K/V shape [S/C,1,H,D]")
    if not query.shape[0] == key.shape[0] == value.shape[0]:
        raise DriftError("reference CP attention requires aligned local sequences")
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    if cp_size == 1:
        # Heavy dependency stays local to the GPU-only diagnostic path.
        from megatron.core.models.hybrid.shared_prefix_fused import flash_composed_forest_attention

        output = flash_composed_forest_attention(query, key, value, forest, scale=scale)
        return output.reshape(output.shape[0], 1, -1).contiguous()

    canonical_inputs = [
        _undo_cp_zigzag(_all_gather_cat(tensor, group=cp_group), cp_size) for tensor in (query, key, value)
    ]
    # Heavy dependency stays local to the GPU-only diagnostic path.
    from megatron.core.models.hybrid.shared_prefix_fused import flash_composed_forest_attention

    output = flash_composed_forest_attention(*canonical_inputs, forest, scale=scale)
    rank_major = _redo_cp_zigzag(output, cp_size)
    local = torch.chunk(rank_major, cp_size, dim=0)[cp_rank]
    if local.shape[0] != query.shape[0] or local.shape[1] != 1:
        raise DriftError("reference CP attention returned an invalid local output shape")
    return local.reshape(local.shape[0], 1, -1).contiguous()


class _SemanticPlan:
    """Map one dense pack or shared star back to the four captured rows."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        padding_multiple: int,
        dense_order: Sequence[int],
    ) -> None:
        self.input_lengths = tuple(int(row["input_length"]) for row in rows)
        self.prompt_length = int(rows[0]["prompt_length"])
        self.padding_multiple = int(padding_multiple)
        self.dense_order = tuple(int(index) for index in dense_order)
        if sorted(self.dense_order) != list(range(v3.BATCH_SIZE)):
            raise DriftError(f"dense source order is not K=4: {self.dense_order!r}")
        self.padded_lengths = tuple(v3._round_up(length, self.padding_multiple) for length in self.input_lengths)
        self.shared_order = tuple(
            sorted(
                range(v3.BATCH_SIZE),
                key=lambda row: (
                    -(self.padded_lengths[row] - self.prompt_length),
                    -(self.input_lengths[row] - self.prompt_length),
                    row,
                ),
            )
        )

    def rows_from_physical(self, physical: torch.Tensor, *, shared: bool) -> list[torch.Tensor]:
        """Return one unpadded semantic tensor per original captured row."""
        if physical.ndim < 2:
            raise DriftError(f"physical semantic tensor must have rank >=2, got {physical.shape}")
        semantic: list[torch.Tensor | None] = [None] * v3.BATCH_SIZE
        if shared:
            if physical.shape[0] < self.prompt_length:
                raise DriftError("shared physical tensor is shorter than its prompt")
            prompt = physical[: self.prompt_length]
            cursor = self.prompt_length
            for row in self.shared_order:
                physical_completion = self.padded_lengths[row] - self.prompt_length
                logical_completion = self.input_lengths[row] - self.prompt_length
                stop = cursor + physical_completion
                if stop > physical.shape[0]:
                    raise DriftError("shared physical tensor ends inside a completion")
                semantic[row] = torch.cat((prompt, physical[cursor : cursor + logical_completion]), dim=0)
                cursor = stop
            if physical.shape[0] - cursor >= self.padding_multiple:
                raise DriftError("shared physical tensor has at least one full padding quantum after its star")
        else:
            cursor = 0
            for row in self.dense_order:
                stop = cursor + self.padded_lengths[row]
                if stop > physical.shape[0]:
                    raise DriftError("dense physical tensor ends inside a packed row")
                semantic[row] = physical[cursor : cursor + self.input_lengths[row]]
                cursor = stop
            if cursor != physical.shape[0]:
                raise DriftError(f"dense physical length differs from its exact pack: {cursor}/{physical.shape[0]}")
        if any(value is None for value in semantic):
            raise DriftError("semantic mapping did not reconstruct every captured row")
        result = [cast(torch.Tensor, value) for value in semantic]
        for row, tensor in enumerate(result):
            if tensor.shape[0] != self.input_lengths[row]:
                raise DriftError(f"semantic row {row} has invalid length {tensor.shape[0]}")
        return result


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(v3._canonical_json_bytes(value)).hexdigest()


def _semantic_hidden_evidence(rows: Sequence[torch.Tensor], *, include_features: bool) -> dict[str, Any]:
    row_evidence: list[dict[str, Any]] = []
    semantic_hasher = hashlib.sha256()
    for row, tensor in enumerate(rows):
        if tensor.ndim != 2 or tensor.shape[1] <= 0:
            raise DriftError(f"semantic hidden row {row} must have shape [tokens, hidden]")
        values = tensor.detach()
        if not torch.isfinite(values).all():
            raise DriftError(f"semantic hidden row {row} contains non-finite values")
        tensor_sha256 = v3._tensor_sha256(values)
        semantic_hasher.update(
            v3._canonical_json_bytes(
                {"row": row, "shape": list(values.shape), "dtype": str(values.dtype), "sha256": tensor_sha256}
            )
        )
        item: dict[str, Any] = {
            "row": row,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": tensor_sha256,
        }
        if include_features:
            width = values.shape[1]
            count = min(FEATURE_DIMENSIONS, width)
            indices = torch.linspace(0, width - 1, count, dtype=torch.float64, device=values.device).to(torch.long)
            fp32 = values.float()
            features = torch.cat(
                (
                    fp32.mean(dim=1, keepdim=True),
                    torch.linalg.vector_norm(fp32, ord=2, dim=1, keepdim=True) / math.sqrt(width),
                    fp32.index_select(1, indices),
                ),
                dim=1,
            )
            item["feature_columns"] = ["mean", "rms", *[f"hidden_{int(index)}" for index in indices.cpu()]]
            item["features"] = features.cpu().tolist()
        else:
            item["feature_columns"] = None
            item["features"] = None
        row_evidence.append(item)
    return {"semantic_sha256": semantic_hasher.hexdigest(), "rows": row_evidence}


def _semantic_router_evidence(rows: Sequence[torch.Tensor], *, include_indices: bool) -> dict[str, Any]:
    serialized: list[list[list[int]]] = []
    topk: int | None = None
    for row, tensor in enumerate(rows):
        if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
            raise DriftError(f"semantic router row {row} has invalid shape {tensor.shape}")
        if tensor.dtype not in (torch.int32, torch.int64):
            raise DriftError(f"semantic router row {row} must contain integer expert IDs")
        if topk is None:
            topk = tensor.shape[1]
        elif tensor.shape[1] != topk:
            raise DriftError("semantic router top-k width changed across captured rows")
        serialized.append([[int(value) for value in token] for token in tensor.cpu().tolist()])
    return {
        "topk": topk,
        "semantic_sha256": _canonical_json_sha256(serialized),
        "rows": serialized if include_indices else None,
    }


class _Collector:
    """Capture the first complete, forward-only backbone pass in one process."""

    def __init__(self, *, arm: str, plan: _SemanticPlan, rank: int) -> None:
        self.arm = arm
        self.plan = plan
        self.rank = rank
        self.shared = arm != "off"
        self.layer_modules: dict[int, torch.nn.Module] = {}
        self.layer_by_identity: dict[int, int] = {}
        self.router_layers: set[int] = set()
        self.hidden: dict[int, dict[str, Any]] = {}
        self.routing: dict[int, dict[str, Any]] = {}
        self.intervals: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.tp_group: torch.distributed.ProcessGroup | None = None
        self.cp_group: torch.distributed.ProcessGroup | None = None
        self.sequence_parallel = False
        self._mamba_original: Callable[..., Any] | None = None

    def install(self, worker: Any) -> None:
        """Install hooks and snapshot local optimizer shard intervals."""
        from megatron.core.utils import unwrap_model
        from megatron.core.transformer.moe.router import TopKRouter

        chunks = unwrap_model(worker.model)
        chunks = chunks if isinstance(chunks, (list, tuple)) else [chunks]
        if len(chunks) != 1:
            raise DriftError(f"PP1 diagnostic expected one model chunk, got {len(chunks)}")
        chunk = chunks[0]
        decoder = chunk.decoder
        self.tp_group = decoder.pg_collection.tp
        self.cp_group = decoder.pg_collection.cp
        self.sequence_parallel = bool(decoder.config.sequence_parallel)

        for name, module in chunk.named_modules():
            layer_match = _DECODER_LAYER_RE.search(name)
            if layer_match is not None:
                layer = int(layer_match.group(1))
                if layer in self.layer_modules:
                    raise DriftError(f"decoder layer {layer} appears more than once")
                self.layer_modules[layer] = module
                self.layer_by_identity[id(module)] = layer
                self.handles.append(module.register_forward_hook(self._layer_hook(layer)))
            router_match = _DECODER_ROUTER_RE.search(name)
            if router_match is not None:
                if not isinstance(module, TopKRouter):
                    raise DriftError(f"decoder router {name} is not TopKRouter")
                layer = int(router_match.group(1))
                self.router_layers.add(layer)
                self.handles.append(module.register_forward_hook(self._router_hook(layer)))
        if not self.layer_modules or not self.router_layers:
            raise DriftError("diagnostic found no decoder layers or MoE routers")
        self._snapshot_intervals(worker)

    def _snapshot_intervals(self, worker: Any) -> None:
        if self.tp_group is None or self.cp_group is None:
            raise DriftError("parallel groups are unavailable while snapshotting optimizer intervals")
        names = {id(parameter): name for name, parameter in v3._named_parameters(worker.model)}
        world_rank = torch.distributed.get_rank()
        tp_rank = torch.distributed.get_rank(self.tp_group)
        cp_rank = torch.distributed.get_rank(self.cp_group)
        seen: set[tuple[int, str]] = set()
        for part_index, part in enumerate(v3._optimizer_parts(worker.optimizer)):
            data_parallel_group = getattr(part, "data_parallel_group", None)
            if data_parallel_group is None:
                raise DriftError(f"optimizer part {part_index} has no data-parallel group")
            optimizer_group_ranks = torch.distributed.get_process_group_ranks(data_parallel_group)
            optimizer_group_rank = torch.distributed.get_rank(data_parallel_group)
            if (
                not optimizer_group_ranks
                or not 0 <= optimizer_group_rank < len(optimizer_group_ranks)
                or optimizer_group_ranks[optimizer_group_rank] != world_rank
            ):
                raise DriftError(f"optimizer part {part_index} has inconsistent data-parallel ranks")
            parameter_map = getattr(part, "model_param_group_index_map", None)
            range_accessor = getattr(part, "_get_model_param_range_map", None)
            if not isinstance(parameter_map, dict) or not callable(range_accessor):
                raise DriftError(f"optimizer part {part_index} lacks local shard interval metadata")
            for parameter in parameter_map:
                name = names.get(id(parameter))
                if name is None:
                    raise DriftError("optimizer interval references an unnamed model parameter")
                range_map = range_accessor(parameter)
                model_range = range_map.get("param") if isinstance(range_map, dict) else None
                start = getattr(model_range, "start", None)
                end = getattr(model_range, "end", None)
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= parameter.numel()
                ):
                    raise DriftError(f"optimizer interval for {name} is invalid: {start}/{end}")
                identity = (part_index, name)
                if identity in seen:
                    raise DriftError(f"optimizer interval for {name} occurs twice in part {part_index}")
                seen.add(identity)
                self.intervals.append(
                    {
                        "rank": world_rank,
                        "tp_rank": tp_rank,
                        "cp_rank": cp_rank,
                        "optimizer_group_rank": optimizer_group_rank,
                        "optimizer_group_ranks": optimizer_group_ranks,
                        "part": part_index,
                        "name": name,
                        "family": v3._family(name),
                        "start": start,
                        "end": end,
                        "model_numel": parameter.numel(),
                    }
                )
        if not self.intervals:
            raise DriftError("optimizer exposes no local shard intervals")

    def _canonical_global(self, local: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None or self.cp_group is None:
            raise DriftError("parallel groups are unavailable during semantic capture")
        value = local.detach()
        if value.ndim == 3 and value.shape[1] == 1:
            pass
        elif value.ndim == 2:
            pass
        else:
            raise DriftError(f"layer/router local tensor has unsupported shape {value.shape}")
        if self.sequence_parallel and torch.distributed.get_world_size(self.tp_group) > 1:
            value = _all_gather_cat(value, group=self.tp_group)
        cp_size = torch.distributed.get_world_size(self.cp_group)
        if cp_size > 1:
            value = _undo_cp_zigzag(_all_gather_cat(value, group=self.cp_group), cp_size)
        return value

    def _capture_hidden(self, layer: int, output: Any) -> None:
        if layer in self.hidden:
            return
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise DriftError(f"decoder layer {layer} did not return a tensor")
        canonical = self._canonical_global(tensor).squeeze(1)
        semantic = self.plan.rows_from_physical(canonical, shared=self.shared)
        self.hidden[layer] = _semantic_hidden_evidence(semantic, include_features=self.rank == 0)

    def _layer_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self._capture_hidden(layer, output)

        return hook

    def _router_hook(self, layer: int) -> Callable[..., None]:
        def hook(module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer in self.routing:
                return
            if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], torch.Tensor):
                raise DriftError(f"decoder router {layer} returned an invalid routing payload")
            raw = output[1]
            if raw.dtype == torch.bool:
                topk = getattr(module, "topk", None)
                if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
                    raise DriftError(f"decoder router {layer} exposes invalid top-k {topk!r}")
                num_tokens = raw.shape[0]
                token_index, expert_index = raw.nonzero(as_tuple=True)
                counts = torch.bincount(token_index, minlength=num_tokens)
                if expert_index.numel() != num_tokens * topk or not torch.all(counts == topk):
                    raise DriftError(f"decoder router {layer} routing map is not exact top-k")
                indices = expert_index.view(num_tokens, topk).to(torch.int32)
            elif raw.dtype in (torch.int32, torch.int64) and raw.ndim == 2:
                indices = raw.to(torch.int32)
            else:
                raise DriftError(f"decoder router {layer} returned unsupported dtype/shape {raw.dtype}/{raw.shape}")
            canonical = self._canonical_global(indices)
            semantic = self.plan.rows_from_physical(canonical, shared=self.shared)
            self.routing[layer] = _semantic_router_evidence(semantic, include_indices=self.rank == 0)

        return hook

    def install_mamba_helper(self) -> None:
        """Wrap the currently selected production or replay CP Mamba helper."""
        if self._mamba_original is not None:
            raise DriftError("Mamba helper was instrumented more than once")
        from megatron.core.models.hybrid import shared_prefix

        original = shared_prefix._forward_mamba_layer_shared_prefix_cp
        self._mamba_original = original

        @wraps(original)
        def counted(layer_module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
            output = original(layer_module, *args, **kwargs)
            layer = self.layer_by_identity.get(id(layer_module))
            if layer is None:
                raise DriftError("shared CP Mamba helper received an unknown decoder layer")
            self._capture_hidden(layer, output)
            return output

        shared_prefix._forward_mamba_layer_shared_prefix_cp = counted

    def restore(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self._mamba_original is not None:
            from megatron.core.models.hybrid import shared_prefix

            shared_prefix._forward_mamba_layer_shared_prefix_cp = self._mamba_original
            self._mamba_original = None

    def evidence(self) -> dict[str, Any]:
        expected_layers = set(self.layer_modules)
        if set(self.hidden) != expected_layers:
            raise DriftError(
                f"hidden capture coverage differs: missing={sorted(expected_layers - set(self.hidden))} "
                f"extra={sorted(set(self.hidden) - expected_layers)}"
            )
        if set(self.routing) != self.router_layers:
            raise DriftError(
                f"router capture coverage differs: missing={sorted(self.router_layers - set(self.routing))} "
                f"extra={sorted(set(self.routing) - self.router_layers)}"
            )
        return {
            "layers": {str(layer): self.hidden[layer] for layer in sorted(self.hidden)},
            "routers": {str(layer): self.routing[layer] for layer in sorted(self.routing)},
            "optimizer_intervals": sorted(
                self.intervals,
                key=lambda item: (item["part"], item["name"], item["start"], item["rank"]),
            ),
        }


def _policy_and_plan(arguments: argparse.Namespace, *, arm: str) -> tuple[list[dict[str, Any]], _SemanticPlan]:
    rows, _summary = v3._read_captured_rows(Path(arguments.batch), expected_sha256=arguments.expected_batch_sha256)
    mode = "observe" if arm == "off" else "train"
    policy, _loss, _fingerprint = v3._build_policy_and_loss_config(
        arguments.model_path,
        repo_root=Path(arguments.repo_root),
        shared_prefix_mode=mode,
    )
    dense_policy, _dense_loss, _dense_fingerprint = v3._build_policy_and_loss_config(
        arguments.model_path,
        repo_root=Path(arguments.repo_root),
        shared_prefix_mode="observe",
    )
    _prepared, dense_order = v3._prepare_batch_for_worker(
        v3._build_batch(rows),
        policy_config=dense_policy,
        shared_prefix_mode="observe",
        stage="logprob",
    )
    if dense_order is None:
        raise DriftError("dense logprob preparation did not return a source order")
    return rows, _SemanticPlan(
        rows,
        padding_multiple=int(policy["make_sequence_length_divisible_by"]),
        dense_order=dense_order,
    )


def _run_arm(arguments: argparse.Namespace) -> None:
    public_arm = arguments.arm
    if public_arm not in PUBLIC_ARMS:
        raise DriftError(f"unsupported v4 arm {public_arm!r}")
    rank = int(os.environ["RANK"])
    _rows, plan = _policy_and_plan(arguments, arm=public_arm)
    underlying_arm = "on-mamba-replay" if public_arm == "on-reference-cp-replay" else public_arm
    output_root = Path(arguments.output_dir)
    base_output = output_root if public_arm != "on-reference-cp-replay" else output_root / "reference_base"
    base_output.mkdir(parents=True, exist_ok=True)

    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    original_init = MegatronPolicyWorkerImpl.__init__
    original_counter_installer = v3._install_shared_prefix_runtime_counters
    collector: _Collector | None = None
    reference_calls = 0

    def worker_init(worker: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal collector
        original_init(worker, *args, **kwargs)
        if collector is not None:
            raise DriftError("diagnostic constructed more than one policy worker")
        collector = _Collector(arm=public_arm, plan=plan, rank=rank)
        collector.install(worker)

    def install_counters() -> tuple[dict[str, int], dict[str, str], Callable[[], None]]:
        if collector is None:
            raise DriftError("runtime counters were installed before the policy worker")
        collector.install_mamba_helper()
        return original_counter_installer()

    original_attention: Callable[..., Any] | None = None
    if public_arm == "on-reference-cp-replay":
        from megatron.core.models.hybrid import shared_prefix_fused

        original_attention = shared_prefix_fused.flash_composed_forest_attention_cp

        @wraps(_reference_cp_attention)
        def reference_attention(*args: Any, **kwargs: Any) -> torch.Tensor:
            nonlocal reference_calls
            reference_calls += 1
            return _reference_cp_attention(*args, **kwargs)

        shared_prefix_fused.flash_composed_forest_attention_cp = reference_attention

    MegatronPolicyWorkerImpl.__init__ = worker_init
    v3._install_shared_prefix_runtime_counters = install_counters
    base_arguments = copy.copy(arguments)
    base_arguments.arm = underlying_arm
    base_arguments.output_dir = str(base_output)
    try:
        v3._run_arm(base_arguments)
    finally:
        MegatronPolicyWorkerImpl.__init__ = original_init
        v3._install_shared_prefix_runtime_counters = original_counter_installer
        if original_attention is not None:
            from megatron.core.models.hybrid import shared_prefix_fused

            shared_prefix_fused.flash_composed_forest_attention_cp = original_attention
        if collector is not None:
            collector.restore()
    if collector is None:
        raise DriftError("policy worker construction did not install the diagnostic collector")
    if public_arm == "on-reference-cp-replay" and reference_calls <= 0:
        raise DriftError("reference CP attention arm did not call its gathered implementation")

    base_path = base_output / f"{underlying_arm}.rank{rank}.json"
    with base_path.open(encoding="utf-8", errors="strict") as stream:
        base_evidence = json.load(
            stream,
            object_pairs_hook=v3._no_duplicate_keys,
            parse_constant=v3._reject_constant,
        )
    diagnostics = collector.evidence()
    evidence = {
        "schema": SCHEMA,
        "arm": public_arm,
        "rank": rank,
        "base_schema": base_evidence.get("schema"),
        "base_arm": underlying_arm,
        "batch_sha256": base_evidence.get("batch", {}).get("sha256"),
        "config_sha256": base_evidence.get("config_sha256"),
        "topology": base_evidence.get("topology"),
        "forward_only": public_arm in FORWARD_ONLY_ARMS,
        "reference_cp_attention": public_arm == "on-reference-cp-replay",
        "reference_cp_attention_calls": reference_calls,
        "te_workaround_attested": public_arm not in FORWARD_ONLY_ARMS,
        "native_optimizer_gate": "separate-required",
        **diagnostics,
    }
    v3._atomic_json(output_root / f"diagnostic.{public_arm}.rank{rank}.json", evidence)
    if rank == 0:
        print(
            "NEMORL_CAPTURED_CITATION_DRIFT_ARM_GREEN "
            f"arm={public_arm} layers={len(diagnostics['layers'])} "
            f"routers={len(diagnostics['routers'])} reference_calls={reference_calls}",
            flush=True,
        )


def _load_diagnostics(root: Path, arm: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    expected_keys = {
        "schema",
        "arm",
        "rank",
        "base_schema",
        "base_arm",
        "batch_sha256",
        "config_sha256",
        "topology",
        "forward_only",
        "reference_cp_attention",
        "reference_cp_attention_calls",
        "te_workaround_attested",
        "native_optimizer_gate",
        "layers",
        "routers",
        "optimizer_intervals",
    }
    for rank in range(v3.WORLD_SIZE):
        path = root / f"diagnostic.{arm}.rank{rank}.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
            raise DriftError(f"missing, symlinked, or writable diagnostic evidence: {path}")
        with path.open(encoding="utf-8", errors="strict") as stream:
            value = json.load(
                stream,
                object_pairs_hook=v3._no_duplicate_keys,
                parse_constant=v3._reject_constant,
            )
        if (
            not isinstance(value, dict)
            or value.get("schema") != SCHEMA
            or value.get("arm") != arm
            or value.get("rank") != rank
        ):
            raise DriftError(f"diagnostic identity differs in {path}")
        v3._exact_dict(value, keys=expected_keys, label=f"{arm} rank {rank} v4 diagnostic")
        evidence.append(value)
    return evidence


def _features(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise DriftError(f"{label} has invalid semantic hidden rows")
    result: list[float] = []
    for row_index, row in enumerate(value["rows"]):
        if not isinstance(row, dict) or not isinstance(row.get("features"), list):
            raise DriftError(f"{label} row {row_index} lacks rank-zero hidden features")
        columns = row.get("feature_columns")
        if not isinstance(columns, list) or len(columns) != FEATURE_DIMENSIONS + 2:
            raise DriftError(f"{label} row {row_index} has invalid feature columns")
        for token_index, token in enumerate(row["features"]):
            values = v3._number_list(token, label=f"{label} row {row_index} token {token_index}")
            if len(values) != len(columns):
                raise DriftError(f"{label} row {row_index} token {token_index} feature width differs")
            result.extend(values)
    if not result:
        raise DriftError(f"{label} has no hidden features")
    return result


def _router_mismatches(reference: Any, candidate: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise DriftError(f"{label} router evidence must be dictionaries")
    reference_rows = reference.get("rows")
    candidate_rows = candidate.get("rows")
    if not isinstance(reference_rows, list) or not isinstance(candidate_rows, list):
        raise DriftError(f"{label} router evidence lacks rank-zero rows")
    if len(reference_rows) != v3.BATCH_SIZE or len(candidate_rows) != v3.BATCH_SIZE:
        raise DriftError(f"{label} router evidence must have K=4 rows")
    mismatches = 0
    first: dict[str, Any] | None = None
    tokens = 0
    for row, (reference_tokens, candidate_tokens) in enumerate(zip(reference_rows, candidate_rows, strict=True)):
        if not isinstance(reference_tokens, list) or not isinstance(candidate_tokens, list):
            raise DriftError(f"{label} row {row} router tokens are invalid")
        if len(reference_tokens) != len(candidate_tokens):
            raise DriftError(f"{label} row {row} router token counts differ")
        for token, (expected, actual) in enumerate(zip(reference_tokens, candidate_tokens, strict=True)):
            expected_ids = v3._int_list(expected, label=f"{label} expected row {row} token {token}")
            actual_ids = v3._int_list(actual, label=f"{label} actual row {row} token {token}")
            tokens += 1
            if expected_ids != actual_ids:
                mismatches += 1
                if first is None:
                    first = {"row": row, "token": token, "off": expected_ids, "candidate": actual_ids}
    return {"tokens": tokens, "mismatches": mismatches, "first_mismatch": first}


def _validate_interval_coverage(values: Sequence[dict[str, Any]], *, arm: str) -> dict[str, Any]:
    groups: dict[tuple[int, str, tuple[int, ...], int], list[tuple[int, int, int, int]]] = {}
    for rank, evidence in enumerate(values):
        intervals = evidence.get("optimizer_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise DriftError(f"{arm} rank {rank} has no optimizer intervals")
        for item in intervals:
            if not isinstance(item, dict):
                raise DriftError(f"{arm} rank {rank} optimizer interval is not an object")
            required = {
                "rank",
                "tp_rank",
                "cp_rank",
                "optimizer_group_rank",
                "optimizer_group_ranks",
                "part",
                "name",
                "family",
                "start",
                "end",
                "model_numel",
            }
            if set(item) != required:
                raise DriftError(f"{arm} rank {rank} optimizer interval schema differs")
            if item["rank"] != rank or not isinstance(item["name"], str) or not item["name"]:
                raise DriftError(f"{arm} rank {rank} optimizer interval identity differs")
            integers = {
                key: v3._int(item[key], label=f"{arm} rank {rank} interval {key}")
                for key in (
                    "tp_rank",
                    "cp_rank",
                    "optimizer_group_rank",
                    "part",
                    "start",
                    "end",
                    "model_numel",
                )
            }
            raw_optimizer_group_ranks = item["optimizer_group_ranks"]
            if not isinstance(raw_optimizer_group_ranks, list):
                raise DriftError(f"{arm} rank {rank} optimizer group ranks are not a list")
            optimizer_group_ranks = tuple(
                v3._int(value, label=f"{arm} rank {rank} optimizer group rank") for value in raw_optimizer_group_ranks
            )
            optimizer_group_rank = integers["optimizer_group_rank"]
            if (
                not optimizer_group_ranks
                or len(set(optimizer_group_ranks)) != len(optimizer_group_ranks)
                or not 0 <= optimizer_group_rank < len(optimizer_group_ranks)
                or optimizer_group_ranks[optimizer_group_rank] != rank
            ):
                raise DriftError(f"{arm} rank {rank} optimizer group membership differs")
            if not 0 <= integers["start"] < integers["end"] <= integers["model_numel"]:
                raise DriftError(f"{arm} rank {rank} optimizer interval bounds differ")
            key = (
                integers["part"],
                item["name"],
                optimizer_group_ranks,
                integers["model_numel"],
            )
            groups.setdefault(key, []).append((integers["start"], integers["end"], optimizer_group_rank, rank))
    normalized: dict[str, Any] = {}
    for (part, name, optimizer_group_ranks, model_numel), intervals in sorted(groups.items()):
        ordered = sorted(intervals)
        cursor = 0
        seen_group_ranks: set[int] = set()
        for start, end, optimizer_group_rank, rank in ordered:
            if optimizer_group_rank in seen_group_ranks:
                raise DriftError(
                    f"{arm} {name} optimizer group {optimizer_group_ranks} has two intervals "
                    f"from group rank {optimizer_group_rank}"
                )
            seen_group_ranks.add(optimizer_group_rank)
            if start != cursor:
                kind = "overlap" if start < cursor else "gap"
                raise DriftError(
                    f"{arm} {name} optimizer group {optimizer_group_ranks} has cross-rank shard {kind}: "
                    f"expected_start={cursor} actual_start={start} rank={rank}"
                )
            cursor = end
        if cursor != model_numel:
            raise DriftError(
                f"{arm} {name} optimizer group {optimizer_group_ranks} shard union " f"ends at {cursor}/{model_numel}"
            )
        group_label = ",".join(map(str, optimizer_group_ranks))
        normalized[f"part{part}:group[{group_label}]:{name}"] = [
            [start, end, optimizer_group_rank] for start, end, optimizer_group_rank, _ in ordered
        ]
    if not normalized:
        raise DriftError(f"{arm} has no cross-rank optimizer interval groups")
    return {
        "parameter_shards": len(normalized),
        "sha256": _canonical_json_sha256(normalized),
        "normalized": normalized,
    }


def _changed_parameter_set(evidence: dict[str, Any], family: str) -> set[str]:
    before, _before_families = v3._validated_optimizer_master_snapshot(evidence, "optimizer_masters_before")
    after, _after_families = v3._validated_optimizer_master_snapshot(evidence, "optimizer_masters_after")
    if set(before) != set(after):
        raise DriftError("optimizer parameter identities changed across the step")
    return {
        name
        for name, item in before.items()
        if item.get("family") == family and item.get("sha256") != after[name].get("sha256")
    }


def _validate_reference_runtime(evidence: dict[str, Any], *, rank: int) -> None:
    """Validate the reference arm without misidentifying it as production CP attention."""
    if evidence.get("worker_shared_prefix_training_enabled") is not True:
        raise DriftError(f"reference rank {rank} did not activate shared-prefix execution")
    expected_implementations = {
        "hybrid_stack": "forward_hybrid_stack_shared_prefix",
        "mamba_cp": "_forward_mamba_layer_shared_prefix_cp_replay",
        "mamba_variant": "mamba_decomposed_replay",
        "attention_cp": "_reference_cp_attention",
    }
    if evidence.get("runtime_path_implementations") != expected_implementations:
        raise DriftError(f"reference rank {rank} runtime implementations differ")
    raw_phases = evidence.get("runtime_path_calls")
    if not isinstance(raw_phases, dict) or set(raw_phases) != {"after_pre_logprob"}:
        raise DriftError(f"reference rank {rank} runtime counter phases differ")
    raw_counts = raw_phases["after_pre_logprob"]
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(v3.RUNTIME_PATH_COUNTERS):
        raise DriftError(f"reference rank {rank} runtime counter schema differs")
    counts = {
        name: v3._int(value, label=f"reference rank {rank} runtime counter {name}")
        for name, value in raw_counts.items()
    }
    if any(counts[name] <= 0 for name in v3.RUNTIME_REQUIRED_COUNTERS):
        raise DriftError(f"reference rank {rank} did not execute every shared-prefix path")
    if counts["mamba_decomposed_replay"] != counts["mamba_cp"] or any(
        counts[name] != 0 for name in v3.MAMBA_RUNTIME_VARIANTS if name != "mamba_decomposed_replay"
    ):
        raise DriftError(f"reference rank {rank} Mamba replay attestation differs")


def _compare(arguments: argparse.Namespace) -> None:
    root = Path(arguments.evidence_dir)
    output = Path(arguments.output)
    if not root.is_absolute() or not root.is_dir() or output.parent != root:
        raise DriftError("v4 compare requires an absolute evidence directory and direct-child output")
    base_output = root / "PARITY_REPORT.v3.json"
    base_status = "RED"
    try:
        v3._compare(argparse.Namespace(evidence_dir=str(root), output=str(base_output)))
    except v3.ParityError:
        pass
    if base_output.is_file():
        with base_output.open(encoding="utf-8", errors="strict") as stream:
            base_report = json.load(stream)
        base_status = base_report.get("status", "RED") if isinstance(base_report, dict) else "RED"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RED",
        "base_v3_status": base_status,
        "thresholds": {
            "hidden_relative_l2": LAYER_RELATIVE_L2_LIMIT,
            "hidden_cosine": LAYER_COSINE_LIMIT,
            "router_mismatches": ROUTER_MISMATCH_LIMIT,
            "logprob_relative_l2": 0.03,
            "logprob_cosine": 0.995,
            "logprob_max_sequence_mean_exp_abs_delta": v3.LOGPROB_STRICT_LIMIT,
            "gradient_norm_relative_l2": 0.05,
            "gradient_norm_cosine": 0.995,
            "gradient_sample_relative_l2": 0.10,
            "gradient_sample_cosine": 0.99,
            "optimizer_update_relative_l2": 0.10,
            "optimizer_update_cosine": 0.99,
        },
        "checks": {},
        "failures": [],
        "layers": {},
        "routers": {},
        "optimizer_families": {},
    }
    v3._record_check(
        report,
        label="paired v3 parity report",
        passed=base_status == "GREEN",
        details={"status": base_status, "path": str(base_output)},
    )
    diagnostics = {arm: _load_diagnostics(root, arm) for arm in PUBLIC_ARMS}
    base_evidence = {
        "off": v3._load_evidence(root, "off"),
        "on": v3._load_evidence(root, "on"),
        "on-mamba-replay": v3._load_evidence(root, "on-mamba-replay"),
        "on-reference-cp-replay": v3._load_evidence(root / "reference_base", "on-mamba-replay"),
    }
    for arm, values in diagnostics.items():
        expected_forward_only = arm in FORWARD_ONLY_ARMS
        expected_base_arm = "on-mamba-replay" if arm == "on-reference-cp-replay" else arm
        for rank, (diagnostic, base) in enumerate(zip(values, base_evidence[arm], strict=True)):
            v3._validated_arm_contract(base, arm=expected_base_arm, rank=rank)
            if arm == "on-reference-cp-replay":
                _validate_reference_runtime(base, rank=rank)
            else:
                v3._validated_runtime_path_evidence(base, arm=expected_base_arm)
            batch = base.get("batch")
            base_batch_sha256 = batch.get("sha256") if isinstance(batch, dict) else None
            if (
                diagnostic.get("base_schema") != v3.SCHEMA
                or diagnostic.get("base_arm") != expected_base_arm
                or diagnostic.get("batch_sha256") != base_batch_sha256
                or diagnostic.get("config_sha256") != base.get("config_sha256")
                or diagnostic.get("topology") != base.get("topology")
            ):
                raise DriftError(f"{arm} rank {rank} diagnostic is not bound to its paired v3 evidence")
            if diagnostic.get("native_optimizer_gate") != "separate-required":
                raise DriftError(f"{arm} rank {rank} does not attest the separate native optimizer gate")
            if diagnostic.get("te_workaround_attested") is not (not expected_forward_only):
                raise DriftError(f"{arm} rank {rank} TE workaround attestation differs")
            if diagnostic.get("forward_only") is not expected_forward_only:
                raise DriftError(f"{arm} rank {rank} forward-only identity differs")
            reference = arm == "on-reference-cp-replay"
            calls = v3._int(
                diagnostic.get("reference_cp_attention_calls"),
                label=f"{arm} rank {rank} reference CP attention calls",
            )
            if diagnostic.get("reference_cp_attention") is not reference or ((calls > 0) is not reference):
                raise DriftError(f"{arm} rank {rank} reference CP attention attestation differs")
        leader = values[0]
        for rank in range(1, v3.WORLD_SIZE):
            for section in ("layers", "routers"):
                leader_hashes = {key: value.get("semantic_sha256") for key, value in leader[section].items()}
                rank_hashes = {key: value.get("semantic_sha256") for key, value in values[rank][section].items()}
                v3._record_check(
                    report,
                    label=f"{arm} {section} rank0/rank{rank}",
                    passed=leader_hashes == rank_hashes,
                    details={"rank0": leader_hashes, f"rank{rank}": rank_hashes},
                )

    off_diag = diagnostics["off"][0]
    first_layer_failure: int | None = None
    for candidate_arm in ("on", "on-mamba-replay", "on-reference-cp-replay"):
        candidate = diagnostics[candidate_arm][0]
        if set(off_diag["layers"]) != set(candidate["layers"]):
            raise DriftError(f"OFF/{candidate_arm} decoder layer identities differ")
        for layer_key in sorted(off_diag["layers"], key=int):
            metrics = v3._vector_metrics(
                _features(off_diag["layers"][layer_key], label=f"OFF layer {layer_key}"),
                _features(candidate["layers"][layer_key], label=f"{candidate_arm} layer {layer_key}"),
                label=f"OFF/{candidate_arm} layer {layer_key} hidden features",
            )
            passed = metrics["relative_l2"] < LAYER_RELATIVE_L2_LIMIT and metrics["cosine"] > LAYER_COSINE_LIMIT
            report["layers"].setdefault(layer_key, {})[candidate_arm] = metrics
            v3._record_check(
                report,
                label=f"OFF/{candidate_arm} layer {layer_key} semantic hidden parity",
                passed=passed,
                details=metrics,
            )
            if candidate_arm == "on" and not passed and first_layer_failure is None:
                first_layer_failure = int(layer_key)
        if set(off_diag["routers"]) != set(candidate["routers"]):
            raise DriftError(f"OFF/{candidate_arm} decoder router identities differ")
        for layer_key in sorted(off_diag["routers"], key=int):
            mismatches = _router_mismatches(
                off_diag["routers"][layer_key],
                candidate["routers"][layer_key],
                label=f"OFF/{candidate_arm} router {layer_key}",
            )
            report["routers"].setdefault(layer_key, {})[candidate_arm] = mismatches
            v3._record_check(
                report,
                label=f"OFF/{candidate_arm} router {layer_key} exact top-k parity",
                passed=mismatches["mismatches"] == ROUTER_MISMATCH_LIMIT,
                details=mismatches,
            )
    report["first_on_hidden_failure_layer"] = first_layer_failure

    off = v3._load_evidence(root, "off")
    on = v3._load_evidence(root, "on")
    replay = v3._load_evidence(root, "on-mamba-replay")
    reference = v3._load_evidence(root / "reference_base", "on-mamba-replay")
    for arm, values in (("on-mamba-replay", replay), ("on-reference-cp-replay", reference)):
        metrics = v3._logprob_metrics(
            off[0]["selected_logprobs_before"],
            values[0]["selected_logprobs_before"],
            label=f"OFF/{arm} pre-update selected logprobs",
        )
        passed = (
            metrics["relative_l2"] < 0.03
            and metrics["cosine"] > 0.995
            and metrics["max_sequence_mean_exp_abs_delta"] < v3.LOGPROB_STRICT_LIMIT
        )
        v3._record_check(
            report,
            label=f"OFF/{arm} strict replay/reference logprob parity",
            passed=passed,
            details=metrics,
        )

    off_intervals = _validate_interval_coverage(diagnostics["off"], arm="off")
    on_intervals = _validate_interval_coverage(diagnostics["on"], arm="on")
    v3._record_check(
        report,
        label="OFF/ON cross-rank optimizer interval coverage",
        passed=off_intervals["normalized"] == on_intervals["normalized"],
        details={
            "off_sha256": off_intervals["sha256"],
            "on_sha256": on_intervals["sha256"],
            "parameter_shards": off_intervals["parameter_shards"],
        },
    )

    family_counts = off[0]["optimizer_state_initialization"]["parameter_coverage"]["full_family_counts"]
    if not isinstance(family_counts, dict) or not family_counts:
        raise DriftError("OFF optimizer coverage has no trainable families")
    families = sorted(family_counts)
    if "embedding_output" not in families or "other" not in families:
        raise DriftError(f"expected embedding_output and other trainable families, got {families}")
    for family in families:
        aggregate = v3._aggregate_family_optimizer_updates(off, on, family)
        report["optimizer_families"][family] = aggregate
        off_changed = {
            (rank, name) for rank, evidence in enumerate(off) for name in _changed_parameter_set(evidence, family)
        }
        on_changed = {
            (rank, name) for rank, evidence in enumerate(on) for name in _changed_parameter_set(evidence, family)
        }
        v3._record_check(
            report,
            label=f"global {family} changed parameter set parity",
            passed=off_changed == on_changed,
            details={
                "off_count": len(off_changed),
                "on_count": len(on_changed),
                "off_only": sorted(off_changed - on_changed)[:20],
                "on_only": sorted(on_changed - off_changed)[:20],
            },
        )
        update = aggregate["update_projection"]
        v3._record_check(
            report,
            label=f"global {family} optimizer update parity v4",
            passed=update["relative_l2"] < 0.10 and update["cosine"] > 0.99,
            details=update,
        )
        for rank, (dense, shared) in enumerate(zip(off, on, strict=True)):
            off_norms, off_samples = v3._family_gradient_vectors(dense, family)
            on_norms, on_samples = v3._family_gradient_vectors(shared, family)
            norm_metrics = v3._vector_metrics(off_norms, on_norms, label=f"rank {rank} {family} gradient norms v4")
            sample_metrics = v3._vector_metrics(
                off_samples, on_samples, label=f"rank {rank} {family} gradient samples v4"
            )
            v3._record_check(
                report,
                label=f"rank {rank} {family} all-family gradient parity v4",
                passed=(
                    norm_metrics["relative_l2"] < 0.05
                    and norm_metrics["cosine"] > 0.995
                    and sample_metrics["relative_l2"] < 0.10
                    and sample_metrics["cosine"] > 0.99
                ),
                details={"norms": norm_metrics, "samples": sample_metrics},
            )

    report["status"] = "GREEN" if not report["failures"] else "RED"
    v3._atomic_json(output, report)
    marker = "GREEN" if report["status"] == "GREEN" else "RED"
    print(
        f"NEMORL_CAPTURED_CITATION_DRIFT_{marker} "
        f"failures={len(report['failures'])} first_on_hidden_failure_layer={first_layer_failure} "
        f"report={output}",
        flush=True,
    )
    if report["status"] != "GREEN":
        raise DriftError(f"sealed RED v4 drift report at {output}")


def _semantic_self_test() -> dict[str, int]:
    rows = [
        {"input_length": 5, "prompt_length": 2},
        {"input_length": 4, "prompt_length": 2},
        {"input_length": 6, "prompt_length": 2},
        {"input_length": 3, "prompt_length": 2},
    ]
    plan = _SemanticPlan(rows, padding_multiple=2, dense_order=[2, 0, 1, 3])
    dense = torch.full((sum(plan.padded_lengths), 1), -1, dtype=torch.int64)
    cursor = 0
    for row in plan.dense_order:
        dense[cursor : cursor + plan.input_lengths[row], 0] = row * 100 + torch.arange(plan.input_lengths[row])
        cursor += plan.padded_lengths[row]
    dense_rows = plan.rows_from_physical(dense, shared=False)
    if any(
        tensor[:, 0].tolist() != [row * 100 + index for index in range(plan.input_lengths[row])]
        for row, tensor in enumerate(dense_rows)
    ):
        raise DriftError("semantic self-test failed dense row recovery")

    shared_length = plan.prompt_length + sum(padded - plan.prompt_length for padded in plan.padded_lengths)
    shared = torch.full((v3._round_up(shared_length, plan.padding_multiple), 1), -1, dtype=torch.int64)
    shared[: plan.prompt_length, 0] = torch.arange(plan.prompt_length)
    cursor = plan.prompt_length
    for row in plan.shared_order:
        logical = plan.input_lengths[row] - plan.prompt_length
        physical = plan.padded_lengths[row] - plan.prompt_length
        shared[cursor : cursor + logical, 0] = row * 100 + torch.arange(plan.prompt_length, plan.input_lengths[row])
        cursor += physical
    shared_rows = plan.rows_from_physical(shared, shared=True)
    if any(
        tensor[:, 0].tolist()
        != [
            *range(plan.prompt_length),
            *[row * 100 + index for index in range(plan.prompt_length, plan.input_lengths[row])],
        ]
        for row, tensor in enumerate(shared_rows)
    ):
        raise DriftError("semantic self-test failed shared row recovery")
    return {"dense_rows": len(dense_rows), "shared_rows": len(shared_rows)}


def _interval_self_test() -> dict[str, int]:
    def rank_evidence(rank: int, cp_rank: int, start: int, end: int) -> dict[str, Any]:
        return {
            "optimizer_intervals": [
                {
                    "rank": rank,
                    "tp_rank": 0,
                    "cp_rank": cp_rank,
                    "optimizer_group_rank": cp_rank,
                    "optimizer_group_ranks": [0, 2],
                    "part": 0,
                    "name": "chunk0.weight",
                    "family": "other",
                    "start": start,
                    "end": end,
                    "model_numel": 8,
                }
            ]
        }

    values = [
        rank_evidence(0, 0, 0, 4),
        {"optimizer_intervals": []},
        rank_evidence(2, 1, 4, 8),
        {"optimizer_intervals": []},
    ]
    # Real TP ranks with no local shard still have other interval records. Add a
    # complete independent group so the synthetic rank payloads remain nonempty.
    for rank in (1, 3):
        values[rank]["optimizer_intervals"].append(
            {
                "rank": rank,
                "tp_rank": 1,
                "cp_rank": 0 if rank == 1 else 1,
                "optimizer_group_rank": 0 if rank == 1 else 1,
                "optimizer_group_ranks": [1, 3],
                "part": 0,
                "name": "chunk0.peer_weight",
                "family": "other",
                "start": 0 if rank == 1 else 4,
                "end": 4 if rank == 1 else 8,
                "model_numel": 8,
            }
        )
    result = _validate_interval_coverage(values, arm="self-test")
    rejected = 0
    broken = copy.deepcopy(values)
    broken[2]["optimizer_intervals"][0]["start"] = 5
    try:
        _validate_interval_coverage(broken, arm="self-test-gap")
    except DriftError:
        rejected += 1
    else:
        raise DriftError("interval self-test accepted a cross-rank gap")
    broken = copy.deepcopy(values)
    broken[2]["optimizer_intervals"][0]["start"] = 3
    try:
        _validate_interval_coverage(broken, arm="self-test-overlap")
    except DriftError:
        rejected += 1
    else:
        raise DriftError("interval self-test accepted a cross-rank overlap")
    return {"parameter_shards": result["parameter_shards"], "rejected": rejected}


def _zigzag_self_test() -> dict[str, int]:
    canonical = torch.arange(24).reshape(8, 3)
    for cp_size in (1, 2, 4):
        rank_major = _redo_cp_zigzag(canonical, cp_size)
        restored = _undo_cp_zigzag(rank_major, cp_size)
        if not torch.equal(canonical, restored):
            raise DriftError(f"CP{cp_size} zigzag self-test failed")
    return {"cp_cases": 3}


def _paired_import_self_test() -> dict[str, int]:
    """Reproduce Deployment-Q's partial namespace and prove sibling precedence."""
    import tempfile
    import types

    package_name = "tools.model_diagnostics.shared_prefix_captured_citation_parity"
    namespace_names = ("tools", "tools.model_diagnostics", package_name)
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in namespace_names}
    test_module_name = "_nemorl_paired_import_self_test"
    try:
        tools_namespace = types.ModuleType("tools")
        tools_namespace.__path__ = []
        diagnostics_namespace = types.ModuleType("tools.model_diagnostics")
        diagnostics_namespace.__path__ = []
        tools_namespace.model_diagnostics = diagnostics_namespace
        sys.modules["tools"] = tools_namespace
        sys.modules["tools.model_diagnostics"] = diagnostics_namespace
        sys.modules.pop(package_name, None)
        try:
            exec(
                "from tools.model_diagnostics import shared_prefix_captured_citation_parity",
                {},
            )
        except ImportError:
            pass
        else:
            raise RuntimeError("partial namespace self-test unexpectedly resolved the package import")
        with tempfile.TemporaryDirectory(prefix="nemorl-drift-import-") as directory:
            sibling = Path(directory) / "shared_prefix_captured_citation_parity.py"
            sibling.write_text("SENTINEL = 'mounted-sibling'\n", encoding="utf-8")
            loaded = _load_paired_module(
                sibling,
                package_name=package_name,
                module_name=test_module_name,
            )
            if getattr(loaded, "SENTINEL", None) != "mounted-sibling":
                raise RuntimeError("paired import self-test did not load the mounted sibling")
    finally:
        sys.modules.pop(test_module_name, None)
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return {"partial_namespace_rejected": 1, "sibling_loaded": 1}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-arm")
    run.add_argument("--arm", choices=PUBLIC_ARMS, required=True)
    run.add_argument("--repo-root", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--batch", required=True)
    run.add_argument("--expected-batch-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("compare")
    compare.add_argument("--evidence-dir", required=True)
    compare.add_argument("--output", required=True)
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run-arm":
            _run_arm(arguments)
        elif arguments.command == "compare":
            _compare(arguments)
        elif arguments.command == "self-test":
            semantic = _semantic_self_test()
            intervals = _interval_self_test()
            zigzag = _zigzag_self_test()
            paired_import = _paired_import_self_test()
            print(
                "CAPTURED_CITATION_DRIFT_SELF_TEST_GREEN "
                f"semantic={semantic} intervals={intervals} zigzag={zigzag} "
                f"paired_import={paired_import}",
                flush=True,
            )
        else:
            raise DriftError(f"unhandled command {arguments.command!r}")
    except (DriftError, v3.ParityError, OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        print(f"CAPTURED_CITATION_DRIFT_ERROR: {error}", file=os.sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
