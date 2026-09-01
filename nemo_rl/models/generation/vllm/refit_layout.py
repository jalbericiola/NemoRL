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

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch


class VllmExpertParamLayout(TypedDict):
    tp_rank: int
    tp_size: int
    local_expert_ids: list[int] | None


class VllmWeightLayout(TypedDict):
    expert_params: dict[str, VllmExpertParamLayout]
    missing_weight_prefixes: list[str]


@dataclass(frozen=True)
class HfExpertWeight:
    parameter_name: str
    expert_id: int
    shard_id: Literal["w1", "w2", "w3"]
    tp_shard_dim: int


_HF_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.(?:mlp|mixer)\.experts)\."
    r"(?P<expert_id>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_HF_PROJECTION_SHARDS: dict[str, Literal["w1", "w2", "w3"]] = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def parse_hf_expert_weight(name: str) -> HfExpertWeight | None:
    match = _HF_EXPERT_WEIGHT_RE.match(name)
    if match is None:
        return None

    projection = match.group("projection")
    shard_id = _HF_PROJECTION_SHARDS[projection]
    parameter_leaf = "w2_weight" if shard_id == "w2" else "w13_weight"
    # vLLM 0.25 stores expert weights on the RoutedExperts submodule of the
    # MoERunner returned by the FusedMoE factory, so the named_parameters()
    # key gains a ".routed_experts." segment.
    return HfExpertWeight(
        parameter_name=f"{match.group('prefix')}.routed_experts.{parameter_leaf}",
        expert_id=int(match.group("expert_id")),
        shard_id=shard_id,
        tp_shard_dim=1 if shard_id == "w2" else 0,
    )


def _expert_parameter_namespace(name: str) -> Literal["main", "mtp"]:
    """Return the model namespace that owns an expert parameter."""
    if re.search(r"(?:^|\.)mtp\.layers\.\d+\.", name) is not None:
        return "mtp"
    return "main"


def _expert_parameter_alias(name: str) -> tuple[Literal["main", "mtp"], str]:
    """Return a prefix-independent, namespace-preserving expert alias."""
    layer_match = re.search(r"(?:^|\.)(layers\.\d+\..+)$", name)
    relative_name = layer_match.group(1) if layer_match is not None else name
    return (
        _expert_parameter_namespace(name),
        relative_name.replace(".routed_experts.", "."),
    )


def resolve_vllm_expert_parameter_name(
    candidate: str,
    available_names: Iterable[str],
    *,
    target_is_mtp_drafter: bool = False,
) -> str | None:
    """Resolve HF/vLLM prefix and ``routed_experts`` namespace aliases.

    Nemotron-H exports ``backbone.layers.*.mixer.experts`` while vLLM owns
    ``model.layers.*.mixer.experts.routed_experts``.  Matching the layer-local
    suffix keeps this conversion explicit without hard-coding either main-model
    prefix. Native ``mtp.layers.*`` parameters belong to vLLM's separate
    drafter and must never alias a main-model ``model.layers.*`` parameter.
    Ambiguous aliases are rejected instead of selecting a target
    nondeterministically. ``target_is_mtp_drafter`` is reserved for a drafter
    whose own vLLM parameter names use ``model.layers.*``; main-model refit
    callers must keep the fail-closed default.
    """
    available = set(available_names)
    if candidate in available:
        return candidate

    alias = _expert_parameter_alias(candidate)
    matches = sorted(
        name
        for name in available
        if _expert_parameter_alias(name) == alias
        or (target_is_mtp_drafter and alias[0] == "mtp" and _expert_parameter_alias(name) == ("main", alias[1]))
    )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous vLLM expert parameter alias for {candidate!r}: {matches}")
    return matches[0] if matches else None


def select_hf_weight_for_vllm_target(
    name: str,
    tensor: torch.Tensor,
    *,
    target_layout: VllmWeightLayout,
) -> torch.Tensor | None:
    """Return the destination-local weight, or ``None`` if not owned.

    Pipeline stages omit complete parameter prefixes. Within an owned stage,
    tensor-parallel MoE layers shard every expert tensor while expert-parallel
    layers place complete experts on selected ranks.
    """
    if any(name.startswith(prefix) for prefix in target_layout["missing_weight_prefixes"]):
        return None

    expert_weight = parse_hf_expert_weight(name)
    if expert_weight is None:
        return tensor

    resolved_parameter_name = resolve_vllm_expert_parameter_name(
        expert_weight.parameter_name, target_layout["expert_params"]
    )
    if resolved_parameter_name is None and _expert_parameter_namespace(expert_weight.parameter_name) == "mtp":
        # The checkpoint-engine layout describes only the main vLLM model.
        # Native MTP weights remain in the policy stream so the receiver can
        # route their full tensors to the separately-owned MTP drafter.
        return tensor
    param_layout = (
        target_layout["expert_params"].get(resolved_parameter_name) if resolved_parameter_name is not None else None
    )
    if param_layout is None:
        # A missing expert parameter belongs to another pipeline stage.
        return None

    local_expert_ids = param_layout["local_expert_ids"]
    if local_expert_ids is not None and expert_weight.expert_id not in local_expert_ids:
        return None

    tp_rank = param_layout["tp_rank"]
    tp_size = param_layout["tp_size"]

    shard_dim = expert_weight.tp_shard_dim
    if tp_size == 1:
        return tensor
    if tensor.shape[shard_dim] % tp_size != 0:
        raise ValueError(
            f"Cannot shard {name} dimension {shard_dim} of size "
            f"{tensor.shape[shard_dim]} across vLLM TP size {tp_size}."
        )

    shard_size = tensor.shape[shard_dim] // tp_size
    return tensor.narrow(shard_dim, tp_rank * shard_size, shard_size).contiguous()
