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

from typing import cast

import torch

from nemo_rl.models.generation.vllm.refit_layout import (
    VllmExpertParamLayout,
    VllmWeightLayout,
    parse_hf_expert_weight,
    resolve_vllm_expert_parameter_name,
)


class VllmShardedExpertRefitMixin:
    """Load destination-local expert shards into vLLM storage."""

    def _is_sharded_refit_weight(self, name: str, tensor: torch.Tensor) -> bool:
        param_names = self._sharded_refit_param_names(name)
        if not param_names:
            return False

        state_dict_info = getattr(self, "state_dict_info", None)
        if state_dict_info is None or name not in state_dict_info:
            return False
        full_shape, _dtype = state_dict_info[name]
        return torch.Size(full_shape) != tensor.shape

    def _sharded_refit_param_names(self, name: str) -> list[str]:
        expert_weight = parse_hf_expert_weight(name)
        return [] if expert_weight is None else [expert_weight.parameter_name]

    def _checkpoint_engine_weight_layout(self) -> VllmWeightLayout:
        from vllm.model_executor.models.utils import get_pp_missing_layer_names

        expert_params: dict[str, VllmExpertParamLayout] = {}
        for name, param in self._get_named_parameters().items():
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue

            if bool(getattr(param, "is_transposed", False)):
                raise ValueError(
                    "Sharded NIXL expert refit requires canonical expert-weight "
                    f"orientation, but {name} is transposed."
                )

            weight_loader = getattr(param, "weight_loader", None)
            owner = getattr(weight_loader, "__self__", None)
            if owner is None:
                raise RuntimeError(f"Could not inspect the vLLM expert weight loader for {name}.")

            self._validate_expert_storage(name, param)

            quant_method = getattr(owner, "quant_method", None)
            backend = getattr(quant_method, "unquantized_backend", None)
            backend_name = getattr(backend, "name", None)
            if getattr(owner, "quant_config", None) is not None or backend_name not in {
                "TRITON",
                "BATCHED_TRITON",
            }:
                raise ValueError(
                    "Sharded NIXL expert refit requires canonical unquantized Triton "
                    f"expert weights, but {name} uses "
                    f"{type(quant_method).__name__}/{backend_name}. Set "
                    "policy.generation.vllm_kwargs.moe_backend=triton."
                )

            moe_config = owner.moe_config
            use_ep = bool(getattr(owner, "use_ep", False))
            local_expert_ids: list[int] | None = None
            if use_ep:
                if bool(moe_config.moe_parallel_config.enable_eplb):
                    raise RuntimeError(
                        "Sharded refit does not support dynamic vLLM expert load "
                        "balancing because ownership can change after metadata "
                        "exchange."
                    )
                expert_map = getattr(owner, "_expert_map", None)
                if expert_map is None:
                    raise RuntimeError(f"vLLM reports EP for {name} without an expert ownership map.")
                logical_num_experts = int(
                    cast(
                        int,
                        getattr(moe_config, "num_logical_experts", expert_map.numel()),
                    )
                )
                global_num_experts = int(
                    cast(
                        int,
                        getattr(owner, "global_num_experts", logical_num_experts),
                    )
                )
                if global_num_experts != logical_num_experts:
                    raise RuntimeError("Sharded refit does not support redundant vLLM experts.")
                local_expert_ids = [
                    expert_id
                    for expert_id, local_id in enumerate(expert_map.detach().cpu().tolist()[:logical_num_experts])
                    if int(local_id) >= 0
                ]

            expert_params[name] = {
                "tp_rank": int(moe_config.tp_rank),
                "tp_size": int(moe_config.tp_size),
                "local_expert_ids": local_expert_ids,
            }

        # parse_hf_expert_weight() builds its lookup key by hardcoding the
        # ".routed_experts." segment that vLLM 0.25 introduced, while these keys
        # come from real named_parameters(). This is the one place that sees
        # both sides, so check them here: on a mismatch every HF expert weight
        # would map to a non-existent parameter,
        # select_hf_weight_for_vllm_target() would return None for all of them,
        # and the checkpoint-engine sender would drop them silently -- the
        # engine would then serve stale experts for the whole run with no
        # exception and no warning, since there is no require_complete()
        # equivalent on this path.
        if expert_params and not all(".routed_experts." in name for name in expert_params):
            raise RuntimeError(
                "vLLM expert parameters are not under the expected "
                "'.routed_experts.' submodule (saw "
                f"{sorted(expert_params)[:3]}). parse_hf_expert_weight() would map "
                "every HF expert weight to a non-existent parameter and the "
                "checkpoint-engine sender would silently skip them, leaving the "
                "engine serving stale experts."
            )

        return {
            "expert_params": expert_params,
            "missing_weight_prefixes": get_pp_missing_layer_names(self.model_runner.model),
        }

    @staticmethod
    def _validate_expert_storage(param_name: str, param: torch.nn.Parameter) -> None:
        """Reject incompatible vLLM storage before advertising shards."""
        is_w13 = param_name.endswith(".w13_weight")
        if param.ndim != 3 or (is_w13 and param.shape[1] % 2 != 0):
            raise RuntimeError(f"Unsupported vLLM expert storage for {param_name}: " f"shape={tuple(param.shape)}.")

    def _local_expert_id(self, param: torch.nn.Parameter, expert_id: int) -> int:
        weight_loader = getattr(param, "weight_loader", None)
        owner = getattr(weight_loader, "__self__", None)
        mapper = getattr(owner, "_map_global_expert_id_to_local_expert_id", None)
        if mapper is None:
            return expert_id
        return int(mapper(expert_id))

    def _load_destination_local_expert_group(
        self,
        param_name: str,
        param: torch.nn.Parameter,
        shard_id: str,
        items: list[tuple[int, torch.Tensor]],
        *,
        split_w13: bool = True,
    ) -> None:
        """Copy destination-local experts into canonical vLLM storage."""
        if shard_id not in {"w1", "w2", "w3"}:
            raise ValueError(f"Unexpected sharded expert shard_id: {shard_id}")

        target = param.data
        if shard_id in {"w1", "w3"} and split_w13:
            shard_size = target.shape[1] // 2
            shard_offset = 0 if shard_id == "w1" else shard_size
            target = target.narrow(1, shard_offset, shard_size)

        sorted_items = sorted(items, key=lambda item: item[0])
        expected_item_shape = target.shape[1:]
        for local_expert_id, source in sorted_items:
            if not 0 <= local_expert_id < target.shape[0]:
                raise ValueError(
                    f"Sharded expert {param_name} local expert id "
                    f"{local_expert_id} is outside destination extent "
                    f"{target.shape[0]}."
                )
            if source.shape != expected_item_shape:
                raise ValueError(
                    f"Sharded expert {param_name}/{shard_id} source for local "
                    f"expert {local_expert_id} has shape {tuple(source.shape)}, "
                    f"but the selected destination region requires "
                    f"{tuple(expected_item_shape)}."
                )

        expert_ids = [expert_id for expert_id, _tensor in sorted_items]
        if not expert_ids:
            return
        contiguous_ids = list(range(expert_ids[0], expert_ids[0] + len(expert_ids)))
        with torch.no_grad():
            if expert_ids == contiguous_ids:
                expert_data = target.narrow(0, expert_ids[0], len(expert_ids))
                loaded_weight = torch.stack([tensor for _expert_id, tensor in sorted_items])
                expert_data.copy_(loaded_weight)
            else:
                for local_expert_id, loaded_weight in sorted_items:
                    target[local_expert_id].copy_(loaded_weight)

    def _load_sharded_expert_weight_groups(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        params = self._get_named_parameters()
        groups: dict[tuple[str, str, bool], list[tuple[int, torch.Tensor]]] = {}
        group_source_names: dict[tuple[str, str, bool], set[str]] = {}
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            expert_weight = parse_hf_expert_weight(name)
            if expert_weight is None:
                remaining_weights.append((name, tensor))
                continue

            mapped_name = resolve_vllm_expert_parameter_name(expert_weight.parameter_name, params)
            param = params.get(mapped_name) if mapped_name is not None else None
            if param is None:
                remaining_weights.append((name, tensor))
                continue

            owner = getattr(getattr(param, "weight_loader", None), "__self__", None)
            if not self._is_sharded_refit_weight(name, tensor) and not bool(getattr(owner, "use_ep", False)):
                remaining_weights.append((name, tensor))
                continue

            if owner is None:
                raise RuntimeError("Could not resolve the vLLM expert weight loader for " f"{mapped_name}.")

            if param.data.ndim != 3 or tensor.ndim != 2:
                raise ValueError(
                    f"Sharded expert {name} requires a 3-D vLLM parameter and "
                    f"2-D source tensor, got {param.data.ndim}-D and {tensor.ndim}-D."
                )

            full_shape, _dtype = self.state_dict_info[name]
            expected_shape = list(full_shape)
            tp_size = int(owner.moe_config.tp_size)
            shard_dim = expert_weight.tp_shard_dim
            expected_shape[shard_dim] //= tp_size
            if tensor.shape != torch.Size(expected_shape):
                raise ValueError(
                    f"Received sharded expert {name} with shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected_shape)} from "
                    f"full shape {tuple(full_shape)} and TP size {tp_size}."
                )

            local_expert_id = self._local_expert_id(param, expert_weight.expert_id)
            if local_expert_id == -1:
                continue

            # Nemotron-H's ``mixer.experts`` block is non-gated. vLLM 0.25
            # stores its sole input projection in ``w13_weight`` with I (not
            # 2I) rows and the official loader treats HF ``up_proj`` as w1.
            # Use vLLM's authoritative activation/storage flag. Missing gate
            # metadata cannot be treated as proof of a non-gated layer: that
            # would turn a dropped gate into a false success.
            split_w13 = bool(getattr(getattr(owner, "moe_config", None), "is_act_and_mul", True))
            shard_id = expert_weight.shard_id
            if not split_w13:
                if name.endswith(".gate_proj.weight"):
                    raise ValueError(
                        "Non-gated vLLM expert destination received an unexpected "
                        f"gate projection: {name}. Refusing to alias it onto w1."
                    )
                if name.endswith(".up_proj.weight"):
                    shard_id = "w1"

            group_key = (mapped_name, shard_id, split_w13)
            groups.setdefault(group_key, []).append((local_expert_id, tensor))
            group_source_names.setdefault(group_key, set()).add(name)

        for (mapped_name, shard_id, split_w13), items in groups.items():
            self._load_destination_local_expert_group(
                mapped_name,
                params[mapped_name],
                shard_id,
                items,
                split_w13=split_w13,
            )
        if groups:
            # These parameters bypass vLLM's load_weights API. Accumulate the
            # exact source expert/shard tokens across the entire transaction;
            # checkpoint-engine buckets are allowed to split experts and
            # w1/w2/w3 between calls, so completeness is checked only by the
            # lifecycle finalizer.
            self._record_direct_main_load(
                {
                    source_name: mapped_name
                    for (
                        mapped_name,
                        _shard_id,
                        _split_w13,
                    ), source_names in group_source_names.items()
                    for source_name in source_names
                }
            )

        return remaining_weights

    def _load_sharded_expert_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        from nemo_rl.models.generation.vllm.quantization import fp8

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            raise ValueError("Sharded NIXL expert refit is not supported for FP8 vLLM models.")

        remaining_weights = self._load_sharded_expert_weight_groups(policy_weights)
        if remaining_weights:
            self._load_full_hf_weights(remaining_weights)
