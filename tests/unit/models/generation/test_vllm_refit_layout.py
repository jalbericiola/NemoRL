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

import pytest
import torch

from nemo_rl.models.generation.vllm.refit_layout import (
    parse_hf_expert_weight,
    resolve_vllm_expert_parameter_name,
    select_hf_weight_for_vllm_target,
)


def _layout(*, tp_rank=0, tp_size=1, local_expert_ids=None):
    return {
        "expert_params": {
            "model.layers.0.mlp.experts.routed_experts.w13_weight": {
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "local_expert_ids": local_expert_ids,
            },
            "model.layers.0.mlp.experts.routed_experts.w2_weight": {
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "local_expert_ids": local_expert_ids,
            },
        },
        "missing_weight_prefixes": [],
    }


def test_parse_hf_expert_weight_maps_vllm_fused_parameters():
    gate = parse_hf_expert_weight("model.layers.0.mlp.experts.7.gate_proj.weight")
    down = parse_hf_expert_weight("model.layers.0.mlp.experts.7.down_proj.weight")

    assert gate is not None
    assert (gate.parameter_name, gate.expert_id, gate.shard_id, gate.tp_shard_dim) == (
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        7,
        "w1",
        0,
    )
    assert down is not None
    assert (down.parameter_name, down.shard_id, down.tp_shard_dim) == (
        "model.layers.0.mlp.experts.routed_experts.w2_weight",
        "w2",
        1,
    )


def test_bridge_nemotron_h_mixer_expert_resolves_to_vllm_model_prefix():
    source_name = "backbone.layers.0.mixer.experts.7.up_proj.weight"
    parsed = parse_hf_expert_weight(source_name)
    assert parsed is not None
    assert parsed.parameter_name == ("backbone.layers.0.mixer.experts.routed_experts.w13_weight")

    layout = {
        "expert_params": {
            "model.layers.0.mixer.experts.routed_experts.w13_weight": {
                "tp_rank": 0,
                "tp_size": 1,
                "local_expert_ids": [7],
            }
        },
        "missing_weight_prefixes": [],
    }
    weight = torch.ones(4, 4)

    assert select_hf_weight_for_vllm_target(source_name, weight, target_layout=layout) is weight


def test_native_mtp_expert_cannot_alias_main_model_expert() -> None:
    candidate = "mtp.layers.2.mixer.experts.routed_experts.w13_weight"
    main_parameter = "model.layers.2.mixer.experts.routed_experts.w13_weight"

    assert resolve_vllm_expert_parameter_name(candidate, [main_parameter]) is None


def test_native_mtp_expert_passes_through_main_model_shard_layout() -> None:
    source_name = "mtp.layers.2.mixer.experts.1.up_proj.weight"
    main_parameter = "model.layers.2.mixer.experts.routed_experts.w13_weight"
    layout = {
        "expert_params": {
            main_parameter: {
                "tp_rank": 1,
                "tp_size": 2,
                "local_expert_ids": [1],
            }
        },
        "missing_weight_prefixes": [],
    }
    weight = torch.arange(32).reshape(8, 4)

    selected = select_hf_weight_for_vllm_target(
        source_name,
        weight,
        target_layout=layout,
    )

    assert selected is weight


def test_native_mtp_expert_aliases_only_within_mtp_namespace() -> None:
    candidate = "mtp.layers.2.mixer.experts.routed_experts.w13_weight"
    drafter_parameter = "language_model.mtp.layers.2.mixer.experts.routed_experts.w13_weight"

    assert resolve_vllm_expert_parameter_name(candidate, [drafter_parameter]) == drafter_parameter


def test_mtp_drafter_can_explicitly_alias_native_source_to_model_layers() -> None:
    candidate = "mtp.layers.2.mixer.experts.routed_experts.w13_weight"
    drafter_parameter = "model.layers.2.mixer.experts.routed_experts.w13_weight"

    assert (
        resolve_vllm_expert_parameter_name(
            candidate,
            [drafter_parameter],
            target_is_mtp_drafter=True,
        )
        == drafter_parameter
    )


def test_select_hf_weight_uses_destination_tp_coordinate():
    gate = torch.arange(32).reshape(8, 4)
    down = torch.arange(32).reshape(4, 8)
    layout = _layout(tp_rank=1, tp_size=2)

    selected_gate = select_hf_weight_for_vllm_target(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        gate,
        target_layout=layout,
    )
    selected_down = select_hf_weight_for_vllm_target(
        "model.layers.0.mlp.experts.0.down_proj.weight",
        down,
        target_layout=layout,
    )

    torch.testing.assert_close(selected_gate, gate[4:])
    torch.testing.assert_close(selected_down, down[:, 4:])


def test_select_hf_weight_filters_ep_ownership_without_tp_slicing():
    local_weight = torch.arange(32).reshape(8, 4)
    remote_weight = local_weight + 100
    layout = _layout(local_expert_ids=[1, 3])

    selected = select_hf_weight_for_vllm_target(
        "model.layers.0.mlp.experts.3.up_proj.weight",
        local_weight,
        target_layout=layout,
    )
    skipped = select_hf_weight_for_vllm_target(
        "model.layers.0.mlp.experts.2.up_proj.weight",
        remote_weight,
        target_layout=layout,
    )

    assert selected is local_weight
    assert skipped is None


def test_select_hf_weight_applies_intra_expert_tp_after_ep_ownership():
    weight = torch.arange(32).reshape(8, 4)

    selected = select_hf_weight_for_vllm_target(
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        weight,
        target_layout=_layout(tp_rank=1, tp_size=2, local_expert_ids=[1, 3]),
    )

    torch.testing.assert_close(selected, weight[4:])


def test_select_hf_weight_rejects_nondivisible_tp_layout():
    with pytest.raises(ValueError, match="across vLLM TP size 3"):
        select_hf_weight_for_vllm_target(
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            torch.ones(8, 4),
            target_layout=_layout(tp_size=3, local_expert_ids=[3]),
        )


def test_select_hf_weight_leaves_non_expert_weights_unchanged():
    weight = torch.arange(16).reshape(4, 4)

    selected = select_hf_weight_for_vllm_target(
        "model.layers.0.self_attn.q_proj.weight",
        weight,
        target_layout=_layout(local_expert_ids=[]),
    )

    assert selected is weight


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
    ],
)
def test_select_hf_weight_skips_weights_absent_from_destination_stage(name):
    layout = _layout(local_expert_ids=[0])
    layout["missing_weight_prefixes"] = ["model.layers.1."]

    selected = select_hf_weight_for_vllm_target(
        name,
        torch.ones(4, 4),
        target_layout=layout,
    )

    assert selected is None


def test_select_hf_weight_skips_experts_absent_from_destination_stage():
    selected = select_hf_weight_for_vllm_target(
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        torch.ones(4, 4),
        target_layout=_layout(local_expert_ids=[0]),
    )

    assert selected is None
