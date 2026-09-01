# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# NOTE: vllm_backend imports `vllm` eagerly at module top, so it is only imported
# inside the test bodies (which are marked @pytest.mark.vllm). This keeps the
# module collectable in the non-vllm unit lane, where these tests are deselected.

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from safetensors.torch import save_file


def _make_collective_update_extension(backend):
    ext = backend.VllmInternalWorkerExtension.__new__(backend.VllmInternalWorkerExtension)
    state_info = object()
    ext.state_dict_info = {"model.weight": state_info}
    ext.model_update_group = object()
    ext.model_runner = SimpleNamespace(model=SimpleNamespace(named_parameters=lambda: []), vllm_config=object())
    ext.model_config = object()
    ext.device = object()
    return ext, state_info


def _write_sharded_checkpoint(model_dir, shards):
    """Write safetensors shards plus a model.safetensors.index.json.

    Args:
        model_dir: Directory (pathlib.Path) to write the checkpoint into.
        shards: Mapping of shard filename -> {weight_name: tensor}.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(model_dir / shard_name))
        for name in tensors:
            weight_map[name] = shard_name
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def _make_extension_with_drafter(
    mtp_start_layer_idx,
    num_mtp_layers,
    *,
    model_type=None,
    destination_names=(),
    loaded_names=None,
):
    """Build a VllmInternalWorkerExtension with a mocked drafter and stubbed refit."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    predictor = SimpleNamespace(mtp_start_layer_idx=mtp_start_layer_idx, num_mtp_layers=num_mtp_layers)

    class _NonGatedMoeOwner:
        def __init__(self):
            self.use_ep = False
            self.moe_config = SimpleNamespace(is_act_and_mul=False)

        def weight_loader(self, *_args, **_kwargs):
            raise AssertionError("unit-test weight loader should not be called")

    owner = _NonGatedMoeOwner()
    destination_params = {}
    for name in destination_names:
        shape = (1, 4, 4) if name.endswith(("w13_weight", "w2_weight")) else (1,)
        param = torch.nn.Parameter(torch.empty(shape))
        if name.endswith(("w13_weight", "w2_weight")):
            param.weight_loader = owner.weight_loader
        destination_params[name] = param

    def load_weights(*, weights):
        if callable(loaded_names):
            return loaded_names(weights)
        return loaded_names

    draft_model = SimpleNamespace(
        model=predictor,
        config=SimpleNamespace(model_type=model_type),
        quant_config=None,
        named_parameters=lambda: list(destination_params.items()),
        named_modules=lambda: [],
        load_weights=MagicMock(side_effect=load_weights),
    )
    ext.model_runner = MagicMock()
    ext.model_runner.drafter.model = draft_model
    # Isolate this test from _load_draft_weights internals.
    ext._load_draft_weights = MagicMock(return_value=loaded_names)
    return ext


def _make_strict_refit_extension(
    backend,
    *,
    destination_params,
    source_names,
    load_weights,
    model_type="nemotron_h",
    quant_config=None,
):
    """Build a lifecycle-capable strict main-model refit fixture."""
    ext = backend.VllmInternalWorkerExtension.__new__(backend.VllmInternalWorkerExtension)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type),
        quant_config=quant_config,
        named_parameters=lambda: list(destination_params.items()),
        load_weights=MagicMock(side_effect=load_weights),
    )
    ext.model_runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(
            quant_config=quant_config,
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        ),
    )
    ext.state_dict_info = {name: object() for name in source_names}
    ext.model_config = object()
    ext.device = torch.device("cpu")
    ext._maybe_process_mtp_drafter_after_loading = MagicMock()
    ext._maybe_process_fp8_kv_cache = MagicMock()
    return ext, model


def _patch_vllm_postload(monkeypatch):
    """Stub the vLLM post-load helpers imported inside load_mtp_weights_from_disk."""
    monkeypatch.setattr("vllm.config.set_current_vllm_config", lambda cfg: contextlib.nullcontext())
    process_weights = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights,
    )
    return process_weights


def _make_mtp_refit_extension(*, method="mtp", from_disk=False, has_drafter=True, draft_model_config=None):
    """Build an extension for exercising the MTP-refit drafter gating.

    The drafter here is fed from the refit stream (co-trained MTP layer), as
    opposed to the disk-load path built by ``_make_extension_with_drafter``.

    Returns:
        (ext, drafter_model): drafter_model is None when has_drafter is False.
    """
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext._mtp_drafter_from_disk = from_disk

    spec_config = None if method is None else SimpleNamespace(method=method, draft_model_config=draft_model_config)
    drafter_model = SimpleNamespace(load_weights=MagicMock()) if has_drafter else None
    ext.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec_config),
        drafter=SimpleNamespace(model=drafter_model) if has_drafter else None,
    )
    return ext, drafter_model


@pytest.mark.vllm
@pytest.mark.parametrize("with_mtp", [False, True])
def test_update_weights_from_collective_processes_weights_after_loading(monkeypatch, with_mtp):
    from nemo_rl.models.generation.vllm import vllm_backend

    call_order = []
    process_calls = []
    draft_model = object() if with_mtp else None
    draft_model_config = object() if with_mtp else None

    def process_weights_after_loading(model, model_config, device):
        call_order.append("process_mtp" if model is draft_model else "process_main")
        process_calls.append((model, model_config, device))

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights_after_loading,
    )
    ext, expected_state_info = _make_collective_update_extension(vllm_backend)
    if with_mtp:
        ext._mtp_drafter_from_disk = False
        ext.model_runner.drafter = SimpleNamespace(model=draft_model)
        ext.model_runner.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp", draft_model_config=draft_model_config)
        )

    @contextlib.contextmanager
    def set_current_vllm_config(config):
        assert config is ext.model_runner.vllm_config
        call_order.append("config_enter")
        try:
            yield
        finally:
            call_order.append("config_exit")

    monkeypatch.setattr("vllm.config.set_current_vllm_config", set_current_vllm_config)

    def load_weights(weights):
        call_order.append("load")
        assert weights == [("model.weight", "weight-value")]
        ext._record_refit_source_names(name for name, _weight in weights)

    def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
        call_order.append("broadcast")
        assert list(iterator) == [("model.weight", expected_state_info)]
        assert group is ext.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", "weight-value")])

    ext._load_weights = load_weights
    ext._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(vllm_backend, "packed_broadcast_consumer", packed_broadcast_consumer)
    monkeypatch.setattr(vllm_backend.gc, "collect", lambda: call_order.append("gc"))
    monkeypatch.setattr(
        vllm_backend.torch.cuda,
        "empty_cache",
        lambda: call_order.append("empty_cache"),
    )

    assert ext.update_weights_from_collective() is True

    expected_process_calls = [(ext.model_runner.model, ext.model_config, ext.device)]
    expected_call_order = [
        "broadcast",
        "load",
        "config_enter",
        "process_main",
        "config_exit",
    ]
    if with_mtp:
        expected_process_calls.append((draft_model, draft_model_config, ext.device))
        expected_call_order.extend(["config_enter", "process_mtp", "config_exit"])
    expected_call_order.extend(["kv", "gc", "empty_cache"])

    assert process_calls == expected_process_calls
    assert call_order == expected_call_order


@pytest.mark.vllm
def test_main_destination_coverage_accumulates_across_refit_batches(monkeypatch):
    """Every fused Q/K/V and gate/up source is independently accepted."""
    from nemo_rl.models.generation.vllm import vllm_backend

    destination_names = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    ]
    source_names = [
        "backbone.layers.0.self_attn.q_proj.weight",
        "backbone.layers.0.self_attn.k_proj.weight",
        "backbone.layers.0.self_attn.v_proj.weight",
        "backbone.layers.0.mlp.gate_proj.weight",
        "backbone.layers.0.mlp.up_proj.weight",
    ]

    def load_weights(*, weights):
        return {
            destination_names[0] if ".self_attn." in source_name else destination_names[1]
            for source_name, _tensor in weights
        }

    ext, model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={name: torch.nn.Parameter(torch.empty(1)) for name in destination_names},
        source_names=source_names,
        load_weights=load_weights,
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with ext._weight_update_lifecycle("collective") as finalize:
        ext._load_full_hf_weights([(name, name.rsplit(".", 2)[-2]) for name in source_names])
        finalize()

    process_weights.assert_called_once_with(model, ext.model_config, ext.device)
    # Three cohorts suffice: q+gate, k+up, then v. The loader is not invoked
    # once per tensor/layer.
    assert model.load_weights.call_count == 3
    ext._maybe_process_mtp_drafter_after_loading.assert_called_once_with()
    ext._maybe_process_fp8_kv_cache.assert_called_once_with()
    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
@pytest.mark.parametrize("skipped_role", ["q_proj", "k_proj", "v_proj"])
def test_main_fused_qkv_coverage_rejects_skipped_source(monkeypatch, skipped_role):
    """One returned qkv destination cannot hide any skipped source component."""
    from nemo_rl.models.generation.vllm import vllm_backend

    destination = "model.layers.0.mixer.qkv_proj.weight"
    source_names = [f"backbone.layers.0.mixer.{role}.weight" for role in ("q_proj", "k_proj", "v_proj")]

    def load_weights(*, weights):
        return set() if skipped_role in weights[0][0] else {destination}

    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={destination: torch.nn.Parameter(torch.empty(1))},
        source_names=source_names,
        load_weights=load_weights,
    )
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="missing required source components"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([(name, name) for name in source_names])
            finalize()

    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
def test_prepare_refit_info_rejects_invalid_strict_schema_before_transfer():
    from nemo_rl.models.generation.vllm import vllm_backend

    destination = "model.layers.0.mixer.qkv_proj.weight"
    source_names = {f"backbone.layers.0.mixer.{role}.weight" for role in ("q_proj", "k_proj")}
    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={destination: torch.nn.Parameter(torch.empty(1))},
        source_names={"old.weight"},
        load_weights=lambda **_kwargs: {destination},
    )
    previous_state = ext.state_dict_info

    with pytest.raises(ValueError, match="missing source roles.*v"):
        ext.prepare_refit_info({name: object() for name in source_names})

    assert ext.state_dict_info is previous_state
    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
def test_prepare_nccl_refit_info_rolls_back_invalid_strict_schema(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    destination = "model.layers.0.mixer.qkv_proj.weight"
    q_source = "backbone.layers.0.mixer.q_proj.weight"
    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={destination: torch.nn.Parameter(torch.empty(1))},
        source_names=set(),
        load_weights=lambda **_kwargs: {destination},
    )
    old_refit_info = object()
    old_param_map = object()
    old_destinations = {"old.destination"}
    old_pairs = {"old.source": "old.destination"}
    ext.nccl_reshard_refit_info = old_refit_info
    ext.hf_to_local_param_map = old_param_map
    ext._nrl_nccl_reshard_destination_names = old_destinations
    ext._nrl_nccl_reshard_source_destinations = old_pairs

    monkeypatch.setattr(
        "nemo_rl.weight_sync.nccl_reshard_utils.restore_refit_info_placements",
        lambda info: info,
    )

    def build_map(_info):
        ext._nrl_nccl_reshard_destination_names = {destination}
        ext._nrl_nccl_reshard_source_destinations = {q_source: destination}
        return object()

    ext.build_hf_to_local_param_map = build_map
    invalid_info = {
        "layer_names": ["backbone.layers.0"],
        "per_layer_params": {
            "backbone.layers.0": [{"name": q_source}],
        },
        "misc_meta": {},
    }

    with pytest.raises(ValueError, match="missing source roles"):
        ext.prepare_nccl_reshard_refit_info(invalid_info)

    assert ext.nccl_reshard_refit_info is old_refit_info
    assert ext.hf_to_local_param_map is old_param_map
    assert ext._nrl_nccl_reshard_destination_names is old_destinations
    assert ext._nrl_nccl_reshard_source_destinations is old_pairs


@pytest.mark.vllm
@pytest.mark.parametrize("accepted_role", ["gate_proj", "up_proj"])
def test_main_fused_gate_up_coverage_rejects_one_component(monkeypatch, accepted_role):
    from nemo_rl.models.generation.vllm import vllm_backend

    destination = "model.layers.0.mlp.gate_up_proj.weight"
    source_names = [f"backbone.layers.0.mlp.{role}.weight" for role in ("gate_proj", "up_proj")]

    def load_weights(*, weights):
        return {destination} if accepted_role in weights[0][0] else set()

    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={destination: torch.nn.Parameter(torch.empty(1))},
        source_names=source_names,
        load_weights=load_weights,
    )
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="missing required source components"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([(name, name) for name in source_names])
            finalize()


@pytest.mark.vllm
def test_main_destination_coverage_rejects_silent_partial_load(monkeypatch):
    """A complete source transfer cannot hide a destination skipped by vLLM."""
    from nemo_rl.models.generation.vllm import vllm_backend

    missing_name = "model.layers.0.mlp.down_proj.weight"
    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={
            name: torch.nn.Parameter(torch.empty(1)) for name in {"model.embed_tokens.weight", missing_name}
        },
        source_names={"model.embed_tokens.weight", missing_name},
        load_weights=lambda **_kwargs: {"model.embed_tokens.weight"},
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="missing destination parameters.*down_proj"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([("model.embed_tokens.weight", "embed")])
            finalize()

    process_weights.assert_not_called()
    ext._maybe_process_mtp_drafter_after_loading.assert_not_called()
    ext._maybe_process_fp8_kv_cache.assert_not_called()
    assert ext._nrl_main_destination_manifest is None


@pytest.mark.vllm
def test_main_destination_coverage_requires_loader_tracking(monkeypatch):
    """A vLLM API regression that drops loaded-name reporting fails closed."""
    from nemo_rl.models.generation.vllm import vllm_backend

    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={"model.weight": torch.nn.Parameter(torch.empty(1))},
        source_names={"model.weight"},
        load_weights=lambda **_kwargs: None,
    )
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(RuntimeError, match="did not report loaded destination"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([("model.weight", "weight")])
            finalize()

    assert ext._nrl_main_destination_manifest is None


@pytest.mark.vllm
def test_main_destination_coverage_includes_nccl_reshard_bulk_targets(monkeypatch):
    """Direct bulk destinations combine with misc vLLM-loader destinations."""
    from nemo_rl.models.generation.vllm import vllm_backend

    bulk_name = "model.layers.0.mlp.gate_up_proj.weight"
    misc_name = "model.embed_tokens.weight"
    gate_name = "backbone.layers.0.mlp.gate_proj.weight"
    up_name = "backbone.layers.0.mlp.up_proj.weight"
    ext, model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={
            bulk_name: torch.nn.Parameter(torch.empty(2, 1)),
            misc_name: torch.nn.Parameter(torch.empty(1)),
        },
        source_names=set(),
        load_weights=lambda **_kwargs: {misc_name},
    )
    ext.nccl_reshard_refit_info = {
        "layer_names": ["layer0"],
        "per_layer_params": {"layer0": [{"name": gate_name}, {"name": up_name}]},
        "misc_meta": {misc_name: {}},
    }
    ext._nrl_nccl_reshard_destination_names = {bulk_name}
    ext._nrl_nccl_reshard_source_destinations = {
        gate_name: bulk_name,
        up_name: bulk_name,
    }
    process_weights = _patch_vllm_postload(monkeypatch)

    with ext._weight_update_lifecycle("nccl_reshard") as finalize:
        ext._load_full_hf_weights([(misc_name, "embed")])
        finalize()

    process_weights.assert_called_once_with(model, ext.model_config, ext.device)


@pytest.mark.vllm
def test_nccl_direct_source_coverage_rejects_swapped_targets(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    source0 = "backbone.layers.0.mlp.down_proj.weight"
    source1 = "backbone.layers.1.mlp.down_proj.weight"
    destination0 = "model.layers.0.mlp.down_proj.weight"
    destination1 = "model.layers.1.mlp.down_proj.weight"
    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={
            destination0: torch.nn.Parameter(torch.empty(1)),
            destination1: torch.nn.Parameter(torch.empty(1)),
        },
        source_names=set(),
        load_weights=lambda **_kwargs: set(),
    )
    ext.nccl_reshard_refit_info = {
        "layer_names": ["layer0", "layer1"],
        "per_layer_params": {
            "layer0": [{"name": source0}],
            "layer1": [{"name": source1}],
        },
        "misc_meta": {},
    }
    ext._nrl_nccl_reshard_destination_names = {destination0, destination1}
    ext._nrl_nccl_reshard_source_destinations = {
        source0: destination1,
        source1: destination0,
    }
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(RuntimeError, match="wrote.*expected"):
        with ext._weight_update_lifecycle("nccl_reshard") as finalize:
            finalize()

    process_weights.assert_not_called()
    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
def test_generic_loader_returning_none_remains_compatible(monkeypatch):
    """Strict loaded-name tracking does not silently broaden to other models."""
    from nemo_rl.models.generation.vllm import vllm_backend

    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={"model.weight": torch.nn.Parameter(torch.empty(1))},
        source_names={"model.weight"},
        load_weights=lambda **_kwargs: None,
        model_type="llama",
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with ext._weight_update_lifecycle("collective") as finalize:
        ext._load_full_hf_weights([("model.weight", "weight")])
        finalize()

    process_weights.assert_called_once()


@pytest.mark.vllm
def test_main_manifest_excludes_draft_and_wrapped_mtp_namespaces(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    destination = "model.layers.0.mixer.qkv_proj.weight"
    main_sources = [f"backbone.layers.0.mixer.{role}.weight" for role in ("q_proj", "k_proj", "v_proj")]
    auxiliary_sources = [f"draft.model.layers.0.mixer.{role}.weight" for role in ("q_proj", "k_proj", "v_proj")] + [
        f"language_model.mtp.layers.0.mixer.{role}.weight" for role in ("q_proj", "k_proj", "v_proj")
    ]
    ext, model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={destination: torch.nn.Parameter(torch.empty(1))},
        source_names=main_sources + auxiliary_sources,
        load_weights=lambda **_kwargs: {destination},
    )
    _patch_vllm_postload(monkeypatch)

    with ext._weight_update_lifecycle("collective") as finalize:
        ext._load_full_hf_weights([(name, name) for name in main_sources])
        finalize()

    assert model.load_weights.call_count == 3


@pytest.mark.vllm
def test_main_manifest_rejects_unsupported_fused_expert_biases(monkeypatch):
    """Destination names alone cannot prove per-expert fused-bias coverage."""
    from nemo_rl.models.generation.vllm import vllm_backend

    destinations = {
        "model.layers.0.mixer.experts.routed_experts.w13_bias",
        "model.layers.0.mixer.experts.routed_experts.w2_bias",
    }
    source_names = {
        "backbone.layers.0.mixer.experts.0.up_proj.bias",
        "backbone.layers.0.mixer.experts.0.down_proj.bias",
    }
    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={name: torch.nn.Parameter(torch.empty(1)) for name in destinations},
        source_names=source_names,
        # Simulate the strongest destination-only result: both fused targets
        # were reported despite there being no per-expert acceptance proof.
        load_weights=lambda **_kwargs: destinations,
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="fused expert-bias coverage is unsupported"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([(name, name) for name in source_names])
            finalize()

    process_weights.assert_not_called()
    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
@pytest.mark.parametrize("loaded_names", [None, {"model.weight"}])
def test_quantized_nemotron_h_refit_marks_strict_coverage_unsupported(monkeypatch, caplog, loaded_names):
    from nemo_rl.models.generation.vllm import vllm_backend

    quant_config = object()
    ext, model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={"model.weight": torch.nn.Parameter(torch.empty(1))},
        source_names={"model.weight"},
        load_weights=lambda **_kwargs: loaded_names,
        model_type="nemotron_h_puzzle",
        quant_config=quant_config,
    )
    _patch_vllm_postload(monkeypatch)

    with caplog.at_level("WARNING"):
        with ext._weight_update_lifecycle("collective") as finalize:
            ext._load_full_hf_weights([("model.weight", "weight")])
            finalize()

    model.load_weights.assert_called_once()
    assert "coverage is disabled for quantized" in caplog.text


@pytest.mark.vllm
def test_zero_batch_transaction_fails_and_resets_without_strict_manifest(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend

    ext, _model = _make_strict_refit_extension(
        vllm_backend,
        destination_params={"model.weight": torch.nn.Parameter(torch.empty(1))},
        source_names={"model.weight"},
        load_weights=lambda **_kwargs: None,
        model_type="llama",
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(RuntimeError, match="empty collective weight update"):
        with ext._weight_update_lifecycle("collective") as finalize:
            finalize()

    process_weights.assert_not_called()
    assert ext._nrl_main_destination_manifest is None
    assert ext._nrl_refit_source_names is None


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_via_ipc_zmq",
        "update_weights_from_collective",
        "nccl_reshard_refit",
    ],
)
@pytest.mark.parametrize("worker_results, expected", [([True, True], True), ([True, False], False)])
def test_sync_weight_updates_check_every_internal_worker(method_name, worker_results, expected):
    """A failure on a later PP rank must not be hidden by rank zero success."""
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": False}}
    worker.llm = SimpleNamespace(collective_rpc=MagicMock(return_value=worker_results))

    assert getattr(worker, method_name)() is expected


@pytest.mark.vllm
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name",
    [
        "update_weights_via_ipc_zmq_async",
        "update_weights_from_collective_async",
        "nccl_reshard_refit_async",
    ],
)
@pytest.mark.parametrize("worker_results, expected", [([True, True], True), ([True, False], False)])
async def test_async_weight_updates_check_every_internal_worker(method_name, worker_results, expected):
    """Async refit also reports failures from every internal PP rank."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=worker_results),
        reset_encoder_cache=AsyncMock(),
    )

    assert await getattr(worker, method_name)() is expected
    if expected:
        worker.llm.reset_encoder_cache.assert_awaited_once_with()
    else:
        worker.llm.reset_encoder_cache.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_async_weight_update_skips_encoder_cache_reset_when_disabled():
    """Text-only and in-flight refit users retain the existing cache behavior."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {"vllm_cfg": {"async_engine": True}}
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(),
    )

    assert await worker.update_weights_from_collective_async() is True
    worker.llm.reset_encoder_cache.assert_not_awaited()


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_async_weight_update_fails_when_encoder_cache_reset_fails():
    """A successful refit must not resume with stale multimodal encoder outputs."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(side_effect=RuntimeError("reset failed")),
    )

    assert await worker.update_weights_from_collective_async() is False


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_nccl_reshard_refit_resets_encoder_cache():
    """NCCL-reshard refits invalidate encoder outputs just like other transports."""
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "vllm_cfg": {
            "async_engine": True,
            "reset_encoder_cache_after_weight_update": True,
        }
    }
    worker.llm = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[True]),
        reset_encoder_cache=AsyncMock(),
    )

    assert await worker.nccl_reshard_refit_async() is True
    worker.llm.reset_encoder_cache.assert_awaited_once_with()


@pytest.mark.vllm
def test_update_weights_via_ipc_acks_manifest_error_and_returns_false(monkeypatch):
    from nemo_rl.models.generation.vllm import vllm_backend
    from nemo_rl.models.policy.utils import IPCProtocol

    class FakeSocket:
        def __init__(self):
            self.sent = []

        def recv_pyobj(self):
            return IPCProtocol.COMPLETE

        def send(self, payload):
            self.sent.append(payload)

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(vllm_backend.VllmInternalWorkerExtension)
    ext.state_dict_info = {"model.weight": (torch.Size([1]), torch.float32)}
    ext.zmq_socket = FakeSocket()
    ext.maybe_init_zmq = lambda: None

    @contextlib.contextmanager
    def lifecycle(_transport):
        yield lambda: pytest.fail("an incomplete transfer must not be finalized")

    ext._weight_update_lifecycle = lifecycle

    assert ext.update_weights_via_ipc_zmq() is False
    assert ext.zmq_socket.sent == [IPCProtocol.ACK.value.encode()]


@pytest.mark.vllm
def test_read_mtp_layer_weights_from_checkpoint_filters_and_reads(tmp_path):
    """Only the requested MTP layer tensors are read, across the shards holding them."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    mtp_block = torch.randn(4, 4)
    mtp_head = torch.randn(2, 4)
    other_layer = torch.randn(4, 4)
    embed = torch.randn(8, 4)
    # MTP layer index is 2; layer 0 and the top-level embed must be ignored.
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.2.mlp.up_proj.weight": mtp_block,  # MTP, shard 1
                "model.layers.0.mlp.up_proj.weight": other_layer,  # non-MTP, same shard
            },
            "model-00002-of-00002.safetensors": {
                "model.layers.2.shared_head.head.weight": mtp_head,  # MTP, shard 2
                "model.embed_tokens.weight": embed,  # non-MTP, no layer index
            },
        },
    )

    weights = _read_mtp_layer_weights_from_checkpoint(str(model_dir), {2})

    by_name = dict(weights)
    assert set(by_name) == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.shared_head.head.weight",
    }
    assert torch.equal(by_name["model.layers.2.mlp.up_proj.weight"], mtp_block)
    assert torch.equal(by_name["model.layers.2.shared_head.head.weight"], mtp_head)


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_loads_only_mtp_layer(tmp_path, monkeypatch):
    """Success path: only MTP-layer weights are handed to the drafter, then post-loaded."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.2.mlp.up_proj.weight": torch.randn(4, 4),  # MTP
                "model.layers.2.embed_tokens.weight": torch.randn(8, 4),  # MTP
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),  # non-MTP
                "model.embed_tokens.weight": torch.randn(8, 4),  # non-MTP
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    process_weights = _patch_vllm_postload(monkeypatch)

    result = ext.load_mtp_weights_from_disk(str(model_dir))

    assert result is True
    ext._load_draft_weights.assert_called_once()
    loaded_names = {name for name, _ in ext._load_draft_weights.call_args[0][0]}
    assert loaded_names == {
        "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.embed_tokens.weight",
    }
    process_weights.assert_called_once()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_supports_nemotron_h_native_namespace(tmp_path, monkeypatch):
    """Nemotron-H's physical MTP blocks remain in the native mtp.layers namespace."""
    model_dir = tmp_path / "ckpt"
    first = torch.randn(4, 4)
    expert = torch.randn(4, 4)
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00002.safetensors": {
                "model.layers.0.mixer.weight": torch.randn(4, 4),
            },
            "model-00002-of-00002.safetensors": {
                "mtp.layers.0.eh_proj.weight": first,
                "mtp.layers.1.mixer.experts.0.up_proj.weight": expert,
            },
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*E",
            }
        )
    )
    destination_names = {
        "model.layers.0.eh_proj.weight",
        "model.layers.1.mixer.experts.w13_weight",
    }
    ext = _make_extension_with_drafter(
        mtp_start_layer_idx=52,
        num_mtp_layers=1,
        model_type="nemotron_h",
        destination_names=destination_names,
        loaded_names=destination_names,
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    assert ext.load_mtp_weights_from_disk(str(model_dir)) is True

    draft_model = ext.model_runner.drafter.model
    loaded = dict(item for call in draft_model.load_weights.call_args_list for item in call.kwargs["weights"])
    assert set(loaded) == {
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
    }
    assert torch.equal(loaded["mtp.layers.0.eh_proj.weight"], first)
    assert torch.equal(loaded["mtp.layers.1.mixer.experts.0.up_proj.weight"], expert)
    process_weights.assert_called_once()


@pytest.mark.vllm
def test_load_native_nemotron_h_mtp_rejects_silent_partial_destination_load(tmp_path, monkeypatch):
    """Source-layer presence is insufficient when vLLM silently skips a tensor."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "mtp.layers.0.eh_proj.weight": torch.randn(4, 4),
                "mtp.layers.1.mixer.experts.0.up_proj.weight": torch.randn(4, 4),
            }
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*E",
            }
        )
    )
    expected_names = {
        "model.layers.0.eh_proj.weight",
        "model.layers.1.mixer.experts.w13_weight",
        "model.layers.1.final_layernorm.weight",
    }
    ext = _make_extension_with_drafter(
        mtp_start_layer_idx=52,
        num_mtp_layers=1,
        model_type="nemotron_h",
        destination_names=expected_names,
        loaded_names=expected_names - {"model.layers.1.final_layernorm.weight"},
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="missing destination parameters.*final_layernorm"):
        ext.load_mtp_weights_from_disk(str(model_dir))

    process_weights.assert_not_called()


@pytest.mark.vllm
@pytest.mark.parametrize("missing_or_skipped", ["missing_v", "skipped_k"])
def test_native_nemotron_h_mtp_qkv_coverage_requires_every_source(tmp_path, monkeypatch, missing_or_skipped):
    """A qkv destination cannot hide a missing or silently skipped component."""
    model_dir = tmp_path / "ckpt"
    roles = ["q_proj", "k_proj"]
    if missing_or_skipped != "missing_v":
        roles.append("v_proj")
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                f"mtp.layers.0.mixer.{role}.weight": torch.randn(4, 4) for role in roles
            }
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "M",
            }
        )
    )
    destination = "model.layers.0.mixer.qkv_proj.weight"

    def loaded_names(weights):
        source_name = weights[0][0]
        if missing_or_skipped == "skipped_k" and ".k_proj." in source_name:
            return set()
        return {destination}

    ext = _make_extension_with_drafter(
        mtp_start_layer_idx=52,
        num_mtp_layers=1,
        model_type="nemotron_h",
        destination_names={destination},
        loaded_names=loaded_names,
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="missing (?:source roles|required source components)"):
        ext.load_mtp_weights_from_disk(str(model_dir))

    process_weights.assert_not_called()


@pytest.mark.vllm
def test_native_nemotron_h_mtp_coverage_requires_loader_tracking(tmp_path, monkeypatch):
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {"model-00001-of-00001.safetensors": {"mtp.layers.0.eh_proj.weight": torch.randn(4, 4)}},
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*",
            }
        )
    )
    ext = _make_extension_with_drafter(
        mtp_start_layer_idx=52,
        num_mtp_layers=1,
        model_type="nemotron_h",
        destination_names={"model.layers.0.eh_proj.weight"},
        loaded_names=None,
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    with pytest.raises(RuntimeError, match="did not report loaded destination"):
        ext.load_mtp_weights_from_disk(str(model_dir))

    process_weights.assert_not_called()


@pytest.mark.vllm
def test_nemotron_h_native_mtp_reader_rejects_missing_physical_block(tmp_path):
    """The native namespace must cover every block in depth × hybrid pattern."""
    import json

    from nemo_rl.models.generation.vllm.vllm_backend import (
        _read_mtp_layer_weights_from_checkpoint,
    )

    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "mtp.layers.0.eh_proj.weight": torch.randn(4, 4),
            }
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*E",
            }
        )
    )

    with pytest.raises(ValueError, match="expected physical layers.*0, 1"):
        _read_mtp_layer_weights_from_checkpoint(str(model_dir), {52})


@pytest.mark.vllm
@pytest.mark.parametrize("is_last_rank", [False, True])
def test_load_mtp_weights_from_disk_without_drafter(tmp_path, monkeypatch, is_last_rank):
    """Only the pipeline stage that owns the drafter requires it to exist."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    ext.model_runner = MagicMock()
    ext.model_runner.drafter = None
    ext._load_draft_weights = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_backend.get_pp_group",
        lambda: SimpleNamespace(is_last_rank=is_last_rank),
    )

    if is_last_rank:
        with pytest.raises(RuntimeError, match="drafter model is unavailable"):
            ext.load_mtp_weights_from_disk(str(tmp_path))
    else:
        assert ext.load_mtp_weights_from_disk(str(tmp_path)) is False
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_mtp_weights_from_disk_raises_when_mtp_weights_missing(tmp_path, monkeypatch):
    """A checkpoint without the MTP layer(s) fails loudly instead of silently."""
    model_dir = tmp_path / "ckpt"
    _write_sharded_checkpoint(
        model_dir,
        {
            "model-00001-of-00001.safetensors": {
                "model.layers.0.mlp.up_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(8, 4),
            }
        },
    )
    ext = _make_extension_with_drafter(mtp_start_layer_idx=2, num_mtp_layers=1)
    _patch_vllm_postload(monkeypatch)

    with pytest.raises(ValueError, match="No MTP layer weights"):
        ext.load_mtp_weights_from_disk(str(model_dir))
    ext._load_draft_weights.assert_not_called()


@pytest.mark.vllm
def test_load_weights_routes_only_policy_weights_to_mtp_drafter(monkeypatch):
    """The MTP path receives policy weights, while Eagle gets draft-prefixed ones."""
    from nemo_rl.models.generation.vllm.quantization import fp8
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    main_model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner = SimpleNamespace(
        model=main_model,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[])),
    )
    ext._load_draft_weights = MagicMock()
    ext._maybe_refit_mtp_drafter = MagicMock()
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _: False)

    policy_weights = [("model.weight", "policy-value")]
    draft_weights = [("weight", "draft-value")]
    ext._load_weights(policy_weights + [("draft.weight", "draft-value")])

    main_model.load_weights.assert_called_once_with(weights=policy_weights)
    ext._load_draft_weights.assert_called_once_with(draft_weights)
    ext._maybe_refit_mtp_drafter.assert_called_once_with(policy_weights)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method, from_disk, has_drafter, expected",
    [
        ("mtp", False, True, True),  # co-trained MTP drafter refit from policy stream
        ("deepseek_mtp", False, True, True),  # same, DeepSeek naming
        ("mtp", True, True, False),  # served once from disk -> leave static weights
        ("eagle3", False, True, False),  # non-MTP drafter uses the draft. prefix path
        (None, False, True, False),  # speculative decoding disabled
        ("mtp", False, False, False),  # vLLM built no drafter
    ],
)
def test_mtp_drafter_refit_enabled(method, from_disk, has_drafter, expected):
    """The refit-into-drafter path only fires for a co-trained MTP drafter."""
    ext, _ = _make_mtp_refit_extension(method=method, from_disk=from_disk, has_drafter=has_drafter)
    assert ext._mtp_drafter_refit_enabled() is expected


@pytest.mark.vllm
def test_maybe_refit_mtp_drafter_loads_when_enabled():
    """A co-trained MTP drafter is fed the (vocab-trimmed) policy weights on refit."""
    ext, drafter_model = _make_mtp_refit_extension(method="mtp", from_disk=False)
    weights = [("mtp.layers.0.weight", "w0")]
    trimmed = [("mtp.layers.0.weight", "trimmed")]
    # Isolate from _trim_vocab_padding, which needs a real vLLM module tree.
    ext._trim_vocab_padding = MagicMock(return_value=trimmed)

    ext._maybe_refit_mtp_drafter(weights)

    ext._trim_vocab_padding.assert_called_once_with(drafter_model, weights)
    drafter_model.load_weights.assert_called_once_with(weights=trimmed)


@pytest.mark.vllm
@pytest.mark.parametrize(
    "method, from_disk",
    [
        ("mtp", True),  # disk-served MTP drafter must not be reloaded on refit
        ("eagle3", False),  # non-MTP drafter is handled elsewhere
    ],
)
def test_maybe_refit_mtp_drafter_noop_when_gated(method, from_disk):
    """The drafter is left untouched for the disk-load path and non-MTP drafters."""
    ext, drafter_model = _make_mtp_refit_extension(method=method, from_disk=from_disk)
    ext._trim_vocab_padding = MagicMock()

    ext._maybe_refit_mtp_drafter([("mtp.layers.0.weight", "w0")])

    ext._trim_vocab_padding.assert_not_called()
    drafter_model.load_weights.assert_not_called()


@pytest.mark.vllm
def test_maybe_process_mtp_drafter_after_loading_when_enabled(monkeypatch):
    """The refit MTP drafter is finalized against its own draft_model_config."""
    draft_model_config = object()
    ext, drafter_model = _make_mtp_refit_extension(method="mtp", from_disk=False, draft_model_config=draft_model_config)
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_called_once_with(drafter_model, draft_model_config, ext.device)


@pytest.mark.vllm
def test_maybe_process_mtp_drafter_after_loading_noop_when_disk_loaded(monkeypatch):
    """The disk-load path already finalized its weights, so refit skips reprocessing."""
    ext, _ = _make_mtp_refit_extension(method="mtp", from_disk=True)
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_not_called()
