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
    ext = backend.VllmInternalWorkerExtension.__new__(
        backend.VllmInternalWorkerExtension
    )
    state_info = object()
    ext.state_dict_info = {"model.weight": state_info}
    ext.model_update_group = object()
    ext.model_runner = SimpleNamespace(model=torch.nn.Module(), vllm_config=object())
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


def _make_extension_with_drafter(mtp_start_layer_idx, num_mtp_layers):
    """Build a VllmInternalWorkerExtension with a mocked drafter and stubbed refit."""
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    ext.device = torch.device("cpu")
    predictor = SimpleNamespace(
        mtp_start_layer_idx=mtp_start_layer_idx, num_mtp_layers=num_mtp_layers
    )
    ext.model_runner = MagicMock()
    ext.model_runner.drafter.model = SimpleNamespace(model=predictor)
    # Isolate this test from _load_draft_weights internals.
    ext._load_draft_weights = MagicMock()
    return ext


def _patch_vllm_postload(monkeypatch):
    """Stub the vLLM post-load helpers imported inside load_mtp_weights_from_disk."""
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config", lambda cfg: contextlib.nullcontext()
    )
    process_weights = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights,
    )
    return process_weights


def _make_padded_routed_expert_reload_fixture(monkeypatch):
    """Build a real vLLM reload root with distinct checkpoint/kernel layouts."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        record_metadata_for_reloading,
    )

    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.local_num_experts = 2
    layer.moe_config = SimpleNamespace(
        has_bias=False,
        is_act_and_mul=False,
        intermediate_size_per_partition_unpadded=3,
        hidden_dim_unpadded=4,
    )
    quant_method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    torch.nn.Module.__init__(quant_method)
    layer.quant_method = quant_method

    def expert_weight_loader(
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        del weight_name
        if shard_id == "w1":
            destination = param.data[expert_id]
        elif shard_id == "w2":
            destination = param.data[expert_id]
        else:
            raise AssertionError(f"unexpected shard: {shard_id}")
        destination.copy_(loaded_weight)
        return True if return_success else None

    for name, shape in (("w13_weight", (2, 3, 4)), ("w2_weight", (2, 4, 3))):
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        param.weight_loader = expert_weight_loader
        layer.register_parameter(name, param)

    # vLLM records these canonical 3-D shapes immediately after construction,
    # before its initial checkpoint load and kernel post-processing.
    record_metadata_for_reloading(layer)

    process_calls = []

    def process_weights_after_loading(_quant_method, routed_experts):
        process_calls.append(routed_experts)
        canonical_w13 = routed_experts.w13_weight.detach()
        canonical_w2 = routed_experts.w2_weight.detach()
        padded_w13 = torch.cat(
            (canonical_w13, torch.zeros_like(canonical_w13[:, :1])), dim=1
        ).reshape(2, 1, 4, 4)
        padded_w2 = torch.cat(
            (canonical_w2, torch.zeros_like(canonical_w2[..., :1])), dim=-1
        ).reshape(2, 4, 2, 2)
        for name, value in (("w13_weight", padded_w13), ("w2_weight", padded_w2)):
            old_param = getattr(routed_experts, name)
            new_param = torch.nn.Parameter(value, requires_grad=False)
            new_param.weight_loader = old_param.weight_loader
            setattr(routed_experts, name, new_param)

    monkeypatch.setattr(
        UnquantizedFusedMoEMethod,
        "process_weights_after_loading",
        process_weights_after_loading,
    )
    # Emulate the cold-load conversion from canonical 3-D tensors to a padded
    # 4-D FlashInfer/TRTLLM runtime layout.
    layer.quant_method.process_weights_after_loading(layer)
    return layer, process_calls


def _make_mtp_refit_extension(
    *, method="mtp", from_disk=False, has_drafter=True, draft_model_config=None
):
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

    spec_config = (
        None
        if method is None
        else SimpleNamespace(method=method, draft_model_config=draft_model_config)
    )
    drafter_model = SimpleNamespace(load_weights=MagicMock()) if has_drafter else None
    ext.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec_config),
        drafter=SimpleNamespace(model=drafter_model) if has_drafter else None,
    )
    return ext, drafter_model


@pytest.mark.vllm
def test_routed_expert_reload_uses_canonical_metadata_and_preserves_kernel_storage(
    monkeypatch,
):
    """A padded 4-D cold kernel must accept 2-D per-expert refit shards safely."""
    from nemo_rl.models.generation.vllm import vllm_backend
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    layer, process_calls = _make_padded_routed_expert_reload_fixture(monkeypatch)
    extension, roots = _make_routed_expert_lifecycle_extension(vllm_backend, layer)
    assert [root.label for root in roots] == ["main.experts"]
    root = roots[0]
    vllm_backend._require_routed_expert_reload_metadata(root)
    legacy_quant_methods = []
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda _model, _config, _device: legacy_quant_methods.append(
            layer.quant_method
        ),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    runtime_params = {
        name: getattr(layer, name) for name in ("w13_weight", "w2_weight")
    }
    runtime_ptrs = {
        name: param.untyped_storage().data_ptr()
        for name, param in runtime_params.items()
    }
    assert tuple(runtime_params["w13_weight"].shape) == (2, 1, 4, 4)
    assert tuple(runtime_params["w2_weight"].shape) == (2, 4, 2, 2)

    expected_w13 = torch.zeros(2, 3, 4)
    expected_w2 = torch.zeros(2, 4, 3)

    def load(name, tensor, shard_id, expert_id):
        param = getattr(layer, name)
        param.weight_loader(
            param,
            tensor,
            f"experts.{expert_id}.{shard_id}.weight",
            shard_id,
            expert_id,
        )

    with extension._weight_update_lifecycle("ipc") as finish:
        info = get_layerwise_info(layer)
        restore_params, _ = info.restore_metadata
        assert tuple(restore_params["w13_weight"].shape) == (2, 3, 4)
        assert tuple(restore_params["w2_weight"].shape) == (2, 4, 3)
        assert info.load_numel_total == 48

        # Split the first root across an IPC-style batch boundary. The deferred
        # BoundArguments must own a clone before the sender reuses this storage.
        up0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected_w13[0].copy_(up0)
        load("w13_weight", up0, "w1", 0)
        vllm_backend._detach_pending_layerwise_weights(
            (layer,), {up0.untyped_storage().data_ptr()}
        )
        deferred_up0 = info.loaded_weights[0][1].arguments["loaded_weight"]
        assert (
            deferred_up0.untyped_storage().data_ptr()
            != up0.untyped_storage().data_ptr()
        )
        up0.fill_(-999)

        values = [
            ("w2_weight", torch.full((4, 3), 30.0), "w2", 0),
            ("w13_weight", torch.full((3, 4), 40.0), "w1", 1),
            ("w2_weight", torch.full((4, 3), 60.0), "w2", 1),
        ]
        for name, tensor, shard_id, expert_id in values:
            if shard_id == "w1":
                expected_w13[expert_id].copy_(tensor)
            else:
                expected_w2[expert_id].copy_(tensor)
            load(name, tensor, shard_id, expert_id)

        assert info.can_load() is False
        finish()

    # Stock vLLM reaches the exact canonical total (unpadded 464 in the real
    # model), converts once, copies into cold kernel storage, and resets itself.
    assert info.can_load() is False
    assert len(process_calls) == 2
    assert legacy_quant_methods == [None]
    assert extension._weight_update_errors_are_fatal() is False
    for name, param in runtime_params.items():
        assert getattr(layer, name) is param
        assert param.untyped_storage().data_ptr() == runtime_ptrs[name]

    expected_padded_w13 = torch.cat(
        (expected_w13, torch.zeros_like(expected_w13[:, :1])), dim=1
    ).reshape(2, 1, 4, 4)
    expected_padded_w2 = torch.cat(
        (expected_w2, torch.zeros_like(expected_w2[..., :1])), dim=-1
    ).reshape(2, 4, 2, 2)
    torch.testing.assert_close(layer.w13_weight, expected_padded_w13)
    torch.testing.assert_close(layer.w2_weight, expected_padded_w2)

    assert layer.quant_method is not None


def _make_routed_expert_lifecycle_extension(backend, layer):
    model = torch.nn.Module()
    model.add_module("experts", layer)
    model_config = SimpleNamespace(quantization=None, dtype=torch.float32)
    vllm_config = object()
    extension = backend.VllmInternalWorkerExtension.__new__(
        backend.VllmInternalWorkerExtension
    )
    extension.device = torch.device("cpu")
    extension.model_config = model_config
    extension.model_runner = SimpleNamespace(model=model, vllm_config=vllm_config)
    extension._nrl_active_routed_expert_reload_roots = ()
    extension._nrl_routed_expert_refit_fatal = False
    extension._maybe_process_mtp_drafter_after_loading = MagicMock()
    extension._maybe_process_fp8_kv_cache = MagicMock()
    roots = tuple(
        backend._find_unquantized_routed_expert_roots(
            model,
            owner="main",
            model_config=model_config,
        )
    )
    extension._get_routed_expert_reload_roots = MagicMock(return_value=roots)
    return extension, roots


@pytest.mark.vllm
def test_routed_expert_roots_include_mtp_drafter_but_not_mamba(monkeypatch):
    """Only main/draft expert owners enter reload; ordinary Mamba state does not."""
    from nemo_rl.models.generation.vllm import vllm_backend

    main_experts, _ = _make_padded_routed_expert_reload_fixture(monkeypatch)
    draft_experts, _ = _make_padded_routed_expert_reload_fixture(monkeypatch)
    main_model = torch.nn.Module()
    main_model.add_module("experts", main_experts)
    main_model.add_module("mamba_conv", torch.nn.Conv1d(2, 2, 3, groups=2))
    draft_model = torch.nn.Module()
    draft_model.add_module("experts", draft_experts)
    draft_model_config = object()
    extension = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    extension.model_config = object()
    extension._mtp_drafter_from_disk = False
    extension.model_runner = SimpleNamespace(
        model=main_model,
        drafter=SimpleNamespace(model=draft_model),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="mtp", draft_model_config=draft_model_config
            )
        ),
    )
    conv_parameter = main_model.mamba_conv.weight

    roots = extension._get_routed_expert_reload_roots()

    assert [(root.owner, root.label) for root in roots] == [
        ("main", "main.experts"),
        ("mtp_drafter", "mtp_drafter.experts"),
    ]
    assert [root.module for root in roots] == [main_experts, draft_experts]
    assert roots[1].model_config is draft_model_config
    assert main_model.mamba_conv.weight is conv_parameter


@pytest.mark.vllm
def test_routed_expert_lifecycle_rejects_partial_root_and_stays_fatal(monkeypatch):
    """COMPLETE cannot process a partially loaded root or leave it recoverable."""
    from nemo_rl.models.generation.vllm import vllm_backend
    from vllm.model_executor.model_loader import reload as reload_module

    layer, _ = _make_padded_routed_expert_reload_fixture(monkeypatch)
    extension, _roots = _make_routed_expert_lifecycle_extension(vllm_backend, layer)
    native_finalize = MagicMock()
    monkeypatch.setattr(reload_module, "finalize_layerwise_reload", native_finalize)
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )

    with pytest.raises(RuntimeError, match="layerwise refit is incomplete"):
        with extension._weight_update_lifecycle("collective") as finish:
            param = layer.w13_weight
            param.weight_loader(
                param,
                torch.ones(3, 4),
                "experts.0.up_proj.weight",
                "w1",
                0,
            )
            finish()

    native_finalize.assert_not_called()
    assert extension._weight_update_errors_are_fatal() is True
    assert extension._nrl_active_routed_expert_reload_roots == ()
    with pytest.raises(RuntimeError, match="poisoned by an earlier"):
        with extension._weight_update_lifecycle("nccl_reshard"):
            pytest.fail("a poisoned worker must reject every later transport")


@pytest.mark.vllm
def test_routed_expert_lifecycle_keeps_poison_when_fp8_postpass_fails(monkeypatch):
    """A failure after COMPLETE must still prevent the worker from serving."""
    from nemo_rl.models.generation.vllm import vllm_backend
    from vllm.model_executor.model_loader import reload as reload_module

    layer, _ = _make_padded_routed_expert_reload_fixture(monkeypatch)
    extension, _roots = _make_routed_expert_lifecycle_extension(vllm_backend, layer)
    extension._maybe_process_fp8_kv_cache.side_effect = RuntimeError("kv failed")
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        reload_module, "initialize_layerwise_reload", lambda _root: None
    )
    monkeypatch.setattr(
        reload_module,
        "finalize_layerwise_reload",
        lambda _root, _config: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        lambda _model, _config, _device: None,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="kv failed"):
        with extension._weight_update_lifecycle("ipc") as finish:
            finish()

    assert extension._weight_update_errors_are_fatal() is True
    assert extension._nrl_active_routed_expert_reload_roots == ()


@pytest.mark.vllm
def test_nccl_reshard_does_not_enter_routed_expert_layerwise_reload(monkeypatch):
    """The bulk in-place transport keeps its existing non-layerwise lifecycle."""
    from nemo_rl.models.generation.vllm import vllm_backend

    extension = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    extension.device = torch.device("cpu")
    extension.model_config = SimpleNamespace(quantization=None, dtype=torch.float32)
    extension.model_runner = SimpleNamespace(
        model=torch.nn.Module(), vllm_config=object()
    )
    extension._nrl_routed_expert_refit_fatal = False
    extension._get_routed_expert_reload_roots = MagicMock()
    extension._maybe_process_mtp_drafter_after_loading = MagicMock()
    extension._maybe_process_fp8_kv_cache = MagicMock()
    process_weights = MagicMock()
    monkeypatch.setattr(
        "vllm.config.set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.process_weights_after_loading",
        process_weights,
    )

    with extension._weight_update_lifecycle("nccl_reshard") as finish:
        finish()

    extension._get_routed_expert_reload_roots.assert_not_called()
    process_weights.assert_called_once_with(
        extension.model_runner.model,
        extension.model_config,
        extension.device,
    )
    extension._maybe_process_fp8_kv_cache.assert_called_once_with()


@pytest.mark.vllm
@pytest.mark.parametrize("with_mtp", [False, True])
def test_update_weights_from_collective_processes_weights_after_loading(
    monkeypatch, with_mtp
):
    from nemo_rl.models.generation.vllm import vllm_backend

    call_order = []
    process_calls = []
    draft_model = torch.nn.Module() if with_mtp else None
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
            speculative_config=SimpleNamespace(
                method="mtp", draft_model_config=draft_model_config
            )
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

    def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
        call_order.append("broadcast")
        assert list(iterator) == [("model.weight", expected_state_info)]
        assert group is ext.model_update_group
        assert src == 0
        post_unpack_func([("model.weight", "weight-value")])

    ext._load_weights = load_weights
    ext._maybe_process_fp8_kv_cache = lambda: call_order.append("kv")
    monkeypatch.setattr(
        vllm_backend, "packed_broadcast_consumer", packed_broadcast_consumer
    )
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
@pytest.mark.parametrize(
    "method_name",
    ["update_weights_via_ipc_zmq", "update_weights_from_collective"],
)
@pytest.mark.parametrize(
    "worker_results, expected", [([True, True], True), ([True, False], False)]
)
def test_sync_weight_updates_check_every_internal_worker(
    method_name, worker_results, expected
):
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
    ["update_weights_via_ipc_zmq_async", "update_weights_from_collective_async"],
)
@pytest.mark.parametrize(
    "worker_results, expected", [([True, True], True), ([True, False], False)]
)
async def test_async_weight_updates_check_every_internal_worker(
    method_name, worker_results, expected
):
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

    ext = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
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
@pytest.mark.parametrize("is_last_rank", [False, True])
def test_load_mtp_weights_from_disk_without_drafter(
    tmp_path, monkeypatch, is_last_rank
):
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
def test_load_mtp_weights_from_disk_raises_when_mtp_weights_missing(
    tmp_path, monkeypatch
):
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
    """Native MTP weights bypass the main loader and reach only the drafter."""
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

    main_weights = [("model.weight", "policy-value")]
    native_mtp_weights = [
        ("mtp.layers.0.mixer.experts.0.up_proj.weight", "mtp-value"),
        ("module.mtp.norm.weight", "prefixed-mtp-value"),
    ]
    policy_weights = main_weights + native_mtp_weights
    draft_weights = [("weight", "draft-value")]
    ext._load_weights(policy_weights + [("draft.weight", "draft-value")])

    main_model.load_weights.assert_called_once_with(weights=main_weights)
    ext._load_draft_weights.assert_called_once_with(draft_weights)
    ext._maybe_refit_mtp_drafter.assert_called_once_with(policy_weights)


@pytest.mark.vllm
def test_load_weights_with_only_native_mtp_weights_skips_main_loader(monkeypatch):
    """An MTP-only IPC bucket must not invoke the main-model autoloader."""
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

    native_mtp_weights = [("mtp.layers.1.weight", "mtp-value")]
    ext._load_weights(native_mtp_weights)

    main_model.load_weights.assert_not_called()
    # The legacy dispatcher calls the Eagle3 loader for every bucket; the
    # production loader treats an empty list as a no-op.
    ext._load_draft_weights.assert_called_once_with([])
    ext._maybe_refit_mtp_drafter.assert_called_once_with(native_mtp_weights)


@pytest.mark.vllm
def test_load_weights_fences_detached_accelerator_clone_before_return(monkeypatch):
    """An alternating collective stream cannot consume an unfinished clone."""
    from nemo_rl.models.generation.vllm import vllm_backend

    extension = vllm_backend.VllmInternalWorkerExtension.__new__(
        vllm_backend.VllmInternalWorkerExtension
    )
    extension.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=[]))
    )
    extension._nrl_active_routed_expert_reload_roots = (
        SimpleNamespace(module=object()),
    )
    events = []
    extension._load_hf_weights = lambda _weights: events.append("load")
    extension._load_draft_weights = lambda _weights: None
    extension._maybe_refit_mtp_drafter = lambda _weights: None
    monkeypatch.setattr(
        vllm_backend,
        "_detach_pending_layerwise_weights",
        lambda _roots, _ptrs: events.append("clone") or True,
    )
    monkeypatch.setattr(
        vllm_backend,
        "_require_bounded_routed_expert_retention",
        lambda _roots: events.append("bounded"),
    )
    monkeypatch.setattr(
        vllm_backend.torch.accelerator,
        "synchronize",
        lambda: events.append("sync"),
    )

    extension._load_weights([("model.weight", torch.ones(1))])

    assert events == ["load", "clone", "sync", "bounded"]


@pytest.mark.vllm
def test_main_loader_rejects_native_mtp_namespace_before_load(monkeypatch):
    """Direct callers cannot bypass the MTP/main ownership split."""
    from nemo_rl.models.generation.vllm.quantization import fp8
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    main_model = SimpleNamespace(load_weights=MagicMock())
    ext.model_runner = SimpleNamespace(model=main_model, vllm_config=object())
    monkeypatch.setattr(fp8, "is_fp8_model", lambda _: False)

    with pytest.raises(RuntimeError, match="native MTP weights reached.*main-model"):
        ext._load_hf_weights([("language_model.mtp.norm.weight", "mtp-value")])

    main_model.load_weights.assert_not_called()


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
    ext, _ = _make_mtp_refit_extension(
        method=method, from_disk=from_disk, has_drafter=has_drafter
    )
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
    ext, drafter_model = _make_mtp_refit_extension(
        method="mtp", from_disk=False, draft_model_config=draft_model_config
    )
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_called_once_with(
        drafter_model, draft_model_config, ext.device
    )


@pytest.mark.vllm
def test_maybe_process_mtp_drafter_after_loading_noop_when_disk_loaded(monkeypatch):
    """The disk-load path already finalized its weights, so refit skips reprocessing."""
    ext, _ = _make_mtp_refit_extension(method="mtp", from_disk=True)
    process_weights = _patch_vllm_postload(monkeypatch)

    ext._maybe_process_mtp_drafter_after_loading()

    process_weights.assert_not_called()
