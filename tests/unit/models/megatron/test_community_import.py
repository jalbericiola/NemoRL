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

"""Unit tests for community_import checkpoint-save strategy shim."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _ensure_package(monkeypatch, name: str) -> ModuleType:
    """Create a minimal package module in sys.modules for import stubbing."""
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent_module = _ensure_package(monkeypatch, parent_name)
        setattr(parent_module, child_name, module)

    return module


def _load_community_import_module(monkeypatch):
    """Import community_import with lightweight dependency stubs."""
    # Stub torch symbols used at import time/type annotations.
    fake_torch = ModuleType("torch")
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.float32 = object()
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Stub megatron imports required by the module.
    _ensure_package(monkeypatch, "megatron")
    _ensure_package(monkeypatch, "megatron.core")
    megatron_bridge = ModuleType("megatron.bridge")
    megatron_bridge.AutoBridge = type("AutoBridge", (), {})
    monkeypatch.setitem(sys.modules, "megatron.bridge", megatron_bridge)
    megatron_transformer = ModuleType("megatron.core.transformer")
    megatron_transformer.ModuleSpec = type("ModuleSpec", (), {})
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", megatron_transformer)

    # Stub MegatronConfig type import.
    nemo_policy = ModuleType("nemo_rl.models.policy")
    nemo_policy.MegatronConfig = dict
    nemo_policy.SHARED_PREFIX_DETERMINISM_MODEL_OVERRIDE_VALUES = {
        "deterministic_mode": True,
        "cross_entropy_loss_fusion": False,
        "tp_comm_overlap": False,
    }
    monkeypatch.setitem(sys.modules, "nemo_rl.models.policy", nemo_policy)

    module_name = "nemo_rl.models.megatron.community_import"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _install_torch_strategy_module(monkeypatch, strategy_cls):
    """Install a fake dist-checkpoint strategy module for local import."""
    _ensure_package(monkeypatch, "megatron")
    _ensure_package(monkeypatch, "megatron.core")
    _ensure_package(monkeypatch, "megatron.core.dist_checkpointing")
    _ensure_package(monkeypatch, "megatron.core.dist_checkpointing.strategies")
    strategy_module = ModuleType("megatron.core.dist_checkpointing.strategies.torch")
    strategy_module.TorchDistSaveShardedStrategy = strategy_cls
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.dist_checkpointing.strategies.torch",
        strategy_module,
    )


def _install_runtime_stubs_for_hf_import(monkeypatch):
    """Install minimal megatron-core stubs needed by import_model_from_hf_name."""
    core_module = _ensure_package(monkeypatch, "megatron.core")

    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.model_parallel_is_initialized = lambda: False
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    core_module.parallel_state = parallel_state

    rerun_state_machine = ModuleType("megatron.core.rerun_state_machine")
    rerun_state_machine.destroy_rerun_state_machine = lambda: None
    monkeypatch.setitem(
        sys.modules, "megatron.core.rerun_state_machine", rerun_state_machine
    )
    core_module.rerun_state_machine = rerun_state_machine

    tensor_parallel = ModuleType("megatron.core.tensor_parallel")
    tensor_parallel.model_parallel_cuda_manual_seed = lambda seed: None
    tensor_parallel_random = ModuleType("megatron.core.tensor_parallel.random")
    tensor_parallel_random._CUDA_RNG_STATE_TRACKER = "stale"
    tensor_parallel_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = True
    tensor_parallel.random = tensor_parallel_random
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(
        sys.modules, "megatron.core.tensor_parallel.random", tensor_parallel_random
    )
    core_module.tensor_parallel = tensor_parallel


def test_prefer_nvrx_is_noop_when_strategy_import_fails(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    # Force this import to fail even if real megatron modules were preloaded by
    # earlier tests in the same process.
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.strategies.torch", None
    )

    # Should not raise when dist-checkpoint strategy is unavailable.
    with module._prefer_nvrx_for_dist_ckpt_save():
        pass


def test_deterministic_model_overrides_apply_before_provider_construction(
    monkeypatch,
):
    module = _load_community_import_module(monkeypatch)
    provider = SimpleNamespace(
        deterministic_mode=False,
        cross_entropy_loss_fusion=True,
        tp_comm_overlap=True,
    )

    original_values = module._apply_deterministic_model_overrides(
        provider,
        {
            "model_overrides": {
                "deterministic_mode": True,
                "cross_entropy_loss_fusion": False,
                "tp_comm_overlap": False,
            }
        },
    )

    assert provider.deterministic_mode is True
    assert provider.cross_entropy_loss_fusion is False
    assert provider.tp_comm_overlap is False
    assert original_values == {
        "deterministic_mode": False,
        "cross_entropy_loss_fusion": True,
        "tp_comm_overlap": True,
    }


def test_deterministic_model_override_rejects_unsupported_provider(monkeypatch):
    module = _load_community_import_module(monkeypatch)

    with pytest.raises(ValueError, match="model_overrides.deterministic_mode"):
        module._apply_deterministic_model_overrides(
            SimpleNamespace(),
            {"model_overrides": {"deterministic_mode": True}},
        )


def test_prefer_nvrx_uses_async_save_and_restores_original_save(monkeypatch):
    module = _load_community_import_module(monkeypatch)

    class FakeAsyncRequest:
        def __init__(self, owner):
            self.owner = owner

        def execute_sync(self):
            self.owner.execute_sync_calls += 1

    class FakeStrategy:
        def __init__(self):
            self.original_save_calls = []
            self.async_save_calls = []
            self.execute_sync_calls = 0

        def save(self, sharded_state_dict, checkpoint_dir):
            self.original_save_calls.append((sharded_state_dict, checkpoint_dir))

        def async_save(self, sharded_state_dict, checkpoint_dir, async_strategy):
            self.async_save_calls.append(
                (sharded_state_dict, checkpoint_dir, async_strategy)
            )
            return FakeAsyncRequest(self)

    _install_torch_strategy_module(monkeypatch, FakeStrategy)
    strategy = FakeStrategy()
    original_save = FakeStrategy.save

    with module._prefer_nvrx_for_dist_ckpt_save():
        strategy.save({"x": 1}, "/tmp/ckpt")
        assert FakeStrategy.save is not original_save

    assert strategy.async_save_calls == [({"x": 1}, "/tmp/ckpt", "nvrx")]
    assert strategy.execute_sync_calls == 1
    assert strategy.original_save_calls == []
    assert FakeStrategy.save is original_save


def test_prefer_nvrx_falls_back_to_original_save_when_nvrx_missing(monkeypatch):
    module = _load_community_import_module(monkeypatch)

    class FakeStrategy:
        def __init__(self):
            self.original_save_calls = []
            self.async_save_calls = []

        def save(self, sharded_state_dict, checkpoint_dir):
            self.original_save_calls.append((sharded_state_dict, checkpoint_dir))

        def async_save(self, sharded_state_dict, checkpoint_dir, async_strategy):
            self.async_save_calls.append(
                (sharded_state_dict, checkpoint_dir, async_strategy)
            )
            raise ModuleNotFoundError("nvrx is unavailable")

    _install_torch_strategy_module(monkeypatch, FakeStrategy)
    strategy = FakeStrategy()

    with module._prefer_nvrx_for_dist_ckpt_save():
        strategy.save({"y": 2}, "/tmp/ckpt")

    assert strategy.async_save_calls == [({"y": 2}, "/tmp/ckpt", "nvrx")]
    assert strategy.original_save_calls == [({"y": 2}, "/tmp/ckpt")]


def test_import_model_from_hf_name_calls_bridge_save(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    _install_runtime_stubs_for_hf_import(monkeypatch)
    # Force this import path to stay unavailable even if real megatron modules
    # were preloaded by earlier tests in the same process.
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.strategies.torch", None
    )

    class FakeProvider:
        def __init__(self):
            self.tensor_model_parallel_size = 1
            self.pipeline_model_parallel_size = 1
            self.context_parallel_size = 1
            self.expert_model_parallel_size = 1
            self.expert_tensor_parallel_size = 1
            self.num_layers_in_first_pipeline_stage = None
            self.num_layers_in_last_pipeline_stage = None
            self.pipeline_dtype = "fp32"

        def finalize(self):
            pass

        def initialize_model_parallel(self, seed):
            self.seed = seed

        def provide_distributed_model(self, wrap_with_ddp, post_wrap_hook):
            config = SimpleNamespace()
            return [SimpleNamespace(config=config)]

    class FakeBridge:
        def __init__(self):
            self.provider = FakeProvider()
            self.saved_model = None
            self.saved_path = None

        def to_megatron_provider(self, load_weights):
            assert load_weights is True
            return self.provider

        def save_megatron_model(self, megatron_model, output_path):
            self.saved_model = megatron_model
            self.saved_path = output_path

    fake_bridge = FakeBridge()

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(hf_model_name, *args, **kwargs):
            # Keep this test focused on bridge-save flow, not HF API defaults.
            assert hf_model_name == "fake/hf-model"
            return fake_bridge

    monkeypatch.setattr(module, "AutoBridge", FakeAutoBridge)

    module.import_model_from_hf_name("fake/hf-model", "/tmp/out")

    assert fake_bridge.saved_model is not None
    assert fake_bridge.saved_path == "/tmp/out"


def test_hf_import_does_not_persist_deterministic_execution_overrides(monkeypatch):
    module = _load_community_import_module(monkeypatch)
    _install_runtime_stubs_for_hf_import(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.strategies.torch", None
    )

    class FakeProvider:
        def __init__(self):
            self.deterministic_mode = False
            self.cross_entropy_loss_fusion = True
            self.tp_comm_overlap = True
            self.tensor_model_parallel_size = 1
            self.pipeline_model_parallel_size = 1
            self.context_parallel_size = 1
            self.expert_model_parallel_size = 1
            self.expert_tensor_parallel_size = 1
            self.num_layers_in_first_pipeline_stage = None
            self.num_layers_in_last_pipeline_stage = None
            self.pipeline_dtype = "fp32"
            self.execution_values = None

        def finalize(self):
            pass

        def initialize_model_parallel(self, seed):
            self.seed = seed

        def provide_distributed_model(self, wrap_with_ddp, post_wrap_hook):
            self.execution_values = (
                self.deterministic_mode,
                self.cross_entropy_loss_fusion,
                self.tp_comm_overlap,
            )
            config = SimpleNamespace(
                deterministic_mode=self.deterministic_mode,
                cross_entropy_loss_fusion=self.cross_entropy_loss_fusion,
                tp_comm_overlap=self.tp_comm_overlap,
            )
            return [SimpleNamespace(config=config)]

    class FakeBridge:
        def __init__(self):
            self.provider = FakeProvider()
            self.saved_config = None

        def to_megatron_provider(self, load_weights):
            assert load_weights is True
            return self.provider

        def save_megatron_model(self, megatron_model, output_path):
            self.saved_config = megatron_model[0].config

    fake_bridge = FakeBridge()

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(hf_model_name, *args, **kwargs):
            return fake_bridge

    monkeypatch.setattr(module, "AutoBridge", FakeAutoBridge)
    megatron_config = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "pipeline_dtype": "float32",
        "sequence_parallel": False,
        "gradient_accumulation_fusion": False,
        "model_overrides": {
            "deterministic_mode": True,
            "cross_entropy_loss_fusion": False,
            "tp_comm_overlap": False,
        },
    }

    module.import_model_from_hf_name(
        "fake/hf-model",
        "/tmp/out",
        megatron_config,
    )

    assert fake_bridge.provider.execution_values == (True, False, False)
    assert fake_bridge.saved_config.deterministic_mode is False
    assert fake_bridge.saved_config.cross_entropy_loss_fusion is True
    assert fake_bridge.saved_config.tp_comm_overlap is True
