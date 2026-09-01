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
import gc
import logging
import re
import socket
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import torch
import zmq
from torch.distributed.tensor.placement_types import Shard

from nemo_rl.models.generation.vllm.checkpoint_engine import (
    VllmCheckpointEngineMixin,
    preinit_nixl_from_vllm_config,
    resolve_rollout_rank,
)
from nemo_rl.models.generation.vllm.refit_layout import (
    parse_hf_expert_weight,
    resolve_vllm_expert_parameter_name,
)
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    _STR_TO_DTYPE,
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
    _extract_layer_prefix,
)

logger = logging.getLogger(__name__)

try:
    import vllm  # noqa: F401
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_worker import Worker as VllmWorker
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


WeightUpdateTransport = Literal["ipc", "collective", "nccl_reshard", "checkpoint_engine"]
WeightUpdateFinalizer = Callable[[], None]


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


class IPCWeightManifestError(RuntimeError):
    """An IPC transfer did not match the prepared state-dict manifest."""


class _IPCWeightManifest:
    """Validate an IPC stream against its prepared state-dict manifest."""

    def __init__(self, expected_keys: Iterable[str]) -> None:
        self.expected_keys = set(expected_keys)
        self.loaded_keys: set[str] = set()
        self.errors: list[str] = []

    def validate_batch(self, keys: Sequence[str]) -> set[str] | None:
        batch_keys: set[str] = set()
        duplicate_keys: set[str] = set()
        for key in keys:
            if key in batch_keys:
                duplicate_keys.add(key)
            batch_keys.add(key)
        duplicate_keys.update(self.loaded_keys & batch_keys)
        unexpected_keys = batch_keys - self.expected_keys
        if duplicate_keys:
            self.errors.append(_format_refit_key_error("duplicate keys", duplicate_keys))
        if unexpected_keys:
            self.errors.append(_format_refit_key_error("unexpected keys", unexpected_keys))
        return None if self.errors else batch_keys

    def record_loaded(self, keys: set[str]) -> None:
        self.loaded_keys.update(keys)

    def record_load_failure(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        if len(message) > 512:
            message = message[:512] + " ..."
        self.errors.append(f"weight load failed: {message}")

    def require_complete(self) -> None:
        details = list(self.errors)
        missing_keys = self.expected_keys - self.loaded_keys
        if missing_keys:
            details.append(_format_refit_key_error("missing keys", missing_keys))
        if details:
            raise IPCWeightManifestError("; ".join(details))


_QKV_SOURCE_RE = re.compile(r"^(?P<prefix>.+)\.(?P<role>q_proj|k_proj|v_proj)\.weight$")
_DENSE_GATE_UP_SOURCE_RE = re.compile(r"^(?P<prefix>.+)\.(?P<role>gate_proj|up_proj)\.weight$")
_GROUPED_EXPERT_SOURCE_RE = re.compile(
    r"^(?P<prefix>.+\.(?:mlp|mixer)\.experts)\." r"(?P<role>gate_proj|up_proj|down_proj)\.weight$"
)


def _refit_parameter_alias(name: str) -> str:
    """Normalize only architecture-owned name aliases, not tensor roles."""
    layer_match = re.search(r"(?:^|\.)(layers\.\d+\..+)$", name)
    relative_name = layer_match.group(1) if layer_match is not None else name
    return relative_name.replace(".routed_experts.", ".")


def _resolve_refit_destination_name(candidate: str, destination_names: Iterable[str]) -> str | None:
    destinations = set(destination_names)
    if candidate in destinations:
        return candidate
    alias = _refit_parameter_alias(candidate)
    matches = sorted(name for name in destinations if _refit_parameter_alias(name) == alias)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous vLLM refit destination alias for {candidate!r}: {matches}")
    return matches[0] if matches else None


def _vllm_model_type(model_runner: Any, model: Any) -> str | None:
    """Read the HF model type across vLLM's public and model-local configs."""
    vllm_config = getattr(model_runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    candidates = (
        getattr(getattr(model, "config", None), "model_type", None),
        getattr(getattr(model_config, "hf_config", None), "model_type", None),
        getattr(model_config, "model_type", None),
    )
    return next((value for value in candidates if isinstance(value, str)), None)


def _is_quantized_vllm_model(model_runner: Any, model: Any) -> bool:
    vllm_config = getattr(model_runner, "vllm_config", None)
    return any(
        value is not None
        for value in (
            getattr(vllm_config, "quant_config", None),
            getattr(model, "quant_config", None),
        )
    )


def _expected_local_expert_ids(param: torch.nn.Parameter) -> set[int] | None:
    """Return global expert IDs owned by this destination, when inspectable."""
    owner = getattr(getattr(param, "weight_loader", None), "__self__", None)
    if bool(getattr(owner, "use_ep", False)):
        expert_map = getattr(owner, "_expert_map", None)
        if expert_map is None:
            raise RuntimeError("Strict refit coverage cannot inspect local expert ownership.")
        return {
            expert_id for expert_id, local_id in enumerate(expert_map.detach().cpu().tolist()) if int(local_id) >= 0
        }
    if param.ndim == 3:
        return set(range(param.shape[0]))
    return None


def _expert_uses_gated_activation(param: torch.nn.Parameter) -> bool:
    """Read vLLM's authoritative fused-MoE storage contract.

    Missing source metadata must never be interpreted as evidence that a gate
    does not exist.  vLLM's ``is_act_and_mul`` controls whether ``w13`` stores
    two input projections; default to gated (the stricter requirement) when an
    older loader does not expose the flag.
    """
    owner = getattr(getattr(param, "weight_loader", None), "__self__", None)
    moe_config = getattr(owner, "moe_config", None)
    return bool(getattr(moe_config, "is_act_and_mul", True))


class _MainModelDestinationManifest:
    """Track destination and fused-source coverage across one strict refit.

    A destination-name set alone is insufficient: vLLM returns the same fused
    ``qkv_proj``, ``gate_up_proj``, or ``w13`` name after loading any one source
    component.  This manifest therefore derives the expected source-component
    contract from the trainer's state-dict metadata and requires each fused
    component to be independently accepted by the loader.  Direct expert and
    NCCL paths contribute exact source names transaction-wide.
    """

    def __init__(
        self,
        expected_params: dict[str, torch.nn.Parameter],
        expected_source_names: Iterable[str],
        *,
        include_native_mtp_sources: bool = False,
        context: str = "main-model",
    ) -> None:
        self.expected_names = set(expected_params)
        self.loaded_names: set[str] = set()
        self.accepted_source_names: set[str] = set()
        self.source_to_destination: dict[str, str] = {}
        self._component_roles: dict[str, dict[int | None, dict[str, set[str]]]] = {}
        self.schema_errors: list[str] = []
        self.tracking_errors: list[str] = []
        self.context = context

        for source_name in expected_source_names:
            source_namespace = set(source_name.split("."))
            if not include_native_mtp_sources and source_namespace.intersection({"mtp", "draft"}):
                continue
            component = self._resolve_source_component(
                source_name,
                expected_params,
                target_is_mtp_drafter=include_native_mtp_sources,
            )
            if component is None:
                continue
            destination_name, role, expert_id = component
            if expert_id is not None:
                local_expert_ids = _expected_local_expert_ids(expected_params[destination_name])
                if local_expert_ids is not None and expert_id not in local_expert_ids:
                    continue
            self.source_to_destination[source_name] = destination_name
            self._component_roles.setdefault(destination_name, {}).setdefault(expert_id, {}).setdefault(
                role, set()
            ).add(source_name)

        self._validate_source_schema(expected_params)

    @staticmethod
    def _resolve_source_component(
        source_name: str,
        expected_params: dict[str, torch.nn.Parameter],
        *,
        target_is_mtp_drafter: bool = False,
    ) -> tuple[str, str, int | None] | None:
        expert_weight = parse_hf_expert_weight(source_name)
        if expert_weight is not None:
            destination = resolve_vllm_expert_parameter_name(
                expert_weight.parameter_name,
                expected_params,
                target_is_mtp_drafter=target_is_mtp_drafter,
            )
            if destination is None:
                return None
            projection = source_name.rsplit(".", 2)[-2]
            return (
                destination,
                projection.removesuffix("_proj"),
                expert_weight.expert_id,
            )

        grouped_match = _GROUPED_EXPERT_SOURCE_RE.match(source_name)
        if grouped_match is not None:
            role = grouped_match.group("role").removesuffix("_proj")
            destination_leaf = "w2_weight" if role == "down" else "w13_weight"
            candidate = f"{grouped_match.group('prefix')}.routed_experts." f"{destination_leaf}"
            destination = resolve_vllm_expert_parameter_name(
                candidate,
                expected_params,
                target_is_mtp_drafter=target_is_mtp_drafter,
            )
            return (destination, role, None) if destination is not None else None

        qkv_match = _QKV_SOURCE_RE.match(source_name)
        if qkv_match is not None:
            role = qkv_match.group("role")[0]
            candidate = f"{qkv_match.group('prefix')}.qkv_proj.weight"
            destination = _resolve_refit_destination_name(candidate, expected_params)
            return (destination, role, None) if destination is not None else None

        gate_up_match = _DENSE_GATE_UP_SOURCE_RE.match(source_name)
        if gate_up_match is not None and ".experts." not in source_name:
            role = gate_up_match.group("role").removesuffix("_proj")
            candidate = f"{gate_up_match.group('prefix')}.gate_up_proj.weight"
            destination = _resolve_refit_destination_name(candidate, expected_params)
            return (destination, role, None) if destination is not None else None

        # Preserve exact source-key association for ordinary 1:1 parameters as
        # well. Prefix-only HF/vLLM aliases (for example backbone.layers ->
        # model.layers) remain safe to resolve through the same layer-relative
        # matcher; architecture-specific renames that cannot be proven stay on
        # destination-only coverage.
        destination = _resolve_refit_destination_name(source_name, expected_params)
        return (destination, "direct", None) if destination is not None else None

    def _validate_source_schema(self, expected_params: dict[str, torch.nn.Parameter]) -> None:
        for destination_name, param in expected_params.items():
            role_groups = self._component_roles.get(destination_name, {})
            if destination_name.endswith(("w13_bias", "w2_bias")):
                # The pinned Nano/RLVR41 contract has ``mlp_bias=False``.  A
                # biased fused-MoE layout packs expert-indexed HF bias sources
                # into shared vLLM destinations, but the current refit
                # metadata does not expose enough slice identity to prove that
                # every expert bias was written.  Never let a single returned
                # destination name masquerade as complete coverage for that
                # unsupported variant.
                self.schema_errors.append(f"{destination_name}: strict fused expert-bias coverage " "is unsupported")
            elif destination_name.endswith("qkv_proj.weight"):
                self._require_roles(destination_name, role_groups, {"q", "k", "v"})
            elif destination_name.endswith("gate_up_proj.weight"):
                self._require_roles(destination_name, role_groups, {"gate", "up"})
            elif destination_name.endswith("w13_weight"):
                expected_ids = _expected_local_expert_ids(param)
                required_roles = {"up"}
                if _expert_uses_gated_activation(param):
                    required_roles.add("gate")
                self._require_roles(
                    destination_name,
                    role_groups,
                    required_roles,
                    expected_expert_ids=expected_ids,
                )
            elif destination_name.endswith("w2_weight"):
                self._require_roles(
                    destination_name,
                    role_groups,
                    {"down"},
                    expected_expert_ids=_expected_local_expert_ids(param),
                )

    def _require_roles(
        self,
        destination_name: str,
        role_groups: dict[int | None, dict[str, set[str]]],
        required_roles: set[str],
        *,
        expected_expert_ids: set[int] | None = None,
    ) -> None:
        grouped_roles = role_groups.get(None)
        if grouped_roles is not None:
            missing_roles = required_roles - set(grouped_roles)
            if missing_roles:
                self.schema_errors.append(f"{destination_name}: missing source roles " f"{sorted(missing_roles)}")
            return

        expert_ids = (
            expected_expert_ids
            if expected_expert_ids is not None
            else {expert_id for expert_id in role_groups if expert_id is not None}
        )
        if not expert_ids:
            missing_roles = required_roles - set(role_groups.get(None, {}))
            self.schema_errors.append(
                f"{destination_name}: no source components for roles " f"{sorted(missing_roles or required_roles)}"
            )
            return
        for expert_id in sorted(expert_ids):
            roles = role_groups.get(expert_id, {})
            missing_roles = required_roles - set(roles)
            if missing_roles:
                self.schema_errors.append(
                    f"{destination_name}: expert {expert_id} missing source roles " f"{sorted(missing_roles)}"
                )

    def expects_component_source(self, source_name: str) -> bool:
        return source_name in self.source_to_destination

    def component_destination(self, source_name: str) -> str:
        return self.source_to_destination[source_name]

    def record_loader_result(self, source_names: set[str], loaded_names: set[str] | None) -> None:
        if not isinstance(loaded_names, set):
            self.tracking_errors.append("the vLLM loader did not report loaded destination parameter names")
            return
        self.loaded_names.update(loaded_names)
        for source_name in source_names:
            destination_name = self.source_to_destination.get(source_name)
            if destination_name is not None and destination_name in loaded_names:
                self.accepted_source_names.add(source_name)

    def record_direct_load(
        self,
        source_destinations: dict[str, str],
        additional_destination_names: set[str] | None = None,
    ) -> None:
        self.loaded_names.update(source_destinations.values())
        self.loaded_names.update(additional_destination_names or set())
        for source_name, actual_destination in source_destinations.items():
            destination_name = self.source_to_destination.get(source_name)
            if destination_name is not None and destination_name == actual_destination:
                self.accepted_source_names.add(source_name)
            elif destination_name is not None:
                self.tracking_errors.append(
                    f"direct source {source_name!r} wrote {actual_destination!r}, " f"expected {destination_name!r}"
                )

    def record_loaded_destinations(self, loaded_names: set[str] | None) -> None:
        self.record_loader_result(set(), loaded_names)

    def require_valid_schema(self) -> None:
        """Reject an impossible strict source contract before any transfer."""
        if self.schema_errors:
            raise ValueError(f"Incomplete vLLM {self.context} refit source schema: " + "; ".join(self.schema_errors))

    def require_complete(self) -> None:
        if self.tracking_errors:
            raise RuntimeError(f"Strict {self.context} refit coverage failed: " + "; ".join(self.tracking_errors))

        details = list(self.schema_errors)
        missing_names = self.expected_names - self.loaded_names
        if missing_names:
            details.append(_format_refit_key_error("missing destination parameters", missing_names))
        missing_sources = set(self.source_to_destination) - self.accepted_source_names
        if missing_sources:
            details.append(_format_refit_key_error("missing required source components", missing_sources))
        if details:
            raise ValueError(f"Incomplete vLLM {self.context} refit coverage: " + "; ".join(details))


class NixlVllmWorker(VllmWorker):
    """vLLM worker that establishes NIXL/UCX before vLLM initialization."""

    def __new__(cls, vllm_config: Any, *args: Any, **kwargs: Any) -> "NixlVllmWorker":
        worker = super().__new__(cls)
        worker._nrl_nixl_preinit_agent = preinit_nixl_from_vllm_config(vllm_config)
        return worker


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key)


def _read_mtp_layer_weights_from_checkpoint(
    model_path: str, mtp_layer_indices: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Read only the MTP draft layer weights from a sharded HF safetensors checkpoint.

    Uses the checkpoint's ``model.safetensors.index.json`` to open only the
    shards that contain the requested transformer layer indices, so the
    multi-terabyte base-model weights are never read from disk.

    Args:
        model_path: Path to the HF checkpoint directory.
        mtp_layer_indices: Transformer layer indices belonging to the MTP module(s).

    Returns:
        A list of ``(weight_name, tensor)`` pairs for the requested layers, with
        tensors on CPU.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Some architectures expose MTP as appended transformer layers (for
    # example model.layers.52.*), while Nemotron-H exports its complete
    # predictor under a native mtp.layers.* namespace. Preserve native names:
    # vLLM's architecture-specific drafter loader owns their mapping.
    native_mtp_names = {name for name in weight_map if name.startswith("mtp.")}
    config_path = os.path.join(model_path, "config.json")
    if native_mtp_names and os.path.exists(config_path):
        with open(config_path) as f:
            hf_config = json.load(f)
        if str(hf_config.get("model_type", "")).startswith("nemotron_h"):
            predictor_count = int(hf_config.get("num_nextn_predict_layers", 0))
            predictor_pattern = str(hf_config.get("mtp_hybrid_override_pattern", ""))
            expected_native_layers = set(range(predictor_count * len(predictor_pattern)))
            native_layer_re = re.compile(r"^mtp\.layers\.(\d+)\.")
            actual_native_layers = {
                int(match.group(1)) for name in native_mtp_names if (match := native_layer_re.match(name)) is not None
            }
            if not expected_native_layers or actual_native_layers != expected_native_layers:
                raise ValueError(
                    "Incomplete Nemotron-H native MTP checkpoint: expected physical "
                    f"layers {sorted(expected_native_layers)}, found "
                    f"{sorted(actual_native_layers)}"
                )
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        match = layer_re.search(name)
        if name in native_mtp_names or (match is not None and int(match.group(1)) in mtp_layer_indices):
            shard_to_names.setdefault(shard, []).append(name)

    weights: list[tuple[str, torch.Tensor]] = []
    for shard, names in shard_to_names.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as reader:
            for name in names:
                weights.append((name, reader.get_tensor(name)))
    return weights


class VllmInternalWorkerExtension:
    # True once the MTP drafter has been served by a one-time disk load (see
    # load_mtp_weights_from_disk); refit then leaves those static weights alone.
    _mtp_drafter_from_disk: bool = False
    _sparse_delta_applier: Any = None
    _nrl_named_parameters: dict[str, torch.nn.Parameter]
    _nrl_main_destination_manifest: _MainModelDestinationManifest | None = None
    _nrl_nccl_reshard_destination_names: set[str] | None = None
    _nrl_nccl_reshard_source_destinations: dict[str, str] | None = None
    _nrl_refit_source_names: set[str] | None = None

    def _get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        params = getattr(self, "_nrl_named_parameters", None)
        if params is None:
            params = dict(self.model_runner.model.named_parameters())
            self._nrl_named_parameters = params
        return params

    def _load_full_hf_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        self._record_refit_source_names(name for name, _tensor in policy_weights)
        manifest = getattr(self, "_nrl_main_destination_manifest", None)
        if manifest is None:
            self.model_runner.model.load_weights(weights=policy_weights)
            return
        self._load_model_weights_with_manifest(self.model_runner.model, policy_weights, manifest)

    @staticmethod
    def _load_model_weights_with_manifest(
        model: Any,
        weights: list[tuple[str, torch.Tensor]],
        manifest: _MainModelDestinationManifest,
    ) -> set[str] | None:
        """Load fused components independently so acceptance is observable."""
        loaded_union: set[str] = set()
        tracking_failed = False
        remaining_weights: list[tuple[str, torch.Tensor]] = []
        component_cohorts: list[tuple[list[tuple[str, torch.Tensor]], set[str]]] = []
        for name, tensor in weights:
            if not manifest.expects_component_source(name):
                remaining_weights.append((name, tensor))
                continue
            destination_name = manifest.component_destination(name)
            for cohort_weights, cohort_destinations in component_cohorts:
                if destination_name not in cohort_destinations:
                    cohort_weights.append((name, tensor))
                    cohort_destinations.add(destination_name)
                    break
            else:
                component_cohorts.append(([(name, tensor)], {destination_name}))

        # Each cohort contains at most one source for any fused destination.
        # A returned destination can therefore be attributed to exactly one
        # component, while q/k/v and expert role+ID cohorts from every layer are
        # loaded together instead of rebuilding vLLM's loader maps per tensor.
        for cohort_weights, _cohort_destinations in component_cohorts:
            source_names = {name for name, _tensor in cohort_weights}
            loaded_names = model.load_weights(weights=cohort_weights)
            manifest.record_loader_result(source_names, loaded_names)
            if isinstance(loaded_names, set):
                loaded_union.update(loaded_names)
            else:
                tracking_failed = True

        if remaining_weights:
            loaded_names = model.load_weights(weights=remaining_weights)
            manifest.record_loader_result({name for name, _tensor in remaining_weights}, loaded_names)
            if isinstance(loaded_names, set):
                loaded_union.update(loaded_names)
            else:
                tracking_failed = True
        return None if tracking_failed else loaded_union

    def _record_refit_source_names(self, source_names: Iterable[str]) -> None:
        active_source_names = getattr(self, "_nrl_refit_source_names", None)
        if active_source_names is not None:
            active_source_names.update(source_names)

    def _record_main_destination_names(self, loaded_names: set[str] | None) -> None:
        """Record destinations initialized by one batch in the active refit."""
        manifest = getattr(self, "_nrl_main_destination_manifest", None)
        if manifest is not None:
            manifest.record_loaded_destinations(loaded_names)

    def _record_direct_main_load(
        self,
        source_destinations: dict[str, str],
        additional_destination_names: set[str] | None = None,
    ) -> None:
        """Record exact sources written without calling vLLM's loader."""
        self._record_refit_source_names(source_destinations)
        manifest = getattr(self, "_nrl_main_destination_manifest", None)
        if manifest is not None:
            manifest.record_direct_load(source_destinations, additional_destination_names)

    def _strict_nemotron_h_refit_model_type(self) -> str | None:
        model_type = _vllm_model_type(self.model_runner, self.model_runner.model)
        if isinstance(model_type, str) and model_type.startswith("nemotron_h"):
            return model_type
        return None

    def _new_main_destination_manifest(self, transport: WeightUpdateTransport) -> _MainModelDestinationManifest | None:
        """Build strict coverage only for the qualified Nemotron-H BF16 path."""
        model = self.model_runner.model
        model_type = self._strict_nemotron_h_refit_model_type()
        if model_type is None:
            return None
        real_quant_probe = getattr(self, "_is_real_quant_model", None)
        is_real_quant = bool(real_quant_probe()) if callable(real_quant_probe) else False
        if _is_quantized_vllm_model(self.model_runner, model) or is_real_quant:
            # A receiver-side rejection here could deadlock IPC/collective or
            # checkpoint-engine senders that have already been dispatched.
            # Preserve the established quantized refit path, but state the
            # qualification boundary explicitly: RLVR41 is BF16 and strict
            # source-component coverage is not claimed for quantized models.
            logger.warning(
                "Strict Nemotron-H source-component refit coverage is disabled "
                "for quantized vLLM model_type=%s; source-component and "
                "destination completeness are not validated by this path, and "
                "the existing quantized loading behavior is being preserved.",
                model_type,
            )
            return None

        logger.info(
            "Strict Nemotron-H BF16 refit coverage active for model_type=%s " "transport=%s",
            model_type,
            transport,
        )
        if transport == "nccl_reshard":
            refit_info = getattr(self, "nccl_reshard_refit_info", None)
            if refit_info is None:
                raise RuntimeError("Nemotron-H NCCL-reshard refit metadata was not prepared.")
            source_names = {
                param_info["name"]
                for layer_name in refit_info["layer_names"]
                for param_info in refit_info["per_layer_params"][layer_name]
            }
            source_names.update(refit_info.get("misc_meta", {}))
        else:
            state_dict_info = getattr(self, "state_dict_info", None)
            if state_dict_info is None:
                raise RuntimeError("Strict Nemotron-H refit coverage requires prepared " "state-dict metadata.")
            source_names = set(state_dict_info)
        return _MainModelDestinationManifest(self._get_named_parameters(), source_names)

    def _load_hf_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        from nemo_rl.models.generation.vllm.quantization import fp8

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            self._record_refit_source_names(name for name, _tensor in policy_weights)
            fp8.load_weights(policy_weights, self.model_runner)
            return
        self._load_full_hf_weights(policy_weights)

    def bind_numa(self) -> bool:
        """Pin this TP worker to its GPU's NUMA-local CPUs/memory.

        Invoked via ``collective_rpc`` on each vLLM TP worker once the engine
        (and CUDA) is up, so the worker's physical GPU id is resolved from its
        local device index (see ``resolve_visible_gpu_id``).
        """
        import torch

        from nemo_rl.distributed.numa_utils import (
            bind_to_gpu_numa,
            resolve_visible_gpu_id,
        )

        gpu_id = resolve_visible_gpu_id(torch.cuda.current_device())
        if gpu_id is None:
            return False
        return bind_to_gpu_numa(gpu_id)

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + resolve_rollout_rank(rank_prefix, world_size - train_world_size)

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        # Free cached torch-allocator blocks so NCCL's P2P transport buffers
        # (raw cudaMalloc at comm init) have headroom; otherwise comm_init OOMs
        # on memory-tight shapes (mirror the train side).
        torch.cuda.empty_cache()
        self.model_update_group.init_nccl_communicator(device=self.device)

    def init_nccl_reshard_comm_group(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Bootstrap this gen worker's nccl_reshard bulk-path comm group(s).

        One comm group per PP stage; gen workers join ALL ``pp_size`` groups
        (they need every stage's layers), created in stage order so the train
        ranks (each in only their own stage) unblock deterministically.
        Non-PP is simply ``pp_size == 1`` that contains all the gen ranks.
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        gen_rank_in_group = train_ranks_per_stage + rank_prefix + local_rank

        # Free cached blocks so NCCL P2P buffers have headroom (see init_collective).
        torch.cuda.empty_cache()
        self.pp_comm_groups = {}  # pyrefly: ignore[implicitly-defined-attribute]
        for stage in range(pp_size):
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=gen_rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=self.device)
            self.pp_comm_groups[stage] = group

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def report_node_hostname(self) -> str:
        """Return the host shared by worker processes on this node."""
        return socket.gethostname()

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = (
                zmq.Context()
            )  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, 120000)  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, 120000)  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        had_previous_state = hasattr(self, "state_dict_info")
        previous_state = getattr(self, "state_dict_info", None)
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
        try:
            manifest = self._new_main_destination_manifest("collective")
            if manifest is not None:
                manifest.require_valid_schema()
        except BaseException:
            # Preparation happens before sender dispatch. Do not retain an
            # invalid strict manifest as the metadata for a later transaction.
            if had_previous_state:
                self.state_dict_info = previous_state
            else:
                del self.state_dict_info
            raise

    def prepare_sparse_delta_refit_info(
        self, state_dict_info: dict[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> list[str]:
        """Reserve scratch space and report weights that require overwrite."""
        applier = self._get_sparse_delta_applier()
        return sorted(applier.discover_native_skips(state_dict_info))

    def _uses_fp8_kv_cache(self) -> bool:
        """Return whether this worker owns an FP8 KV cache."""
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        kv_cache_dtype = getattr(cache_config, "cache_dtype", None)
        return kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        if not self._uses_fp8_kv_cache():
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM. MTP parameters do not use the `draft.` prefix; they remain
        in the policy stream and are forwarded separately by
        ``_maybe_refit_mtp_drafter``.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _get_drafter_model(self) -> Any:
        """Return the vLLM drafter's underlying model, or None if absent.

        The drafter holds the speculative-decoding draft model (Eagle3 or MTP),
        which vLLM keeps as a module separate from the main model. Typed ``Any``
        because these are dynamic vLLM model classes whose ``load_weights`` /
        ``mtp_start_layer_idx`` members are not visible through ``nn.Module``.
        """
        draft_owner = getattr(self.model_runner, "drafter", None)
        return getattr(draft_owner, "model", None) if draft_owner else None

    def _load_draft_weights(self, draft_weights: list[tuple[str, torch.Tensor]]) -> set[str] | None:
        if not draft_weights:
            return set()
        self._record_refit_source_names(f"draft.{name}" for name, _tensor in draft_weights)

        draft_model = self._get_drafter_model()
        if draft_model is None:
            logger.warning("[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update.")
            return None
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        return draft_model.load_weights(weights=draft_weights)

    def _mtp_drafter_refit_enabled(self) -> bool:
        """Whether MTP drafter weights should be refreshed from the refit stream.

        For MTP speculative decoding where the trainer co-trains the MTP layer
        (``mtp_num_layers > 0``), the MTP weights are exported as part of the
        policy weight stream during refit (without the ``draft.`` prefix used by
        Eagle3), so the drafter must be fed those weights on every refit.

        Returns False when the MTP weights were instead loaded once from disk
        (see ``load_mtp_weights_from_disk``) — the path used when the trainer
        does not co-train the MTP layer — to avoid clobbering and re-processing
        those static weights.
        """
        if self._mtp_drafter_from_disk:
            return False
        spec_config = getattr(self.model_runner.vllm_config, "speculative_config", None)
        method = getattr(spec_config, "method", None) if spec_config else None
        if method not in ("deepseek_mtp", "mtp"):
            return False
        return self._get_drafter_model() is not None

    def _maybe_refit_mtp_drafter(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load refit weights into an MTP drafter co-trained with the policy.

        The drafter's ``load_weights`` selects the MTP-specific parameters (and
        shared embed_tokens / lm_head) it needs from the full policy weight
        stream. Megatron pads the vocab dimension, so weights are trimmed to the
        drafter's expected vocab size first, matching ``_load_draft_weights``.
        """
        if not self._mtp_drafter_refit_enabled():
            return
        draft_model = self._get_drafter_model()
        if draft_model is None:
            return
        weights = self._trim_vocab_padding(draft_model, weights)
        draft_model.load_weights(weights=weights)

    def _maybe_process_mtp_drafter_after_loading(self) -> None:
        """Finalize MTP drafter weights after a refit (e.g. MoE grouped-GEMM layout).

        Mirrors the main-model post-processing so the freshly refit MTP layers
        are converted to their runtime layout. Skipped for the disk-load path,
        which already processes its weights once at startup.
        """
        if not self._mtp_drafter_refit_enabled():
            return
        draft_model = self._get_drafter_model()
        if draft_model is None:
            return

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = self.model_runner.vllm_config.speculative_config.draft_model_config
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)

    def load_mtp_weights_from_disk(self, model_path: str) -> bool:
        """Load only the MTP (multi-token-prediction) draft weights from disk.

        Used when an MTP speculative-decoding policy runs with
        ``load_format="dummy"``: the main model receives real weights via refit,
        but the MTP draft layer is not covered by refit (the trainer runs with
        ``mtp_num_layers=0``), so its weights must come from the checkpoint. Only
        the MTP layer(s) are read, avoiding a full base-model load (~1.3 TB for
        DeepSeek-V3) on every inference replica.

        Args:
            model_path: Path to the HF checkpoint directory.

        Returns:
            bool: True if MTP weights were loaded.
        """
        draft_model = self._get_drafter_model()
        if draft_model is None:
            # vLLM places the speculative drafter only on the last pipeline
            # stage. Its absence is expected on every earlier stage, but means
            # the engine cannot serve speculative decoding on the owning stage.
            if get_pp_group().is_last_rank:
                raise RuntimeError(
                    "[mtp] vLLM speculative_config is set for MTP but the drafter "
                    "model is unavailable; cannot load MTP weights from disk."
                )
            return False

        predictor = draft_model.model
        mtp_layer_indices = set(
            range(
                predictor.mtp_start_layer_idx,
                predictor.mtp_start_layer_idx + predictor.num_mtp_layers,
            )
        )
        weights = _read_mtp_layer_weights_from_checkpoint(model_path, mtp_layer_indices)
        if not weights:
            raise ValueError(
                f"No MTP layer weights for layers {sorted(mtp_layer_indices)} "
                f"found in checkpoint at {model_path}. The checkpoint must "
                f"include MTP layer weights to run deepseek_mtp speculative decoding."
            )

        draft_model_type = getattr(getattr(draft_model, "config", None), "model_type", None)
        if (
            isinstance(draft_model_type, str)
            and draft_model_type.startswith("nemotron_h")
            and any(name.startswith("mtp.") for name, _ in weights)
        ):
            if getattr(draft_model, "quant_config", None) is not None:
                raise RuntimeError(
                    "Strict native Nemotron-H MTP coverage is currently "
                    "supported only for non-quantized vLLM drafters."
                )
            expected_params = {
                name: param
                for name, param in draft_model.named_parameters()
                if re.match(r"^model\.layers\.\d+\.", name)
            }
            if not expected_params:
                raise RuntimeError(
                    "The native Nemotron-H MTP drafter exposes no physical " "model.layers.* parameters to validate."
                )
            weights = self._trim_vocab_padding(draft_model, weights)
            manifest = _MainModelDestinationManifest(
                expected_params,
                (name for name, _tensor in weights),
                include_native_mtp_sources=True,
                context="native Nemotron-H MTP",
            )
            self._load_model_weights_with_manifest(draft_model, weights, manifest)
            manifest.require_complete()
        else:
            self._load_draft_weights(weights)

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = self.model_runner.vllm_config.speculative_config.draft_model_config
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)
        # Mark that the MTP drafter is served from a one-time disk load so refit
        # does not re-load or re-process these static weights.
        self._mtp_drafter_from_disk = True
        logger.info(
            "[mtp] Loaded MTP draft weights for layers %s from %s",
            sorted(mtp_layer_indices),
            model_path,
        )
        return True

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, dispatches policy weights through the configured refit loader,
        and loads draft weights into the drafter model.
        """
        if "Gemma3ForConditionalGeneration" in self.model_runner.vllm_config.model_config.architectures:
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        self._load_hf_weights(policy_weights)
        # Eagle3 draft weights are exported with the `draft.` prefix.
        self._load_draft_weights(draft_weights)
        # MTP drafters co-trained with the policy receive their weights from the
        # policy stream (no `draft.` prefix), so feed it the policy weights too.
        self._maybe_refit_mtp_drafter(policy_weights)

    def _get_sparse_delta_applier(self) -> Any:
        if self._sparse_delta_applier is None:
            # Avoid importing sparse-refit code for existing refit transports.
            from nemo_rl.models.generation.vllm.vllm_sparse_delta import (
                VllmSparseDeltaApplier,
            )

            self._sparse_delta_applier = VllmSparseDeltaApplier(
                self.model_runner,
                self.device,
            )
        return self._sparse_delta_applier

    @contextmanager
    def _weight_update_lifecycle(self, transport: WeightUpdateTransport) -> Iterator[WeightUpdateFinalizer]:
        """Provide setup/finalization around a transport-owned weight update."""
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        manifest = self._new_main_destination_manifest(transport)
        bulk_destination_names: set[str] | None = None
        bulk_source_destinations: dict[str, str] | None = None
        if transport == "nccl_reshard":
            bulk_destination_names = getattr(self, "_nrl_nccl_reshard_destination_names", None)
            bulk_source_destinations = getattr(self, "_nrl_nccl_reshard_source_destinations", None)
            if manifest is not None and (bulk_destination_names is None or bulk_source_destinations is None):
                raise RuntimeError("NCCL-reshard source/destination coverage was not prepared " "before refit.")
        self._nrl_refit_source_names = set()
        self._nrl_main_destination_manifest = manifest
        if transport == "nccl_reshard" and bulk_source_destinations:
            assert bulk_destination_names is not None
            # The bulk path writes these registered parameters directly and
            # validates a LocalParamSpec for every source tensor. The lifecycle
            # begins only after all bulk receives complete successfully, so
            # seed their exact source components before loading misc tensors.
            self._record_direct_main_load(bulk_source_destinations, bulk_destination_names)

        def finalize() -> None:
            source_names = getattr(self, "_nrl_refit_source_names", None)
            if not source_names:
                raise RuntimeError(
                    f"Refusing to finalize an empty {transport} weight update; " "the transaction received no weights."
                )
            if manifest is not None:
                manifest.require_complete()
            with set_current_vllm_config(self.model_runner.vllm_config):
                process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
            self._maybe_process_mtp_drafter_after_loading()

        try:
            yield finalize
            # Preserve the IPC lifetime boundary: the COMPLETE ACK is sent before
            # this optional second pass, just as it was before lifecycle hooks.
            self._maybe_process_fp8_kv_cache()
        finally:
            # A failed or completed transaction must never leak accumulated
            # destinations or source tokens into the next refit.
            self._nrl_main_destination_manifest = None
            self._nrl_refit_source_names = None

    def _weight_update_errors_are_fatal(self) -> bool:
        """Whether transport errors should propagate instead of returning False."""
        return False

    def _synchronize_before_ipc_data_ack(self) -> None:
        """Fence work consuming one IPC data batch before its acknowledgment."""
        torch.cuda.current_stream().synchronize()

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weight = None
        weights = None

        try:
            self.maybe_init_zmq()
            manifest = _IPCWeightManifest(self.state_dict_info)
            with self._weight_update_lifecycle("ipc") as finalize:
                while True:
                    # Blocking receive with timeout (this is the main operation)
                    payload = self.zmq_socket.recv_pyobj()

                    if payload == IPCProtocol.COMPLETE:
                        # A REP socket must reply even when validation or finalization
                        # fails, otherwise the sender remains blocked until timeout.
                        try:
                            manifest.require_complete()
                            finalize()
                        finally:
                            self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                        break

                    batch_keys = None
                    batch_error = None
                    try:
                        ipc_handle, list_keys, used_bytes = payload
                        batch_keys = manifest.validate_batch(list_keys)
                        if batch_keys is None:
                            continue

                        buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)
                        weights = []
                        offset = 0
                        for key in list_keys:
                            shape, dtype = self.state_dict_info[key]  # pyrefly
                            if isinstance(shape, list):
                                shape = torch.Size(shape)

                            size_in_bytes = dtype.itemsize * shape.numel()
                            weight = buffer[offset : offset + size_in_bytes].view(dtype=dtype).view(shape)
                            weights.append((key, weight))
                            offset += calculate_aligned_size(size_in_bytes)

                        assert offset == used_bytes, (
                            "Offset is not equal to used bytes, usually indicate "
                            "inaccurate info like keys or cached dtype in "
                            "state_dict_info"
                        )
                        self._load_weights(weights)
                    except Exception as error:
                        batch_error = error
                        # The manifest only keeps the exception message; log
                        # the full traceback and the batch contents so loader
                        # failures stay diagnosable from worker logs.
                        batch_desc = ", ".join(f"{k}: {tuple(w.shape)} {w.dtype}" for k, w in (weights or [])[:40])
                        logger.exception("IPC weight batch load failed (batch: %s)", batch_desc)
                    finally:
                        # Synchronize before releasing or ACKing an IPC allocation,
                        # including when a loader failed after scheduling CUDA work.
                        if buffer is not None:
                            try:
                                self._synchronize_before_ipc_data_ack()
                            except Exception as error:
                                if batch_error is None:
                                    batch_error = error

                        if batch_error is not None:
                            manifest.record_load_failure(batch_error)
                        elif batch_keys is not None:
                            manifest.record_loaded(batch_keys)

                        # Drop every view before ACK permits sender-side reuse.
                        del weight, weights, buffer
                        weight = None
                        weights = None
                        buffer = None
                        self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            if self._weight_update_errors_are_fatal():
                raise
            logger.exception(
                "Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: %s",
                e,
            )
            return False

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_from_collective")
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. " "Please call prepare_refit_info when initializing the worker."
        )

        try:
            with self._weight_update_lifecycle("collective") as finalize:
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self._load_weights,
                )
                finalize()

        except Exception as e:
            if self._weight_update_errors_are_fatal():
                raise
            logger.exception(
                "Error in VllmInternalWorkerExtension.update_weights_from_collective: %s",
                e,
            )
            return False

        gc.collect()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_decoded_sparse_payload(self, *payloads: bytes | str) -> dict[str, Any]:
        applier = self._get_sparse_delta_applier()
        return applier.update_weights_from_decoded_sparse_payload(*payloads)

    def synchronize_device(self) -> None:
        self._get_sparse_delta_applier().synchronize_device()

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        return self._get_sparse_delta_applier().finish_sparse_delta_refit()

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Restore per-layer param metadata and build the HF→vLLM mapping.

        Done once ahead of refit; the cached mapping is reused by every
        ``nccl_reshard_refit`` call.
        """
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            restore_refit_info_placements,
        )

        prepared_attributes = (
            "nccl_reshard_refit_info",
            "hf_to_local_param_map",
            "_nrl_nccl_reshard_destination_names",
            "_nrl_nccl_reshard_source_destinations",
        )
        previous_state = {name: (hasattr(self, name), getattr(self, name, None)) for name in prepared_attributes}
        try:
            self.nccl_reshard_refit_info = restore_refit_info_placements(
                refit_info
            )  # pyrefly: ignore[implicitly-defined-attribute]
            # Build HFToLocalParamMap (see nccl_reshard_utils)
            self.hf_to_local_param_map = (
                self.build_hf_to_local_param_map(  # pyrefly: ignore[implicitly-defined-attribute]
                    self.nccl_reshard_refit_info
                )
            )
            manifest = self._new_main_destination_manifest("nccl_reshard")
            if manifest is not None:
                # Catch missing q/k/v, fused roles, and explicitly unsupported
                # expert-bias schemas during coordinated setup, before the
                # first nccl.m2n bulk write can mutate live vLLM parameters.
                manifest.require_valid_schema()
        except BaseException:
            for name, (had_previous_value, previous_value) in previous_state.items():
                if had_previous_value:
                    setattr(self, name, previous_value)
                elif hasattr(self, name):
                    delattr(self, name)
            raise

    def build_hf_to_local_param_map(self, refit_info: dict) -> HFToLocalParamMap:
        """Build the vLLM-backend ``hf_to_local_param_map`` (HFToLocalParamMap).

        Wraps the ``(vllm_param, merged_slice)`` resolution from
        ``_build_hf_to_gen_backend_mapping`` into ``LocalParamSpec``s:
        - direct (slice ``None``): ``base`` is the live vLLM param; receive in place.
        - merged (dense ``gate_up_proj`` / grouped-expert ``w13``): ``pre`` allocs a
          recv buffer for this component's ``region`` slice, ``post`` copies it back
          (region recomputed each refit to track live storage).
        """

        def _merged_param_spec(vllm_param, merged_slice):
            def pre(_base: torch.Tensor) -> RefitCtx:
                region = vllm_param.data[merged_slice]
                return RefitCtx(buf=torch.empty_like(region), extra={"region": region})

            def post(ctx: RefitCtx) -> None:
                ctx.extra["region"].copy_(ctx.buf)

            return LocalParamSpec(base=vllm_param, pre=pre, post=post)

        def _bf16_to_mxfp8_receiver_quant_spec(
            value_param: torch.Tensor,
            scale_param: torch.Tensor,
            merged_slice: tuple[slice, ...] | None,
        ) -> LocalParamSpec:
            def pre(_base: torch.Tensor) -> RefitCtx:
                value_region = value_param.data if merged_slice is None else value_param.data[merged_slice]
                scale_region = scale_param.data if merged_slice is None else scale_param.data[merged_slice]
                return RefitCtx(
                    buf=torch.empty_like(value_region, dtype=torch.bfloat16),
                    extra={"value_region": value_region, "scale_region": scale_region},
                )

            def post(ctx: RefitCtx) -> None:
                from nemo_rl.models.generation.vllm.quantization.fp8 import (
                    quantize_mxfp8_weight,
                )

                value, scale = quantize_mxfp8_weight(ctx.buf)
                ctx.extra["value_region"].copy_(value)
                ctx.extra["scale_region"].copy_(scale)

            return LocalParamSpec(base=value_param.data, pre=pre, post=post)

        # Get dict of vllm_param and merged_slice for each hf_name
        vllm_param_map_and_slices = self._build_hf_to_gen_backend_mapping(refit_info)
        param_info_by_name = {
            param_info["name"]: param_info
            for layer_name in refit_info["layer_names"]
            for param_info in refit_info["per_layer_params"][layer_name]
        }

        def _destination_global_shape(
            local_shape: torch.Size,
            param_info: dict[str, Any],
        ) -> tuple[int, ...] | None:
            """Reconstruct a target's logical shape from its local storage.

            Production NCCL metadata always carries the destination mesh and
            placements.  A few legacy mapping-only unit fixtures omit them; a
            TP=1 fixture is still provable, while a synthetic TP>1 fixture is
            left to the slice-geometry checks below.
            """
            placements = param_info.get("dst_placements")
            mesh = param_info.get("dst_mesh_info")
            mesh_tensor = getattr(mesh, "mesh", getattr(mesh, "_mesh", None))
            if placements is None or mesh_tensor is None:
                if int(refit_info.get("gen_tp_size", 1)) == 1:
                    return tuple(local_shape)
                return None
            if len(placements) != mesh_tensor.ndim:
                raise ValueError(
                    "build_hf_to_local_param_map: destination placement rank "
                    f"{len(placements)} does not match mesh rank "
                    f"{mesh_tensor.ndim} for {param_info['name']!r}"
                )

            global_shape = list(local_shape)
            for mesh_dim, placement in enumerate(placements):
                if not isinstance(placement, Shard):
                    continue
                tensor_dim = int(placement.dim) % len(global_shape)
                global_shape[tensor_dim] *= int(mesh_tensor.shape[mesh_dim])
            return tuple(global_shape)

        # The real nccl.m2n API receives local tensors/meshes/placements but
        # discards DTensorRef.global_shape.  Prove before any transfer that the
        # selected live destination region implies exactly the advertised HF
        # global shape.  This catches missing grouped experts as well as a
        # direct non-gated w13/w2 or dense-down shape mismatch.
        for hf_name, (vllm_param, merged_slice) in vllm_param_map_and_slices.items():
            local_region = vllm_param.data if merged_slice is None else vllm_param.data[merged_slice]
            param_info = param_info_by_name[hf_name]
            if "global_shape" not in param_info:
                raise ValueError("build_hf_to_local_param_map: missing global_shape metadata " f"for {hf_name!r}")
            target_global_shape = _destination_global_shape(local_region.shape, param_info)
            source_global_shape = tuple(param_info["global_shape"])
            if target_global_shape is not None and target_global_shape != source_global_shape:
                raise ValueError(
                    "build_hf_to_local_param_map: source/destination shape "
                    f"mismatch for {hf_name!r}: source global shape "
                    f"{source_global_shape}, selected vLLM target implies "
                    f"{target_global_shape}"
                )

        # Destination-name coverage cannot prove that every region of a fused
        # tensor has a source. Validate the static NCCL map itself: a direct
        # source must be the only writer for its target, while merged slices
        # must be disjoint and cover one complete local dimension without gaps.
        mappings_by_param: dict[
            int,
            list[tuple[str, torch.Tensor, tuple[slice, ...] | None]],
        ] = {}
        for hf_name, (vllm_param, merged_slice) in vllm_param_map_and_slices.items():
            mappings_by_param.setdefault(id(vllm_param), []).append((hf_name, vllm_param, merged_slice))
        for mappings in mappings_by_param.values():
            direct_mappings = [item for item in mappings if item[2] is None]
            if direct_mappings:
                if len(mappings) != 1:
                    raise ValueError(
                        "build_hf_to_local_param_map: a direct target has "
                        f"multiple source mappings {[item[0] for item in mappings]}"
                    )
                continue

            intervals: list[tuple[int, int, int, str]] = []
            for hf_name, vllm_param, merged_slice in mappings:
                assert merged_slice is not None
                normalized = tuple(merged_slice) + (slice(None),) * (vllm_param.ndim - len(merged_slice))
                partial_dims: list[tuple[int, int, int]] = []
                for dim, (dim_slice, dim_size) in enumerate(zip(normalized, vllm_param.shape)):
                    if not isinstance(dim_slice, slice):
                        raise ValueError(
                            "build_hf_to_local_param_map: fused target slices "
                            f"must use slices, got {dim_slice!r} for {hf_name!r}"
                        )
                    start, stop, step = dim_slice.indices(dim_size)
                    if step != 1:
                        raise ValueError(
                            "build_hf_to_local_param_map: fused target slices " f"must be contiguous for {hf_name!r}"
                        )
                    if (start, stop) != (0, dim_size):
                        partial_dims.append((dim, start, stop))
                if len(partial_dims) != 1:
                    raise ValueError(
                        "build_hf_to_local_param_map: each merged source must "
                        "cover exactly one partial target dimension; "
                        f"{hf_name!r} covers {partial_dims}"
                    )
                dim, start, stop = partial_dims[0]
                intervals.append((dim, start, stop, hf_name))

            sliced_dims = {dim for dim, _start, _stop, _name in intervals}
            if len(sliced_dims) != 1:
                raise ValueError(
                    "build_hf_to_local_param_map: fused source slices disagree " f"on target dimension: {intervals}"
                )
            sliced_dim = next(iter(sliced_dims))
            ordered_intervals = sorted(intervals, key=lambda item: item[1])
            cursor = 0
            for _dim, start, stop, hf_name in ordered_intervals:
                if start != cursor or stop <= start:
                    raise ValueError(
                        "build_hf_to_local_param_map: incomplete or overlapping "
                        f"fused target coverage before {hf_name!r}: "
                        f"expected offset {cursor}, got [{start}, {stop})"
                    )
                cursor = stop
            target_size = mappings[0][1].shape[sliced_dim]
            if cursor != target_size:
                raise ValueError(
                    "build_hf_to_local_param_map: incomplete fused target "
                    f"coverage; covered [0, {cursor}), target size {target_size}"
                )

        vllm_params = dict(self.model_runner.model.named_parameters())
        vllm_names_by_id = {id(param): name for name, param in vllm_params.items()}
        specs = {}
        bulk_destination_names: set[str] = set()
        bulk_source_destinations: dict[str, str] = {}
        for hf_name, (vllm_param, merged_slice) in vllm_param_map_and_slices.items():
            vllm_name = vllm_names_by_id.get(id(vllm_param))
            if vllm_name is None:
                raise ValueError(
                    f"build_hf_to_local_param_map: resolved vLLM target for "
                    f"{hf_name!r} is not a registered model parameter"
                )
            bulk_destination_names.add(vllm_name)
            bulk_source_destinations[hf_name] = vllm_name
            wire_dtype_value = param_info_by_name[hf_name].get("dtype")
            wire_dtype = (
                wire_dtype_value if isinstance(wire_dtype_value, torch.dtype) else _STR_TO_DTYPE.get(wire_dtype_value)
            )
            if wire_dtype is None:
                raise ValueError(
                    f"build_hf_to_local_param_map: unsupported wire dtype " f"{wire_dtype_value!r} for {hf_name!r}"
                )
            if wire_dtype == torch.bfloat16 and vllm_param.dtype == torch.float8_e4m3fn:
                scale_names = (
                    vllm_name + "_scale_from_checkpoint",
                    vllm_name + "_scale",
                )
                scale_name = next((name for name in scale_names if name in vllm_params), None)
                scale_param = vllm_params.get(scale_name) if scale_name is not None else None
                if scale_param is None:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 target {vllm_name!r} "
                        f"for {hf_name!r} has no scale parameter among "
                        f"{scale_names!r}"
                    )
                assert scale_name is not None
                bulk_destination_names.add(scale_name)
                value_region = vllm_param if merged_slice is None else vllm_param[merged_slice]
                scale_region = scale_param if merged_slice is None else scale_param[merged_slice]
                if value_region.shape[-1] % 32 != 0:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 target for {hf_name!r} "
                        f"must have K divisible by 32, got {tuple(value_region.shape)}"
                    )
                expected_scale_shape = (
                    *value_region.shape[:-1],
                    value_region.shape[-1] // 32,
                )
                if tuple(scale_region.shape) != expected_scale_shape:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 scale target "
                        f"{scale_name!r} for {hf_name!r} has shape "
                        f"{tuple(scale_region.shape)}, expected {expected_scale_shape}"
                    )
                if scale_param.dtype != torch.uint8:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 scale target "
                        f"{scale_name!r} has dtype {scale_param.dtype}, expected torch.uint8"
                    )
                specs[hf_name] = _bf16_to_mxfp8_receiver_quant_spec(vllm_param, scale_param, merged_slice)
            elif wire_dtype != vllm_param.dtype:
                raise ValueError(
                    f"build_hf_to_local_param_map: wire dtype {wire_dtype} does not "
                    f"match target dtype {vllm_param.dtype} for {hf_name!r}"
                )
            else:
                specs[hf_name] = (
                    LocalParamSpec(base=vllm_param.data)
                    if merged_slice is None
                    else _merged_param_spec(vllm_param, merged_slice)
                )
        self._nrl_nccl_reshard_destination_names = bulk_destination_names
        self._nrl_nccl_reshard_source_destinations = bulk_source_destinations
        return HFToLocalParamMap(specs=specs)

    def _build_hf_to_gen_backend_mapping(self, refit_info):
        """Map each FFN HF param name to its gen-backend param and slice.

        Only ``gate_proj`` / ``up_proj`` / ``down_proj`` ``.weight``
        (dense MLP and MoE experts) reach here.
        Returns ``hf_name -> (vllm_param, merged_param_slice or None)``; the
        slice (``None`` for a 1:1 direct map) is the local region of a fused
        vLLM param this HF piece occupies, applied by the LocalParamSpec
        pre/post hooks.  The three shapes:

          - grouped MoE experts: gate/up -> ``w13_weight`` halves (dim 1),
            down -> ``w2_weight`` (direct).
          - dense MLP gate/up    -> ``gate_up_proj`` halves (dim 0).
          - dense MLP down       -> ``down_proj`` (direct 1:1).
        """
        vllm_params = dict(self.model_runner.model.named_parameters())
        mapping = {}

        # Collect FFN param names + global shapes from refit_info, plus the
        # grouped-expert tag (gate_proj/up_proj/down_proj) for MoE params.
        hf_shapes = {}  # hf_name -> global_shape
        hf_grouped = {}  # hf_name -> "gate_proj"|"up_proj"|"down_proj" (MoE only)
        for layer_name in refit_info["layer_names"]:
            # p is a dict of param info
            for p in refit_info["per_layer_params"][layer_name]:
                hf_shapes[p["name"]] = tuple(p["global_shape"])
                if p.get("grouped_expert_proj"):
                    hf_grouped[p["name"]] = p["grouped_expert_proj"]

        # Resolve an HF FFN name to its vLLM param name.  The two differ only in
        # the module prefix before ``layers.N`` (e.g. NemotronH's HF ``backbone.``
        # vs vLLM ``model.``); the layer-relative suffix is identical.  Index the
        # real vLLM names by that suffix so any prefix rename resolves generically
        # instead of hardcoding per-model swaps.  Matching-prefix models (most)
        # hit the exact-name fast path and never touch the index.
        def _layer_relative(name: str) -> str:
            prefix = _extract_layer_prefix(name)
            return name[len(prefix) + 1 :] if prefix else name

        vllm_by_relative = {_layer_relative(n): n for n in vllm_params}

        # vLLM 0.25 moved the fused-MoE expert weights onto a nested
        # ``routed_experts`` submodule, so real names carry a
        # ``.routed_experts.`` segment that the name built from the HF side
        # below does not (``...mlp.experts.w13_weight`` vs
        # ``...mlp.experts.routed_experts.w13_weight``).  Index the real names
        # with that segment dropped so either layout resolves; on a 0.20-style
        # model this index is identical to ``vllm_by_relative``.
        vllm_by_relative_flat = {_layer_relative(n).replace(".routed_experts.", "."): n for n in vllm_params}

        def _to_vllm_name(n: str) -> str:
            if n in vllm_params:
                return n
            relative = _layer_relative(n)
            if relative in vllm_by_relative:
                return vllm_by_relative[relative]
            return vllm_by_relative_flat.get(relative, n)

        for hf_name in hf_shapes:
            # 1) Grouped MoE expert params (gate_proj/up_proj/down_proj, each
            #    [E, ...]). vLLM fuses them as w13_weight (gate||up on the
            #    intermediate axis) and w2_weight (down). The received
            #    Shard(1)/Shard(2) shard is placed into the right w13/w2 region by
            #    the LocalParamSpec pre/post hooks (for the gated w13 halves).
            # Caveat: Dispatch on the grouped_expert_proj TAG, NOT the suffix,
            #   so dense gate_proj/up_proj (-> gate_up_proj, rule below) don't collide.
            grouped_proj = hf_grouped.get(hf_name)
            if grouped_proj is not None:
                # e.g.) expert_prefix = model.layers.3.mlp.experts
                expert_prefix = hf_name.rsplit(f".{grouped_proj}.weight", 1)[0]
                vllm_suffix = "w2_weight" if grouped_proj == "down_proj" else "w13_weight"
                # e.g.) vllm_name = model.layers.3.mlp.experts.w13_weight
                vllm_name = _to_vllm_name(f"{expert_prefix}.{vllm_suffix}")
                if vllm_name not in vllm_params:
                    raise ValueError(
                        f"_build_hf_to_gen_backend_mapping: grouped expert {hf_name!r} has "
                        f"no vLLM target {vllm_name!r}; refit would silently drop "
                        f"the expert weights."
                    )
                # vllm_param is a torch.Tensor corresponding to the vllm_name
                vllm_param = vllm_params[vllm_name]
                if grouped_proj == "down_proj" or not _expert_uses_gated_activation(vllm_param):
                    # Case for non-gated MLP layer or down_proj (w2)
                    # Weights are not merged, so the mapping is 1:1
                    mapping[hf_name] = (vllm_param, None)
                else:
                    # Gated MLP: vLLM fuses gate (w1) + up (w3) into w13 along the
                    # intermediate axis (dim 1).  Standard layout is [gate; up]:
                    # gate -> [:, :P, :], up -> [:, P:2P, :].  The FlashInfer
                    # CUTLASS unquantized MoE backend instead stores w13 as
                    # [w3; w1] = [up; gate]
                    P = vllm_param.shape[1] // 2
                    # Write canonical [gate; up], following vLLM's load_weights
                    # behavior. Per-MoE-backend layout diversity is resolved later by
                    # process_weights_after_loading at the end of nccl_reshard_refit.
                    sl = slice(0, P) if grouped_proj == "gate_proj" else slice(P, 2 * P)
                    mapping[hf_name] = (vllm_param, (slice(None), sl, slice(None)))
                continue

            # 2) Direct 1:1 (dense down_proj; also non-gated dense up_proj, which
            #    vLLM keeps unmerged).
            vllm_direct = _to_vllm_name(hf_name)
            if vllm_direct in vllm_params:
                mapping[hf_name] = (vllm_params[vllm_direct], None)
                continue

            # 3) Gated dense MLP: gate/up fuse into gate_up_proj along dim 0,
            #    [gate; up] -> gate=[0:I_local], up=[I_local:2*I_local], where
            #    I_local = intermediate // gen TP (even split, gate==up size).
            if hf_name.endswith(("gate_proj.weight", "up_proj.weight")):
                is_gate = hf_name.endswith("gate_proj.weight")
                suffix = "gate_proj.weight" if is_gate else "up_proj.weight"
                prefix = hf_name[: -len(suffix)]
                vllm_name = _to_vllm_name(prefix + "gate_up_proj.weight")
                if vllm_name in vllm_params:
                    gate_source_name = prefix + "gate_proj.weight"
                    up_source_name = prefix + "up_proj.weight"
                    missing_sources = {name for name in (gate_source_name, up_source_name) if name not in hf_shapes}
                    if missing_sources:
                        raise ValueError(
                            "_build_hf_to_gen_backend_mapping: incomplete dense "
                            "gate/up source family: "
                            + _format_refit_key_error("missing source components", missing_sources)
                        )
                    tp = refit_info.get("gen_tp_size", 1)
                    gate_local = hf_shapes[gate_source_name][0] // tp
                    up_local = hf_shapes[up_source_name][0] // tp
                    sl = slice(0, gate_local) if is_gate else slice(gate_local, gate_local + up_local)
                    mapping[hf_name] = (vllm_params[vllm_name], (sl,))
                    continue

            raise ValueError(
                f"_build_hf_to_gen_backend_mapping: no vLLM param for {hf_name!r} "
                f"(no grouped-expert / direct / gate_up-merge match). Only FFN "
                f"gate/up/down weights should reach the bulk path."
            )

        return mapping

    def nccl_reshard_refit(self) -> bool:
        """Receive weights from training workers via xferdtensor.

        Each HF param's ``LocalParamSpec`` (from ``hf_to_local_param_map``,
        built once in ``prepare_nccl_reshard_refit_info``) provides the dst buffer:
        for a direct param xferdtensor receives straight into the live vLLM
        param (no hooks); for a merged param (dense gate_up_proj, grouped w13)
        ``pre`` allocates a temp recv buffer and ``post`` copies the TP-local
        slice back into the live merged param.
        """
        import os
        from collections import OrderedDict

        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        def _recv_one_param(param_info, group, stream):
            # Coverage guard: every bulk param must have a spec; a missing entry
            # would silently discard its weights.
            spec = self.hf_to_local_param_map.get(param_info["name"])
            assert spec is not None, (
                f"nccl_reshard_refit: {param_info['name']!r} has no spec in "
                "hf_to_local_param_map (would silently discard its weights)"
            )
            # spec.pre/post run on the caller's current stream (this stage's
            # stream); xferdtensor should use the same stream.
            ctx = spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
            dst_tensor = DTensorRef(ctx.buf, param_info["global_shape"])
            xferdtensor(
                None,
                param_info["src_mesh_info"],
                param_info["src_placements"],
                dst_tensor,
                param_info["dst_mesh_info"],
                param_info["dst_placements"],
                group,
                stream,
            )
            if spec.post is not None:
                spec.post(ctx)

        # Group params by PP stage so different stages' bulk reshards run
        # concurrently on their own streams.  Non-PP = single stage 0 (params
        # carry no "pp_stage" key), so this collapses to one stage / one stream.
        stage_params = OrderedDict()
        for layer_name in self.nccl_reshard_refit_info["layer_names"]:
            for p in self.nccl_reshard_refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(p.get("pp_stage", 0), []).append(p)

        num_streams = max(
            1,
            min(int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params)),
        )

        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            # synchronize the last run in the same stream
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            stage_stream = streams[idx % num_streams]
            with torch.cuda.stream(stage_stream):
                group = self.pp_comm_groups[stage]
                for p in params:
                    _recv_one_param(p, group, stage_stream)
                ev = torch.cuda.Event()
                ev.record()
                events[idx] = ev

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        import time

        with self._weight_update_lifecycle("nccl_reshard") as finalize:
            misc_t0 = time.perf_counter()
            self._receive_and_load_misc_params()
            torch.cuda.synchronize()
            if torch.distributed.get_rank() == 0:
                print(
                    f"[nccl_reshard_refit] misc recv+load (gen side): " f"{time.perf_counter() - misc_t0:.2f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()

            # Finalize post-load weight processing: dense Linear + attention/MLA,
            # the per-MoE-backend w13 layout (FlashInfer CUTLASS/TRTLLM) that the
            # canonical [gate; up] bulk write above defers to here, and the MTP
            # drafter's mirror of the same. The FP8 KV-cache per-layer k/v scales
            # are finalized by the lifecycle on exit.
            finalize()

            torch.cuda.empty_cache()
        return True

    def _receive_and_load_misc_params(self) -> None:
        """Receive misc params via packed_broadcast and load via vLLM."""
        misc_meta = self.nccl_reshard_refit_info.get("misc_meta", {})
        if not misc_meta:
            return

        misc_state_dict_info = {
            name: (tuple(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]]) for name, meta in misc_meta.items()
        }

        packed_broadcast_consumer(
            iterator=iter(misc_state_dict_info.items()),
            group=self.model_update_group,
            src=0,
            post_unpack_func=self._load_weights,
        )

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()


class VllmInternalWorkerExtensionWithCheckpointEngine(VllmCheckpointEngineMixin, VllmInternalWorkerExtension):
    """vLLM worker extension with checkpoint-engine refit support."""
