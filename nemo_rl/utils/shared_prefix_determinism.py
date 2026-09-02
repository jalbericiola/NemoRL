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

"""Dependency-neutral constants for shared-prefix deterministic execution."""

from collections.abc import Mapping
from types import MappingProxyType


SHARED_PREFIX_DETERMINISM_ENV_VAR_VALUES: Mapping[str, str] = MappingProxyType(
    {
        "MAMBA_DETERMINISTIC": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
    }
)
SHARED_PREFIX_DETERMINISM_MODEL_OVERRIDE_VALUES: Mapping[str, bool] = MappingProxyType(
    {
        "deterministic_mode": True,
        "cross_entropy_loss_fusion": False,
        "tp_comm_overlap": False,
    }
)
SHARED_PREFIX_FORBIDDEN_DETERMINISM_ENV_VAR_NAMES = frozenset(
    {"TRITON_CACHE_AUTOTUNING"}
)
SHARED_PREFIX_FORBIDDEN_DETERMINISM_ENV_VAR_PREFIXES = ("TRITON_AUTOTUNE_BLOCK",)
