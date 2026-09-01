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
"""Source-bound, process-contained loader for CPU worker unit tests.

Megatron-Bridge imports every registered model from its package ``__init__``.
The CPU gate's pinned image intentionally lacks Transformer Engine, so importing
that registry would prevent tests of NeMo-RL's dependency-independent worker
state machine.  This helper executes the production worker under a private
``nemo_rl.*`` module identity while fail-closing its unused Bridge/setup/config
import boundaries.  It then removes every isolated boundary while retaining the
source-bound Megatron train/data modules needed by later CPU tests.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import pathlib
import sys
import types

import pytest

_MISSING = object()
CANONICAL_WORKER_NAME = "nemo_rl.models.policy.workers.megatron_policy_worker"
PRIVATE_WORKER_NAME = "nemo_rl.models.policy.workers._cpu_test_private_megatron_policy_worker"


def _source_package_spec(name: str, search_path: object) -> object:
    spec = importlib.machinery.PathFinder.find_spec(name, search_path)
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise AssertionError(f"{name} does not resolve to a source package")
    origin = pathlib.Path(spec.origin).resolve(strict=True)
    locations = tuple(pathlib.Path(location).resolve(strict=True) for location in spec.submodule_search_locations)
    if origin.name != "__init__.py" or origin.parent not in locations:
        raise AssertionError(f"{name} origin and search path do not agree")
    return spec


def _literal_capability(setup_source: pathlib.Path) -> str:
    values = []
    for statement in ast.parse(setup_source.read_text()).body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY"
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            values.append(statement.value.value)
    if len(values) != 1:
        raise AssertionError("setup source must define exactly one literal shared-prefix MTP capability")
    return values[0]


def _seed_fail_closed_boundaries() -> set[types.ModuleType]:
    try:
        import megatron
    except ModuleNotFoundError:
        pytest.skip("megatron namespace is unavailable", allow_module_level=True)

    # A fully imported Bridge needs no isolation.  This path also preserves a
    # canonical worker that an earlier broad-suite test may already have loaded.
    if "megatron.bridge" in sys.modules:
        return set()
    namespace_paths = getattr(megatron, "__path__", None)
    if namespace_paths is None:
        raise AssertionError("megatron is not a package namespace")
    bridge_spec = importlib.machinery.PathFinder.find_spec("megatron.bridge", namespace_paths)
    if bridge_spec is None:
        pytest.skip("megatron.bridge is unavailable", allow_module_level=True)
    bridge_spec = _source_package_spec("megatron.bridge", namespace_paths)
    bridge_origin = pathlib.Path(bridge_spec.origin).resolve(strict=True)
    bridge_package = importlib.util.module_from_spec(bridge_spec)
    sys.modules["megatron.bridge"] = bridge_package

    models_spec = _source_package_spec("megatron.bridge.models", bridge_package.__path__)
    models_package = importlib.util.module_from_spec(models_spec)
    sys.modules["megatron.bridge.models"] = models_package

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU worker test unexpectedly invoked an isolated import boundary")

    installed = {bridge_package, models_package}
    bridge_stubs = {
        "megatron.bridge.training.checkpointing": (
            bridge_origin.parent / "training" / "checkpointing.py",
            ("maybe_finalize_async_save", "save_checkpoint"),
        ),
        "megatron.bridge.training.utils.pg_utils": (
            bridge_origin.parent / "training" / "utils" / "pg_utils.py",
            ("get_pg_collection",),
        ),
        "megatron.bridge.training.utils.train_utils": (
            bridge_origin.parent / "training" / "utils" / "train_utils.py",
            (
                "logical_and_across_model_parallel_group",
                "reduce_max_stat_across_model_parallel_group",
            ),
        ),
        "megatron.bridge.utils.common_utils": (
            bridge_origin.parent / "utils" / "common_utils.py",
            ("get_rank_safe",),
        ),
    }
    for module_name, (source, names) in bridge_stubs.items():
        source = source.resolve(strict=True)
        module_spec = importlib.util.spec_from_file_location(module_name, source)
        if module_spec is None:
            raise AssertionError(f"could not source-bind {module_name}")
        module = importlib.util.module_from_spec(module_spec)
        for name in names:
            setattr(module, name, unavailable)
        sys.modules[module_name] = module
        installed.add(module)

    nemo_spec = importlib.util.find_spec("nemo_rl")
    if nemo_spec is None or nemo_spec.origin is None:
        raise AssertionError("nemo_rl does not resolve to a source package")
    nemo_root = pathlib.Path(nemo_spec.origin).resolve(strict=True).parent
    megatron_root = nemo_root / "models" / "megatron"

    # ``train.py`` imports this module only for MegatronModule annotations.  Do
    # not execute the real NeMo config module: that would import Bridge's model
    # registry and defeat this CPU-only boundary.  The sole exposed symbol is
    # the real MCore class, so any unexpected config dependency fails closed.
    config_source = (megatron_root / "config.py").resolve(strict=True)
    config_name = "nemo_rl.models.megatron.config"
    config_spec = importlib.util.spec_from_file_location(config_name, config_source)
    if config_spec is None:
        raise AssertionError("could not source-bind NeMo config boundary")
    config_module = importlib.util.module_from_spec(config_spec)
    from megatron.core.transformer import MegatronModule

    config_module.MegatronModule = MegatronModule
    sys.modules[config_name] = config_module
    installed.add(config_module)

    setup_source = (megatron_root / "setup.py").resolve(strict=True)
    setup_name = "nemo_rl.models.megatron.setup"
    setup_spec = importlib.util.spec_from_file_location(setup_name, setup_source)
    if setup_spec is None:
        raise AssertionError("could not source-bind NeMo setup boundary")
    setup_module = importlib.util.module_from_spec(setup_spec)
    setup_module.SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY = _literal_capability(setup_source)
    for name in (
        "_get_mcore_shared_prefix_training_capability",
        "build_inference_model",
        "finalize_megatron_setup",
        "handle_model_import",
        "setup_distributed",
        "setup_model_and_optimizer",
        "setup_reference_model_state",
        "validate_and_set_config",
        "validate_model_paths",
    ):
        setattr(setup_module, name, unavailable)
    sys.modules[setup_name] = setup_module
    installed.add(setup_module)
    return installed


def _require_source_module(name: str, source: pathlib.Path) -> types.ModuleType:
    module = sys.modules.get(name)
    origin = getattr(module, "__file__", None)
    if not isinstance(module, types.ModuleType) or origin is None:
        raise AssertionError(f"{name} was not loaded as a source module")
    if pathlib.Path(origin).resolve(strict=True) != source:
        raise AssertionError(f"{name} did not load from its production source")
    return module


def _snapshot(prefix: str) -> dict[str, types.ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if isinstance(module, types.ModuleType) and (name == prefix or name.startswith(f"{prefix}."))
    }


def _restore_subtree(
    prefix: str,
    previous: dict[str, types.ModuleType],
    *,
    preserve: dict[str, types.ModuleType] | None = None,
) -> set[types.ModuleType]:
    preserve = preserve or {}
    target = dict(previous)
    for name, module in preserve.items():
        target.setdefault(name, module)

    current = _snapshot(prefix)
    removed = {module for name, module in current.items() if target.get(name, _MISSING) is not module}
    for name in current:
        if name not in target:
            del sys.modules[name]
    sys.modules.update(target)

    # Importlib also attaches children to package objects.  Remove references
    # to discarded modules, then attach every restored/preserved child to the
    # exact parent object now installed in sys.modules.
    for name in current:
        if name in target:
            continue
        parent_name, separator, child_name = name.rpartition(".")
        if not separator:
            continue
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        child = getattr(parent, child_name, _MISSING)
        if any(child is module for module in removed):
            delattr(parent, child_name)

    for name, module in sorted(target.items(), key=lambda item: item[0].count(".")):
        parent_name, separator, child_name = name.rpartition(".")
        if not separator:
            continue
        parent = sys.modules.get(parent_name)
        if isinstance(parent, types.ModuleType):
            setattr(parent, child_name, module)
    return removed


def _load() -> tuple[
    types.ModuleType,
    pathlib.Path,
    tuple[types.ModuleType, ...],
    object,
    dict[str, types.ModuleType],
    dict[str, pathlib.Path],
]:
    if PRIVATE_WORKER_NAME in sys.modules:
        raise AssertionError("private worker alias was unexpectedly preloaded")
    before_bridge = _snapshot("megatron.bridge")
    before_nemo = _snapshot("nemo_rl")
    before_canonical = sys.modules.get(CANONICAL_WORKER_NAME, _MISSING)

    nemo_spec = importlib.util.find_spec("nemo_rl")
    if nemo_spec is None or nemo_spec.origin is None:
        raise AssertionError("nemo_rl does not resolve to a source package")
    worker_source = (
        pathlib.Path(nemo_spec.origin).resolve(strict=True).parent
        / "models"
        / "policy"
        / "workers"
        / "megatron_policy_worker.py"
    ).resolve(strict=True)
    worker_spec = importlib.util.spec_from_file_location(PRIVATE_WORKER_NAME, worker_source)
    if worker_spec is None or worker_spec.loader is None:
        raise AssertionError("could not create private production-worker spec")
    worker_module = importlib.util.module_from_spec(worker_spec)
    boundaries: set[types.ModuleType] = set()
    preserved_nemo: dict[str, types.ModuleType] = {}
    preserved_sources: dict[str, pathlib.Path] = {}
    succeeded = False
    try:
        boundaries = _seed_fail_closed_boundaries()
        sys.modules[PRIVATE_WORKER_NAME] = worker_module
        worker_spec.loader.exec_module(worker_module)

        function_source = pathlib.Path(worker_module._should_use_router_replay.__code__.co_filename).resolve(
            strict=True
        )
        if function_source != worker_source:
            raise AssertionError("private worker functions are not production-source bound")

        nemo_root = worker_source.parents[3]
        models_root = nemo_root / "models"
        megatron_root = models_root / "megatron"
        preserved_sources = {
            "nemo_rl": (nemo_root / "__init__.py").resolve(strict=True),
            "nemo_rl.models": (models_root / "__init__.py").resolve(strict=True),
            "nemo_rl.models.megatron": (megatron_root / "__init__.py").resolve(strict=True),
            "nemo_rl.models.megatron.data": (megatron_root / "data.py").resolve(strict=True),
            "nemo_rl.models.megatron.train": (megatron_root / "train.py").resolve(strict=True),
        }
        preserved_nemo = {name: _require_source_module(name, source) for name, source in preserved_sources.items()}
        for name, module in preserved_nemo.items():
            previous = before_nemo.get(name, _MISSING)
            if previous is not _MISSING and previous is not module:
                raise AssertionError(f"{name} replaced a preexisting module")
        succeeded = True
    finally:
        removed_bridge = _restore_subtree("megatron.bridge", before_bridge)
        removed_nemo = _restore_subtree(
            "nemo_rl",
            before_nemo,
            preserve=preserved_nemo if succeeded else None,
        )
        boundaries.update(removed_bridge)
        boundaries.update(removed_nemo)
        boundaries.discard(worker_module)
        if sys.modules.get(PRIVATE_WORKER_NAME) is worker_module:
            del sys.modules[PRIVATE_WORKER_NAME]

    return (
        worker_module,
        worker_source,
        tuple(boundaries),
        before_canonical,
        dict(preserved_nemo),
        dict(preserved_sources),
    )


(
    WORKER_MODULE,
    WORKER_SOURCE,
    ISOLATED_IMPORT_MODULES,
    _CANONICAL_BEFORE,
    PRESERVED_NEMO_MODULES,
    PRESERVED_NEMO_SOURCES,
) = _load()
SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY = WORKER_MODULE.SUPPORTED_SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY


def assert_import_isolation() -> None:
    """Regression assertion for canonical-module and package-cache hygiene."""

    assert PRIVATE_WORKER_NAME not in sys.modules
    assert pathlib.Path(WORKER_MODULE.__file__).resolve(strict=True) == WORKER_SOURCE
    canonical_now = sys.modules.get(CANONICAL_WORKER_NAME, _MISSING)
    assert canonical_now is _CANONICAL_BEFORE
    assert PRESERVED_NEMO_MODULES.keys() == PRESERVED_NEMO_SOURCES.keys()
    for name, expected_module in PRESERVED_NEMO_MODULES.items():
        assert sys.modules.get(name) is expected_module
        assert _require_source_module(name, PRESERVED_NEMO_SOURCES[name]) is expected_module
        parent_name, separator, child_name = name.rpartition(".")
        if separator:
            parent = sys.modules.get(parent_name)
            assert isinstance(parent, types.ModuleType)
            assert getattr(parent, child_name, _MISSING) is expected_module
    for isolated in ISOLATED_IMPORT_MODULES:
        assert all(module is not isolated for module in sys.modules.values())


assert_import_isolation()
