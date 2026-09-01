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

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.model_diagnostics import shared_prefix_native_optimizer_gate as gate


def test_native_optimizer_gate_source_contract_is_green():
    assert gate._source_contract() == {
        "schema": gate.SCHEMA,
        "forbidden_direct_calls": 0,
        "production_closure_calls": 1,
    }


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "optimizer.initialize_state(parameter)",
        "_initialize_optimizer_state_for_probe(model, optimizer, scheduler)",
    ),
)
def test_native_optimizer_gate_source_contract_rejects_diagnostic_initializer_calls(monkeypatch, forbidden_call):
    source = (
        "def probe(production_initializer, optimizer, parameter, model, scheduler):\n"
        "    production_initializer(optimizer, None)\n"
        f"    {forbidden_call}\n"
    )
    monkeypatch.setattr(gate, "_read_regular_file_bytes", lambda *_args, **_kwargs: (source.encode(), None))

    with pytest.raises(gate.NativeOptimizerGateError, match="forbidden direct initializer calls"):
        gate._source_contract()


def test_native_optimizer_gate_source_contract_requires_one_production_call(monkeypatch):
    monkeypatch.setattr(
        gate,
        "_read_regular_file_bytes",
        lambda *_args, **_kwargs: (b"def probe():\n    pass\n", None),
    )

    with pytest.raises(gate.NativeOptimizerGateError, match="production MCore closure exactly once"):
        gate._source_contract()


def test_immutable_source_file_rejects_source_outside_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    outside.chmod(0o444)

    with pytest.raises(gate.NativeOptimizerGateError, match="outside --repo-root"):
        gate._require_immutable_source_file(outside, repo_root=repo_root, label="adversarial")


def test_immutable_source_file_rejects_writable_source(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source = repo_root / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(gate.NativeOptimizerGateError, match="writable"):
        gate._require_immutable_source_file(source, repo_root=repo_root, label="adversarial")


def test_descriptor_safe_read_rejects_path_swap_during_open(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    replacement = tmp_path / "replacement.py"
    source.write_bytes(b"authorized\n")
    replacement.write_bytes(b"substituted\n")
    real_open = gate.os.open

    def swap_then_open(path, flags):
        replacement.replace(source)
        return real_open(path, flags)

    monkeypatch.setattr(gate.os, "open", swap_then_open)
    with pytest.raises(gate.NativeOptimizerGateError, match="changed while being opened"):
        gate._read_regular_file_bytes(source, label="adversarial")


def test_bound_source_loader_records_exact_compiled_bytes(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    package_root = repo_root / "nemo_rl"
    package_root.mkdir(parents=True)
    source = package_root / "__init__.py"
    source_bytes = b"VALUE = 7\n"
    source.write_bytes(source_bytes)
    audit = {}
    loader = gate._BoundSourceLoader("nemo_rl", str(source), repo_root=repo_root, audit=audit)
    monkeypatch.setattr(gate.sys, "dont_write_bytecode", True)

    code = loader.get_code("nemo_rl")

    assert code is not None
    assert audit == {
        "nemo_rl": {
            "module": "nemo_rl",
            "relative_path": "nemo_rl/__init__.py",
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size": len(source_bytes),
        }
    }


def test_bound_source_loader_rejects_existing_cached_bytecode(tmp_path):
    repo_root = tmp_path / "repo"
    source = repo_root / "nemo_rl/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 7\n", encoding="utf-8")
    cached = source.parent / "__pycache__/__init__.cpython-313.pyc"
    cached.parent.mkdir()
    cached.write_bytes(b"cached")
    loader = gate._BoundSourceLoader("nemo_rl", str(source), repo_root=repo_root, audit={})

    with pytest.raises(gate.NativeOptimizerGateError, match="attempted cached bytecode"):
        loader.get_data(str(cached))


def test_loaded_module_manifest_rejects_editable_import_outside_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "nemo_rl.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    outside.chmod(0o444)
    modules = {"nemo_rl": SimpleNamespace(__file__=str(outside))}

    with pytest.raises(gate.NativeOptimizerGateError, match="outside --repo-root"):
        gate._loaded_module_source_manifest(repo_root, modules, prefixes=("nemo_rl",))


def test_loaded_module_manifest_rejects_source_not_compiled_by_bound_loader(tmp_path):
    repo_root = tmp_path / "repo"
    package_root = repo_root / "nemo_rl"
    package_root.mkdir(parents=True)
    source = package_root / "__init__.py"
    source.write_text("value = 1\n", encoding="utf-8")
    source.chmod(0o444)
    package_root.chmod(0o555)
    repo_root.chmod(0o555)
    modules = {"nemo_rl": SimpleNamespace(__file__=str(source), __cached__=None)}
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="not compiled from its bound source bytes"):
            gate._loaded_module_source_manifest(
                repo_root,
                modules,
                prefixes=("nemo_rl",),
                expected_roots={"nemo_rl": Path("nemo_rl")},
                executed_sources={},
            )
    finally:
        package_root.chmod(0o755)
        repo_root.chmod(0o755)


def test_immutable_source_file_rejects_writable_parent_directory(tmp_path):
    repo_root = tmp_path / "repo"
    source_parent = repo_root / "package"
    source_parent.mkdir(parents=True)
    source = source_parent / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    source.chmod(0o444)
    repo_root.chmod(0o555)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="permits source substitution"):
            gate._require_immutable_source_file(source, repo_root=repo_root, label="adversarial")
    finally:
        source_parent.chmod(0o755)
        repo_root.chmod(0o755)


def test_immutable_tree_rejects_writable_empty_directory(tmp_path):
    repo_root = tmp_path / "repo"
    tree_root = repo_root / "tree"
    writable_empty = tree_root / "replaceable"
    writable_empty.mkdir(parents=True)
    source = tree_root / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    source.chmod(0o444)
    tree_root.chmod(0o555)
    repo_root.chmod(0o555)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="permits source substitution"):
            gate._immutable_tree_fingerprint(tree_root, repo_root=repo_root, label="adversarial")
    finally:
        writable_empty.chmod(0o755)
        tree_root.chmod(0o755)
        repo_root.chmod(0o755)


def test_immutable_tree_rejects_cached_bytecode(tmp_path):
    repo_root = tmp_path / "repo"
    tree_root = repo_root / "tree"
    cache_root = tree_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source = tree_root / "source.py"
    cached = cache_root / "source.cpython-313.pyc"
    source.write_text("value = 1\n", encoding="utf-8")
    cached.write_bytes(b"cached")
    for path in (source, cached):
        path.chmod(0o444)
    for path in (cache_root, tree_root, repo_root):
        path.chmod(0o555)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="forbidden cached bytecode"):
            gate._immutable_tree_fingerprint(tree_root, repo_root=repo_root, label="adversarial")
    finally:
        cache_root.chmod(0o755)
        tree_root.chmod(0o755)
        repo_root.chmod(0o755)


def test_loaded_module_manifest_rejects_existing_cached_bytecode(tmp_path):
    repo_root = tmp_path / "repo"
    package_root = repo_root / "nemo_rl"
    cache_root = package_root / "__pycache__"
    cache_root.mkdir(parents=True)
    source = package_root / "__init__.py"
    cached = cache_root / "__init__.cpython-313.pyc"
    source.write_text("value = 1\n", encoding="utf-8")
    cached.write_bytes(b"cached")
    source.chmod(0o444)
    cached.chmod(0o444)
    package_root.chmod(0o555)
    repo_root.chmod(0o555)
    modules = {"nemo_rl": SimpleNamespace(__file__=str(source), __cached__=str(cached))}
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="forbidden cached bytecode"):
            gate._loaded_module_source_manifest(
                repo_root,
                modules,
                prefixes=("nemo_rl",),
                expected_roots={"nemo_rl": Path("nemo_rl")},
            )
    finally:
        cache_root.chmod(0o755)
        package_root.chmod(0o755)
        repo_root.chmod(0o755)


def _sealed_outer_metadata(tmp_path, *, expected_override: bytes | None = None):
    repo_root = tmp_path / "repo"
    metadata_root = repo_root / gate.DEPLOYMENT_METADATA_ROOT_RELATIVE
    metadata_root.mkdir(parents=True)
    identities = b'{"schema":"test"}\n'
    identities_sha256 = hashlib.sha256(identities).hexdigest()
    manifest = f"{identities_sha256}  SOURCE_IDENTITIES.json\n".encode("ascii")
    outer_sha256 = hashlib.sha256(manifest).hexdigest()
    expected = f"package_manifest_sha256={outer_sha256}\n".encode("ascii")
    payloads = {
        gate.OUTER_MANIFEST_RELATIVE.name: manifest,
        gate.OUTER_EXPECTED_RELATIVE.name: expected_override or expected,
        gate.SOURCE_IDENTITIES_RELATIVE.name: identities,
    }
    for name, payload in payloads.items():
        path = metadata_root / name
        path.write_bytes(payload)
        path.chmod(0o444)
    metadata_root.chmod(0o555)
    repo_root.chmod(0o555)
    return repo_root, metadata_root, outer_sha256


def test_outer_package_binding_verifies_raw_manifest_and_expected_file(tmp_path):
    repo_root, metadata_root, outer_sha256 = _sealed_outer_metadata(tmp_path)
    try:
        assert gate._outer_package_binding(repo_root, outer_package_expected=outer_sha256) == {
            "manifest_relative_path": gate.OUTER_MANIFEST_RELATIVE.as_posix(),
            "manifest_sha256": outer_sha256,
            "expected_relative_path": gate.OUTER_EXPECTED_RELATIVE.as_posix(),
            "expected_sha256": outer_sha256,
        }
    finally:
        metadata_root.chmod(0o755)
        repo_root.chmod(0o755)


def test_outer_package_binding_rejects_nonexact_expected_file(tmp_path):
    repo_root, metadata_root, outer_sha256 = _sealed_outer_metadata(
        tmp_path, expected_override=b"package_manifest_sha256=not-the-digest\n"
    )
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="does not exactly bind"):
            gate._outer_package_binding(repo_root, outer_package_expected=outer_sha256)
    finally:
        metadata_root.chmod(0o755)
        repo_root.chmod(0o755)


def _seal_runtime_materialization(repo_root, metadata_root, outer_sha256, inner_sha256, *, newline=False):
    manifest = {
        "schema": gate.RUNTIME_MATERIALIZATION_SCHEMA,
        "base": {
            "root": gate.Q_BASE_ROOT,
            "ready": gate.Q_BASE_READY,
            "package_manifest_sha256": gate.Q_BASE_PACKAGE_SHA256,
        },
        "outer_package_manifest_sha256": outer_sha256,
        "runtime_source_package_sha256": inner_sha256,
        "runtime_layout": gate.RUNTIME_LAYOUT,
        "repositories": {
            name: {
                "base_head": "1" * 40,
                "head": "2" * 40,
                "overlay_archive_sha256": "3" * 64,
                "overlay_payload_manifest_sha256": "4" * 64,
            }
            for name in ("NemoRL", "Megatron-Bridge", "Megatron-LM")
        },
        "runtime_repo_root": str(repo_root),
    }
    raw = gate._canonical_json_bytes(manifest) + (b"\n" if newline else b"")
    metadata_root.chmod(0o755)
    path = repo_root / gate.RUNTIME_MATERIALIZATION_RELATIVE
    path.write_bytes(raw)
    path.chmod(0o444)
    metadata_root.chmod(0o555)
    return manifest


def test_runtime_materialization_binding_is_canonical_and_two_digest_bound(tmp_path):
    repo_root, metadata_root, outer_sha256 = _sealed_outer_metadata(tmp_path)
    inner_sha256 = "a" * 64
    manifest = _seal_runtime_materialization(repo_root, metadata_root, outer_sha256, inner_sha256)
    try:
        result = gate._runtime_materialization_binding(
            repo_root,
            outer_package_sha256=outer_sha256,
            runtime_source_package_sha256=inner_sha256,
        )
        assert result["relative_path"] == gate.RUNTIME_MATERIALIZATION_RELATIVE.as_posix()
        assert result["manifest"] == manifest
        assert result["sha256"] == hashlib.sha256(gate._canonical_json_bytes(manifest)).hexdigest()
    finally:
        metadata_root.chmod(0o755)
        repo_root.chmod(0o755)


def test_runtime_materialization_binding_rejects_noncanonical_bytes(tmp_path):
    repo_root, metadata_root, outer_sha256 = _sealed_outer_metadata(tmp_path)
    inner_sha256 = "a" * 64
    _seal_runtime_materialization(repo_root, metadata_root, outer_sha256, inner_sha256, newline=True)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="exact canonical JSON bytes"):
            gate._runtime_materialization_binding(
                repo_root,
                outer_package_sha256=outer_sha256,
                runtime_source_package_sha256=inner_sha256,
            )
    finally:
        metadata_root.chmod(0o755)
        repo_root.chmod(0o755)


def _raw_optimizer_snapshots():
    before_moments = {
        state_key: {
            "sha256": "0" * 64,
            "tensor_count": 1,
            "numel": 8,
            "nonzero": 0,
            "l2": 0.0,
        }
        for state_key in ("exp_avg", "exp_avg_sq")
    }
    after_moments = {
        state_key: {
            "sha256": digest_character * 64,
            "tensor_count": 1,
            "numel": 8,
            "nonzero": 4,
            "l2": 1.0,
        }
        for state_key, digest_character in (("exp_avg", "1"), ("exp_avg_sq", "2"))
    }
    return (
        {
            "parts": [
                {
                    "part": 0,
                    "parameter_count": 1,
                    "group_steps": [{"group": 0, "parameter_count": 1, "present": False, "value": None}],
                    "moments": before_moments,
                }
            ]
        },
        {
            "parts": [
                {
                    "part": 0,
                    "parameter_count": 1,
                    "group_steps": [{"group": 0, "parameter_count": 1, "present": True, "value": 1}],
                    "moments": after_moments,
                }
            ]
        },
    )


def _source_binding_evidence(*, source_marker="a", module_marker="a"):
    def source(relative_path, marker):
        return {"relative_path": relative_path, "sha256": marker * 64, "size": 1, "mode": 0o444}

    outer_sha256 = "b" * 64
    expected_outer_bytes = f"package_manifest_sha256={outer_sha256}\n".encode("ascii")
    explicit_sources = {
        "gate": source("tools/model_diagnostics/shared_prefix_native_optimizer_gate.py", source_marker),
        "captured_parity_helpers": source(
            "tools/model_diagnostics/shared_prefix_captured_citation_parity.py", source_marker
        ),
        "tools_package": source("tools/__init__.py", source_marker),
        "nemo_rl_package": source("nemo_rl/__init__.py", source_marker),
        "bridge_package": source(gate.BRIDGE_PACKAGE_RELATIVE.as_posix(), source_marker),
        "mcore_optimizer": source(gate.MCORE_OPTIMIZER_RELATIVE.as_posix(), source_marker),
        "outer_package_manifest": source(gate.OUTER_MANIFEST_RELATIVE.as_posix(), "b"),
        "outer_package_expected": {
            "relative_path": gate.OUTER_EXPECTED_RELATIVE.as_posix(),
            "sha256": hashlib.sha256(expected_outer_bytes).hexdigest(),
            "size": len(expected_outer_bytes),
            "mode": 0o444,
        },
        "source_identities": source(gate.SOURCE_IDENTITIES_RELATIVE.as_posix(), source_marker),
        "recipe": source("examples/nemo_gym/nemotron-3.5-nano/rlvr.yaml", source_marker),
        "pyproject": source("pyproject.toml", source_marker),
        "lockfile": source("uv.lock", source_marker),
    }
    source_trees = {
        name: {
            "root": root.as_posix(),
            "sha256": source_marker * 64,
            "directory_count": 1,
            "file_count": 1,
            "total_bytes": 1,
        }
        for name, root in gate.SOURCE_TREE_ROOTS.items()
    }
    modules = [
        {"module": module_name, **source(relative_path, module_marker)}
        for module_name, relative_path in (
            ("nemo_rl", "nemo_rl/__init__.py"),
            ("megatron.bridge", gate.BRIDGE_PACKAGE_RELATIVE.as_posix()),
            (
                "megatron.core",
                (gate.MCORE_ROOT_RELATIVE / "megatron/core/__init__.py").as_posix(),
            ),
            (
                "megatron.training",
                (gate.MCORE_ROOT_RELATIVE / "megatron/training/__init__.py").as_posix(),
            ),
            ("tools", "tools/__init__.py"),
        )
    ]
    module_digest = hashlib.sha256()
    for module in modules:
        module_digest.update(gate._canonical_json_bytes(module))
    imported_modules = {
        "prefixes": ["nemo_rl", "megatron.bridge", "megatron.core", "megatron.training", "tools"],
        "sha256": module_digest.hexdigest(),
        "file_backed_module_count": len(modules),
        "namespace_modules": [],
        "modules": modules,
    }
    runtime_source_package_sha256 = gate._source_package_sha256(explicit_sources, source_trees)
    outer_package = {
        "manifest_relative_path": gate.OUTER_MANIFEST_RELATIVE.as_posix(),
        "manifest_sha256": outer_sha256,
        "expected_relative_path": gate.OUTER_EXPECTED_RELATIVE.as_posix(),
        "expected_sha256": outer_sha256,
    }
    materialization_manifest = {
        "schema": gate.RUNTIME_MATERIALIZATION_SCHEMA,
        "base": {
            "root": gate.Q_BASE_ROOT,
            "ready": gate.Q_BASE_READY,
            "package_manifest_sha256": gate.Q_BASE_PACKAGE_SHA256,
        },
        "outer_package_manifest_sha256": outer_sha256,
        "runtime_source_package_sha256": runtime_source_package_sha256,
        "runtime_layout": gate.RUNTIME_LAYOUT,
        "repositories": {
            name: {
                "base_head": "1" * 40,
                "head": "2" * 40,
                "overlay_archive_sha256": "3" * 64,
                "overlay_payload_manifest_sha256": "4" * 64,
            }
            for name in ("NemoRL", "Megatron-Bridge", "Megatron-LM")
        },
        "runtime_repo_root": "/immutable/repo",
    }
    materialization = {
        "relative_path": gate.RUNTIME_MATERIALIZATION_RELATIVE.as_posix(),
        "sha256": hashlib.sha256(gate._canonical_json_bytes(materialization_manifest)).hexdigest(),
        "manifest": materialization_manifest,
    }
    binding_digest = hashlib.sha256()
    binding_digest.update(gate._canonical_json_bytes({"repo_root": "/immutable/repo"}))
    binding_digest.update(gate._canonical_json_bytes(outer_package))
    binding_digest.update(bytes.fromhex(runtime_source_package_sha256))
    binding_digest.update(gate._canonical_json_bytes(explicit_sources))
    binding_digest.update(gate._canonical_json_bytes(source_trees))
    binding_digest.update(gate._canonical_json_bytes(materialization))
    binding_digest.update(gate._canonical_json_bytes(imported_modules))
    return {
        "repo_root": "/immutable/repo",
        "outer_package": outer_package,
        "materialization": materialization,
        "runtime_source_package_sha256": runtime_source_package_sha256,
        "sha256": binding_digest.hexdigest(),
        "explicit_sources": explicit_sources,
        "source_trees": source_trees,
        "imported_modules": imported_modules,
    }


def _rank_evidence(rank, *, family_changes):
    before, after = _raw_optimizer_snapshots()
    families = {
        family: {
            "changed": changed,
            "before_sha256": "5" * 64,
            "after_sha256": ("6" if changed else "5") * 64,
            "tensor_count": 1,
            "numel": 8,
        }
        for family, changed in family_changes.items()
    }
    return {
        "rank": rank,
        "source_binding": _source_binding_evidence(),
        "native_step": {
            "raw_optimizer_before": before,
            "raw_optimizer_after": after,
            "raw_optimizer_step": {"parts": [{"part": 0}]},
            "optimizer_update": {
                "locally_present_required_families": sorted(family_changes),
                "families": families,
            },
        },
    }


def _full_rank_evidence(rank):
    before, after = _raw_optimizer_snapshots()
    raw_step = gate._validate_raw_optimizer_step(before, after)
    families = {
        family: {
            "changed": True,
            "before_sha256": "5" * 64,
            "after_sha256": "6" * 64,
            "tensor_count": 1,
            "numel": 8,
        }
        for family in gate.GATE_REQUIRED_FAMILIES
    }
    batch_preparation = {
        stage: {
            "capacity": gate.GATE_PACKING_TOKENS,
            "microbatches": None,
            "source_order": None,
            "plan": {
                "execution_units": 1,
                "row_indices": [3, 2, 1, 0],
                "slot_ids": [0] * gate.GATE_BATCH_SIZE,
                "shared_layout": True,
                "physical_length": 316,
                "capacity": gate.GATE_PACKING_TOKENS,
            },
            "worker_num_microbatches": 1,
        }
        for stage in ("logprob", "train")
    }
    metrics = {
        "global_loss": 1.0,
        "grad_norm": 2.0,
        "mtp": {name: 0.5 for name in gate.GATE_MTP_METRIC_KEYS},
    }
    return {
        "schema": gate.SCHEMA,
        "rank": rank,
        "local_rank": rank,
        "batch": {
            "sha256": gate.GATE_FIXTURE_SHA256,
            "rows": gate.GATE_BATCH_SIZE,
            "width": 200,
            "input_lengths": [150, 151, 152, 153],
            "prompt_length": 100,
            "rewards": [0.0, 1.0, 0.0, 1.0],
            "selected_tokens": gate.GATE_SELECTED_TOKENS,
        },
        "batch_preparation": batch_preparation,
        "config_sha256": "c" * 64,
        "source_binding": _source_binding_evidence(),
        "topology": gate.GATE_TOPOLOGY,
        "shared_prefix_mode": "train",
        "lazy_state_before_backward": {"parts": 1, "parameters": 1},
        "production_initialization": {
            "method": "mcore-production-init-state-fn",
            "closures": [
                {
                    "module": "megatron.core.optimizer",
                    "qualname": "factory.<locals>.init_state_fn",
                    "signature": "(opt, config=None)",
                }
            ],
            "initialized_parameters": 1,
            "bf16_remainder_parameters": 1,
            "state_tensor_numel": 3,
            "primary_sha256": "d" * 64,
            "gradient_sha256": "e" * 64,
            "group_steps": [{"part": 0}],
            "scheduler_steps": 0,
            "unchanged": True,
        },
        "native_step": {
            "scheduler_steps_before": 0,
            "scheduler_steps_after": gate.GATE_BATCH_SIZE,
            "raw_optimizer_before": before,
            "raw_optimizer_after": after,
            "raw_optimizer_step": raw_step,
            "metrics": metrics,
            "optimizer_update": {
                "changed_parameters": len(families),
                "locally_present_required_families": sorted(families),
                "families": families,
            },
        },
    }


def _sealed_evidence_root(tmp_path, per_rank=None, *, raw_overrides=None):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    per_rank = per_rank or [_full_rank_evidence(rank) for rank in range(gate.GATE_WORLD_SIZE)]
    raw_overrides = raw_overrides or {}
    manifest_lines = []
    for rank, evidence in enumerate(per_rank):
        name = f"native-optimizer.rank{rank}.json"
        raw = raw_overrides.get(rank, gate._canonical_json_bytes(evidence))
        path = evidence_root / name
        path.write_bytes(raw)
        path.chmod(0o444)
        manifest_lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n")
    manifest_path = evidence_root / "EVIDENCE.sha256"
    manifest_path.write_text("".join(manifest_lines), encoding="ascii")
    manifest_path.chmod(0o444)
    evidence_root.chmod(0o555)
    return evidence_root, per_rank


def test_source_binding_evidence_rejects_tampered_mcore_hash():
    binding = _source_binding_evidence()
    binding["explicit_sources"]["mcore_optimizer"]["sha256"] = "e" * 64

    with pytest.raises(gate.NativeOptimizerGateError, match="source-package digest mismatch"):
        gate._validate_source_binding_evidence(binding)


def test_source_binding_evidence_rejects_tampered_repo_root():
    binding = _source_binding_evidence()
    binding["repo_root"] = "/different/repo"

    with pytest.raises(gate.NativeOptimizerGateError, match="materialization repository root mismatch"):
        gate._validate_source_binding_evidence(binding)


def test_runtime_source_binding_rejects_builder_expected_mismatch():
    binding = _source_binding_evidence()

    with pytest.raises(gate.NativeOptimizerGateError, match="RUNTIME_SOURCE_PACKAGE_EXPECTED"):
        gate._require_expected_runtime_source_package(binding["runtime_source_package_sha256"], "e" * 64)


def test_source_binding_evidence_rejects_outer_digest_disagreement():
    binding = _source_binding_evidence()
    binding["outer_package"]["expected_sha256"] = "e" * 64

    with pytest.raises(gate.NativeOptimizerGateError, match="differs from manifest digest"):
        gate._validate_source_binding_evidence(binding)


def test_source_binding_evidence_rejects_self_asserted_materialization_root():
    binding = _source_binding_evidence()
    binding["materialization"]["manifest"]["runtime_repo_root"] = "/attacker/runtime"
    binding["materialization"]["sha256"] = hashlib.sha256(
        gate._canonical_json_bytes(binding["materialization"]["manifest"])
    ).hexdigest()

    with pytest.raises(gate.NativeOptimizerGateError, match="repository root mismatch"):
        gate._validate_source_binding_evidence(binding)


def test_source_binding_evidence_requires_full_bridge_and_mcore_trees():
    binding = _source_binding_evidence()
    del binding["source_trees"]["megatron_core"]

    with pytest.raises(gate.NativeOptimizerGateError, match="full source trees"):
        gate._validate_source_binding_evidence(binding)


def test_source_binding_evidence_rejects_noncanonical_bridge_root():
    binding = _source_binding_evidence()
    binding["explicit_sources"]["bridge_package"][
        "relative_path"
    ] = "alternate/Megatron-Bridge/src/megatron/bridge/__init__.py"

    with pytest.raises(gate.NativeOptimizerGateError, match="Bridge package path is not canonical"):
        gate._validate_source_binding_evidence(binding)


def test_source_binding_evidence_rejects_imported_mcore_outside_canonical_vendor_root():
    binding = _source_binding_evidence()
    for module in binding["imported_modules"]["modules"]:
        if module["module"] == "megatron.core":
            module["relative_path"] = "alternate/Megatron-LM/megatron/core/__init__.py"

    with pytest.raises(gate.NativeOptimizerGateError, match="outside canonical vendored root"):
        gate._validate_source_binding_evidence(binding)


def test_per_rank_family_gate_rejects_union_any_local_false_green():
    per_rank = [
        _rank_evidence(0, family_changes={"attention": True, "mamba": False}),
        _rank_evidence(1, family_changes={"attention": True, "mamba": True}),
    ]

    with pytest.raises(gate.NativeOptimizerGateError, match="rank 0 did not update"):
        gate._validate_per_rank_required_family_updates(
            per_rank, required_families={"attention", "mamba"}, world_size=2
        )


def test_per_rank_gate_rejects_source_package_mismatch():
    per_rank = [
        _rank_evidence(0, family_changes={"attention": True}),
        _rank_evidence(1, family_changes={"attention": True}),
    ]
    per_rank[1]["source_binding"] = _source_binding_evidence(source_marker="e")

    with pytest.raises(gate.NativeOptimizerGateError, match="immutable source package differs"):
        gate._validate_per_rank_required_family_updates(per_rank, required_families={"attention"}, world_size=2)


def test_per_rank_gate_rejects_rank_specific_loaded_module_manifest():
    per_rank = [
        _rank_evidence(0, family_changes={"attention": True}),
        _rank_evidence(1, family_changes={"attention": True}),
    ]
    per_rank[1]["source_binding"] = _source_binding_evidence(module_marker="e")

    with pytest.raises(gate.NativeOptimizerGateError, match="immutable source package differs"):
        gate._validate_per_rank_required_family_updates(per_rank, required_families={"attention"}, world_size=2)


def test_validate_evidence_root_accepts_exact_four_rank_bundle(tmp_path):
    evidence_root, per_rank = _sealed_evidence_root(tmp_path)
    try:
        result = gate._validate_evidence_root(evidence_root)
        assert result == {
            "result_sha256": hashlib.sha256(gate._canonical_json_bytes(per_rank)).hexdigest(),
            "outer_package_sha256": "b" * 64,
            "runtime_source_package_sha256": per_rank[0]["source_binding"]["runtime_source_package_sha256"],
            "source_binding_sha256": per_rank[0]["source_binding"]["sha256"],
            "materialization_sha256": per_rank[0]["source_binding"]["materialization"]["sha256"],
            "ranks": gate.GATE_WORLD_SIZE,
            "selected_tokens": gate.GATE_SELECTED_TOKENS,
        }
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_cli_emits_exact_single_green_line(tmp_path, capsys):
    evidence_root, _ = _sealed_evidence_root(tmp_path)
    try:
        gate.main(["validate-evidence", "--evidence-root", str(evidence_root)])
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("NEMORL_SHARED_PREFIX_NATIVE_OPTIMIZER_EVIDENCE_GREEN ")
        for field in (
            "result_sha256=",
            "outer_package_sha256=",
            "runtime_source_package_sha256=",
            "source_binding_sha256=",
            "materialization_sha256=",
            "ranks=4",
            f"selected_tokens={gate.GATE_SELECTED_TOKENS}",
        ):
            assert field in lines[0]
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_duplicate_rank_json_keys(tmp_path):
    evidence_root, _ = _sealed_evidence_root(tmp_path, raw_overrides={0: b'{"rank":0,"rank":0}'})
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="duplicate JSON key"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_self_asserted_raw_step_summary(tmp_path):
    per_rank = [_full_rank_evidence(rank) for rank in range(gate.GATE_WORLD_SIZE)]
    per_rank[0]["native_step"]["raw_optimizer_step"] = {"parts": []}
    evidence_root, _ = _sealed_evidence_root(tmp_path, per_rank)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="not derived evidence"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_non_shared_batch_preparation(tmp_path):
    per_rank = [_full_rank_evidence(rank) for rank in range(gate.GATE_WORLD_SIZE)]
    per_rank[0]["batch_preparation"] = {"shared_prefix_mode": "train"}
    evidence_root, _ = _sealed_evidence_root(tmp_path, per_rank)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="invalid stages"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_missing_mtp_metric(tmp_path):
    per_rank = [_full_rank_evidence(rank) for rank in range(gate.GATE_WORLD_SIZE)]
    del per_rank[0]["native_step"]["metrics"]["mtp"]["mtp_5_loss"]
    evidence_root, _ = _sealed_evidence_root(tmp_path, per_rank)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="all MTP heads"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_self_asserted_family_change(tmp_path):
    per_rank = [_full_rank_evidence(rank) for rank in range(gate.GATE_WORLD_SIZE)]
    family = per_rank[0]["native_step"]["optimizer_update"]["families"]["mamba"]
    family["after_sha256"] = family["before_sha256"]
    evidence_root, _ = _sealed_evidence_root(tmp_path, per_rank)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="change flag is not derived"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


def test_validate_evidence_root_rejects_extra_inventory(tmp_path):
    evidence_root, _ = _sealed_evidence_root(tmp_path)
    evidence_root.chmod(0o755)
    extra = evidence_root / "summary.json"
    extra.write_text("{}", encoding="utf-8")
    extra.chmod(0o444)
    evidence_root.chmod(0o555)
    try:
        with pytest.raises(gate.NativeOptimizerGateError, match="invalid inventory"):
            gate._validate_evidence_root(evidence_root)
    finally:
        evidence_root.chmod(0o755)


@pytest.mark.parametrize("state_key", ("exp_avg", "exp_avg_sq"))
def test_raw_optimizer_step_gate_rejects_unchanged_adam_moment(state_key):
    before, after = _raw_optimizer_snapshots()
    after["parts"][0]["moments"][state_key] = before["parts"][0]["moments"][state_key].copy()

    with pytest.raises(gate.NativeOptimizerGateError, match=rf"{state_key} did not change"):
        gate._validate_raw_optimizer_step(before, after)


def test_raw_optimizer_step_gate_rejects_missing_counter_advance():
    before, after = _raw_optimizer_snapshots()
    after["parts"][0]["group_steps"][0].update({"present": False, "value": None})

    with pytest.raises(gate.NativeOptimizerGateError, match="counter did not advance exactly 0->1"):
        gate._validate_raw_optimizer_step(before, after)
