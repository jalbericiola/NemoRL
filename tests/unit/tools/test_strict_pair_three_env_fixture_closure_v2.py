from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.unit.tools import test_nano35_single_env_pair as pair_harness

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_CONTRACTS = {
    "reasoning_gym": {
        "name": "reasoning_gym_example.jsonl",
        "sha256": ("da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"),
        "agent": "reasoning_gym_simple_agent",
        "verifier": None,
    },
    "citation": {
        "name": "citation_example.jsonl",
        "sha256": ("d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc"),
        "agent": "citation_format_simple_agent",
        "verifier": "string_match",
    },
    "freeform": {
        "name": "freeform_example.jsonl",
        "sha256": ("8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399"),
        "agent": "freeform_formatting_simple_agent",
        "verifier": "regex",
    },
}
STRICT_PAIR_SOURCE_CLOSURE = (
    "nemo_rl/utils/strict_model_transport.py",
    "nemo_rl/models/generation/vllm/vllm_worker_async.py",
    "nemo_rl/experience/rollout_manager.py",
    "tests/unit/tools/data/reasoning_gym_example.jsonl",
    "tests/unit/tools/data/citation_example.jsonl",
    "tests/unit/tools/data/freeform_example.jsonl",
)


def _literal_assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1
    return ast.literal_eval(matches[0].value)


def _literal_tuple_assignment(path: Path, name: str) -> tuple[str, ...]:
    value = _literal_assignment(path, name)
    assert type(value) is tuple
    assert all(type(member) is str for member in value)
    return value


def test_pair79_roster_is_ordered_identically_across_producer_and_v2() -> None:
    from nemo_rl.utils.strict_captured_replay_manifest_v2 import (
        SLURM_EXPORT_ALLOWED_NAMES as V2_MANIFEST_NAMES,
    )

    contract = (
        REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"^STRICT_PAIR_SLURM_EXPORT_ALLOWED_NAMES=\(\n(.*?)^\)$",
        contract,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    producer_names = tuple(match.group(1).split())
    entrypoint_names = _literal_tuple_assignment(
        REPO_ROOT / "examples/nemo_gym/run_strict_captured_replay_v2.py",
        "_SLURM_EXPORT_ALLOWED_NAMES",
    )
    expected = pair_harness.SLURM_EXPORT_ALLOWED_NAMES
    assert len(expected) == 79
    assert producer_names == expected == V2_MANIFEST_NAMES == entrypoint_names


def test_pair79_source_schema_is_identical_across_all_producer_and_v2_boundaries() -> (
    None
):
    expected = "nemo-rl-strict-slurm-export-file-v3"
    contract = (
        REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/strict_pair_contract.sh"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO_ROOT / "examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh"
    ).read_text(encoding="utf-8")
    contract_schemas = re.findall(
        r'^\s+"schema": "(nemo-rl-strict-slurm-export-file-v\d+)",$',
        contract,
        flags=re.MULTILINE,
    )
    launcher_schemas = re.findall(
        r"STRICT_PAIR_SLURM_EXPORT_PREPARED[^\n]+schema="
        r"(nemo-rl-strict-slurm-export-file-v\d+)",
        launcher,
    )
    manifest_schema = _literal_assignment(
        REPO_ROOT / "nemo_rl/utils/strict_captured_replay_manifest_v2.py",
        "PAIR_SLURM_EXPORT_SCHEMA",
    )
    entrypoint_schema = _literal_assignment(
        REPO_ROOT / "examples/nemo_gym/run_strict_captured_replay_v2.py",
        "_PAIR_SLURM_EXPORT_SCHEMA",
    )
    assert contract_schemas == [expected]
    assert launcher_schemas == [expected]
    assert manifest_schema == expected
    assert entrypoint_schema == expected


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_fixture(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha256
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    encoded_rows = raw.splitlines(keepends=True)
    assert len(encoded_rows) == 5
    assert all(row.endswith(b"\n") and row != b"\n" for row in encoded_rows)

    rows = []
    for encoded_row in encoded_rows:
        parsed = json.loads(
            encoded_row[:-1].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        assert isinstance(parsed, dict)
        rows.append(parsed)
    return rows


def _assert_profile_schema(
    rows: list[dict[str, Any]], *, agent_name: str, verifier_type: str | None
) -> None:
    pattern_ids: set[str] = set()
    for row in rows:
        assert row["agent_ref"] == {
            "type": "responses_api_agents",
            "name": agent_name,
        }
        params = row["responses_create_params"]
        assert isinstance(params, dict)
        assert set(params) == {"input"}
        assert isinstance(params["input"], list) and len(params["input"]) == 1
        message = params["input"][0]
        assert isinstance(message, dict)
        assert set(message) == {"role", "content"}
        assert message["role"] == "user"
        assert isinstance(message["content"], str) and message["content"]

        if verifier_type is None:
            assert set(row) == {
                "responses_create_params",
                "question",
                "answer",
                "metadata",
                "agent_ref",
            }
            assert isinstance(row["question"], str) and row["question"]
            assert isinstance(row["answer"], str) and row["answer"]
            assert isinstance(row["metadata"], dict)
            continue

        assert set(row) == {"responses_create_params", "verifier", "agent_ref"}
        verifier = row["verifier"]
        assert isinstance(verifier, dict)
        assert verifier["type"] == verifier_type
        if verifier_type == "string_match":
            assert set(verifier) == {"type", "patterns", "expected_markers"}
            patterns = verifier["patterns"]
            markers = verifier["expected_markers"]
            assert isinstance(patterns, list) and patterns
            assert isinstance(markers, list) and len(markers) == len(patterns)
            assert all(
                isinstance(pattern, str)
                and isinstance(marker, str)
                and re.fullmatch(pattern, marker)
                for pattern, marker in zip(patterns, markers, strict=True)
            )
        else:
            assert set(verifier) == {
                "type",
                "pattern_id",
                "verify_regex",
                "verify_min_matches",
            }
            pattern_id = verifier["pattern_id"]
            assert isinstance(pattern_id, str) and pattern_id
            assert pattern_id not in pattern_ids
            pattern_ids.add(pattern_id)
            patterns = verifier["verify_regex"]
            assert isinstance(patterns, list) and len(patterns) == 1
            assert isinstance(patterns[0], str) and patterns[0]
            re.compile(patterns[0])
            assert verifier["verify_min_matches"] == 3


def _close_synthetic_deployment(
    root: Path,
    kind: str,
    original_builder: Any,
    *,
    fixture_kind: str = "valid",
    omit_acceptance_binding: bool = False,
) -> tuple[Path, Path, Path, dict[str, str]]:
    deployment, nemo_root, job_wrapper, environment = original_builder(
        root, kind, fixture_kind
    )
    assert kind == "valid"

    deployment.chmod(0o700)
    nemo_root.chmod(0o700)
    paths_to_add = list(STRICT_PAIR_SOURCE_CLOSURE)
    for relative in paths_to_add:
        source_path = REPO_ROOT / relative
        destination = nemo_root / relative
        if destination.exists():
            assert destination.is_file() and not destination.is_symlink()
            assert destination.read_bytes() == source_path.read_bytes()
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    if omit_acceptance_binding:
        launcher_relative = "examples/nemo_gym/nemotron-3.5-nano/launch_pair.sh"
        launcher = nemo_root / launcher_relative
        launcher.chmod(0o700)
        binding = (
            '    EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256="'
            '${STRICT_PAIR_ACCEPTANCE_POLICY_SHA256}" \\\n'
        )
        source = launcher.read_text(encoding="ascii")
        assert source.count(binding) == 1
        launcher.write_text(source.replace(binding, ""), encoding="ascii")
        paths_to_add.append(launcher_relative)

    subprocess.run(
        ["git", "-C", str(nemo_root), "add", "--", *paths_to_add],
        check=True,
    )
    staged = subprocess.run(
        ["git", "-C", str(nemo_root), "diff", "--cached", "--quiet", "HEAD", "--"]
    )
    assert staged.returncode in {0, 1}
    if staged.returncode == 1:
        subprocess.run(
            [
                "git",
                "-C",
                str(nemo_root),
                "-c",
                "user.name=NeMo RL test",
                "-c",
                "user.email=nemo-rl-test@nvidia.com",
                "commit",
                "--amend",
                "--no-edit",
                "-q",
            ],
            check=True,
        )

    manifest = deployment / "NemoRL.runnable.sha256"
    retained_entries = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        _, separator, raw_path = line.partition("  ")
        assert separator == "  "
        if not raw_path.startswith(f"{nemo_root}/"):
            retained_entries.append(line)
    tracked = subprocess.check_output(
        ["git", "-C", str(nemo_root), "ls-files", "--recurse-submodules", "-z"]
    ).split(b"\0")
    source_entries = []
    for raw_relative in tracked:
        if not raw_relative:
            continue
        tracked_path = nemo_root / raw_relative.decode()
        if tracked_path.is_symlink() or tracked_path.is_dir():
            continue
        source_entries.append(f"{pair_harness._sha256(tracked_path)}  {tracked_path}")
    manifest.chmod(0o600)
    manifest.write_text(
        "\n".join(sorted([*retained_entries, *source_entries])) + "\n",
        encoding="ascii",
    )
    manifest.chmod(0o400)
    environment["EXPECTED_NEMO_HEAD"] = subprocess.check_output(
        ["git", "-C", str(nemo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    gym_root = nemo_root / "3rdparty/Gym-workspace/Gym"
    environment["EXPECTED_GYM_TREE"] = subprocess.check_output(
        ["git", "-C", str(gym_root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    environment["EXPECTED_NEMO_RUNNABLE_MANIFEST_SHA256"] = pair_harness._sha256(
        manifest
    )

    pair_harness._seal_tracked_files(nemo_root)
    nemo_root.chmod(0o500)
    deployment.chmod(0o500)
    return deployment, nemo_root, job_wrapper, environment


def _run_real_prepare_contract(
    root: Path,
    environment_name: str,
    original_builder: Any,
) -> subprocess.CompletedProcess[str]:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model":"test"}\n', encoding="ascii")
    container = root / "nemo-rl.sqsh"
    container.write_bytes(b"test training container\n")
    sandbox_container = root / "sandbox.sqsh"
    sandbox_container.write_bytes(b"test sandbox container\n")
    results = root / "results"
    results.mkdir(mode=0o700)
    persistent_cache = root / "cache"
    persistent_cache.mkdir()
    hf_home = root / "hf-cache"
    hf_home.mkdir()

    deployment, nemo_root, _, deployment_environment = _close_synthetic_deployment(
        root, "valid", original_builder
    )
    recipe_dir = nemo_root / "examples/nemo_gym/nemotron-3.5-nano"
    harness = root / "strict-pair-prepare-harness.sh"
    harness.write_text(
        """#!/bin/bash -p
set -euo pipefail
source "${CONTRACT_PATH}"
strict_pair_select_environment
strict_pair_load_runtime_tools
strict_pair_prepare_contract "${PROJECT_ROOT}" "${RECIPE_DIR}"
printf '%s\\t%s\\t%s\\t%s\\t%s\\n' \\
  "${STRICT_PAIR_ENVIRONMENT}" \\
  "${STRICT_PAIR_FIXTURE_RELATIVE}" \\
  "${STRICT_PAIR_FIXTURE_SHA256}" \\
  "${STRICT_PAIR_FIXTURE_ROWS}" \\
  "${TRAIN_PATH}"
""",
        encoding="ascii",
    )
    harness.chmod(0o500)
    child_environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "HOME": str(root),
        "CONTRACT_PATH": str(recipe_dir / "strict_pair_contract.sh"),
        "PROJECT_ROOT": str(nemo_root),
        "RECIPE_DIR": str(recipe_dir),
        "PAIR_ID": f"fixture-closure-{environment_name}",
        "STRICT_PAIR_ENVIRONMENT": environment_name,
        "RESULTS_DIR": str(results),
        "PERSISTENT_CACHE": str(persistent_cache),
        "HF_HOME": str(hf_home),
        "MODEL_PATH": str(model),
        "CONTAINER": str(container),
        "SANDBOX_CONTAINER": str(sandbox_container),
        **deployment_environment,
    }
    return subprocess.run(
        [str(pair_harness.HOST_TOOLS["bash"]), "-p", str(harness)],
        env=child_environment,
        capture_output=True,
        text=True,
    )


def _run_closed_pair(
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit_acceptance_binding: bool = False,
    **environment: str,
) -> pair_harness.PairRun:
    original_builder = pair_harness._make_deployment
    monkeypatch.setattr(
        pair_harness,
        "_make_deployment",
        lambda root, kind, fixture_kind="valid": _close_synthetic_deployment(
            root,
            kind,
            original_builder,
            fixture_kind=fixture_kind,
            omit_acceptance_binding=omit_acceptance_binding,
        ),
    )
    return pair_harness._run_pair(**environment)


@pytest.mark.parametrize("environment", tuple(FIXTURE_CONTRACTS))
def test_fixture_closure_runs_real_strict_pair_preparation(
    tmp_path: Path, environment: str
) -> None:
    contract = FIXTURE_CONTRACTS[environment]
    fixture = REPO_ROOT / "tests/unit/tools/data" / contract["name"]
    rows = _load_strict_fixture(fixture, contract["sha256"])
    _assert_profile_schema(
        rows,
        agent_name=contract["agent"],
        verifier_type=contract["verifier"],
    )

    run = _run_real_prepare_contract(
        tmp_path.resolve(), environment, pair_harness._make_deployment
    )
    assert run.returncode == 0, run.stderr
    selected_environment, relative, digest, rows_raw, train_path = (
        run.stdout.strip().split("\t")
    )
    assert selected_environment == environment
    assert relative == f"tests/unit/tools/data/{contract['name']}"
    assert digest == contract["sha256"]
    assert rows_raw == "5"
    assert train_path.endswith(f"/tests/unit/tools/data/{contract['name']}")


def test_parent_forwards_authoritative_acceptance_policy_and_ignores_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poison = "0" * 64
    run = _run_closed_pair(
        monkeypatch,
        STRICT_PAIR_ENVIRONMENT="citation",
        EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256=poison,
    )
    assert run.process.returncode == 0, run.process.stderr
    assert run.manifest is not None
    manifest = json.loads(run.manifest)
    expected = manifest["acceptance"]["policy_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", expected)
    assert expected != poison
    assert set(run.export_payloads) == {"off", "on"}
    for payload in run.export_payloads.values():
        values = dict(pair_harness._parse_slurm_export_payload(payload))
        assert values["EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256"] == (
            expected.encode("ascii")
        )


def test_arm_rejects_omitted_acceptance_policy_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_closed_pair(monkeypatch, omit_acceptance_binding=True)
    assert run.process.returncode == 2
    assert run.manifest is None
    assert (
        run.process.stderr.count(
            "EXPECTED_STRICT_PAIR_ACCEPTANCE_POLICY_SHA256 must be an explicit "
            "lowercase SHA-256."
        )
        == 2
    )
    assert (
        "strict pair export preparation failed: off_status=2 on_status=2"
        in run.process.stderr
    )
